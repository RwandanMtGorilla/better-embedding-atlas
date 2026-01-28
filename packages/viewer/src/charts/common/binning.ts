// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";
import * as d3 from "d3";
import type { ScaleType } from "./types.js";

export interface Binning {
  scale: ScaleClass & { domain: [number, number]; constant?: number };
  binStart: number;
  binSize: number;
}

export interface ScaleClass {
  type: "linear" | "log" | "symlog";
  expr(x: SQL.ExprNode, constant: number): SQL.ExprNode;
  forward(x: number, constant: number): number;
  reverse(x: number, constant: number): number;
}

const scaleTypes: Record<string, ScaleClass> = {
  linear: { type: "linear", expr: (x) => x, forward: (x) => x, reverse: (x) => x },
  log: {
    type: "log",
    expr: (x) => SQL.cond(SQL.gt(x, 0), SQL.log(x), SQL.literal("nan")),
    forward: (x) => Math.log10(x),
    reverse: (x) => Math.pow(10, x),
  },
  symlog: {
    type: "symlog",
    expr: (x: SQL.ExprNode, constant: number) =>
      SQL.mul(SQL.sign(x), SQL.ln(SQL.add(1, SQL.abs(SQL.div(x, constant))))),
    forward: (x: number, constant: number) => Math.sign(x) * Math.log1p(Math.abs(x) / constant),
    reverse: (x: number, constant: number) => Math.sign(x) * Math.expm1(Math.abs(x)) * constant,
  },
};

function roundToNearest(value: number, array: number[]): number {
  let minV = value;
  let minD = Infinity;
  for (let v of array) {
    let d = Math.abs(value - v);
    if (d < minD) {
      minD = d;
      minV = v;
    }
  }
  return minV;
}

/** Date interval types for date binning */
export type DateInterval = "year" | "month" | "week" | "day" | "hour" | "minute";

/** Extended binning with optional date metadata */
export interface DateBinning extends Binning {
  _isDate?: boolean;
  _dateInterval?: DateInterval;
}

/** Get approximate milliseconds for a date interval */
function getIntervalMs(interval: DateInterval): number {
  switch (interval) {
    case "year":
      return 365.25 * 24 * 60 * 60 * 1000;
    case "month":
      return 30.44 * 24 * 60 * 60 * 1000;
    case "week":
      return 7 * 24 * 60 * 60 * 1000;
    case "day":
      return 24 * 60 * 60 * 1000;
    case "hour":
      return 60 * 60 * 1000;
    case "minute":
      return 60 * 1000;
  }
}

/** Infer appropriate date interval based on time range */
export function inferDateInterval(minMs: number, maxMs: number, desiredCount: number = 20): DateInterval {
  const rangeMs = maxMs - minMs;
  const rangeDays = rangeMs / (1000 * 60 * 60 * 24);

  // Choose interval so we get roughly desiredCount bins
  if (rangeDays > 365 * 5) return "year";
  if (rangeDays > 365) return "month";
  if (rangeDays > 60) return "week";
  if (rangeDays > 3) return "day";
  if (rangeDays > 0.1) return "hour";
  return "minute";
}

/** Infer binning for date values (in epoch milliseconds) */
export function inferDateBinning(
  stats: { min: number; max: number; count: number },
  options: { desiredCount?: number } = {},
): DateBinning {
  const { min, max } = stats;
  const desiredCount = options.desiredCount ?? 20;

  const interval = inferDateInterval(min, max, desiredCount);
  const intervalMs = getIntervalMs(interval);

  // Align to time boundaries
  const alignedMin = Math.floor(min / intervalMs) * intervalMs;
  const alignedMax = Math.ceil(max / intervalMs) * intervalMs;

  return {
    scale: {
      ...scaleTypes.linear,
      domain: [alignedMin, alignedMax],
    },
    binStart: alignedMin,
    binSize: intervalMs,
    _isDate: true,
    _dateInterval: interval,
  };
}

export function inferBinning(
  stats: { min: number; minPositive: number; max: number; median: number; count: number },
  options: {
    scale?: ScaleType | null;
    desiredCount?: number;
  } = {},
): Binning {
  let { min, max, median, count } = stats;

  // Infer scale type
  let scaleType = options.scale;
  if (scaleType == "band") {
    scaleType = null;
  }
  if (scaleType == null) {
    scaleType = "linear";
    if (count >= 100 && min >= 0 && median < max * 0.05) {
      scaleType = min > 0 ? "log" : "symlog";
    }
  }

  if (min <= 0 && scaleType == "log") {
    if (max <= 0) {
      // Log scale with no positive value, we'll just do a default domain of [1, 10]
      min = 1;
      max = 10;
    } else {
      min = Math.min(stats.minPositive, max / 10);
    }
  }

  let desiredCount = options.desiredCount ?? 5;

  switch (scaleType) {
    case "linear": {
      let s = d3.scaleLinear().domain([min, max]).nice(desiredCount);
      let ticks = s.ticks(desiredCount);
      return {
        scale: { ...scaleTypes.linear, domain: s.domain() as any },
        binStart: s.domain()[0],
        binSize: ticks[1] - ticks[0],
      };
    }
    case "log": {
      let s = d3.scaleLog().domain([min, max]).nice();
      let binStart = Math.log10(s.domain()[0]);
      let binSize = (Math.log10(s.domain()[1]) - binStart) / desiredCount;
      binSize = roundToNearest(binSize, [0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]);
      return {
        scale: { ...scaleTypes.log, domain: s.domain() as any },
        binStart: binStart,
        binSize: binSize,
      };
    }
    case "symlog": {
      let absMax = Math.max(Math.abs(min), Math.abs(max));
      let constant = absMax >= 100 ? 1 : absMax > 0 ? absMax / 1e5 : 1;
      let sMin = scaleTypes.symlog.forward(min, constant);
      let sMax = scaleTypes.symlog.forward(max, constant);
      return {
        scale: { ...scaleTypes.symlog, domain: [min, max], constant: constant },
        binStart: sMin,
        binSize: (sMax - sMin) / desiredCount,
      };
    }
    default:
      throw new Error("invalid scale type");
  }
}
