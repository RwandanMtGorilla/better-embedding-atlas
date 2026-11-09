// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { measureText } from "../measure_text.js";
import { type Point, type Rectangle } from "../utils.js";
import type { LabelTileSnapshot } from "./label_tile_requests.js";
import { labelTileNeighbors } from "./local_label_tiles.js";
import type { Label } from "./types.js";
import { dynamicLabelPlacement } from "./worker/index.js";

export interface LabelWithPlacement {
  id?: string;
  text: string;
  fontSize: number;
  bounds: Rectangle;
  locationAtZero: Point;
  coordinate: Point;
  placement: { minScale: number; maxScale: number } | null;
}

export async function layoutLabels(
  normalScale: number,
  labels: Label[],
  fontFamily: string,
): Promise<LabelWithPlacement[]> {
  if (labels.length === 0) {
    return [];
  }
  let minLevel = labels.reduce((a, b) => Math.min(a, b.level ?? 0), 0);
  let maxLevel = labels.reduce((a, b) => Math.max(a, b.level ?? 0), 0);
  let globalMaxScale = normalScale * 0.5;
  let threshold = normalScale * 2;

  let result: LabelWithPlacement[] = labels.map((cluster) => {
    let location = { x: cluster.x, y: cluster.y };
    let level = cluster.level ?? 0;
    let fontSize = level == 0 ? 14 : 12;
    let size = measureText({
      text: cluster.text,
      fontSize: fontSize,
      fontFamily: fontFamily,
    });
    size.width += 4;
    size.height += 4;
    return {
      text: cluster.text,
      fontSize: fontSize,
      bounds: {
        xMin: location.x - size.width / 2,
        xMax: location.x + size.width / 2,
        yMin: location.y - size.height / 2,
        yMax: location.y + size.height / 2,
      },
      locationAtZero: location,
      priority: cluster.priority,
      minScale: level == maxLevel ? null : 1 / (threshold * Math.pow(2, level) * 1.2),
      maxScale: level == minLevel ? null : 1 / (threshold * Math.pow(2, level - 1)),
      coordinate: { x: cluster.x, y: cluster.y },
      placement: null,
    };
  });
  let placements = await dynamicLabelPlacement(result, { globalMaxScale: 1 / globalMaxScale });
  for (let i = 0; i < placements.length; i++) {
    let placement = placements[i];
    if (placement != null) {
      let maxScale = 1 / placement.minScale;
      let minScale = 1 / placement.maxScale;
      result[i].placement = { minScale, maxScale };
    }
  }
  return result;
}

/** Place a tile's labels only after its full, fixed neighborhood is available. */
export function layoutTileLabels(
  { visible, results }: LabelTileSnapshot,
  pixelsPerUnit: number,
  fontFamily: string,
): LabelWithPlacement[] {
  const fontSize = 12;
  const padding = 4;
  let tileWidth = visible[0]?.size * pixelsPerUnit;
  if (!(tileWidth >= fontSize + padding) || !Number.isFinite(tileWidth)) {
    return [];
  }

  type Candidate = LabelWithPlacement & { id: string; priority: number };
  let measured = new Map<string, Candidate[]>();
  function measureTile(key: string): Candidate[] {
    let cached = measured.get(key);
    if (cached != null) {
      return cached;
    }
    let candidates: Candidate[] = [];
    for (let label of results.get(key) ?? []) {
      let text = label.text
        .split("\n")
        .map((line) => fitLabelLine(line, tileWidth - padding, fontSize, fontFamily))
        .join("\n");
      let size = measureText({ text, fontSize, fontFamily });
      if (size.height + padding > tileWidth) {
        continue;
      }
      let coordinate = { x: label.x, y: label.y };
      let halfWidth = (size.width + padding) / pixelsPerUnit / 2;
      let halfHeight = (size.height + padding) / pixelsPerUnit / 2;
      candidates.push({
        id: label.id,
        priority: label.priority,
        text,
        fontSize,
        coordinate,
        locationAtZero: coordinate,
        bounds: {
          xMin: coordinate.x - halfWidth,
          xMax: coordinate.x + halfWidth,
          yMin: coordinate.y - halfHeight,
          yMax: coordinate.y + halfHeight,
        },
        placement: { minScale: 0, maxScale: Infinity },
      });
    }
    measured.set(key, candidates);
    return candidates;
  }

  let placed: LabelWithPlacement[] = [];
  for (let tile of visible) {
    if (!results.get(tile.key)?.length) {
      continue;
    }
    let neighbors = labelTileNeighbors(tile);
    if (neighbors.some(({ key }) => !results.has(key))) {
      continue;
    }
    let candidates = neighbors.flatMap(({ key }) => measureTile(key));
    for (let label of measureTile(tile.key)) {
      // A higher priority candidate blocks this label even if that candidate is itself hidden.
      // This limits placement dependencies to one ring of tiles.
      let blocked = candidates.some(
        (other) =>
          (other.priority > label.priority || (other.priority === label.priority && other.id < label.id)) &&
          label.bounds.xMin < other.bounds.xMax &&
          label.bounds.xMax > other.bounds.xMin &&
          label.bounds.yMin < other.bounds.yMax &&
          label.bounds.yMax > other.bounds.yMin,
      );
      if (!blocked) {
        placed.push(label);
      }
    }
  }
  return placed;
}

function fitLabelLine(text: string, maxWidth: number, fontSize: number, fontFamily: string): string {
  let width = (text: string) => measureText({ text, fontSize, fontFamily }).width;
  if (width(text) <= maxWidth) {
    return text;
  }
  let characters = Array.from(text);
  let lo = 0;
  let hi = characters.length;
  let fitted = width("…") <= maxWidth ? "…" : "";
  while (lo < hi) {
    let mid = Math.floor((lo + hi) / 2);
    let candidate = characters.slice(0, mid + 1).join("") + "…";
    if (width(candidate) <= maxWidth) {
      fitted = candidate;
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return fitted;
}
