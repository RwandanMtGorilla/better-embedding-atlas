// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { stemmer } from "stemmer";

import type { Rectangle } from "../utils.js";
import { stopWords as defaultStopWords } from "./stop_words.js";

export interface TextSummarizerOptions {
  regions: Rectangle[][];
  levels?: number[];
  stopWords?: string[] | null;
}

interface Candidate {
  stem: string;
  word: string;
  score: number;
  region: number;
}

/** BM25-style region keywords with document/region IDF and same-level diversification. */
export class TextSummarizer {
  private segmenter: Intl.Segmenter;
  private binning: XYBinning;
  private stopWords: Set<string>;
  private key2RegionIndices: Map<number, number[]>;
  private frequencyPerClass: Map<string, number>[];
  private documentFrequencyPerClass: Map<string, number>[];
  private documentCountPerClass: number[];
  private documentFrequency = new Map<string, number>();
  private documentCount = 0;
  private levels: number[];
  private regions: Rectangle[][];

  /** Create a new TextSummarizer */
  constructor(options: TextSummarizerOptions) {
    this.binning = XYBinning.inferFromRegions(options.regions);
    this.segmenter = new Intl.Segmenter(undefined, { granularity: "word" });
    this.stopWords = new Set((options.stopWords ?? defaultStopWords).map((word) => word.toLowerCase()));

    this.frequencyPerClass = options.regions.map(() => new Map());
    this.documentFrequencyPerClass = options.regions.map(() => new Map());
    this.documentCountPerClass = options.regions.map(() => 0);
    this.levels = options.regions.map((_, i) => options.levels?.[i] ?? 0);
    this.regions = options.regions;

    // Generate key2RegionIndices, a map from xy key to region index
    this.key2RegionIndices = new Map();
    for (let i = 0; i < options.regions.length; i++) {
      let keys = this.binning.keys(options.regions[i]);
      for (let k of keys) {
        let v = this.key2RegionIndices.get(k);
        if (v != null) {
          v.push(i);
        } else {
          this.key2RegionIndices.set(k, [i]);
        }
      }
    }
  }

  /** Add data to the summarizer */
  add(data: { x: ArrayLike<number>; y: ArrayLike<number>; text: ArrayLike<string | null | undefined> }) {
    for (let i = 0; i < data.text.length; i++) {
      let text = data.text[i];
      if (typeof text !== "string" || text.trim().length === 0) continue;
      let words: string[] = [];
      let stems = new Set<string>();
      for (let s of this.segmenter.segment(text)) {
        let word = s.segment.trim();
        let lower = word.toLowerCase();
        if (!s.isWordLike || this.stopWords.has(lower) || /^\p{N}+$/u.test(lower)) continue;
        words.push(word);
        stems.add(stemmer(lower));
      }
      if (words.length === 0) continue;
      this.documentCount++;
      for (let stem of stems) incrementMap(this.documentFrequency, stem, 1);

      // Bins are only a lookup accelerator: do not count nearby points outside the region.
      let x = data.x[i];
      let y = data.y[i];
      let indices = (this.key2RegionIndices.get(this.binning.key(x, y)) ?? []).filter((idx) =>
        this.regions[idx].some((r) => x >= r.xMin && x <= r.xMax && y >= r.yMin && y <= r.yMax),
      );
      // Each point gets one vote per stem, regardless of repetitions or spelling variants.
      for (let idx of indices) {
        this.documentCountPerClass[idx]++;
        for (let stem of stems) incrementMap(this.documentFrequencyPerClass[idx], stem, 1);
      }
      for (let word of words) {
        for (let idx of indices) {
          incrementMap(this.frequencyPerClass[idx], word, 1);
        }
      }
    }
  }

  summarize(limit: number = 4): string[][] {
    let result: Candidate[][] = this.regions.map(() => []);
    if (!Number.isFinite(limit) || limit <= 0) return result.map(() => []);
    limit = Math.floor(limit);
    let counts = this.frequencyPerClass.map(aggregateByStem);
    let lengths = this.documentFrequencyPerClass.map((m) =>
      Array.from(m.values()).reduce((sum, count) => sum + count, 0),
    );
    let groups = new Map<number, number[]>();
    for (let i = 0; i < counts.length; i++) {
      let group = groups.get(this.levels[i]) ?? [];
      group.push(i);
      groups.set(this.levels[i], group);
    }
    for (let indices of groups.values()) {
      // Empty regions neither supply candidates nor distort the length/IDF statistics.
      let active = indices.filter((i) => lengths[i] > 0);
      if (active.length === 0) continue;
      let averageLength = active.reduce((sum, i) => sum + lengths[i], 0) / active.length;
      let regionFrequency = new Map<string, number>();
      for (let i of active) {
        for (let stem of counts[i].keys()) incrementMap(regionFrequency, stem, 1);
      }
      let candidates: Candidate[] = [];
      for (let region of active) {
        for (let [stem, [word]] of counts[region]) {
          let tf = this.documentFrequencyPerClass[region].get(stem)!;
          let coverage = tf / this.documentCountPerClass[region];
          let documentIDF = inverseDocumentFrequency(this.documentCount, this.documentFrequency.get(stem)!);
          let regionIDF = inverseDocumentFrequency(active.length, regionFrequency.get(stem)!);
          let saturation = (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (lengths[region] / averageLength)));
          // Coverage squared limits rare-word IDF rewards for isolated documents.
          candidates.push({ stem, word, region, score: documentIDF * regionIDF * saturation * coverage ** 2 });
        }
      }
      candidates.sort((a, b) => b.score - a.score || compareText(a.stem, b.stem) || a.region - b.region);
      let used = new Set<string>();
      for (let round = 0; round < limit; round++) {
        let assigned = new Set<number>();
        for (let candidate of candidates) {
          if (assigned.has(candidate.region) || used.has(candidate.stem)) continue;
          result[candidate.region].push(candidate);
          assigned.add(candidate.region);
          used.add(candidate.stem);
        }
        if (assigned.size === 0) break;
      }
      // Only after diversification, fill remaining slots with local topic words.
      for (let candidate of candidates) {
        let selected = result[candidate.region];
        if (selected.length < limit && !selected.some((x) => x.stem === candidate.stem)) selected.push(candidate);
      }
    }
    return result.map((selected) => selected.map((x) => x.word));
  }
}

class XYBinning {
  private xMin: number;
  private yMin: number;
  private xStep: number;
  private yStep: number;

  constructor(xMin: number, yMin: number, xStep: number, yStep: number) {
    this.xMin = xMin;
    this.yMin = yMin;
    this.xStep = xStep;
    this.yStep = yStep;
  }

  static inferFromRegions(regions: Rectangle[][]): XYBinning {
    let xMin = Number.POSITIVE_INFINITY;
    let yMin = Number.POSITIVE_INFINITY;
    let xMax = Number.NEGATIVE_INFINITY;
    let yMax = Number.NEGATIVE_INFINITY;
    for (let region of regions) {
      for (let rect of region) {
        if (rect.xMin < xMin) {
          xMin = rect.xMin;
        }
        if (rect.xMax > xMax) {
          xMax = rect.xMax;
        }
        if (rect.yMin < yMin) {
          yMin = rect.yMin;
        }
        if (rect.yMax > yMax) {
          yMax = rect.yMax;
        }
      }
    }
    if (xMin < xMax && yMin < yMax) {
      return new XYBinning(xMin, yMin, (xMax - xMin) / 200, (yMax - yMin) / 200);
    } else {
      return new XYBinning(0, 0, 1, 1);
    }
  }

  key(x: number, y: number) {
    let ix = Math.floor((x - this.xMin) / this.xStep);
    let iy = Math.floor((y - this.yMin) / this.yStep);
    return ix + iy * 32768;
  }

  keys(rects: Rectangle[]): Set<number> {
    let keys = new Set<number>();
    for (let { xMin, yMin, xMax, yMax } of rects) {
      let xiLowerBound = Math.floor((xMin - this.xMin) / this.xStep);
      let xiUpperBound = Math.floor((xMax - this.xMin) / this.xStep);
      let yiLowerBound = Math.floor((yMin - this.yMin) / this.yStep);
      let yiUpperBound = Math.floor((yMax - this.yMin) / this.yStep);
      for (let xi = xiLowerBound; xi <= xiUpperBound; xi++) {
        for (let yi = yiLowerBound; yi <= yiUpperBound; yi++) {
          let p = yi * 32768 + xi;
          keys.add(p);
        }
      }
    }
    return keys;
  }
}

function incrementMap<K>(map: Map<K, number>, key: K, value: number) {
  let c = map.get(key) ?? 0;
  map.set(key, c + value);
}

/** Aggregate words by their stems and track the most frequent version.
 * Returns a map with stemmed words as keys, and the most frequent version and total count as values. */
function aggregateByStem(inputMap: Map<string, number>): Map<string, [string, number]> {
  let result = new Map<string, [string, number]>();
  for (let [word, count] of inputMap.entries()) {
    let lower = word.toLowerCase();
    let s = stemmer(lower);
    let value = result.get(s);
    if (value != null) {
      value[1] += count;
      let previousCount = inputMap.get(value[0]) ?? 0;
      if (previousCount < count || (previousCount === count && compareText(word, value[0]) < 0)) {
        value[0] = word;
      }
    } else {
      result.set(s, [word, count]);
    }
  }
  return result;
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function inverseDocumentFrequency(total: number, frequency: number): number {
  return Math.log1p((total - frequency + 0.5) / (frequency + 0.5));
}
