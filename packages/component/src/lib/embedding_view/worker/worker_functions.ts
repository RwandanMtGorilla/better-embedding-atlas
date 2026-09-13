// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { findClusters } from "@embedding-atlas/density-clustering";
import { dynamicLabelPlacement } from "../../dynamic_label_placement/dynamic_label_placement.js";
import { TextSummarizer, type TextSummarizerOptions } from "../../text_summarizer/text_summarizer.js";

export { dynamicLabelPlacement, findClusters };

let textSummarizers = new Map<string, TextSummarizer>();

export function textSummarizerCreate(options: TextSummarizerOptions) {
  let key = new Date().getTime() + "-" + Math.random();
  textSummarizers.set(key, new TextSummarizer(options));
  return key;
}

export function textSummarizerDestroy(key: string) {
  return textSummarizers.delete(key);
}

export function textSummarizerAdd(
  key: string,
  data: { x: ArrayLike<number>; y: ArrayLike<number>; text: ArrayLike<string | null | undefined> },
) {
  textSummarizers.get(key)?.add(data);
}

export function textSummarizerSummarize(key: string) {
  return textSummarizers.get(key)?.summarize() ?? [];
}
