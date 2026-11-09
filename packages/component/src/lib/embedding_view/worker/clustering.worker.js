// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import {
  dynamicLabelPlacement,
  findClusters,
  findLabelTileClusters,
  textSummarizerAdd,
  textSummarizerCreate,
  textSummarizerDestroy,
  textSummarizerSummarize,
} from "./worker_functions.js";

/** @type Record<string, (...args: any[]) => any> */
let functions = {
  dynamicLabelPlacement,
  findClusters,
  findLabelTileClusters,
  textSummarizerCreate,
  textSummarizerAdd,
  textSummarizerDestroy,
  textSummarizerSummarize,
};

onmessage = async (msg) => {
  let { id, name, payload } = msg.data;
  try {
    let result = await functions[name](...payload);
    postMessage({ id, payload: result });
  } catch (error) {
    postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};

postMessage({
  ready: true,
});

export {};
