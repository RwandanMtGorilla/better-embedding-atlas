// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { LabelTile } from "../local_label_tiles.js";
import type { LabelTileCluster } from "./label_tile_clusters.js";
import type * as Functions from "./worker_functions.js";

let global_worker: Promise<Worker> | null = null;
let nextRequestId = 0;
let callbacks = new Map<number, { resolve: (payload: any) => void; reject: (error: Error) => void }>();
function worker(): Promise<Worker> {
  if (global_worker == null) {
    let w = new Worker(new URL("./clustering.worker.js", import.meta.url), { type: "module" });
    let connection = new Promise<Worker>((resolve, reject) => {
      function fail(error: Error) {
        if (global_worker !== connection) {
          return;
        }
        global_worker = null;
        w.terminate();
        reject(error);
        for (let callback of callbacks.values()) {
          callback.reject(error);
        }
        callbacks.clear();
      }
      w.onerror = (event) => fail(new Error(event.message));
      w.onmessageerror = () => fail(new Error("Could not read the label worker response"));
      w.onmessage = (msg) => {
        if (msg.data.ready) {
          resolve(w);
          return;
        }
        let { id, payload, error } = msg.data;
        let callback = callbacks.get(id);
        if (callback != null) {
          callbacks.delete(id);
          if (error != null) {
            callback.reject(new Error(error));
          } else {
            callback.resolve(payload);
          }
        }
      };
    });
    global_worker = connection;
  }
  return global_worker;
}

async function invokeWorker(name: string, payload: any, transfer: Transferable[] = []): Promise<any> {
  let w = await worker();
  return new Promise((resolve, reject) => {
    let id = nextRequestId++;
    callbacks.set(id, { resolve, reject });
    try {
      w.postMessage({ id: id, name: name, payload: payload }, transfer);
    } catch (error) {
      callbacks.delete(id);
      reject(error);
    }
  });
}

type PromiseReturn<F extends (...args: any) => any> = (...args: Parameters<F>) => Promise<ReturnType<F>>;

export let findClusters: PromiseReturn<typeof Functions.findClusters> = (densityMap, width, height, options) => {
  return invokeWorker("findClusters", [densityMap, width, height, options], [densityMap.buffer]);
};

export async function findLabelTileClusters(
  counts: Float64Array,
  tile: LabelTile,
  threshold: number | undefined,
  signal: AbortSignal,
): Promise<LabelTileCluster[]> {
  signal.throwIfAborted();
  // A fresh WASM instance gives each tile the same tie-breaking state for density merges.
  let w = new Worker(new URL("./clustering.worker.js", import.meta.url), { type: "module" });
  let abort: (() => void) | undefined;
  try {
    return await new Promise((resolve, reject) => {
      abort = () => reject(signal.reason);
      signal.addEventListener("abort", abort, { once: true });
      w.onerror = (event) => reject(new Error(event.message));
      w.onmessageerror = () => reject(new Error("Could not read the tile clustering response"));
      w.onmessage = ({ data: { ready, payload, error } }) => {
        if (ready) return;
        if (error != null) reject(new Error(error));
        else resolve(payload);
      };
      w.postMessage({ id: 0, name: "findLabelTileClusters", payload: [counts, tile, threshold] }, [counts.buffer]);
    });
  } finally {
    if (abort != null) signal.removeEventListener("abort", abort);
    w.terminate();
  }
}

export let dynamicLabelPlacement: PromiseReturn<typeof Functions.dynamicLabelPlacement> = (...args) => {
  return invokeWorker("dynamicLabelPlacement", args);
};

export let textSummarizerCreate: PromiseReturn<typeof Functions.textSummarizerCreate> = (...args) => {
  return invokeWorker("textSummarizerCreate", args);
};

export let textSummarizerDestroy: PromiseReturn<typeof Functions.textSummarizerDestroy> = (...args) => {
  return invokeWorker("textSummarizerDestroy", args);
};

export let textSummarizerAdd: PromiseReturn<typeof Functions.textSummarizerAdd> = (...args) => {
  return invokeWorker("textSummarizerAdd", args);
};

export let textSummarizerSummarize: PromiseReturn<typeof Functions.textSummarizerSummarize> = (...args) => {
  return invokeWorker("textSummarizerSummarize", args);
};
