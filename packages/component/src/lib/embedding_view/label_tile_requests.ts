// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { labelTilesWithNeighbors, type LabelTile, type LocalLabel, type QueryLabelTile } from "./local_label_tiles.js";

const FOREGROUND_CACHE_SIZE = 64;
const PREFETCH_CACHE_SIZE = 72;

export interface LabelTileSnapshot {
  query: QueryLabelTile;
  visible: LabelTile[];
  results: ReadonlyMap<string, LocalLabel[]>;
}

/** Keep settled tiles visible and drain obsolete work before starting the next tile. */
export class LabelTileRequests {
  private query: QueryLabelTile | null = null;
  private visible: LabelTile[] = [];
  private needed = new Map<string, LabelTile>();
  private results = new Map<string, LocalLabel[]>();
  private prefetched = new Map<string, LocalLabel[]>();
  private prefetch: LabelTile[] = [];
  private prefetchAllowed = false;
  private failed = new Set<string>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private prefetchTimer: ReturnType<typeof setTimeout> | null = null;
  private settled = false;
  private running: { key: string; controller: AbortController } | null = null;

  constructor(private publish: (snapshot: LabelTileSnapshot | null) => void) {}

  setViewport(query: QueryLabelTile | null, visible: LabelTile[], getPrefetch?: () => LabelTile[]) {
    if (query !== this.query) {
      this.clear();
      this.query = query;
    }
    this.stopTimers();
    this.failed.clear();
    if (visible.length !== this.visible.length || visible.some((tile, i) => tile.key !== this.visible[i].key)) {
      this.visible = visible;
      // Prioritize the tiles on screen, then load their fixed neighbors for collision checks.
      this.needed = labelTilesWithNeighbors(visible);
      for (let key of this.needed.keys()) {
        let cached = this.results.get(key) ?? this.prefetched.get(key);
        if (cached != null) {
          this.prefetched.delete(key);
          this.results.delete(key);
          this.results.set(key, cached);
        }
      }
      this.trimResults();
      this.publishSnapshot();
    }
    if (this.running != null && !this.needed.has(this.running.key)) {
      this.running.controller.abort();
    }
    if (query == null) {
      return;
    }
    if (Array.from(this.needed.keys()).some((key) => !this.results.has(key))) {
      this.timer = setTimeout(() => {
        this.timer = null;
        this.settled = true;
        this.runNext();
      }, 300);
    }
    if (getPrefetch != null) {
      this.prefetchTimer = setTimeout(() => {
        this.prefetchTimer = null;
        this.prefetch = getPrefetch().slice(0, PREFETCH_CACHE_SIZE);
        // Keep matching entries ahead of older, unrelated entries as the window fills.
        for (let { key } of this.prefetch) {
          let cached = this.prefetched.get(key);
          if (cached != null) {
            this.cachePrefetched(key, cached);
          }
        }
        this.runNext();
      }, 600);
    }
  }

  setPrefetchAllowed(allowed: boolean) {
    this.prefetchAllowed = allowed;
    if (allowed) {
      this.runNext();
    } else if (this.running != null && !this.needed.has(this.running.key)) {
      this.running.controller.abort();
    }
  }

  clear() {
    this.stopTimers();
    this.running?.controller.abort();
    this.query = null;
    this.visible = [];
    this.needed.clear();
    this.results.clear();
    this.prefetched.clear();
    this.failed.clear();
    this.publish(null);
  }

  private stopTimers() {
    if (this.timer != null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.prefetchTimer != null) {
      clearTimeout(this.prefetchTimer);
      this.prefetchTimer = null;
    }
    this.prefetch = [];
    this.settled = false;
  }

  private trimResults() {
    for (let key of this.results.keys()) {
      if (this.results.size <= FOREGROUND_CACHE_SIZE) {
        break;
      }
      if (!this.needed.has(key)) {
        this.results.delete(key);
      }
    }
  }

  private cachePrefetched(key: string, labels: LocalLabel[]) {
    this.prefetched.delete(key);
    this.prefetched.set(key, labels);
    if (this.prefetched.size > PREFETCH_CACHE_SIZE) {
      let oldest = this.prefetched.keys().next().value;
      if (oldest != null) {
        this.prefetched.delete(oldest);
      }
    }
  }

  private publishSnapshot() {
    if (this.query == null || this.visible.length === 0) {
      this.publish(null);
      return;
    }
    let results = new Map<string, LocalLabel[]>();
    for (let key of this.needed.keys()) {
      let labels = this.results.get(key);
      if (labels != null) {
        results.set(key, labels);
      }
    }
    this.publish({ query: this.query, visible: this.visible, results });
  }

  private runNext() {
    let query = this.query;
    if (this.running != null || query == null) {
      return;
    }
    let missing = Array.from(this.needed.values()).filter(({ key }) => !this.results.has(key));
    if (missing.length > 0) {
      let tile = this.settled ? missing.find(({ key }) => !this.failed.has(key)) : undefined;
      if (tile != null) {
        void this.runTile(query, tile);
      }
      return;
    }
    if (!this.prefetchAllowed || this.prefetchTimer != null) {
      return;
    }
    let tile = this.prefetch.find(
      ({ key }) => !this.results.has(key) && !this.prefetched.has(key) && !this.failed.has(key),
    );
    if (tile != null) {
      // Yield between background tiles so a viewport update can take precedence.
      this.prefetchTimer = setTimeout(() => {
        this.prefetchTimer = null;
        if (this.prefetchAllowed) {
          void this.runTile(query, tile);
        }
      }, 0);
    }
  }

  private async runTile(query: QueryLabelTile, tile: LabelTile) {
    let controller = new AbortController();
    let signal = controller.signal;
    this.running = { key: tile.key, controller };
    let isPrefetch = () => !this.needed.has(tile.key);
    try {
      let labels = await query(tile, signal, isPrefetch);
      if (signal.aborted) {
        return;
      }
      if (isPrefetch()) {
        this.cachePrefetched(tile.key, labels);
      } else {
        this.results.set(tile.key, labels);
        this.trimResults();
        this.publishSnapshot();
      }
    } catch (error) {
      if (!signal.aborted) {
        this.failed.add(tile.key);
        if (!isPrefetch()) {
          console.warn("Could not generate local labels", error);
        }
      }
    } finally {
      this.running = null;
      this.runNext();
    }
  }
}
