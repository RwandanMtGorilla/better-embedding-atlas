// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { labelTileNeighbors, type LabelTile, type LocalLabel, type QueryLabelTile } from "./local_label_tiles.js";

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
  private failed = new Set<string>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private settled = false;
  private running: { key: string; controller: AbortController } | null = null;

  constructor(private publish: (snapshot: LabelTileSnapshot | null) => void) {}

  setViewport(query: QueryLabelTile | null, visible: LabelTile[]) {
    if (query !== this.query) {
      this.clear();
      this.query = query;
    }
    this.stopTimer();
    this.failed.clear();
    if (visible.length !== this.visible.length || visible.some((tile, i) => tile.key !== this.visible[i].key)) {
      this.visible = visible;
      // Prioritize the tiles on screen, then load their fixed neighbors for collision checks.
      this.needed = new Map(visible.map((tile) => [tile.key, tile]));
      for (let tile of visible) {
        for (let neighbor of labelTileNeighbors(tile)) {
          this.needed.set(neighbor.key, neighbor);
        }
      }
      for (let key of this.needed.keys()) {
        let cached = this.results.get(key);
        if (cached != null) {
          this.results.delete(key);
          this.results.set(key, cached);
        }
      }
      this.publishSnapshot();
    }
    if (this.running != null && !this.needed.has(this.running.key)) {
      this.running.controller.abort();
    }
    if (query != null && Array.from(this.needed.keys()).some((key) => !this.results.has(key))) {
      this.timer = setTimeout(() => {
        this.timer = null;
        this.settled = true;
        void this.runNext();
      }, 300);
    }
  }

  clear() {
    this.stopTimer();
    this.running?.controller.abort();
    this.query = null;
    this.visible = [];
    this.needed.clear();
    this.results.clear();
    this.failed.clear();
    this.publish(null);
  }

  private stopTimer() {
    if (this.timer != null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.settled = false;
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

  private async runNext() {
    let query = this.query;
    if (!this.settled || this.running != null || query == null) {
      return;
    }
    let tile = Array.from(this.needed.values()).find(({ key }) => !this.results.has(key) && !this.failed.has(key));
    if (tile == null) {
      return;
    }
    let controller = new AbortController();
    let signal = controller.signal;
    this.running = { key: tile.key, controller };
    try {
      let labels = await query(tile, signal);
      if (signal.aborted) {
        return;
      }
      this.results.set(tile.key, labels);
      if (this.results.size > 64) {
        let oldest = Array.from(this.results.keys()).find((key) => !this.needed.has(key));
        if (oldest != null) {
          this.results.delete(oldest);
        }
      }
      this.publishSnapshot();
    } catch (error) {
      if (!signal.aborted) {
        this.failed.add(tile.key);
        console.warn("Could not generate local labels", error);
      }
    } finally {
      this.running = null;
      void this.runNext();
    }
  }
}
