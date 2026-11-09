import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LabelTileRequests, type LabelTileSnapshot } from "../src/lib/embedding_view/label_tile_requests.js";
import {
  localLabelGrid,
  visibleLabelTiles,
  type LocalLabel,
  type QueryLabelTile,
} from "../src/lib/embedding_view/local_label_tiles.js";

const grid = localLabelGrid(0.5, 2);
const tileAt = (column: number) => visibleLabelTiles(grid, { xMin: column, xMax: column + 1, yMin: 0, yMax: 1 })[0];

describe("local label tile requests", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("debounces movement and drains obsolete work before starting its replacement", async () => {
    const publish = vi.fn<(snapshot: LabelTileSnapshot | null) => void>();
    const requests = new LabelTileRequests(publish);
    let finishFirst!: (labels: LocalLabel[]) => void;
    const query = vi
      .fn<QueryLabelTile>()
      .mockResolvedValue([])
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finishFirst = resolve;
          }),
      );

    requests.setViewport(query, [tileAt(0)]);
    await vi.advanceTimersByTimeAsync(299);
    requests.setViewport(query, [tileAt(10)]);
    await vi.advanceTimersByTimeAsync(299);
    expect(query).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(query.mock.calls[0][0]).toEqual(tileAt(10));

    requests.setViewport(query, [tileAt(20)]);
    await vi.advanceTimersByTimeAsync(300);
    expect(query.mock.calls[0][1].aborted).toBe(true);
    expect(query).toHaveBeenCalledTimes(1);
    finishFirst([{ id: "obsolete", x: 10.5, y: 0.5, text: "obsolete", priority: 1 }]);
    await vi.advanceTimersByTimeAsync(0);
    expect(query.mock.calls[1][0]).toEqual(tileAt(20));
    expect(publish.mock.lastCall?.[0]?.results.has(tileAt(10).key)).toBe(false);
    expect(publish.mock.lastCall?.[0]?.results.get(tileAt(20).key)).toEqual([]);
  });

  it("publishes shared tiles immediately and clears them when the source changes", async () => {
    const publish = vi.fn<(snapshot: LabelTileSnapshot | null) => void>();
    const requests = new LabelTileRequests(publish);
    const query = vi.fn<QueryLabelTile>().mockResolvedValue([]);
    requests.setViewport(query, [tileAt(0)]);
    await vi.advanceTimersByTimeAsync(300);
    expect(query).toHaveBeenCalledTimes(9);

    requests.setViewport(query, [tileAt(0), tileAt(1)]);
    expect(publish.mock.lastCall?.[0]?.results.has(tileAt(0).key)).toBe(true);
    await vi.advanceTimersByTimeAsync(300);
    expect(query).toHaveBeenCalledTimes(12);

    requests.setViewport(query, []);
    requests.setViewport(query, [tileAt(0)]);
    await vi.advanceTimersByTimeAsync(300);
    expect(query).toHaveBeenCalledTimes(12);

    const replacement = vi.fn<QueryLabelTile>().mockResolvedValue([]);
    requests.setViewport(replacement, [tileAt(0)]);
    expect(publish.mock.lastCall?.[0]?.results.size).toBe(0);
    await vi.advanceTimersByTimeAsync(300);
    expect(replacement).toHaveBeenCalledTimes(9);
  });

  it("reuses empty tiles and evicts least recently used tiles after 64 entries", async () => {
    const requests = new LabelTileRequests(vi.fn());
    const query = vi.fn<QueryLabelTile>().mockResolvedValue([]);
    async function visit(column: number) {
      requests.setViewport(query, [tileAt(column)]);
      await vi.advanceTimersByTimeAsync(300);
    }
    for (let i = 0; i < 7; i++) {
      await visit(i * 10);
    }
    await visit(0);
    expect(query).toHaveBeenCalledTimes(63);
    await visit(70);
    await visit(0);
    expect(query).toHaveBeenCalledTimes(72);
    await visit(10);
    expect(query).toHaveBeenCalledTimes(80);
  });

  it("leaves failed tiles missing and retries on a later view update", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const publish = vi.fn<(snapshot: LabelTileSnapshot | null) => void>();
    const requests = new LabelTileRequests(publish);
    const query = vi.fn<QueryLabelTile>().mockResolvedValue([]).mockRejectedValueOnce(new Error("query failed"));
    requests.setViewport(query, [tileAt(0)]);
    await vi.advanceTimersByTimeAsync(300);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(publish.mock.lastCall?.[0]?.results.size).toBe(8);
    expect(publish.mock.lastCall?.[0]?.results.has(tileAt(0).key)).toBe(false);

    requests.setViewport(query, [tileAt(0)]);
    await vi.advanceTimersByTimeAsync(300);
    expect(query).toHaveBeenCalledTimes(10);
    expect(publish.mock.lastCall?.[0]?.results.get(tileAt(0).key)).toEqual([]);
  });
});
