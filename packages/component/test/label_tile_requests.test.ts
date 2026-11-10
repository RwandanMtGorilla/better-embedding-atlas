import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LabelTileRequests, type LabelTileSnapshot } from "../src/lib/embedding_view/label_tile_requests.js";
import {
  labelTilesWithNeighbors,
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
    requests.setPrefetchAllowed(true);
    requests.setViewport(query, [tileAt(0)], () => [tileAt(5)]);
    await vi.runAllTimersAsync();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(query).toHaveBeenCalledTimes(9);
    expect(publish.mock.lastCall?.[0]?.results.size).toBe(8);
    expect(publish.mock.lastCall?.[0]?.results.has(tileAt(0).key)).toBe(false);

    requests.setViewport(query, [tileAt(0)]);
    await vi.advanceTimersByTimeAsync(300);
    expect(query).toHaveBeenCalledTimes(10);
    expect(publish.mock.lastCall?.[0]?.results.get(tileAt(0).key)).toEqual([]);
  });

  it("waits for quiet and foreground readiness, then publishes warmed tiles on the next viewport update", async () => {
    const publish = vi.fn<(snapshot: LabelTileSnapshot | null) => void>();
    const requests = new LabelTileRequests(publish);
    const current = labelTilesWithNeighbors([tileAt(0)]);
    const next = labelTilesWithNeighbors([tileAt(1)]);
    const candidates = [...next.values()].filter(({ key }) => !current.has(key));
    const getPrefetch = vi.fn(() => candidates);
    let finishForeground!: (labels: LocalLabel[]) => void;
    const query = vi
      .fn<QueryLabelTile>()
      .mockResolvedValue([])
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finishForeground = resolve;
          }),
      );
    requests.setPrefetchAllowed(true);
    requests.setViewport(query, [tileAt(0)], getPrefetch);
    await vi.advanceTimersByTimeAsync(599);
    expect(getPrefetch).not.toHaveBeenCalled();
    expect(query).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(getPrefetch).toHaveBeenCalledTimes(1);
    expect(query).toHaveBeenCalledTimes(1);

    requests.setPrefetchAllowed(false);
    finishForeground([]);
    await vi.advanceTimersByTimeAsync(0);
    expect(query).toHaveBeenCalledTimes(9);
    expect(query.mock.calls.every(([, , isPrefetch]) => !isPrefetch())).toBe(true);
    const publications = publish.mock.calls.length;
    requests.setPrefetchAllowed(true);
    await vi.runAllTimersAsync();
    expect(query.mock.calls.slice(9).map(([tile]) => tile)).toEqual(candidates);
    expect(query.mock.calls.slice(9).every(([, , isPrefetch]) => isPrefetch())).toBe(true);
    expect(publish).toHaveBeenCalledTimes(publications);

    requests.setViewport(query, [tileAt(1)]);
    expect(publish.mock.lastCall?.[0]?.results).toEqual(new Map([...next.keys()].map((key) => [key, []])));
    await vi.runAllTimersAsync();
    expect(query).toHaveBeenCalledTimes(12);
  });

  it("promotes an in-flight prefetch when the tile becomes required", async () => {
    const publish = vi.fn<(snapshot: LabelTileSnapshot | null) => void>();
    const requests = new LabelTileRequests(publish);
    const tile = tileAt(5);
    let finish!: (labels: LocalLabel[]) => void;
    const query = vi
      .fn<QueryLabelTile>()
      .mockResolvedValue([])
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      );
    requests.setPrefetchAllowed(true);
    requests.setViewport(query, [], () => [tile]);
    await vi.advanceTimersByTimeAsync(610);
    const [, signal, isPrefetch] = query.mock.calls[0];
    expect(isPrefetch()).toBe(true);
    requests.setViewport(query, [tile]);
    expect(signal.aborted).toBe(false);
    expect(isPrefetch()).toBe(false);

    finish([]);
    await vi.advanceTimersByTimeAsync(0);
    expect(publish.mock.lastCall?.[0]?.results.get(tile.key)).toEqual([]);
    expect(query).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(300);
    expect(query).toHaveBeenCalledTimes(9);
    expect(query.mock.calls.filter(([requested]) => requested.key === tile.key)).toHaveLength(1);
  });

  it("drains obsolete prefetch work and clears both caches across a source change", async () => {
    const publish = vi.fn<(snapshot: LabelTileSnapshot | null) => void>();
    const requests = new LabelTileRequests(publish);
    const current = tileAt(0);
    const warmed = tileAt(5);
    const pending = tileAt(6);
    let finish!: (labels: LocalLabel[]) => void;
    const obsolete = [{ id: "obsolete", x: 6.5, y: 0.5, text: "obsolete", priority: 1 }];
    const query = vi.fn<QueryLabelTile>((tile) =>
      tile.key === pending.key
        ? new Promise((resolve) => {
            finish = resolve;
          })
        : Promise.resolve(obsolete),
    );
    const replacement = vi.fn<QueryLabelTile>().mockResolvedValue([]);
    requests.setPrefetchAllowed(true);
    requests.setViewport(query, [current], () => [warmed, pending]);
    await vi.advanceTimersByTimeAsync(610);
    expect(query).toHaveBeenCalledTimes(11);
    const [, signal] = query.mock.lastCall!;

    requests.setViewport(query, [tileAt(20)], () => [pending]);
    expect(signal.aborted).toBe(true);
    requests.setViewport(replacement, [current, warmed, pending]);
    expect(publish.mock.lastCall?.[0]?.results.size).toBe(0);
    await vi.advanceTimersByTimeAsync(700);
    expect(replacement).not.toHaveBeenCalled();
    finish(obsolete);
    await vi.advanceTimersByTimeAsync(0);
    expect(replacement.mock.calls[0][0]).toEqual(current);
    expect(publish.mock.lastCall?.[0]?.results.get(pending.key)).toEqual([]);
  });

  it("keeps 72 prefetched results independently and reuses entries shared by successive windows", async () => {
    const requests = new LabelTileRequests(vi.fn());
    const query = vi.fn<QueryLabelTile>().mockResolvedValue([]);
    const visible = [tileAt(0)];
    const original = Array.from({ length: 72 }, (_, i) => tileAt(10 + i));
    const added = Array.from({ length: 12 }, (_, i) => tileAt(100 + i));
    requests.setPrefetchAllowed(true);
    requests.setViewport(query, visible, () => original);
    await vi.runAllTimersAsync();
    expect(query).toHaveBeenCalledTimes(81);

    requests.setViewport(query, visible, () => [...added, ...original.slice(0, 60)]);
    await vi.runAllTimersAsync();
    expect(query.mock.calls.slice(81).map(([tile]) => tile)).toEqual(added);

    const evicted = original[71];
    requests.setViewport(query, visible, () => [original[0], evicted]);
    await vi.runAllTimersAsync();
    expect(query.mock.calls.slice(93).map(([tile]) => tile)).toEqual([evicted]);
  });
});
