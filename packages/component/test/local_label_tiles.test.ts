import { describe, expect, it, vi } from "vitest";
import { layoutTileLabels } from "../src/lib/embedding_view/labels.js";
import {
  labelTileAnalysis,
  labelTileNeighbors,
  labelTilesWithNeighbors,
  labelViewportBounds,
  localLabelGrid,
  localLabelLevel,
  prefetchLabelTiles,
  visibleLabelTiles,
  type LocalLabel,
} from "../src/lib/embedding_view/local_label_tiles.js";

vi.mock("../src/lib/measure_text.js", () => ({
  measureText: ({ text, fontSize }: { text: string; fontSize: number }) => {
    const lines = text.split("\n");
    return { width: Math.max(...lines.map((line) => Array.from(line).length * 8)), height: lines.length * fontSize };
  },
}));

describe("stable local label tiles", () => {
  it("uses fixed zoom bands and a power-of-two grid sized for the longest side", () => {
    const initial = 0.1;
    expect(localLabelLevel(initial, initial * (4 - 1e-12))).toBeNull();
    expect(localLabelLevel(initial, initial * 4)).toBe(2);
    expect(localLabelLevel(initial, initial * (8 - 1e-12))).toBe(2);
    expect(localLabelLevel(initial, initial * 8)).toBe(3);
    expect(localLabelLevel(initial, initial * 16)).toBe(4);
    expect(localLabelGrid(initial, 2).size).toBe(8);
    expect(localLabelGrid(initial, 3).size).toBe(4);
    expect(localLabelGrid(0.5 / (64 + 2 ** -46), 2).size).toBe(128);
    expect(labelViewportBounds({ x: 0, y: 0, scale: 0.25 }, 1000, 250)).toEqual({
      xMin: -4,
      xMax: 4,
      yMin: -1,
      yMax: 1,
    });
  });

  it("uses half-open tiles with shared world bins on either side of zero", () => {
    const grid = localLabelGrid(0.5, 2);
    const tiles = visibleLabelTiles(grid, { xMin: -1, xMax: 1, yMin: 0, yMax: 1 });
    expect(tiles.map(({ column, row }) => [column, row])).toEqual([
      [-1, 0],
      [0, 0],
    ]);
    const [left, right] = tiles.map(labelTileAnalysis);
    expect(left.firstColumn + 512).toBe(right.firstColumn);
    expect(left.bounds.xMax).toBe(0.125);
    expect(right.bounds.xMin).toBe(-0.125);
  });

  it("uses half-open bounds for the possible peak centers within a tile", () => {
    const grid = localLabelGrid(0.5, 2);
    const columns = (xMin: number, xMax: number) =>
      visibleLabelTiles(grid, { xMin, xMax, yMin: 0, yMax: 1 }).map(({ column }) => column);
    expect(columns(-Number.EPSILON, 1)).toEqual([0]);
    expect(columns(-1 / 1024, 1 / 1024)).toEqual([-1]);
  });

  it("uses the same cross-border collision decisions regardless of which cores are visible", () => {
    const grid = localLabelGrid(0.5, 2);
    const [left, right] = visibleLabelTiles(grid, { xMin: -1, xMax: 1, yMin: 0, yMax: 1 });
    const results = new Map<string, LocalLabel[]>();
    for (const tile of [...labelTileNeighbors(left), ...labelTileNeighbors(right)]) {
      results.set(tile.key, []);
    }
    results.set(left.key, [
      { id: "a", text: "left", x: -0.1, y: 0.5, priority: 10 },
      { id: "long", text: "long".repeat(20), x: -0.55, y: 0.1, priority: 5 },
    ]);
    results.set(right.key, [{ id: "b", text: "right", x: 0.1, y: 0.5, priority: 10 }]);
    const query = vi.fn();
    const place = (visible: (typeof left)[]) => layoutTileLabels({ query, results, visible }, 100, "sans-serif");
    expect(place([left]).map(({ id }) => id)).toEqual(["a", "long"]);
    expect(place([right])).toEqual([]);
    expect(place([left, right])).toEqual(place([left]));
    const [long] = place([left]).filter(({ id }) => id === "long");
    expect(long.text).toMatch(/…$/);
    expect(long.bounds.xMax - long.bounds.xMin).toBeLessThanOrEqual(1);

    results.delete(right.key);
    expect(place([left])).toEqual([]);
  });
});

describe("local label prefetch coverage", () => {
  const initial = 0.5;
  const maxScale = 100 * initial;

  it("covers a one-tile pan in each direction, including the new collision neighbors", () => {
    const width = 350;
    const height = 1000;
    const viewport = { x: -0.18, y: 0.32, scale: 5.6 };
    const grid = localLabelGrid(initial, 3);
    const required = (view: typeof viewport) =>
      labelTilesWithNeighbors(visibleLabelTiles(grid, labelViewportBounds(view, width, height)));
    const current = required(viewport);
    const nearby = prefetchLabelTiles(initial, viewport, width, height, maxScale).filter(({ level }) => level === 3);
    expect(nearby.every(({ key }) => !current.has(key))).toBe(true);
    const ready = new Set([...current.keys(), ...nearby.map(({ key }) => key)]);
    for (const [dx, dy] of [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ]) {
      const next = required({ ...viewport, x: viewport.x + dx * grid.size, y: viewport.y + dy * grid.size });
      expect([...next.keys()].every((key) => ready.has(key))).toBe(true);
    }
  });

  it("covers both adjacent bands for edge and corner zoom anchors", () => {
    const viewport = { x: -0.19, y: 0.32, scale: 5.6 };
    for (const [width, height] of [
      [1000, 350],
      [350, 1000],
      [1000, 1000],
    ]) {
      const candidates = prefetchLabelTiles(initial, viewport, width, height, maxScale);
      const keys = new Set(candidates.map(({ key }) => key));
      expect(keys.size).toBe(candidates.length);
      expect(keys.size).toBeLessThanOrEqual(72);
      for (const level of [2, 4]) {
        const grid = localLabelGrid(initial, level);
        for (const scale of [grid.scale, 1.99 * grid.scale]) {
          const shift = (1 / viewport.scale - 1 / scale) / Math.max(width, height);
          for (const px of [-width, 0, width]) {
            for (const py of [-height, 0, height]) {
              const next = { x: viewport.x + px * shift, y: viewport.y + py * shift, scale };
              const required = labelTilesWithNeighbors(
                visibleLabelTiles(grid, labelViewportBounds(next, width, height)),
              );
              expect([...required.keys()].every((key) => keys.has(key))).toBe(true);
            }
          }
        }
      }
    }
  });

  it("covers the first local band when corner zoom rounds past a tile edge", () => {
    const view = { x: 68, y: 68, scale: 0.25 };
    const scale = 0.75;
    const center = view.x - 1 / view.scale + 1 / scale;
    const next = { x: center, y: center, scale };
    const grid = localLabelGrid(0.125, 2);
    const needed = labelTilesWithNeighbors(visibleLabelTiles(grid, labelViewportBounds(next, 1000, 1000)));
    const warmed = new Set(prefetchLabelTiles(0.125, view, 1000, 1000, 12.5).map(({ key }) => key));
    expect(needed.size).toBeGreaterThan(0);
    expect([...needed.keys()].every((key) => warmed.has(key))).toBe(true);
  });

  it("starts warming at 2x, orders the closer zoom band first and respects the zoom limit", () => {
    const levels = (ratio: number) => [
      ...new Set(
        prefetchLabelTiles(initial, { x: 0.21, y: -0.21, scale: initial * ratio }, 800, 800, maxScale).map(
          ({ level }) => level,
        ),
      ),
    ];
    expect(levels(2 - 1e-12)).toEqual([]);
    expect(levels(2)).toEqual([2]);
    expect(levels(4 - 1e-12)).toEqual([2]);
    expect(levels(4)).toEqual([2, 3]);
    expect(levels(8)).toEqual([3, 2, 4]);
    expect(levels(8 * Math.SQRT2)).toEqual([3, 4, 2]);
    expect(levels(100)).toEqual([6, 5]);
  });
});
