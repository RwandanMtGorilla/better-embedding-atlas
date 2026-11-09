import { describe, expect, it, vi } from "vitest";
import { layoutTileLabels } from "../src/lib/embedding_view/labels.js";
import {
  labelTileAnalysis,
  labelTileNeighbors,
  labelViewportBounds,
  localLabelGrid,
  localLabelLevel,
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
