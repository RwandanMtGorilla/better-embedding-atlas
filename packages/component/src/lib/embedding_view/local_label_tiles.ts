// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Rectangle, ViewportState } from "../utils.js";
import type { Label } from "./types.js";

export const LABEL_TILE_CELLS = 512;
export const LABEL_TILE_PADDING = 64;
export const LABEL_TILE_MAP_SIZE = LABEL_TILE_CELLS + 2 * LABEL_TILE_PADDING;

export interface LabelGrid {
  level: number;
  size: number;
  scale: number;
}

export interface LabelTile {
  level: number;
  column: number;
  row: number;
  size: number;
  key: string;
}

export interface LocalLabel extends Label {
  id: string;
  priority: number;
}

export type QueryLabelTile = (tile: LabelTile, signal: AbortSignal, isPrefetch: () => boolean) => Promise<LocalLabel[]>;

export function localLabelLevel(initialScale: number | undefined, scale: number): number | null {
  if (initialScale == null || !(initialScale > 0) || !(scale >= 4 * initialScale)) {
    return null;
  }
  let ratio = scale / initialScale;
  if (!Number.isFinite(ratio)) {
    return null;
  }
  let level = Math.floor(Math.log2(ratio));
  // log2 can round up immediately below a band boundary.
  return scale < initialScale * 2 ** level ? level - 1 : level;
}

export function localLabelGrid(initialScale: number, level: number): LabelGrid {
  let extent = 2 / (4 * initialScale);
  let firstLevelSize = 2 ** Math.ceil(Math.log2(extent));
  if (firstLevelSize < extent) {
    firstLevelSize *= 2;
  }
  return { level, size: firstLevelSize / 2 ** (level - 2), scale: initialScale * 2 ** level };
}

export function labelViewportBounds({ x, y, scale }: ViewportState, width: number, height: number): Rectangle {
  let longSide = Math.max(width, height);
  let halfWidth = width / longSide / scale;
  let halfHeight = height / longSide / scale;
  return { xMin: x - halfWidth, xMax: x + halfWidth, yMin: y - halfHeight, yMax: y + halfHeight };
}

function labelTile(grid: Pick<LabelGrid, "level" | "size">, column: number, row: number): LabelTile {
  let { level, size } = grid;
  return { level, size, column, row, key: `${level}:${size}:${column}:${row}` };
}

export function visibleLabelTiles(grid: LabelGrid, bounds: Rectangle): LabelTile[] {
  let { size } = grid;
  let minColumn = Math.floor(bounds.xMin / size);
  let maxColumn = Math.ceil(bounds.xMax / size) - 1;
  let minRow = Math.floor(bounds.yMin / size);
  let maxRow = Math.ceil(bounds.yMax / size) - 1;
  // Leave space for both neighbor rings and retain exact integer bin coordinates.
  if (
    bounds.xMin >= bounds.xMax ||
    bounds.yMin >= bounds.yMax ||
    ![minColumn, maxColumn, minRow, maxRow].every((index) =>
      Number.isSafeInteger((Math.abs(index) + 3) * LABEL_TILE_CELLS),
    )
  ) {
    return [];
  }
  // A label's density peak is at a cell center. A sliver at the tile edge cannot contain one.
  let halfCell = size / LABEL_TILE_CELLS / 2;
  if ((minColumn + 1) * size - halfCell < bounds.xMin) minColumn++;
  if (maxColumn * size + halfCell >= bounds.xMax) maxColumn--;
  if ((minRow + 1) * size - halfCell < bounds.yMin) minRow++;
  if (maxRow * size + halfCell >= bounds.yMax) maxRow--;
  let tiles: LabelTile[] = [];
  for (let row = minRow; row <= maxRow; row++) {
    for (let column = minColumn; column <= maxColumn; column++) {
      tiles.push(labelTile(grid, column, row));
    }
  }
  return tiles;
}

export function labelTileNeighbors(tile: LabelTile): LabelTile[] {
  let tiles: LabelTile[] = [];
  for (let row = tile.row - 1; row <= tile.row + 1; row++) {
    for (let column = tile.column - 1; column <= tile.column + 1; column++) {
      tiles.push(labelTile(tile, column, row));
    }
  }
  return tiles;
}

export function labelTilesWithNeighbors(tiles: LabelTile[]): Map<string, LabelTile> {
  let result = new Map(tiles.map((tile) => [tile.key, tile]));
  for (let tile of tiles) {
    for (let neighbor of labelTileNeighbors(tile)) {
      result.set(neighbor.key, neighbor);
    }
  }
  return result;
}

function tilesNearBounds(grid: LabelGrid, bounds: Rectangle): Map<string, LabelTile> {
  // Allow for rounding when applying a pan or an anchored zoom to world coordinates.
  let margin = grid.size / LABEL_TILE_CELLS / 4;
  let padded = {
    xMin: bounds.xMin - margin,
    xMax: bounds.xMax + margin,
    yMin: bounds.yMin - margin,
    yMax: bounds.yMax + margin,
  };
  return labelTilesWithNeighbors(visibleLabelTiles(grid, padded));
}

export function prefetchLabelTiles(
  initialScale: number,
  viewport: ViewportState,
  width: number,
  height: number,
  maxScale: number,
): LabelTile[] {
  let { scale } = viewport;
  if (
    !(initialScale > 0 && scale >= 2 * initialScale && width > 0 && height > 0) ||
    !Number.isFinite(scale / initialScale)
  ) {
    return [];
  }
  let level = localLabelLevel(initialScale, scale) ?? 1;
  let bounds = labelViewportBounds(viewport, width, height);
  let pan: LabelTile[] = [];
  if (level >= 2) {
    let grid = localLabelGrid(initialScale, level);
    let required = labelTilesWithNeighbors(visibleLabelTiles(grid, bounds));
    let nearby = tilesNearBounds(grid, bounds);
    let surrounding = labelTilesWithNeighbors(Array.from(nearby.values()));
    for (let key of required.keys()) {
      surrounding.delete(key);
    }
    pan = Array.from(surrounding.values());
  }

  let finerGrid = localLabelGrid(initialScale, level + 1);
  let finer: LabelTile[] = [];
  if (finerGrid.scale <= maxScale) {
    // Zooming in around any point on screen stays within the current bounds.
    finer = Array.from(tilesNearBounds(finerGrid, bounds).values());
  }
  let coarser: LabelTile[] = [];
  if (level > 2) {
    let grid = localLabelGrid(initialScale, level - 1);
    if (grid.scale <= maxScale) {
      // At the lowest scale in that band, add the largest possible shift of the center.
      let envelopeScale = grid.scale / (2 - grid.scale / scale);
      let envelope = labelViewportBounds({ ...viewport, scale: envelopeScale }, width, height);
      coarser = Array.from(tilesNearBounds(grid, envelope).values());
    }
  }
  let finerFirst = scale / (finerGrid.scale / 2) >= Math.SQRT2;
  return finerFirst ? [...pan, ...finer, ...coarser] : [...pan, ...coarser, ...finer];
}

export function labelTileAnalysis(tile: LabelTile) {
  let cellSize = tile.size / LABEL_TILE_CELLS;
  let firstColumn = tile.column * LABEL_TILE_CELLS - LABEL_TILE_PADDING;
  let firstRow = tile.row * LABEL_TILE_CELLS - LABEL_TILE_PADDING;
  let bounds: Rectangle = {
    xMin: firstColumn * cellSize,
    xMax: (firstColumn + LABEL_TILE_MAP_SIZE) * cellSize,
    yMin: firstRow * cellSize,
    yMax: (firstRow + LABEL_TILE_MAP_SIZE) * cellSize,
  };
  return { cellSize, firstColumn, firstRow, bounds };
}
