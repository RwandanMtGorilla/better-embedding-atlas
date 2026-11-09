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

export type QueryLabelTile = (tile: LabelTile, signal: AbortSignal) => Promise<LocalLabel[]>;

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
  // Leave space for the neighbor ring and retain exact integer bin coordinates.
  if (
    bounds.xMin >= bounds.xMax ||
    bounds.yMin >= bounds.yMax ||
    ![minColumn, maxColumn, minRow, maxRow].every((index) =>
      Number.isSafeInteger((Math.abs(index) + 3) * LABEL_TILE_CELLS),
    )
  ) {
    return [];
  }
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
