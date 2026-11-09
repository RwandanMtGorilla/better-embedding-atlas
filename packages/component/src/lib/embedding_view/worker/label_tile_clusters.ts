// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { findClusters } from "@embedding-atlas/density-clustering";
import type { Rectangle } from "../../utils.js";
import {
  LABEL_TILE_CELLS,
  LABEL_TILE_MAP_SIZE,
  LABEL_TILE_PADDING,
  labelTileAnalysis,
  type LabelTile,
} from "../local_label_tiles.js";

export interface LabelTileCluster {
  id: string | null;
  x: number;
  y: number;
  sumDensity: number;
  rects: Rectangle[];
}

// The tile has about half the raster resolution of the overview, so use half its fine bandwidth.
const blurRadius = 8;
const weights = Array.from({ length: 2 * blurRadius + 1 }, (_, i) => Math.exp(-((i - blurRadius) ** 2) / 12.5));
const weightSum = weights.reduce((a, b) => a + b, 0);
const kernel = weights.map((weight) => weight / weightSum);

function blurCounts(counts: Float64Array): Float32Array {
  let size = LABEL_TILE_MAP_SIZE;
  let horizontal = new Float64Array(counts.length);
  let density = new Float64Array(counts.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let count = counts[y * size + x];
      if (count > 0) {
        for (let dx = Math.max(-blurRadius, -x); dx <= Math.min(blurRadius, size - 1 - x); dx++) {
          horizontal[y * size + x + dx] += count * kernel[dx + blurRadius];
        }
      }
    }
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let count = horizontal[y * size + x];
      if (count > 0) {
        for (let dy = Math.max(-blurRadius, -y); dy <= Math.min(blurRadius, size - 1 - y); dy++) {
          density[(y + dy) * size + x] += count * kernel[dy + blurRadius];
        }
      }
    }
  }
  return new Float32Array(density);
}

export async function findLabelTileClusters(
  counts: Float64Array,
  tile: LabelTile,
  densityThreshold = 0.005,
): Promise<LabelTileCluster[]> {
  let clusters = await findClusters(blurCounts(counts), LABEL_TILE_MAP_SIZE, LABEL_TILE_MAP_SIZE, {
    unionThreshold: 5,
  });
  let maxDensity = clusters.reduce((maximum, cluster) => Math.max(maximum, cluster.sumDensity), 0);
  let { cellSize, firstColumn, firstRow } = labelTileAnalysis(tile);
  let end = LABEL_TILE_PADDING + LABEL_TILE_CELLS;
  return clusters
    .filter(
      (cluster) => cluster.sumDensity / maxDensity > densityThreshold && cluster.boundaryRectApproximation?.length,
    )
    .sort(
      (a, b) =>
        a.maxDensityLocation[1] - b.maxDensityLocation[1] ||
        a.maxDensityLocation[0] - b.maxDensityLocation[0] ||
        a.identifier - b.identifier,
    )
    .map((cluster) => {
      let [column, row] = cluster.maxDensityLocation;
      let ownsPeak = column >= LABEL_TILE_PADDING && column < end && row >= LABEL_TILE_PADDING && row < end;
      let peakColumn = firstColumn + column;
      let peakRow = firstRow + row;
      return {
        id: ownsPeak ? `${tile.key}:${peakColumn}:${peakRow}` : null,
        x: (peakColumn + 0.5) * cellSize,
        y: (peakRow + 0.5) * cellSize,
        sumDensity: cluster.sumDensity,
        rects: cluster.boundaryRectApproximation!.map(([x1, y1, x2, y2]) => ({
          xMin: (firstColumn + Math.min(x1, x2)) * cellSize,
          xMax: (firstColumn + Math.max(x1, x2)) * cellSize,
          yMin: (firstRow + Math.min(y1, y2)) * cellSize,
          yMax: (firstRow + Math.max(y1, y2)) * cellSize,
        })),
      };
    });
}
