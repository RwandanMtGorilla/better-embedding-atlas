// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "../api.js";
import type { ChartSpec } from "../charts/spec/spec.js";
import type { ColumnDesc } from "../utils/database.js";

export interface CollectionViewState extends EmbeddingAtlasState {
  fitId?: string | null;
  representativeItemColumn?: string;
}

export interface CollectionViewOptions {
  fitId: string | null;
  state?: CollectionViewState | null;
  onStateChange: (state: CollectionViewState) => void;
}

export function restoreCollectionState(
  state: CollectionViewState,
  data: EmbeddingAtlasProps["data"] | undefined,
  columns: ColumnDesc[],
  fitId: string | null,
  representativeItemColumn?: string,
): EmbeddingAtlasState {
  const types = new Map(columns.map((column) => [column.name, column.jsType]));
  const projectionChanged = state.fitId !== undefined && state.fitId !== fitId;
  const mapField = (name: string) =>
    name === state.representativeItemColumn ? (representativeItemColumn ?? name) : name;
  const charts = { ...state.charts };
  const chartStates = { ...state.chartStates };

  for (const [id, chart] of Object.entries(charts)) {
    if (chart.type === "embedding" && data?.projection) {
      const category = chart.data.category == null ? undefined : mapField(chart.data.category);
      const type = types.get(category ?? "");
      charts[id] = {
        ...chart,
        data: {
          ...data.projection,
          text: data.text,
          category: type === "string" || type === "number" ? category : undefined,
        },
      };
      if (projectionChanged && chartStates[id]) {
        const { viewport, brush, ...rest } = chartStates[id];
        chartStates[id] = rest;
      }
    } else if (chart.type === "table") {
      const fields: string[] = chart.columns.map(mapField).filter((name: string) => types.has(name));
      charts[id] = {
        ...chart,
        columns: fields.length ? fields : columns.map((c) => c.name).filter((name) => !name.startsWith("__")),
      };
    } else if (chart.type === "count-plot" && typeof chart.data.field === "string") {
      const field = mapField(chart.data.field);
      if (types.has(field)) {
        charts[id] = { ...chart, data: { ...chart.data, field } };
      } else {
        delete charts[id];
        delete chartStates[id];
      }
    } else if (
      (chart.type === "content-viewer" && !types.has(chart.field)) ||
      (chart as ChartSpec).layers?.some(
        (layer) =>
          layer.from == null &&
          Object.values(layer.encoding ?? {}).some(
            (encoding) =>
              encoding && "field" in encoding && typeof encoding.field === "string" && !types.has(encoding.field),
          ),
      )
    ) {
      delete charts[id];
      delete chartStates[id];
    }
  }
  return { ...state, charts, chartStates };
}
