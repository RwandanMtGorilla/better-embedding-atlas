<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import {
    coordinator as defaultCoordinator,
    isSelection,
    makeClient,
    Priority,
    type Coordinator,
    type MosaicClient,
  } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { untrack } from "svelte";

  import EmbeddingViewImpl from "./EmbeddingViewImpl.svelte";

  import { deepEquals, type Point, type Rectangle, type ViewportState } from "../utils.js";
  import type { EmbeddingViewMosaicProps } from "./embedding_view_mosaic_api.js";
  import { LABEL_TILE_MAP_SIZE, labelTileAnalysis, type LabelTile, type QueryLabelTile } from "./local_label_tiles.js";
  import {
    DataPointQuery,
    predicateForDataPoints,
    predicateForRangeSelection,
    queryApproximateDensity,
  } from "./mosaic_client.js";
  import type { DataPoint, DataPointID } from "./types.js";
  import {
    findLabelTileClusters,
    textSummarizerAdd,
    textSummarizerCreate,
    textSummarizerDestroy,
    textSummarizerSummarize,
  } from "./worker/index.js";

  let {
    coordinator = defaultCoordinator(),
    table,
    x,
    y,
    category = null,
    text = null,
    identifier = null,
    filter = null,
    categoryColors = null,
    tooltip = null,
    additionalFields = null,
    selection = null,
    rangeSelection = null,
    rangeSelectionValue = null,
    width = null,
    height = null,
    pixelRatio = null,
    config = null,
    theme = null,
    viewportState = null,
    labels = null,
    customTooltip = null,
    customOverlay = null,
    onViewportState = null,
    onTooltip = null,
    onSelection = null,
    onRangeSelection = null,
    cache = null,
  }: EmbeddingViewMosaicProps = $props();

  let xData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let yData: Float32Array<ArrayBuffer> = $state.raw(new Float32Array());
  let categoryData: Uint8Array<ArrayBuffer> | null = $state.raw(null);
  let categoryCount: number = $state.raw(1);
  let totalCount: number = $state.raw(1);
  let maxDensity: number = $state.raw(1);
  let defaultViewportState: ViewportState | null = $state.raw(null);

  let effectiveTooltip: DataPoint | null = $state.raw(null);
  let effectiveSelection: DataPoint[] | null = $state.raw(null);
  let effectiveRangeSelection: Rectangle | Point[] | null = $state.raw(null);

  let clientId: MosaicClient | null = $state.raw(null);
  // The predicate for the point data currently available for local labels.
  let labelFilter: { predicate?: SQL.FilterExpr } | null = $state.raw(null);

  $effect(() => {
    // Let Svelte track the dependencies.
    let deps = { coordinator, source: { table, x, y, category }, filter };

    let client: MosaicClient | null = null;
    let didDestroy = false;
    let pendingQueries = 0;
    let requestedFilter: typeof labelFilter = null;

    async function initClient() {
      let source = deps.source;
      xData = new Float32Array();
      yData = new Float32Array();
      categoryData = null;
      let approxDensity = await queryApproximateDensity(deps.coordinator, source);
      if (didDestroy) {
        return;
      }
      let scaler = approxDensity.scaler * 0.95; // shrink a bit so the point is not exactly on the edge.
      defaultViewportState = { x: approxDensity.centerX, y: approxDensity.centerY, scale: scaler };
      totalCount = approxDensity.totalCount;
      maxDensity = approxDensity.maxDensity;
      categoryCount = approxDensity.categoryCount;

      // A client is a thing that queries data from a selection with user-defined query
      client = makeClient({
        coordinator: deps.coordinator,
        selection: deps.filter ?? undefined,
        filterStable: false,
        query: (predicate) => {
          requestedFilter = { predicate };
          return SQL.Query.from(source.table)
            .select({
              x: SQL.sql`${SQL.column(source.x)}::FLOAT`,
              y: SQL.sql`${SQL.column(source.y)}::FLOAT`,
              ...(source.category != null ? { c: SQL.sql`${SQL.column(source.category)}::UTINYINT` } : {}),
            })
            .where(predicate);
        },
        queryPending: () => {
          pendingQueries++;
          if (!didDestroy) {
            labelFilter = null;
          }
        },
        queryError: () => {
          pendingQueries--;
          requestedFilter = null;
        },
        queryResult: (data: any) => {
          if (didDestroy) {
            return;
          }
          pendingQueries--;
          let xArray = data.getChild("x").toArray();
          let yArray = data.getChild("y").toArray();
          let categoryArray = data.getChild("c")?.toArray() ?? null;
          // Ensure that the arrays are typed arrays.
          if (xArray != null && !(xArray instanceof Float32Array)) {
            xArray = new Float32Array(xArray);
          }
          if (yArray != null && !(yArray instanceof Float32Array)) {
            yArray = new Float32Array(yArray);
          }
          if (categoryArray != null && !(categoryArray instanceof Uint8Array)) {
            categoryArray = new Uint8Array(categoryArray);
          }
          xData = xArray;
          yData = yArray;
          categoryData = categoryArray;
          if (pendingQueries === 0) {
            labelFilter = requestedFilter;
          }
          updateTooltip(null);
          updateSelection(null);
        },
      });
      (client as any).reset = () => {
        reset();
      };
      clientId = client;
    }

    initClient();

    return () => {
      didDestroy = true;
      labelFilter = null;
      clientId = null;
      client?.destroy();
    };
  });

  // Tooltip
  $effect(() => {
    if (isSelection(tooltip)) {
      let client = clientId;
      if (client == null) {
        return;
      }
      let captured = tooltip;
      effectiveTooltip = (captured.valueFor(client) ?? null) as any;
      let listener = () => {
        effectiveTooltip = (captured.valueFor(client) ?? null) as any;
      };

      $effect(() => {
        let value = effectiveTooltip;
        let source = { x, y, category, identifier };
        captured.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          predicate: value != null ? predicateForDataPoints(source, [value]) : null,
          value: value,
        });
      });

      captured.addEventListener("value", listener);
      return () => {
        captured.removeEventListener("value", listener);
        captured.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          value: null,
          predicate: null,
        });
      };
    } else if (tooltip == null || typeof tooltip == "object") {
      effectiveTooltip = tooltip;
    } else {
      if (effectiveTooltip?.identifier == tooltip) {
        return;
      }
      let obsolete = false;
      queryPoints([tooltip]).then((value) => {
        if (obsolete) {
          return;
        }
        if (value.length > 0) {
          effectiveTooltip = value[0];
        } else {
          effectiveTooltip = null;
        }
      });
      return () => {
        obsolete = true;
      };
    }
  });

  function updateTooltip(value: DataPoint | null) {
    if (deepEquals(tooltip, value)) {
      return;
    }
    effectiveTooltip = value;
    onTooltip?.(value);
  }

  // Selection
  $effect(() => {
    if (isSelection(selection)) {
      let client = clientId;
      if (client == null) {
        return;
      }
      let captured = selection;
      effectiveSelection = (captured.valueFor(client) ?? null) as any;
      let listener = () => {
        effectiveSelection = (captured.valueFor(client) ?? null) as any;
      };

      $effect(() => {
        let value = effectiveSelection;
        let source = { x, y, category, identifier };
        captured.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          predicate: value != null ? predicateForDataPoints(source, value) : null,
          value: value,
        });
      });

      captured.addEventListener("value", listener);
      return () => {
        captured.removeEventListener("value", listener);
        captured.update({
          source: client,
          clients: new Set<MosaicClient>().add(client),
          value: null,
          predicate: null,
        });
      };
    } else if (selection == null) {
      effectiveSelection = null;
    } else if (selection.length == 0) {
      effectiveSelection = [];
    } else {
      if (selection.every((x) => typeof x == "object")) {
        effectiveSelection = selection;
      } else {
        let obsolete = false;
        queryPoints(selection).then((value) => {
          if (obsolete) {
            return;
          }
          effectiveSelection = value;
        });
        return () => {
          obsolete = true;
        };
      }
    }
  });

  function updateSelection(value: DataPoint[] | null) {
    if (deepEquals(selection, value)) {
      return;
    }
    effectiveSelection = value;
    onSelection?.(value);
  }

  // Range Selection
  $effect(() => {
    let client = clientId;
    if (client == null) {
      return;
    }
    let captured = rangeSelection;
    if (captured == null) {
      return;
    }

    $effect(() => {
      let value = effectiveRangeSelection;
      let source = { x, y };
      let clause = {
        source: client,
        clients: new Set<MosaicClient>().add(client),
        predicate: value != null ? predicateForRangeSelection(source, value) : null,
        value: value,
      };
      captured.update(clause);
      captured.activate(clause);
    });

    return () => {
      captured.update({
        source: client,
        clients: new Set<MosaicClient>().add(client),
        value: null,
        predicate: null,
      });
    };
  });

  $effect(() => {
    if (
      !deepEquals(
        untrack(() => effectiveRangeSelection),
        rangeSelectionValue,
      )
    ) {
      effectiveRangeSelection = rangeSelectionValue;
    }
  });

  // Reset tooltip, selection, and range selection.
  function reset() {
    updateSelection(null);
    updateTooltip(null);
    onRangeSelection?.(null);
    effectiveRangeSelection = null;
  }

  // Point query
  let pointQuery = $derived(
    new DataPointQuery(coordinator, { table, x, y, category, text, identifier, additionalFields }),
  );

  async function querySelection(px: number, py: number, unitDistance: number): Promise<DataPoint | null> {
    return await pointQuery.queryClosestPoint(filter?.predicate?.(clientId), px, py, unitDistance);
  }

  async function queryPoints(identifiers: DataPointID[]): Promise<DataPoint[]> {
    return await pointQuery.queryPoints(identifiers);
  }

  // Cluster Labels
  interface LabelSource {
    coordinator: Coordinator;
    table: string;
    x: string;
    y: string;
    text: string;
    stopWords: string[] | null;
  }

  let configuredStopWords = $derived(config?.autoLabelStopWords ?? null);
  let stopWords = $derived(configuredStopWords == null ? null : Array.from(configuredStopWords));
  let labelDensityThreshold = $derived(config?.autoLabelDensityThreshold ?? undefined);
  let labelSource = $derived(text == null ? null : { coordinator, table, x, y, text, stopWords });
  let queryClusterLabels = $derived.by(() => {
    let source = labelSource;
    return source == null
      ? null
      : (clusters: Rectangle[][], context?: { levels: number[] }) =>
          summarizeClusterLabels(source, clusters, context?.levels);
  });
  let queryLabelTile = $derived.by<QueryLabelTile | null | undefined>(() => {
    let source = labelSource;
    let filterSnapshot = labelFilter;
    let densityThreshold = labelDensityThreshold;
    if (source == null) {
      return undefined;
    }
    if (filterSnapshot == null) {
      return null;
    }
    return (tile, signal, isPrefetch) =>
      generateLabelTile(source, filterSnapshot.predicate, tile, densityThreshold, signal, isPrefetch);
  });

  async function generateLabelTile(
    source: LabelSource,
    predicate: SQL.FilterExpr | undefined,
    tile: LabelTile,
    densityThreshold: number | undefined,
    signal: AbortSignal,
    isPrefetch: () => boolean,
  ) {
    signal.throwIfAborted();
    let priority = () => (isPrefetch() ? Priority.Low : Priority.Normal);
    let { bounds, cellSize, firstColumn, firstRow } = labelTileAnalysis(tile);
    let x = SQL.column(source.x);
    let y = SQL.column(source.y);
    let rowsInTile = SQL.Query.from(source.table).where(
      SQL.gte(x, bounds.xMin),
      SQL.lt(x, bounds.xMax),
      SQL.gte(y, bounds.yMin),
      SQL.lt(y, bounds.yMax),
    );
    if (predicate != null) {
      rowsInTile.where(predicate);
    }
    // Both neighboring tiles use the same world bins throughout their shared buffer.
    let binX = SQL.sql`(FLOOR(${x}::DOUBLE / ${cellSize}) - ${firstColumn})::INTEGER`;
    let binY = SQL.sql`(FLOOR(${y}::DOUBLE / ${cellSize}) - ${firstRow})::INTEGER`;
    let histogram = rowsInTile
      .clone()
      .select({ bin_x: binX, bin_y: binY, bin_count: SQL.sql`COUNT(*)::DOUBLE` })
      .groupby(binX, binY);
    let bins = await source.coordinator.query(histogram, { cache: false, priority: priority() });
    signal.throwIfAborted();
    if (bins.numRows === 0) {
      return [];
    }
    let columns = bins.getChild("bin_x").toArray();
    let rows = bins.getChild("bin_y").toArray();
    let counts = bins.getChild("bin_count").toArray();
    let grid = new Float64Array(LABEL_TILE_MAP_SIZE * LABEL_TILE_MAP_SIZE);
    for (let i = 0; i < counts.length; i++) {
      grid[rows[i] * LABEL_TILE_MAP_SIZE + columns[i]] = counts[i];
    }
    let clusters = await findLabelTileClusters(grid, tile, densityThreshold, signal);
    signal.throwIfAborted();
    if (!clusters.some((cluster) => cluster.id != null)) {
      return [];
    }
    let summaries = await summarizeClusterLabels(
      source,
      clusters.map((cluster) => cluster.rects),
      undefined,
      {
        query: rowsInTile,
        signal,
        priority,
      },
    );
    signal.throwIfAborted();
    return clusters.flatMap(({ id, x, y, sumDensity }, i) => {
      let text = summaries[i];
      return id != null && text != null ? [{ id, x, y, text, priority: sumDensity }] : [];
    });
  }

  async function summarizeClusterLabels(
    { coordinator, table, x, y, text, stopWords }: LabelSource,
    clusters: Rectangle[][],
    levels?: number[],
    local?: { query: SQL.SelectQuery; signal: AbortSignal; priority: () => number },
  ): Promise<(string | null)[]> {
    let signal = local?.signal;
    signal?.throwIfAborted();
    let query = (local?.query ?? SQL.Query.from(table)).select({
      x: SQL.column(x),
      y: SQL.column(y),
      text: SQL.column(text),
    });
    if (local != null) {
      // Sort by the three projected columns so pagination is stable, even if source names differ.
      query.orderby(SQL.literal(1), SQL.literal(2), SQL.literal(3));
    }
    let summarizer = await textSummarizerCreate({
      regions: clusters,
      levels,
      stopWords,
      minRegionDocumentFrequency: local == null ? undefined : 2,
    });
    try {
      let chunkSize = 10000;
      for (let start = 0; ; start += chunkSize) {
        signal?.throwIfAborted();
        let rows = await coordinator.query(query.offset(start).limit(chunkSize), {
          cache: local == null,
          priority: local?.priority() ?? Priority.Normal,
        });
        signal?.throwIfAborted();
        let texts = rows.getChild("text");
        await textSummarizerAdd(summarizer, {
          x: rows.getChild("x").toArray(),
          y: rows.getChild("y").toArray(),
          text: texts.toArray(),
        });
        if (texts.length < chunkSize) {
          break;
        }
      }
      signal?.throwIfAborted();
      let result = await textSummarizerSummarize(summarizer);
      return result.map((words) => {
        if (words.length === 0) {
          return null;
        }
        return words.length > 2 ? words.slice(0, 2).join("-") + "-\n" + words.slice(2).join("-") : words.join("-");
      });
    } finally {
      await textSummarizerDestroy(summarizer);
    }
  }
</script>

<EmbeddingViewImpl
  width={width ?? 800}
  height={height ?? 800}
  pixelRatio={pixelRatio ?? 2}
  theme={theme}
  config={config}
  data={{ x: xData, y: yData, category: categoryData }}
  totalCount={totalCount}
  maxDensity={maxDensity}
  categoryCount={categoryCount}
  categoryColors={categoryColors}
  defaultViewportState={defaultViewportState}
  querySelection={querySelection}
  queryClusterLabels={queryClusterLabels}
  queryLabelTile={queryLabelTile}
  labels={labels}
  customTooltip={customTooltip}
  customOverlay={customOverlay}
  tooltip={effectiveTooltip}
  onTooltip={updateTooltip}
  selection={effectiveSelection}
  onSelection={updateSelection}
  viewportState={viewportState}
  onViewportState={onViewportState}
  rangeSelection={effectiveRangeSelection}
  onRangeSelection={(v) => {
    effectiveRangeSelection = v;
    onRangeSelection?.(v);
  }}
  cache={cache}
/>
