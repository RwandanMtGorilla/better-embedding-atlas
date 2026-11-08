<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { debounce } from "@embedding-atlas/utils";
  import { Coordinator } from "@uwdata/mosaic-core";
  import { onMount, untrack } from "svelte";

  import EmbeddingAtlas from "../EmbeddingAtlas.svelte";
  import MessagesView from "./components/MessagesView.svelte";

  import type { EmbeddingAtlasProps, EmbeddingAtlasState } from "../api.js";
  import { systemColorScheme } from "../utils/color_scheme.js";
  import { columnDescriptions } from "../utils/database.js";
  import type { ExportFormat } from "../utils/mosaic_exporter.js";
  import { getQueryPayload, setQueryPayload } from "../utils/query_payload.js";
  import {
    restoreCollectionState,
    type CollectionViewOptions,
    type CollectionViewState,
  } from "./collection_view_state.js";
  import type { DataSource } from "./data_source.js";

  const coordinator = new Coordinator();

  interface Props {
    dataSource: DataSource;
    collection?: CollectionViewOptions;
  }

  let { dataSource, collection }: Props = $props();

  let ready = $state(false);
  let error = $state(false);
  let status = $state("Loading...");
  let initialState: EmbeddingAtlasState | null = $state.raw(null);
  let config: Partial<EmbeddingAtlasProps> | null = $state.raw(null);
  let active = false;
  let stateWrite = 0;

  onMount(() => {
    active = true;
    const initialized = initialize();
    return () => {
      active = false;
      coordinator.clear();
      void initialized.then(() => dataSource.dispose?.()).catch(console.error);
    };
  });

  async function initialize() {
    try {
      const saved: CollectionViewState | null = collection?.state ?? (await getQueryPayload("state")) ?? null;
      if (!active) return;
      status = "Initializing database...";
      const props = await dataSource.initializeCoordinator(coordinator, "dataset", (s) => {
        status = s;
      });
      if (!active) return;

      initialState = saved;
      if (collection && saved) {
        const columns = await columnDescriptions(coordinator, "dataset");
        if (!active) return;
        initialState = restoreCollectionState(
          saved,
          props.data,
          columns,
          collection.fitId,
          dataSource.representativeItemColumn,
        );
      }
      config = props;
      ready = true;
    } catch (e) {
      if (active) {
        error = true;
        status = e instanceof Error ? e.message : String(e);
      }
    }
  }

  async function onExportSelection(predicate: string | null, format: ExportFormat) {
    if (dataSource.downloadSelection) {
      await dataSource.downloadSelection(predicate, format);
    }
  }

  async function onDownloadArchive() {
    if (dataSource.downloadArchive) {
      await dataSource.downloadArchive();
    }
  }

  const saveState = debounce((state: CollectionViewState, write: number) => {
    if (active) {
      void setQueryPayload("state", state, "object", () => active && write === stateWrite);
    }
  }, 200);

  function onStateChange(state: EmbeddingAtlasState) {
    if (!active) return;
    // Writing the snapshot back to the parent must not subscribe the chart's effect to it.
    untrack(() => {
      const saved: CollectionViewState = { ...state, predicate: undefined };
      if (collection) {
        saved.fitId = collection.fitId;
        saved.representativeItemColumn = dataSource.representativeItemColumn;
        collection.onStateChange(saved);
      }
      saveState(saved, ++stateWrite);
    });
  }
</script>

<div class="fixed left-0 right-0 top-0 bottom-0">
  {#if ready && config != null}
    <EmbeddingAtlas
      coordinator={coordinator}
      data={{
        ...(config.data ?? { id: "id" }),
        // table is loaded with the name "dataset" above.
        table: "dataset",
      }}
      searcher={config.searcher}
      embeddingViewConfig={config.embeddingViewConfig}
      embeddingViewLabels={config.embeddingViewLabels}
      defaultChartsConfig={config.defaultChartsConfig}
      chartTheme={config.chartTheme}
      stylesheet={config.stylesheet}
      initialState={initialState ?? config.initialState}
      onExportApplication={dataSource.downloadArchive ? onDownloadArchive : null}
      onExportSelection={dataSource.downloadSelection ? onExportSelection : null}
      onStateChange={onStateChange}
      cache={dataSource.cache}
    />
  {:else}
    <div
      class="w-full h-full grid place-content-center select-none text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
      class:dark={$systemColorScheme == "dark"}
    >
      <MessagesView messages={[{ text: status, error: error }]} />
    </div>
  {/if}
</div>
