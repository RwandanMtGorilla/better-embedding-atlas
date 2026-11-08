<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import LoadingPage from "./LoadingPage.svelte";
  import BackendViewer from "../BackendViewer.svelte";
  import type { CollectionViewState } from "../collection_view_state.js";
  import { getCollectionDataUrl, subscribeProgress, type LoadingProgressEvent } from "../services/api.js";

  interface Props {
    params: { name: string };
  }

  let { params }: Props = $props();

  let collectionName = $derived(params.name);
  let progress = $state.raw<LoadingProgressEvent | null>(null);
  let savedState = $state.raw<CollectionViewState | null>(null);
  let connectionError = $state(false);

  $effect.pre(() => {
    const name = collectionName;
    progress = null;
    savedState = null;
    connectionError = false;
    const stream = subscribeProgress(
      name,
      (event) => {
        progress = event;
        connectionError = false;
      },
      () => {
        connectionError = true;
      },
    );
    return () => stream.close();
  });

  let notice = $derived(
    connectionError
      ? "连接中断，正在重连..."
      : progress?.refresh_error
        ? "更新失败，将自动重试"
        : progress?.refreshing
          ? "正在更新数据..."
          : null,
  );
</script>

{#if progress?.revision && progress.row_count !== 0}
  {#key `${collectionName}/${progress.revision}`}
    <BackendViewer
      serverUrl={getCollectionDataUrl(collectionName)}
      revision={progress.revision}
      collection={{
        fitId: progress.fit_id ?? null,
        state: savedState,
        onStateChange: (state) => {
          savedState = state;
        },
      }}
    />
  {/key}
{:else}
  <LoadingPage collectionName={collectionName} progress={progress} />
{/if}

{#if notice}
  <p role="status" class="fixed bottom-3 right-3 z-50 rounded bg-slate-800 px-3 py-2 text-sm text-white">
    {notice}
  </p>
{/if}
