<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import LoadingPage from "./LoadingPage.svelte";
  import BackendViewer from "../BackendViewer.svelte";
  import type { CollectionViewState } from "../collection_view_state.js";
  import {
    getCollectionDataUrl,
    requestFullCollectionSync,
    subscribeProgress,
    type LoadingProgressEvent,
  } from "../services/api.js";

  interface Props {
    params: { name: string };
  }

  let { params }: Props = $props();

  let collectionName = $derived(params.name);
  let progress = $state.raw<LoadingProgressEvent | null>(null);
  let savedState = $state.raw<CollectionViewState | null>(null);
  let connectionError = $state(false);
  let syncRequested = $state(false);
  let syncRequestError = $state(false);

  $effect.pre(() => {
    const name = collectionName;
    progress = null;
    savedState = null;
    connectionError = false;
    syncRequested = false;
    syncRequestError = false;
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

  async function syncCollection() {
    const name = collectionName;
    syncRequested = true;
    syncRequestError = false;
    try {
      await requestFullCollectionSync(name);
    } catch {
      if (name === collectionName) syncRequestError = true;
    } finally {
      if (name === collectionName) syncRequested = false;
    }
  }

  let notice = $derived(
    connectionError
      ? "连接中断，正在重连..."
      : syncRequestError
        ? "无法启动全量同步，请重试"
        : progress?.refresh_error
          ? "更新失败，将自动重试"
          : progress?.refreshing
            ? (progress.logs?.at(-1)?.text ?? "正在更新数据...")
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

{#if notice || progress?.revision}
  <div class="fixed bottom-3 right-3 z-50 flex items-center gap-2">
    {#if notice}
      <p role="status" class="rounded bg-slate-800 px-3 py-2 text-sm text-white">{notice}</p>
    {/if}
    {#if progress?.revision}
      <button
        type="button"
        onclick={syncCollection}
        disabled={syncRequested || progress.refreshing}
        title="重新核对全部记录，包括同一 ID 被覆盖的内容"
        class="rounded border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
      >
        全量同步
      </button>
    {/if}
  </div>
{/if}
