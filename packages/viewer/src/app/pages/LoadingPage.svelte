<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { IconSpinner, IconError, IconCheck, IconEmbeddingView } from "../../assets/icons.js";
  import { subscribeProgress, type LoadingProgressEvent, type LogEntry } from "../services/api.js";

  interface Props {
    collectionName: string;
    onReady: () => void;
  }

  let { collectionName, onReady }: Props = $props();

  let progress: LoadingProgressEvent | null = $state(null);
  let logs: LogEntry[] = $state([]);
  let eventSource: EventSource | null = null;

  const statusLabels: Record<string, string> = {
    pending: "等待中",
    loading_metadata: "加载文档和元数据",
    loading_embeddings: "加载向量数据",
    computing_projection: "计算UMAP投影",
    ready: "加载完成",
    error: "加载失败",
  };

  onMount(() => {
    eventSource = subscribeProgress(
      collectionName,
      (event) => {
        progress = event;
        if (event.logs) {
          logs = event.logs;
        }
        if (event.status === "ready") {
          setTimeout(() => {
            onReady();
          }, 500);
        }
      },
      (error) => {
        console.error("Progress stream error:", error);
        progress = {
          status: "error",
          progress: 0,
          message: "连接失败",
          error: "无法连接到服务器",
        };
      },
    );
  });

  onDestroy(() => {
    if (eventSource) {
      eventSource.close();
    }
  });
</script>

<div class="fixed inset-0 flex items-center justify-center bg-slate-200 dark:bg-slate-900 p-4">
  <div class="w-full max-w-2xl">
    <div class="p-6 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-md shadow-lg">
      <!-- Header -->
      <div class="flex items-center gap-3 mb-6">
        <IconEmbeddingView class="w-8 h-8 text-blue-500 flex-shrink-0" />
        <div class="flex-1 min-w-0">
          <h2 class="text-lg font-semibold text-slate-800 dark:text-slate-200 truncate">
            {collectionName}
          </h2>
          <p class="text-sm text-slate-500 dark:text-slate-400">
            {progress ? statusLabels[progress.status] || progress.status : "正在连接..."}
          </p>
        </div>
      </div>

      <!-- Progress Bar -->
      {#if progress}
        <div class="mb-6">
          <div class="flex justify-between items-center mb-2">
            <span class="text-sm text-slate-500 dark:text-slate-400">加载进度</span>
            <span class="text-sm font-mono text-slate-600 dark:text-slate-300">
              {Math.round(progress.progress)}%
            </span>
          </div>
          <div class="h-2 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
            <div
              class="h-full bg-blue-500 transition-all duration-300 ease-out"
              style="width: {progress.progress}%"
            ></div>
          </div>
        </div>

        <!-- Success/Error Messages -->
        {#if progress.status === "ready"}
          <div class="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md mb-4">
            <IconCheck class="w-5 h-5 text-green-500 flex-shrink-0" />
            <span class="text-sm text-green-700 dark:text-green-300">知识库加载成功，正在跳转...</span>
          </div>
        {:else if progress.status === "error"}
          <div class="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md mb-4">
            <IconError class="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div class="flex-1">
              <p class="text-sm text-red-700 dark:text-red-300 font-medium mb-1">加载失败</p>
              {#if progress.error}
                <p class="text-sm text-red-600 dark:text-red-400">{progress.error}</p>
              {/if}
            </div>
          </div>
        {/if}
      {/if}

      <!-- Logs Display -->
      {#if logs.length > 0}
        <div class="mt-4">
          <h3 class="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">加载日志</h3>
          <div class="flex flex-col gap-1 p-3 border rounded-md bg-slate-50 dark:bg-slate-900 border-slate-300 dark:border-slate-700 max-h-64 overflow-y-auto">
            {#each logs as log, i}
              {@const isLast = i === logs.length - 1}
              <div
                class="flex items-start leading-5 {isLast
                  ? 'text-slate-500 dark:text-slate-400'
                  : 'text-slate-300 dark:text-slate-600'}"
              >
                <div class="w-7 flex-none">
                  {#if isLast && !log.error}
                    <IconSpinner class="w-5 h-5 text-blue-500" />
                  {:else if log.error}
                    <IconError class="w-5 h-5 text-red-400" />
                  {/if}
                </div>
                <div class="flex-1" class:text-red-400={log.error}>
                  {log.text}
                </div>
                {#if isLast && log.progress > 0}
                  <div class="flex-none font-mono text-sm">
                    {log.progress.toFixed(0)}%
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Actions -->
      {#if progress?.status === "error"}
        <div class="flex justify-center mt-6">
          <button
            onclick={() => (window.location.hash = "#/")}
            class="px-3 py-1.5 h-[28px] text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
          >
            返回首页
          </button>
        </div>
      {/if}
    </div>
  </div>
</div>
