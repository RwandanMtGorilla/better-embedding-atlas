<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount } from "svelte";
  import { IconEmbeddingView, IconError, IconRight, IconSpinner } from "../../assets/icons.js";
  import { fetchCollections, type CollectionInfo } from "../services/api.js";

  let collections: CollectionInfo[] = $state([]);
  let loading = $state(true);
  let error = $state<string | null>(null);

  onMount(async () => {
    try {
      collections = await fetchCollections();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  });

  function formatCount(count: number): string {
    if (count >= 1000000) {
      return (count / 1000000).toFixed(1) + "M";
    } else if (count >= 1000) {
      return (count / 1000).toFixed(1) + "K";
    }
    return count.toString();
  }
</script>

<div class="min-h-screen bg-slate-200 dark:bg-slate-900 p-8">
  <div class="max-w-6xl mx-auto">
    <header class="mb-8">
      <h1 class="text-2xl font-semibold text-slate-800 dark:text-slate-200 mb-2">知识库列表</h1>
      <p class="text-slate-500 dark:text-slate-400">选择一个知识库开始探索</p>
    </header>

    {#if loading}
      <div class="flex flex-col items-center justify-center py-16 gap-3">
        <IconSpinner class="w-8 h-8 text-blue-500" />
        <p class="text-slate-500 dark:text-slate-400">正在加载知识库列表...</p>
      </div>
    {:else if error}
      <div class="p-6 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md">
        <div class="flex items-start gap-3">
          <IconError class="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div class="flex-1">
            <h2 class="text-lg font-medium text-slate-800 dark:text-slate-200 mb-1">加载失败</h2>
            <p class="text-slate-500 dark:text-slate-400 mb-3">{error}</p>
            <button
              onclick={() => window.location.reload()}
              class="px-3 py-1.5 h-[28px] text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
            >
              重试
            </button>
          </div>
        </div>
      </div>
    {:else if collections.length === 0}
      <div class="p-12 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md text-center">
        <IconEmbeddingView class="w-12 h-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
        <h2 class="text-lg font-medium text-slate-800 dark:text-slate-200 mb-1">暂无知识库</h2>
        <p class="text-slate-500 dark:text-slate-400">ChromaDB 中没有找到任何知识库</p>
      </div>
    {:else}
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {#each collections as collection}
          <a
            href="#/collection/{encodeURIComponent(collection.name)}"
            class="group flex items-center gap-3 p-3 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md hover:shadow-md dark:hover:border-slate-600 transition-all"
          >
            <div class="flex-shrink-0">
              <IconEmbeddingView class="w-10 h-10 text-blue-500" />
            </div>
            <div class="flex-1 min-w-0">
              <h2 class="text-base font-medium text-slate-800 dark:text-slate-200 truncate mb-0.5">
                {collection.name}
              </h2>
              <p class="text-sm text-slate-500 dark:text-slate-400">
                {formatCount(collection.count)} 条文档
              </p>
              {#if collection.metadata && Object.keys(collection.metadata).length > 0}
                <div class="flex flex-wrap gap-1 mt-1">
                  {#each Object.entries(collection.metadata).slice(0, 2) as [key, value]}
                    <span class="text-xs px-1.5 py-0.5 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded">
                      {key}: {value}
                    </span>
                  {/each}
                </div>
              {/if}
            </div>
            <div class="flex-shrink-0">
              <IconRight class="w-4 h-4 text-slate-400 dark:text-slate-500 group-hover:text-blue-500 transition-colors" />
            </div>
          </a>
        {/each}
      </div>
    {/if}
  </div>
</div>
