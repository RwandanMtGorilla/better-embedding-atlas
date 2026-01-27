<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount } from "svelte";
  import LoadingPage from "./LoadingPage.svelte";
  import BackendViewer from "../BackendViewer.svelte";
  import { fetchCollectionStatus, getCollectionDataUrl } from "../services/api.js";

  interface Props {
    params: { name: string };
  }

  let { params }: Props = $props();

  let isReady = $state(false);
  let serverUrl = $derived(getCollectionDataUrl(params.name));

  // Only run on mount (will re-mount when key changes)
  onMount(async () => {
    try {
      const status = await fetchCollectionStatus(params.name);
      if (status.status === "ready") {
        isReady = true;
      }
    } catch (err) {
      // Collection not loaded yet, will show loading page
      console.log("Collection not loaded yet, will trigger loading");
    }
  });

  function handleReady() {
    isReady = true;
  }
</script>

{#key params.name}
  {#if isReady}
    <BackendViewer {serverUrl} />
  {:else}
    <LoadingPage collectionName={params.name} onReady={handleReady} />
  {/if}
{/key}
