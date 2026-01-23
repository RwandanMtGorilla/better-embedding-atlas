<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { _ } from "../../i18n";
  import ToggleButton from "../../widgets/ToggleButton.svelte";
  import { getSections, type ListLayoutState } from "./ListLayout.svelte";

  import { IconEmbeddingView, IconMenu, IconTable } from "../../assets/icons.js";

  import type { LayoutOptionsProps } from "../layout.js";

  let { charts, state, onStateChange }: LayoutOptionsProps<ListLayoutState> = $props();

  let sections = $derived(getSections(charts));
</script>

<div class="flex gap-0.5 items-center">
  {#if sections.embedding.length > 0}
    <ToggleButton
      icon={IconEmbeddingView}
      title={$_("layout.showHideEmbedding")}
      bind:checked={
        () => state.showEmbedding ?? true,
        (v) => {
          onStateChange({ showEmbedding: v });
        }
      }
    />
  {/if}
  <ToggleButton
    icon={IconTable}
    title={$_("layout.showHideTable")}
    bind:checked={
      () => state.showTable ?? true,
      (v) => {
        onStateChange({ showTable: v });
      }
    }
  />
  <ToggleButton
    icon={IconMenu}
    title={$_("layout.showHideCharts")}
    bind:checked={
      () => state.showCharts ?? true,
      (v) => {
        onStateChange({ showCharts: v });
      }
    }
  />
</div>
