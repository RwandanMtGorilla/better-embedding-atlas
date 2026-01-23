<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { untrack } from "svelte";

  import { _ } from "../../i18n";
  import Button from "../../widgets/Button.svelte";
  import ComboBox from "../../widgets/ComboBox.svelte";
  import SegmentedControl from "../../widgets/SegmentedControl.svelte";
  import Select from "../../widgets/Select.svelte";

  import { EMBEDDING_ATLAS_VERSION } from "../../constants.js";
  import { jsTypeFromDBType } from "../../utils/database.js";

  // Predefined embedding models. The default is the first model.
  const textModels = [
    "Xenova/all-MiniLM-L6-v2",
    "Xenova/paraphrase-multilingual-mpnet-base-v2",
    "Xenova/multilingual-e5-small",
    "Xenova/multilingual-e5-base",
    "Xenova/multilingual-e5-large",
  ];
  const imageModels = [
    "Xenova/dinov2-small",
    "Xenova/dinov2-base",
    "Xenova/dinov2-large",
    "Xenova/dino-vitb8",
    "Xenova/dino-vits8",
    "Xenova/dino-vitb16",
    "Xenova/dino-vits16",
  ];

  export interface Settings {
    version: string;
    text?: string;
    embedding?:
      | {
          precomputed: { x: string; y: string; neighbors?: string };
        }
      | { compute: { column: string; type: "text" | "image"; model: string } };
  }

  interface Props {
    columns: { column_name: string; column_type: string }[];
    onConfirm: (value: Settings) => void;
  }

  let { columns, onConfirm }: Props = $props();

  let embeddingMode = $state<"precomputed" | "from-text" | "from-image" | "none">("precomputed");

  let textColumn: string | undefined = $state(undefined);

  let embeddingXColumn: string | undefined = $state(undefined);
  let embeddingYColumn: string | undefined = $state(undefined);
  let embeddingNeighborsColumn: string | undefined = $state(undefined);
  let embeddingTextColumn: string | undefined = $state(undefined);
  let embeddingTextModel: string | undefined = $state(undefined);
  let embeddingImageColumn: string | undefined = $state(undefined);
  let embeddingImageModel: string | undefined = $state(undefined);

  let numericalColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "number"));
  let stringColumns = $derived(columns.filter((x) => jsTypeFromDBType(x.column_type) == "string"));

  $effect.pre(() => {
    let c = textColumn;
    if (untrack(() => embeddingTextColumn == undefined)) {
      embeddingTextColumn = c;
    }
  });

  function confirm() {
    let value: Settings = { version: EMBEDDING_ATLAS_VERSION, text: textColumn };
    if (embeddingMode == "precomputed" && embeddingXColumn != undefined && embeddingYColumn != undefined) {
      value.embedding = {
        precomputed: {
          x: embeddingXColumn,
          y: embeddingYColumn,
          neighbors: embeddingNeighborsColumn != undefined ? embeddingNeighborsColumn : undefined,
        },
      };
    }
    if (embeddingMode == "from-text" && embeddingTextColumn != undefined) {
      let model = embeddingTextModel?.trim() ?? "";
      if (model == undefined || model == "") {
        model = textModels[0];
      }
      value.embedding = { compute: { column: embeddingTextColumn, type: "text", model: model } };
    }
    if (embeddingMode == "from-image" && embeddingImageColumn != undefined) {
      let model = embeddingImageModel?.trim() ?? "";
      if (model == undefined || model == "") {
        model = imageModels[0];
      }
      value.embedding = { compute: { column: embeddingImageColumn, type: "image", model: model } };
    }
    onConfirm?.(value);
  }
</script>

<div
  class="flex flex-col p-4 w-[40rem] border rounded-md bg-slate-50 border-slate-300 dark:bg-slate-900 dark:border-slate-700"
>
  <div class="flex flex-col gap-2 pb-4">
    <!-- Text column -->
    <h2 class="text-slate-500 dark:text-slate-500">{$_("settings.searchTooltip.title")}</h2>
    <p class="text-sm text-slate-400 dark:text-slate-600">
      {$_("settings.searchTooltip.description")}
    </p>
    <div class="w-full flex flex-row items-center">
      <div class="w-[6rem] dark:text-slate-400">{$_("settings.searchTooltip.textLabel")}</div>
      <Select
        class="flex-1 min-w-0"
        value={textColumn}
        onChange={(v) => (textColumn = v)}
        options={[
          { value: undefined, label: $_("common.none") },
          ...stringColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
        ]}
      />
    </div>
    <div class="my-2"></div>
    <!-- Embedding Config -->
    <h2 class="text-slate-500 dark:text-slate-500">{$_("settings.embeddingView.title")}</h2>
    <p class="text-sm text-slate-400 dark:text-slate-600">
      {$_("settings.embeddingView.description")}
    </p>
    <div class="flex items-start">
      <SegmentedControl
        value={embeddingMode}
        onChange={(v) => (embeddingMode = v as any)}
        options={[
          { value: "precomputed", label: $_("settings.embeddingView.modes.precomputed") },
          { value: "from-text", label: $_("settings.embeddingView.modes.fromText") },
          { value: "from-image", label: $_("settings.embeddingView.modes.fromImage") },
          { value: "none", label: $_("settings.embeddingView.modes.none") },
        ]}
      />
    </div>
    {#if embeddingMode == "precomputed"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">{$_("settings.embeddingView.labels.x")}</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingXColumn}
          onChange={(v) => (embeddingXColumn = v)}
          options={[
            { value: undefined, label: $_("common.none") },
            ...numericalColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">{$_("settings.embeddingView.labels.y")}</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingYColumn}
          onChange={(v) => (embeddingYColumn = v)}
          options={[
            { value: undefined, label: $_("common.none") },
            ...numericalColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">{$_("settings.embeddingView.labels.neighbors")}</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingNeighborsColumn}
          onChange={(v) => (embeddingNeighborsColumn = v)}
          options={[
            { value: undefined, label: $_("common.none") },
            ...columns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        {$_("settings.embeddingView.neighborsHelp")}
      </p>
    {:else if embeddingMode == "from-text"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">{$_("settings.embeddingView.labels.text")}</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingTextColumn}
          onChange={(v) => (embeddingTextColumn = v)}
          options={[
            { value: undefined, label: $_("common.none") },
            ...stringColumns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">{$_("settings.embeddingView.labels.model")}</div>
        <ComboBox
          className="flex-1"
          value={embeddingTextModel}
          placeholder="(default {textModels[0]})"
          onChange={(v) => (embeddingTextModel = v)}
          options={textModels}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        {$_("settings.embeddingView.computeHelp")}
      </p>
    {:else if embeddingMode == "from-image"}
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">{$_("settings.embeddingView.labels.image")}</div>
        <Select
          class="flex-1 min-w-0"
          value={embeddingImageColumn}
          onChange={(v) => (embeddingImageColumn = v)}
          options={[
            { value: undefined, label: $_("common.none") },
            ...columns.map((x) => ({ value: x.column_name, label: `${x.column_name} (${x.column_type})` })),
          ]}
        />
      </div>
      <div class="w-full flex flex-row items-center">
        <div class="w-[6rem] dark:text-slate-400">{$_("settings.embeddingView.labels.model")}</div>
        <ComboBox
          className="flex-1"
          value={embeddingImageModel}
          placeholder="(default {imageModels[0]})"
          onChange={(v) => (embeddingImageModel = v)}
          options={imageModels}
        />
      </div>
      <p class="text-sm text-slate-400 dark:text-slate-600">
        {$_("settings.embeddingView.computeHelp")}
      </p>
    {/if}
  </div>
  <div class="w-full flex flex-row items-center mt-4">
    <div class="flex-1"></div>
    <Button
      label={$_("settings.confirm")}
      class="w-40 justify-center"
      onClick={() => {
        confirm();
      }}
    />
  </div>
</div>
