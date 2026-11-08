<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Viewer from "./Viewer.svelte";

  import { BackendDataSource } from "./backend_data_source.js";
  import type { CollectionViewOptions } from "./collection_view_state.js";

  interface Props {
    serverUrl?: string;
    revision?: string;
    collection?: CollectionViewOptions;
  }

  let { serverUrl = getDefaultServerUrl(), revision, collection }: Props = $props();

  function getDefaultServerUrl(): string {
    if (import.meta.env.MODE == "development") {
      return "http://localhost:5055/data/";
    }
    return "./data/";
  }

  let dataSource = new BackendDataSource(serverUrl, revision);
</script>

<Viewer dataSource={dataSource} collection={collection} />
