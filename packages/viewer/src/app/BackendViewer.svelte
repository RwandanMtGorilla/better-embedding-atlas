<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Viewer from "./Viewer.svelte";

  import { BackendDataSource } from "./backend_data_source.js";

  interface Props {
    serverUrl?: string;
  }

  let { serverUrl = getDefaultServerUrl() }: Props = $props();

  function getDefaultServerUrl(): string {
    if (import.meta.env.MODE == "development") {
      return "http://localhost:5055/data/";
    }
    return "./data/";
  }

  let dataSource = new BackendDataSource(serverUrl);
</script>

<Viewer dataSource={dataSource} />
