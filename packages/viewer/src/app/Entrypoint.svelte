<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Router from "svelte-spa-router";

  import BackendViewer from "./BackendViewer.svelte";
  import FileViewer from "./FileViewer.svelte";
  import TestDataViewer from "./TestDataViewer.svelte";
  import HomePage from "./pages/HomePage.svelte";
  import CollectionViewer from "./pages/CollectionViewer.svelte";

  import { resolveAppConfig } from "./app_config.js";

  const config = resolveAppConfig();

  // Multi-collection mode routes
  const isMultiCollection = config.home === "multi-collection";

  const routes: any = isMultiCollection
    ? {
        "/": HomePage,
        "/collection/:name": CollectionViewer,
      }
    : {
        "/": config.home == "file-viewer" ? FileViewer : BackendViewer,
      };

  if (import.meta.env.DEV) {
    routes["/test"] = TestDataViewer;
    routes["/file"] = FileViewer;
  }
</script>

<Router routes={routes} />
