// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * API client for multi-collection mode.
 */

export interface CollectionInfo {
  name: string;
  count: number;
  metadata?: Record<string, unknown>;
}

export interface LogEntry {
  text: string;
  progress: number;
  error: boolean;
  timestamp: number;
}

export interface LoadingProgressEvent {
  status: "pending" | "loading_metadata" | "loading_embeddings" | "computing_projection" | "ready" | "error";
  progress: number;
  message: string;
  error?: string;
  logs?: LogEntry[];
}

const API_BASE = import.meta.env.DEV ? "http://localhost:5055" : "";

/**
 * Fetch all available collections from ChromaDB.
 */
export async function fetchCollections(): Promise<CollectionInfo[]> {
  const resp = await fetch(`${API_BASE}/api/collections`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch collections: ${resp.statusText}`);
  }
  const data = await resp.json();
  return data.collections;
}

/**
 * Fetch the loading status of a collection.
 */
export async function fetchCollectionStatus(name: string): Promise<LoadingProgressEvent> {
  const resp = await fetch(`${API_BASE}/api/collection/${encodeURIComponent(name)}/status`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch status: ${resp.statusText}`);
  }
  return await resp.json();
}

/**
 * Subscribe to real-time loading progress updates via SSE.
 *
 * @param name Collection name
 * @param onProgress Callback for progress events
 * @param onError Callback for errors
 * @returns EventSource instance (call .close() to unsubscribe)
 */
export function subscribeProgress(
  name: string,
  onProgress: (event: LoadingProgressEvent) => void,
  onError?: (error: Event) => void,
): EventSource {
  const url = `${API_BASE}/api/collection/${encodeURIComponent(name)}/progress`;
  const eventSource = new EventSource(url);

  eventSource.addEventListener("progress", (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data) as LoadingProgressEvent;
      onProgress(data);

      // Auto-close on terminal states
      if (data.status === "ready" || data.status === "error") {
        eventSource.close();
      }
    } catch (err) {
      console.error("Failed to parse progress event:", err);
    }
  });

  eventSource.onerror = (e) => {
    if (onError) {
      onError(e);
    } else {
      console.error("SSE connection error:", e);
    }
    eventSource.close();
  };

  return eventSource;
}

/**
 * Get the base URL for a collection's data endpoints.
 */
export function getCollectionDataUrl(name: string): string {
  return `${API_BASE}/collection/${encodeURIComponent(name)}/data/`;
}
