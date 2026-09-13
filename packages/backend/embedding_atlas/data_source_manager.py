# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Data source manager for multi-collection mode with LRU caching."""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .chroma_client import ChromaDBClient, chroma_data_to_dataframe
from .data_source import DataSource
from .incremental_umap import (
    IncrementalUMAPConfig,
    IncrementalUMAPProcessor,
)
from .options import make_embedding_atlas_props
from .projection import Projection, _run_umap
from .question_clustering import cached_cluster_questions, add_question_cluster_column
from .utils import Hasher, cache_path
from .version import __version__

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LoadingStatus(Enum):
    """Status of data source loading."""

    PENDING = "pending"
    LOADING_METADATA = "loading_metadata"
    LOADING_EMBEDDINGS = "loading_embeddings"
    COMPUTING_PROJECTION = "computing_projection"
    READY = "ready"
    ERROR = "error"


@dataclass
class LoadingProgress:
    """Progress information for data source loading."""

    status: LoadingStatus
    progress: float = 0.0  # 0-100
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class ManagedDataSource:
    """A managed data source with loading state."""

    collection_name: str
    data_source: DataSource | None = None
    loading_progress: LoadingProgress = field(
        default_factory=lambda: LoadingProgress(LoadingStatus.PENDING)
    )
    loading_task: asyncio.Task | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    logs: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_log(self, message: str, progress: float = 0.0, error: bool = False):
        """Add a log entry and notify subscribers."""
        log_entry = {
            "text": message,
            "progress": progress,
            "error": error,
            "timestamp": time.time(),
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.collection_name}] {message}")
        # Notify subscribers
        for queue in self.subscribers:
            try:
                queue.put_nowait(self.loading_progress)
            except asyncio.QueueFull:
                pass

    def update_progress(
        self,
        status: LoadingStatus,
        progress: float = 0.0,
        message: str = "",
        error: str | None = None,
    ):
        """Update progress and notify all subscribers."""
        self.loading_progress = LoadingProgress(
            status=status,
            progress=progress,
            message=message,
            error=error,
        )
        # Notify subscribers
        for queue in self.subscribers:
            try:
                queue.put_nowait(self.loading_progress)
            except asyncio.QueueFull:
                pass  # Skip if queue is full


def find_column_name(existing_names, candidate):
    """Find a unique column name."""
    if candidate not in existing_names:
        return candidate
    else:
        index = 1
        while True:
            s = f"{candidate}_{index}"
            if s not in existing_names:
                return s
            index += 1


class DataSourceManager:
    """
    Manages multiple ChromaDB data sources with LRU caching.

    This class handles:
    - Listing available collections from ChromaDB
    - Loading data sources on demand
    - LRU eviction when max_cached is exceeded
    - Progress reporting via async queues
    """

    def __init__(
        self,
        chroma_host: str,
        chroma_port: int,
        max_cached: int = 5,
        umap_args: dict | None = None,
        duckdb_uri: str = "server",
        enable_incremental: bool = False,
        incremental_config: IncrementalUMAPConfig | None = None,
    ):
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.max_cached = max_cached
        self.umap_args = umap_args or {}
        self.duckdb_uri = duckdb_uri
        self.enable_incremental = enable_incremental
        self.incremental_config = incremental_config or IncrementalUMAPConfig()

        # LRU cache using OrderedDict
        self._sources: OrderedDict[str, ManagedDataSource] = OrderedDict()
        self._lock = threading.RLock()

        # ChromaDB client for listing collections
        self._chroma_client: ChromaDBClient | None = None

    def _get_chroma_client(self) -> ChromaDBClient:
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            self._chroma_client = ChromaDBClient(
                host=self.chroma_host, port=self.chroma_port
            )
        return self._chroma_client

    def list_collections(self) -> list[dict]:
        """List all available collections from ChromaDB."""
        try:
            client = self._get_chroma_client()
            return client.list_collections_with_info()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def get_managed(self, collection_name: str) -> ManagedDataSource:
        """Get or create a managed data source entry."""
        with self._lock:
            if collection_name not in self._sources:
                self._sources[collection_name] = ManagedDataSource(
                    collection_name=collection_name
                )
            else:
                # Move to end (most recently used)
                self._sources.move_to_end(collection_name)
            return self._sources[collection_name]

    def subscribe_progress(self, collection_name: str) -> asyncio.Queue:
        """Subscribe to progress updates for a collection."""
        managed = self.get_managed(collection_name)
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        managed.subscribers.append(queue)

        # Send current progress immediately
        try:
            queue.put_nowait(managed.loading_progress)
        except asyncio.QueueFull:
            pass

        return queue

    def unsubscribe_progress(self, collection_name: str, queue: asyncio.Queue):
        """Unsubscribe from progress updates."""
        with self._lock:
            if collection_name in self._sources:
                managed = self._sources[collection_name]
                if queue in managed.subscribers:
                    managed.subscribers.remove(queue)

    async def get_or_load(self, collection_name: str) -> ManagedDataSource:
        """Get a data source, loading it if necessary."""
        managed = self.get_managed(collection_name)

        # Already loaded
        if managed.data_source is not None:
            return managed

        # Already loading
        if managed.loading_task is not None and not managed.loading_task.done():
            await managed.loading_task
            return managed

        # Start loading
        managed.loading_task = asyncio.create_task(self._load_data_source(managed))
        await managed.loading_task
        return managed

    def _evict_if_needed(self):
        """Evict least recently used data sources if over limit."""
        with self._lock:
            while len(self._sources) > self.max_cached:
                # Find oldest entry that is ready and has no subscribers
                for name in list(self._sources.keys()):
                    managed = self._sources[name]
                    if (
                        managed.loading_progress.status == LoadingStatus.READY
                        and len(managed.subscribers) == 0
                    ):
                        logger.info(f"Evicting data source: {name}")
                        del self._sources[name]
                        break
                else:
                    # No evictable entries found
                    break

    async def _load_data_source(self, managed: ManagedDataSource):
        """Load a data source from ChromaDB."""
        collection_name = managed.collection_name

        try:
            # Read paired documents and embeddings in a single response.
            managed.add_log("开始加载知识库", progress=0)
            managed.update_progress(
                LoadingStatus.LOADING_METADATA,
                progress=0,
                message="正在从 ChromaDB 获取文档、元数据和向量...",
            )
            managed.add_log("正在从 ChromaDB 获取文档、元数据和向量...", progress=5)

            client = self._get_chroma_client()
            data = await asyncio.to_thread(
                client.get_snapshot, collection_name
            )
            row_count = len(data.ids)

            managed.add_log(f"已获取 {row_count} 条文档", progress=20)
            managed.update_progress(
                LoadingStatus.LOADING_METADATA,
                progress=20,
                message=f"已获取 {row_count} 条文档",
            )

            embeddings = data.embeddings
            managed.add_log("正在进行代表问法聚类...", progress=25)
            managed.update_progress(
                LoadingStatus.COMPUTING_PROJECTION,
                progress=25,
                message="正在进行代表问法聚类...",
            )
            clusters = await asyncio.to_thread(cached_cluster_questions, data.documents, embeddings)
            managed.add_log(
                f"聚类完成：{len(clusters.representatives)} 个代表问法；"
                f"精确问法覆盖率 {clusters.unique_coverage:.2%}，"
                f"请求加权覆盖率 {clusters.request_coverage:.2%}",
                progress=50,
            )

            # Content fingerprints invalidate projections after vector edits too.
            hasher = Hasher()
            hasher.update({
                "version": 2,
                "source": "chromadb",
                "collection": collection_name,
                "ids": data.ids,
                "embeddings": embeddings,
                "umap_args": self.umap_args,
            })
            legacy_cache_file = cache_path("projections") / hasher.hexdigest()
            managed.update_progress(
                LoadingStatus.COMPUTING_PROJECTION,
                progress=60,
                message="正在计算 UMAP 投影...",
            )
            if self.enable_incremental:
                source_hasher = Hasher()
                source_hasher.update([self.chroma_host, self.chroma_port, collection_name, self.umap_args])
                incremental_cache_file = cache_path("projections") / ("questions_" + source_hasher.hexdigest())
                # Changing an existing vector is treated as removal + addition,
                # causing the incremental processor to recompute the projection.
                content_ids = []
                for row_id, vector in zip(data.ids, embeddings):
                    row_hasher = Hasher()
                    row_hasher.update([row_id, vector])
                    content_ids.append(row_hasher.hexdigest())
                processor = IncrementalUMAPProcessor(self.incremental_config)
                result = await asyncio.to_thread(
                    processor.compute_or_update, embeddings, content_ids,
                    incremental_cache_file, self.umap_args,
                )
                proj = result.to_projection()
            elif Projection.exists(legacy_cache_file):
                proj = await asyncio.to_thread(Projection.load, legacy_cache_file)
            else:
                proj = await asyncio.to_thread(_run_umap, embeddings, self.umap_args)
                await asyncio.to_thread(Projection.save, legacy_cache_file, proj)
            managed.add_log("UMAP 投影已就绪", progress=78)

            managed.add_log("正在构建数据框...", progress=80)
            managed.update_progress(
                LoadingStatus.COMPUTING_PROJECTION,
                progress=80,
                message="正在构建数据框...",
            )

            # Step 4: Build DataFrame
            df = chroma_data_to_dataframe(data)
            add_question_cluster_column(df, clusters.labels)

            # Add projection columns
            x_column = find_column_name(df.columns, "projection_x")
            y_column = find_column_name(df.columns, "projection_y")
            neighbors_column = find_column_name(df.columns, "__neighbors")

            df[x_column] = proj.projection[:, 0]
            df[y_column] = proj.projection[:, 1]
            df[neighbors_column] = [
                {"distances": d.tolist(), "ids": i.tolist()}
                for i, d in zip(proj.knn_indices, proj.knn_distances)
            ]

            # Add row index
            id_column = find_column_name(df.columns, "__row_index__")
            df[id_column] = range(df.shape[0])

            # Build ChromaDB ID to row index mapping
            chroma_id_to_row_index = {
                chroma_id: idx for idx, chroma_id in enumerate(data.ids)
            }

            # Build props
            props = make_embedding_atlas_props(
                row_id=id_column,
                x=x_column,
                y=y_column,
                neighbors=neighbors_column,
                text="document",
            )

            metadata = {"props": props}

            # Build identifier
            hasher = Hasher()
            hasher.update(__version__)
            hasher.update(collection_name)
            hasher.update(metadata)
            identifier = hasher.hexdigest()

            # Build chroma config
            chroma_config = {
                "host": self.chroma_host,
                "port": self.chroma_port,
                "collection": collection_name,
                "id_to_row_index": chroma_id_to_row_index,
            }

            managed.add_log("正在初始化数据源...", progress=95)
            managed.update_progress(
                LoadingStatus.COMPUTING_PROJECTION,
                progress=95,
                message="正在初始化数据源...",
            )

            # Create DataSource
            data_source = DataSource(
                identifier, df, metadata, chroma_config=chroma_config
            )
            managed.data_source = data_source

            managed.add_log("加载完成", progress=100)
            managed.update_progress(
                LoadingStatus.READY,
                progress=100,
                message="加载完成",
            )

            # Evict old entries if needed
            self._evict_if_needed()

            logger.info(f"Successfully loaded collection: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {e}")
            managed.add_log(f"加载失败: {str(e)}", progress=0, error=True)
            managed.update_progress(
                LoadingStatus.ERROR,
                progress=0,
                message="加载失败",
                error=str(e),
            )
