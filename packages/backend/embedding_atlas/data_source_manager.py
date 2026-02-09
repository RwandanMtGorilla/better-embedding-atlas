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
    IncrementalProjectionCache,
    IncrementalUMAPConfig,
    IncrementalUMAPProcessor,
)
from .options import make_embedding_atlas_props
from .projection import Projection, _run_umap
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
            # Step 1: Load documents and metadata
            managed.add_log("开始加载知识库", progress=0)
            managed.update_progress(
                LoadingStatus.LOADING_METADATA,
                progress=0,
                message="正在从 ChromaDB 获取文档和元数据...",
            )
            managed.add_log("正在从 ChromaDB 获取文档和元数据...", progress=5)

            client = self._get_chroma_client()
            data = await asyncio.to_thread(
                client.get_documents_and_metadata, collection_name
            )
            row_count = len(data.ids)

            managed.add_log(f"已获取 {row_count} 条文档", progress=20)
            managed.update_progress(
                LoadingStatus.LOADING_METADATA,
                progress=20,
                message=f"已获取 {row_count} 条文档",
            )

            # Step 2: Check cache and compute projection
            # Use collection name as cache key for incremental mode
            incremental_cache_file = cache_path("projections") / f"incremental_{collection_name}"

            # Legacy cache key based on content hash
            hasher = Hasher()
            hasher.update(
                {
                    "version": 1,
                    "source": "chromadb",
                    "collection": collection_name,
                    "documents": data.documents,
                    "row_count": row_count,
                    "umap_args": self.umap_args,
                }
            )
            cache_key = hasher.hexdigest()
            legacy_cache_file = cache_path("projections") / cache_key

            proj = None

            # Try incremental mode first if enabled
            if self.enable_incremental:
                managed.add_log("增量模式已启用，检查增量缓存...", progress=25)

                if IncrementalProjectionCache.supports_incremental(incremental_cache_file):
                    managed.add_log("发现增量缓存，正在加载...", progress=30)
                    managed.update_progress(
                        LoadingStatus.LOADING_METADATA,
                        progress=30,
                        message="发现增量缓存，正在检查...",
                    )

                    # Load cached IDs to check if we need incremental update
                    try:
                        cached = await asyncio.to_thread(
                            IncrementalProjectionCache.load,
                            incremental_cache_file,
                            False,  # Don't load model yet
                        )

                        cached_id_set = set(cached.ids)
                        new_id_set = set(data.ids)

                        # Check if IDs match exactly
                        if cached_id_set == new_id_set and cached.ids == data.ids:
                            managed.add_log("ID 完全匹配，使用缓存投影", progress=35)
                            proj = cached.to_projection()
                        elif new_id_set - cached_id_set:
                            # New IDs added - try incremental update
                            added_count = len(new_id_set - cached_id_set)
                            removed_count = len(cached_id_set - new_id_set)

                            if removed_count > 0:
                                managed.add_log(
                                    f"检测到 {removed_count} 条数据被删除，需要全量重算",
                                    progress=35,
                                )
                            else:
                                managed.add_log(
                                    f"检测到 {added_count} 条新数据，尝试增量计算",
                                    progress=35,
                                )

                                # Need to load embeddings for incremental update
                                managed.add_log("正在从 ChromaDB 获取向量数据...", progress=40)
                                managed.update_progress(
                                    LoadingStatus.LOADING_EMBEDDINGS,
                                    progress=40,
                                    message="正在从 ChromaDB 获取向量数据...",
                                )

                                embeddings = await asyncio.to_thread(
                                    client.get_embeddings, collection_name
                                )
                                managed.add_log(f"已获取 {len(embeddings)} 个向量", progress=50)

                                managed.add_log("正在执行增量 UMAP 计算...", progress=60)
                                managed.update_progress(
                                    LoadingStatus.COMPUTING_PROJECTION,
                                    progress=60,
                                    message=f"正在执行增量 UMAP 计算 ({added_count} 个新向量)...",
                                )

                                processor = IncrementalUMAPProcessor(self.incremental_config)
                                incremental_result = await asyncio.to_thread(
                                    processor.compute_or_update,
                                    embeddings,
                                    data.ids,
                                    incremental_cache_file,
                                    self.umap_args,
                                )
                                proj = incremental_result.to_projection()
                                managed.add_log("增量 UMAP 计算完成", progress=75)
                        else:
                            managed.add_log("缓存 ID 不匹配，需要重新计算", progress=35)

                    except Exception as e:
                        logger.warning(f"Failed to use incremental cache: {e}")
                        managed.add_log(f"增量缓存加载失败: {e}", progress=35)

            # Fallback to legacy cache if incremental didn't work
            if proj is None and Projection.exists(legacy_cache_file):
                managed.add_log("检查传统缓存...", progress=30)
                managed.update_progress(
                    LoadingStatus.LOADING_METADATA,
                    progress=30,
                    message="发现缓存的投影数据，正在加载...",
                )
                proj = await asyncio.to_thread(Projection.load, legacy_cache_file)

                if proj.projection.shape[0] != row_count:
                    logger.warning(
                        f"Cache row count mismatch ({proj.projection.shape[0]} vs {row_count}), recomputing..."
                    )
                    managed.add_log("缓存数据不匹配，需要重新计算", progress=30)
                    proj = None
                else:
                    managed.add_log("缓存投影加载成功", progress=35)

            # Step 3: Load embeddings and compute projection if needed
            if proj is None:
                managed.add_log("正在从 ChromaDB 获取向量数据...", progress=40)
                managed.update_progress(
                    LoadingStatus.LOADING_EMBEDDINGS,
                    progress=40,
                    message="正在从 ChromaDB 获取向量数据...",
                )

                embeddings = await asyncio.to_thread(
                    client.get_embeddings, collection_name
                )
                managed.add_log(f"已获取 {len(embeddings)} 个向量", progress=50)

                managed.add_log(f"正在计算 UMAP 投影 ({len(embeddings)} 个向量)...", progress=60)
                managed.update_progress(
                    LoadingStatus.COMPUTING_PROJECTION,
                    progress=60,
                    message=f"正在计算 UMAP 投影 ({len(embeddings)} 个向量)...",
                )

                if self.enable_incremental:
                    # Use incremental processor for full computation (saves model for future)
                    processor = IncrementalUMAPProcessor(self.incremental_config)
                    incremental_result = await asyncio.to_thread(
                        processor.compute_or_update,
                        embeddings,
                        data.ids,
                        incremental_cache_file,
                        self.umap_args,
                    )
                    proj = incremental_result.to_projection()
                    managed.add_log("UMAP 投影计算完成（已保存增量缓存）", progress=75)
                else:
                    proj = await asyncio.to_thread(_run_umap, embeddings, self.umap_args)
                    managed.add_log("UMAP 投影计算完成", progress=75)

                    # Save to legacy cache
                    await asyncio.to_thread(Projection.save, legacy_cache_file, proj)
                    managed.add_log("投影已保存到缓存", progress=78)

            managed.add_log("正在构建数据框...", progress=80)
            managed.update_progress(
                LoadingStatus.COMPUTING_PROJECTION,
                progress=80,
                message="正在构建数据框...",
            )

            # Step 4: Build DataFrame
            df = chroma_data_to_dataframe(data)

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
