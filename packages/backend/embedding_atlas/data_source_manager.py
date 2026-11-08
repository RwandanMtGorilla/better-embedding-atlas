# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Data source manager for multi-collection mode with LRU caching."""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
from uuid import uuid4

import duckdb
import numpy as np

from .chroma_client import ChromaCollectionData, ChromaDBClient, chroma_data_to_dataframe
from .data_source import DataSource
from .incremental_umap import (
    IncrementalProjectionCache,
    IncrementalUMAPConfig,
    IncrementalUMAPProcessor,
)
from .options import make_embedding_atlas_props
from .projection import Projection, _run_umap
from .question_clustering import (
    QuestionClusteringConfig,
    add_question_cluster_column,
    cached_cluster_questions,
    update_collection_clusters,
)
from .utils import Hasher, cache_path
from .version import __version__

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = 15.0


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
class CollectionRevision:
    """A published snapshot. In-flight requests keep its data and connection alive."""

    revision: str
    row_count: int
    fit_id: str | None = None
    data_source: DataSource | None = None
    query_connection: duckdb.DuckDBPyConnection | None = None
    incremental_cache: IncrementalProjectionCache | None = None


@dataclass
class ManagedDataSource:
    """Loading and refresh state for one collection."""

    collection_name: str
    published: CollectionRevision | None = None
    loading_progress: LoadingProgress = field(
        default_factory=lambda: LoadingProgress(LoadingStatus.PENDING)
    )
    loading_task: asyncio.Task | None = None
    last_check: float = -float("inf")
    refreshing: bool = False
    refresh_error: str | None = None
    subscribers: list[asyncio.Queue[None]] = field(default_factory=list)
    logs: list[dict] = field(default_factory=list)

    def progress_event(self) -> dict:
        published = self.published
        return self.loading_progress.to_dict() | {
            "revision": published.revision if published else None,
            "row_count": published.row_count if published else None,
            "fit_id": published.fit_id if published else None,
            "refreshing": self.refreshing,
            "refresh_error": self.refresh_error,
        }

    def notify(self):
        # Queue notifications, then read the latest state when the stream sends it.
        for queue in self.subscribers:
            if not queue.full():
                queue.put_nowait(None)

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
        self.notify()

    def report(self, status: LoadingStatus, progress: float, message: str):
        if self.published is None:
            self.loading_progress = LoadingProgress(status, progress, message)
        self.add_log(message, progress)


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
        enable_incremental: bool = True,
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
        self._client_lock = threading.Lock()

    def _get_chroma_client(self) -> ChromaDBClient:
        """Get or create ChromaDB client."""
        with self._client_lock:
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

    def subscribe_progress(self, collection_name: str) -> asyncio.Queue[None]:
        """Subscribe to progress updates for a collection."""
        managed = self.get_managed(collection_name)
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        managed.subscribers.append(queue)
        queue.put_nowait(None)
        return queue

    def unsubscribe_progress(self, collection_name: str, queue: asyncio.Queue[None]):
        """Unsubscribe from progress updates."""
        with self._lock:
            if collection_name in self._sources:
                managed = self._sources[collection_name]
                if queue in managed.subscribers:
                    managed.subscribers.remove(queue)
        self._evict_if_needed()

    async def get_or_load(self, collection_name: str) -> ManagedDataSource:
        """Get a data source, loading it if necessary."""
        managed = self.get_managed(collection_name)

        if managed.published is not None:
            return managed

        if managed.loading_task is None or managed.loading_task.done():
            managed.loading_task = asyncio.create_task(self._load_data_source(managed))
            managed.loading_task.add_done_callback(self._evict_if_needed)
        await asyncio.shield(managed.loading_task)
        return managed

    def check_for_refresh(self, managed: ManagedDataSource):
        now = time.monotonic()
        if (
            managed.published is None
            or now - managed.last_check < REFRESH_INTERVAL
            or managed.loading_task is not None and not managed.loading_task.done()
        ):
            return
        managed.last_check = now
        managed.loading_task = asyncio.create_task(self._refresh_if_changed(managed))
        managed.loading_task.add_done_callback(self._evict_if_needed)

    async def _refresh_if_changed(self, managed: ManagedDataSource):
        try:
            client = await asyncio.to_thread(self._get_chroma_client)
            count = await asyncio.to_thread(client.get_collection_count, managed.collection_name)
        except Exception as error:
            logger.warning("Could not check collection %s: %s", managed.collection_name, error)
            managed.refresh_error = str(error)
            managed.notify()
            return

        if count != managed.published.row_count:
            managed.refreshing = True
            managed.refresh_error = None
            managed.notify()
            await self._load_data_source(managed)
        elif managed.refresh_error is not None:
            managed.refresh_error = None
            managed.notify()

    def _evict_if_needed(self, _task=None):
        """Evict least recently used data sources if over limit."""
        with self._lock:
            while len(self._sources) > self.max_cached:
                # Find oldest entry that is ready and has no subscribers
                for name in list(self._sources.keys()):
                    managed = self._sources[name]
                    if (
                        managed.loading_progress.status == LoadingStatus.READY
                        and len(managed.subscribers) == 0
                        and (managed.loading_task is None or managed.loading_task.done())
                    ):
                        logger.info(f"Evicting data source: {name}")
                        del self._sources[name]
                        break
                else:
                    # No evictable entries found
                    break

    async def _load_data_source(self, managed: ManagedDataSource):
        """Prepare a snapshot, then publish its data, projection and query table together."""
        name = managed.collection_name
        previous = managed.published
        managed.logs.clear()
        try:
            managed.report(LoadingStatus.LOADING_METADATA, 5, "正在从 ChromaDB 获取文档、元数据和向量...")
            client = await asyncio.to_thread(self._get_chroma_client)
            data = await asyncio.to_thread(client.get_snapshot, name, allow_empty=True)
            managed.report(LoadingStatus.LOADING_METADATA, 20, f"已获取 {len(data.ids)} 条文档")

            cache_file = self._incremental_cache_file(name) if self.enable_incremental else None
            cached = None
            if data.ids:
                if self.enable_incremental:
                    if previous is not None:
                        cached = previous.incremental_cache
                    elif cache_file is not None:
                        cached = await asyncio.to_thread(IncrementalProjectionCache.load, cache_file)
                prepared = await self._prepare_revision(managed, data, cached)
            else:
                prepared = CollectionRevision(uuid4().hex, 0)

            if cache_file is not None and (
                prepared.incremental_cache is not cached or not data.ids
            ):
                await asyncio.to_thread(IncrementalProjectionCache.save, cache_file, prepared.incremental_cache)

            managed.published = prepared
            managed.refresh_error = None
            message = "暂无条目" if prepared.row_count == 0 else ("更新完成" if previous else "加载完成")
            managed.loading_progress = LoadingProgress(LoadingStatus.READY, 100, message)
            managed.add_log(message, progress=100)
        except Exception as error:
            logger.exception("Failed to prepare collection %s", name)
            if previous is None:
                managed.loading_progress = LoadingProgress(LoadingStatus.ERROR, error=str(error), message="加载失败")
                managed.add_log(f"加载失败 {error}", error=True)
            else:
                managed.refresh_error = str(error)
                managed.add_log(f"更新失败 {error}", error=True)
        finally:
            managed.refreshing = False
            managed.notify()

    def _incremental_cache_file(self, name: str):
        hasher = Hasher()
        hasher.update([self.chroma_host, self.chroma_port, name, self.umap_args])
        try:
            return cache_path("projections") / f"collection_{hasher.hexdigest()}"
        except OSError as error:
            logger.warning("Incremental cache unavailable for %s: %s", name, error)
            return None

    async def _prepare_revision(
        self,
        managed: ManagedDataSource,
        data: ChromaCollectionData,
        cached: IncrementalProjectionCache | None,
    ) -> CollectionRevision:
        config = QuestionClusteringConfig.from_env()
        incremental_cache = None
        if self.enable_incremental:
            managed.report(LoadingStatus.COMPUTING_PROJECTION, 30, "正在计算 UMAP 投影...")
            content_ids = await asyncio.to_thread(_embedding_content_ids, data.ids, data.embeddings)
            processor = IncrementalUMAPProcessor(self.incremental_config)
            result = await asyncio.to_thread(
                processor.update, data.embeddings, content_ids, cached, self.umap_args
            )
            previous_clusters = cached.clusters if cached is not None and cached.fit_id == result.fit_id else None
            managed.report(LoadingStatus.COMPUTING_PROJECTION, 60, "正在计算代表项...")
            clusters = await asyncio.to_thread(
                update_collection_clusters, content_ids, data.documents, data.embeddings, previous_clusters, config
            )
            incremental_cache = result if clusters is result.clusters else replace(result, clusters=clusters)
            labels = [clusters.rows[row_id][1] for row_id in content_ids]
            representative_count = len(clusters.representative_ids)
            projection, fit_id = result.to_projection(), result.fit_id
        else:
            managed.report(LoadingStatus.COMPUTING_PROJECTION, 25, "正在计算代表项...")
            result = await asyncio.to_thread(cached_cluster_questions, data.documents, data.embeddings, config)
            labels = result.labels
            representative_count = len(result.representatives)
            managed.report(LoadingStatus.COMPUTING_PROJECTION, 60, "正在计算 UMAP 投影...")
            projection, fit_id = await asyncio.to_thread(self._full_projection, managed.collection_name, data)

        distinct_labels = {text: label for text, label in zip(data.documents, labels) if isinstance(text, str)}
        unique_coverage = sum(label is not None for label in distinct_labels.values()) / max(1, len(distinct_labels))
        row_coverage = sum(label is not None for label in labels) / len(labels)
        managed.add_log(
            f"已计算 {representative_count} 个代表项，不同文本覆盖率 {unique_coverage:.2%}，"
            f"条目覆盖率 {row_coverage:.2%}",
            progress=78,
        )

        managed.report(LoadingStatus.COMPUTING_PROJECTION, 80, "正在构建数据源...")
        return await asyncio.to_thread(
            self._build_revision, managed.collection_name, data, projection, labels, fit_id, incremental_cache
        )

    def _full_projection(self, name: str, data: ChromaCollectionData) -> tuple[Projection, str]:
        hasher = Hasher()
        hasher.update({
            "version": 2,
            "source": "chromadb",
            "collection": name,
            "ids": data.ids,
            "embeddings": data.embeddings,
            "umap_args": self.umap_args,
        })
        fit_id = hasher.hexdigest()
        path = cache_path("projections") / fit_id
        if Projection.exists(path):
            try:
                return Projection.load(path), fit_id
            except (OSError, ValueError) as error:
                logger.warning("Could not load projection cache %s: %s", path, error)
        projection = _run_umap(data.embeddings, self.umap_args)
        Projection.save(path, projection)
        return projection, fit_id

    def _build_revision(
        self,
        name: str,
        data: ChromaCollectionData,
        projection: Projection,
        labels: list[str | None],
        fit_id: str,
        incremental_cache: IncrementalProjectionCache | None,
    ) -> CollectionRevision:
        df = chroma_data_to_dataframe(data)
        representative_column = add_question_cluster_column(df, labels)
        x_column = find_column_name(df.columns, "projection_x")
        y_column = find_column_name(df.columns, "projection_y")
        neighbors_column = find_column_name(df.columns, "__neighbors")
        df[x_column] = projection.projection[:, 0]
        df[y_column] = projection.projection[:, 1]
        df[neighbors_column] = [
            {"distances": distances.tolist(), "ids": indices.tolist()}
            for indices, distances in zip(projection.knn_indices, projection.knn_distances)
        ]
        id_column = find_column_name(df.columns, "__row_index__")
        df[id_column] = range(len(df))
        metadata = {
            "props": make_embedding_atlas_props(
                row_id=id_column, x=x_column, y=y_column, neighbors=neighbors_column, text="document"
            ),
            "representative_item_column": representative_column,
        }

        # Reuse viewer caches for the exact same snapshot, and isolate each changed revision.
        hasher = Hasher()
        hasher.update([
            __version__, self.chroma_host, self.chroma_port, name, fit_id,
            data.ids, data.documents, data.metadatas, metadata, labels, projection.projection,
        ])
        revision = hasher.hexdigest()
        source = DataSource(revision, df, metadata, chroma_config={
            "host": self.chroma_host,
            "port": self.chroma_port,
            "collection": name,
            "id_to_row_index": {row_id: i for i, row_id in enumerate(data.ids)},
        })

        connection = None
        if self.duckdb_uri == "server":
            connection = duckdb.connect(":memory:")
            try:
                connection.sql("CREATE TABLE dataset AS (SELECT * FROM df)")
            except Exception:
                connection.close()
                raise
        return CollectionRevision(revision, len(df), fit_id, source, connection, incremental_cache)


def _embedding_content_ids(ids: list[str], embeddings: np.ndarray) -> list[str]:
    """An edit to an existing vector is a removal and addition for UMAP."""
    content_ids = []
    for row_id, vector in zip(ids, embeddings, strict=True):
        hasher = Hasher()
        hasher.update([row_id, vector])
        content_ids.append(hasher.hexdigest())
    return content_ids
