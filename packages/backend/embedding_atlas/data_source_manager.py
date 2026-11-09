# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Data source manager for multi-collection mode with LRU caching."""

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from heapq import merge
from operator import itemgetter

import duckdb
import numpy as np

from .chroma_client import ChromaCollectionData, ChromaDBClient, ChromaDeltaError, chroma_data_to_dataframe
from .collection_snapshot import CollectionSnapshotStore, SnapshotArtifacts, SnapshotCacheError, SnapshotIndex
from .data_source import DataSource
from .incremental_umap import (
    IncrementalProjectionCache,
    IncrementalUMAPConfig,
    IncrementalUMAPProcessor,
)
from .options import make_embedding_atlas_props
from .projection import _run_umap
from .question_clustering import (
    CLUSTER_CACHE_VERSION,
    CollectionQuestionClusters,
    QuestionClusteringConfig,
    add_question_cluster_column,
    update_collection_clusters,
)
from .utils import Hasher, cache_path
from .version import __version__

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = 15.0
ID_CHECK_INTERVAL = 60.0


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
    index: SnapshotIndex | None = None
    artifacts: SnapshotArtifacts | None = None
    checkpoint_saved: bool = False


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
    last_id_check: float = -float("inf")
    force_pending: bool = False
    force_running: bool = False
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

        # Shared ChromaDB client for collection listing and snapshot reads.
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

        if not self._is_loading(managed):
            self._start_loading(managed)
        await asyncio.shield(managed.loading_task)
        return managed

    @staticmethod
    def _is_loading(managed: ManagedDataSource) -> bool:
        return managed.loading_task is not None and not managed.loading_task.done()

    def _start_loading(self, managed: ManagedDataSource, *, poll: bool = False):
        managed.loading_task = asyncio.create_task(self._load_data_source(managed, poll=poll))
        managed.loading_task.add_done_callback(self._evict_if_needed)

    def check_for_refresh(self, managed: ManagedDataSource):
        if (
            managed.published is None
            or time.monotonic() - managed.last_check < REFRESH_INTERVAL
            or self._is_loading(managed)
        ):
            return
        self._start_loading(managed, poll=True)

    def request_full_refresh(self, collection_name: str):
        managed = self.get_managed(collection_name)
        if not managed.force_running:
            managed.force_pending = True
            if not self._is_loading(managed):
                self._start_loading(managed)

    def _evict_if_needed(self, _task=None):
        """Evict least recently used data sources if over limit."""
        with self._lock:
            while len(self._sources) > self.max_cached:
                # Find the oldest finished entry without subscribers.
                for name in list(self._sources.keys()):
                    managed = self._sources[name]
                    if (
                        managed.loading_progress.status in (LoadingStatus.READY, LoadingStatus.ERROR)
                        and len(managed.subscribers) == 0
                        and (managed.loading_task is None or managed.loading_task.done())
                    ):
                        logger.info(f"Evicting data source: {name}")
                        del self._sources[name]
                        break
                else:
                    # No evictable entries found
                    break

    async def _load_data_source(self, managed: ManagedDataSource, *, poll: bool = False):
        """Serialize automatic refreshes and coalesce manual full sync requests."""
        while True:
            force = managed.force_pending
            managed.force_pending = False
            managed.force_running = force
            managed.last_check = time.monotonic()
            try:
                await self._sync_once(managed, poll=poll, force=force)
            except Exception as error:
                if managed.published is None or managed.refreshing:
                    logger.exception("Failed to prepare collection %s", managed.collection_name)
                else:
                    logger.warning("Could not check collection %s: %s", managed.collection_name, error)
                if managed.published is None:
                    managed.loading_progress = LoadingProgress(LoadingStatus.ERROR, error=str(error), message="加载失败")
                    managed.add_log(f"加载失败 {error}", error=True)
                else:
                    managed.refresh_error = str(error)
                    if managed.refreshing:
                        managed.add_log(f"更新失败 {error}", error=True)
                    else:
                        managed.notify()
                if force:
                    managed.force_pending = True
                    return  # Retry on the next check, even if the row count is unchanged.
            finally:
                managed.force_running = False
                if managed.refreshing:
                    managed.refreshing = False
                    managed.notify()
            if not managed.force_pending:
                return
            poll = False

    def _snapshot_store(self, name: str) -> CollectionSnapshotStore | None:
        hasher = Hasher()
        hasher.update([self.chroma_host, self.chroma_port, name])
        try:
            return CollectionSnapshotStore(cache_path("projections") / f"collection_{hasher.hexdigest()}.sqlite3")
        except OSError as error:
            logger.warning("Collection cache unavailable for %s: %s", name, error)
            return None

    async def _sync_once(self, managed: ManagedDataSource, *, poll: bool, force: bool):
        name, previous = managed.collection_name, managed.published
        started = time.perf_counter()
        if previous is None or force:
            self._begin_sync(managed)

        client = await asyncio.to_thread(self._get_chroma_client)
        reader = await asyncio.to_thread(client.open_collection, name)
        base = previous.index if previous is not None else None
        if poll and not force and base is not None and base.collection_id == reader.uuid:
            count = await asyncio.to_thread(reader.count)
            if count == len(base.ids) and time.monotonic() - managed.last_id_check < ID_CHECK_INTERVAL:
                self._clear_refresh_error(managed)
                return

        store = await asyncio.to_thread(self._snapshot_store, name)
        if previous is None and store is not None:
            base = await self._read_local(store, store.read_index)
        if base is not None and base.collection_id != reader.uuid:
            base = None

        additions = None
        remote_ids = None
        if base is not None and not force:
            remote_ids = await asyncio.to_thread(reader.get_ids)
            if previous is not None and remote_ids == base.ids:
                managed.last_id_check = time.monotonic()
                self._clear_refresh_error(managed)
                return

        if not managed.refreshing and previous is not None:
            self._begin_sync(managed)

        config = QuestionClusteringConfig.from_env()
        projection_key, cluster_key = self._artifact_keys(config)
        prior = None
        cached = None
        if base is not None:
            if previous is not None:
                prior, cached = previous.artifacts, previous.incremental_cache
            elif store is not None and base.ids:
                prior = await self._read_local(store, store.read_artifacts, base)

        data = None
        mode = "full"
        if base is not None and store is not None and remote_ids is not None:
            if remote_ids == base.ids:
                mode = "local"
                managed.report(LoadingStatus.LOADING_METADATA, 20, f"正在恢复 {len(base.ids)} 条本地记录...")
                use_artifacts = _artifacts_match(prior, projection_key, cluster_key)
                data = await self._read_local(store, store.read_rows, base, include_vectors=not use_artifacts)
            elif (previous is None or previous.checkpoint_saved) and set(base.ids).issubset(remote_ids):
                new_ids = sorted(set(remote_ids).difference(base.ids))
                managed.report(LoadingStatus.LOADING_EMBEDDINGS, 20, f"正在读取 {len(new_ids)} 条新增记录...")
                try:
                    additions = await asyncio.to_thread(self._read_append, client, name, reader, base, remote_ids, new_ids)
                except ChromaDeltaError as error:
                    logger.info("[%s] Append no longer matches the ID check: %s", name, error)
                    reader = await asyncio.to_thread(client.open_collection, name)
                    if reader.uuid != base.collection_id:
                        base, prior, cached = None, None, None
                else:
                    mode = "append"
                    data = await self._read_local(
                        store, store.read_rows, base, include_vectors=True, additions=additions
                    )

        if data is None:
            additions = None
            mode = "full"
            message = "正在全量核对 ChromaDB 记录..." if previous is not None or force else "正在从 ChromaDB 首次加载记录..."
            managed.report(LoadingStatus.LOADING_EMBEDDINGS, 20, message)
            data = await asyncio.to_thread(reader.get_snapshot, allow_empty=True)

        received = time.perf_counter()
        index = base if mode == "local" else await asyncio.to_thread(
            _snapshot_index, reader.uuid, data, base=base, additions=additions
        )
        same_rows = base is not None and index.fingerprints == base.fingerprints
        reuse_artifacts = _artifacts_match(prior, projection_key, cluster_key)
        if previous is not None and same_rows and (not index.ids or reuse_artifacts):
            managed.last_id_check = time.monotonic()
            self._clear_refresh_error(managed)
            managed.add_log("数据未变化", progress=100)
            return

        incremental = None
        artifacts = None
        restored = mode == "local" and (reuse_artifacts or not index.ids)
        if index.ids:
            if restored:
                artifacts = prior
                incremental = await asyncio.to_thread(self._restore_incremental, index, prior, config, data.documents)
            else:
                old_documents = None
                if prior is not None and base is not None:
                    if mode == "local":
                        old_documents = data.documents
                    elif mode == "append" and previous is None:
                        old_documents = await asyncio.to_thread(_select_documents, data, base.ids)
                    elif mode == "full" and index.content_ids == base.content_ids:
                        if previous is not None:
                            old_documents = await asyncio.to_thread(previous.data_source.dataset["document"].tolist)
                        elif store is not None:
                            old_data = await self._read_local(store, store.read_rows, base)
                            old_documents = old_data.documents if old_data is not None else None
                managed.report(LoadingStatus.COMPUTING_PROJECTION, 30, "正在准备 UMAP 投影和代表项...")
                artifacts, incremental = await asyncio.to_thread(
                    self._compute_artifacts, data, index, config, projection_key, cluster_key,
                    base, prior, cached, old_documents, store,
                )
                if prior is None or artifacts.labels is not prior.labels:
                    unique_coverage, row_coverage = await asyncio.to_thread(_label_coverage, data.documents, artifacts.labels)
                    managed.add_log(
                        f"已计算 {len(artifacts.representative_ids)} 个代表项，不同文本覆盖率 {unique_coverage:.2%}，"
                        f"条目覆盖率 {row_coverage:.2%}", progress=78,
                    )

        if not restored:
            fit_id = artifacts.fit_id if artifacts is not None else None
            revision = await asyncio.to_thread(_hash, [
                self.chroma_host, self.chroma_port, name, reader.uuid,
                projection_key, cluster_key, fit_id, index.fingerprints,
            ])
            index = replace(index, revision=revision)
        managed.report(LoadingStatus.COMPUTING_PROJECTION, 80, "正在构建数据源...")
        prepared = await asyncio.to_thread(self._build_revision, name, data, index, artifacts, incremental)
        built = time.perf_counter()

        if restored:
            prepared.checkpoint_saved = True
        elif store is not None:
            # An append commits just the new records. Rebuilding derived artifacts
            # from the local checkpoint leaves all its original records alone.
            records = additions if mode == "append" else data
            if mode == "local":
                records = ChromaCollectionData([], [], [], np.empty((0, index.dimension), dtype=np.float32))
            try:
                await asyncio.to_thread(
                    store.save, index, records, artifacts, incremental.umap_model if incremental is not None else None,
                    base=base if mode != "full" else None,
                )
                prepared.checkpoint_saved = True
            except Exception as error:
                logger.warning("Could not save collection snapshot %s: %s", name, error)

        managed.published = prepared
        managed.last_id_check = time.monotonic()
        managed.refresh_error = None
        message = "暂无条目" if not prepared.row_count else ("更新完成" if previous else "加载完成")
        managed.loading_progress = LoadingProgress(LoadingStatus.READY, 100, message)
        managed.add_log(message, progress=100)
        logger.info(
            "[%s] Sync %s: %d rows; read %.3fs, prepare %.3fs, checkpoint %.3fs",
            name, mode, prepared.row_count, received - started, built - received, time.perf_counter() - built,
        )

    @staticmethod
    def _begin_sync(managed: ManagedDataSource):
        managed.logs.clear()
        managed.refreshing = managed.published is not None
        managed.refresh_error = None
        managed.report(LoadingStatus.LOADING_METADATA, 5, "正在检查 ChromaDB 集合...")

    @staticmethod
    def _clear_refresh_error(managed: ManagedDataSource):
        if managed.refresh_error is not None:
            managed.refresh_error = None
            managed.notify()

    @staticmethod
    async def _read_local(store, method, *args, **kwargs):
        try:
            return await asyncio.to_thread(method, *args, **kwargs)
        except (OSError, SnapshotCacheError) as error:
            logger.warning("Could not use collection cache %s: %s", store.path, error)
            return None

    @staticmethod
    def _read_append(client, name, reader, base, expected_ids, new_ids):
        data = reader.get_rows(new_ids)
        if base.ids and data.embeddings.shape[1] != base.dimension:
            raise ChromaDeltaError("New embeddings do not match the cached dimensions")
        latest = client.open_collection(name)
        if latest.uuid != reader.uuid or not set(expected_ids).issubset(latest.get_ids()):
            raise ChromaDeltaError("An expected ID disappeared during the read")
        return data

    def _artifact_keys(self, config):
        return (
            _hash([__version__, 1, self.enable_incremental, self.incremental_config.save_model,
                   dict(sorted(self.umap_args.items()))]),
            _hash([CLUSTER_CACHE_VERSION, asdict(config)]),
        )

    def _restore_incremental(self, index, artifacts, config, documents, cached=None):
        if not self.enable_incremental:
            return None
        clusters = None
        if artifacts.cluster_key == self._artifact_keys(config)[1]:
            if cached is not None and cached.clusters is not None:
                clusters = cached.clusters
            elif documents is not None:
                clusters = CollectionQuestionClusters(
                    CLUSTER_CACHE_VERSION, config,
                    dict(zip(index.content_ids, zip(documents, artifacts.labels, strict=True), strict=True)),
                    artifacts.representative_ids,
                )
        if cached is not None:
            return cached if clusters is cached.clusters else replace(cached, clusters=clusters)
        projection = artifacts.projection
        return IncrementalProjectionCache(
            projection.projection, projection.knn_indices, projection.knn_distances,
            index.content_ids, artifacts.fit_count, artifacts.fit_id, clusters=clusters,
        )

    def _compute_artifacts(
        self, data, index, config, projection_key, cluster_key, base, prior, cached, old_documents, store
    ):
        same_vectors = (
            prior is not None and prior.projection_key == projection_key and base.content_ids == index.content_ids
        )
        if prior is None or prior.projection_key != projection_key:
            cached = None
        elif self.enable_incremental:
            cached = self._restore_incremental(base, prior, config, old_documents, cached)

        result = None
        if same_vectors:
            projection, fit_id, fit_count = prior.projection, prior.fit_id, prior.fit_count
            result = cached
        elif self.enable_incremental:
            def load_model():
                return store.read_model(base, cached.fit_id) if store is not None and cached is not None else None

            result = IncrementalUMAPProcessor(self.incremental_config).update(
                data.embeddings, index.content_ids, cached, self.umap_args, load_model=load_model
            )
            projection, fit_id, fit_count = result.to_projection(), result.fit_id, result.fit_count
        else:
            projection = None
            fit_count = len(index.ids)
            fit_id = _hash([projection_key, index.collection_id, index.content_ids])

        if same_vectors and prior.cluster_key == cluster_key and old_documents == data.documents:
            labels, representative_ids = prior.labels, prior.representative_ids
        else:
            previous_clusters = cached.clusters if cached is not None and cached.fit_id == fit_id else None
            clusters = update_collection_clusters(
                index.content_ids, data.documents, data.embeddings, previous_clusters, config
            )
            labels = [clusters.rows[row_id][1] for row_id in index.content_ids]
            representative_ids = clusters.representative_ids
            if result is not None and clusters is not result.clusters:
                result = replace(result, clusters=clusters)
        if projection is None:
            projection = _run_umap(data.embeddings, self.umap_args)
        return SnapshotArtifacts(projection_key, cluster_key, fit_id, fit_count, projection, labels, representative_ids), result

    def _build_revision(
        self,
        name: str,
        data: ChromaCollectionData,
        index: SnapshotIndex,
        artifacts: SnapshotArtifacts | None,
        incremental_cache: IncrementalProjectionCache | None,
    ) -> CollectionRevision:
        if artifacts is None:
            return CollectionRevision(index.revision, 0, index=index)
        projection = artifacts.projection
        df = chroma_data_to_dataframe(data)
        representative_column = add_question_cluster_column(df, artifacts.labels)
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

        source = DataSource(index.revision, df, metadata, chroma_config={
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
        return CollectionRevision(
            index.revision, len(df), artifacts.fit_id, source, connection, incremental_cache, index, artifacts
        )


def _hash(value) -> str:
    hasher = Hasher()
    hasher.update(value)
    return hasher.hexdigest()


def _artifacts_match(artifacts, projection_key, cluster_key):
    return artifacts is not None and artifacts.projection_key == projection_key and artifacts.cluster_key == cluster_key


def _select_documents(data: ChromaCollectionData, ids: list[str]) -> list[str | None]:
    by_id = dict(zip(data.ids, data.documents, strict=True))
    return [by_id[row_id] for row_id in ids]


def _label_coverage(documents, labels):
    distinct = {text: label for text, label in zip(documents, labels, strict=True) if isinstance(text, str)}
    return (
        sum(label is not None for label in distinct.values()) / max(1, len(distinct)),
        sum(label is not None for label in labels) / len(labels),
    )


def _snapshot_index(collection_id, data, *, base=None, additions=None):
    incoming = data if additions is None else additions
    content_ids = _embedding_content_ids(incoming.ids, incoming.embeddings)
    fingerprints = [
        hashlib.sha256(json.dumps([content_id, document, metadata], sort_keys=True).encode()).hexdigest()
        for content_id, document, metadata in zip(content_ids, incoming.documents, incoming.metadatas, strict=True)
    ]
    if additions is not None:
        original = zip(base.ids, base.content_ids, base.fingerprints, strict=True)
        added = zip(incoming.ids, content_ids, fingerprints, strict=True)
        content_ids, fingerprints = [], []
        for _, content_id, fingerprint in merge(original, added, key=itemgetter(0)):
            content_ids.append(content_id)
            fingerprints.append(fingerprint)
    return SnapshotIndex(collection_id, "", data.ids, content_ids, fingerprints, data.embeddings.shape[1])


def _embedding_content_ids(ids: list[str], embeddings: np.ndarray) -> list[str]:
    """An edit to an existing vector is a removal and addition for UMAP."""
    content_ids = []
    for row_id, vector in zip(ids, embeddings, strict=True):
        hasher = Hasher()
        hasher.update([row_id, vector])
        content_ids.append(hasher.hexdigest())
    return content_ids
