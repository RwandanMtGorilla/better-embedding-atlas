import asyncio
import json
import sqlite3
import threading
import unittest
import zipfile
from contextlib import aclosing, closing
from io import BytesIO
from unittest.mock import Mock, patch

import httpx
import joblib
import numpy as np
import pandas as pd

from embedding_atlas.chroma_client import ChromaCollectionData, ChromaCollectionReader, ChromaDBClient
from embedding_atlas.collection_snapshot import CollectionSnapshotStore
from embedding_atlas.data_source_manager import DataSourceManager, _embedding_content_ids
from embedding_atlas.incremental_umap import IncrementalProjectionCache, IncrementalUMAPConfig, IncrementalUMAPProcessor
from embedding_atlas.projection import Projection
from embedding_atlas.question_clustering import cluster_questions
from embedding_atlas.server import make_multi_server

from test_question_clustering import cache_directory


class StubMapper:
    def __init__(self):
        self.transformed = []
        self.fail = False

    def transform(self, embeddings):
        self.transformed.append(len(embeddings))
        if self.fail:
            raise ValueError("transform failed")
        return embeddings + 1000


def neighbors(embeddings, args):
    count = len(embeddings)
    return ((np.arange(count) + 1) % count)[:, None], np.zeros((count, 1)), None


def fit_umap(embeddings, args):
    indices, distances, _ = neighbors(embeddings, args)
    coordinates = np.column_stack((np.arange(len(embeddings)), np.zeros(len(embeddings))))
    return Projection(coordinates, indices, distances), StubMapper()


def collection(count):
    return ChromaCollectionData(
        [f"{i * 2:04d}" for i in range(count)],
        ["left"] * count,
        [None] * count,
        np.tile(np.array([1, 0], dtype=np.float32), (count, 1)),
    )


class FakeCollection:
    def __init__(self, data, uuid="items-v1"):
        self.data = data
        self.id = uuid
        self.get = Mock(side_effect=self.fetch)
        self.count = Mock(side_effect=lambda: len(self.data.ids))

    def fetch(self, *, include, ids=None):
        data = self.data
        if ids is None:
            rows = range(len(data.ids))
        else:
            wanted = set(ids)
            rows = [i for i in range(len(data.ids) - 1, -1, -1) if data.ids[i] in wanted]
        return {
            "ids": [data.ids[i] for i in rows],
            "documents": [data.documents[i] for i in rows] if "documents" in include else None,
            "metadatas": [data.metadatas[i] for i in rows] if "metadatas" in include else None,
            "embeddings": data.embeddings[rows] if "embeddings" in include else None,
        }

    def calls_for(self, kind):
        return [
            call for call in self.get.call_args_list
            if kind == ("ids" if not call.kwargs["include"] else "delta" if "ids" in call.kwargs else "full")
        ]


class CollectionRefreshTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.projections = self.enterContext(cache_directory())
        self.viewer_cache = self.enterContext(cache_directory())
        self.static = self.enterContext(cache_directory())
        for module, directory in (
            ("data_source_manager", self.projections),
            ("question_clustering", self.projections),
            ("data_source", self.viewer_cache),
        ):
            self.enterContext(patch(f"embedding_atlas.{module}.cache_path", return_value=directory))
        self.fit = self.enterContext(patch("embedding_atlas.incremental_umap._fit_umap", side_effect=fit_umap))
        self.full_neighbors = self.enterContext(patch("embedding_atlas.incremental_umap._umap_neighbors", side_effect=neighbors))
        self.cluster = self.enterContext(patch(
            "embedding_atlas.question_clustering.cluster_questions", wraps=cluster_questions
        ))
        self.source = FakeCollection(collection(100))
        self.client = ChromaDBClient.__new__(ChromaDBClient)
        self.client._client = Mock()
        self.client._client.get_collection.return_value = self.source

    def set_collection(self, data):
        self.source.data = data

    def manager(self, mode="wasm"):
        manager = DataSourceManager("test", 1234, duckdb_uri=mode)
        manager._chroma_client = self.client
        return manager

    async def refresh(self, manager, managed):
        managed.last_check = -float("inf")
        manager.check_for_refresh(managed)
        await managed.loading_task

    async def full_refresh(self, manager, managed):
        manager.request_full_refresh(managed.collection_name)
        await managed.loading_task

    async def test_cumulative_growth_and_representatives_survive_restart(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        initial = managed.published

        updated = collection(110)
        updated.ids[100:105] = ["0001", "0003", "0005", "0007", "0009"]
        updated.documents[100:105] = ["left", "uncovered", None, "similar", "similar"]
        updated.embeddings[100:105] = [[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]]
        order = np.argsort(updated.ids)
        self.set_collection(ChromaCollectionData(
            [updated.ids[i] for i in order], [updated.documents[i] for i in order],
            [None] * 110, updated.embeddings[order],
        ))
        with patch("embedding_atlas.collection_snapshot.joblib.dump", wraps=joblib.dump) as dump, patch(
            "embedding_atlas.data_source_manager._embedding_content_ids", wraps=_embedding_content_ids,
        ) as hash_rows:
            await self.refresh(manager, managed)
            dump.assert_not_called()
            self.assertEqual(len(hash_rows.call_args.args[0]), 10)
        latest = managed.published
        self.assertTrue(latest.checkpoint_saved)
        self.assertEqual(len(self.source.calls_for("full")), 1)
        self.assertEqual(set(self.source.calls_for("delta")[0].kwargs["ids"]), set(updated.ids) - set(initial.index.ids))
        self.full_neighbors.assert_not_called()
        frame = latest.data_source.dataset.set_index("id")
        old_frame = initial.data_source.dataset.set_index("id")
        pd.testing.assert_series_equal(frame.loc[old_frame.index, "projection_x"], old_frame.projection_x)
        pd.testing.assert_series_equal(frame.loc[old_frame.index, "representative_item"], old_frame.representative_item)
        self.assertEqual(frame.loc[updated.ids[100:105], "representative_item"].tolist(),
                         ["left", None, None, "left", "left"])
        self.assertEqual(latest.incremental_cache.umap_model.transformed, [10])
        self.assertEqual(latest.fit_id, initial.fit_id)
        self.assertEqual(self.cluster.call_count, 1)

        current = self.source.data
        extension = collection(121)
        manager = self.manager()
        with patch.object(CollectionSnapshotStore, "read_model", side_effect=AssertionError("model read on restart")), patch(
            "embedding_atlas.data_source_manager._embedding_content_ids", side_effect=AssertionError("vectors read on restart")
        ):
            managed = await manager.get_or_load("items")
        self.assertEqual(managed.published.revision, latest.revision)
        self.assertEqual(len(self.source.calls_for("full")), 1)
        self.assertEqual(len(self.source.calls_for("delta")), 1)
        self.assertIsNone(managed.published.incremental_cache.umap_model)
        for count in (120, 121):
            self.set_collection(ChromaCollectionData(
                current.ids + extension.ids[110:count],
                current.documents + extension.documents[110:count],
                [None] * count,
                np.concatenate([current.embeddings, extension.embeddings[110:count]]),
            ))
            await self.refresh(manager, managed)
            if count == 120:
                self.assertEqual(managed.published.fit_id, initial.fit_id)
                self.assertEqual(managed.published.incremental_cache.fit_count, 100)
                self.assertEqual(managed.published.incremental_cache.umap_model.transformed, [10])
                self.assertEqual(self.cluster.call_count, 1)
        self.assertEqual(managed.published.incremental_cache.fit_count, 121)
        self.assertNotEqual(managed.published.fit_id, initial.fit_id)
        self.assertEqual(self.fit.call_count, 2)
        self.assertEqual(self.cluster.call_count, 2)
        self.assertEqual(len(self.source.calls_for("full")), 1)

    async def test_reordered_chroma_response_does_not_change_revision(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        initial = managed.published
        data = collection(100)
        data.ids.reverse()
        self.set_collection(data)
        await self.full_refresh(manager, managed)
        self.assertIs(managed.published, initial)
        self.assertEqual(self.cluster.call_count, 1)
        frame = managed.published.data_source.dataset
        self.assertEqual(frame.projection_x.iloc[:2].tolist(), [0, 1])

    async def test_text_edit_reclusters_and_transform_failure_refits(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        first_fit = managed.published.fit_id
        updated = collection(101)
        updated.documents[0] = "renamed"
        self.set_collection(updated)
        await self.full_refresh(manager, managed)
        self.assertEqual(managed.published.fit_id, first_fit)
        self.assertEqual(self.cluster.call_count, 2)

        managed.published.incremental_cache.umap_model.fail = True
        updated = collection(102)
        updated.documents[0] = "renamed"
        self.set_collection(updated)
        await self.refresh(manager, managed)
        self.assertNotEqual(managed.published.fit_id, first_fit)
        self.assertEqual(self.cluster.call_count, 3)

    async def test_unchanged_count_reuses_revision_and_removal_refits(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        initial = managed.published
        await self.refresh(manager, managed)
        self.assertEqual(len(self.source.calls_for("full")), 1)
        self.assertEqual(len(self.source.calls_for("ids")), 0)
        self.assertIs(managed.published, initial)

        self.set_collection(collection(99))
        await self.refresh(manager, managed)
        self.assertNotEqual(managed.published.fit_id, initial.fit_id)
        self.assertEqual(managed.published.row_count, 99)
        self.assertEqual(self.fit.call_count, 2)
        self.assertEqual(self.cluster.call_count, 2)

        app = make_multi_server(manager, str(self.static), duckdb_uri="wasm")
        params = {"revision": managed.published.revision}
        prefix = "/collection/items/data/"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            metadata = await http.get(prefix + "metadata.json", params=params)
            self.assertEqual(metadata.json()["database"], {"type": "wasm", "load": True})
            dataset = await http.get(prefix + "dataset.parquet", params=params)
            frame = pd.read_parquet(BytesIO(dataset.content))
            self.assertEqual(frame.representative_item.tolist(), ["left"] * 99)

    async def test_failed_refresh_keeps_disk_and_published_revision(self):
        manager = self.manager("server")
        managed = await manager.get_or_load("items")
        initial = managed.published
        path = next(self.projections.glob("*.sqlite3"))
        saved = path.read_bytes()
        self.set_collection(collection(101))
        with patch.object(manager, "_build_revision", side_effect=ValueError("query table failed")):
            await self.refresh(manager, managed)
        self.assertIs(managed.published, initial)
        self.assertEqual(managed.refresh_error, "query table failed")
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual(initial.query_connection.sql("SELECT count(*) FROM dataset").fetchone(), (100,))

        await self.refresh(manager, managed)
        self.assertIsNone(managed.refresh_error)
        self.assertEqual(managed.published.row_count, 101)
        self.assertEqual(managed.published.fit_id, initial.fit_id)

    async def test_empty_collection_and_corrupt_cache_rebuild(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        fit_id = managed.published.fit_id
        self.set_collection(collection(0))
        await self.refresh(manager, managed)
        self.assertEqual(managed.published.row_count, 0)
        self.assertIsNone(managed.published.data_source)
        self.assertEqual(self.fit.call_count, 1)
        self.set_collection(collection(100))
        manager = self.manager()
        managed = await manager.get_or_load("items")
        self.assertNotEqual(managed.published.fit_id, fit_id)
        path = next(self.projections.glob("*.sqlite3"))
        path.write_bytes(b"broken cache")
        reloaded = await self.manager().get_or_load("items")
        self.assertNotEqual(reloaded.published.fit_id, managed.published.fit_id)
        restored = await self.manager().get_or_load("items")
        self.assertEqual(restored.published.fit_id, reloaded.published.fit_id)
        self.assertEqual(self.fit.call_count, 3)

    async def test_equal_count_replacement_and_recreated_collection(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        initial = managed.published
        changed = collection(100)
        changed.ids[0] = "replacement"
        changed.metadatas[1] = {"tag": "edited in place"}
        self.set_collection(changed)
        await self.refresh(manager, managed)
        self.assertIs(managed.published, initial)
        self.assertFalse(self.source.calls_for("ids"))

        managed.last_id_check = -float("inf")
        await self.refresh(manager, managed)
        replaced = managed.published
        self.assertEqual(replaced.row_count, 100)
        frame = replaced.data_source.dataset.set_index("id")
        self.assertNotIn("0000", frame.index)
        self.assertEqual(frame.loc["0002", "tag"], "edited in place")
        self.assertEqual(len(self.source.calls_for("full")), 2)
        self.assertFalse(self.source.calls_for("delta"))

        recreated = FakeCollection(changed, "items-v2")
        self.client._client.get_collection.return_value = recreated
        await self.refresh(manager, managed)
        self.assertEqual(managed.published.index.collection_id, "items-v2")
        self.assertNotEqual(managed.published.fit_id, replaced.fit_id)
        self.assertEqual(len(recreated.calls_for("full")), 1)

    async def test_disappearing_old_id_during_append_falls_back_to_full(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        added = collection(101)
        self.set_collection(added)

        def concurrent_delete(*, include, ids=None):
            response = self.source.fetch(include=include, ids=ids)
            if ids is not None:
                self.set_collection(ChromaCollectionData(
                    added.ids[1:], added.documents[1:], added.metadatas[1:], added.embeddings[1:]
                ))
            return response

        self.source.get.side_effect = concurrent_delete
        await self.refresh(manager, managed)
        self.assertEqual(managed.published.index.ids, added.ids[1:])
        self.assertEqual(len(self.source.calls_for("delta")), 1)
        self.assertEqual(len(self.source.calls_for("full")), 2)

    async def test_manual_sync_reuses_artifacts_independently_and_skips_no_op(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        initial = managed.published
        edited = collection(100)
        edited.metadatas[0] = {"tag": "manual"}
        self.set_collection(edited)
        await self.refresh(manager, managed)
        self.assertIs(managed.published, initial)
        await self.full_refresh(manager, managed)
        metadata_edit = managed.published
        self.assertEqual(metadata_edit.data_source.dataset.tag.iloc[0], "manual")
        self.assertIs(metadata_edit.artifacts.projection, initial.artifacts.projection)
        self.assertIs(metadata_edit.artifacts.labels, initial.artifacts.labels)
        self.assertEqual(self.fit.call_count, 1)
        self.assertEqual(self.cluster.call_count, 1)

        await self.full_refresh(manager, managed)
        self.assertIs(managed.published, metadata_edit)

        edited.documents[0] = "new text"
        await self.full_refresh(manager, managed)
        text_edit = managed.published
        self.assertIs(text_edit.artifacts.projection, initial.artifacts.projection)
        self.assertEqual(text_edit.data_source.dataset.document.iloc[0], "new text")
        self.assertEqual(self.fit.call_count, 1)
        self.assertEqual(self.cluster.call_count, 2)

    async def test_manual_endpoint_coalesces_requests_and_retries_a_failed_full_read(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        initial = managed.published
        updated = collection(100)
        updated.metadatas[0] = {"tag": "retried"}
        self.set_collection(updated)
        entered, release = threading.Event(), threading.Event()

        def fail_full_read(**_):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release the full read")
            raise ConnectionError("Chroma unavailable")

        self.source.get.side_effect = fail_full_read
        app = make_multi_server(manager, str(self.static), duckdb_uri="wasm")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            try:
                self.assertEqual((await http.post("/api/collection/items/refresh")).status_code, 202)
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                self.assertTrue(managed.refreshing)
                self.assertEqual((await http.post("/api/collection/items/refresh")).status_code, 202)
                self.assertIs(managed.published, initial)
            finally:
                release.set()
            await managed.loading_task

        self.assertEqual(len(self.source.calls_for("full")), 2)
        self.assertIs(managed.published, initial)
        self.assertEqual(managed.refresh_error, "Chroma unavailable")
        self.source.get.side_effect = self.source.fetch
        await self.refresh(manager, managed)
        self.assertIsNone(managed.refresh_error)
        self.assertEqual(managed.published.data_source.dataset.tag.iloc[0], "retried")
        self.assertEqual(len(self.source.calls_for("full")), 3)

    async def test_evicted_collection_reopens_from_disk(self):
        manager = self.manager()
        manager.max_cached = 1
        other = FakeCollection(collection(10), "other-v1")
        self.client._client.get_collection.side_effect = lambda name: self.source if name == "items" else other
        initial = (await manager.get_or_load("items")).published
        await manager.get_or_load("other")
        self.assertNotIn("items", manager._sources)
        restored = (await manager.get_or_load("items")).published
        self.assertEqual(restored.revision, initial.revision)
        self.assertEqual(len(self.source.calls_for("full")), 1)
        self.assertEqual(len(self.source.calls_for("ids")), 1)

    async def test_restart_with_additions_reuses_the_fit_and_fetches_only_new_rows(self):
        initial = (await self.manager().get_or_load("items")).published
        self.set_collection(collection(110))
        restored = (await self.manager().get_or_load("items")).published
        self.assertEqual(restored.row_count, 110)
        self.assertEqual(restored.fit_id, initial.fit_id)
        self.assertEqual(self.cluster.call_count, 1)
        self.assertEqual(len(self.source.calls_for("full")), 1)
        self.assertEqual(self.source.calls_for("delta")[0].kwargs["ids"], collection(110).ids[100:])

    async def test_changed_cluster_config_reuses_local_records_and_projection(self):
        initial = (await self.manager().get_or_load("items")).published
        with patch.dict("os.environ", {"QUESTION_CLUSTER_SIMILARITY_THRESHOLD": "0.74"}):
            restored = (await self.manager().get_or_load("items")).published
        self.assertEqual(restored.fit_id, initial.fit_id)
        self.assertNotEqual(restored.revision, initial.revision)
        self.assertEqual(self.cluster.call_count, 2)
        self.assertEqual(self.fit.call_count, 1)
        self.assertEqual(len(self.source.calls_for("full")), 1)
        self.assertFalse(self.source.calls_for("delta"))

    async def test_corrupt_artifacts_rebuild_from_local_records(self):
        initial = (await self.manager().get_or_load("items")).published
        path = next(self.projections.glob("*.sqlite3"))
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE artifacts SET indices = x'0001'")
        rebuilt = (await self.manager().get_or_load("items")).published
        self.assertNotEqual(rebuilt.fit_id, initial.fit_id)
        self.assertTrue(rebuilt.checkpoint_saved)
        self.assertEqual(len(self.source.calls_for("full")), 1)
        self.assertEqual(self.fit.call_count, 2)

    async def test_unavailable_cache_falls_back_to_full_reads(self):
        with patch("embedding_atlas.data_source_manager.cache_path", side_effect=PermissionError("read only")), patch(
            "embedding_atlas.data_source.cache_path", side_effect=PermissionError("read only")
        ):
            manager = self.manager()
            managed = await manager.get_or_load("items")
            initial = managed.published
            self.assertFalse(initial.checkpoint_saved)
            self.set_collection(collection(101))
            await self.refresh(manager, managed)
            self.assertEqual(managed.published.row_count, 101)
            self.assertEqual(managed.published.fit_id, initial.fit_id)
            self.assertEqual(len(self.source.calls_for("full")), 2)
            self.assertFalse(self.source.calls_for("delta"))

    async def test_concurrent_parquet_requests_and_archive_serialize_once(self):
        manager = self.manager()
        snapshot = (await manager.get_or_load("items")).published
        app = make_multi_server(manager, str(self.static), duckdb_uri="wasm")
        frame = snapshot.data_source.dataset
        params = {"revision": snapshot.revision}
        prefix = "/collection/items/data/"
        with patch.object(frame, "to_parquet", wraps=frame.to_parquet) as serialize:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
                first, second, head, archive = await asyncio.gather(
                    http.get(prefix + "dataset.parquet", params=params),
                    http.get(prefix + "dataset.parquet", params=params),
                    http.head(prefix + "dataset.parquet", params=params),
                    http.get(prefix + "archive.zip", params=params),
                )
            serialize.assert_called_once()
        self.assertEqual([res.status_code for res in (first, second, head, archive)], [200] * 4)
        self.assertEqual(first.content, second.content)
        self.assertEqual(head.headers["content-length"], str(len(first.content)))
        with zipfile.ZipFile(BytesIO(archive.content)) as exported:
            self.assertEqual(exported.read("data/dataset.parquet"), first.content)

    async def test_watchers_share_one_refresh_and_endpoints_use_the_new_revision(self):
        manager = self.manager("server")
        with patch("embedding_atlas.data_source_manager.REFRESH_INTERVAL", 0.02):
            app = make_multi_server(manager, str(self.static), duckdb_uri="server")
            watch = next(route.endpoint for route in app.routes if route.path == "/api/collection/{name}/progress")

            async def ready(stream, previous=None):
                async for event in stream:
                    data = json.loads(event["data"])
                    if data["revision"] and data["revision"] != previous:
                        return data

            first = (await watch("items", watch=True)).body_iterator
            second = (await watch("items", watch=True)).body_iterator
            async with aclosing(first), aclosing(second):
                loaded, also_loaded = await asyncio.wait_for(asyncio.gather(ready(first), ready(second)), 5)
                self.assertEqual(loaded["revision"], also_loaded["revision"])
                self.set_collection(collection(101))
                updated, also_updated = await asyncio.wait_for(asyncio.gather(
                    ready(first, loaded["revision"]), ready(second, loaded["revision"])
                ), 5)
                self.assertEqual(updated["revision"], also_updated["revision"])
                self.assertEqual(updated["row_count"], 101)
                self.assertEqual(self.source.count.call_count, 1)
                self.assertEqual(len(self.source.calls_for("full")), 1)
                self.assertEqual(self.source.calls_for("delta")[0].kwargs["ids"], ["0200"])
            self.assertEqual(manager.get_managed("items").subscribers, [])

        prefix = "/collection/items/data/"
        params = {"revision": updated["revision"]}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            stale = await http.get(prefix + "metadata.json", params={"revision": loaded["revision"]})
            self.assertEqual(stale.status_code, 409)
            metadata = await http.get(prefix + "metadata.json", params=params)
            self.assertEqual(metadata.json()["representative_item_column"], "representative_item")
            query = await http.post(prefix + "query", params=params, json={
                "type": "json", "sql": "SELECT count(*) AS count FROM dataset WHERE representative_item = 'left'",
            })
            self.assertEqual(query.json(), [{"count": 101}])
            selection = await http.post(prefix + "selection", params=params, json={
                "predicate": "id = '0200'", "format": "json",
            })
            self.assertEqual(selection.json()[0]["representative_item"], "left")
            archive = await http.get(prefix + "archive.zip", params=params)
            with zipfile.ZipFile(BytesIO(archive.content)) as exported:
                frame = pd.read_parquet(BytesIO(exported.read("data/dataset.parquet")))
            self.assertEqual(frame.representative_item.tolist(), ["left"] * 101)


class CollectionReaderTests(unittest.TestCase):
    def test_explicit_ids_are_batched_and_aligned(self):
        data = collection(2055)
        data.documents = data.ids.copy()
        source = FakeCollection(data)
        reader = ChromaCollectionReader(source)
        requested = data.ids[3:2053]
        records = reader.get_rows(requested[::-1])
        self.assertEqual(records.ids, requested)
        self.assertEqual(records.documents, requested)
        self.assertEqual([len(call.kwargs["ids"]) for call in source.calls_for("delta")], [1000, 1000, 50])
        self.assertFalse(source.calls_for("full"))


class IncrementalNeighborTests(unittest.TestCase):
    def test_merge_matches_exact_neighbors_with_interleaved_new_ids(self):
        embeddings = np.random.default_rng(42).normal(size=(10, 2)).astype(np.float32)
        ids = [str(i) for i in range(len(embeddings))]
        old = np.array([0, 1, 3, 4, 5, 7, 8, 9])
        k = 4

        def brute(vectors, metric):
            vectors = vectors.astype(np.float64)
            if metric == "cosine":
                unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
                distances = 1 - unit @ unit.T
            else:
                distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)
            np.fill_diagonal(distances, 0)
            nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
            return nearest, np.take_along_axis(distances, nearest, axis=1)

        for metric in ("cosine", "euclidean"):
            with self.subTest(metric=metric):
                indices, distances = brute(embeddings[old], metric)
                cache = IncrementalProjectionCache(
                    embeddings[old].copy(), indices, distances, [ids[i] for i in old], len(old), "fit", StubMapper()
                )
                processor = IncrementalUMAPProcessor(IncrementalUMAPConfig(0.5, 1))
                with patch("embedding_atlas.incremental_umap._umap_neighbors") as rebuild, patch(
                    "embedding_atlas.incremental_umap._NEIGHBOR_BLOCK_SIZE", 3
                ):
                    result = processor.update(embeddings, ids, cache, {"metric": metric})
                    rebuild.assert_not_called()
                expected_ids, expected_distances = brute(embeddings, metric)
                np.testing.assert_array_equal(result.knn_indices, expected_ids)
                np.testing.assert_allclose(result.knn_distances, expected_distances, atol=1e-6)
                np.testing.assert_array_equal(result.projection[old], cache.projection)
