import asyncio
import json
import unittest
import zipfile
from contextlib import aclosing
from io import BytesIO
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pandas as pd

from embedding_atlas.chroma_client import ChromaCollectionData
from embedding_atlas.data_source_manager import DataSourceManager
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
        self.enterContext(patch("embedding_atlas.incremental_umap._umap_neighbors", side_effect=neighbors))
        self.cluster = self.enterContext(patch(
            "embedding_atlas.question_clustering.cluster_questions", wraps=cluster_questions
        ))
        self.client = Mock()
        self.set_collection(collection(100))

    def set_collection(self, data):
        self.client.get_snapshot.return_value = data
        self.client.get_collection_count.return_value = len(data.ids)

    def manager(self, mode="wasm"):
        manager = DataSourceManager("test", 1234, duckdb_uri=mode)
        manager._chroma_client = self.client
        return manager

    async def refresh(self, manager, managed):
        managed.last_check = -float("inf")
        manager.check_for_refresh(managed)
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
        await self.refresh(manager, managed)
        latest = managed.published
        frame = latest.data_source.dataset.set_index("id")
        old_frame = initial.data_source.dataset.set_index("id")
        pd.testing.assert_series_equal(frame.loc[old_frame.index, "projection_x"], old_frame.projection_x)
        pd.testing.assert_series_equal(frame.loc[old_frame.index, "representative_item"], old_frame.representative_item)
        self.assertEqual(frame.loc[updated.ids[100:105], "representative_item"].tolist(),
                         ["left", None, None, "left", "left"])
        self.assertEqual(latest.incremental_cache.umap_model.transformed, [10])
        self.assertEqual(latest.fit_id, initial.fit_id)
        self.assertEqual(self.cluster.call_count, 1)

        current = self.client.get_snapshot.return_value
        extension = collection(121)
        manager = self.manager()
        managed = await manager.get_or_load("items")
        self.assertEqual(managed.published.revision, latest.revision)
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
                self.assertEqual(managed.published.incremental_cache.umap_model.transformed, [10, 10])
                self.assertEqual(self.cluster.call_count, 1)
        self.assertEqual(managed.published.incremental_cache.fit_count, 121)
        self.assertNotEqual(managed.published.fit_id, initial.fit_id)
        self.assertEqual(self.fit.call_count, 2)
        self.assertEqual(self.cluster.call_count, 2)

    async def test_reorder_remaps_neighbors_and_reuses_labels(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        initial = managed.published
        data = collection(100)
        data.ids.reverse()
        self.set_collection(data)
        await manager._load_data_source(managed)
        self.assertEqual(managed.published.fit_id, initial.fit_id)
        self.assertEqual(self.cluster.call_count, 1)
        frame = managed.published.data_source.dataset
        self.assertEqual(frame.projection_x.iloc[:2].tolist(), [99, 98])
        self.assertEqual(frame["__neighbors"].iloc[0]["ids"], [99])
        self.assertEqual(frame["__neighbors"].iloc[1]["ids"], [0])

    async def test_text_edit_reclusters_and_transform_failure_refits(self):
        manager = self.manager()
        managed = await manager.get_or_load("items")
        first_fit = managed.published.fit_id
        updated = collection(101)
        updated.documents[0] = "renamed"
        self.set_collection(updated)
        await self.refresh(manager, managed)
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
        self.client.get_snapshot.assert_called_once()
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
        path = next(self.projections.glob("*.incremental.joblib"))
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
        path = next(self.projections.glob("*.incremental.joblib"))
        path.write_bytes(b"broken cache")
        reloaded = await self.manager().get_or_load("items")
        self.assertNotEqual(reloaded.published.fit_id, managed.published.fit_id)
        restored = await self.manager().get_or_load("items")
        self.assertEqual(restored.published.fit_id, reloaded.published.fit_id)
        self.assertEqual(self.fit.call_count, 3)

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
                self.assertEqual(self.client.get_collection_count.call_count, 1)
                self.assertEqual(self.client.get_snapshot.call_count, 2)
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
