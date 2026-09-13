from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from embedding_atlas.question_clustering import (
    QuestionClusteringConfig, add_question_cluster_column, cluster_questions,
    cached_cluster_questions,
)
from embedding_atlas.chroma_client import ChromaDBClient
from embedding_atlas.projection import Projection, compute_vector_projection, _projection_for_texts
from embedding_atlas.chroma_client import ChromaCollectionData
from embedding_atlas.data_source_manager import DataSourceManager, ManagedDataSource, LoadingStatus


@contextmanager
def cache_directory():
    # mkdir with inherited permissions also works in restricted Windows runners.
    directory = Path(__file__).resolve().parent / ("cache_" + uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        for file in directory.iterdir():
            file.unlink()
        directory.rmdir()


def angle(degrees):
    radians = np.deg2rad(degrees)
    return [np.cos(radians), np.sin(radians)]


class QuestionClusteringTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(cache_directory())
        self.enterContext(patch("embedding_atlas.question_clustering.cache_path", return_value=directory))

    def test_frequency_first_and_original_row_mapping(self):
        texts = ["rare", "frequent", "frequent", "other"]
        vectors = np.array([angle(10), angle(0), angle(0), angle(90)]) * 7
        original = vectors.copy()
        result = cluster_questions(texts, vectors)
        self.assertEqual(result.labels, ["frequent", "frequent", "frequent", "other"])
        self.assertEqual(result.request_coverage, 1)
        np.testing.assert_array_equal(vectors, original)

    def test_exact_recount_does_not_inherit_merged_coverage(self):
        texts = ["C", "C", "C", "A", "A", "B"]
        result = cluster_questions(texts, [angle(d) for d in [0, 0, 0, 40, 40, 53]])
        self.assertEqual(result.representatives, ["C"])
        self.assertEqual(result.merged_unique_coverage, 1)
        self.assertAlmostEqual(result.unique_coverage, 2 / 3)
        self.assertAlmostEqual(result.request_coverage, 5 / 6)
        self.assertIsNone(result.labels[-1])

    def test_both_targets_are_required(self):
        texts = ["popular"] * 100 + [f"q{i}" for i in range(9)]
        vectors = np.eye(10)[[0] * 100 + list(range(1, 10))]
        result = cluster_questions(texts, vectors)
        self.assertEqual(len(result.representatives), 8)
        self.assertEqual(result.unique_coverage, .8)
        self.assertGreaterEqual(result.request_coverage, .901)

    def test_request_target_continues_after_unique_target(self):
        result = cluster_questions([str(i) for i in range(10)], np.eye(10))
        self.assertEqual(len(result.representatives), 10)

    def test_exact_text_keeps_whitespace_and_first_vector(self):
        result = cluster_questions(["q", "q", " q"], [angle(0), angle(90), angle(90)])
        self.assertEqual(result.labels, ["q", "q", " q"])

    def test_invalid_vectors_and_config(self):
        for vectors in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[1, 0], [0, 1]]):
            with self.assertRaises(ValueError):
                cluster_questions(["q"], vectors)
        with self.assertRaises(ValueError):
            QuestionClusteringConfig(similarity_threshold=1.1)
        with self.assertRaises(ValueError):
            QuestionClusteringConfig(unique_weight=.5)

    def test_missing_text_and_empty_input(self):
        self.assertEqual(cluster_questions([], np.empty((0, 2))).labels, [])
        result = cluster_questions([None, "q"], [angle(0), angle(0)])
        self.assertEqual(result.labels, [None, "q"])
        self.assertEqual(result.request_coverage, .5)

    def test_column_collision(self):
        frame = pd.DataFrame({"representative_question": ["original"]})
        name = add_question_cluster_column(frame, ["new"])
        self.assertEqual(name, "representative_question_1")
        self.assertEqual(frame.iloc[0, 0], "original")

    def test_snapshot_pairs_and_sorts_in_one_request(self):
        client = ChromaDBClient.__new__(ChromaDBClient)
        collection = Mock()
        client._client = Mock()
        client._client.get_collection.return_value = collection
        collection.get.return_value = dict(ids=["b", "a"], documents=["B", "A"],
                                           metadatas=[None, None], embeddings=[[0, 1], [1, 0]])
        data = client.get_snapshot("test")
        collection.get.assert_called_once_with(include=["documents", "metadatas", "embeddings"])
        self.assertEqual(data.ids, ["a", "b"])
        self.assertEqual(data.documents, ["A", "B"])
        np.testing.assert_array_equal(data.embeddings, [[1, 0], [0, 1]])
        self.assertEqual(data.embeddings.dtype, np.float32)

    def test_clustering_precedes_umap_and_preserves_rows(self):
        frame = pd.DataFrame({"text": ["a", "b", "a"], "vector": [[1, 0], [0, 1], [1, 0]]}, index=[8, 3, 6])
        def umap(vectors, args):
            self.assertEqual(frame.representative_question.tolist(), ["a", "b", "a"])
            return Projection(np.zeros((3, 2)), np.zeros((3, 1), dtype=int), np.zeros((3, 1)))
        with patch("embedding_atlas.projection._run_umap", side_effect=umap):
            compute_vector_projection(frame, "vector", text="text")
        self.assertEqual(frame.index.tolist(), [8, 3, 6])

    def test_text_projection_cache_preserves_labels(self):
        projection = Projection(np.zeros((2, 2)), np.zeros((2, 1), dtype=int), np.zeros((2, 1)))
        callback = Mock(return_value=np.eye(2))
        callback.__name__ = "test_embeddings"
        with cache_directory() as directory:
            with patch("embedding_atlas.projection.cache_path", return_value=Path(directory)), patch(
                "embedding_atlas.projection._run_umap", return_value=projection
            ) as umap:
                first = _projection_for_texts(["a", "b"], text_projector=callback)
                cached = _projection_for_texts(["a", "b"], text_projector=callback)
                self.assertEqual(first.question_labels, cached.question_labels)
                self.assertEqual(cached.question_labels, ["a", "b"])
                callback.assert_called_once()
                umap.assert_called_once()


class DataSourceClusteringTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(cache_directory())
        self.enterContext(patch("embedding_atlas.question_clustering.cache_path", return_value=directory))

    async def test_cached_projection_still_gets_current_labels(self):
        manager = DataSourceManager("test", 1234)
        client = Mock()
        client.get_snapshot.return_value = ChromaCollectionData(
            ["a", "b"], ["A", "B"], [None, None], np.eye(2, dtype=np.float32)
        )
        manager._chroma_client = client
        projection = Projection(np.zeros((2, 2)), np.zeros((2, 1), dtype=int), np.zeros((2, 1)))
        managed = ManagedDataSource("test")
        module = "embedding_atlas.data_source_manager."
        with patch(module + "cache_path", return_value=Path(__file__).parent), patch(
            module + "Projection.exists", return_value=True
        ), patch(module + "Projection.load", return_value=projection), patch(module + "DataSource") as source:
            await manager._load_data_source(managed)
            self.assertEqual(managed.loading_progress.status, LoadingStatus.READY)
            frame = source.call_args.args[1]
            self.assertEqual(frame.representative_question.tolist(), ["A", "B"])
            self.assertEqual(frame.id.tolist(), ["a", "b"])
            client.get_embeddings.assert_not_called()

    async def test_incremental_vector_edit_changes_content_ids(self):
        manager = DataSourceManager("test", 1234, enable_incremental=True)
        client = Mock()
        data = ChromaCollectionData(["a", "b"], ["A", "B"], [None, None], np.eye(2, dtype=np.float32))
        client.get_snapshot.return_value = data
        manager._chroma_client = client
        projection = Projection(np.zeros((2, 2)), np.zeros((2, 1), dtype=int), np.zeros((2, 1)))
        module = "embedding_atlas.data_source_manager."
        with patch(module + "cache_path", return_value=Path(__file__).parent), patch(
            module + "IncrementalUMAPProcessor"
        ) as processor, patch(module + "DataSource") as source:
            processor.return_value.compute_or_update.return_value.to_projection.return_value = projection
            await manager._load_data_source(ManagedDataSource("test"))
            before = processor.return_value.compute_or_update.call_args.args[1]
            data.embeddings[1] = [1, 0]
            await manager._load_data_source(ManagedDataSource("test"))
            after = processor.return_value.compute_or_update.call_args.args[1]
            self.assertEqual(before[0], after[0])
            self.assertNotEqual(before[1], after[1])
            self.assertEqual(source.call_args.args[1].representative_question.tolist(), ["A", "A"])


class QuestionClusterCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(cache_directory())
        self.enterContext(patch("embedding_atlas.question_clustering.cache_path", return_value=self.directory))
        self.compute = self.enterContext(patch(
            "embedding_atlas.question_clustering.cluster_questions", wraps=cluster_questions
        ))

    def test_disk_hit_restores_all_fields_without_computation(self):
        texts = ["C", "C", "C", "A", "A", "B", None]
        vectors = np.array([angle(d) for d in [0, 0, 0, 40, 40, 53, 90]], dtype=np.float32)
        first = cached_cluster_questions(texts, vectors)
        # Distinct objects and no in-memory result shared with the cache reader.
        first.labels[0] = "modified locally"
        second = cached_cluster_questions(list(texts), vectors.copy())
        self.assertEqual(self.compute.call_count, 1)
        self.assertEqual(second.labels, ["C"] * 5 + [None, None])
        self.assertEqual(second.representatives, ["C"])
        self.assertEqual(second.unique_coverage, first.unique_coverage)
        self.assertEqual(second.request_coverage, first.request_coverage)
        self.assertEqual(second.merged_unique_coverage, first.merged_unique_coverage)
        self.assertEqual(second.merged_request_coverage, first.merged_request_coverage)
        self.assertEqual(len(list(self.directory.iterdir())), 1)

    def test_text_vector_order_shape_and_every_parameter_invalidate(self):
        config = QuestionClusteringConfig()
        texts = ["a", "b", "a"]
        vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        cases = [
            (texts, vectors, config),
            (["changed", "b", "a"], vectors, config),
            (texts, np.array([[1, 0], [0, 1], [2, 0]]), config),
            (list(reversed(texts)), vectors[::-1].copy(), config),
            (["a", "b"], vectors.reshape(2, 3), config),
        ]
        # Reverse a non-palindromic paired sequence to test stable order.
        cases[3] = ([texts[i] for i in [1, 0, 2]], vectors[[1, 0, 2]], config)
        for changes in (
            {"near_duplicate_threshold": .98}, {"similarity_threshold": .74},
            {"unique_weight": .5, "request_weight": .5},
            {"unique_target": .85}, {"request_target": .95},
        ):
            cases.append((texts, vectors, replace(config, **changes)))
        for index, (questions, embeddings, settings) in enumerate(cases, 1):
            with self.subTest(index=index):
                cached_cluster_questions(questions, embeddings, settings)
                self.assertEqual(self.compute.call_count, index)
                cached_cluster_questions(questions, embeddings, settings)
                self.assertEqual(self.compute.call_count, index)

    def test_environment_parameters_invalidate(self):
        with patch.dict("os.environ", {"QUESTION_CLUSTER_SIMILARITY_THRESHOLD": "0.73"}):
            cached_cluster_questions(["a", "b"], np.eye(2))
        with patch.dict("os.environ", {"QUESTION_CLUSTER_SIMILARITY_THRESHOLD": "0.8"}):
            cached_cluster_questions(["a", "b"], np.eye(2))
            cached_cluster_questions(["a", "b"], np.eye(2))
        self.assertEqual(self.compute.call_count, 2)

    def test_corrupt_cache_is_recomputed_and_replaced(self):
        expected = cached_cluster_questions(["a", "b"], np.eye(2))
        path = next(self.directory.glob("*.json"))
        for corrupt in ('{"incomplete":', '{}', '[]'):
            path.write_text(corrupt, encoding="utf-8")
            self.assertEqual(cached_cluster_questions(["a", "b"], np.eye(2)), expected)
        self.assertEqual(self.compute.call_count, 4)
        self.assertEqual(cached_cluster_questions(["a", "b"], np.eye(2)), expected)
        self.assertEqual(self.compute.call_count, 4)

    def test_unwritable_cache_does_not_block_result(self):
        with patch("embedding_atlas.question_clustering.cache_path", side_effect=PermissionError("read only")):
            result = cached_cluster_questions(["a", "b"], np.eye(2))
        self.assertEqual(result.labels, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
