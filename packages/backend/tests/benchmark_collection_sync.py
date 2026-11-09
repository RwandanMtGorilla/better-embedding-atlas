"""Benchmark the real Atlas pipeline with a synthetic, in-process Chroma source.

Run cold, warm and append (in that order) in separate Python processes with
the same --cache-dir. Requires psutil for native process memory measurements.
"""

import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil

from embedding_atlas.chroma_client import ChromaDBClient
from embedding_atlas.data_source_manager import DataSourceManager
from embedding_atlas.projection import _umap_neighbors

N, D, APPENDED = 50_000, 384, 10


class FixtureCollection:
    id = "benchmark-fixture-v1"

    def __init__(self):
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((32, D), dtype=np.float32)
        self.vectors = rng.standard_normal((N + APPENDED, D), dtype=np.float32) * np.float32(0.04)
        for group, center in enumerate(centers):
            self.vectors[group::len(centers)] += center
        self.ids = [f"item-{i:05}" for i in range(N + APPENDED)]
        self.documents = [f"Topic {i % len(centers):02}" for i in range(N + APPENDED)]
        self.metadatas = [{"topic": i % len(centers)} for i in range(N + APPENDED)]
        self.count_rows = N
        self.requests = []

    def get_collection(self, _name):
        return self

    def count(self):
        return self.count_rows

    def get(self, *, include, ids=None):
        if ids is None:
            rows = slice(self.count_rows)
            count = self.count_rows
        else:
            rows = [int(row_id.removeprefix("item-")) for row_id in ids]
            count = len(rows)
        self.requests.append(("ids" if not include else "full" if ids is None else "append", count))

        def select(items):
            return items[rows] if isinstance(rows, slice) else [items[row] for row in rows]

        return {
            "ids": select(self.ids),
            "documents": select(self.documents) if include else None,
            "metadatas": select(self.metadatas) if include else None,
            "embeddings": self.vectors[rows].copy() if include else None,
        }


async def run(phase, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if phase == "cold" and any(cache_dir.glob("collection_*.sqlite3")):
        raise ValueError("Use a new cache directory for the cold run")
    if phase != "cold" and not any(cache_dir.glob("collection_*.sqlite3")):
        raise ValueError("Run the cold benchmark first")

    source = FixtureCollection()
    client = ChromaDBClient.__new__(ChromaDBClient)
    client._client = source
    manager = DataSourceManager("benchmark", 8000, umap_args={"metric": "cosine"})
    manager._chroma_client = client
    process = psutil.Process()

    with (
        patch("embedding_atlas.data_source_manager.cache_path", return_value=cache_dir),
        patch("embedding_atlas.data_source.cache_path", return_value=cache_dir),
        patch("embedding_atlas.question_clustering.cache_path", return_value=cache_dir),
        patch("embedding_atlas.projection._umap_neighbors", wraps=_umap_neighbors) as full_fit_neighbors,
        patch("embedding_atlas.incremental_umap._umap_neighbors", wraps=_umap_neighbors) as fallback_neighbors,
    ):
        fit_id = None
        if phase == "append":
            managed = await manager.get_or_load("benchmark")
            fit_id = managed.published.fit_id
            source.count_rows += APPENDED
            source.requests.clear()

        done = threading.Event()
        baseline = process.memory_info().rss
        peak = baseline

        def sample_rss():
            nonlocal peak
            while not done.wait(0.01):
                peak = max(peak, process.memory_info().rss)

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        started = time.perf_counter()
        try:
            if phase == "append":
                managed.last_check = -float("inf")
                manager.check_for_refresh(managed)
                await managed.loading_task
            else:
                managed = await manager.get_or_load("benchmark")
        finally:
            elapsed = time.perf_counter() - started
            peak = max(peak, process.memory_info().rss)
            done.set()
            sampler.join()

        assert managed.published is not None and managed.published.row_count == source.count_rows, managed.logs
        assert managed.published.checkpoint_saved, "Checkpoint was not saved or restored"
        full_neighbor_calls = full_fit_neighbors.call_count + fallback_neighbors.call_count
        ids = sum(count for kind, count in source.requests if kind == "ids")
        records = sum(count for kind, count in source.requests if kind != "ids")
        assert records == {"cold": N, "warm": 0, "append": APPENDED}[phase], source.requests
        if phase == "append":
            assert fit_id == managed.published.fit_id and full_neighbor_calls == 0

        # Report float32 bytes and serialized text fields separately. HTTP/JSON
        # framing and textual encoding of vectors vary by Chroma deployment.
        start = N if phase == "append" else 0
        record_fields = (
            [source.ids[start : start + records], source.documents[start : start + records],
             source.metadatas[start : start + records]] if records else []
        )
        id_bytes = sum(
            len(json.dumps(source.ids[:count], separators=(",", ":")).encode())
            for kind, count in source.requests if kind == "ids"
        )
        print(json.dumps({
            "phase": phase,
            "elapsed_s": round(elapsed, 3),
            "peak_rss_mib": round(peak / 2**20, 1),
            "rss_increase_mib": round((peak - baseline) / 2**20, 1),
            "ids_only_rows": ids,
            "full_record_rows": records,
            "vector_bytes_float32": records * D * 4,
            "text_field_bytes": id_bytes + (len(json.dumps(record_fields, separators=(",", ":")).encode())
                                           if record_fields else 0),
            "full_neighbor_calls": full_neighbor_calls,
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cold", "warm", "append"))
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.phase, args.cache_dir.resolve()))
