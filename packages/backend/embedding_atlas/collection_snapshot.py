# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""A local checkpoint of a published Chroma collection, with vectors loaded on demand."""

import json
import logging
import sqlite3
from bisect import bisect_left
from contextlib import closing, contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np

from .chroma_client import ChromaCollectionData
from .projection import Projection

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_SCHEMA = f"""
PRAGMA page_size = 8192;
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS snapshot (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    collection_id TEXT NOT NULL, revision TEXT NOT NULL,
    row_count INTEGER NOT NULL, dimension INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY, content_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
    document TEXT, metadata TEXT
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY, embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    projection_key TEXT NOT NULL, cluster_key TEXT NOT NULL,
    fit_id TEXT NOT NULL, fit_count INTEGER NOT NULL,
    coordinates BLOB NOT NULL, indices BLOB NOT NULL, distances BLOB NOT NULL,
    labels TEXT NOT NULL, representatives TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    fit_id TEXT NOT NULL, value BLOB NOT NULL
);
PRAGMA user_version = {_SCHEMA_VERSION};
COMMIT;
"""


class SnapshotCacheError(Exception):
    """The local checkpoint cannot be used for the expected revision."""


@dataclass(frozen=True)
class SnapshotIndex:
    collection_id: str
    revision: str
    ids: list[str]
    content_ids: list[str]
    fingerprints: list[str]
    dimension: int


@dataclass(frozen=True)
class SnapshotArtifacts:
    projection_key: str
    cluster_key: str
    fit_id: str
    fit_count: int
    projection: Projection
    labels: list[str | None]
    representative_ids: list[str]


def _array_bytes(array: np.ndarray) -> bytes:
    with BytesIO() as output:
        np.save(output, array, allow_pickle=False)
        return output.getvalue()


class CollectionSnapshotStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _read(self, expected: SnapshotIndex | None = None):
        try:
            with closing(sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)) as conn, conn:
                conn.execute("BEGIN")
                if conn.execute("PRAGMA user_version").fetchone()[0] != _SCHEMA_VERSION:
                    raise SnapshotCacheError("Unsupported collection snapshot format")
                if expected is not None:
                    found = conn.execute("SELECT collection_id, revision FROM snapshot").fetchone()
                    if found != (expected.collection_id, expected.revision):
                        raise SnapshotCacheError("Local collection snapshot changed")
                yield conn
        except (sqlite3.Error, OSError, ValueError, TypeError, EOFError) as error:
            raise SnapshotCacheError(str(error)) from error

    def read_index(self) -> SnapshotIndex | None:
        if not self.path.exists():
            return None
        with self._read() as conn:
            state = conn.execute("SELECT collection_id, revision, row_count, dimension FROM snapshot").fetchone()
            if state is None:
                return None
            collection_id, revision, count, dimension = state
            ids, content_ids, fingerprints = [], [], []
            for row_id, content_id, fingerprint in conn.execute(
                "SELECT id, content_id, fingerprint FROM records ORDER BY id"
            ):
                ids.append(row_id)
                content_ids.append(content_id)
                fingerprints.append(fingerprint)
            if len(ids) != count or dimension < 0 or bool(dimension) != bool(ids):
                raise SnapshotCacheError("Invalid collection snapshot index")
            return SnapshotIndex(collection_id, revision, ids, content_ids, fingerprints, dimension)

    def read_rows(
        self,
        index: SnapshotIndex,
        *,
        include_vectors: bool = False,
        additions: ChromaCollectionData | None = None,
    ) -> ChromaCollectionData:
        """Materialize an aligned result. On append, allocate the merged matrix only once."""
        if additions is None:
            ids, dimension = index.ids, index.dimension
            row_positions = None
        else:
            ids = sorted(index.ids + additions.ids)
            dimension = index.dimension if index.ids else additions.embeddings.shape[1]
            row_positions = {row_id: i for i, row_id in enumerate(ids)}

        vectors = np.empty((len(ids), dimension), dtype=np.float32) if include_vectors else None
        documents, metadatas = [None] * len(ids), [None] * len(ids)
        with self._read(index) as conn:
            fields = "r.id, r.content_id, r.fingerprint, r.document, r.metadata"
            sql = (
                f"SELECT {fields}, v.embedding FROM records r JOIN vectors v USING (id) ORDER BY r.id"
                if include_vectors else f"SELECT {fields} FROM records r ORDER BY r.id"
            )
            for i, (record, expected) in enumerate(zip(
                conn.execute(sql), zip(index.ids, index.content_ids, index.fingerprints, strict=True), strict=True
            )):
                if record[:3] != expected:
                    raise SnapshotCacheError("Local collection rows are not aligned")
                row = i if row_positions is None else row_positions[record[0]]
                documents[row] = record[3]
                metadata = None if record[4] is None else json.loads(record[4])
                if metadata is not None and not isinstance(metadata, dict):
                    raise SnapshotCacheError("Invalid cached metadata")
                metadatas[row] = metadata
                if vectors is not None:
                    if len(record[5]) != dimension * 4:
                        raise SnapshotCacheError("Invalid cached embedding dimension")
                    vectors[row] = np.frombuffer(record[5], dtype="<f4")

        if additions is not None:
            for i, row_id in enumerate(additions.ids):
                row = row_positions[row_id]
                documents[row], metadatas[row] = additions.documents[i], additions.metadatas[i]
                if vectors is not None:
                    vectors[row] = additions.embeddings[i]
        return ChromaCollectionData(ids, documents, metadatas, vectors)

    def read_artifacts(self, index: SnapshotIndex) -> SnapshotArtifacts | None:
        with self._read(index) as conn:
            result = conn.execute("""
                SELECT projection_key, cluster_key, fit_id, fit_count, coordinates, indices, distances,
                    labels, representatives FROM artifacts
                """).fetchone()
            if result is None:
                return None
            key, cluster_key, fit_id, fit_count, coordinates, indices, distances, labels, representatives = result
            arrays = [np.load(BytesIO(blob), allow_pickle=False) for blob in (coordinates, indices, distances)]
            if any(not isinstance(array, np.ndarray) for array in arrays):
                raise SnapshotCacheError("Invalid collection projection arrays")
            coordinates, indices, distances = arrays
            labels, representatives = json.loads(labels), json.loads(representatives)
            count = len(index.ids)
            if (
                not fit_id or not 0 < fit_count <= count
                or coordinates.shape != (count, 2) or not np.issubdtype(coordinates.dtype, np.floating)
                or indices.ndim != 2 or indices.shape[0] != count or indices.shape != distances.shape
                or not np.issubdtype(indices.dtype, np.signedinteger)
                or not np.issubdtype(distances.dtype, np.floating)
                or np.any((indices < -1) | (indices >= count))
                or not isinstance(labels, list) or len(labels) != count
                or any(label is not None and not isinstance(label, str) for label in labels)
                or not isinstance(representatives, list)
                or any(not isinstance(row_id, str) for row_id in representatives)
                or not set(representatives).issubset(index.content_ids)
            ):
                raise SnapshotCacheError("Invalid collection projection or labels")
            return SnapshotArtifacts(
                key, cluster_key, fit_id, fit_count, Projection(coordinates, indices, distances), labels, representatives
            )

    def read_model(self, index: SnapshotIndex, fit_id: str) -> object | None:
        try:
            with self._read(index) as conn:
                row = conn.execute("SELECT singleton FROM model WHERE fit_id = ?", (fit_id,)).fetchone()
                if row is None:
                    return None
                # Avoid copying the entire compressed model out of SQLite before loading it.
                with conn.blobopen("model", "value", row[0], readonly=True) as value:
                    return joblib.load(value)
        except Exception as error:
            logger.warning("Could not restore collection model from %s: %s", self.path, error)
            return None

    def save(
        self,
        index: SnapshotIndex,
        data: ChromaCollectionData,
        artifacts: SnapshotArtifacts | None,
        model: object | None,
        *,
        base: SnapshotIndex | None = None,
    ):
        try:
            self._write(self.path, index, data, artifacts, model, base)
        except (sqlite3.DatabaseError, SnapshotCacheError) as error:
            code = getattr(error, "sqlite_errorcode", 0)
            corrupt = (code & 0xFF) in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)
            if base is not None or not (corrupt or isinstance(error, SnapshotCacheError)):
                raise
            # Keep the previous file in place until a full, valid checkpoint is ready.
            replacement = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                self._write(replacement, index, data, artifacts, model, None)
                replacement.replace(self.path)
            finally:
                replacement.unlink(missing_ok=True)

    def _write(self, path, index, data, artifacts, model, base):
        with closing(sqlite3.connect(path)) as conn, conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                conn.executescript(_SCHEMA)
            elif version != _SCHEMA_VERSION:
                raise SnapshotCacheError("Unsupported collection snapshot format")
            else:
                tables = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                if not {"snapshot", "records", "vectors", "artifacts", "model"}.issubset(tables):
                    raise SnapshotCacheError("Incomplete collection snapshot schema")

            conn.execute("BEGIN IMMEDIATE")
            if base is not None:
                state = conn.execute("SELECT collection_id, revision FROM snapshot").fetchone()
                if state != (base.collection_id, base.revision):
                    raise SnapshotCacheError("Cannot append to a different collection snapshot")
                positions = (bisect_left(index.ids, row_id) for row_id in data.ids)
                hashes = ((index.content_ids[i], index.fingerprints[i]) for i in positions)
            else:
                conn.execute("DELETE FROM records")
                conn.execute("DELETE FROM vectors")
                hashes = zip(index.content_ids, index.fingerprints, strict=True)
            conn.executemany(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                ((row_id, content_id, fingerprint, document, None if metadata is None else json.dumps(metadata))
                 for row_id, (content_id, fingerprint), document, metadata in zip(
                     data.ids, hashes, data.documents, data.metadatas, strict=True
                 )),
            )
            conn.executemany(
                "INSERT INTO vectors VALUES (?, ?)",
                ((row_id, np.asarray(vector, dtype="<f4").tobytes())
                 for row_id, vector in zip(data.ids, data.embeddings, strict=True)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO snapshot VALUES (1, ?, ?, ?, ?)",
                (index.collection_id, index.revision, len(index.ids), index.dimension),
            )

            if artifacts is None:
                conn.execute("DELETE FROM artifacts")
                conn.execute("DELETE FROM model")
                return
            projection = artifacts.projection
            conn.execute("""
                INSERT OR REPLACE INTO artifacts VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                artifacts.projection_key, artifacts.cluster_key, artifacts.fit_id, artifacts.fit_count,
                _array_bytes(projection.projection), _array_bytes(projection.knn_indices),
                _array_bytes(projection.knn_distances), json.dumps(artifacts.labels, ensure_ascii=False),
                json.dumps(artifacts.representative_ids),
            ))
            # The fitted model is immutable across transforms. Never serialize it on an append.
            fit = conn.execute("SELECT fit_id FROM model").fetchone()
            if fit != (artifacts.fit_id,):
                conn.execute("DELETE FROM model")
                if model is not None:
                    with BytesIO() as output:
                        joblib.dump(model, output, compress=3)
                        conn.execute("INSERT INTO model VALUES (1, ?, ?)", (artifacts.fit_id, output.getvalue()))
