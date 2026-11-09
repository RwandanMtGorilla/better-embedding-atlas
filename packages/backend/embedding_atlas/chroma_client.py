# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""ChromaDB client wrapper for embedding-atlas."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChromaDBError(Exception):
    """Base exception for ChromaDB operations."""

    pass


class ChromaDBConnectionError(ChromaDBError):
    """Failed to connect to ChromaDB server."""

    pass


class ChromaDBCollectionNotFoundError(ChromaDBError):
    """Requested collection does not exist."""

    pass


class ChromaDeltaError(ChromaDBError):
    """The requested records no longer match an ID probe."""


@dataclass
class ChromaCollectionData:
    """Container for ChromaDB collection data."""

    ids: list[str]
    documents: list[str | None]
    metadatas: list[dict[str, Any] | None]
    embeddings: np.ndarray | None = None


class ChromaCollectionReader:
    """Read one collection by UUID, even if its name is later reused."""

    DELTA_BATCH_SIZE = 1000

    def __init__(self, collection):
        self._collection = collection

    @property
    def uuid(self) -> str:
        return str(self._collection.id)

    def count(self) -> int:
        return self._collection.count()

    def get_ids(self) -> list[str]:
        ids = self._collection.get(include=[])["ids"]
        if any(not isinstance(row_id, str) for row_id in ids) or len(set(ids)) != len(ids):
            raise ChromaDBError("Invalid collection ID list")
        return sorted(ids)

    def get_snapshot(self, *, allow_empty: bool = False) -> ChromaCollectionData:
        """Read paired records in one response to avoid pagination across writes."""
        return self._parse_records(self._get_records(), allow_empty=allow_empty)

    def get_rows(self, ids: list[str]) -> ChromaCollectionData:
        """Read explicit IDs in bounded batches; never align by response order."""
        ids = sorted(ids)
        batches = []
        for offset in range(0, len(ids), self.DELTA_BATCH_SIZE):
            requested = ids[offset : offset + self.DELTA_BATCH_SIZE]
            result = self._get_records(ids=requested)
            try:
                batch = self._parse_records(result, allow_empty=True)
            except (ChromaDBError, KeyError, TypeError, ValueError) as error:
                raise ChromaDeltaError("Invalid Chroma response for new IDs") from error
            if batch.ids != requested or (batches and batch.embeddings.shape[1] != batches[0].embeddings.shape[1]):
                raise ChromaDeltaError("New Chroma records changed during the read")
            batches.append(batch)

        if not batches:
            return ChromaCollectionData([], [], [], np.empty((0, 0), dtype=np.float32))
        if len(batches) == 1:
            return batches[0]
        return ChromaCollectionData(
            ids,
            [document for batch in batches for document in batch.documents],
            [metadata for batch in batches for metadata in batch.metadatas],
            np.concatenate([batch.embeddings for batch in batches]),
        )

    def _get_records(self, **kwargs):
        return self._collection.get(include=["documents", "metadatas", "embeddings"], **kwargs)

    @staticmethod
    def _parse_records(result, *, allow_empty: bool) -> ChromaCollectionData:
        ids = result["ids"]
        embeddings = result["embeddings"]
        if allow_empty and not ids and (embeddings is None or len(embeddings) == 0):
            return ChromaCollectionData([], [], [], np.empty((0, 0), dtype=np.float32))
        if not ids or embeddings is None or len(embeddings) != len(ids):
            raise ChromaDBError("Collection must contain one embedding per record")
        if any(not isinstance(row_id, str) for row_id in ids) or len(set(ids)) != len(ids):
            raise ChromaDBError("Snapshot contains invalid or duplicate record IDs")
        documents = result["documents"]
        metadatas = result["metadatas"]
        if documents is None:
            documents = [None] * len(ids)
        if metadatas is None:
            metadatas = [None] * len(ids)
        if len(documents) != len(ids) or len(metadatas) != len(ids):
            raise ChromaDBError("Snapshot records are not aligned")
        vectors = np.asarray(embeddings, dtype=np.float32, order="C")
        if vectors.ndim != 2 or not vectors.shape[1]:
            raise ChromaDBError("Snapshot contains invalid embedding dimensions")
        if all(a < b for a, b in zip(ids, ids[1:])):
            return ChromaCollectionData(list(ids), list(documents), list(metadatas), vectors)
        order = sorted(range(len(ids)), key=ids.__getitem__)
        return ChromaCollectionData(
            [ids[i] for i in order], [documents[i] for i in order], [metadatas[i] for i in order], vectors[order]
        )


@dataclass
class VectorSearchResult:
    """Container for vector search results."""

    ids: list[str]
    distances: list[float]


class ChromaDBClient:
    """Wrapper for ChromaDB HTTP client with pagination support."""

    DEFAULT_BATCH_SIZE = 5000

    def __init__(self, host: str, port: int):
        """
        Initialize ChromaDB client.

        Args:
            host: ChromaDB server host
            port: ChromaDB server port
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb package is required for --chroma-collection. "
                "Please install it with: pip install chromadb"
            )

        self.host = host
        self.port = port

        try:
            self._client = chromadb.HttpClient(host=host, port=port)
            # Test connection by listing collections
            self._client.list_collections()
            logger.info("Connected to ChromaDB at %s:%d", host, port)
        except Exception as e:
            raise ChromaDBConnectionError(
                f"Failed to connect to ChromaDB at {host}:{port}. "
                f"Please ensure the server is running. Error: {e}"
            )

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the total number of items in a collection.

        Args:
            collection_name: Name of the ChromaDB collection

        Returns:
            Total row count
        """
        try:
            return self.open_collection(collection_name).count()
        except Exception as e:
            if "does not exist" in str(e).lower():
                collections = [c.name for c in self._client.list_collections()]
                raise ChromaDBCollectionNotFoundError(
                    f"Collection '{collection_name}' not found. "
                    f"Available collections: {collections}"
                )
            raise

    def open_collection(self, collection_name: str) -> ChromaCollectionReader:
        return ChromaCollectionReader(self._client.get_collection(collection_name))

    def get_documents_and_metadata(
        self,
        collection_name: str,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> ChromaCollectionData:
        """
        Fetch only documents and metadata from collection (without embeddings).
        Used for cache validation before deciding whether to fetch embeddings.

        Args:
            collection_name: Name of the ChromaDB collection
            batch_size: Number of items to fetch per batch
            show_progress: Whether to show progress bar

        Returns:
            ChromaCollectionData with documents and metadata (embeddings=None)
        """
        import tqdm

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        try:
            collection = self._client.get_collection(collection_name)
        except Exception as e:
            if "does not exist" in str(e).lower():
                collections = [c.name for c in self._client.list_collections()]
                raise ChromaDBCollectionNotFoundError(
                    f"Collection '{collection_name}' not found. "
                    f"Available collections: {collections}"
                )
            raise

        total_count = collection.count()

        logger.info(
            "Fetching documents and metadata from collection '%s' (%d items)...",
            collection_name,
            total_count,
        )

        all_ids: list[str] = []
        all_documents: list[str | None] = []
        all_metadatas: list[dict[str, Any] | None] = []

        iterator = range(0, total_count, batch_size)
        if show_progress:
            iterator = tqdm.tqdm(
                iterator,
                desc="Fetching documents",
                unit="batch",
                total=(total_count + batch_size - 1) // batch_size,
            )

        for offset in iterator:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_ids.extend(result["ids"])
            if result["documents"] is not None:
                all_documents.extend(result["documents"])
            else:
                all_documents.extend([None] * len(result["ids"]))
            if result["metadatas"] is not None:
                all_metadatas.extend(result["metadatas"])
            else:
                all_metadatas.extend([None] * len(result["ids"]))

        return ChromaCollectionData(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas,
            embeddings=None,
        )

    def get_embeddings(
        self,
        collection_name: str,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Fetch only embeddings from collection.
        Called after cache miss to get embedding vectors for UMAP.

        Args:
            collection_name: Name of the ChromaDB collection
            batch_size: Number of items to fetch per batch
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings with shape (N, embedding_dim)
        """
        import tqdm

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        collection = self._client.get_collection(collection_name)
        total_count = collection.count()

        logger.info(
            "Fetching embeddings from collection '%s' (%d items)...",
            collection_name,
            total_count,
        )

        all_embeddings: list[list[float]] = []

        iterator = range(0, total_count, batch_size)
        if show_progress:
            iterator = tqdm.tqdm(
                iterator,
                desc="Fetching embeddings",
                unit="batch",
                total=(total_count + batch_size - 1) // batch_size,
            )

        for offset in iterator:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings"],
            )
            if result["embeddings"] is not None:
                all_embeddings.extend(result["embeddings"])

        if not all_embeddings:
            raise ChromaDBError(
                f"Collection '{collection_name}' has no embeddings. "
                "Please ensure the collection contains embedding vectors."
            )

        return np.array(all_embeddings)

    def get_snapshot(self, collection_name: str, *, allow_empty: bool = False) -> ChromaCollectionData:
        return self.open_collection(collection_name).get_snapshot(allow_empty=allow_empty)

    def get_all_data(
        self,
        collection_name: str,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> ChromaCollectionData:
        """
        Fetch all data including embeddings from collection.

        Args:
            collection_name: Name of the ChromaDB collection
            batch_size: Number of items to fetch per batch
            show_progress: Whether to show progress bar

        Returns:
            ChromaCollectionData with all data including embeddings
        """
        import tqdm

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        try:
            collection = self._client.get_collection(collection_name)
        except Exception as e:
            if "does not exist" in str(e).lower():
                collections = [c.name for c in self._client.list_collections()]
                raise ChromaDBCollectionNotFoundError(
                    f"Collection '{collection_name}' not found. "
                    f"Available collections: {collections}"
                )
            raise

        total_count = collection.count()

        logger.info(
            "Fetching all data from collection '%s' (%d items)...",
            collection_name,
            total_count,
        )

        all_ids: list[str] = []
        all_documents: list[str | None] = []
        all_metadatas: list[dict[str, Any] | None] = []
        all_embeddings: list[list[float]] = []

        iterator = range(0, total_count, batch_size)
        if show_progress:
            iterator = tqdm.tqdm(
                iterator,
                desc="Fetching data",
                unit="batch",
                total=(total_count + batch_size - 1) // batch_size,
            )

        for offset in iterator:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            all_ids.extend(result["ids"])
            if result["documents"] is not None:
                all_documents.extend(result["documents"])
            else:
                all_documents.extend([None] * len(result["ids"]))
            if result["metadatas"] is not None:
                all_metadatas.extend(result["metadatas"])
            else:
                all_metadatas.extend([None] * len(result["ids"]))
            if result["embeddings"] is not None:
                all_embeddings.extend(result["embeddings"])

        embeddings = np.array(all_embeddings) if all_embeddings else None

        return ChromaCollectionData(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas,
            embeddings=embeddings,
        )

    def list_collections_with_info(self) -> list[dict]:
        """
        List all collections with their metadata.

        Returns:
            List of dicts containing collection info:
            - name: Collection name
            - count: Number of items in collection
            - metadata: Collection metadata (if any)
        """
        collections = self._client.list_collections()
        result = []
        for col in collections:
            result.append({
                "name": col.name,
                "count": col.count(),
                "metadata": col.metadata,
            })
        return result

    def vector_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 100,
    ) -> VectorSearchResult:
        """
        Perform vector similarity search on a collection.

        Args:
            collection_name: Name of the ChromaDB collection
            query_embedding: Query vector for similarity search
            n_results: Maximum number of results to return

        Returns:
            VectorSearchResult with ids and distances
        """
        try:
            collection = self._client.get_collection(collection_name)
        except Exception as e:
            if "does not exist" in str(e).lower():
                collections = [c.name for c in self._client.list_collections()]
                raise ChromaDBCollectionNotFoundError(
                    f"Collection '{collection_name}' not found. "
                    f"Available collections: {collections}"
                )
            raise

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances"],
        )

        return VectorSearchResult(
            ids=results["ids"][0] if results["ids"] else [],
            distances=results["distances"][0] if results["distances"] else [],
        )


def chroma_data_to_dataframe(data: ChromaCollectionData) -> pd.DataFrame:
    """
    Convert ChromaDB collection data to pandas DataFrame.

    Metadata fields are flattened to top-level columns.

    Args:
        data: ChromaCollectionData from ChromaDB client

    Returns:
        pandas DataFrame with columns: id, document, + flattened metadata fields
    """
    records = []
    for i in range(len(data.ids)):
        record: dict[str, Any] = {
            "id": data.ids[i],
            "document": data.documents[i] if data.documents else None,
        }

        # Flatten metadata to top-level columns
        if data.metadatas and i < len(data.metadatas) and data.metadatas[i]:
            for key, value in data.metadatas[i].items():
                # Avoid overwriting id and document columns
                if key not in ("id", "document"):
                    record[key] = value

        records.append(record)

    return pd.DataFrame(records)
