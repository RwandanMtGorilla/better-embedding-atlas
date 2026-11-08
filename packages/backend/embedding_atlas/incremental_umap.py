# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Reuse an existing UMAP fit when a collection grows."""

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np

from .projection import Projection, _fit_umap, _umap_neighbors
from .question_clustering import CollectionQuestionClusters
from .utils import logger

_CACHE_VERSION = 1


@dataclass
class IncrementalUMAPConfig:
    max_new_points_ratio: float = 0.2
    min_samples_for_incremental: int = 100
    save_model: bool = True


@dataclass
class IncrementalProjectionCache:
    projection: np.ndarray
    knn_indices: np.ndarray
    knn_distances: np.ndarray
    ids: list[str]  # ordered content IDs matching the projection rows
    fit_count: int
    fit_id: str
    umap_model: object | None = None
    clusters: CollectionQuestionClusters | None = None

    @staticmethod
    def load(path: Path) -> "IncrementalProjectionCache | None":
        path = path.with_suffix(".incremental.joblib")
        try:
            version, cached = joblib.load(path)
            if version != _CACHE_VERSION:
                raise ValueError("Unsupported incremental cache version")
            if cached is not None:
                if not isinstance(cached, IncrementalProjectionCache):
                    raise ValueError("Invalid incremental cache")
                count = len(cached.ids)
                if (
                    not 0 < cached.fit_count <= count
                    or not cached.fit_id
                    or cached.projection.shape != (count, 2)
                    or cached.knn_indices.shape != cached.knn_distances.shape
                    or cached.knn_indices.ndim != 2
                    or cached.knn_indices.shape[0] != count
                    or len(set(cached.ids)) != count
                    or cached.clusters is not None and not cached.clusters.matches(cached.ids)
                ):
                    raise ValueError("Invalid incremental cache")
            return cached
        except FileNotFoundError:
            return None
        except Exception as error:
            logger.warning("Could not load incremental cache %s: %s", path, error)
            return None

    @staticmethod
    def save(path: Path, value: "IncrementalProjectionCache | None") -> None:
        path = path.with_suffix(".incremental.joblib")
        temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            # Projection, fit baseline, and representatives must advance together.
            joblib.dump((_CACHE_VERSION, value), temporary, compress=3)
            temporary.replace(path)
        except Exception as error:
            logger.warning("Could not save incremental cache %s: %s", path, error)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def to_projection(self) -> Projection:
        return Projection(self.projection, self.knn_indices, self.knn_distances)


class IncrementalUMAPProcessor:
    """The fit ID changes only when the projection is rebuilt."""

    def __init__(self, config: IncrementalUMAPConfig | None = None):
        self.config = config or IncrementalUMAPConfig()

    def compute_or_update(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        cache_path: Path,
        umap_args: dict | None = None,
    ) -> IncrementalProjectionCache:
        cached = IncrementalProjectionCache.load(cache_path)
        result = self.update(embeddings, ids, cached, umap_args)
        if result is not cached:
            IncrementalProjectionCache.save(cache_path, result)
        return result

    def update(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        cached: IncrementalProjectionCache | None,
        umap_args: dict | None = None,
    ) -> IncrementalProjectionCache:
        """Prepare an update without persisting anything."""
        if len(embeddings) != len(ids):
            raise ValueError("Embeddings and IDs must have the same length")
        umap_args = umap_args or {}
        if cached is not None:
            if cached.ids == ids:
                return cached

            cached_rows = {row_id: i for i, row_id in enumerate(cached.ids)}
            previous_rows = np.array([cached_rows.get(row_id, -1) for row_id in ids], dtype=int)
            new_rows = np.flatnonzero(previous_rows == -1)
            if not len(new_rows) and len(ids) == len(cached.ids):
                knn_indices = cached.knn_indices[previous_rows].copy()
                present = knn_indices >= 0
                knn_indices[present] = np.argsort(previous_rows)[knn_indices[present]]
                return replace(
                    cached,
                    ids=ids,
                    projection=cached.projection[previous_rows],
                    knn_indices=knn_indices,
                    knn_distances=cached.knn_distances[previous_rows],
                )

            ratio = (len(ids) - cached.fit_count) / cached.fit_count
            if (
                cached.umap_model is not None
                and len(ids) - len(new_rows) == len(cached.ids)
                and cached.fit_count >= self.config.min_samples_for_incremental
                and ratio <= self.config.max_new_points_ratio
            ):
                try:
                    new_projection = np.asarray(cached.umap_model.transform(embeddings[new_rows]))
                    if new_projection.shape != (len(new_rows), 2):
                        raise ValueError("Unexpected UMAP transform shape")
                except Exception as error:
                    logger.warning("UMAP transform failed, rebuilding projection: %s", error)
                else:
                    knn_indices, knn_distances, _ = _umap_neighbors(embeddings, umap_args)
                    projection = np.empty((len(ids), 2), dtype=cached.projection.dtype)
                    old_rows = previous_rows >= 0
                    projection[old_rows] = cached.projection[previous_rows[old_rows]]
                    projection[new_rows] = new_projection
                    logger.info("Transformed %d new rows (%.2f%% growth since fit)", len(new_rows), 100 * ratio)
                    return replace(
                        cached,
                        projection=projection,
                        knn_indices=knn_indices,
                        knn_distances=knn_distances,
                        ids=ids,
                        clusters=None,
                    )

        return self._full_compute(embeddings, ids, umap_args)

    def _full_compute(
        self, embeddings: np.ndarray, ids: list[str], umap_args: dict
    ) -> IncrementalProjectionCache:
        projection, mapper = _fit_umap(embeddings, umap_args)
        return IncrementalProjectionCache(
            projection.projection, projection.knn_indices, projection.knn_distances, ids, len(ids), uuid4().hex,
            mapper if self.config.save_model else None,
        )


def run_incremental_umap(
    embeddings: np.ndarray,
    ids: list[str],
    cache_path: Path,
    umap_args: dict | None = None,
    config: IncrementalUMAPConfig | None = None,
) -> Projection:
    processor = IncrementalUMAPProcessor(config)
    return processor.compute_or_update(embeddings, ids, cache_path, umap_args).to_projection()
