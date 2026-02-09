# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""
Incremental UMAP computation module.

Supports adding new data points to an existing UMAP projection without
full recomputation, using the transform() method of umap-learn.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np

from .projection import Projection
from .utils import logger


@dataclass
class IncrementalUMAPConfig:
    """Configuration for incremental UMAP computation."""

    # Maximum ratio of new points for incremental computation (default: 20%)
    max_new_points_ratio: float = 0.2

    # Minimum number of cached samples to consider incremental computation
    min_samples_for_incremental: int = 100

    # Whether to save the UMAP model for future incremental updates
    save_model: bool = True


@dataclass
class IncrementalProjectionCache:
    """Extended projection cache that supports incremental computation."""

    projection: np.ndarray  # (N, 2) 2D coordinates
    knn_indices: np.ndarray  # (N, k) KNN indices
    knn_distances: np.ndarray  # (N, k) KNN distances
    ids: list[str]  # Ordered list of IDs matching projection rows
    umap_model: Optional[object] = None  # Serialized UMAP model

    @staticmethod
    def exists(path: Path, require_model: bool = False) -> bool:
        """Check if cache exists at the given path."""
        basic_exists = Projection.exists(path) and path.with_suffix(".ids.json").exists()
        if not require_model:
            return basic_exists
        return basic_exists and path.with_suffix(".umap_model.pkl").exists()

    @staticmethod
    def supports_incremental(path: Path) -> bool:
        """Check if the cache supports incremental computation."""
        return path.with_suffix(".umap_model.pkl").exists()

    @staticmethod
    def load(path: Path, load_model: bool = True) -> "IncrementalProjectionCache":
        """Load cache from disk."""
        proj = Projection.load(path)

        with open(path.with_suffix(".ids.json"), "r", encoding="utf-8") as f:
            ids = json.load(f)

        umap_model = None
        model_path = path.with_suffix(".umap_model.pkl")
        if load_model and model_path.exists():
            import joblib

            try:
                umap_model = joblib.load(model_path)
            except Exception as e:
                logger.warning("Failed to load UMAP model: %s", e)

        return IncrementalProjectionCache(
            projection=proj.projection,
            knn_indices=proj.knn_indices,
            knn_distances=proj.knn_distances,
            ids=ids,
            umap_model=umap_model,
        )

    def save(self, path: Path, save_model: bool = True) -> None:
        """Save cache to disk."""
        # Save basic projection data
        Projection.save(
            path,
            Projection(
                projection=self.projection,
                knn_indices=self.knn_indices,
                knn_distances=self.knn_distances,
            ),
        )

        # Save IDs
        with open(path.with_suffix(".ids.json"), "w", encoding="utf-8") as f:
            json.dump(self.ids, f)

        # Save UMAP model if available
        if save_model and self.umap_model is not None:
            import joblib

            joblib.dump(
                self.umap_model, path.with_suffix(".umap_model.pkl"), compress=3
            )

    def to_projection(self) -> Projection:
        """Convert to basic Projection object."""
        return Projection(
            projection=self.projection,
            knn_indices=self.knn_indices,
            knn_distances=self.knn_distances,
        )


class IncrementalUMAPProcessor:
    """Processor for incremental UMAP computation."""

    def __init__(self, config: Optional[IncrementalUMAPConfig] = None):
        self.config = config or IncrementalUMAPConfig()

    def compute_or_update(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        cache_path: Path,
        umap_args: dict = {},
    ) -> IncrementalProjectionCache:
        """
        Compute or incrementally update UMAP projection.

        Args:
            embeddings: High-dimensional embedding vectors (N, D)
            ids: List of unique IDs for each row, matching embeddings order
            cache_path: Path to cache files (without extension)
            umap_args: UMAP parameters

        Returns:
            IncrementalProjectionCache with projection results
        """
        if len(embeddings) != len(ids):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and ids ({len(ids)}) must have the same length"
            )

        # Check if incremental-capable cache exists
        if not IncrementalProjectionCache.supports_incremental(cache_path):
            logger.info("No incremental cache found, running full UMAP computation")
            return self._full_compute(embeddings, ids, cache_path, umap_args)

        # Load existing cache
        try:
            cached = IncrementalProjectionCache.load(cache_path, load_model=True)
        except Exception as e:
            logger.warning("Failed to load cache: %s, running full computation", e)
            return self._full_compute(embeddings, ids, cache_path, umap_args)

        if cached.umap_model is None:
            logger.info("Cache exists but no UMAP model, running full computation")
            return self._full_compute(embeddings, ids, cache_path, umap_args)

        # Determine computation strategy
        strategy = self._determine_strategy(cached.ids, ids)
        logger.info("Incremental strategy: %s", strategy)

        if strategy["action"] == "use_cache":
            logger.info("IDs match exactly, using cached projection")
            return cached

        if strategy["action"] == "full":
            logger.info("Strategy requires full recomputation: %s", strategy["reason"])
            return self._full_compute(embeddings, ids, cache_path, umap_args)

        # Incremental computation
        return self._incremental_compute(
            embeddings, ids, cached, cache_path, umap_args, strategy
        )

    def _determine_strategy(
        self, cached_ids: list[str], new_ids: list[str]
    ) -> dict:
        """
        Determine the computation strategy based on ID comparison.

        Returns:
            dict with 'action' ('use_cache', 'incremental', 'full') and details
        """
        cached_id_set = set(cached_ids)
        new_id_set = set(new_ids)

        # Check for exact match (order doesn't matter for this check)
        if cached_id_set == new_id_set:
            # Check if order is the same
            if cached_ids == new_ids:
                return {"action": "use_cache", "reason": "exact_match"}
            else:
                # Same IDs but different order - can reorder cached projection
                return {
                    "action": "reorder",
                    "reason": "same_ids_different_order",
                }

        # Find added and removed IDs
        added_ids = new_id_set - cached_id_set
        removed_ids = cached_id_set - new_id_set

        # If there are removed IDs, we need full recomputation
        # (UMAP model was trained on those points)
        if removed_ids:
            return {
                "action": "full",
                "reason": f"removed_ids ({len(removed_ids)} removed)",
                "removed_count": len(removed_ids),
            }

        # Only additions - check ratio
        if not added_ids:
            return {"action": "use_cache", "reason": "no_changes"}

        cached_count = len(cached_ids)
        added_count = len(added_ids)
        new_ratio = added_count / cached_count if cached_count > 0 else float("inf")

        # Check minimum sample threshold
        if cached_count < self.config.min_samples_for_incremental:
            return {
                "action": "full",
                "reason": f"cached_count ({cached_count}) < min_samples ({self.config.min_samples_for_incremental})",
            }

        # Check new points ratio
        if new_ratio > self.config.max_new_points_ratio:
            return {
                "action": "full",
                "reason": f"new_ratio ({new_ratio:.2%}) > max_ratio ({self.config.max_new_points_ratio:.0%})",
            }

        return {
            "action": "incremental",
            "added_ids": list(added_ids),
            "added_count": added_count,
            "new_ratio": new_ratio,
        }

    def _full_compute(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        cache_path: Path,
        umap_args: dict,
    ) -> IncrementalProjectionCache:
        """Execute full UMAP computation."""
        import umap
        from umap.umap_ import nearest_neighbors

        logger.info(
            "Running full UMAP for %d points with shape %s...",
            len(embeddings),
            embeddings.shape,
        )

        metric = umap_args.get("metric", "cosine")
        n_neighbors = umap_args.get("n_neighbors", 15)

        # Compute KNN
        knn = nearest_neighbors(
            embeddings,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=None,
            angular=False,
            random_state=None,
        )

        # Create and fit UMAP
        mapper = umap.UMAP(**umap_args, precomputed_knn=knn)
        projection = mapper.fit_transform(embeddings)

        result = IncrementalProjectionCache(
            projection=projection,
            knn_indices=knn[0],
            knn_distances=knn[1],
            ids=ids,
            umap_model=mapper if self.config.save_model else None,
        )

        # Save cache
        result.save(cache_path, save_model=self.config.save_model)
        logger.info("Full UMAP computation complete, cache saved to %s", cache_path)

        return result

    def _incremental_compute(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        cached: IncrementalProjectionCache,
        cache_path: Path,
        umap_args: dict,
        strategy: dict,
    ) -> IncrementalProjectionCache:
        """Execute incremental UMAP computation."""
        from umap.umap_ import nearest_neighbors

        logger.info(
            "Running incremental UMAP: %d cached + %d new points",
            len(cached.ids),
            strategy["added_count"],
        )

        # Build ID to index mappings
        cached_id_to_idx = {id_: i for i, id_ in enumerate(cached.ids)}
        new_id_set = set(ids)
        added_id_set = set(strategy["added_ids"])

        # Separate old and new indices in the new data
        old_indices = []  # Indices in new data that exist in cache
        new_indices = []  # Indices in new data that are new

        for i, id_ in enumerate(ids):
            if id_ in added_id_set:
                new_indices.append(i)
            else:
                old_indices.append(i)

        # Reorder cached projections to match new order
        reordered_projection = np.zeros((len(ids), 2), dtype=cached.projection.dtype)

        for new_idx in old_indices:
            id_ = ids[new_idx]
            cached_idx = cached_id_to_idx[id_]
            reordered_projection[new_idx] = cached.projection[cached_idx]

        # Transform new points using the cached UMAP model
        if new_indices:
            new_embeddings = embeddings[new_indices]
            logger.info("Transforming %d new points...", len(new_indices))

            try:
                new_projection = cached.umap_model.transform(new_embeddings)
                for i, new_idx in enumerate(new_indices):
                    reordered_projection[new_idx] = new_projection[i]
            except Exception as e:
                logger.warning(
                    "UMAP transform failed: %s, falling back to full computation", e
                )
                return self._full_compute(embeddings, ids, cache_path, umap_args)

        # Recompute KNN on the complete dataset
        metric = umap_args.get("metric", "cosine")
        n_neighbors = umap_args.get("n_neighbors", 15)

        logger.info("Recomputing KNN for %d total points...", len(embeddings))
        knn = nearest_neighbors(
            embeddings,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=None,
            angular=False,
            random_state=None,
        )

        result = IncrementalProjectionCache(
            projection=reordered_projection,
            knn_indices=knn[0],
            knn_distances=knn[1],
            ids=ids,
            umap_model=cached.umap_model,  # Keep the original model
        )

        # Save updated cache
        result.save(cache_path, save_model=self.config.save_model)
        logger.info(
            "Incremental UMAP complete: %d new points added, cache updated",
            strategy["added_count"],
        )

        return result


def run_incremental_umap(
    embeddings: np.ndarray,
    ids: list[str],
    cache_path: Path,
    umap_args: dict = {},
    config: Optional[IncrementalUMAPConfig] = None,
) -> Projection:
    """
    Convenience function to run incremental UMAP.

    Args:
        embeddings: High-dimensional embedding vectors (N, D)
        ids: List of unique IDs for each row
        cache_path: Path to cache files
        umap_args: UMAP parameters
        config: Incremental computation configuration

    Returns:
        Projection object with 2D coordinates and KNN data
    """
    processor = IncrementalUMAPProcessor(config)
    result = processor.compute_or_update(embeddings, ids, cache_path, umap_args)
    return result.to_projection()
