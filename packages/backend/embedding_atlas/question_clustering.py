"""Deterministic representative-question coverage in the original vector space."""

import heapq
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from uuid import uuid4

import numpy as np

from .utils import cache_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionClusteringConfig:
    near_duplicate_threshold: float = 0.97
    similarity_threshold: float = 0.73
    unique_weight: float = 0.4
    request_weight: float = 0.6
    unique_target: float = 0.8
    request_target: float = 0.901

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if self.near_duplicate_threshold < self.similarity_threshold:
            raise ValueError("Near-duplicate threshold must be >= coverage threshold")
        if not np.isclose(self.unique_weight + self.request_weight, 1):
            raise ValueError("Unique and request weights must sum to 1")

    @classmethod
    def from_env(cls):
        return cls(**{
            name: float(os.environ[f"QUESTION_CLUSTER_{name.upper()}"])
            for name in cls.__dataclass_fields__
            if f"QUESTION_CLUSTER_{name.upper()}" in os.environ
        })


@dataclass
class QuestionClusters:
    labels: list[str | None]
    representatives: list[str]
    unique_coverage: float
    request_coverage: float
    merged_unique_coverage: float
    merged_request_coverage: float


CLUSTER_CACHE_VERSION = 1


def cached_cluster_questions(texts, embeddings, config=None) -> QuestionClusters:
    """Restore results only for identical ordered float32 records and settings."""
    config = config or QuestionClusteringConfig.from_env()
    # Match the algorithm's treatment of missing/non-text values. Hash raw
    # float32 vectors before normalization, including duplicate rows and shape.
    texts = [text if isinstance(text, str) else None for text in texts]
    vectors = np.asarray(embeddings, dtype="<f4", order="C")
    header = {
        "version": CLUSTER_CACHE_VERSION,
        "texts": texts,
        "shape": vectors.shape,
        "config": asdict(config),
    }
    digest = hashlib.sha256(
        json.dumps(header, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    digest.update(vectors.tobytes(order="C"))
    key = digest.hexdigest()
    path = None
    try:
        path = cache_path("projections") / f"question-clusters-{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["version"] != CLUSTER_CACHE_VERSION or payload["key"] != key:
                raise ValueError("Cluster cache identity mismatch")
            result = QuestionClusters(**payload["result"])
            if not isinstance(result.labels, list) or len(result.labels) != len(texts):
                raise ValueError("Cluster cache row count mismatch")
            if not isinstance(result.representatives, list) or not all(
                isinstance(value, str) for value in result.representatives
            ):
                raise ValueError("Invalid cached representatives")
            representatives = set(result.representatives)
            if not representatives.issubset({text for text in texts if text is not None}):
                raise ValueError("Cached representatives are absent from the input")
            if not all(
                label is None or isinstance(label, str) and label in representatives
                for label in result.labels
            ):
                raise ValueError("Invalid cached labels")
            for name, value in asdict(result).items():
                if name.endswith("coverage") and (
                    not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1
                ):
                    raise ValueError("Invalid cached coverage")
            logger.info(
                "Using cached question clustering: %d representatives; "
                "exact coverage unique=%.2f%% requests=%.2f%% (%s)",
                len(result.representatives), 100 * result.unique_coverage,
                100 * result.request_coverage, path,
            )
            return result
    except (OSError, ValueError, TypeError, KeyError) as error:
        logger.warning("Question clustering cache unavailable; recomputing: %s", error)

    result = cluster_questions(texts, vectors, config)
    if path is not None:
        # Unique temporary files + atomic replace prevent partial reads when
        # workers/processes compute the same collection concurrently.
        temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                json.dump({"version": CLUSTER_CACHE_VERSION, "key": key, "result": asdict(result)},
                          output, ensure_ascii=False, allow_nan=False)
            temporary.replace(path)
            logger.info("Saved question clustering cache: %s", path)
        except OSError as error:
            logger.warning("Could not save question clustering cache: %s", error)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return result


def cluster_questions(texts, embeddings, config=None) -> QuestionClusters:
    """Preserve input order; ties use first occurrence, duplicates use first vector.

    Missing texts remain unassigned and count as uncovered requests. Nonfinite
    and zero vectors are rejected. Similarity work uses O(ND + N) memory,
    with lazy greedy updates instead of storing an N x N similarity matrix.
    """
    config = config or QuestionClusteringConfig.from_env()
    vectors = np.array(embeddings, dtype=np.float32, copy=True)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise ValueError("Expected one embedding per question, with shape (N, D)")
    if not np.isfinite(vectors).all():
        raise ValueError("Question embeddings contain nonfinite values")
    # Scale first to prevent overflow for large, otherwise valid float32 vectors.
    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Question embeddings contain zero vectors")
    vectors /= scale
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    lookup, unique_texts, first, counts, inverse = {}, [], [], [], []
    for row, text in enumerate(texts):
        if not isinstance(text, str):
            inverse.append(-1)
            continue
        if text not in lookup:
            lookup[text] = len(first)
            unique_texts.append(text)
            first.append(row)
            counts.append(0)
        index = lookup[text]
        counts[index] += 1
        inverse.append(index)
    n = len(first)
    if not n:
        return QuestionClusters([None] * len(texts), [], 0, 0, 0, 0)
    vectors = vectors[first]
    counts = np.asarray(counts, dtype=np.int64)
    groups = np.full(n, -1, dtype=np.int64)
    seeds = []
    for index in np.argsort(-counts, kind="stable"):
        if groups[index] >= 0:
            continue
        members = (groups < 0) & (vectors @ vectors[index] >= config.near_duplicate_threshold)
        members[index] = True
        groups[members] = len(seeds)
        seeds.append(index)
    candidates = vectors[seeds]
    unique_mass = np.bincount(groups)
    request_mass = np.bincount(groups, weights=counts)
    covered = np.zeros(len(seeds), dtype=bool)
    masses = config.unique_weight * unique_mass / n + config.request_weight * request_mass / len(texts)

    def neighborhood(index):
        mask = candidates @ candidates[index] >= config.similarity_threshold
        mask[index] = True
        return mask

    # Heap keys also encode stable input order for equal marginal scores.
    heap = [(-float(masses[neighborhood(i)].sum()), seeds[i], i, 0) for i in range(len(seeds))]
    heapq.heapify(heap)
    selected = []
    merged_unique = merged_request = 0.0
    while heap and (merged_unique < config.unique_target or merged_request < config.request_target):
        neg_score, seed, index, iteration = heapq.heappop(heap)
        mask = neighborhood(index)
        if iteration != len(selected):
            score = float(masses[mask & ~covered].sum())
            heapq.heappush(heap, (-score, seed, index, len(selected)))
            continue
        if -neg_score <= 0:
            break
        selected.append(seed)
        covered |= mask
        merged_unique = float(unique_mass[covered].sum() / n)
        merged_request = float(request_mass[covered].sum() / len(texts))

    # Re-evaluate exact phrasings: merged coverage is only an approximation.
    best = np.full(n, -np.inf, dtype=np.float32)
    nearest = np.full(n, -1, dtype=np.int64)
    for seed in selected:
        similarity = vectors @ vectors[seed]
        similarity[seed] = 1.0
        update = similarity > best
        nearest[update] = seed
        best[update] = similarity[update]
    exact_covered = best >= config.similarity_threshold
    unique_coverage = float(exact_covered.mean())
    request_coverage = float(counts[exact_covered].sum() / len(texts))
    labels = [unique_texts[nearest[i]] if i >= 0 and exact_covered[i] else None for i in inverse]
    logger.info(
        "Question clustering: %d requests, %d exact questions, %d near-duplicate groups, %d representatives; "
        "merged coverage unique=%.2f%% requests=%.2f%%; exact coverage unique=%.2f%% requests=%.2f%%",
        len(texts), n, len(seeds), len(selected), 100 * merged_unique, 100 * merged_request,
        100 * unique_coverage, 100 * request_coverage,
    )
    if unique_coverage < config.unique_target or request_coverage < config.request_target:
        logger.warning("Exact-question coverage is below target; consider adjusting question clustering thresholds")
    return QuestionClusters(labels, [unique_texts[i] for i in selected], unique_coverage,
                            request_coverage, merged_unique, merged_request)


def add_question_cluster_column(data_frame, labels):
    """Add a categorical coloring column without overwriting source metadata."""
    name = "representative_question"
    suffix = 1
    while name in data_frame.columns:
        name = f"representative_question_{suffix}"
        suffix += 1
    data_frame[name] = labels
    return name
