"""Deterministic representative-item coverage in the original vector space."""

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


@dataclass
class CollectionQuestionClusters:
    version: int
    config: QuestionClusteringConfig
    rows: dict[str, tuple[str | None, str | None]]  # content ID -> (text, label)
    representative_ids: list[str]  # selected order, using the first row for each text

    def matches(self, ids: list[str]) -> bool:
        if self.rows.keys() != set(ids):
            return False
        representatives = {self.rows[row_id][0] for row_id in self.representative_ids}
        return (
            None not in representatives
            and len(representatives) == len(self.representative_ids)
            and all(
                (label is None or (text is not None and label in representatives))
                for text, label in self.rows.values()
            )
        )


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


def _normalized_embeddings(embeddings, count):
    vectors = np.array(embeddings, dtype=np.float32, copy=True)
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] == 0:
        raise ValueError("Expected one embedding per item, with shape (N, D)")
    if not np.isfinite(vectors).all():
        raise ValueError("Item embeddings contain nonfinite values")
    # Scale first to prevent overflow for large, otherwise valid float32 vectors.
    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Item embeddings contain zero vectors")
    vectors /= scale
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def cluster_questions(texts, embeddings, config=None) -> QuestionClusters:
    """Preserve input order; ties use first occurrence, duplicates use first vector.

    Missing texts remain unassigned and count as uncovered requests. Nonfinite
    and zero vectors are rejected. Similarity work uses O(ND + N) memory,
    with lazy greedy updates instead of storing an N x N similarity matrix.
    """
    config = config or QuestionClusteringConfig.from_env()
    vectors = _normalized_embeddings(embeddings, len(texts))

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


def update_collection_clusters(
    ids: list[str],
    documents: list[str | None],
    embeddings: np.ndarray,
    previous: CollectionQuestionClusters | None,
    config: QuestionClusteringConfig,
) -> CollectionQuestionClusters:
    """Keep existing assignments and classify new texts against fixed representatives."""
    texts = [text if isinstance(text, str) else None for text in documents]
    rows_by_id = {row_id: i for i, row_id in enumerate(ids)}
    if (
        previous is None
        or previous.version != CLUSTER_CACHE_VERSION
        or previous.config != config
        or any(row_id not in rows_by_id or texts[rows_by_id[row_id]] != text
               for row_id, (text, _) in previous.rows.items())
    ):
        result = cached_cluster_questions(texts, embeddings, config)
        first_ids = {}
        for row_id, text in zip(ids, texts):
            first_ids.setdefault(text, row_id)
        return CollectionQuestionClusters(
            CLUSTER_CACHE_VERSION,
            config,
            dict(zip(ids, zip(texts, result.labels))),
            [first_ids[text] for text in result.representatives],
        )

    if len(ids) == len(previous.rows):
        return previous

    new_rows = [i for i, row_id in enumerate(ids) if row_id not in previous.rows]
    new_vectors = _normalized_embeddings(embeddings[new_rows], len(new_rows))
    labels = {text: label for text, label in previous.rows.values() if text is not None}
    new_texts = {}
    for offset, row in enumerate(new_rows):
        text = texts[row]
        if text is not None and text not in labels:
            new_texts.setdefault(text, offset)

    if new_texts and previous.representative_ids:
        representative_rows = [rows_by_id[row_id] for row_id in previous.representative_ids]
        representatives = _normalized_embeddings(embeddings[representative_rows], len(representative_rows))
        candidates = new_vectors[list(new_texts.values())]
        best = np.full(len(candidates), -np.inf, dtype=np.float32)
        nearest = np.zeros(len(candidates), dtype=np.int64)
        for index, representative in enumerate(representatives):
            similarity = candidates @ representative
            closer = similarity > best
            nearest[closer] = index
            best[closer] = similarity[closer]
        for text, index, similarity in zip(new_texts, nearest, best):
            if similarity >= config.similarity_threshold:
                labels[text] = texts[representative_rows[index]]

    rows = previous.rows.copy()
    rows.update((ids[row], (texts[row], labels.get(texts[row]))) for row in new_rows)
    return CollectionQuestionClusters(CLUSTER_CACHE_VERSION, config, rows, previous.representative_ids)


def add_question_cluster_column(data_frame, labels):
    """Add a categorical coloring column without overwriting source metadata."""
    name = "representative_item"
    suffix = 1
    while name in data_frame.columns:
        name = f"representative_item_{suffix}"
        suffix += 1
    data_frame[name] = labels
    return name
