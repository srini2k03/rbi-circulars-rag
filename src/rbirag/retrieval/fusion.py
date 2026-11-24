"""Score fusion for hybrid retrieval and reranking.

Two ways to combine the BM25 and dense rankings of one query:

* **RRF** (:func:`rbirag.retrieval.hybrid.reciprocal_rank_fusion`): ranks only.
* **Weighted score fusion** (:func:`weighted_fusion`): each retriever's raw scores are
  normalised per query (min-max to [0, 1], or z-scores), multiplied by the retriever's
  weight and summed. A candidate missing from one list gets that list's floor (0 after
  min-max, the minimum z-score after z-normalisation), i.e. "no better than the worst
  candidate it returned".

Normalisation is per query and per retriever, so BM25's unbounded scores and the narrow
cosine band of the embedder are put on one scale only relative to their own candidates; the
weights are tuned on the dev split (see :mod:`rbirag.eval.benchmark`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rbirag.retrieval.hybrid import reciprocal_rank_fusion
from rbirag.retrieval.params import ScoreNorm

__all__ = [
    "interpolate",
    "normalise",
    "rank_map",
    "reciprocal_rank_fusion",
    "sort_scores",
    "weighted_fusion",
]


def normalise(scores: Sequence[float], method: ScoreNorm = "minmax") -> list[float]:
    """Per-query normalisation. A constant list maps to all ones (min-max) / zeros (z)."""
    if not scores:
        return []
    if method == "minmax":
        lo, hi = min(scores), max(scores)
        if hi - lo <= 1e-12:
            return [1.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]
    if method == "zscore":
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        sd = math.sqrt(var)
        if sd <= 1e-12:
            return [0.0] * len(scores)
        return [(s - mean) / sd for s in scores]
    raise ValueError(f"unknown normalisation: {method!r}")


def sort_scores(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    """Best first; ties broken by id so every run orders candidates identically."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float],
    *,
    method: ScoreNorm = "minmax",
) -> list[tuple[str, float]]:
    """``score(d) = sum_i w_i * norm_i(d)`` over ``(id, raw score)`` lists."""
    if len(rankings) != len(weights):
        raise ValueError("one weight per ranking is required")
    fused: dict[str, float] = {}
    normed: list[dict[str, float]] = []
    floors: list[float] = []
    for ranking in rankings:
        values = normalise([s for _, s in ranking], method)
        normed.append({cid: v for (cid, _), v in zip(ranking, values, strict=True)})
        floors.append(min(values) if values and method == "zscore" else 0.0)
    ids = {cid for ranking in rankings for cid, _ in ranking}
    for cid in ids:
        fused[cid] = sum(
            w * norm.get(cid, floor) for w, norm, floor in zip(weights, normed, floors, strict=True)
        )
    return sort_scores(fused)


def interpolate(
    first_stage: Sequence[tuple[str, float]],
    rerank_scores: Mapping[str, float],
    *,
    rerank_weight: float,
) -> list[tuple[str, float]]:
    """Final score of reranked candidates: ``w * minmax(rerank) + (1 - w) * minmax(first)``.

    ``rerank_weight=1`` is pure cross-encoder reranking. Lower weights keep some of the
    first-stage evidence, which helps when the cross-encoder (trained on web search
    queries) is less reliable on regulatory text than BM25.
    """
    pool = [(cid, s) for cid, s in first_stage if cid in rerank_scores]
    if not pool:
        return []
    if rerank_weight >= 1.0:
        return sort_scores({cid: rerank_scores[cid] for cid, _ in pool})
    first = normalise([s for _, s in pool], "minmax")
    rr = normalise([rerank_scores[cid] for cid, _ in pool], "minmax")
    mixed = {
        cid: rerank_weight * r + (1 - rerank_weight) * f
        for (cid, _), f, r in zip(pool, first, rr, strict=True)
    }
    return sort_scores(mixed)


def rank_map(ranking: Sequence[tuple[str, float]]) -> dict[str, int]:
    """``id -> 1-based rank``."""
    return {cid: i for i, (cid, _) in enumerate(ranking, start=1)}
