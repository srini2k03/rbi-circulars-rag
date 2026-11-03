"""Reciprocal Rank Fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks.

    Only ranks matter, not raw scores, so BM25 and cosine scores never need calibrating
    against each other. Ties are broken by best single rank, then by id (deterministic).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # count each id once per ranking (its best position)
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))
    return fused[:limit] if limit is not None else fused
