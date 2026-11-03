"""Retrieval confidence from reranker scores (calibration of the threshold comes later)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel


class Confidence(BaseModel):
    top_score: float | None
    probability: float  # sigmoid(top_score / temperature)
    margin: float | None  # top_score - second_score
    threshold: float
    is_confident: bool


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def compute_confidence(
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    temperature: float = 1.0,
    min_margin: float | None = None,
) -> Confidence:
    """Confidence that the best reranked chunk answers the query.

    ``probability = sigmoid(max(scores) / temperature)``; confident iff
    ``probability >= threshold`` (and, optionally, the top-1/top-2 margin >= ``min_margin``).
    No scores means no confidence.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return Confidence(
            top_score=None, probability=0.0, margin=None, threshold=threshold, is_confident=False
        )
    ordered = sorted(scores, reverse=True)
    top = float(ordered[0])
    margin = float(top - ordered[1]) if len(ordered) > 1 else None
    prob = sigmoid(top / temperature)
    confident = prob >= threshold
    if min_margin is not None and margin is not None and margin < min_margin:
        confident = False
    return Confidence(
        top_score=top, probability=prob, margin=margin, threshold=threshold, is_confident=confident
    )
