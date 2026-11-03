"""Calibrate the confidence-gate threshold on the dev split.

Each dev question contributes the gate probability (sigmoid of the top rerank score) and
whether it is answerable. The gate refuses when ``p < tau``. For each candidate ``tau`` we
compute refusal precision/recall/F1 (positives = unanswerable) and the answerable coverage
(share of answerable questions that pass the gate). The chosen ``tau`` maximises
F-beta of refusal (beta < 1 weights precision: wrongly refusing answerable questions is
worse than occasionally answering an unanswerable one, which the LLM may still refuse),
breaking ties by higher coverage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel


class CalibrationPoint(BaseModel):
    threshold: float
    refusal_precision: float
    refusal_recall: float
    refusal_f: float
    answerable_coverage: float


class Calibration(BaseModel):
    threshold: float
    beta: float
    n_dev: int
    n_unanswerable: int
    chosen: CalibrationPoint
    curve: list[CalibrationPoint]


def evaluate_threshold(
    probs: Sequence[float], answerable: Sequence[bool], tau: float, beta: float = 0.5
) -> CalibrationPoint:
    refused = [p < tau for p in probs]
    tp = sum(1 for r, a in zip(refused, answerable, strict=True) if r and not a)
    fp = sum(1 for r, a in zip(refused, answerable, strict=True) if r and a)
    fn = sum(1 for r, a in zip(refused, answerable, strict=True) if not r and not a)
    n_ans = sum(answerable)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    f = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    coverage = (n_ans - fp) / n_ans if n_ans else 0.0
    return CalibrationPoint(
        threshold=tau,
        refusal_precision=precision,
        refusal_recall=recall,
        refusal_f=f,
        answerable_coverage=coverage,
    )


def calibrate_threshold(
    probs: Sequence[float],
    answerable: Sequence[bool],
    *,
    beta: float = 0.5,
    grid: Sequence[float] | None = None,
) -> Calibration:
    if len(probs) != len(answerable):
        raise ValueError("probs and answerable must align")
    if not probs:
        raise ValueError("no dev questions to calibrate on")
    thresholds = list(grid) if grid is not None else [round(t, 3) for t in np.linspace(0, 1, 101)]
    curve = [evaluate_threshold(probs, answerable, t, beta) for t in thresholds]
    best = max(curve, key=lambda c: (c.refusal_f, c.answerable_coverage, -c.threshold))
    return Calibration(
        threshold=best.threshold,
        beta=beta,
        n_dev=len(probs),
        n_unanswerable=sum(1 for a in answerable if not a),
        chosen=best,
        curve=curve,
    )
