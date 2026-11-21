"""Choose the abstention threshold on the dev split, after the fact.

The guardrails turn a substantive verdict whose confidence is below the threshold into
``insufficient_evidence``. Predictions store the confidence the guardrails computed, so
the effect of any threshold can be simulated on recorded dev predictions without new LLM
calls. The chosen threshold is then fixed for the test run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from rbirag.compliance.evaluation.dataset import EvalItem
from rbirag.compliance.evaluation.metrics import macro_f1
from rbirag.compliance.records import SUBSTANTIVE_VERDICTS, ClausePrediction

GRID = tuple(round(0.05 * i, 2) for i in range(0, 20))


class ThresholdRow(BaseModel):
    threshold: float
    accuracy: float
    macro_f1: float
    abstained: int  # substantive verdicts turned into insufficient_evidence


def apply_threshold(pred: ClausePrediction, threshold: float) -> str:
    if (
        pred.verdict in SUBSTANTIVE_VERDICTS
        and pred.confidence is not None
        and pred.confidence < threshold
    ):
        return "insufficient_evidence"
    return pred.verdict


def sweep(
    items: Sequence[EvalItem],
    predictions: Mapping[str, ClausePrediction],
    grid: Sequence[float] = GRID,
) -> list[ThresholdRow]:
    pairs = [
        (it.label.verdict, predictions[it.item_id]) for it in items if it.item_id in predictions
    ]
    rows: list[ThresholdRow] = []
    for t in grid:
        golds = [g for g, _ in pairs]
        preds = [
            apply_threshold(p, t) if p.error is None else "insufficient_evidence" for _, p in pairs
        ]
        rows.append(
            ThresholdRow(
                threshold=t,
                accuracy=sum(g == p for g, p in zip(golds, preds, strict=True))
                / max(1, len(pairs)),
                macro_f1=macro_f1(golds, preds),
                abstained=sum(
                    1
                    for (_, p), v in zip(pairs, preds, strict=True)
                    if v == "insufficient_evidence" and p.verdict in SUBSTANTIVE_VERDICTS
                ),
            )
        )
    return rows


def choose(rows: Sequence[ThresholdRow]) -> ThresholdRow:
    """Highest macro-F1; ties go to the lowest threshold (abstain as little as possible)."""
    return max(rows, key=lambda r: (round(r.macro_f1, 6), -r.threshold))


def threshold_table(rows: Sequence[ThresholdRow], chosen: ThresholdRow) -> str:
    lines = ["| threshold | accuracy | macro-F1 | abstained |", "|---|---|---|---|"]
    for r in rows:
        mark = " **(chosen)**" if r.threshold == chosen.threshold else ""
        lines.append(
            f"| {r.threshold:.2f}{mark} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {r.abstained} |"
        )
    return "\n".join(lines)
