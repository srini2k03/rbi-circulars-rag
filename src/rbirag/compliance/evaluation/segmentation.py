"""Clause-splitting accuracy on the synthetic policies, as text and as rendered PDFs."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from rbirag.compliance.clauses import split_clauses
from rbirag.compliance.evaluation.dataset import Dataset, render_pdf, render_policy
from rbirag.compliance.evaluation.metrics import SegmentationScore, segmentation_score
from rbirag.compliance.policy import load_policy


class SegmentationRow(BaseModel):
    policy_id: str
    source: str  # "text" | "pdf"
    score: SegmentationScore


def segmentation_report(dataset: Dataset, *, pdf: bool = True) -> list[SegmentationRow]:
    rows: list[SegmentationRow] = []
    with tempfile.TemporaryDirectory() as tmp:
        for policy in dataset.policies:
            labels = dataset.labels_for(policy.policy_id)
            gold = [f"{lb.number} {lb.text}" for lb in labels]
            text = render_policy(policy, labels)
            doc = load_policy(text.encode("utf-8"), filename=f"{policy.policy_id}.md")
            predicted = [c.text for c in split_clauses(doc.text)]
            rows.append(
                SegmentationRow(
                    policy_id=policy.policy_id,
                    source="text",
                    score=segmentation_score(gold, predicted),
                )
            )
            if pdf:
                path = render_pdf(text, Path(tmp) / f"{policy.policy_id}.pdf")
                predicted = [c.text for c in split_clauses(load_policy(path).text)]
                rows.append(
                    SegmentationRow(
                        policy_id=policy.policy_id,
                        source="pdf",
                        score=segmentation_score(gold, predicted),
                    )
                )
    return rows


def segmentation_table(rows: list[SegmentationRow]) -> str:
    lines = ["| policy | source | gold | predicted | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r.score
        lines.append(
            f"| {r.policy_id} | {r.source} | {s.gold} | {s.predicted} | "
            f"{s.precision:.2f} | {s.recall:.2f} | {s.f1:.2f} |"
        )
    for source in sorted({r.source for r in rows}):
        sub = [r.score for r in rows if r.source == source]
        matched = sum(s.matched for s in sub)
        gold = sum(s.gold for s in sub)
        pred = sum(s.predicted for s in sub)
        p, rec = matched / pred if pred else 0.0, matched / gold if gold else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        lines.append(f"| **all** | {source} | {gold} | {pred} | {p:.2f} | {rec:.2f} | {f1:.2f} |")
    return "\n".join(lines)
