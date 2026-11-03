"""AI review of model-generated Q/A pairs, kept separate from human verification.

An AI assistant (not a person) reviewed a stratified sample of the candidates against the
source documents. Its decisions live in ``data/eval/qa_ai_reviewed.jsonl``, one record per
reviewed pair, keyed by ``qa_id`` and append-only: pairs the daily generator adds later
are simply unreviewed, and a review only applies while the candidate still has the
question and answer that were reviewed.

Applying the reviews (:func:`apply_ai_reviews`) drops AI-rejected pairs, substitutes the
AI-edited question/answer, and annotates each reviewed pair with ``ai_review``. It never
sets ``verified`` or ``review_status``: those mean a *human* decision, and a human
decision always takes precedence over an AI one.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from rbirag.eval.dataset import AI_REVIEWER, AIDecision, AIReview, QAPair, QuestionType

__all__ = [
    "AI_REVIEWER",
    "AI_REVIEW_FILE",
    "SELECTION_METHOD",
    "AIDecision",
    "AIReview",
    "AIReviewRecord",
    "append_ai_reviews",
    "apply_ai_reviews",
    "read_ai_reviews",
    "select_for_review",
]

logger = logging.getLogger(__name__)

AI_REVIEW_FILE = "qa_ai_reviewed.jsonl"
SELECTION_METHOD = (
    "answerable pairs passing the automatic checks with no human decision, sorted by qa_id "
    "and shuffled with random.Random(seed); question-type quotas proportional to the pool "
    "(largest remainder); slots filled round-robin over types, each time taking the first "
    "shuffled pair whose year and department are least represented so far; unanswerable "
    "pairs (if any) shuffled with the same seed, first 10 taken"
)


class AIReviewRecord(BaseModel):
    """One line of ``qa_ai_reviewed.jsonl``."""

    qa_id: str
    qtype: QuestionType
    answerable: bool
    year: int | None = None
    department: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    original_question: str
    original_answer: str
    question: str  # final wording (== original unless edited)
    answer: str
    review: AIReview
    selection_seed: int | None = None


def read_ai_reviews(path: Path) -> dict[str, AIReviewRecord]:
    """Records keyed by ``qa_id``; for a repeated id the last line wins (append-only)."""
    out: dict[str, AIReviewRecord] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = AIReviewRecord.model_validate(json.loads(line))
                out[rec.qa_id] = rec
    return out


def append_ai_reviews(records: Iterable[AIReviewRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
            n += 1
    return n


def apply_ai_reviews(
    pairs: Sequence[QAPair], reviews: dict[str, AIReviewRecord]
) -> tuple[list[QAPair], list[str]]:
    """Pairs with AI edits applied and AI rejections removed, plus the rejected ids.

    A review is ignored when a human already decided on the pair, or when the candidate's
    question/answer no longer match what was reviewed (it would describe another pair).
    """
    out: list[QAPair] = []
    rejected: list[str] = []
    for p in pairs:
        rec = reviews.get(p.qa_id)
        if rec is None or p.review_status != "pending":
            out.append(p)
            continue
        if (p.question, p.answer) != (rec.original_question, rec.original_answer):
            logger.warning("AI review of %s is stale (pair changed); ignoring it", p.qa_id)
            out.append(p)
            continue
        if rec.review.decision == "reject":
            rejected.append(p.qa_id)
            continue
        update: dict[str, object] = {"ai_review": rec.review}
        if rec.review.decision == "edit":
            update |= {"question": rec.question, "answer": rec.answer}
        out.append(p.model_copy(update=update))
    return out, rejected


def select_for_review(
    pairs: Sequence[QAPair], n_answerable: int = 50, n_unanswerable: int = 10, seed: int = 0
) -> list[QAPair]:
    """Stratified, seeded sample for review (see :data:`SELECTION_METHOD`)."""
    from rbirag.eval.qa_generate import allocate

    rng = random.Random(seed)
    pool = sorted(
        (
            p
            for p in pairs
            if p.answerable and p.auto_check is not False and p.review_status == "pending"
        ),
        key=lambda p: p.qa_id,
    )
    rng.shuffle(pool)
    by_type: dict[str, list[QAPair]] = {}
    for p in pool:
        by_type.setdefault(p.qtype, []).append(p)
    quota = allocate({t: len(v) for t, v in by_type.items()}, n_answerable)
    years: Counter[int | None] = Counter()
    depts: Counter[str | None] = Counter()
    picked: list[QAPair] = []
    while any(quota[t] > 0 for t in quota):
        for t in sorted(quota):
            if quota[t] <= 0:
                continue
            members = by_type[t]
            best = min(
                range(len(members)),
                key=lambda i: (years[members[i].year] + depts[members[i].department], i),
            )
            p = members.pop(best)
            years[p.year] += 1
            depts[p.department] += 1
            picked.append(p)
            quota[t] -= 1
    unans = sorted((p for p in pairs if not p.answerable), key=lambda p: p.qa_id)
    rng.shuffle(unans)
    return picked + unans[:n_unanswerable]
