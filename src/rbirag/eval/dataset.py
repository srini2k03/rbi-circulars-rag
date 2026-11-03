"""The Q/A test-set model, JSONL I/O and deterministic dev/test splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["factual", "numeric", "procedural", "multi_chunk", "unanswerable"]
ReviewStatus = Literal["pending", "accepted", "edited", "rejected"]
AIDecision = Literal["accept", "edit", "reject"]
AI_REVIEWER = "ai-assistant (Claude)"


class GoldSpan(BaseModel):
    """Character span of the evidence inside ``Document.text`` (chunker-independent)."""

    doc_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class AIReview(BaseModel):
    """A review by an AI assistant. It is never a human verification (see ``verified``)."""

    reviewer: str = AI_REVIEWER
    decision: AIDecision
    notes: str
    reviewed_at: str


class QAPair(BaseModel):
    qa_id: str
    question: str
    answer: str  # reference answer; empty for unanswerable questions
    qtype: QuestionType
    answerable: bool = True
    gold: list[GoldSpan] = Field(default_factory=list)
    evidence: str = ""  # quoted evidence used to locate ``gold``
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_text: str = ""  # the passage shown to the generator (and to the reviewer)
    doc_title: str | None = None
    year: int | None = None
    department: str | None = None
    generator_model: str | None = None
    auto_check: bool | None = None  # passed the automatic consistency checks
    review_status: ReviewStatus = "pending"
    verified: bool = False  # True only when a human accepted or edited it
    reviewer_note: str = ""
    ai_review: AIReview | None = None  # set in memory from qa_ai_reviewed.jsonl, not by humans

    @property
    def gold_doc_ids(self) -> set[str]:
        return {g.doc_id for g in self.gold}


def write_qa_jsonl(pairs: Iterable[QAPair], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for p in pairs:
            # the AI review lives in its own file (qa_ai_reviewed.jsonl), never in these
            fh.write(p.model_dump_json(exclude={"ai_review"}) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_qa_jsonl(path: Path) -> list[QAPair]:
    out: list[QAPair] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(QAPair.model_validate(json.loads(line)))
    return out


def usable(pairs: Sequence[QAPair]) -> list[QAPair]:
    """Pairs that count in evaluation: not rejected by a human, and not auto-failed."""
    return [p for p in pairs if p.review_status != "rejected" and p.auto_check is not False]


def split_of(qa_id: str, dev_fraction: float, seed: int = 13) -> Literal["dev", "test"]:
    """Stable hash split, independent of file order and of how many pairs exist."""
    h = hashlib.sha256(f"{seed}:{qa_id}".encode()).digest()
    return "dev" if int.from_bytes(h[:8], "big") / 2**64 < dev_fraction else "test"


def split(
    pairs: Sequence[QAPair], dev_fraction: float = 0.3, seed: int = 13
) -> tuple[list[QAPair], list[QAPair]]:
    dev = [p for p in pairs if split_of(p.qa_id, dev_fraction, seed) == "dev"]
    test = [p for p in pairs if split_of(p.qa_id, dev_fraction, seed) == "test"]
    return dev, test
