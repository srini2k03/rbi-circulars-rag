"""The Q/A test-set model, JSONL I/O and deterministic dev/test splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
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
    # sha256 of doc.text[start:end] in the corpus the span was located in. The offsets
    # are only meaningful for that corpus version; verify_gold() checks them.
    text_sha256: str | None = None


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


SPLIT_METHOD = (
    "by document: every pair whose gold evidence comes from the same notification lands in "
    "the same split (seeded SHA-256 of the gold doc id; unanswerable pairs, which have no "
    "document, are split by qa_id)"
)


def split_key(pair: QAPair) -> str:
    """The unit of splitting: the (first) gold document, so no document's questions are in
    both splits; unanswerable pairs have none and are split individually."""
    docs = sorted(pair.gold_doc_ids)
    return f"doc:{docs[0]}" if docs else f"qa:{pair.qa_id}"


def split_of(key: str, dev_fraction: float, seed: int = 13) -> Literal["dev", "test"]:
    """Stable hash split, independent of file order and of how many pairs exist."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return "dev" if int.from_bytes(h[:8], "big") / 2**64 < dev_fraction else "test"


def split(
    pairs: Sequence[QAPair], dev_fraction: float = 0.3, seed: int = 13
) -> tuple[list[QAPair], list[QAPair]]:
    """Deterministic dev/test split by document (see :data:`SPLIT_METHOD`)."""
    dev = [p for p in pairs if split_of(split_key(p), dev_fraction, seed) == "dev"]
    test = [p for p in pairs if split_of(split_key(p), dev_fraction, seed) == "test"]
    return dev, test


def split_summary(
    dev: Sequence[QAPair], test: Sequence[QAPair], *, dev_fraction: float, seed: int
) -> dict[str, object]:
    return {
        "method": SPLIT_METHOD,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "dev_questions": len(dev),
        "test_questions": len(test),
        "dev_documents": len({d for p in dev for d in p.gold_doc_ids}),
        "test_documents": len({d for p in test for d in p.gold_doc_ids}),
    }


def span_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GoldCheck(BaseModel):
    """Result of checking gold spans against the corpus in use."""

    checked: int = 0
    unpinned: int = 0  # pairs whose spans carry no hash (cannot be verified)
    mismatched: list[str] = Field(default_factory=list)  # qa_ids excluded
    missing_docs: list[str] = Field(default_factory=list)  # doc_ids not in the corpus

    @property
    def message(self) -> str:
        if not self.mismatched:
            return f"gold spans verified for {self.checked} pairs ({self.unpinned} unpinned)"
        return (
            f"{len(self.mismatched)} pairs EXCLUDED: their gold spans no longer match the "
            "corpus (the corpus release changed since the pairs were made; re-locate or "
            f"regenerate them). Missing documents: {len(self.missing_docs)}."
        )


def verify_gold(
    pairs: Sequence[QAPair], texts: Mapping[str, str]
) -> tuple[list[QAPair], GoldCheck]:
    """Keep pairs whose gold spans still point at the same text in ``texts`` (doc_id ->
    Document.text); pairs with a changed or missing span are excluded and reported."""
    ok: list[QAPair] = []
    check = GoldCheck()
    missing: set[str] = set()
    for p in pairs:
        if not p.gold:
            ok.append(p)
            continue
        bad = False
        pinned = True
        for g in p.gold:
            text = texts.get(g.doc_id)
            if text is None or g.end > len(text):
                bad = True
                if text is None:
                    missing.add(g.doc_id)
                continue
            if g.text_sha256 is None:
                pinned = False
            elif span_sha256(text[g.start : g.end]) != g.text_sha256:
                bad = True
        if bad:
            check.mismatched.append(p.qa_id)
        else:
            check.checked += 1
            check.unpinned += int(not pinned)
            ok.append(p)
    check.missing_docs = sorted(missing)
    return ok, check


def pin_gold(pairs: Sequence[QAPair], texts: Mapping[str, str]) -> tuple[list[QAPair], int]:
    """Add hashes to spans that have none, from ``texts`` (the corpus they were made in)."""
    out: list[QAPair] = []
    pinned = 0
    for p in pairs:
        changed = False
        gold: list[GoldSpan] = []
        for g in p.gold:
            text = texts.get(g.doc_id)
            if g.text_sha256 is None and text is not None and g.end <= len(text):
                g = g.model_copy(update={"text_sha256": span_sha256(text[g.start : g.end])})
                changed = True
            gold.append(g)
        if changed:
            pinned += 1
            p = p.model_copy(update={"gold": gold})
        out.append(p)
    return out, pinned
