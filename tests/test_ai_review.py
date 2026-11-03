"""AI review records: selection, edit overrides, rejection filtering and labelling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rbirag.eval.ai_review import (
    AI_REVIEWER,
    AIReview,
    AIReviewRecord,
    append_ai_reviews,
    apply_ai_reviews,
    read_ai_reviews,
    select_for_review,
)
from rbirag.eval.dataset import QAPair, QuestionType, read_qa_jsonl, write_qa_jsonl
from rbirag.eval.report import describe_test_set, run_metadata

TYPES: list[QuestionType] = ["factual", "numeric", "procedural", "multi_chunk"]


def pair(i: int, **kw: object) -> QAPair:
    base: dict[str, object] = {
        "qa_id": f"q{i:03d}",
        "question": f"question {i}?",
        "answer": f"answer {i}",
        "qtype": TYPES[i % 4],
        "year": 2000 + i % 7,
        "department": f"dept {i % 5}",
        "auto_check": True,
    }
    base.update(kw)
    return QAPair.model_validate(base)


def record(p: QAPair, decision: str, **kw: str) -> AIReviewRecord:
    return AIReviewRecord(
        qa_id=p.qa_id,
        qtype=p.qtype,
        answerable=p.answerable,
        original_question=p.question,
        original_answer=p.answer,
        question=kw.get("question", p.question),
        answer=kw.get("answer", p.answer),
        review=AIReview(decision=decision, notes="note", reviewed_at="2026-11-03"),  # type: ignore[arg-type]
    )


def test_apply_edits_rejects_and_keeps_verified_human_only() -> None:
    pairs = [pair(i) for i in range(4)]
    reviews = {
        r.qa_id: r
        for r in [
            record(pairs[0], "accept"),
            record(pairs[1], "edit", question="better question?", answer="better answer"),
            record(pairs[2], "reject"),
        ]
    }
    out, rejected = apply_ai_reviews(pairs, reviews)
    assert rejected == ["q002"]
    by_id = {p.qa_id: p for p in out}
    assert set(by_id) == {"q000", "q001", "q003"}
    assert (by_id["q001"].question, by_id["q001"].answer) == ("better question?", "better answer")
    assert by_id["q000"].question == "question 0?"
    assert by_id["q001"].ai_review is not None
    assert by_id["q001"].ai_review.reviewer == AI_REVIEWER == "ai-assistant (Claude)"
    assert by_id["q003"].ai_review is None
    assert not any(p.verified for p in out)
    assert all(p.review_status == "pending" for p in out)


def test_human_decision_wins_and_stale_reviews_are_ignored() -> None:
    human = pair(0, review_status="accepted", verified=True)
    changed = pair(1)
    reviews = {
        "q000": record(human, "reject"),
        "q001": record(changed.model_copy(update={"question": "old wording?"}), "reject"),
    }
    out, rejected = apply_ai_reviews([human, changed], reviews)
    assert rejected == []
    assert [p.ai_review for p in out] == [None, None]
    assert out[0].verified is True


def test_records_are_append_only_and_keyed_by_id(tmp_path: Path) -> None:
    path = tmp_path / "qa_ai_reviewed.jsonl"
    p = pair(0)
    append_ai_reviews([record(p, "accept")], path)
    append_ai_reviews([record(p, "edit", answer="fixed"), record(pair(1), "reject")], path)
    reviews = read_ai_reviews(path)
    assert len(path.read_text().splitlines()) == 3
    assert reviews["q000"].review.decision == "edit"  # the last line wins
    assert read_ai_reviews(tmp_path / "missing.jsonl") == {}


def test_new_candidates_do_not_disturb_the_review() -> None:
    old = [pair(i) for i in range(3)]
    reviews = {"q001": record(old[1], "reject"), "q002": record(old[2], "edit", answer="a2")}
    grown = old + [pair(i) for i in range(3, 10)]  # the daily generator adds pairs
    out, rejected = apply_ai_reviews(sorted(grown, key=lambda p: p.qa_id), reviews)
    assert rejected == ["q001"]
    assert len(out) == 9
    assert {p.qa_id for p in out if p.ai_review is not None} == {"q002"}


def test_ai_review_is_not_serialised_into_qa_files(tmp_path: Path) -> None:
    p = pair(0).model_copy(
        update={"ai_review": AIReview(decision="accept", notes="n", reviewed_at="2026-11-03")}
    )
    path = tmp_path / "qa.jsonl"
    write_qa_jsonl([p], path)
    assert "ai_review" not in path.read_text()
    assert read_qa_jsonl(path)[0].ai_review is None


def test_selection_is_seeded_stratified_and_skips_unusable_pairs() -> None:
    pairs = [pair(i) for i in range(80)]
    pairs.append(pair(80, auto_check=False))
    pairs.append(pair(81, review_status="accepted", verified=True))
    pairs += [pair(90 + i, qtype="unanswerable", answerable=False, answer="") for i in range(12)]
    a = select_for_review(pairs, 20, 10, seed=7)
    b = select_for_review(list(reversed(pairs)), 20, 10, seed=7)
    assert [p.qa_id for p in a] == [p.qa_id for p in b]  # independent of file order
    assert select_for_review(pairs, 20, 10, seed=8) != a
    answerable = [p for p in a if p.answerable]
    assert len(answerable) == 20
    assert sum(not p.answerable for p in a) == 10
    assert {"q080", "q081"}.isdisjoint(p.qa_id for p in a)
    assert Counter(p.qtype for p in answerable) == dict.fromkeys(TYPES, 5)
    assert len({p.department for p in answerable}) == 5
    assert len({p.year for p in answerable}) == 7


@pytest.mark.parametrize(
    ("source", "kind"),
    [("data/eval/qa_candidates.jsonl", "model-generated"), ("data/eval/qa_v1.jsonl", "reviewed")],
)
def test_labels_say_ai_reviewed_never_human_verified(source: str, kind: str) -> None:
    pairs = [pair(i) for i in range(5)]
    out, rejected = apply_ai_reviews(
        pairs, {"q000": record(pairs[0], "accept"), "q001": record(pairs[1], "reject")}
    )
    text = describe_test_set(source, out, len(rejected))
    assert text == (
        f"{kind} test set (`{source}`): 4 model-generated pairs "
        "(1 AI-reviewed by Claude, 0 human-verified); 1 AI-rejected pairs excluded"
    )
    meta = run_metadata(out, test_set=source, ai_rejected=len(rejected))
    assert meta["human_verified"] == 0
    assert meta["unverified_model_generated"] == 4
    assert meta["ai_review"]["reviewed_in_test_set"] == 1
    assert meta["ai_review"]["rejected_excluded"] == 1
    assert "not human verification" in meta["ai_review"]["reviewer"]


def test_committed_review_file_matches_the_candidates() -> None:
    """The committed AI review is consistent with the committed candidates."""
    root = Path(__file__).resolve().parents[1] / "data" / "eval"
    reviews = read_ai_reviews(root / "qa_ai_reviewed.jsonl")
    if not reviews:
        pytest.skip("no AI review committed")
    candidates = {p.qa_id: p for p in read_qa_jsonl(root / "qa_candidates.jsonl")}
    for rec in reviews.values():
        cand = candidates[rec.qa_id]
        assert (cand.question, cand.answer) == (rec.original_question, rec.original_answer)
        assert rec.review.reviewer == AI_REVIEWER
        assert (rec.review.decision == "edit") == (
            (rec.question, rec.answer) != (rec.original_question, rec.original_answer)
        )
        assert rec.review.notes
    assert len({r.selection_seed for r in reviews.values()}) == 1
