"""Human review sampling, the labelling loop and agreement statistics."""

from __future__ import annotations

from pathlib import Path

from rbirag.eval.human_review import (
    AnswerReview,
    agreement,
    cohen_kappa,
    read_reviews,
    review_loop,
    sample_for_review,
)

ITEMS = [
    {"qa_id": f"q{i}", "config": "c", "backend": "b", "judge": 1.0 if i % 2 else 0.5,
     "verdict": "supported" if i % 3 else "unsupported"}
    for i in range(12)
]  # fmt: skip


def test_sample_is_stratified_and_seeded() -> None:
    a = sample_for_review(ITEMS, 4, seed=1)
    assert a == sample_for_review(ITEMS, 4, seed=1) and len(a) == 4
    strata = {((x["judge"] or 0) >= 0.999, x["verdict"]) for x in a}
    assert len(strata) == 4  # one from each (judge, verifier) stratum first


def test_review_loop_is_resumable(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    answers = iter(["f", "y", "ok", "u", "n", "", "q"])
    added = review_loop(ITEMS[:5], lambda it: it["qa_id"], path, reviewer="alice",
                        ask=lambda _: next(answers), echo=lambda _: None)  # fmt: skip
    assert added == 2
    reviews = read_reviews(path)
    assert reviews[("q0", "c", "b")].faithful == "faithful" and reviews[("q0", "c", "b")].correct
    assert reviews[("q1", "c", "b")].faithful == "unfaithful"
    again = iter(["s", "q"])
    assert review_loop(ITEMS[:5], str, path, reviewer="alice", ask=lambda _: next(again),
                       echo=lambda _: None) == 0  # fmt: skip


def test_agreement_needs_human_labels() -> None:
    checks = {("q1", "c", "b"): ITEMS[1], ("q2", "c", "b"): ITEMS[2]}
    ai_only = {
        k: AnswerReview(qa_id=k[0], config="c", backend="b", reviewer="ai-assistant (LLM)",
                        faithful="faithful", reviewed_at="t")
        for k in checks
    }  # fmt: skip
    rep = agreement(ai_only, checks)
    assert rep["human_reviews"] == 0 and "not ground truth" in rep["note"]
    human = {k: r.model_copy(update={"reviewer": "alice"}) for k, r in ai_only.items()}
    rep = agreement(human, checks)
    judge = rep["llm_judge (faithfulness = 1)"]
    assert judge["n"] == 2 and judge["agreement"] == 0.5
    assert cohen_kappa([True, False], [True, False]) == 1.0
    assert cohen_kappa([], []) is None
