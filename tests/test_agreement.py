"""Judge agreement metrics."""

from __future__ import annotations

import pytest

from rbirag.eval.agreement import (
    agreement_markdown,
    agreement_report,
    cohen_kappa,
    judge_agreement,
    pearson,
    spearman,
)
from rbirag.eval.gen_metrics import GenerationRecord


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    # textbook example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    # systematic disagreement is worse than chance
    assert cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)
    # quadratic weights: an off-by-one disagreement costs less than off-by-four
    near = cohen_kappa([1, 2, 3, 4, 5], [2, 2, 3, 4, 5], weights="quadratic")
    far = cohen_kappa([1, 2, 3, 4, 5], [5, 2, 3, 4, 5], weights="quadratic")
    assert near is not None and far is not None and near > far
    assert cohen_kappa([], []) is None
    assert cohen_kappa([1, 1], [1, 1]) is None  # chance agreement is total
    with pytest.raises(ValueError, match="same items"):
        cohen_kappa([1], [1, 2])


def test_correlations() -> None:
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)  # monotone
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 2], [1, 2]) is None  # too few
    assert pearson([1, 1, 1], [1, 2, 3]) is None  # constant


def rec(i: int, verdicts: dict[str, dict[str, float | None]]) -> GenerationRecord:
    return GenerationRecord(
        qa_id=str(i), config="c", backend="b", model="m", answerable=True, question="q?",
        answer="a", refused=False, refusal_reason=None, citation_valid=True, n_citations=1,
        n_invalid_citations=0, n_uncited_sentences=0, verdicts=verdicts,
    )  # fmt: skip


def v(f: float | None, rel: int | None) -> dict[str, float | None]:
    return {"faithfulness": f, "relevance": None if rel is None else (rel - 1) / 4}


def test_judge_agreement_on_overlap_only() -> None:
    records = [
        rec(1, {"small": v(1.0, 5), "big": v(1.0, 5)}),
        rec(2, {"small": v(0.5, 3), "big": v(0.5, 4)}),
        rec(3, {"small": v(0.0, 1), "big": v(0.25, 1)}),
        rec(4, {"small": v(1.0, 4), "big": v(1.0, 4)}),
        rec(5, {"small": v(1.0, 5)}),  # only one judge: not counted
        rec(6, {"small": v(None, None), "big": v(1.0, 5)}),  # failed attempt: not counted
        rec(7, {"small": v(None, 2), "other": v(1.0, 2)}),
    ]
    a = judge_agreement(records, "small", "big")
    assert a["n"] == 4
    f, r = a["faithfulness"], a["relevance"]
    assert f["n"] == 4
    assert f["fully_supported_agreement"] == 1.0
    assert f["fully_supported_kappa"] == 1.0
    assert f["mean_abs_diff"] == pytest.approx(0.0625, abs=1e-3)
    assert r["exact_agreement"] == 0.75
    assert r["within_one_agreement"] == 1.0
    assert r["kappa_quadratic"] is not None and r["kappa_quadratic"] > r["kappa"]
    report = agreement_report(records, "small")
    assert [x["other"] for x in report] == ["big", "other"]
    assert report[1]["faithfulness"]["n"] == 0
    md = agreement_markdown(report)
    assert md.count("\n") == 4 and "| big | 4 |" in md
