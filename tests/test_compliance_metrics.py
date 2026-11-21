from __future__ import annotations

import math

import pytest

from rbirag.compliance.evaluation.dataset import EvalItem, Label
from rbirag.compliance.evaluation.metrics import (
    classify_failure,
    confusion_matrix,
    entity_of,
    expected_calibration_error,
    macro_f1,
    paired_bootstrap_difference,
    per_class,
    retrieval_recall_at_k,
    risk_coverage,
    score_item,
    score_predictions,
    segmentation_score,
    summarize,
)
from rbirag.compliance.evaluation.report import render_report
from rbirag.compliance.records import ClausePrediction, ToolCall

TITLES = {
    "rbi-1": "Reserve Bank of India (Commercial Banks - Know Your Customer) Directions, 2025",
    "rbi-2": "Reserve Bank of India (Commercial Banks - Know Your Customer) Amendment Directions",
    "rbi-9": "Reserve Bank of India (Urban Co-operative Banks - Know Your Customer) Directions",
    "rbi-5": "Master Direction - Know Your Customer (KYC) Direction, 2016",
}


def item(
    number: str,
    verdict: str,
    citations: list[list[str]] | None = None,
    *,
    policy: str = "kyc",
    tags: list[str] | None = None,
    tools: list[str] | None = None,
    acceptable: list[str] | None = None,
) -> EvalItem:
    label = Label(
        item_id=f"{policy}/{number}",
        policy_id=policy,
        section=number.split(".", maxsplit=1)[0],
        number=number,
        text="The Bank shall do something specific here.",
        verdict=verdict,  # type: ignore[arg-type]
        citations=citations or [],
        acceptable=acceptable or [],
        rationale="because the rule says so",
        tags=tags or [],  # type: ignore[arg-type]
        expected_tools=tools or [],  # type: ignore[arg-type]
    )
    return EvalItem(label=label, split="test", policy_title="P", section_title="S")


def search(*ids: str) -> ToolCall:
    return ToolCall(
        name="search_regulations", arguments={"query": "q"}, valid=True, result_doc_ids=list(ids)
    )


def fetch(doc_id: str, valid: bool = True) -> ToolCall:
    return ToolCall(
        name="fetch_notification",
        arguments={"doc_id": doc_id},
        valid=valid,
        result_doc_ids=[doc_id] if valid else [],
        error=None if valid else "unknown notification",
    )


COMPARE = ToolCall(name="compare_clause", arguments={}, valid=True)


def pred(
    item_id: str,
    verdict: str,
    citations: list[str] | None = None,
    calls: list[ToolCall] | None = None,
    **kw: object,
) -> ClausePrediction:
    return ClausePrediction(
        item_id=item_id,
        verdict=verdict,  # type: ignore[arg-type]
        citations=citations or [],
        tool_calls=calls or [],
        **kw,  # type: ignore[arg-type]
    )


def test_entity_of() -> None:
    assert entity_of(TITLES["rbi-9"]) == "Urban Co-operative Banks"
    assert entity_of(TITLES["rbi-1"]) == "Commercial Banks"
    assert entity_of(TITLES["rbi-5"]) is None and entity_of(None) is None


def test_per_class_macro_f1_and_confusion() -> None:
    golds = ["compliant", "compliant", "conflict", "gap"]
    preds = ["compliant", "conflict", "conflict", "compliant"]
    pc = per_class(golds, preds)
    assert pc["compliant"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    )
    assert pc["conflict"]["f1"] == pytest.approx(2 / 3)
    assert pc["gap"]["f1"] == 0.0
    # classes absent from gold (not_applicable, insufficient_evidence) are skipped
    assert macro_f1(golds, preds) == pytest.approx((0.5 + 2 / 3 + 0.0) / 3)
    cm = confusion_matrix(golds, preds)
    assert cm["compliant"]["conflict"] == 1 and cm["gap"]["compliant"] == 1
    assert sum(sum(r.values()) for r in cm.values()) == 4


def test_full_success_with_grounded_trajectory() -> None:
    it = item(
        "3.2", "conflict", [["rbi-1", "rbi-2"]], tools=["search_regulations", "compare_clause"]
    )
    p = pred(
        "kyc/3.2",
        "conflict",
        ["rbi-2"],
        [search("rbi-1", "rbi-2"), fetch("rbi-2"), COMPARE],
        confidence=0.9,
        input_tokens=100,
        output_tokens=20,
        latency_ms=1500.0,
        llm_calls=3,
    )
    s = score_item(it, p, titles=TITLES)
    assert s.correct and s.success and s.failure is None
    assert (s.cited, s.cited_allowed, s.groups, s.groups_hit) == (1, 1, 1, 1)
    assert (s.calls, s.valid_calls, s.fetches, s.grounded_fetches) == (3, 3, 1, 1)
    assert s.tool_selection_ok and s.steps == 3


def test_citation_precision_and_recall_with_groups_and_acceptable() -> None:
    it = item("1.1", "gap", [["rbi-1"], ["rbi-7", "rbi-8"]], acceptable=["rbi-2"])
    p = pred("kyc/1.1", "gap", ["rbi-1", "rbi-2", "rbi-9", "rbi-1"], [search("rbi-1"), COMPARE])
    s = score_item(it, p, titles=TITLES)
    assert s.cited == 3 and s.cited_allowed == 2  # duplicates removed; rbi-9 not allowed
    assert s.groups_hit == 1 and s.citation_recall == 0.5
    assert s.correct and not s.success and s.failure == "citation_miss"
    assert s.wrong_entity_citations == 1


def test_tool_selection_rules() -> None:
    it = item("2.1", "compliant", [["rbi-1"]], tools=["search_regulations", "fetch_notification"])
    # substantive verdict without compare_clause, and no fetch
    s = score_item(it, pred("kyc/2.1", "compliant", ["rbi-1"], [search("rbi-1")]))
    assert not s.tool_selection_ok
    # ungrounded fetch (id never shown by a search) is legal but not grounded
    s = score_item(
        it, pred("kyc/2.1", "compliant", ["rbi-1"], [search("rbi-2"), fetch("rbi-1"), COMPARE])
    )
    assert s.tool_selection_ok and s.fetches == 1 and s.grounded_fetches == 0
    # step budget
    many = [search("rbi-1")] * 9 + [fetch("rbi-1"), COMPARE]
    s = score_item(it, pred("kyc/2.1", "compliant", ["rbi-1"], many), max_steps=8)
    assert not s.tool_selection_ok and s.failure == "step_budget"
    # abstention needs no compare_clause
    ie = item("2.2", "insufficient_evidence", tags=["not_in_corpus"], tools=["search_regulations"])
    s = score_item(ie, pred("kyc/2.2", "insufficient_evidence", [], [search("rbi-1")]))
    assert s.tool_selection_ok and s.success


def test_invalid_tool_call_blocks_success() -> None:
    it = item("2.3", "compliant", [["rbi-1"]])
    p = pred(
        "kyc/2.3", "compliant", ["rbi-1"], [search("rbi-1"), fetch("rbi-404", valid=False), COMPARE]
    )
    s = score_item(it, p)
    assert s.correct and not s.success and s.failure == "invalid_tool_call"
    assert s.valid_calls == 2 and s.calls == 3


@pytest.mark.parametrize(
    ("gold", "predicted", "tags", "cited", "expected"),
    [
        ("insufficient_evidence", "compliant", [], [], "unsupported_verdict"),
        ("conflict", "insufficient_evidence", [], [], "over_abstention"),
        ("not_applicable", "compliant", [], [], "scope_error"),
        ("compliant", "not_applicable", [], [], "scope_error"),
        ("conflict", "compliant", [], ["rbi-9"], "wrong_entity"),
        ("conflict", "compliant", ["outdated_rule"], ["rbi-5"], "outdated_rule_missed"),
        ("gap", "compliant", [], ["rbi-1"], "missed_issue"),
        ("compliant", "conflict", [], ["rbi-1"], "false_alarm"),
        ("gap", "conflict", [], ["rbi-1"], "conflict_gap_confusion"),
    ],
)
def test_failure_taxonomy(
    gold: str, predicted: str, tags: list[str], cited: list[str], expected: str
) -> None:
    groups = [["rbi-1"]] if gold in {"compliant", "conflict", "gap"} else []
    if gold == "not_applicable":
        tags = [*tags, "out_of_scope"]
    if gold == "insufficient_evidence":
        tags = [*tags, "not_in_corpus"]
    it = item("4.1", gold, groups, tags=tags)
    s = score_item(it, pred("kyc/4.1", predicted, cited, [search(*cited), COMPARE]), titles=TITLES)
    assert s.failure == expected
    assert classify_failure(s) == expected


def test_run_error_and_missing_prediction() -> None:
    it = item("5.1", "compliant", [["rbi-1"]])
    s = score_item(it, pred("kyc/5.1", "compliant", ["rbi-1"], error="timeout"))
    assert not s.correct and s.failure == "run_error"
    scores = score_predictions([it], {})
    assert scores[0].error and scores[0].failure == "run_error"


def test_ece_and_risk_coverage() -> None:
    assert expected_calibration_error([0.9, 0.9], [True, True]) == pytest.approx(0.1)
    assert expected_calibration_error([0.95, 0.15], [True, False]) == pytest.approx(0.1)
    assert math.isnan(expected_calibration_error([], []))
    curve, aurc = risk_coverage([0.9, 0.2, 0.8, 0.1], [True, False, True, True])
    assert curve[0] == (0.25, 1.0) and curve[1] == (0.5, 1.0)
    assert curve[2] == pytest.approx((0.75, 2 / 3)) and curve[3] == (1.0, 0.75)
    assert aurc == pytest.approx((0 + 0 + 1 / 3 + 0.25) / 4)


def test_segmentation_and_retrieval_recall() -> None:
    seg = segmentation_score(["1.1 A  b", "1.2 C", "1.3 D"], ["1.1 A b", "1.2 C\nextra", "1.3 D"])
    assert (seg.matched, seg.precision, seg.recall) == (2, 2 / 3, 2 / 3)
    assert segmentation_score(["4.2 low-risk customers"], ["4.2 low-\nrisk customers"]).f1 == 1.0
    assert retrieval_recall_at_k(["a", "b", "c"], [["c"], ["x", "b"]], k=2) == 0.5
    assert retrieval_recall_at_k(["a", "b", "c"], [["c"], ["x", "b"]], k=3) == 1.0
    assert math.isnan(retrieval_recall_at_k(["a"], [], k=1))


def test_paired_difference() -> None:
    est = paired_bootstrap_difference([1, 1, 1, 0], [0, 1, 0, 0])
    assert est.mean == pytest.approx(0.5) and est.low <= 0.5 <= est.high
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_difference([1], [1, 0])


def test_summarize_end_to_end() -> None:
    items = [
        item("1.1", "compliant", [["rbi-1"]], tags=["paraphrase"]),
        item("1.2", "conflict", [["rbi-1"]], tags=["numeric_threshold"]),
        item("2.1", "not_applicable", tags=["out_of_scope"], policy="hr"),
        item("2.2", "insufficient_evidence", tags=["not_in_corpus"], policy="hr"),
    ]
    preds = {
        "kyc/1.1": pred(
            "kyc/1.1",
            "compliant",
            ["rbi-1"],
            [search("rbi-1"), COMPARE],
            confidence=0.9,
            latency_ms=1000.0,
            cost_usd=0.001,
            input_tokens=500,
        ),
        "kyc/1.2": pred(
            "kyc/1.2",
            "compliant",
            ["rbi-1", "rbi-9"],
            [search("rbi-1"), COMPARE],
            confidence=0.7,
            latency_ms=3000.0,
            cost_usd=0.002,
            input_tokens=700,
        ),
        "hr/2.1": pred("hr/2.1", "not_applicable", [], [], confidence=0.8, latency_ms=500.0),
        "hr/2.2": pred(
            "hr/2.2", "insufficient_evidence", [], [search()], confidence=0.6, latency_ms=800.0
        ),
    }
    scores = score_predictions(items, preds, titles=TITLES)
    report = summarize(scores, system="agent", n_resamples=200)
    assert report.n == 4 and report.documents == 2
    assert report.verdict_accuracy.mean == pytest.approx(0.75)
    assert report.citation_precision.mean == pytest.approx(2 / 3)
    assert report.citation_recall.mean == pytest.approx(1.0)
    assert report.task_success.mean == pytest.approx(0.75)
    assert report.cost_usd_per_document.mean == pytest.approx((0.003 + 0.0) / 2)
    assert report.latency_ms_per_document.mean == pytest.approx((4000 + 1300) / 2)
    assert report.failures == {"wrong_entity": 1}
    assert report.document_success == 1.0
    assert report.ece is not None and report.aurc is not None
    assert report.accuracy_at_coverage["50%"] == 1.0
    assert report.by_tag["numeric_threshold"]["accuracy"] == 0.0
    assert report.verdict_accuracy.low <= 0.75 <= report.verdict_accuracy.high
    md = render_report([report])
    assert "| verdict accuracy |" in md and "wrong_entity" in md and "| gold \\ predicted |" in md


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no scored items"):
        summarize([])


def test_threshold_sweep_and_choice() -> None:
    from rbirag.compliance.evaluation.threshold import apply_threshold, choose, sweep

    items = [
        item("1.1", "compliant", [["rbi-1"]]),
        item("1.2", "insufficient_evidence", tags=["not_in_corpus"]),
        item("1.3", "conflict", [["rbi-1"]]),
    ]
    preds = {
        "kyc/1.1": pred("kyc/1.1", "compliant", ["rbi-1"], confidence=0.9),
        "kyc/1.2": pred("kyc/1.2", "conflict", ["rbi-1"], confidence=0.3),  # wrong, low confidence
        "kyc/1.3": pred("kyc/1.3", "conflict", ["rbi-1"], confidence=0.6),
    }
    assert apply_threshold(preds["kyc/1.2"], 0.5) == "insufficient_evidence"
    assert apply_threshold(preds["kyc/1.1"], 0.5) == "compliant"
    rows = sweep(items, preds, grid=(0.0, 0.5, 0.7))
    assert [r.accuracy for r in rows] == pytest.approx([2 / 3, 1.0, 2 / 3])
    assert [r.abstained for r in rows] == [0, 1, 2]
    assert choose(rows).threshold == 0.5
