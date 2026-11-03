from __future__ import annotations

import math

import numpy as np
import pytest

from rbirag.chunking.base import Chunk
from rbirag.eval.calibrate import calibrate_threshold, evaluate_threshold
from rbirag.eval.dataset import GoldSpan, QAPair, split, split_of, usable
from rbirag.eval.gen_metrics import GenerationRecord, refusal_counts, summarize
from rbirag.eval.retrieval_metrics import (
    covers,
    first_hits,
    mrr_at_k,
    ndcg_at_k,
    query_metrics,
    recall_at_k,
)
from rbirag.eval.stats import bootstrap_ci, bootstrap_ratio_ci


def chunk(doc: str, start: int, end: int, i: int = 0) -> Chunk:
    return Chunk(chunk_id=f"{doc}::{i}", doc_id=doc, index=i, text="x" * (end - start), start=start,
                 end=end, n_tokens=1, chunker="t")  # fmt: skip


def test_rank_metrics_hand_computed() -> None:
    ranks = [2, None]  # two gold items, one found at rank 2
    assert recall_at_k(ranks, 1) == 0.0
    assert recall_at_k(ranks, 2) == 0.5
    assert mrr_at_k(ranks, 10) == 0.5
    assert mrr_at_k(ranks, 1) == 0.0
    # DCG = 1/log2(3); IDCG (2 gold) = 1 + 1/log2(3)
    assert ndcg_at_k(ranks, 10) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([1], 10) == 1.0
    assert ndcg_at_k([3], 10) == pytest.approx(1 / math.log2(4))
    assert recall_at_k([], 5) == 0.0


def test_span_cover_rules() -> None:
    gold = GoldSpan(doc_id="d", start=100, end=200)
    assert covers(chunk("d", 50, 160), gold)  # 60% of the gold span
    assert not covers(chunk("d", 50, 130), gold)  # 30%
    assert covers(chunk("d", 120, 160), gold)  # chunk fully inside the span
    assert not covers(chunk("e", 100, 200), gold)


def test_query_metrics_doc_vs_chunk() -> None:
    qa = QAPair(qa_id="q", question="?", answer="a", qtype="factual",
                gold=[GoldSpan(doc_id="d", start=100, end=200)])  # fmt: skip
    retrieved = [chunk("d", 0, 90, 0), chunk("x", 0, 50, 1), chunk("d", 90, 210, 2)]
    doc = query_metrics(retrieved, qa, level="doc")
    ch = query_metrics(retrieved, qa, level="chunk")
    assert doc.mrr == 1.0  # right document at rank 1
    assert ch.mrr == pytest.approx(1 / 3)  # right passage at rank 3
    assert ch.recall[1] == 0.0
    assert ch.recall[3] == 1.0
    assert first_hits(retrieved, qa, "chunk") == [3]


def test_bootstrap_ci() -> None:
    est = bootstrap_ci([1.0] * 20)
    assert (est.mean, est.low, est.high) == (1.0, 1.0, 1.0)
    vals = list(np.linspace(0, 1, 50))
    a = bootstrap_ci(vals, seed=1)
    b = bootstrap_ci(vals, seed=1)
    assert a == b  # deterministic
    assert a.low < a.mean < a.high
    assert a.mean == pytest.approx(0.5)
    assert math.isnan(bootstrap_ci([]).mean)
    r = bootstrap_ratio_ci([1, 1, 0, 0], [1, 1, 1, 1])
    assert r.mean == 0.5
    assert 0.0 <= r.low <= 0.5 <= r.high <= 1.0


def test_calibration_hand_case() -> None:
    probs = [0.9, 0.8, 0.3, 0.1, 0.6]
    answerable = [True, True, False, False, True]
    p = evaluate_threshold(probs, answerable, 0.5)
    assert (p.refusal_precision, p.refusal_recall, p.answerable_coverage) == (1.0, 1.0, 1.0)
    p = evaluate_threshold(probs, answerable, 0.85)  # refuses 0.8, 0.3, 0.1, 0.6
    assert p.refusal_precision == 0.5
    assert p.answerable_coverage == pytest.approx(1 / 3)
    cal = calibrate_threshold(probs, answerable, grid=[0.0, 0.2, 0.5, 0.7, 0.85])
    assert cal.threshold in {0.5}
    assert cal.chosen.refusal_f == 1.0
    assert cal.n_unanswerable == 2
    with pytest.raises(ValueError, match="no dev"):
        calibrate_threshold([], [])


def rec(
    qa_id: str, answerable: bool, refused: bool, faith: float | None = None
) -> GenerationRecord:
    return GenerationRecord(
        qa_id=qa_id, config="c", backend="b", model="m", answerable=answerable, question="q",
        answer="a", refused=refused, refusal_reason=None, citation_valid=True, n_citations=1,
        n_invalid_citations=0, n_uncited_sentences=0, judge_faithfulness=faith,
    )  # fmt: skip


def test_generation_summary_refusal_math() -> None:
    records = [
        rec("1", True, False, 1.0),
        rec("2", True, False, 0.5),
        rec("3", True, True),  # false refusal
        rec("4", False, True),  # correct refusal
        rec("5", False, False),  # answered an unanswerable question
    ]
    assert refusal_counts(records) == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    s = summarize(records, n_resamples=200)
    assert s["refusal_precision"].mean == 0.5
    assert s["refusal_recall"].mean == 0.5
    assert s["judge_faithfulness"].mean == 0.75  # only answered answerable questions
    assert s["answer_rate"].mean == pytest.approx(2 / 3)


def test_splits_and_usable() -> None:
    pairs = [QAPair(qa_id=f"q{i}", question="?", answer="", qtype="factual") for i in range(200)]
    dev, test = split(pairs, 0.3)
    assert len(dev) + len(test) == 200
    assert 40 <= len(dev) <= 80
    assert split_of("q1", 0.3) == split_of("q1", 0.3)
    pairs[0] = pairs[0].model_copy(update={"review_status": "rejected"})
    pairs[1] = pairs[1].model_copy(update={"auto_check": False})
    assert len(usable(pairs)) == 198


def test_ndcg_counts_each_rank_once_and_stays_in_unit_interval() -> None:
    # two gold spans inside the rank-1 chunk: a perfect ranking
    assert ndcg_at_k([1, 1], 10) == 1.0
    # previously this summed gain twice; now bounded
    assert ndcg_at_k([1, 1, 2], 10) == 1.0
    # one span found at rank 1, one never found: ideal needs two results
    assert ndcg_at_k([1, None], 10) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    # two spans in one chunk at rank 3
    assert ndcg_at_k([3, 3], 10) == pytest.approx(1 / math.log2(4))
    # found only beyond k still needs an ideal slot
    assert ndcg_at_k([12, 1], 10) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_chunk_level_ndcg_with_two_spans_in_one_chunk() -> None:
    qa = QAPair(qa_id="q", question="?", answer="a", qtype="multi_chunk",
                gold=[GoldSpan(doc_id="d", start=10, end=40), GoldSpan(doc_id="d", start=60, end=90)])  # fmt: skip
    m = query_metrics([chunk("d", 0, 100, 0), chunk("d", 100, 200, 1)], qa, level="chunk")
    assert m.ndcg == 1.0
    assert m.recall[1] == 1.0
