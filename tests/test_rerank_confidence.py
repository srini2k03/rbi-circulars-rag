from __future__ import annotations

import math

import pytest

from rbirag.retrieval.confidence import compute_confidence, sigmoid
from rbirag.retrieval.rerank import CrossEncoderReranker, FakeReranker, build_reranker


def test_fake_reranker_scores_coverage() -> None:
    r = FakeReranker()
    scores = r.score("kyc high risk", ["kyc for high risk customers", "kyc only", "nothing"])
    assert scores[0] == pytest.approx(5.0)
    assert scores[1] == pytest.approx(10 / 3 - 5)
    assert scores[2] == pytest.approx(-5.0)
    assert r.score("the", ["x"]) == [-5.0]
    assert r.score("x", []) == []


def test_build_reranker_lazy() -> None:
    assert build_reranker("none") is None
    assert isinstance(build_reranker("fake"), FakeReranker)
    ce = build_reranker("cross-encoder")
    assert isinstance(ce, CrossEncoderReranker) and ce._model is None
    assert ce.score("q", []) == []
    with pytest.raises(ValueError, match="unknown reranker"):
        build_reranker("x")


def test_sigmoid_stable() -> None:
    assert sigmoid(0) == 0.5
    assert sigmoid(-1000) == pytest.approx(0.0)
    assert sigmoid(1000) == pytest.approx(1.0)
    assert sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2)))


def test_confidence_from_scores() -> None:
    c = compute_confidence([1.0, 3.0, -2.0], threshold=0.9)
    assert c.top_score == 3.0
    assert c.margin == pytest.approx(2.0)
    assert c.probability == pytest.approx(sigmoid(3.0))  # 0.9526
    assert c.is_confident
    assert not compute_confidence([1.0, 3.0], threshold=0.97).is_confident


def test_confidence_temperature_and_margin() -> None:
    assert compute_confidence([2.0], temperature=2.0).probability == pytest.approx(sigmoid(1.0))
    low_margin = compute_confidence([3.0, 2.9], threshold=0.5, min_margin=0.5)
    assert not low_margin.is_confident
    assert compute_confidence([3.0], min_margin=0.5).is_confident  # single result: no margin


def test_confidence_empty_and_negative() -> None:
    empty = compute_confidence([])
    assert empty.probability == 0.0 and not empty.is_confident and empty.top_score is None
    assert not compute_confidence([-4.0, -6.0]).is_confident


def test_confidence_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        compute_confidence([1.0], threshold=1.5)
    with pytest.raises(ValueError, match="temperature"):
        compute_confidence([1.0], temperature=0)
