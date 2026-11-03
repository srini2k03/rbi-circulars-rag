from __future__ import annotations

import pytest

from rbirag.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_hand_computed_k60() -> None:
    bm25 = ["a", "b", "c"]
    dense = ["b", "d", "a"]
    fused = dict(reciprocal_rank_fusion([bm25, dense], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)  # 0.032266...
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)  # 0.032522...
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)
    order = [d for d, _ in reciprocal_rank_fusion([bm25, dense], k=60)]
    assert order == ["b", "a", "d", "c"]


def test_rrf_small_k_example() -> None:
    # k=1: a = 1/2 + 1/4 = 0.75 ; b = 1/3 + 1/2 = 0.8333 ; c = 1/4 ; d = 1/3
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=1)
    assert [d for d, _ in fused] == ["b", "a", "d", "c"]
    assert [round(s, 4) for _, s in fused] == [0.8333, 0.75, 0.3333, 0.25]


def test_rrf_weights() -> None:
    # dense weighted 3x: a = 1/61 + 3/63 ; b = 1/62 + 3/61
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60, weights=[1.0, 3.0]))
    assert fused["b"] == pytest.approx(1 / 62 + 3 / 61)
    assert fused["b"] > fused["a"]
    # zero weight on a ranking removes its influence
    only_first = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[1.0, 0.0])
    assert [d for d, _ in only_first] == ["a", "b"]


def test_rrf_ties_deterministic_and_limit() -> None:
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert [d for d, _ in fused] == ["x", "y"]  # same best rank (1) -> by id
    assert len(reciprocal_rank_fusion([["a", "b", "c"]], limit=2)) == 2


def test_rrf_duplicates_counted_once_and_empty() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_validation() -> None:
    with pytest.raises(ValueError, match="one weight"):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a"]], k=-1)
