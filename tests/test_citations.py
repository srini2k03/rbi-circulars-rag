from __future__ import annotations

import pytest

from rbirag.generation.citations import (
    expand_citation,
    extract_citations,
    is_refusal,
    parse_citations,
    split_sentences,
    strip_citations,
)
from rbirag.generation.prompts import REFUSAL_TEXT


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("1", [1]),
        ("1, 3", [1, 3]),
        ("2-4", [2, 3, 4]),
        ("1,3-4", [1, 3, 4]),
        ("1" + chr(0x2013) + "3", [1, 2, 3]),
    ],
)
def test_expand_citation(group: str, expected: list[int]) -> None:
    assert expand_citation(group) == expected


def test_extract_and_strip() -> None:
    text = "Banks must comply [1][3]. The limit is Rs. 5 lakh [2, 4]. See also [1-2]."
    assert extract_citations(text) == [1, 3, 2, 4, 1, 2]
    assert strip_citations(text) == "Banks must comply. The limit is Rs. 5 lakh. See also."


def test_parse_valid_answer() -> None:
    answer = "KYC must be updated every two years for high-risk customers [1]. Low-risk customers every ten years [2]."
    report = parse_citations(answer, n_sources=3)
    assert report.cited_ids == [1, 2]
    assert report.invalid_ids == []
    assert report.uncited_sentences == []
    assert report.valid
    assert [s.citations for s in report.sentences] == [[1], [2]]


def test_parse_flags_invalid_and_uncited() -> None:
    answer = (
        "Banks must report within 30 days [7]. The penalty is a late submission fee as specified."
    )
    report = parse_citations(answer, n_sources=5)
    assert report.invalid_ids == [7]
    assert report.uncited_sentences == ["The penalty is a late submission fee as specified."]
    assert not report.valid


def test_uncited_answer_is_invalid_but_short_headers_are_ignored() -> None:
    report = parse_citations("Summary:\n- Banks must comply with the rule [1].", n_sources=1)
    assert report.valid
    assert not parse_citations("Banks must comply with the new rule.", n_sources=2).valid


def test_refusal_detection() -> None:
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal("I don't know.")
    assert is_refusal("The provided sources do not contain the answer.")
    assert not is_refusal("Banks need not know the customer's employer [1].")
    report = parse_citations(REFUSAL_TEXT, n_sources=3)
    assert report.is_refusal
    assert report.valid  # refusals need no citations


def test_split_sentences_keeps_trailing_citations() -> None:
    assert split_sentences(
        "A rule applies [1]. Another rule applies [2][3].\n- Bullet item [1]."
    ) == [
        "A rule applies [1].",
        "Another rule applies [2][3].",
        "Bullet item [1].",
    ]
