from __future__ import annotations

import itertools

from rbirag.compliance.sections import select_passages, split_passages, table_of_contents
from rbirag.generation.llm import approx_tokens
from tests.conftest import make_doc

TEXT = (
    "RBI/2025-26/1\nApril 1, 2025\n\nDear Sir,\n\nMaster Direction on Things\n\n"
    "1. Introduction\n\nThese directions apply to all commercial banks in India.\n\n"
    "2. Periodic updation of KYC\n\nPeriodic updation shall be carried out at least once in "
    "every two years for high risk customers and once in every ten years for low risk.\n\n"
    "3. Record keeping\n\nRecords shall be preserved for five years after the relationship "
    "ends.\n\n"
    "4. Reporting\n\n" + "Reports shall be filed with FIU-IND electronically. " * 60
)
DOC = make_doc(TEXT, doc_id="rbi-1", title="Master Direction on Things")


def test_passages_cover_the_text_in_order() -> None:
    passages = split_passages(DOC, max_chars=500)
    assert [p.index for p in passages] == list(range(len(passages)))
    assert all(TEXT[p.start : p.end] == p.text for p in passages)
    assert all(len(p.text) <= 500 for p in passages)
    assert all(a.end <= b.start for a, b in itertools.pairwise(passages))
    # the long "4. Reporting" section was split into several passages
    assert sum(1 for p in passages if p.number == "4") > 1


def test_query_selects_the_relevant_section_within_budget() -> None:
    passages, truncated = select_passages(
        DOC, "periodic updation high risk customers", max_tokens=80, max_chars=500
    )
    assert truncated
    assert passages[0].number == "2"
    assert "two years for high risk" in passages[0].text
    assert passages[0].score is not None and passages[0].score > 0
    assert sum(approx_tokens(p.text) for p in passages) <= 80


def test_selected_passages_come_back_in_document_order() -> None:
    passages, _ = select_passages(
        DOC, "records preserved introduction commercial banks", max_tokens=400, max_chars=500
    )
    starts = [p.start for p in passages]
    assert starts == sorted(starts) and len(passages) >= 2


def test_no_query_returns_leading_passages_and_tiny_budget_trims() -> None:
    passages, truncated = select_passages(DOC, None, max_tokens=10_000)
    assert not truncated and passages[0].start == 0
    one, truncated = select_passages(DOC, "FIU-IND electronically", max_tokens=20, max_chars=5000)
    assert truncated and len(one) == 1 and len(one[0].text) <= 80
    assert one[0].end - one[0].start == len(one[0].text)


def test_unmatched_query_falls_back_to_document_order() -> None:
    passages, _ = select_passages(DOC, "zzzz qqqq", max_tokens=60, max_chars=500)
    assert passages and passages[0].start == 0 and passages[0].score is None


def test_table_of_contents_lists_top_level_headings() -> None:
    toc = table_of_contents(DOC)
    assert "1. Introduction" in toc and "4. Reporting" in toc


def test_empty_document() -> None:
    empty = make_doc("", doc_id="e")
    assert split_passages(empty) == []
    assert select_passages(empty, "x", max_tokens=10) == ([], False)
