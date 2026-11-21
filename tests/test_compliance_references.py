from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from rbirag.compliance.references import (
    ReferenceGraph,
    classify_relation,
    normalise_circular_number,
    normalise_ref_number,
    normalise_title,
    sentence_around,
)
from tests.conftest import make_doc

MD = make_doc(
    "RBI/2015-16/108\nDBR.AML.BC.No.81/14.01.001/2015-16\nFebruary 25, 2016\n\n"
    "Master Direction - Know Your Customer (KYC) Direction, 2016\n\n"
    "1. Periodic updation shall be carried out once in every two years for high risk.",
    doc_id="rbi-100",
    title="Master Direction - Know Your Customer (KYC) Direction, 2016 (Updated as on May 4, 2023)",
    date=dt.date(2016, 2, 25),
    circular_number="RBI/2015-16/108",
    ref_number="DBR.AML.BC.No.81/14.01.001/2015-16",
)
AMENDMENT = make_doc(
    "RBI/2024-2025/87\nDOR.AML.REC.49/14.01.001/2024-25\nNovember 06, 2024\n\nDear Sir,\n\n"
    "Please refer to the Master Direction - Know Your Customer (KYC) Direction, 2016 dated "
    "February 25, 2016. 2. On a review, paragraph 38 of the Master Direction has been amended "
    "as given in the Annex.",
    doc_id="rbi-200",
    title="Amendment to the Master Direction - Know Your Customer (KYC) Direction, 2016",
    date=dt.date(2024, 11, 6),
    circular_number="RBI/2024-25/87",
    ref_number="DOR.AML.REC.49/14.01.001/2024-25",
)
OLD_CIRCULAR = make_doc(
    "RBI/2019-20/70\nDOR.CRE.REC.12/21.04.048/2019-20\nSeptember 26, 2019\n\n"
    "1. Banks shall report frauds within 21 days.",
    doc_id="rbi-150",
    title="Reporting of frauds",
    date=dt.date(2019, 9, 26),
    circular_number="RBI/2019-20/70",
    ref_number="DOR.CRE.REC.12/21.04.048/2019-20",
)
DIRECTIONS = make_doc(
    "RBI/2025-26/200\nDOR.FRM.REC.1/23.04.001/2025-26\nNovember 28, 2025\n\n"
    "1. Frauds shall be reported within 14 days of classification.\n\n"
    "9. Repeal Provisions\n\n"
    "The following circulars stand repealed from the date of these Directions:\n"
    "Sl. No. Circular Date Subject\n"
    "1 DOR.CRE.REC.12/21.04.048/2019-20 September 26, 2019 Reporting of frauds\n\n"
    "10. Other instructions\n\n"
    "Banks shall also follow RBI/2024-25/87 as amended from time to time.",
    doc_id="rbi-300",
    title="Reserve Bank of India (Commercial Banks - Fraud Risk Management) Directions, 2025",
    date=dt.date(2025, 11, 28),
    circular_number="RBI/2025-26/200",
    ref_number="DOR.FRM.REC.1/23.04.001/2025-26",
)
WITHDRAWAL_LIST = make_doc(
    "RBI/2022-23/50\nDoR.RRA.31/01.01.101/2022-23\nMay 2, 2022\n\n"
    "Sl. Circular Subject\n1 RBI/2019-20/70 Reporting of frauds\n",
    doc_id="rbi-250",
    title="Regulations Review Authority (RRA 2.0) - Withdrawal of Circulars",
    date=dt.date(2022, 5, 2),
    circular_number="RBI/2022-23/50",
)
ALL = [MD, AMENDMENT, OLD_CIRCULAR, DIRECTIONS, WITHDRAWAL_LIST]


@pytest.fixture(scope="module")
def graph() -> ReferenceGraph:
    return ReferenceGraph.build(ALL)


def test_normalisation() -> None:
    assert normalise_circular_number("RBI / 2024-2025 / 87") == "RBI/2024-25/87"
    assert normalise_circular_number("rbi/fed/2017-18/60") == "RBI/FED/2017-18/60"
    assert normalise_ref_number("DOR.AML.REC.49/ 14.01.001/2024-2025") == (
        "DOR.AML.REC.49/14.01.001/2024-25"
    )
    assert normalise_title("Master Direction \u2013 KYC (Updated as on May 4, 2023)") == (
        "master direction - kyc"
    )


@pytest.mark.parametrize(
    ("context", "heading", "expected"),
    [
        ("the circular stands repealed", "", "repeals"),
        ("This circular supersedes the circular dated", "", "supersedes"),
        ("the instructions are hereby withdrawn", "", "supersedes"),
        ("paragraph 3 has been amended as under", "", "amends"),
        ("in terms of RBI/2024-25/87 as amended from time to time", "", "mentions"),
        ("refer to the circular dated", "", "mentions"),
        ("1 DOR.X/2019-20 Reporting of frauds", "9. Repeal Provisions", "repeals"),
        ("1 RBI/2019-20/70", "RRA 2.0 - Withdrawal of Circulars", "supersedes"),
        ("difficulty in withdrawal of deposits", "", "mentions"),
    ],
)
def test_classify_relation(context: str, heading: str, expected: str) -> None:
    assert classify_relation(context, heading) == expected


def test_sentence_window_stops_at_sentence_breaks() -> None:
    text = "Chapter XI - Repeal. As specified under RBI/2024-25/87 dated May 1. Next one."
    pos = text.index("RBI/")
    assert sentence_around(text, pos, pos + 14) == "As specified under RBI/2024-25/87 dated May 1."


def test_title_and_number_citations_are_linked(graph: ReferenceGraph) -> None:
    rels = {(r.source, r.relation) for r in graph.cited_by("rbi-100")}
    assert ("rbi-200", "amends") in rels  # cited by its title, with "amended" nearby
    assert graph.replaced_by("rbi-100") == []


def test_repeal_table_row_inherits_the_section_heading(graph: ReferenceGraph) -> None:
    refs = {r.source: r for r in graph.cited_by("rbi-150")}
    assert refs["rbi-300"].relation == "repeals"
    assert "Reporting of frauds" in refs["rbi-300"].evidence
    assert refs["rbi-250"].relation == "supersedes"  # listed in a withdrawal circular
    assert graph.replaced_by("rbi-150") == ["rbi-300", "rbi-250"]  # strongest first


def test_boilerplate_amendment_phrase_is_a_mention(graph: ReferenceGraph) -> None:
    refs = {r.source: r.relation for r in graph.cited_by("rbi-200")}
    assert refs["rbi-300"] == "mentions"


def test_no_self_or_backwards_references(graph: ReferenceGraph) -> None:
    for d in ALL:
        for r in graph.cites(d.doc_id):
            assert r.target != d.doc_id
    # rbi-150 (2019) never "cites" rbi-300 (2025) even if its number appeared in the text
    later = make_doc("See RBI/2025-26/200.", doc_id="rbi-140", date=dt.date(2019, 1, 1))
    g = ReferenceGraph.build([*ALL, later])
    assert all(r.target != "rbi-300" for r in g.cites("rbi-140"))
    assert len(g) == len(ReferenceGraph.build(ALL))


def test_graph_cache_round_trip_and_invalidation(tmp_path: Path, graph: ReferenceGraph) -> None:
    docs = {d.doc_id: d for d in ALL}
    path = tmp_path / "graph.json.gz"
    built = ReferenceGraph.load_or_build(docs, path)
    assert path.exists() and built.references == graph.references
    loaded = ReferenceGraph.load_or_build(docs, path)
    for doc_id in docs:
        assert loaded.cited_by(doc_id) == graph.cited_by(doc_id)
        assert loaded.replaced_by(doc_id) == graph.replaced_by(doc_id)

    # a changed corpus invalidates the cache
    changed = {**docs, "rbi-250": WITHDRAWAL_LIST.model_copy(update={"text": "no list"})}
    rebuilt = ReferenceGraph.load_or_build(changed, path)
    assert len(rebuilt) < len(graph)
    assert ReferenceGraph.fingerprint(changed) != ReferenceGraph.fingerprint(docs)

    # an unreadable cache is rebuilt, never fatal
    path.write_bytes(b"not gzip")
    assert len(ReferenceGraph.load_or_build(docs, path)) == len(graph)
    assert ReferenceGraph.load_or_build(docs, None).references == graph.references
