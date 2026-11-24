"""Document intelligence: metadata, references, the relation graph and diffs."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from typer.testing import CliRunner

from rbirag.document import Document, write_documents_jsonl
from rbirag.ingest.metadata import detect_sections
from rbirag.intel.cli import load_graph, save_graph
from rbirag.intel.extract import build_intel, classify_category, extract_effective, relation_type
from rbirag.intel.graph import RelationGraph, compare_documents
from rbirag.intel.review import precision_report, wilson


def doc(doc_id: str, title: str, text: str, day: str, **kw: object) -> Document:
    return Document(doc_id=doc_id, url=f"https://www.rbi.org.in/{doc_id}", title=title,
                    text=text, sections=detect_sections(text), date=dt.date.fromisoformat(day),
                    **kw)  # type: ignore[arg-type]  # fmt: skip


MD = "Master Direction - Know Your Customer (KYC) Direction, 2016"
DOCS = [
    doc("a", MD, "RBI/2015-16/108\nDBR.AML.BC.No.81/14.01.001/2015-16\n\nFebruary 25, 2016\n\n"
        "1. These Directions shall come into force with immediate effect.\n\n"
        "2. Periodic updation shall be carried out at least once in every two years for high "
        "risk customers.\n\n12 Amended vide circular DOR.AML.REC.5/14.01.001/2020-21 dated "
        "January 1, 2021.\n", "2016-02-25", circular_number="RBI/2015-16/108",
        ref_number="DBR.AML.BC.No.81/14.01.001/2015-16"),
    doc("b", "Amendment to the Master Direction on KYC", "RBI/2020-21/50\n"
        "DOR.AML.REC.5/14.01.001/2020-21\n\nJanuary 1, 2021\n\nPlease refer to the "
        f"{MD} dated February 25, 2016. Paragraph 2 of the {MD} is amended as under.\n\n"
        "The amendments shall come into effect from April 1, 2021.\n", "2021-01-01",
        circular_number="RBI/2020-21/50", ref_number="DOR.AML.REC.5/14.01.001/2020-21"),
    doc("c", "Review of KYC instructions", "RBI/2021-22/10\nDOR.AML.REC.9/14.01.001/2021-22\n\n"
        "April 2, 2022\n\nThis circular supersedes the circular DOR.AML.REC.5/14.01.001/2020-21 "
        "dated January 1, 2021. It also refers to DBR.No.Leg.BC.99/09.07.005/2010-11 dated "
        "May 5, 2010.\n", "2022-04-02", circular_number="RBI/2021-22/10",
        ref_number="DOR.AML.REC.9/14.01.001/2021-22"),
    doc("e", MD + " (Updated as on May 10, 2023)", "RBI/2015-16/108\n"
        "DBR.AML.BC.No.81/14.01.001/2015-16\n\nFebruary 25, 2016\n\n1. These Directions shall "
        "come into force with immediate effect.\n\n2. Periodic updation shall be carried out "
        "at least once in every three years for high risk customers.\n\n3. A new paragraph on "
        "video based customer identification was inserted.\n", "2023-05-10",
        circular_number="RBI/2015-16/108", ref_number="DBR.AML.BC.No.81/14.01.001/2015-16"),
]  # fmt: skip


def test_metadata_and_categories() -> None:
    intel, _ = build_intel(DOCS)
    assert intel["a"].category == "master_direction" and intel["e"].version_note
    assert intel["a"].effective.kind == "immediate"
    assert intel["a"].effective.date == dt.date(2016, 2, 25)
    assert intel["b"].effective.kind == "explicit"
    assert intel["b"].effective.date == dt.date(2021, 4, 1)
    assert intel["b"].is_amendment and intel["a"].family == intel["e"].family
    assert classify_category("Master Circular - Lead Bank Scheme", None, "") == "master_circular"
    assert classify_category("X", "A.P. (DIR Series) Circular No. 3", "") == "ap_dir_circular"
    assert classify_category("RBI (Credit Risk) Directions, 2025", None, "") == "directions"
    mc = extract_effective("These guidelines take effect with effect from May 1, 2020.", None,
                           "master_circular")  # fmt: skip
    assert mc.kind == "none"


def test_relation_cues() -> None:
    assert relation_type("12 Amended vide circular X dated ...", list_row=None)[0] == "amended_by"
    assert relation_type("Repealed vide Circular X", list_row=None)[0] == "amended_by"
    assert relation_type("This circular supersedes the circular X", list_row=None)[0] == (
        "supersedes"
    )
    assert relation_type("in partial modification of the notification X", list_row=None)[0] == (
        "partially_modifies"
    )
    assert relation_type("para 4 of circular X stands modified as under", list_row=None)[0] == (
        "amends"
    )
    for neutral in ("the Modified Interest Subvention Scheme of circular X",
                    "the Master Direction X as amended from time to time",
                    "Certain amendments to the entries in the List were notified vide X"):  # fmt: skip
        assert relation_type(neutral, list_row=None)[0] == "refers_to", neutral
    assert (
        relation_type("3. X 01.01.2020 Amendment to Master Direction", list_row="consolidated")[0]
        == "consolidates"
    )


def test_graph_relations_latest_and_unknowns(tmp_path: Path) -> None:
    intel, relations = build_intel(DOCS)
    by = {(r.citing, r.cited, r.type) for r in relations}
    assert ("a", "b", "amended_by") in by  # footnote in the consolidated direction
    assert ("c", "b", "supersedes") in by
    assert any(r.citing == "b" and r.cited in {"a", "e"} and r.type == "amends" for r in relations)
    graph = RelationGraph(intel, relations)
    assert [e.actor for e in graph.changed_by("a")] == ["b"]
    assert graph.versions("a") == ["a", "e"] and graph.original("e") == "a"
    latest = graph.latest_in_corpus("a")
    assert latest.latest == "e" and not latest.is_latest
    assert "not a verified statement of the current regulatory position" in latest.disclaimer
    assert graph.latest_in_corpus("b").latest == "c"
    assert graph.latest_in_corpus("b").how == ["superseded by"]
    unknown = graph.unresolved("c")
    assert len(unknown) == 1 and unknown[0].resolution == "not_in_corpus"
    assert unknown[0].cited is None and unknown[0].confidence == "low"
    assert dict(graph.related("b"))["c"].startswith("supersedes")
    stats = graph.stats()
    assert stats["references_unknown"] >= 1 and stats["edges_by_type"]["supersedes"] == 1
    export = graph.export()
    assert set(export) == {"newest", "disclaimer", "docs", "edges", "families"}
    path = tmp_path / "g.json.gz"
    save_graph(intel, relations, path)
    again = load_graph(path)
    assert again.latest_in_corpus("a").latest == "e" and len(again.edges) == len(graph.edges)


def test_compare_documents() -> None:
    d = compare_documents(DOCS[0], DOCS[3])
    assert any("two years" in a and "three years" in b for a, b in d.changed)
    assert any("video based" in s for s in d.added)
    assert any("Amended vide" in s for s in d.removed)
    assert 0 < d.similarity < 1


def test_precision_report(tmp_path: Path) -> None:
    sample = tmp_path / "s.jsonl"
    sample.write_text(
        json.dumps({"id": "r1", "kind": "relation", "type": "amends"}) + "\n"
        + json.dumps({"id": "r2", "kind": "relation", "type": "amends"}) + "\n"
        + json.dumps({"id": "e1", "kind": "effective_date"}) + "\n"
    )  # fmt: skip
    sample.with_suffix(".meta.json").write_text(
        json.dumps({"resolved_relations_by_type": {"amends": 5}})
    )
    review = tmp_path / "r.jsonl"
    review.write_text(
        json.dumps({"id": "r1", "reviewer": "ai-assistant (LLM)", "target_correct": True,
                    "type_correct": True}) + "\n"
        + json.dumps({"id": "r2", "reviewer": "ai-assistant (LLM)", "target_correct": True,
                      "type_correct": False}) + "\n"
        + json.dumps({"id": "e1", "reviewer": "ai-assistant (LLM)", "date_correct": True}) + "\n"
    )  # fmt: skip
    rep = precision_report(sample, review)
    assert rep["label"].startswith("AI-checked") and not rep["human_verified"]
    assert rep["amends"]["both"]["precision"] == 0.5 and rep["amends"]["target"]["precision"] == 1
    assert rep["effective_date"]["both"]["precision"] == 1.0
    assert rep["population_weighted_precision"]["precision"] == 0.5
    lo, hi = wilson(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0


def test_intel_cli(tmp_path: Path) -> None:
    from rbirag.cli import app

    docs = tmp_path / "docs.jsonl"
    write_documents_jsonl(DOCS, docs)
    runner = CliRunner()
    res = runner.invoke(app, ["intel", "build", "--docs", str(docs), "--out", str(tmp_path),
                              "--export", str(tmp_path / "site.json")])  # fmt: skip
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["intel", "show", "a", "--graph", str(tmp_path / "graph.json.gz")])
    assert res.exit_code == 0, res.output
    shown = json.loads(res.output)
    assert shown["latest_in_corpus"]["latest"] == "e" and shown["original"] == "a"
    res = runner.invoke(app, ["intel", "diff", "a", "e", "--docs", str(docs)])
    assert res.exit_code == 0 and json.loads(res.output)["added"]
    res = runner.invoke(app, ["intel", "show", "zzz", "--graph", str(tmp_path / "graph.json.gz")])
    assert res.exit_code != 0


def test_multi_doc_cli(tmp_path: Path) -> None:
    from rbirag.cli import app

    docs = tmp_path / "docs.jsonl"
    write_documents_jsonl(DOCS, docs)
    intel, relations = build_intel(DOCS)
    save_graph(intel, relations, tmp_path / "graph.json.gz")
    res = CliRunner().invoke(app, ["eval", "multi-doc", "--docs", str(docs), "--graph",
                                   str(tmp_path / "graph.json.gz"), "--out-dir", str(tmp_path),
                                   "--no-log"])  # fmt: skip
    assert res.exit_code == 0, res.output
    out = json.loads((tmp_path / "multi_doc.json").read_text())
    assert out["summary"]["questions"] >= 1
    assert out["summary"]["both_in_top5_plus_graph"] >= out["summary"]["both_in_top5"]
    assert all(len(r["gold"]) == 2 for r in out["rows"])
