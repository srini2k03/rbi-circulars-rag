"""Gold spans are offsets into one corpus release; they are hashed and verified."""

from __future__ import annotations

import json
from pathlib import Path

from rbirag.chunking import SectionChunker
from rbirag.document import Document, write_documents_jsonl
from rbirag.eval.dataset import GoldSpan, QAPair, pin_gold, span_sha256, verify_gold, write_qa_jsonl
from rbirag.eval.qa_generate import Sample, generate_pair, pin_from_chunks
from rbirag.generation.llm import ScriptedLLM
from rbirag.index.build import chunk_documents


def qa(doc: Document, phrase: str, *, pinned: bool = True, qa_id: str = "q") -> QAPair:
    i = doc.text.index(phrase)
    g = GoldSpan(doc_id=doc.doc_id, start=i, end=i + len(phrase),
                 text_sha256=span_sha256(phrase) if pinned else None)  # fmt: skip
    return QAPair(qa_id=qa_id, question="?", answer="a", qtype="factual", gold=[g])


def test_verify_gold(fixture_docs: list[Document]) -> None:
    doc = next(d for d in fixture_docs if d.source_id == 12655)
    good = qa(doc, "at least once in every two years", qa_id="good")
    unpinned = qa(doc, "three advance intimations", pinned=False, qa_id="unpinned")
    texts = {d.doc_id: d.text for d in fixture_docs}
    ok, check = verify_gold([good, unpinned], texts)
    assert [p.qa_id for p in ok] == ["good", "unpinned"]
    assert (check.checked, check.unpinned, check.mismatched) == (2, 1, [])
    # the corpus release changed: text shifted under the same offsets
    shifted = {**texts, doc.doc_id: "PREFIX " + doc.text}
    ok, check = verify_gold([good, unpinned], shifted)
    assert [p.qa_id for p in ok] == ["unpinned"]  # cannot be verified, kept and counted
    assert check.mismatched == ["good"]
    assert "EXCLUDED" in check.message
    ok, check = verify_gold([good], {})
    assert ok == []
    assert check.missing_docs == [doc.doc_id]


def test_pin_gold_and_pin_from_chunks(fixture_docs: list[Document]) -> None:
    doc = next(d for d in fixture_docs if d.source_id == 12655)
    pair = qa(doc, "three advance intimations", pinned=False)
    pinned, n = pin_gold([pair], {d.doc_id: d.text for d in fixture_docs})
    assert n == 1
    assert pinned[0].gold[0].text_sha256 == span_sha256("three advance intimations")
    chunks = chunk_documents(fixture_docs, SectionChunker(max_tokens=80, min_tokens=20))
    from_chunks, n2 = pin_from_chunks([pair], chunks)
    assert n2 == 1
    assert from_chunks[0].gold[0].text_sha256 == pinned[0].gold[0].text_sha256


def test_generated_pairs_carry_span_hashes(fixture_docs: list[Document]) -> None:
    chunks = chunk_documents(fixture_docs, SectionChunker(max_tokens=80, min_tokens=20))
    c = next(c for c in chunks if "two years" in c.text)
    reply = json.dumps({"question": "How often must KYC be updated for high risk customers?",
                        "answer": "Every two years.", "evidence": "at least once in every two years"})  # fmt: skip
    pair = generate_pair(ScriptedLLM([reply]), Sample((c,), "numeric"))
    assert pair is not None
    g = pair.gold[0]
    doc = next(d for d in fixture_docs if d.doc_id == g.doc_id)
    assert g.text_sha256 == span_sha256(doc.text[g.start : g.end])


def test_retrieval_and_report_exclude_mismatched_pairs(
    fixture_docs: list[Document], tmp_path: Path, monkeypatch: object
) -> None:
    from typer.testing import CliRunner

    from rbirag.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RBIRAG_EMBEDDER", "fake")
    monkeypatch.setenv("RBIRAG_RERANKER", "fake")
    write_documents_jsonl(fixture_docs, tmp_path / "data/processed/documents.jsonl")
    doc = next(d for d in fixture_docs if d.source_id == 12655)
    good = qa(doc, "at least once in every two years", qa_id="good")
    stale = qa(doc, "three advance intimations", qa_id="stale")
    stale = stale.model_copy(
        update={"gold": [stale.gold[0].model_copy(update={"text_sha256": "0" * 64})]}
    )
    write_qa_jsonl([good, stale], tmp_path / "data/eval/qa_v1.jsonl")
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "eval",
            "retrieval",
            "--chunkers",
            "section",
            "--sizes",
            "64",
            "--modes",
            "bm25",
            "--tag",
            "t",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "1 pairs EXCLUDED" in r.output
    check = json.loads((tmp_path / "results/retrieval/gold_check-t.json").read_text())
    assert check["mismatched"] == ["stale"]
    per_query = (tmp_path / "results/retrieval/per_query-t.jsonl").read_text()
    assert '"stale"' not in per_query
    r = runner.invoke(app, ["eval", "report"])
    assert r.exit_code == 0, r.output
    meta = json.loads((tmp_path / "results/run_metadata.json").read_text())
    assert meta["gold_mismatch_excluded"] == ["stale"]
    assert "1 pairs excluded" in (tmp_path / "results/README.md").read_text()
