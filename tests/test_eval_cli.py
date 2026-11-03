"""Offline smoke of the evaluation CLI: retrieval ablation -> calibrate -> plan -> generate
(fake LLM + fake judge) -> report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rbirag.cli import app
from rbirag.eval.dataset import GoldSpan, QAPair, write_qa_jsonl
from tests.conftest import HTML_DIR, REAL_FIXTURES

runner = CliRunner()


def invoke(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result.output


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "data" / "raw" / "html"
    raw.mkdir(parents=True)
    for name in REAL_FIXTURES:
        sid = name.split("_")[1].split(".")[0]
        shutil.copy(HTML_DIR / name, raw / f"{sid}.html")
    monkeypatch.setenv("RBIRAG_EMBEDDER", "fake")
    monkeypatch.setenv("RBIRAG_RERANKER", "fake")
    monkeypatch.setenv("RBIRAG_JUDGE_BACKEND", "fake")
    monkeypatch.setenv("RBIRAG_QA_BACKEND", "fake")
    invoke("ingest")
    from rbirag.document import read_documents_jsonl

    docs = {
        d.source_id: d for d in read_documents_jsonl(tmp_path / "data/processed/documents.jsonl")
    }

    def gold(sid: int, phrase: str) -> GoldSpan:
        d = docs[sid]
        i = d.text.index(phrase)
        return GoldSpan(doc_id=d.doc_id, start=i, end=i + len(phrase))

    pairs = [
        QAPair(qa_id=f"a{i}", question=q, answer="x", qtype="factual", gold=[gold(sid, ph)], verified=i == 0,
               review_status="accepted" if i == 0 else "pending")
        for i, (q, sid, ph) in enumerate([
            ("How often is periodic updation of KYC for high risk customers?", 12655, "once in every two years"),
            ("Differential interest rates on bulk deposits of small finance banks", 12701, "differential rates"),
            ("Form ODI reporting through the FIRMS portal deadline", 12590, "within 30 days"),
            ("Advance intimations for KYC updation by letter", 12655, "three advance intimations"),
        ])
    ] + [
        QAPair(qa_id=f"u{i}", question=q, answer="", qtype="unanswerable", answerable=False)
        for i, q in enumerate(["SEBI expense ratio cap for index funds", "GST rate on gold making charges"])
    ]  # fmt: skip
    write_qa_jsonl(pairs, tmp_path / "data/eval/qa_v1.jsonl")
    return tmp_path


def test_eval_pipeline_cli(workspace: Path) -> None:
    out = invoke(
        "eval", "retrieval", "--chunkers", "section,fixed", "--sizes", "64,128",
        "--modes", "bm25,dense,hybrid_rerank", "--tag", "smoke",
    )  # fmt: skip
    assert "section-64-hybrid_rerank" in out
    rows = json.loads((workspace / "results/retrieval/rows-smoke.json").read_text())
    assert len(rows) == 2 * 2 * 3

    invoke("eval", "calibrate")
    cal = json.loads((workspace / "results/calibration.json").read_text())
    assert 0 <= cal["threshold"] <= 1

    out = invoke("eval", "plan", "--backends", "fake", "--subset-size", "2")
    assert "planned LLM calls" in out
    plan = json.loads((workspace / "results/generation/plan.json").read_text())
    assert len(plan["full"]) == 1
    assert len(plan["subset"]) == 2

    out = invoke("eval", "generate", "--backend", "fake", "--nli", "fake")
    status = json.loads((workspace / "results/generation/status-fake-all.json").read_text())
    assert status["complete"] is True
    records = list((workspace / "results/generation").glob("records-fake-*.jsonl"))
    assert len(records) == 3
    # rerun is a no-op (resumable)
    invoke("eval", "generate", "--backend", "fake", "--nli", "fake")
    status = json.loads((workspace / "results/generation/status-fake-all.json").read_text())
    assert all(c["generated"] == 0 for c in status["configs"].values())

    out = invoke("eval", "report")
    assert "README.md" in out
    md = (workspace / "results/README.md").read_text()
    assert "## Retrieval ablation" in md
    assert "## Generation" in md
    # the fake judge cannot produce valid verdicts, so no config counts as measured
    assert "No configuration is complete yet." in md
    assert "incomplete (" in md
    assert "Models that actually answered" in md
    assert "human-verified: 1" in md


def test_qa_generate_cli_with_fake_backend(workspace: Path) -> None:
    invoke("index", "--chunker", "section", "--no-dense")
    result = runner.invoke(
        app,
        ["eval", "qa-generate", "--chunks", str(workspace / "indexes/section/chunks.jsonl"),
         "--n-answerable", "3", "--n-unanswerable", "0", "--stats-out", str(workspace / "stats.json")],
    )  # fmt: skip
    # the fake LLM refuses, so the run attempted work but produced nothing: exit code 3
    assert result.exit_code == 3, result.output
    stats = json.loads((workspace / "stats.json").read_text())
    assert stats["new_items"] == 0
    assert stats["llm_calls_this_run"] == 3
    assert not (workspace / "data/eval/qa_candidates.jsonl").exists()  # nothing to commit


def test_fatal_configuration_error_aborts_without_output(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rbirag.generation.rag as rag
    from rbirag.generation.llm import ConfigurationError, ScriptedLLM

    def boom(messages: object, system: object) -> str:
        raise ConfigurationError("Fatal configuration error: model 'x' was not found (404).")

    monkeypatch.setattr(rag, "build_llm", lambda *a, **k: ScriptedLLM(boom))
    invoke("index", "--chunker", "section", "--no-dense")
    result = runner.invoke(
        app,
        ["eval", "qa-generate", "--chunks", str(workspace / "indexes/section/chunks.jsonl"),
         "--n-answerable", "3", "--stats-out", str(workspace / "stats.json")],
    )  # fmt: skip
    assert result.exit_code == 2
    assert "Fatal configuration error" in result.output
    assert not (workspace / "stats.json").exists()
    assert not (workspace / "data/eval/qa_candidates.jsonl").exists()
