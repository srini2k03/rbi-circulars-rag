"""End-to-end smoke on the fixtures: ingest -> chunk (x3) -> index -> query (x4 modes)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rbirag.cli import app
from rbirag.document import read_documents_jsonl
from tests.conftest import HTML_DIR, REAL_FIXTURES

runner = CliRunner()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "data" / "raw" / "html"
    raw.mkdir(parents=True)
    for name in (*REAL_FIXTURES, "notification_13050_empty.html"):
        sid = name.split("_")[1].split(".")[0]
        shutil.copy(HTML_DIR / name, raw / f"{sid}.html")
    return raw.parent


def invoke(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result.output


def test_full_pipeline_smoke(raw_dir: Path, tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    out = invoke("ingest", "--raw-dir", str(raw_dir), "--stats-out", str(stats_path))
    assert "wrote 3 documents" in out
    stats = json.loads(stats_path.read_text())
    assert stats["documents"] == 3
    docs = read_documents_jsonl(tmp_path / "data" / "processed" / "documents.jsonl")
    assert {d.doc_id for d in docs} == {"rbi-12701", "rbi-12655", "rbi-12590"}

    for chunker in ("fixed", "semantic", "section"):
        out = invoke("index", "--chunker", chunker, "--embedder", "fake")
        assert "chunks" in out
        info = json.loads((tmp_path / "indexes" / chunker / "index_info.json").read_text())
        assert info["documents"] == 3 and info["chunks"] > 0
        for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
            out = invoke(
                "query", "periodic updation of KYC for high risk customers",
                "--chunker", chunker, "--mode", mode, "--embedder", "fake",
                "--reranker", "fake", "--top-k", "3", "--json", "--retrieve-only",
            )  # fmt: skip
            result = json.loads(out)
            assert result["mode"] == mode
            assert result["results"][0]["chunk"]["doc_id"] == "rbi-12655", (chunker, mode)
            assert result["timings_ms"]["total"] >= 0
            if mode == "hybrid_rerank":
                assert result["confidence"]["is_confident"] is True

    text_out = invoke(
        "query", "bulk deposits", "--mode", "bm25", "--chunker", "section", "--retrieve-only"
    )
    assert "[1]" in text_out
    assert "rbi-12701" in text_out

    # answer generation with the deterministic fake LLM, streamed and as JSON
    question = "How often must KYC be updated for high risk customers?"
    common = ["--chunker", "section", "--mode", "hybrid_rerank", "--embedder", "fake",
              "--reranker", "fake", "--llm", "fake"]  # fmt: skip
    streamed = invoke("query", question, *common)
    assert "[1]" in streamed
    assert "fake:fake-extractive" in streamed
    answer = json.loads(invoke("query", question, *common, "--json"))
    assert answer["refused"] is False
    assert answer["citations"]["cited_ids"] == [1]
    assert answer["sources"][0]["doc_id"] == "rbi-12655"
    assert answer["cost"]["usd"] == 0.0
    assert set(answer["timings_ms"]) >= {"retrieve", "rerank", "generate", "total"}
    gated = json.loads(invoke("query", "monsoon cricket score in Chennai", *common, "--json"))
    assert gated["refused"] is True
    assert gated["refusal_reason"] == "low_confidence"


def test_cli_help_lists_commands() -> None:
    out = invoke("--help")
    for cmd in ("fetch", "ingest", "index", "query", "eval"):
        assert cmd in out
