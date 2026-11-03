"""Offline smoke of the evaluation CLI: retrieval ablation -> calibrate -> plan -> generate
(fake LLM + fake judge) -> report."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rbirag.cli import app
from rbirag.eval.ai_review import AIReview, AIReviewRecord, append_ai_reviews
from rbirag.eval.dataset import GoldSpan, QAPair, read_qa_jsonl, write_qa_jsonl
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
    assert (
        "**Test set: reviewed test set (`data/eval/qa_v1.jsonl`): 6 model-generated pairs "
        "(0 AI-reviewed by Claude, 1 human-verified).**" in md
    )
    assert "## AI-reviewed subset" not in md  # no AI review file in this workspace
    meta = json.loads((workspace / "results/run_metadata.json").read_text())
    assert meta["test_set_file"] == "data/eval/qa_v1.jsonl"

    # every planned item is answered and has a judge attempt -> nothing pending
    prog = json.loads(invoke("eval", "progress", "--backends", "fake"))
    assert prog["plan_exists"] is True
    assert prog["done"] is True
    # drop one record line -> that item is pending again
    rec_file = sorted((workspace / "results/generation").glob("records-fake-*.jsonl"))[0]
    lines = rec_file.read_text().splitlines()
    first_id = json.loads(lines[0])["qa_id"]
    rec_file.write_text("\n".join(x for x in lines if json.loads(x)["qa_id"] != first_id) + "\n")
    prog = json.loads(invoke("eval", "progress", "--backends", "fake"))
    assert prog["done"] is False
    assert prog["pending"] >= 1


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


def test_candidates_are_used_and_labelled_until_reviewed_set_exists(workspace: Path) -> None:
    reviewed = workspace / "data/eval/qa_v1.jsonl"
    reviewed.rename(workspace / "data/eval/qa_candidates.jsonl")
    assert json.loads(invoke("eval", "progress"))["plan_exists"] is False
    invoke(
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
    )
    invoke("eval", "report")
    md = (workspace / "results/README.md").read_text()
    assert "model-generated test set (`data/eval/qa_candidates.jsonl`)" in md
    assert "6 model-generated pairs (0 AI-reviewed by Claude, 0 human-verified)" in md
    meta = json.loads((workspace / "results/run_metadata.json").read_text())
    assert meta["test_set"].startswith("model-generated test set")


def test_qa_status_targets(workspace: Path) -> None:
    out = json.loads(invoke("eval", "qa-status", "--candidates", str(workspace / "data/eval/qa_v1.jsonl"),
                            "--n-answerable", "4", "--n-unanswerable", "2"))  # fmt: skip
    assert out["targets_met"] is True
    out = json.loads(
        invoke("eval", "qa-status", "--candidates", str(workspace / "data/eval/qa_v1.jsonl"))
    )
    assert out["targets_met"] is False
    assert (out["answerable"], out["unanswerable"]) == (4, 2)
    missing = json.loads(invoke("eval", "qa-status", "--candidates", str(workspace / "nope.jsonl")))
    assert missing["targets_met"] is False


def test_qa_status_counts_auto_failed_but_not_rejected(workspace: Path) -> None:
    path = workspace / "data/eval/qa_v1.jsonl"
    pairs = read_qa_jsonl(path)
    pairs[1] = pairs[1].model_copy(update={"auto_check": False})
    pairs[2] = pairs[2].model_copy(update={"review_status": "rejected"})
    write_qa_jsonl(pairs, path)
    out = json.loads(invoke("eval", "qa-status", "--candidates", str(path)))
    assert out["answerable"] == 3  # the auto-failed pair counts, the rejected one does not
    assert out["passing_auto_checks"] == 4


def _ai_review(pair: QAPair, decision: str, question: str | None = None) -> AIReviewRecord:
    return AIReviewRecord(
        qa_id=pair.qa_id, qtype=pair.qtype, answerable=pair.answerable,
        original_question=pair.question, original_answer=pair.answer,
        question=question or pair.question, answer=pair.answer,
        review=AIReview(decision=decision, notes="n", reviewed_at="2026-11-03"),  # type: ignore[arg-type]
    )  # fmt: skip


def test_ai_review_filters_edits_and_labels_the_report(workspace: Path) -> None:
    """AI-rejected pairs leave the test set, AI edits are used, results are re-filtered,
    and every label says 'AI-reviewed by Claude', never human-verified."""
    path = workspace / "data/eval/qa_v1.jsonl"
    pairs = {p.qa_id: p for p in read_qa_jsonl(path)}
    invoke("eval", "retrieval", "--chunkers", "section", "--sizes", "64", "--modes", "bm25",
           "--tag", "t")  # fmt: skip
    edited_q = "How often must KYC of high risk customers be updated periodically?"
    append_ai_reviews(
        [
            _ai_review(pairs["a1"], "accept"),
            _ai_review(pairs["a2"], "reject"),
            _ai_review(pairs["a3"], "edit", question=edited_q),
            _ai_review(pairs["a0"], "reject"),  # a0 was human-accepted: the human decision wins
        ],
        workspace / "data/eval/qa_ai_reviewed.jsonl",
    )
    invoke("eval", "report")
    meta = json.loads((workspace / "results/run_metadata.json").read_text())
    assert meta["questions"] == 5  # a2 excluded
    assert meta["human_verified"] == 1  # unchanged: AI review never counts as human
    assert meta["ai_review"]["reviewed_in_test_set"] == 2
    assert meta["ai_review"]["rejected_excluded"] == 1
    assert meta["test_set"] == (
        "reviewed test set (`data/eval/qa_v1.jsonl`): 5 model-generated pairs "
        "(2 AI-reviewed by Claude, 1 human-verified); 1 AI-rejected pairs excluded"
    )
    md = (workspace / "results/README.md").read_text()
    assert "## AI-reviewed subset (2 pairs)" in md
    assert "not human verification" in md
    # headline retrieval rows were recomputed without the rejected pair and the pre-edit run
    row = json.loads((workspace / "results/retrieval/rows-t.json").read_text())[0]
    assert row["n_queries"] == 4
    with (workspace / "results/retrieval_ablation.csv").open() as fh:
        assert next(csv.DictReader(fh))["n_queries"] == "2"  # a0, a1 (a3 ran before its edit)
    with (workspace / "results/retrieval_ablation_ai_reviewed.csv").open() as fh:
        assert next(csv.DictReader(fh))["n_queries"] == "1"  # a1 only

    # the pipeline now uses the edited wording, and the source files are untouched
    from rbirag.config import get_settings
    from rbirag.eval.cli import load_qa

    used = {p.qa_id: p for p in load_qa(get_settings())}
    assert "a2" not in used
    assert used["a3"].question == edited_q
    assert used["a3"].ai_review is not None and used["a3"].verified is False
    assert used["a0"].ai_review is None  # human decision took precedence
    assert "ai_review" not in path.read_text()
    status = json.loads(invoke("eval", "qa-status", "--candidates", str(path)))
    assert (status["ai_reviewed"], status["ai_rejected"], status["human_verified"]) == (4, 2, 1)
