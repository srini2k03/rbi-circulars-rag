"""Dev/test hygiene: split by document, select and calibrate on dev, report on test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rbirag.cli import app
from rbirag.eval.dataset import GoldSpan, QAPair, split, split_key, write_qa_jsonl

runner = CliRunner()


def pair(i: int, doc: str | None) -> QAPair:
    if doc is None:
        return QAPair(
            qa_id=f"u{i}",
            question=f"unanswerable {i}?",
            answer="",
            qtype="unanswerable",
            answerable=False,
        )
    return QAPair(qa_id=f"q{i}", question=f"question {i}?", answer="a", qtype="factual",
                  gold=[GoldSpan(doc_id=doc, start=0, end=10)])  # fmt: skip


def test_split_is_by_document_and_deterministic() -> None:
    pairs = [pair(i, f"doc{i % 12}") for i in range(120)] + [
        pair(1000 + i, None) for i in range(20)
    ]
    dev, test = split(pairs, 0.3, seed=7)
    assert len(dev) + len(test) == 140
    dev_docs = {d for p in dev for d in p.gold_doc_ids}
    test_docs = {d for p in test for d in p.gold_doc_ids}
    assert dev_docs and test_docs
    assert not dev_docs & test_docs  # no document contributes to both splits
    assert split(pairs, 0.3, seed=7) == (dev, test)
    assert split_key(pairs[0]) == "doc:doc0"
    assert split_key(pairs[-1]) == "qa:u1019"
    assert {p.qa_id for p in dev} != {p.qa_id for p in split(pairs, 0.3, seed=8)[0]}


def _metrics(qa_id: str, v: float) -> dict:  # type: ignore[type-arg]
    return {
        "qa_id": qa_id,
        "level": "chunk",
        "recall": {"1": v, "3": v, "5": v, "10": v},
        "mrr": v,
        "ndcg": v,
    }


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[QAPair], list[QAPair]]:
    monkeypatch.chdir(tmp_path)
    pairs = [pair(i, f"doc{i}") for i in range(40)] + [pair(100 + i, None) for i in range(6)]
    write_qa_jsonl(pairs, tmp_path / "data/eval/qa_v1.jsonl")
    dev, test = split(pairs, 0.3, 13)
    assert dev and test
    dev_ids = {p.qa_id for p in dev}
    rows, items = [], []
    # config A is best on dev, config B is best on test
    for config, dev_v, test_v in (
        ("section-512-hybrid_rerank", 0.9, 0.1),
        ("fixed-256-hybrid_rerank", 0.2, 0.95),
    ):
        rows.append({"config": config, "chunker": config.split("-")[0], "size": int(config.split("-")[1]),
                     "mode": "hybrid_rerank", "n_chunks": 10, "chunk_ndcg@10": 0.5})  # fmt: skip
        for p in pairs:
            v = dev_v if p.qa_id in dev_ids else test_v
            items.append({"config": config, "qa_id": p.qa_id, "question": p.question, "answerable": p.answerable,
                          "retrieved": [], "doc": _metrics(p.qa_id, v) if p.answerable else None,
                          "chunk": _metrics(p.qa_id, v) if p.answerable else None,
                          "confidence": v if p.answerable else 0.05, "top_score": 0.0, "latency_ms": 1.0})  # fmt: skip
    out = tmp_path / "results/retrieval"
    out.mkdir(parents=True)
    (out / "rows-x.json").write_text(json.dumps(rows))
    (out / "per_query-x.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    return tmp_path, dev, test


def invoke(*args: str) -> str:
    r = runner.invoke(app, list(args))
    assert r.exit_code == 0, r.output + repr(r.exception)
    return r.output


def test_plan_selects_on_dev_and_runs_on_test(ws: tuple[Path, list[QAPair], list[QAPair]]) -> None:
    root, dev, test = ws
    out = invoke("eval", "plan", "--backends", "fake", "--subset-configs", "0")
    assert f"{len(dev)} dev (selection) / {len(test)} test (reported)" in out
    plan = json.loads((root / "results/generation/plan.json").read_text())
    assert plan["full"] == ["section-512-hybrid_rerank"]  # best on dev, worse on test
    assert set(plan["all_ids"]) == {p.qa_id for p in test}  # dev never generated/reported


def test_calibrate_uses_dev_config_and_dev_items(
    ws: tuple[Path, list[QAPair], list[QAPair]],
) -> None:
    root, dev, _ = ws
    invoke("eval", "calibrate")
    cal = json.loads((root / "results/calibration.json").read_text())
    assert cal["config"] == "section-512-hybrid_rerank"
    assert cal["n_dev"] == len(dev)


def test_report_headline_is_test_only_with_dev_section(
    ws: tuple[Path, list[QAPair], list[QAPair]],
) -> None:
    root, dev, test = ws
    invoke("eval", "report")
    md = (root / "results/README.md").read_text()
    assert f"{len(test)} test questions" in md
    assert f"{len(dev)} dev questions" in md
    assert "## Dev split (selection only, not a result)" in md
    head = md.split("## Retrieval ablation")[1].split("## Dev split")[0]
    # on the test split config B wins, which the headline table must show first
    assert head.index("fixed-256-hybrid_rerank") < head.index("section-512-hybrid_rerank")
    meta = json.loads((root / "results/run_metadata.json").read_text())
    assert meta["split"]["test_questions"] == len(test)
    assert meta["split"]["dev_questions"] == len(dev)
    n_test_answerable = sum(p.answerable for p in test)
    import csv

    with (root / "results/retrieval_ablation.csv").open() as fh:
        assert all(int(r["n_queries"]) == n_test_answerable for r in csv.DictReader(fh))


def test_split_requires_per_query_results(ws: tuple[Path, list[QAPair], list[QAPair]]) -> None:
    root, _, _ = ws
    (root / "results/retrieval/per_query-x.jsonl").unlink()
    r = runner.invoke(app, ["eval", "plan", "--backends", "fake"])
    assert r.exit_code != 0
    assert "per-query retrieval results are missing" in r.output
