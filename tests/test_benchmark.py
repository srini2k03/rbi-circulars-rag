"""Retrieval benchmark: score cache, paired statistics, the end-to-end run and its replay."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from rbirag.document import Document, write_documents_jsonl
from rbirag.eval.benchmark import (
    BenchContext,
    Strategy,
    auroc,
    build_bench_index,
    doc_index,
    run_benchmark,
    strategy_grid,
    title_family,
    title_relatedness,
    write_outputs,
)
from rbirag.eval.dataset import GoldSpan, QAPair, span_sha256, write_qa_jsonl
from rbirag.eval.history import read_experiments
from rbirag.eval.score_cache import CachedDense, CachedReranker, CacheMissError, ScoreCache
from rbirag.eval.stats import holm_adjust, paired_comparison
from rbirag.index.build import build_index
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.expansion import AcronymExpander
from rbirag.retrieval.params import RetrievalParams
from rbirag.retrieval.rerank import FakeReranker

QUESTIONS = [
    ("rbi-12655", "periodic updation", "How often is periodic updation of KYC done?"),
    ("rbi-12701", "bulk deposit", "What is the bulk deposit limit for small finance banks?"),
    ("rbi-12590", "FIRMS", "How is Form ODI reported on the FIRMS portal?"),
    ("rbi-12655", "high risk", "What applies to high risk customers?"),
    ("rbi-12701", "interest rate", "Can banks offer differential interest rates?"),
    ("rbi-12590", "overseas", "What are the rules for overseas direct investment?"),
]


def make_pairs(docs: list[Document]) -> list[QAPair]:
    text = {d.doc_id: d.text for d in docs}
    pairs = []
    for i, (doc_id, needle, question) in enumerate(QUESTIONS):
        start = text[doc_id].lower().find(needle.lower())
        assert start >= 0, needle
        end = min(len(text[doc_id]), start + 200)
        gold = GoldSpan(doc_id=doc_id, start=start, end=end,
                        text_sha256=span_sha256(text[doc_id][start:end]))  # fmt: skip
        pairs.append(QAPair(qa_id=f"q{i}", question=question, answer="x", qtype="factual",
                            gold=[gold]))  # fmt: skip
    pairs.append(QAPair(qa_id="u1", question="Who won the cricket world cup?", answer="",
                        qtype="unanswerable", answerable=False))  # fmt: skip
    return pairs


def test_score_cache_roundtrip_and_misses(tmp_path: Path) -> None:
    cache = ScoreCache()
    live_rr = FakeReranker()
    rr = CachedReranker(cache, live_rr)
    scores = rr.score("kyc updation", ["kyc updation every two years", "bulk deposits"])
    assert cache.misses == 2
    assert rr.score("kyc updation", ["bulk deposits"]) == [scores[1]]
    assert cache.misses == 2  # served from the cache

    class Live:
        def search(self, query: str, k: int = 10, *, flt: object = None) -> list[tuple[str, float]]:
            return [(f"c{i}", 1.0 / (i + 1)) for i in range(k)]

    dense = CachedDense(cache, Live())
    assert [c for c, _ in dense.search("q", 3)] == ["c0", "c1", "c2"]
    path = tmp_path / "cache.jsonl.gz"
    cache.save(path)
    loaded = ScoreCache.load(path)
    assert loaded.dense == cache.dense and loaded.rerank == cache.rerank
    replay = CachedReranker(loaded)
    assert replay.score("kyc updation", ["bulk deposits"]) == [scores[1]]
    with pytest.raises(CacheMissError):
        replay.score("kyc updation", ["never scored"])
    with pytest.raises(CacheMissError):
        CachedDense(loaded).search("another query", 3)
    with pytest.raises(ValueError, match="top-100"):
        CachedDense(loaded).search("q", 500)
    assert ScoreCache.load(tmp_path / "missing.gz").dense == {}


def test_paired_comparison_and_holm() -> None:
    a = [0.9, 0.8, 0.7, 0.9, 1.0] * 6
    b = [0.5, 0.6, 0.7, 0.4, 0.5] * 6
    cmp = paired_comparison(a, b, n_resamples=2000)
    assert cmp.diff == pytest.approx(np.mean(a) - np.mean(b))
    assert cmp.low <= cmp.diff <= cmp.high and cmp.p_value < 0.01
    assert (cmp.wins, cmp.losses) == (24, 0)
    same = paired_comparison(a, a, n_resamples=500)
    assert same.diff == 0 and same.p_value == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(paired_comparison([], []).diff)


def test_auroc_and_title_family() -> None:
    assert auroc([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert auroc([1, 1], [True, False]) == 0.5
    assert auroc([1], [True]) is None
    assert title_family("Master Direction - KYC Direction, 2016 (updated as on May 4, 2023)") == (
        "master direction kyc direction"
    )


def _context(docs: list[Document], pairs: list[QAPair], caches: dict[str, ScoreCache],
             *, live: bool) -> BenchContext:  # fmt: skip
    from rbirag.config import Settings

    s = Settings(bootstrap_samples=200)
    emb = FakeEmbedder(64)

    def factory(name: str):  # type: ignore[no-untyped-def]
        def build(chunks):  # type: ignore[no-untyped-def]
            return build_index(chunks, embedder=emb, collection=f"t_{name}").dense

        return build if live else None

    primary = build_bench_index(docs, "fixed", 128, s, cache=caches["p"],
                                live_dense=factory("p"),
                                live_reranker=FakeReranker() if live else None)  # fmt: skip
    child = build_bench_index(docs, "fixed", 48, s, cache=caches["c"], live_dense=factory("c"),
                              live_reranker=FakeReranker() if live else None)  # fmt: skip
    return BenchContext(
        settings=s, docs=docs, dev=pairs[:3] + pairs[-1:], test=pairs[3:-1] + pairs[-1:],
        primary=primary, child=child, doc_bm25=doc_index(docs),
        acronyms=AcronymExpander.default(), related=title_relatedness(docs),
        doc_meta={d.doc_id: d.metadata() for d in docs},
    )  # fmt: skip


def _comparable(rows: list[dict]) -> list[dict]:  # type: ignore[type-arg]
    drop = {"latency_ms_p50", "latency_ms_p95"}
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def test_benchmark_runs_and_replays_identically(
    fixture_docs: list[Document], tmp_path: Path
) -> None:
    pairs = make_pairs(fixture_docs)
    caches = {"p": ScoreCache(), "c": ScoreCache()}
    grid = [
        st for st in strategy_grid(have_dense=True, have_reranker=True)
        if st.family in {"bm25", "bm25+rm3", "hybrid weighted", "bm25 + rerank", "dense",
                         "parent-child bm25", "doc-first", "filter:oracle-year"}
    ]  # fmt: skip
    live = run_benchmark(_context(fixture_docs, pairs, caches, live=True), strategies=grid)
    assert {r["family"] for r in live.rows} >= {"bm25", "hybrid weighted", "bm25 + rerank"}
    assert any(t["family"] == "hybrid weighted + rerank" for t in live.tuning)  # derived
    assert live.recommended and "oracle" not in live.recommended
    assert all("vs_bm25" in r for r in live.rows if r["strategy"] != "bm25")
    assert set(live.abstention) >= {"bm25_top_score", "dense_top_cosine"}
    assert len(live.document_level) == 2
    for name, cache in caches.items():
        cache.save(tmp_path / f"{name}.gz")
    replay_caches = {n: ScoreCache.load(tmp_path / f"{n}.gz") for n in caches}
    replay = run_benchmark(
        _context(fixture_docs, pairs, replay_caches, live=False), strategies=grid
    )
    assert _comparable(replay.rows) == _comparable(live.rows)
    assert replay.cache_misses == 0
    write_outputs(replay, tmp_path / "out")
    md = (tmp_path / "out" / "strategies.md").read_text()
    assert "test chunk nDCG@10" in md and "Recommended" in md
    assert (tmp_path / "out" / "tuning.csv").read_text().count("\n") == len(replay.tuning) + 1


def test_strategy_label_and_grid_shapes() -> None:
    no_models = strategy_grid(have_dense=False, have_reranker=False)
    assert {s.params.mode for s in no_models} == {"bm25"}
    full = strategy_grid(have_dense=True, have_reranker=True)
    assert len({s.name for s in full}) == len(full)  # unique names
    st = Strategy("x", "f", RetrievalParams(mode="bm25"), index="child", doc_first=5)
    assert "doc-first top-5" in st.label and "256-token children" in st.label


def test_bench_cli_fake_models_and_replay(
    fixture_docs: list[Document], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rbirag.cli import app

    docs = tmp_path / "docs.jsonl"
    write_documents_jsonl(fixture_docs, docs)
    qa = tmp_path / "qa.jsonl"
    write_qa_jsonl(make_pairs(fixture_docs), qa)
    monkeypatch.setenv("RBIRAG_DEV_FRACTION", "0.5")
    monkeypatch.setenv("RBIRAG_BOOTSTRAP_SAMPLES", "100")
    monkeypatch.setenv("RBIRAG_EMBEDDING_CACHE_DIR", "")
    out = tmp_path / "bench"
    args = ["eval", "bench", "--docs", str(docs), "--qa", str(qa), "--out-dir", str(out),
            "--primary", "fixed-128", "--child", "fixed-48", "--embedder", "fake",
            "--reranker", "fake", "--families", "bm25 + rerank,dense"]  # fmt: skip
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    first = json.loads((out / "strategies.json").read_text())
    assert (out / "score_cache-fixed-128.jsonl.gz").exists()
    assert first["live_timing"] and "identical_top10_vs_cached" in first["live_timing"][0]
    with gzip.open(out / "per_query.jsonl.gz", "rt") as fh:
        assert all("category" in json.loads(line) for line in fh)
    res = CliRunner().invoke(app, [*args, "--no-models"])
    assert res.exit_code == 0, res.output
    second = json.loads((out / "strategies.json").read_text())
    assert _comparable(second["rows"]) == _comparable(first["rows"])
    log = read_experiments(Path("results"))
    assert [e["replayed_from_cache"] for e in log] == [False, True]
