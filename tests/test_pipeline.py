from __future__ import annotations

from pathlib import Path

import pytest

from rbirag.chunking import SectionChunker
from rbirag.document import Document
from rbirag.index.build import build_index, chunk_documents
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.pipeline import MODES, Retriever
from rbirag.retrieval.rerank import FakeReranker


@pytest.fixture
def retriever(fixture_docs: list[Document]) -> Retriever:
    chunks = chunk_documents(fixture_docs, SectionChunker(max_tokens=120, min_tokens=30))
    bundle = build_index(chunks, embedder=FakeEmbedder(128))
    return Retriever(
        bundle.store,
        bm25=bundle.bm25,
        dense=bundle.dense,
        reranker=FakeReranker(),
        candidate_k=20,
        rerank_k=10,
    )


QUESTIONS = {
    "How often is periodic updation of KYC for high risk customers?": "rbi-12655",
    "Differential interest rate on bulk deposits for small finance banks": "rbi-12701",
    "Form ODI reporting through the FIRMS portal": "rbi-12590",
}


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_the_right_document(retriever: Retriever, mode: str) -> None:
    for question, doc_id in QUESTIONS.items():
        result = retriever.retrieve(question, top_k=3, mode=mode)  # type: ignore[arg-type]
        assert result.mode == mode
        assert result.results, question
        assert result.results[0].chunk.doc_id == doc_id, (mode, question)
        assert [r.rank for r in result.results] == list(range(1, len(result.results) + 1))
        assert "total" in result.timings_ms
        assert all(v >= 0 for v in result.timings_ms.values())


def test_stage_timings_and_scores_per_mode(retriever: Retriever) -> None:
    r = retriever.retrieve("bulk deposits", mode="bm25")
    assert set(r.timings_ms) == {"bm25", "total"} and r.confidence is None
    assert "bm25" in r.results[0].stage_scores
    r = retriever.retrieve("bulk deposits", mode="dense")
    assert set(r.timings_ms) == {"dense", "total"}
    r = retriever.retrieve("bulk deposits", mode="hybrid")
    assert set(r.timings_ms) == {"bm25", "dense", "fusion", "total"}
    assert r.results[0].score == r.results[0].stage_scores["rrf"]
    r = retriever.retrieve("bulk deposits", mode="hybrid_rerank", top_k=2)
    assert set(r.timings_ms) == {"bm25", "dense", "fusion", "rerank", "total"}
    assert r.confidence is not None
    assert r.results[0].stage_scores["rerank"] == r.results[0].score
    assert len(r.results) == 2
    scores = [x.score for x in r.results]
    assert scores == sorted(scores, reverse=True)


def test_confidence_low_for_unrelated_query(retriever: Retriever) -> None:
    r = retriever.retrieve("monsoon cricket score in Chennai", mode="hybrid_rerank")
    assert r.confidence is not None and not r.confidence.is_confident


def test_mode_validation(retriever: Retriever, fixture_docs: list[Document]) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        retriever.retrieve("x", mode="magic")  # type: ignore[arg-type]
    bm25_only = Retriever(retriever.store, bm25=retriever.bm25)
    assert bm25_only.retrieve("deposits", mode="bm25").results
    with pytest.raises(ValueError, match="dense index"):
        bm25_only.retrieve("deposits", mode="hybrid")
    no_rerank = Retriever(retriever.store, bm25=retriever.bm25, dense=retriever.dense)
    with pytest.raises(ValueError, match="reranker"):
        no_rerank.retrieve("deposits", mode="hybrid_rerank")


def test_make_retriever_uses_every_retrieval_setting(fixture_docs: list[Document]) -> None:
    from rbirag.config import Settings
    from rbirag.retrieval.pipeline import make_retriever

    chunks = chunk_documents(fixture_docs, SectionChunker(max_tokens=120, min_tokens=30))
    b = build_index(chunks, embedder=FakeEmbedder(32))
    s = Settings(candidate_k=7, rerank_k=4, rrf_k=10, bm25_weight=0.5, dense_weight=2.0,
                 confidence_threshold=0.3, confidence_temperature=4.0, retrieval_mode="hybrid")  # fmt: skip
    r = make_retriever(s, b.store, bm25=b.bm25, dense=b.dense, reranker=FakeReranker())
    assert (r.candidate_k, r.rerank_k, r.rrf_k) == (7, 4, 10)
    assert (r.bm25_weight, r.dense_weight) == (0.5, 2.0)
    assert (r.confidence_threshold, r.confidence_temperature, r.mode) == (0.3, 4.0, "hybrid")
    assert (
        make_retriever(
            s, b.store, bm25=b.bm25, dense=b.dense, reranker=None, confidence_threshold=0.9
        ).confidence_threshold
        == 0.9
    )


def test_ablation_and_generation_share_confidence_temperature(
    fixture_docs: list[Document], tmp_path: Path
) -> None:
    import json

    from rbirag.config import Settings
    from rbirag.eval.ablate import run_retrieval_ablation
    from rbirag.eval.dataset import QAPair
    from rbirag.retrieval.confidence import sigmoid

    qa = [
        QAPair(
            qa_id="q",
            question="periodic updation of KYC for high risk customers",
            answer="",
            qtype="unanswerable",
            answerable=False,
        )
    ]
    path = tmp_path / "pq.jsonl"
    run_retrieval_ablation(fixture_docs, qa, settings=Settings(confidence_temperature=5.0), embedder=FakeEmbedder(32),
                           reranker=FakeReranker(), chunkers=["section"], sizes=[128], modes=["hybrid_rerank"],
                           per_query_path=path)  # fmt: skip
    item = json.loads(path.read_text().splitlines()[0])
    assert item["confidence"] == pytest.approx(sigmoid(item["top_score"] / 5.0))
