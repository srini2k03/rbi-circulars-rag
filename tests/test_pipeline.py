from __future__ import annotations

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
