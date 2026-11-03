"""The :class:`Retriever`: bm25 | dense | hybrid | hybrid_rerank, with stage timings."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import get_args

from pydantic import BaseModel, Field

from rbirag.chunking.base import Chunk
from rbirag.config import RetrievalMode, Settings
from rbirag.index.bm25 import BM25Index
from rbirag.index.dense import QdrantIndex
from rbirag.index.store import ChunkStore
from rbirag.retrieval.confidence import Confidence, compute_confidence
from rbirag.retrieval.hybrid import DEFAULT_RRF_K, reciprocal_rank_fusion
from rbirag.retrieval.rerank import Reranker

MODES: tuple[str, ...] = get_args(RetrievalMode)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float  # the score that produced the final order (bm25 / cosine / rrf / rerank)
    rank: int
    stage_scores: dict[str, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    mode: str
    results: list[ScoredChunk]
    timings_ms: dict[str, float]
    confidence: Confidence | None = None


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        *,
        bm25: BM25Index | None = None,
        dense: QdrantIndex | None = None,
        reranker: Reranker | None = None,
        mode: RetrievalMode = "hybrid_rerank",
        candidate_k: int = 50,
        rerank_k: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        confidence_threshold: float = 0.5,
        confidence_temperature: float = 1.0,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.mode: RetrievalMode = mode
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.confidence_threshold = confidence_threshold
        self.confidence_temperature = confidence_temperature

    def _check(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode in {"bm25", "hybrid", "hybrid_rerank"} and self.bm25 is None:
            raise ValueError(f"mode {mode!r} needs a BM25 index")
        if mode in {"dense", "hybrid", "hybrid_rerank"} and self.dense is None:
            raise ValueError(f"mode {mode!r} needs a dense index")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("mode 'hybrid_rerank' needs a reranker")

    def retrieve(
        self, query: str, *, top_k: int = 5, mode: RetrievalMode | None = None
    ) -> RetrievalResult:
        mode = mode or self.mode
        self._check(mode)
        timings: dict[str, float] = {}

        @contextmanager
        def timed(stage: str) -> Iterator[None]:
            t0 = time.perf_counter()
            yield
            timings[stage] = round((time.perf_counter() - t0) * 1000, 3)

        depth = top_k if mode in {"bm25", "dense"} else max(self.candidate_k, top_k)
        stage_scores: dict[str, dict[str, float]] = {}
        sparse: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []
        with timed("total"):
            if mode != "dense":
                assert self.bm25 is not None
                with timed("bm25"):
                    sparse = self.bm25.search(query, depth)
                for cid, s in sparse:
                    stage_scores.setdefault(cid, {})["bm25"] = s
            if mode != "bm25":
                assert self.dense is not None
                with timed("dense"):
                    dense = self.dense.search(query, depth)
                for cid, s in dense:
                    stage_scores.setdefault(cid, {})["dense"] = s

            ranked: list[tuple[str, float]]
            if mode == "bm25":
                ranked = sparse
            elif mode == "dense":
                ranked = dense
            else:
                with timed("fusion"):
                    ranked = reciprocal_rank_fusion(
                        [[c for c, _ in sparse], [c for c, _ in dense]],
                        k=self.rrf_k,
                        weights=[self.bm25_weight, self.dense_weight],
                    )
                for cid, s in ranked:
                    stage_scores[cid]["rrf"] = s

            confidence: Confidence | None = None
            if mode == "hybrid_rerank":
                assert self.reranker is not None
                pool = [cid for cid, _ in ranked[: max(self.rerank_k, top_k)]]
                with timed("rerank"):
                    scores = self.reranker.score(query, [self.store.get(c).text for c in pool])
                for cid, s in zip(pool, scores, strict=True):
                    stage_scores[cid]["rerank"] = s
                ranked = sorted(zip(pool, scores, strict=True), key=lambda x: (-x[1], x[0]))
                confidence = compute_confidence(
                    scores,
                    threshold=self.confidence_threshold,
                    temperature=self.confidence_temperature,
                )

        results = [
            ScoredChunk(
                chunk=self.store.get(cid), score=score, rank=i, stage_scores=stage_scores[cid]
            )
            for i, (cid, score) in enumerate(ranked[:top_k], start=1)
            if cid in self.store
        ]
        return RetrievalResult(
            query=query, mode=mode, results=results, timings_ms=timings, confidence=confidence
        )


def make_retriever(
    settings: Settings,
    store: ChunkStore,
    *,
    bm25: BM25Index | None,
    dense: QdrantIndex | None,
    reranker: Reranker | None,
    mode: RetrievalMode | None = None,
    confidence_threshold: float | None = None,
) -> Retriever:
    """The one place a :class:`Retriever` is configured from settings (ablation, generation
    eval, API and CLI all use it), so fusion weights, candidate depths and the confidence
    temperature can never differ between the numbers reported and the system served."""
    return Retriever(
        store,
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        mode=mode or settings.retrieval_mode,
        candidate_k=settings.candidate_k,
        rerank_k=settings.rerank_k,
        rrf_k=settings.rrf_k,
        bm25_weight=settings.bm25_weight,
        dense_weight=settings.dense_weight,
        confidence_threshold=(
            settings.confidence_threshold if confidence_threshold is None else confidence_threshold
        ),
        confidence_temperature=settings.confidence_temperature,
    )
