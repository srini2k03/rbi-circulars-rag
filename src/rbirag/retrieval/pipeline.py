"""The :class:`Retriever`: BM25 / dense / hybrid (RRF or weighted) / cross-encoder reranking,
with query expansion, metadata filters, evidence thresholds, parent-context windows,
per-stage timings and a per-passage score trace.

Every knob lives in :class:`~rbirag.retrieval.params.RetrievalParams`; the benchmark
(:mod:`rbirag.eval.benchmark`) evaluates exactly these code paths.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, get_args, runtime_checkable

from pydantic import BaseModel, Field

from rbirag.chunking.base import Chunk
from rbirag.config import RetrievalMode, Settings
from rbirag.index.bm25 import BM25Index
from rbirag.index.store import ChunkStore
from rbirag.retrieval.confidence import Confidence, compute_confidence
from rbirag.retrieval.expansion import AcronymExpander, ExpandedQuery, expand_query
from rbirag.retrieval.fusion import interpolate, rank_map, weighted_fusion
from rbirag.retrieval.hybrid import reciprocal_rank_fusion
from rbirag.retrieval.params import RetrievalFilter, RetrievalParams
from rbirag.retrieval.parents import ParentResolver
from rbirag.retrieval.rerank import Reranker

MODES: tuple[str, ...] = get_args(RetrievalMode)


@runtime_checkable
class DenseSearcher(Protocol):
    """Anything that answers dense top-k queries (:class:`QdrantIndex`, or the benchmark's
    score cache)."""

    def search(
        self, query: str, k: int = 10, *, flt: RetrievalFilter | None = None
    ) -> list[tuple[str, float]]: ...


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float  # the score that produced the final order (bm25 / cosine / fused / rerank)
    rank: int
    # raw score per stage: bm25, dense (cosine), rrf or fused, rerank (cross-encoder logit)
    stage_scores: dict[str, float] = Field(default_factory=dict)
    # 1-based rank per stage (bm25, dense, fusion, rerank)
    stage_ranks: dict[str, int] = Field(default_factory=dict)
    # parent-context mode: the retrieved (child) passage the window was built around
    child_chunk_id: str | None = None


class RetrievalTrace(BaseModel):
    """What the retriever did, for inspection in the API / UI (no user text beyond the
    query terms)."""

    params_hash: str
    params: str  # RetrievalParams.describe()
    query_terms: list[str] = Field(default_factory=list)
    expansion: dict[str, float] = Field(default_factory=dict)  # added term -> weight
    expansion_notes: list[str] = Field(default_factory=list)
    filters: dict[str, Any] | None = None
    candidates: dict[str, int] = Field(default_factory=dict)  # per stage
    dropped: dict[str, int] = Field(default_factory=dict)  # by evidence thresholds


class RetrievalResult(BaseModel):
    query: str
    mode: str
    results: list[ScoredChunk]
    timings_ms: dict[str, float]
    confidence: Confidence | None = None
    trace: RetrievalTrace | None = None


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        *,
        bm25: BM25Index | None = None,
        dense: DenseSearcher | None = None,
        reranker: Reranker | None = None,
        params: RetrievalParams | None = None,
        acronyms: AcronymExpander | None = None,
        # legacy keyword arguments (kept for callers that predate RetrievalParams)
        mode: RetrievalMode | None = None,
        candidate_k: int | None = None,
        rerank_k: int | None = None,
        rrf_k: int | None = None,
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
        confidence_threshold: float | None = None,
        confidence_temperature: float | None = None,
    ) -> None:
        base = params or RetrievalParams()
        legacy: dict[str, Any] = {
            "mode": mode,
            "bm25_candidates": candidate_k,
            "dense_candidates": candidate_k,
            "rerank_depth": rerank_k,
            "rrf_k": rrf_k,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "confidence_threshold": confidence_threshold,
            "confidence_temperature": confidence_temperature,
        }
        update = {k: v for k, v in legacy.items() if v is not None}
        self.params = base.model_copy(update=update) if update else base
        RetrievalParams.model_validate(self.params.model_dump())  # re-validate overrides
        self.store = store
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self._acronyms = acronyms
        self._parents: ParentResolver | None = None
        self._masks: dict[RetrievalFilter, Any] = {}

    # -- configuration accessors (read by the API, the RAG pipeline and the CLI) ----------
    @property
    def mode(self) -> RetrievalMode:
        return self.params.mode

    @property
    def acronyms(self) -> AcronymExpander:
        if self._acronyms is None:
            self._acronyms = AcronymExpander.default()
        return self._acronyms

    def _check(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode in {"bm25", "bm25_rerank", "hybrid", "hybrid_rerank"} and self.bm25 is None:
            raise ValueError(f"mode {mode!r} needs a BM25 index")
        if mode in {"dense", "hybrid", "hybrid_rerank"} and self.dense is None:
            raise ValueError(f"mode {mode!r} needs a dense index")
        if mode in {"hybrid_rerank", "bm25_rerank"} and self.reranker is None:
            raise ValueError(f"mode {mode!r} needs a reranker")

    def _bm25_mask(self, flt: RetrievalFilter) -> Any:
        assert self.bm25 is not None
        if flt not in self._masks:
            self._masks[flt] = self.bm25.mask(lambda cid: flt.matches(self.store.get(cid).metadata))
        return self._masks[flt]

    def expand(self, query: str, params: RetrievalParams | None = None) -> ExpandedQuery:
        """The weighted BM25 query actually run (after the configured expansion)."""
        p = params or self.params
        assert self.bm25 is not None
        return expand_query(
            query,
            self.bm25,
            expansion=p.expansion,
            acronyms=self.acronyms if p.expansion.startswith("acronyms") else None,
            acronym_weight=p.acronym_weight,
            fb_docs=p.rm3_fb_docs,
            fb_terms=p.rm3_fb_terms,
            orig_weight=p.rm3_orig_weight,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode | None = None,
        filters: RetrievalFilter | None = None,
        params: RetrievalParams | None = None,
    ) -> RetrievalResult:
        p = params or self.params
        if mode is not None and mode != p.mode:
            p = p.model_copy(update={"mode": mode})
        self._check(p.mode)
        flt = filters if filters is not None and not filters.is_empty else None
        timings: dict[str, float] = {}

        @contextmanager
        def timed(stage: str) -> Iterator[None]:
            t0 = time.perf_counter()
            yield
            timings[stage] = round((time.perf_counter() - t0) * 1000, 3)

        trace = RetrievalTrace(
            params_hash=p.config_hash(),
            params=p.describe(),
            filters=flt.model_dump(mode="json", exclude_defaults=True) if flt else None,
        )
        single = p.mode in {"bm25", "dense"}
        stage_scores: dict[str, dict[str, float]] = {}
        stage_ranks: dict[str, dict[str, int]] = {}

        def note(stage: str, ranking: list[tuple[str, float]]) -> None:
            for rank, (cid, s) in enumerate(ranking, start=1):
                stage_scores.setdefault(cid, {})[stage] = s
                stage_ranks.setdefault(cid, {})[stage] = rank
            trace.candidates[stage] = len(ranking)

        sparse: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []
        confidence: Confidence | None = None
        with timed("total"):
            if p.uses_bm25:
                assert self.bm25 is not None
                depth = top_k if single else max(p.bm25_candidates, top_k)
                if p.mode == "bm25_rerank":
                    depth = max(p.bm25_candidates, p.rerank_depth, top_k)
                with timed("bm25"):
                    eq = self.expand(query, p)
                    mask = self._bm25_mask(flt) if flt else None
                    sparse = self.bm25.search_terms(eq.terms, depth, allowed=mask)
                trace.query_terms = eq.original
                trace.expansion = {t: round(w, 4) for t, w in eq.added.items()}
                trace.expansion_notes = eq.notes
                note("bm25", sparse)
            if p.uses_dense:
                assert self.dense is not None
                depth = top_k if single else max(p.dense_candidates, top_k)
                with timed("dense"):
                    dense = self.dense.search(query, depth, flt=flt)
                if p.dense_min_similarity is not None:
                    kept = [(c, s) for c, s in dense if s >= p.dense_min_similarity]
                    trace.dropped["dense_min_similarity"] = len(dense) - len(kept)
                    dense = kept
                note("dense", dense)

            ranked: list[tuple[str, float]]
            if p.mode in {"bm25", "bm25_rerank"}:
                ranked = sparse
            elif p.mode == "dense":
                ranked = dense
            else:
                with timed("fusion"):
                    if p.fusion == "rrf":
                        ranked = reciprocal_rank_fusion(
                            [[c for c, _ in sparse], [c for c, _ in dense]],
                            k=p.rrf_k,
                            weights=[p.bm25_weight, p.dense_weight],
                        )
                    else:
                        ranked = weighted_fusion(
                            [sparse, dense], [p.bm25_weight, p.dense_weight], method=p.score_norm
                        )
                for cid, s in ranked:
                    stage_scores[cid]["rrf" if p.fusion == "rrf" else "fused"] = s
                for cid, r in rank_map(ranked).items():
                    stage_ranks[cid]["fusion"] = r
                trace.candidates["fusion"] = len(ranked)

            if p.uses_reranker:
                assert self.reranker is not None
                pool = ranked[: max(p.rerank_depth, top_k)]
                with timed("rerank"):
                    scores = self.reranker.score(query, [self.store.get(c).text for c, _ in pool])
                rr = {cid: s for (cid, _), s in zip(pool, scores, strict=True)}
                for cid, s in rr.items():
                    stage_scores[cid]["rerank"] = s
                by_rerank = sorted(rr.items(), key=lambda kv: (-kv[1], kv[0]))
                for cid, r in rank_map(by_rerank).items():
                    stage_ranks[cid]["rerank"] = r
                trace.candidates["rerank"] = len(pool)
                ranked = interpolate(pool, rr, rerank_weight=p.rerank_weight)
                confidence = compute_confidence(
                    scores,
                    threshold=p.confidence_threshold,
                    temperature=p.confidence_temperature,
                )
                if p.rerank_min_score is not None:
                    kept = [(c, s) for c, s in ranked if rr[c] >= p.rerank_min_score]
                    trace.dropped["rerank_min_score"] = len(ranked) - len(kept)
                    ranked = kept

        # parent windows can merge several hits into one, so look further down the list
        keep = top_k * 4 if p.parent == "window" else top_k
        results = [
            ScoredChunk(
                chunk=self.store.get(cid),
                score=score,
                rank=i,
                stage_scores=stage_scores.get(cid, {}),
                stage_ranks=stage_ranks.get(cid, {}),
            )
            for i, (cid, score) in enumerate(ranked[:keep], start=1)
            if cid in self.store
        ]
        if p.parent == "window" and results:
            with timed("parent"):
                results = self.parents.expand(results, p.parent_tokens)[:top_k]
        return RetrievalResult(
            query=query,
            mode=p.mode,
            results=results,
            timings_ms=timings,
            confidence=confidence,
            trace=trace,
        )

    @property
    def parents(self) -> ParentResolver:
        if self._parents is None:
            self._parents = ParentResolver(self.store)
        return self._parents


def make_retriever(
    settings: Settings,
    store: ChunkStore,
    *,
    bm25: BM25Index | None,
    dense: DenseSearcher | None,
    reranker: Reranker | None,
    mode: RetrievalMode | None = None,
    confidence_threshold: float | None = None,
    params: RetrievalParams | None = None,
) -> Retriever:
    """The one place a :class:`Retriever` is configured from settings (ablation, generation
    eval, API and CLI all use it), so fusion weights, candidate depths and the confidence
    temperature can never differ between the numbers reported and the system served."""
    p = params or RetrievalParams.from_settings(settings)
    update: dict[str, Any] = {}
    if mode is not None:
        update["mode"] = mode
    if confidence_threshold is not None:
        update["confidence_threshold"] = confidence_threshold
    return Retriever(
        store,
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        params=p.model_copy(update=update) if update else p,
    )
