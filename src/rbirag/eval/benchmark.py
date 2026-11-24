"""Retrieval strategy benchmark: every strategy on the same held-out questions.

Strategies (each a :class:`~rbirag.retrieval.params.RetrievalParams` run through the real
:class:`~rbirag.retrieval.pipeline.Retriever`):

* first stage: BM25, BM25 + acronym expansion, BM25 + RM3, dense;
* fusion: RRF (original k=60, and k/weights tuned), weighted score fusion (min-max or
  z-score, weight tuned);
* reranking: BM25 -> cross-encoder and hybrid -> cross-encoder, depth and the
  cross-encoder/first-stage interpolation weight tuned;
* metadata-filtered retrieval: filters detected in the question, and oracle filters
  (the gold document's department / year: an upper bound, not a deployable system);
* parent-child: retrieve 256-token passages, return ~1024-token windows around them;
* document-first: rank whole documents, then passages inside the top-N documents.

Protocol: every tunable is chosen on the **dev** split (by chunk-level nDCG@10); the
chosen configuration of each family is then evaluated once on the **test** split, with
95% bootstrap CIs, a paired randomisation test against plain BM25 and Holm-adjusted
p-values. Dense and cross-encoder scores come from :mod:`rbirag.eval.score_cache` so the
whole comparison can be replayed without the models.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import logging
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from rbirag.chunking import build_chunker
from rbirag.chunking.base import Chunk
from rbirag.config import Settings
from rbirag.document import Document
from rbirag.eval.ablate import settings_for
from rbirag.eval.calibrate import calibrate_threshold, evaluate_threshold
from rbirag.eval.dataset import QAPair
from rbirag.eval.retrieval_metrics import context_coverage, first_hits, query_metrics
from rbirag.eval.score_cache import CachedDense, CachedReranker, ScoreCache
from rbirag.eval.stats import bootstrap_ci, holm_adjust, paired_comparison
from rbirag.index.bm25 import BM25Index, bm25_tokenize
from rbirag.index.store import ChunkStore
from rbirag.query.filters import detect_filters
from rbirag.retrieval.confidence import sigmoid
from rbirag.retrieval.expansion import AcronymExpander
from rbirag.retrieval.params import RetrievalFilter, RetrievalParams
from rbirag.retrieval.pipeline import DenseSearcher, RetrievalResult, Retriever
from rbirag.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

METRIC = "chunk_ndcg@10"  # dev selection metric (same as the original ablation)
EVAL_K = 10  # results retrieved per question for the metrics
CONTEXT_K = 5  # passages handed to the generator
POOL_K = 50  # first-stage depth used to tell "ranked low" from "never retrieved"
FilterMode = Literal["none", "auto", "oracle_department", "oracle_year"]


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    params: RetrievalParams
    index: Literal["primary", "child"] = "primary"
    filter_mode: FilterMode = "none"
    doc_first: int = 0  # >0: passages only from the top-N documents (document-level BM25)
    reference: bool = False  # an original configuration, reported as-is (not tuned)

    @property
    def label(self) -> str:
        extra = []
        if self.filter_mode != "none":
            extra.append(f"filter={self.filter_mode}")
        if self.doc_first:
            extra.append(f"doc-first top-{self.doc_first}")
        if self.index == "child":
            extra.append("256-token children")
        return self.params.describe() + (f" ({', '.join(extra)})" if extra else "")


# ----------------------------------------------------------------------------- grids


def strategy_grid(*, have_dense: bool, have_reranker: bool) -> list[Strategy]:
    """All configurations tried on dev, grouped into families. Within a family the first
    entry is the simplest; ties on dev keep the earlier (simpler) configuration."""
    P = RetrievalParams  # noqa: N806 (short alias keeps the grid readable)
    out: list[Strategy] = [Strategy("bm25", "bm25", P(mode="bm25"), reference=True)]
    out += [
        Strategy(f"bm25+acr w={w}", "bm25+acronyms", P(mode="bm25", expansion="acronyms",
                 acronym_weight=w))
        for w in (0.25, 0.5, 1.0)
    ]  # fmt: skip
    out += [
        Strategy(
            f"bm25+rm3 d={d} t={t} o={o}",
            "bm25+rm3",
            P(mode="bm25", expansion="rm3", rm3_fb_docs=d, rm3_fb_terms=t, rm3_orig_weight=o),
        )
        for d in (3, 5, 10)
        for t in (10, 20, 40)
        for o in (0.5, 0.7, 0.85)
    ]
    out += [
        Strategy(f"doc-first top-{n}", "doc-first", P(mode="bm25"), doc_first=n)
        for n in (3, 5, 10, 20)
    ]
    out += [
        Strategy("bm25 auto-filter", "filter:auto", P(mode="bm25"), filter_mode="auto"),
        Strategy("bm25 oracle-dept", "filter:oracle-department", P(mode="bm25"),
                 filter_mode="oracle_department"),
        Strategy("bm25 oracle-year", "filter:oracle-year", P(mode="bm25"),
                 filter_mode="oracle_year"),
    ]  # fmt: skip
    out += [
        Strategy(f"parent bm25 child256->{t}", "parent-child bm25",
                 P(mode="bm25", parent="window", parent_tokens=t), index="child")
        for t in (768, 1024)
    ]  # fmt: skip
    if have_dense:
        out.append(Strategy("dense", "dense", P(mode="dense"), reference=True))
        out.append(Strategy("hybrid rrf k=60", "hybrid rrf (original)", P(mode="hybrid"),
                            reference=True))  # fmt: skip
        out += [
            Strategy(f"hybrid rrf k={k} wb={w}", "hybrid rrf (tuned)",
                     P(mode="hybrid", rrf_k=k, bm25_weight=w, dense_weight=1.0))
            for k in (10, 30, 60, 100)
            for w in (1.0, 1.5, 2.0, 3.0)
        ]  # fmt: skip
        out += [
            Strategy(f"hybrid {norm} a={a:.1f}", "hybrid weighted",
                     P(mode="hybrid", fusion="weighted", score_norm=norm, bm25_weight=a,
                       dense_weight=round(1 - a, 2)))
            for norm in ("minmax", "zscore")
            for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ]  # fmt: skip
    if have_reranker:
        out += [
            Strategy(f"bm25+rerank d={d} w={w}", "bm25 + rerank",
                     P(mode="bm25_rerank", rerank_depth=d, rerank_weight=w))
            for d in (10, 20, 30, 50)
            for w in (1.0, 0.75, 0.5, 0.25)
        ]  # fmt: skip
        out += [
            Strategy(f"parent bm25+rerank child256->1024 w={w}", "parent-child bm25 + rerank",
                     P(mode="bm25_rerank", rerank_depth=20, rerank_weight=w, parent="window",
                       parent_tokens=1024), index="child")
            for w in (1.0, 0.5)
        ]  # fmt: skip
    if have_dense and have_reranker:
        out.append(Strategy("hybrid_rerank (original)", "hybrid rrf + rerank (original)",
                            P(mode="hybrid_rerank"), reference=True))  # fmt: skip
    return out


def derived_grid(selected: dict[str, Strategy], dev_scores: dict[str, float]) -> list[Strategy]:
    """Second-stage families built on the dev-selected first-stage configurations: the
    selected weighted fusion with reranking, and with the best query expansion (only if
    that expansion beat plain BM25 on dev)."""
    out: list[Strategy] = []
    fused = selected.get("hybrid weighted")
    if fused is None:
        return out
    base = fused.params
    if "bm25 + rerank" in selected:  # a reranker is available
        out += [
            Strategy(f"hybrid weighted+rerank d={d} w={w}", "hybrid weighted + rerank",
                     base.model_copy(update={"mode": "hybrid_rerank", "rerank_depth": d,
                                             "rerank_weight": w}))
            for d in (10, 20, 30, 50)
            for w in (1.0, 0.75, 0.5, 0.25)
        ]  # fmt: skip
    plain = dev_scores.get("bm25", 0.0)
    expansions = [selected[f] for f in ("bm25+acronyms", "bm25+rm3") if f in selected]
    best = max(expansions, key=lambda st: dev_scores[st.name], default=None)
    if best is not None and dev_scores[best.name] > plain:
        e = best.params
        out.append(
            Strategy(
                f"hybrid weighted+{e.expansion}",
                "hybrid weighted + expansion",
                base.model_copy(update={
                    "expansion": e.expansion, "acronym_weight": e.acronym_weight,
                    "rm3_fb_docs": e.rm3_fb_docs, "rm3_fb_terms": e.rm3_fb_terms,
                    "rm3_orig_weight": e.rm3_orig_weight,
                }),
            )
        )  # fmt: skip
    return out


# ----------------------------------------------------------------------------- results


class QueryResult(BaseModel):
    strategy: str
    family: str
    qa_id: str
    split: Literal["dev", "test"]
    answerable: bool
    retrieved: list[str]  # chunk ids, best first (top EVAL_K)
    retrieved_docs: list[str]
    doc: dict[str, Any] | None = None
    chunk: dict[str, Any] | None = None
    coverage: float | None = None  # character-level gold coverage of the top CONTEXT_K
    context_tokens: int = 0
    confidence: float | None = None
    category: str | None = None  # failure category (answerable questions)
    filter_applied: bool = False
    latency_ms: float = 0.0


@dataclass
class BenchIndex:
    chunker: str
    size: int
    chunks: list[Chunk]
    store: ChunkStore
    bm25: BM25Index
    cache: ScoreCache
    dense: DenseSearcher | None
    reranker: Reranker | None


@dataclass
class BenchContext:
    settings: Settings
    docs: list[Document]
    dev: list[QAPair]
    test: list[QAPair]
    primary: BenchIndex
    child: BenchIndex | None
    doc_bm25: BM25Index
    acronyms: AcronymExpander
    related: Callable[[str], set[str]]
    doc_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    pool: dict[str, list[Chunk]] = field(default_factory=dict)  # first-stage pools by qa_id
    selected: dict[str, Strategy] = field(default_factory=dict)  # set by run_benchmark
    test_results: dict[str, list[QueryResult]] = field(default_factory=dict)

    def split_of(self, qa_id: str) -> Literal["dev", "test"]:
        return "dev" if qa_id in {p.qa_id for p in self.dev} else "test"


def build_bench_index(
    docs: Sequence[Document],
    chunker: str,
    size: int,
    settings: Settings,
    *,
    cache: ScoreCache,
    live_dense: Callable[[list[Chunk]], DenseSearcher] | None,
    live_reranker: Reranker | None,
) -> BenchIndex:
    """Chunks + BM25 for one configuration; dense and reranker scores through the cache
    (the live dense index is only built if the cache misses)."""
    s = settings_for(settings, size)
    chunker_obj = build_chunker(chunker, s)
    chunks = [c for d in docs for c in chunker_obj.chunk(d)]
    store = ChunkStore(chunks)
    bm25 = BM25Index().build([c.chunk_id for c in chunks], [c.text for c in chunks])
    dense: DenseSearcher | None = None
    if live_dense is not None or cache.dense:
        dense = CachedDense(cache, _LazyDense(live_dense, chunks) if live_dense else None)
    reranker: Reranker | None = None
    if live_reranker is not None or cache.rerank:
        reranker = CachedReranker(cache, live_reranker)
    return BenchIndex(chunker, size, chunks, store, bm25, cache, dense, reranker)


class _LazyDense:
    """Builds the (slow) live dense index only on the first cache miss."""

    def __init__(self, factory: Callable[[list[Chunk]], DenseSearcher], chunks: list[Chunk]):
        self.factory, self.chunks = factory, chunks
        self._index: DenseSearcher | None = None

    def search(
        self, query: str, k: int = 10, *, flt: RetrievalFilter | None = None
    ) -> list[tuple[str, float]]:
        if self._index is None:
            logger.info("building the live dense index (%d chunks)", len(self.chunks))
            self._index = self.factory(self.chunks)
        return self._index.search(query, k, flt=flt)


def doc_index(docs: Sequence[Document]) -> BM25Index:
    """Document-level BM25 over title + full text."""
    return BM25Index().build([d.doc_id for d in docs], [f"{d.title}\n{d.text}" for d in docs])


def title_family(title: str) -> str:
    """Title with dates, years, version notes and numbering removed ("Master Direction -
    KYC Direction, 2016 (updated as on ...)" -> "master direction kyc direction")."""
    t = re.sub(r"\(updated as on[^)]*\)", " ", title, flags=re.IGNORECASE)
    t = re.sub(r"\b(19|20)\d{2}(-\d{2,4})?\b", " ", t)
    return " ".join(bm25_tokenize(t))


def title_relatedness(docs: Sequence[Document]) -> Callable[[str], set[str]]:
    """Documents in the same title family, or with title-token Jaccard >= 0.5. (Replaced
    by the citation graph when it is available; see :mod:`rbirag.intel`.)"""
    toks = {d.doc_id: set(title_family(d.title).split()) for d in docs}
    fam: dict[str, set[str]] = {}
    for d in docs:
        fam.setdefault(title_family(d.title), set()).add(d.doc_id)
    family_of = {d.doc_id: title_family(d.title) for d in docs}

    def related(doc_id: str) -> set[str]:
        out = set(fam.get(family_of.get(doc_id, ""), set()))
        mine = toks.get(doc_id, set())
        if mine:
            for other, t in toks.items():
                if t and len(mine & t) / len(mine | t) >= 0.5:
                    out.add(other)
        out.discard(doc_id)
        return out

    return related


# ----------------------------------------------------------------------------- running


def _filter_for(strategy: Strategy, pair: QAPair, ctx: BenchContext) -> RetrievalFilter | None:
    if strategy.filter_mode == "auto":
        return detect_filters(pair.question).filter
    if not pair.gold:
        return None
    meta = ctx.doc_meta.get(sorted(pair.gold_doc_ids)[0], {})
    if strategy.filter_mode == "oracle_department" and meta.get("department"):
        return RetrievalFilter(departments=(str(meta["department"]),))
    if strategy.filter_mode == "oracle_year" and meta.get("date"):
        year = int(str(meta["date"])[:4])
        return RetrievalFilter(date_from=dt.date(year, 1, 1), date_to=dt.date(year, 12, 31))
    return None


def run_strategy(
    strategy: Strategy, pairs: Sequence[QAPair], ctx: BenchContext
) -> list[QueryResult]:
    idx = ctx.child if strategy.index == "child" else ctx.primary
    if idx is None:
        raise ValueError(f"{strategy.name} needs the child index")
    retriever = Retriever(
        idx.store,
        bm25=idx.bm25,
        dense=idx.dense,
        reranker=idx.reranker,
        params=strategy.params,
        acronyms=ctx.acronyms,
    )
    out: list[QueryResult] = []
    for pair in pairs:
        flt = _filter_for(strategy, pair, ctx)
        if strategy.doc_first:
            top_docs = ctx.doc_bm25.search(pair.question, strategy.doc_first)
            ids = tuple(d for d, _ in top_docs)
            flt = RetrievalFilter(doc_ids=ids) if ids else flt
        t0 = time.perf_counter()
        result = retriever.retrieve(pair.question, top_k=EVAL_K, filters=flt)
        elapsed = (time.perf_counter() - t0) * 1000
        out.append(score_result(strategy, pair, result, ctx, flt is not None, elapsed))
    return out


def score_result(
    strategy: Strategy,
    pair: QAPair,
    result: RetrievalResult,
    ctx: BenchContext,
    filtered: bool,
    elapsed_ms: float,
) -> QueryResult:
    chunks = [r.chunk for r in result.results]
    conf = result.confidence
    qr = QueryResult(
        strategy=strategy.name,
        family=strategy.family,
        qa_id=pair.qa_id,
        split=ctx.split_of(pair.qa_id),
        answerable=pair.answerable,
        retrieved=[c.chunk_id for c in chunks],
        retrieved_docs=[c.doc_id for c in chunks],
        context_tokens=sum(c.n_tokens for c in chunks[:CONTEXT_K]),
        confidence=conf.probability if conf else None,
        filter_applied=filtered,
        latency_ms=round(elapsed_ms, 3),
    )
    if pair.answerable and pair.gold:
        qr.doc = query_metrics(chunks, pair, level="doc").model_dump()
        qr.chunk = query_metrics(chunks, pair, level="chunk").model_dump()
        qr.coverage = context_coverage(chunks[:CONTEXT_K], pair)
        qr.category = failure_category(chunks, pair, ctx)
    return qr


FAILURE_CATEGORIES: dict[str, str] = {
    "top1": "a gold passage is ranked first",
    "in_context": "a gold passage is in ranks 2-5 (inside the generator's context)",
    "ranked_low": "a gold passage is retrieved, but below the context cut-off (ranks 6-10) or "
    f"only in a first-stage top-{POOL_K} list",
    "right_doc_wrong_passage": "the gold document is in the context, but not the gold passage",
    "related_doc": "the context has a related document (same title family), not the gold one",
    "not_retrieved": "neither the gold passage nor its document (nor a related one) is found",
}


def failure_category(retrieved: Sequence[Chunk], pair: QAPair, ctx: BenchContext) -> str:
    ranks = [r for r in first_hits(retrieved, pair, "chunk") if r is not None]
    best = min(ranks) if ranks else None
    if best == 1:
        return "top1"
    if best is not None and best <= CONTEXT_K:
        return "in_context"
    if best is not None or _in_pool(pair, ctx):
        return "ranked_low"
    context_docs = {c.doc_id for c in retrieved[:CONTEXT_K]}
    if context_docs & pair.gold_doc_ids:
        return "right_doc_wrong_passage"
    related = set().union(*(ctx.related(d) for d in pair.gold_doc_ids)) if pair.gold else set()
    if context_docs & related:
        return "related_doc"
    return "not_retrieved"


def _in_pool(pair: QAPair, ctx: BenchContext) -> bool:
    """Whether a gold passage is in the plain BM25 or dense top-POOL_K of the primary index."""
    if pair.qa_id not in ctx.pool:
        idx = ctx.primary
        ids = [c for c, _ in idx.bm25.search(pair.question, POOL_K)]
        if idx.dense is not None:
            ids += [c for c, _ in idx.dense.search(pair.question, POOL_K)]
        ctx.pool[pair.qa_id] = [idx.store.get(c) for c in dict.fromkeys(ids)]
    return any(r is not None for r in first_hits(ctx.pool[pair.qa_id], pair, "chunk"))


# ----------------------------------------------------------------------------- aggregation


def metric_values(results: Iterable[QueryResult], metric: str) -> list[float]:
    """Per-question values of e.g. ``chunk_ndcg@10``, ``doc_recall@5``, ``chunk_precision@5``,
    ``chunk_context_precision``, ``coverage`` (answerable questions only)."""
    out: list[float] = []
    for r in results:
        if r.chunk is None or r.doc is None:
            continue
        if metric == "coverage":
            out.append(r.coverage or 0.0)
            continue
        level, name = metric.split("_", 1)
        m = r.doc if level == "doc" else r.chunk
        if name.startswith(("recall@", "precision@")):
            kind, k = name.split("@")
            out.append(float(m[kind][k] if k in m[kind] else m[kind][int(k)]))
        elif name == "context_precision":
            out.append(float(m["context_precision"] or 0.0))
        else:
            out.append(float(m[name.split("@")[0]]))
    return out


SUMMARY_METRICS = (
    "doc_recall@1", "doc_recall@5", "doc_recall@10",
    "chunk_recall@1", "chunk_recall@3", "chunk_recall@5", "chunk_recall@10",
    "chunk_precision@5", "chunk_context_precision", "chunk_mrr@10", "chunk_ndcg@10",
    "doc_ndcg@10", "coverage",
)  # fmt: skip


def summarize(results: Sequence[QueryResult], *, n_resamples: int, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {"n": sum(1 for r in results if r.chunk is not None)}
    for metric in SUMMARY_METRICS:
        est = bootstrap_ci(metric_values(results, metric), n_resamples=n_resamples, seed=seed)
        row[metric] = est.mean
        row[f"{metric}_ci"] = [est.low, est.high]
    answerable = [r for r in results if r.chunk is not None]
    row["context_tokens_mean"] = (
        float(np.mean([r.context_tokens for r in answerable])) if answerable else None
    )
    lat = [r.latency_ms for r in results]
    row["latency_ms_p50"] = float(np.percentile(lat, 50)) if lat else None
    row["latency_ms_p95"] = float(np.percentile(lat, 95)) if lat else None
    row["filter_applied"] = sum(r.filter_applied for r in results)
    row["categories"] = dict(Counter(r.category for r in answerable if r.category))
    return row


# ----------------------------------------------------------------------------- the benchmark


class BenchmarkReport(BaseModel):
    generated_at: str
    commit: str | None
    primary: str
    child: str | None
    n_dev: int
    n_test: int
    n_dev_answerable: int
    n_test_answerable: int
    selection_metric: str = METRIC
    tuning: list[dict[str, Any]] = Field(default_factory=list)  # every dev configuration
    rows: list[dict[str, Any]] = Field(default_factory=list)  # selected per family, on test
    recommended: str | None = None
    abstention: dict[str, Any] = Field(default_factory=dict)
    document_level: list[dict[str, Any]] = Field(default_factory=list)
    failure_categories: dict[str, str] = Field(default_factory=lambda: dict(FAILURE_CATEGORIES))
    notes: list[str] = Field(default_factory=list)
    cache_misses: int = 0
    live_timing: list[dict[str, Any]] = Field(default_factory=list)
    selected_strategies: list[str] = Field(default_factory=list)


def run_benchmark(
    ctx: BenchContext,
    *,
    strategies: Sequence[Strategy],
    per_query_path: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> BenchmarkReport:
    s = ctx.settings
    dev_ans = [p for p in ctx.dev if p.answerable and p.gold]
    report = BenchmarkReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=git_commit(),
        primary=f"{ctx.primary.chunker}-{ctx.primary.size} ({len(ctx.primary.chunks)} chunks)",
        child=f"{ctx.child.chunker}-{ctx.child.size} ({len(ctx.child.chunks)} chunks)"
        if ctx.child
        else None,
        n_dev=len(ctx.dev),
        n_test=len(ctx.test),
        n_dev_answerable=len(dev_ans),
        n_test_answerable=sum(1 for p in ctx.test if p.answerable and p.gold),
    )
    # 1. dev: every configuration of every family (then the derived second-stage families)
    dev_scores: dict[str, float] = {}
    tried: list[Strategy] = []

    def select() -> dict[str, Strategy]:
        chosen: dict[str, Strategy] = {}
        for st in tried:  # first (simplest) wins ties
            best = chosen.get(st.family)
            if best is None or dev_scores[st.name] > dev_scores[best.name] + 1e-12:
                chosen[st.family] = st
        return chosen

    queue = [st for st in strategies if not (st.index == "child" and ctx.child is None)]
    derived_done = False
    while queue:
        st = queue.pop(0)
        tried.append(st)
        res = run_strategy(st, dev_ans, ctx)
        vals = metric_values(res, METRIC)
        dev_scores[st.name] = float(np.mean(vals)) if vals else 0.0
        report.tuning.append(
            {
                "strategy": st.name,
                "family": st.family,
                "label": st.label,
                "params_hash": st.params.config_hash(),
                "params": st.params.model_dump(mode="json", exclude_defaults=True),
                "dev_" + METRIC: dev_scores[st.name],
                "dev_doc_recall@5": float(np.mean(metric_values(res, "doc_recall@5") or [0.0])),
            }
        )
        if on_progress:
            on_progress(f"dev {st.name}: {METRIC}={dev_scores[st.name]:.3f}")
        if not queue and not derived_done:
            derived_done = True
            queue = derived_grid(select(), dev_scores)
    # 2. select per family
    selected = select()
    # 3. test: selected configurations only
    all_pairs = list(ctx.test)
    per_query: dict[str, list[QueryResult]] = {}
    for family, st in selected.items():
        res = run_strategy(st, all_pairs, ctx)
        per_query[st.name] = res
        if per_query_path is not None:
            _append_jsonl(per_query_path, (r.model_dump() for r in res))
        if on_progress:
            on_progress(f"test {family}: {st.name}")
    baseline = per_query.get("bm25")
    rows: list[dict[str, Any]] = []
    for family, st in selected.items():
        res = per_query[st.name]
        row = {
            "family": family,
            "strategy": st.name,
            "label": st.label,
            "reference": st.reference,
            "params_hash": st.params.config_hash(),
            "params": st.params.model_dump(mode="json", exclude_defaults=True),
            "dev_" + METRIC: dev_scores[st.name],
            "tried_on_dev": sum(1 for t in tried if t.family == family),
            **summarize(res, n_resamples=s.bootstrap_samples, seed=s.seed),
        }
        if baseline is not None and st.name != "bm25":
            cmp = paired_comparison(
                metric_values(res, METRIC), metric_values(baseline, METRIC), seed=s.seed
            )
            row["vs_bm25"] = cmp.model_dump()
        rows.append(row)
    pvals = [r["vs_bm25"]["p_value"] for r in rows if "vs_bm25" in r]
    for r, adj in zip([r for r in rows if "vs_bm25" in r], holm_adjust(pvals), strict=True):
        r["vs_bm25"]["p_holm"] = adj
    rows.sort(key=lambda r: -r[METRIC])
    report.rows = rows
    deployable = [
        st for st in selected.values() if not st.filter_mode.startswith("oracle")
    ]  # oracle filters are upper bounds, not systems
    if deployable:
        report.recommended = max(deployable, key=lambda st: dev_scores[st.name]).name
    report.selected_strategies = [st.name for st in selected.values()]
    ctx.selected = selected
    ctx.test_results = per_query
    report.abstention = abstention_analysis(ctx)
    report.document_level = document_level(ctx)
    report.cache_misses = ctx.primary.cache.misses + (ctx.child.cache.misses if ctx.child else 0)
    return report


# ----------------------------------------------------------------------------- live timing


def live_timing(
    ctx: BenchContext,
    strategies: Sequence[Strategy],
    *,
    dense: dict[str, DenseSearcher | None],
    reranker: Reranker | None,
    cached: dict[str, list[QueryResult]],
) -> list[dict[str, Any]]:
    """Run each strategy with the *live* models over the test questions: wall-clock
    latency per stage (the cached run's timings exclude model time), and whether the live
    top-10 equals the cached top-10 (it must: same code, same scores up to rounding)."""
    rows: list[dict[str, Any]] = []
    for st in strategies:
        idx = ctx.child if st.index == "child" else ctx.primary
        if idx is None:
            continue
        live_dense = dense.get(st.index)
        if (st.params.uses_dense and live_dense is None) or (
            st.params.uses_reranker and reranker is None
        ):
            continue
        retriever = Retriever(
            idx.store, bm25=idx.bm25, dense=live_dense, reranker=reranker,
            params=st.params, acronyms=ctx.acronyms,
        )  # fmt: skip
        pairs = list(ctx.test)
        if pairs:  # warm up (model load, first-call overheads)
            retriever.retrieve(pairs[0].question, top_k=EVAL_K)
        reference = {r.qa_id: r.retrieved for r in cached.get(st.name, [])}
        stages: dict[str, list[float]] = {}
        same = compared = 0
        for pair in pairs:
            flt = _filter_for(st, pair, ctx)
            if st.doc_first:
                ids = tuple(d for d, _ in ctx.doc_bm25.search(pair.question, st.doc_first))
                flt = RetrievalFilter(doc_ids=ids) if ids else flt
            result = retriever.retrieve(pair.question, top_k=EVAL_K, filters=flt)
            for stage, ms in result.timings_ms.items():
                stages.setdefault(stage, []).append(ms)
            if pair.qa_id in reference:
                compared += 1
                same += [r.chunk.chunk_id for r in result.results] == reference[pair.qa_id]
        row: dict[str, Any] = {"strategy": st.name, "n": len(pairs)}
        for stage, values in sorted(stages.items()):
            row[f"{stage}_ms_p50"] = float(np.percentile(values, 50))
            row[f"{stage}_ms_p95"] = float(np.percentile(values, 95))
        row["identical_top10_vs_cached"] = f"{same}/{compared}"
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- abstention


def abstention_analysis(ctx: BenchContext) -> dict[str, Any]:
    """Can a retrieval signal tell unanswerable questions apart? For each signal: AUROC on
    test, and refusal precision/recall at the threshold chosen on dev (F0.5, as the gate)."""
    signals: dict[str, Callable[[QAPair], float | None]] = {}
    idx = ctx.primary

    def bm25_top(p: QAPair) -> float | None:
        hits = idx.bm25.search(p.question, 1)
        return hits[0][1] if hits else 0.0

    signals["bm25_top_score"] = bm25_top
    if idx.dense is not None:

        def dense_top(p: QAPair) -> float | None:
            assert idx.dense is not None
            hits = idx.dense.search(p.question, 1)
            return hits[0][1] if hits else 0.0

        signals["dense_top_cosine"] = dense_top
    if idx.reranker is not None:
        rr = idx.reranker

        def rerank_top(p: QAPair) -> float | None:
            pool = [idx.store.get(c).text for c, _ in idx.bm25.search(p.question, 20)]
            scores = rr.score(p.question, pool) if pool else []
            return sigmoid(max(scores)) if scores else 0.0

        signals["rerank_top_prob (bm25 top-20)"] = rerank_top
    out: dict[str, Any] = {
        "n_dev_unanswerable": sum(1 for p in ctx.dev if not p.answerable),
        "n_test_unanswerable": sum(1 for p in ctx.test if not p.answerable),
    }
    for name, fn in signals.items():
        dev_vals = [(fn(p) or 0.0, p.answerable) for p in ctx.dev]
        test_vals = [(fn(p) or 0.0, p.answerable) for p in ctx.test]
        lo = min(v for v, _ in dev_vals + test_vals)
        hi = max(v for v, _ in dev_vals + test_vals)
        span = (hi - lo) or 1.0
        dev_norm = [((v - lo) / span, a) for v, a in dev_vals]
        test_norm = [((v - lo) / span, a) for v, a in test_vals]
        cal = calibrate_threshold([v for v, _ in dev_norm], [a for _, a in dev_norm])
        point = evaluate_threshold(
            [v for v, _ in test_norm], [a for _, a in test_norm], cal.threshold
        )
        out[name] = {
            "auroc_test": auroc([v for v, _ in test_vals], [a for _, a in test_vals]),
            "threshold_raw": lo + cal.threshold * span,
            "test": point.model_dump(),
        }
    return out


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    """P(score of a random positive > score of a random negative), ties count half."""
    pos = [s for s, y in zip(scores, positive, strict=True) if y]
    neg = [s for s, y in zip(scores, positive, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# ----------------------------------------------------------------------------- document level


def document_level(ctx: BenchContext) -> list[dict[str, Any]]:
    """Document retrieval on test: passage BM25 aggregated to documents (first occurrence,
    i.e. max-passage) vs a document-level BM25 index."""
    pairs = [p for p in ctx.test if p.answerable and p.gold]
    rows: list[dict[str, Any]] = []
    rankers: dict[str, Callable[[QAPair], list[str]]] = {
        "passage BM25 -> documents (max-passage)": lambda p: list(
            dict.fromkeys(ctx.primary.store.get(c).doc_id
                          for c, _ in ctx.primary.bm25.search(p.question, 100))
        )[:10],
        "document-level BM25 (title + full text)": lambda p: [
            d for d, _ in ctx.doc_bm25.search(p.question, 10)
        ],
    }  # fmt: skip
    for name, rank_docs in rankers.items():
        recall: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
        rr: list[float] = []
        for p in pairs:
            docs = rank_docs(p)
            gold = p.gold_doc_ids
            for k, vals in recall.items():
                vals.append(len(gold & set(docs[:k])) / len(gold))
            first = next((i for i, d in enumerate(docs, 1) if d in gold), None)
            rr.append(1.0 / first if first else 0.0)
        row: dict[str, Any] = {"ranker": name, "n": len(pairs)}
        for k, vals in recall.items():
            est = bootstrap_ci(vals, n_resamples=ctx.settings.bootstrap_samples, seed=13)
            row[f"doc_recall@{k}"] = est.mean
            row[f"doc_recall@{k}_ci"] = [est.low, est.high]
        row["doc_mrr@10"] = float(np.mean(rr)) if rr else None
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- I/O


def _append_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener: Any = gzip.open if path.suffix == ".gz" else open
    with opener(path, "at", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    opener: Any = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_outputs(report: BenchmarkReport, out_dir: Path) -> None:
    """``strategies.json`` (everything), ``strategies.csv`` (test rows), ``tuning.csv``
    (every dev configuration) and ``strategies.md`` (tables for the README)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategies.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
    cols = ["family", "strategy", "label", "params_hash", "n", "dev_" + METRIC, *SUMMARY_METRICS,
            "context_tokens_mean", "latency_ms_p50", "latency_ms_p95"]  # fmt: skip
    lines = [",".join([*cols, "diff_vs_bm25", "p_holm"])]
    for r in report.rows:
        vals = [_csv_cell(r.get(c)) for c in cols]
        cmp = r.get("vs_bm25") or {}
        vals += [_csv_cell(cmp.get("diff")), _csv_cell(cmp.get("p_holm"))]
        lines.append(",".join(vals))
    (out_dir / "strategies.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tcols = ["family", "strategy", "params_hash", "dev_" + METRIC, "dev_doc_recall@5"]
    tl = [",".join(tcols)] + [",".join(_csv_cell(t.get(c)) for c in tcols) for t in report.tuning]
    (out_dir / "tuning.csv").write_text("\n".join(tl) + "\n", encoding="utf-8")
    (out_dir / "strategies.md").write_text(render_markdown(report), encoding="utf-8")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return f'"{text}"' if "," in text or '"' in text else text


def _ci(row: dict[str, Any], metric: str) -> str:
    lo, hi = row.get(f"{metric}_ci") or [float("nan")] * 2
    return f"{row[metric]:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown(report: BenchmarkReport) -> str:
    out = [
        f"Retrieval strategies on the **test** split ({report.n_test_answerable} answerable "
        f"questions; {report.n_dev_answerable} dev questions used for every choice). "
        f"Index: {report.primary}. Generated {report.generated_at} at commit "
        f"`{(report.commit or 'unknown')[:7]}`.",
        "",
        "| strategy (dev-selected) | dev nDCG@10 | test chunk nDCG@10 [95% CI] | chunk R@5 | "
        "P@5 | context prec. | coverage@5 | doc R@5 | Δ nDCG vs BM25 [95% CI] | p (Holm) | "
        "ctx tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rows:
        cmp = r.get("vs_bm25")
        delta = f"{cmp['diff']:+.3f} [{cmp['low']:+.3f}, {cmp['high']:+.3f}]" if cmp else "baseline"
        p = f"{cmp['p_holm']:.3f}" if cmp else "-"
        star = " (oracle)" if "oracle" in r["family"] else ""
        out.append(
            f"| {r['family']}{star}: `{r['label']}` | {r['dev_' + METRIC]:.3f} | "
            f"{_ci(r, METRIC)} | {r['chunk_recall@5']:.3f} | {r['chunk_precision@5']:.3f} | "
            f"{r['chunk_context_precision']:.3f} | {r['coverage']:.3f} | "
            f"{r['doc_recall@5']:.3f} | {delta} | {p} | {r['context_tokens_mean']:.0f} |"
        )
    if report.recommended:
        out += ["", f"Recommended (best deployable strategy on dev): `{report.recommended}`."]
    return "\n".join(out) + "\n"
