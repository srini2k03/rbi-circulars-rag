"""Ablations: retrieval over mode x chunker x size, and generation on selected configs.

Retrieval ablation needs no LLM. Generation evaluation runs the full test set on the best
retrieval config(s) and a fixed-size subset on a few contrasting configs, so a free-tier
API quota (and CPU time for the local model) stays manageable. Every stage is resumable:
per-query outputs are appended to JSONL and finished items are skipped on rerun.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
from pydantic import BaseModel

from rbirag.chunking import build_chunker
from rbirag.chunking.base import Chunk
from rbirag.config import ChunkerName, RetrievalMode, Settings
from rbirag.document import Document
from rbirag.eval.dataset import QAPair
from rbirag.eval.gen_metrics import GenerationRecord
from rbirag.eval.judge import judge_answer
from rbirag.eval.nli import NLIScorer, nli_check
from rbirag.eval.retrieval_metrics import DEFAULT_KS, query_metrics
from rbirag.eval.stats import bootstrap_ci
from rbirag.generation.citations import strip_citations
from rbirag.generation.llm import (
    LLM,
    ConfigurationError,
    LLMError,
    QuotaExhaustedError,
    RequestRejectedError,
)
from rbirag.generation.rag import RAGPipeline
from rbirag.index.build import IndexBundle, build_index, chunk_documents
from rbirag.index.embedder import Embedder, FloatArray
from rbirag.retrieval.pipeline import Retriever
from rbirag.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

MODES: tuple[str, ...] = get_args(RetrievalMode)
CHUNKERS: tuple[str, ...] = get_args(ChunkerName)
SIZES: tuple[int, ...] = (256, 512, 1024)


@dataclass(frozen=True)
class RetrievalConfig:
    chunker: str
    size: int
    mode: str

    @property
    def name(self) -> str:
        return f"{self.chunker}-{self.size}-{self.mode}"

    @classmethod
    def parse(cls, name: str) -> RetrievalConfig:
        chunker, size, mode = name.split("-", 2)
        return cls(chunker, int(size), mode)


def grid(
    chunkers: Sequence[str] = CHUNKERS, sizes: Sequence[int] = SIZES, modes: Sequence[str] = MODES
) -> list[RetrievalConfig]:
    return [RetrievalConfig(c, s, m) for c in chunkers for s in sizes for m in modes]


class CachingEmbedder:
    """Memoises document embeddings by text (sentence embeddings are reused across sizes)."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self._cache: dict[str, FloatArray] = {}

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self.inner.embed_documents(missing)
            for t, v in zip(missing, vecs, strict=True):
                self._cache[t] = v
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])

    def embed_query(self, text: str) -> FloatArray:
        return self.inner.embed_query(text)


def settings_for(settings: Settings, size: int) -> Settings:
    return settings.model_copy(
        update={
            "chunk_size": size,
            "chunk_overlap": max(0, size // 8),
            "section_min_tokens": max(16, size // 8),
        }
    )


def build_config_index(
    docs: Sequence[Document],
    chunker: str,
    size: int,
    settings: Settings,
    embedder: Embedder,
    *,
    collection: str | None = None,
) -> tuple[list[Chunk], IndexBundle]:
    s = settings_for(settings, size)
    chunks = chunk_documents(docs, build_chunker(chunker, s, embedder=embedder))
    bundle = build_index(
        chunks,
        embedder=embedder,
        collection=collection or f"abl_{chunker}_{size}",
        batch_size=s.qdrant_upsert_batch_size,
    )
    return chunks, bundle


# ----------------------------------------------------------------------------- retrieval


class RetrievalQueryResult(BaseModel):
    config: str
    qa_id: str
    answerable: bool
    retrieved: list[str]
    doc: dict[str, Any] | None = None
    chunk: dict[str, Any] | None = None
    confidence: float | None = None
    top_score: float | None = None
    latency_ms: float


def evaluate_retrieval(
    retriever: Retriever,
    config: RetrievalConfig,
    qa: Sequence[QAPair],
    *,
    k: int = 10,
) -> list[RetrievalQueryResult]:
    out: list[RetrievalQueryResult] = []
    mode: Any = config.mode
    for pair in qa:
        result = retriever.retrieve(pair.question, top_k=k, mode=mode)
        chunks = [r.chunk for r in result.results]
        conf = result.confidence
        item = RetrievalQueryResult(
            config=config.name,
            qa_id=pair.qa_id,
            answerable=pair.answerable,
            retrieved=[c.chunk_id for c in chunks],
            confidence=conf.probability if conf else None,
            top_score=conf.top_score if conf else None,
            latency_ms=result.timings_ms.get("total", 0.0),
        )
        if pair.answerable and pair.gold:
            item.doc = query_metrics(chunks, pair, level="doc").model_dump()
            item.chunk = query_metrics(chunks, pair, level="chunk").model_dump()
        out.append(item)
    return out


def summarize_retrieval(
    config: RetrievalConfig,
    results: Sequence[RetrievalQueryResult],
    *,
    n_chunks: int,
    n_resamples: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config": config.name,
        "chunker": config.chunker,
        "size": config.size,
        "mode": config.mode,
        "n_chunks": n_chunks,
    }
    scored = [r for r in results if r.doc is not None and r.chunk is not None]
    row["n_queries"] = len(scored)
    for level in ("doc", "chunk"):
        per = [getattr(r, level) for r in scored]
        for kk in DEFAULT_KS:
            est = bootstrap_ci(
                [p["recall"][kk] if kk in p["recall"] else p["recall"][str(kk)] for p in per],
                n_resamples=n_resamples,
                seed=seed,
            )
            row[f"{level}_recall@{kk}"] = est.mean
            row[f"{level}_recall@{kk}_ci"] = [est.low, est.high]
        for metric in ("mrr", "ndcg"):
            est = bootstrap_ci([p[metric] for p in per], n_resamples=n_resamples, seed=seed)
            row[f"{level}_{metric}@10"] = est.mean
            row[f"{level}_{metric}@10_ci"] = [est.low, est.high]
    lat = [r.latency_ms for r in results]
    row["latency_ms_p50"] = float(np.percentile(lat, 50)) if lat else None
    row["latency_ms_p95"] = float(np.percentile(lat, 95)) if lat else None
    return row


def run_retrieval_ablation(
    docs: Sequence[Document],
    qa: Sequence[QAPair],
    *,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker | None,
    chunkers: Sequence[str] = CHUNKERS,
    sizes: Sequence[int] = SIZES,
    modes: Sequence[str] = MODES,
    per_query_path: Path | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    cached_embedder = CachingEmbedder(embedder)
    rows: list[dict[str, Any]] = []
    for chunker in chunkers:
        for size in sizes:
            t0 = time.perf_counter()
            chunks, bundle = build_config_index(docs, chunker, size, settings, cached_embedder)
            logger.info(
                "built %s-%s: %d chunks in %.0fs",
                chunker,
                size,
                len(chunks),
                time.perf_counter() - t0,
            )
            retriever = Retriever(
                bundle.store,
                bm25=bundle.bm25,
                dense=bundle.dense,
                reranker=reranker,
                candidate_k=settings.candidate_k,
                rerank_k=settings.rerank_k,
                rrf_k=settings.rrf_k,
                bm25_weight=settings.bm25_weight,
                dense_weight=settings.dense_weight,
            )
            for mode in modes:
                if mode == "hybrid_rerank" and reranker is None:
                    continue
                config = RetrievalConfig(chunker, size, mode)
                results = evaluate_retrieval(retriever, config, qa)
                if per_query_path is not None:
                    append_jsonl(per_query_path, (r.model_dump() for r in results))
                row = summarize_retrieval(
                    config, results, n_chunks=len(chunks), n_resamples=settings.bootstrap_samples
                )
                rows.append(row)
                if on_row:
                    on_row(row)
            if bundle.dense is not None:
                bundle.dense.close()
    return rows


# ----------------------------------------------------------------------------- planning


class GenerationPlan(BaseModel):
    full: list[str]
    subset: list[str]
    subset_ids: list[str]
    all_ids: list[str]

    def items(self) -> list[tuple[str, list[str]]]:
        return [(c, self.all_ids) for c in self.full] + [(c, self.subset_ids) for c in self.subset]


def _stable_order(ids: Iterable[str], seed: int) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def plan_generation(
    rows: Sequence[dict[str, Any]],
    qa: Sequence[QAPair],
    *,
    full_configs: int = 1,
    subset_configs: int = 2,
    subset_size: int = 50,
    metric: str = "chunk_ndcg@10",
    seed: int = 13,
) -> GenerationPlan:
    """Best ``full_configs`` by ``metric`` on every question; the best config of other modes
    on a ``subset_size`` answerable subset (+ proportional unanswerable questions)."""
    ranked = sorted(rows, key=lambda r: (-(r.get(metric) or 0.0), r["config"]))
    full = [r["config"] for r in ranked[:full_configs]]
    used_modes = {RetrievalConfig.parse(c).mode for c in full}
    subset: list[str] = []
    for r in ranked:
        if len(subset) >= subset_configs:
            break
        mode = r["mode"]
        if r["config"] in full or mode in used_modes:
            continue
        subset.append(r["config"])
        used_modes.add(mode)
    answerable = _stable_order([p.qa_id for p in qa if p.answerable], seed)
    unanswerable = _stable_order([p.qa_id for p in qa if not p.answerable], seed)
    n_un = round(subset_size * len(unanswerable) / max(1, len(answerable)))
    subset_ids = answerable[:subset_size] + unanswerable[:n_un]
    return GenerationPlan(
        full=full, subset=subset, subset_ids=subset_ids, all_ids=[p.qa_id for p in qa]
    )


def planned_calls(
    plan: GenerationPlan, qa: Sequence[QAPair], backends: Sequence[str]
) -> dict[str, int]:
    """Upper bounds: one generation call per (config, question, backend) - the confidence
    gate can only reduce this - and one judge call per answerable answer."""
    answerable = {p.qa_id for p in qa if p.answerable}
    gen_per_backend = sum(len(ids) for _, ids in plan.items())
    judge_per_backend = sum(sum(1 for i in ids if i in answerable) for _, ids in plan.items())
    out = {f"generate[{b}]": gen_per_backend for b in backends}
    out["judge (all backends)"] = judge_per_backend * len(backends)
    out["api_calls_upper_bound"] = (
        sum(v for k, v in out.items() if k.startswith("generate[") and "ollama" not in k)
        + out["judge (all backends)"]
    )
    return out


# ----------------------------------------------------------------------------- generation


def append_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def load_records(path: Path) -> dict[tuple[str, str, str], GenerationRecord]:
    """Last record wins per (config, backend, qa_id)."""
    out: dict[tuple[str, str, str], GenerationRecord] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = GenerationRecord.model_validate_json(line)
            except ValueError:
                continue
            out[(rec.config, rec.backend, rec.qa_id)] = rec
    return out


def _needs_judge(rec: GenerationRecord, has_judge: bool) -> bool:
    return (
        has_judge
        and rec.error is None
        and rec.answerable
        and not rec.refused
        and rec.judge_faithfulness is None
        and rec.judge_relevance is None
        and (rec.judge_error is None or rec.judge_error.startswith("quota"))
    )


def _cosine(a: FloatArray, b: FloatArray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


@dataclass
class GenerationRun:
    complete: bool
    generated: int
    judged: int
    skipped: int
    stopped_because: str | None = None


def run_generation_eval(
    pipeline: RAGPipeline,
    qa: Sequence[QAPair],
    *,
    config: str,
    records_path: Path,
    judge: LLM | None = None,
    nli: NLIScorer | None = None,
    embedder: Embedder | None = None,
) -> GenerationRun:
    """Answer (and judge) each question once; resumable and quota-aware."""
    existing = load_records(records_path)
    backend = pipeline.llm.backend
    generated = judged = skipped = 0
    for pair in qa:
        key = (config, backend, pair.qa_id)
        rec = existing.get(key)
        try:
            if rec is None or rec.error is not None:
                try:
                    ans = pipeline.answer(pair.question)
                except RequestRejectedError as exc:  # per-item: record, retry on rerun
                    logger.warning("answer for %s rejected: %s", pair.qa_id, str(exc)[:160])
                    rec = _error_record(pair, config, backend, f"rejected: {str(exc)[:200]}")
                    existing[key] = rec
                    append_jsonl(records_path, [rec.model_dump()])
                    continue
                sources = [s.text for s in ans.sources]
                rec = GenerationRecord(
                    qa_id=pair.qa_id,
                    config=config,
                    backend=backend,
                    model=ans.model,
                    answerable=pair.answerable,
                    question=pair.question,
                    answer=ans.answer,
                    refused=ans.refused,
                    refusal_reason=ans.refusal_reason,
                    confidence=ans.confidence.probability if ans.confidence else None,
                    citation_valid=ans.citations.valid,
                    n_citations=len(ans.citations.cited_ids),
                    n_invalid_citations=len(ans.citations.invalid_ids),
                    n_uncited_sentences=len(ans.citations.uncited_sentences),
                    input_tokens=ans.usage.input_tokens + ans.usage.cache_read_input_tokens,
                    output_tokens=ans.usage.output_tokens,
                    cost_usd=ans.cost.usd,
                    latency_ms=ans.timings_ms,
                )
                if not ans.refused:
                    if nli is not None:
                        res = nli_check(nli, ans.citations, sources)
                        rec.nli_faithfulness = res.faithfulness
                        rec.citation_accuracy = res.citation_accuracy
                    if embedder is not None:
                        q = embedder.embed_query(pair.question)
                        a = embedder.embed_documents([strip_citations(ans.answer)])[0]
                        rec.emb_relevance = _cosine(q, a)
                generated += 1
                existing[key] = rec
                append_jsonl(records_path, [rec.model_dump()])
            else:
                skipped += 1
            if judge is not None and _needs_judge(rec, True):
                sources_text = _sources_for(pipeline, pair.question)
                jr = judge_answer(judge, pair.question, rec.answer, sources_text)
                rec = rec.model_copy(
                    update={
                        "judge_model": jr.model,
                        "judge_faithfulness": jr.verdict.faithfulness if jr.verdict else None,
                        "judge_relevance": jr.verdict.relevance_score if jr.verdict else None,
                        "judge_error": jr.error,
                        "judge_input_tokens": jr.usage.input_tokens,
                        "judge_output_tokens": jr.usage.output_tokens,
                    }
                )
                judged += 1
                existing[key] = rec
                append_jsonl(records_path, [rec.model_dump()])
        except ConfigurationError:
            raise
        except (QuotaExhaustedError, LLMError) as exc:
            # Quota, or the provider keeps failing (e.g. every model overloaded): stop
            # cleanly; everything finished so far is saved and a rerun resumes.
            logger.warning("stopping %s/%s early: %s", config, backend, exc)
            return GenerationRun(
                complete=False,
                generated=generated,
                judged=judged,
                skipped=skipped,
                stopped_because=type(exc).__name__,
            )
    return GenerationRun(complete=True, generated=generated, judged=judged, skipped=skipped)


def _error_record(pair: QAPair, config: str, backend: str, error: str) -> GenerationRecord:
    return GenerationRecord(
        qa_id=pair.qa_id, config=config, backend=backend, model=None,
        answerable=pair.answerable, question=pair.question, answer="", refused=False,
        refusal_reason=None, citation_valid=False, n_citations=0, n_invalid_citations=0,
        n_uncited_sentences=0, error=error,
    )  # fmt: skip


def _sources_for(pipeline: RAGPipeline, question: str) -> list[str]:
    """Retrieval is deterministic, so re-retrieving reproduces the sources that were shown."""
    result = pipeline.retriever.retrieve(question, top_k=pipeline.top_k, mode=pipeline.mode)
    return [r.chunk.text for r in result.results]
