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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime
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
from rbirag.eval.records import all_verdicts, load_records

__all__ = ["load_records"]  # re-exported for callers of the old location
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
from rbirag.retrieval.pipeline import Retriever, make_retriever
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
    dense: bool = True,
) -> tuple[list[Chunk], IndexBundle]:
    """Chunk + index the corpus for one config; ``dense=False`` skips the (slow) dense
    index, e.g. for BM25-only retrieval."""
    s = settings_for(settings, size)
    chunks = chunk_documents(docs, build_chunker(chunker, s, embedder=embedder))
    bundle = build_index(
        chunks,
        embedder=embedder if dense else None,
        collection=collection or f"abl_{chunker}_{size}",
        batch_size=s.qdrant_upsert_batch_size,
    )
    return chunks, bundle


# ----------------------------------------------------------------------------- retrieval


class RetrievalQueryResult(BaseModel):
    config: str
    qa_id: str
    question: str | None = None  # the wording that was run (detects later edits)
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
            question=pair.question,
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
            retriever = make_retriever(
                settings, bundle.store, bm25=bundle.bm25, dense=bundle.dense, reranker=reranker
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
    # backends evaluated only on the best config and the subset (e.g. a free-tier API whose
    # daily token quota cannot cover the full test set)
    subset_only_backends: list[str] = []

    def items(self, backend: str | None = None) -> list[tuple[str, list[str]]]:
        return plan_items(self.model_dump(), backend)


def plan_items(plan: dict[str, Any], backend: str | None = None) -> list[tuple[str, list[str]]]:
    """(config, question ids) to run for ``backend`` according to a plan (dict form, as
    stored in plan.json). A subset-only backend gets the best config on the subset."""
    if backend is not None and backend in (plan.get("subset_only_backends") or []):
        return [(plan["full"][0], list(plan["subset_ids"]))] if plan["full"] else []
    return [(c, list(plan["all_ids"])) for c in plan["full"]] + [
        (c, list(plan["subset_ids"])) for c in plan["subset"]
    ]


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
    subset_only_backends: Sequence[str] = (),
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
        full=full, subset=subset, subset_ids=subset_ids, all_ids=[p.qa_id for p in qa],
        subset_only_backends=list(subset_only_backends),
    )  # fmt: skip


def planned_calls(
    plan: GenerationPlan, qa: Sequence[QAPair], backends: Sequence[str]
) -> dict[str, int]:
    """Upper bounds: one generation call per (config, question, backend) - the confidence
    gate can only reduce this - and one judge call per answerable answer."""
    answerable = {p.qa_id for p in qa if p.answerable}
    out: dict[str, int] = {}
    judge = 0
    for b in backends:
        items = plan.items(b)
        out[f"generate[{b}]"] = sum(len(ids) for _, ids in items)
        judge += sum(sum(1 for i in ids if i in answerable) for _, ids in items)
    out["judge (all backends)"] = judge
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


def needs_judge(rec: GenerationRecord, accepted_judges: AbstractSet[str] | None = None) -> bool:
    """An answer still needs a verdict: it was answered (not refused) for an answerable
    question and has no verdict yet (or only a quota error). With ``accepted_judges``, a
    verdict from any other judge does not count either: the answer is judged again (the
    old verdict is kept in ``verdicts``) so every reported number comes from one judge."""
    if rec.error is not None or not rec.answerable or rec.refused:
        return False
    if accepted_judges is not None and rec.judge_model not in accepted_judges:
        # unless an accepted judge already tried (a failed attempt is kept in verdicts)
        return not any(m in rec.verdicts for m in accepted_judges)
    return (
        rec.judge_faithfulness is None
        and rec.judge_relevance is None
        and (rec.judge_error is None or rec.judge_error.startswith("quota"))
    )


def _cosine(a: FloatArray, b: FloatArray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class JudgeOnlyLLM:
    """Stands in for the generator in judge-only runs; it must never be called."""

    def __init__(self, backend: str, model: str = "judge-only") -> None:
        self._backend = backend
        self._model = model

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model(self) -> str:
        return self._model

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("judge-only run tried to generate an answer")

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("judge-only run tried to generate an answer")


@dataclass
class GenerationRun:
    complete: bool
    generated: int
    judged: int
    skipped: int
    stopped_because: str | None = None
    judge_stopped: str | None = None
    judge_seconds: float = 0.0  # wall time spent in judge calls (seconds per verdict = /judged)


def run_generation_eval(
    pipeline: RAGPipeline,
    qa: Sequence[QAPair],
    *,
    config: str,
    records_path: Path,
    judge: LLM | None = None,
    nli: NLIScorer | None = None,
    embedder: Embedder | None = None,
    judge_only: bool = False,
    deadline: float | None = None,
    seed_from: Path | None = None,
    clock: Callable[[], float] = time.time,
    accepted_judges: AbstractSet[str] | None = None,
) -> GenerationRun:
    """Answer (and judge) each question once; resumable, quota- and time-aware.

    * ``judge_only``: never generate; questions without a usable record are skipped and
      only missing verdicts are produced (no generator server needed).
    * ``deadline`` (epoch seconds): stop cleanly before starting an item past it, so a CI
      job ends with its records flushed instead of being cancelled at its timeout.
    * ``seed_from``: also treat records in this file as existing (e.g. the committed
      canonical file when writing to a separate shard file), without rewriting them.
    * ``accepted_judges``: verdicts from other judges are redone by ``judge`` (and kept in
      the record's ``verdicts`` history).
    * If the *judge* runs out of quota, generation continues without verdicts (they are
      produced by a later judge run); if the *generator* runs out, the run stops.
    """
    existing = load_records(seed_from) if seed_from is not None else {}
    existing.update(load_records(records_path))
    backend = pipeline.llm.backend
    generated = judged = skipped = 0
    judge_stopped: str | None = None
    judge_seconds = 0.0

    def partial(reason: str) -> GenerationRun:
        return GenerationRun(
            complete=False, generated=generated, judged=judged, skipped=skipped,
            stopped_because=reason, judge_stopped=judge_stopped,
            judge_seconds=round(judge_seconds, 1),
        )  # fmt: skip

    for pair in qa:
        if deadline is not None and clock() >= deadline:
            logger.warning("time budget reached during %s/%s; stopping cleanly", config, backend)
            return partial("deadline")
        key = (config, backend, pair.qa_id)
        rec = existing.get(key)
        if rec is not None and rec.question != pair.question:
            rec = None  # the question was edited (e.g. by a review): answer it afresh
        if judge_only and (rec is None or rec.error is not None):
            skipped += 1
            continue
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
                    cached=ans.cached,
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
        except ConfigurationError:
            raise
        except (QuotaExhaustedError, LLMError) as exc:
            # The generator is out of quota or keeps failing: stop cleanly; everything
            # finished so far is saved and a rerun resumes.
            logger.warning("stopping %s/%s early: %s", config, backend, exc)
            return partial(type(exc).__name__)
        if judge is None or judge_stopped or not needs_judge(rec, accepted_judges):
            continue
        try:
            sources_text = _sources_for(pipeline, pair.question)
            started = time.perf_counter()
            jr = judge_answer(judge, pair.question, rec.answer, sources_text)
            judge_seconds += time.perf_counter() - started
        except ConfigurationError:
            raise
        except (QuotaExhaustedError, LLMError) as exc:
            # Verdicts can be produced later (judge-only runs); keep generating.
            judge_stopped = type(exc).__name__
            logger.warning("judge stopped (%s); continuing without verdicts", exc)
            if judge_only:
                return partial(judge_stopped)
            continue
        history = all_verdicts(rec)
        # a failed attempt is kept too (all None), so the item is not retried forever
        history[jr.model] = {
            "faithfulness": jr.verdict.faithfulness if jr.verdict else None,
            "relevance": jr.verdict.relevance_score if jr.verdict else None,
        }
        rec = rec.model_copy(
            update={
                "verdicts": history,
                "judge_model": jr.model,
                "judge_faithfulness": jr.verdict.faithfulness if jr.verdict else None,
                "judge_relevance": jr.verdict.relevance_score if jr.verdict else None,
                "judge_error": jr.error,
                "judge_cached": jr.cached,
                "judge_input_tokens": jr.usage.input_tokens,
                "judge_output_tokens": jr.usage.output_tokens,
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        judged += 1
        if judged == 1 or judged % 5 == 0:
            logger.info(
                "%s/%s: %d verdicts, %.1f s per verdict (judge %s)",
                config, backend, judged, judge_seconds / judged, jr.model,
            )  # fmt: skip
        existing[key] = rec
        append_jsonl(records_path, [rec.model_dump()])
    if judge_stopped:
        return GenerationRun(
            complete=False, generated=generated, judged=judged, skipped=skipped,
            stopped_because=f"judge: {judge_stopped}", judge_stopped=judge_stopped,
            judge_seconds=round(judge_seconds, 1),
        )  # fmt: skip
    return GenerationRun(
        complete=True, generated=generated, judged=judged, skipped=skipped,
        judge_seconds=round(judge_seconds, 1),
    )  # fmt: skip


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
