"""The versioned API: ``/v1/...``.

Endpoints: ``ask`` (answer with verified citations), ``retrieve`` (inspect passages and
every stage score), ``citations/verify``, ``retrieval/compare``, ``eval/run`` (guarded),
``documents/{id}`` and ``documents/{id}/relations``, ``health`` and ``ready``.
Every POST is rate-limited per client; bodies are size-limited by the app middleware.
"""

from __future__ import annotations

import concurrent.futures
import hmac
import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from rbirag.api.logging import request_id_var
from rbirag.api.metrics import RequestLog, RequestLogEntry
from rbirag.api.ratelimit import RateLimiter
from rbirag.api.schemas import CostOut, Health, Usage
from rbirag.api.schemas_v1 import (
    AskRequest,
    AskResponse,
    CompareItem,
    CompareRequest,
    CompareResponse,
    ConfidenceOut,
    DocumentOut,
    EvalRunRequest,
    EvalRunResponse,
    FilterIn,
    PassageOut,
    Ready,
    RelationsOut,
    RetrieveRequest,
    RetrieveResponse,
    StrategyIn,
    VerifyRequest,
)
from rbirag.api.service import RAGService, ServiceUnavailableError, default_backend
from rbirag.config import Settings
from rbirag.generation.prompts import REFUSAL_TEXT
from rbirag.generation.rag import RAGAnswer, RAGPipeline, Source, to_sources
from rbirag.generation.verify import NOT_ESTABLISHED, SourceRef, verify_answer
from rbirag.query.understanding import QueryAnalysis, analyse_query
from rbirag.retrieval.confidence import Confidence
from rbirag.retrieval.params import RetrievalFilter

logger = logging.getLogger(__name__)

NEEDS_CLARIFICATION = (
    "The question is too unspecific to answer from the RBI notifications: {reasons}. "
    "Please name the regulation, the regulated entity or the topic."
)
OUT_OF_SCOPE = "This question is outside the RBI notifications indexed here: {reason}."
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="v1")


def _filter(f: FilterIn | None) -> RetrievalFilter | None:
    if f is None:
        return None
    flt = RetrievalFilter(
        departments=tuple(f.departments),
        categories=tuple(f.categories),
        date_from=f.date_from,
        date_to=f.date_to,
        doc_ids=tuple(f.doc_ids),
    )
    return None if flt.is_empty else flt


def _overrides(s: StrategyIn | None) -> dict[str, Any]:
    return {} if s is None else s.model_dump(exclude_none=True)


def _passages(sources: list[Source], cited: set[int]) -> list[PassageOut]:
    return [
        PassageOut(
            n=s.n,
            chunk_id=s.chunk_id,
            doc_id=s.doc_id,
            title=s.title,
            date=s.date,
            department=s.department,
            circular_number=s.circular_number,
            ref_number=s.ref_number,
            url=s.url,
            pdf_url=s.pdf_url,
            section=s.section,
            text=s.text,
            score=s.score,
            stage_scores=s.stage_scores,
            stage_ranks=s.stage_ranks,
            cited=s.n in cited,
        )
        for s in sources
    ]


def _confidence(c: Confidence | None) -> ConfidenceOut:
    if c is None:
        return ConfidenceOut()
    return ConfidenceOut(probability=c.probability, top_score=c.top_score, threshold=c.threshold)


def build_v1_router(
    svc: Callable[[], RAGService], log: RequestLog, settings: Settings
) -> APIRouter:
    router = APIRouter(prefix="/v1")
    limiter = RateLimiter(settings.api_rate_limit_per_minute)

    def rate_limited(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(client)
        if wait > 0:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(max(1, round(wait)))},
            )

    def rid() -> str:
        return request_id_var.get() or "-"

    def unavailable(exc: Exception) -> HTTPException:
        return HTTPException(status_code=503, detail=str(exc))

    # -- ops -------------------------------------------------------------------------
    @router.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        """Liveness plus what is loaded (no model or LLM call)."""
        return svc().health()

    @router.get("/ready", response_model=Ready, tags=["ops"])
    def ready() -> Ready:
        """Readiness: 200 when the index can serve queries, 503 otherwise."""
        service = svc()
        checks = {
            "index_loaded": service.bundle is not None,
            "bm25": service.bundle is not None and len(service.bundle.bm25) > 0,
            "dense_index": bool(service.bundle and service.bundle.dense is not None),
            "reranker": service.reranker is not None,
        }
        out = Ready(
            ready=checks["index_loaded"] and checks["bm25"],
            index_loaded=checks["index_loaded"],
            relation_graph=service.graph is not None,
            checks=checks,
        )
        if not out.ready:
            raise HTTPException(status_code=503, detail=out.model_dump())
        return out

    # -- ask -------------------------------------------------------------------------
    @router.post(
        "/ask", response_model=AskResponse, tags=["query"], dependencies=[Depends(rate_limited)]
    )
    def ask(req: AskRequest) -> AskResponse:
        """Answer with numbered citations that are verified against the sources. Returns
        the query analysis (type, ambiguity, follow-up rewrite, scope, suggested filters),
        every passage with its stage scores, the verification report, context checks and
        per-stage timings. Out-of-scope and unanswerably vague questions are refused
        before any retrieval or LLM call."""
        service = svc()
        started = time.perf_counter()
        analysis = analyse_query(req.question, history=req.history, vocabulary=service.vocabulary)
        early = _early_refusal(req, analysis)
        if early is not None:
            resp = _refusal_response(req, analysis, early, started)
            _log(log, "/v1/ask", resp)
            return resp
        flt = _filter(req.filters)
        if flt is None and req.apply_suggested_filters and analysis.filters:
            flt = RetrievalFilter.model_validate(analysis.filters)
        try:
            params = service.params_with(_overrides(req.strategy))
            with service.read():
                retriever = service.retriever_for(params)
                backend = req.backend or default_backend(settings)
                pipeline = RAGPipeline.from_settings(retriever, service.llm(backend), settings)
                pipeline.gate = req.gate and settings.confidence_gate
                pipeline.filters = flt
                pipeline.relation = service.relation
                if req.top_k:
                    pipeline.top_k = req.top_k
                if req.abstain_if_unsupported:
                    pipeline.verify_policy = "abstain"
                question = analysis.retrieval_query if req.use_rewrite else analysis.normalized
                future = _EXECUTOR.submit(pipeline.answer, question)
                answer = future.result(timeout=settings.api_timeout_seconds)
        except ServiceUnavailableError as exc:
            log.add(
                RequestLogEntry(
                    request_id=rid(),
                    endpoint="/v1/ask",
                    status=503,
                    error=str(exc),
                    component="index",
                )
            )
            raise unavailable(exc) from exc
        except concurrent.futures.TimeoutError as exc:
            log.add(
                RequestLogEntry(
                    request_id=rid(),
                    endpoint="/v1/ask",
                    status=504,
                    error="timeout",
                    component="generation",
                )
            )
            raise HTTPException(status_code=504, detail="answer timed out") from exc
        except ValueError as exc:  # invalid strategy combination
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.add(
                RequestLogEntry(
                    request_id=rid(),
                    endpoint="/v1/ask",
                    status=502,
                    error=str(exc)[:300],
                    component="generation",
                )
            )
            logger.exception("ask failed")
            raise HTTPException(
                status_code=502, detail=f"generation failed: {str(exc)[:200]}"
            ) from exc
        resp = _answer_response(req, analysis, answer, backend)
        _log(log, "/v1/ask", resp)
        return resp

    # -- retrieve --------------------------------------------------------------------
    @router.post(
        "/retrieve",
        response_model=RetrieveResponse,
        tags=["query"],
        dependencies=[Depends(rate_limited)],
    )
    def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        """The passages a strategy retrieves, with BM25 score, dense similarity, fusion
        rank, cross-encoder score and the query expansion that was applied (no LLM)."""
        service = svc()
        analysis = analyse_query(req.question, history=req.history, vocabulary=service.vocabulary)
        try:
            params = service.params_with(_overrides(req.strategy))
            with service.read():
                retriever = service.retriever_for(params)
                result = service.cached_retrieve(
                    retriever,
                    analysis.retrieval_query,
                    top_k=req.top_k,
                    filters=_filter(req.filters),
                )
        except ServiceUnavailableError as exc:
            raise unavailable(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RetrieveResponse(
            request_id=rid(),
            query=analysis,
            passages=_passages(to_sources(result.results), set()),
            retrieval=result.trace,
            confidence=_confidence(result.confidence),
            timings_ms=result.timings_ms,
        )

    # -- verify ----------------------------------------------------------------------
    @router.post("/citations/verify", tags=["query"], dependencies=[Depends(rate_limited)])
    def verify(req: VerifyRequest) -> dict[str, Any]:
        """Verify an answer's ``[n]`` citations against the given sources (texts, or chunk
        ids looked up in the index)."""
        service = svc()
        refs: list[SourceRef] = []
        for s in req.sources:
            text, doc_id = s.text, s.doc_id
            if text is None:
                if (
                    not s.chunk_id
                    or service.bundle is None
                    or s.chunk_id not in service.bundle.store
                ):
                    raise HTTPException(
                        status_code=422, detail=f"source {s.n}: give text or a known chunk_id"
                    )
                chunk = service.bundle.store.get(s.chunk_id)
                text, doc_id = chunk.text, chunk.doc_id
            refs.append(SourceRef(n=s.n, doc_id=doc_id or f"source-{s.n}", text=text))
        from rbirag.generation.rag import build_verifier

        scorer = build_verifier(
            settings.verifier if settings.verifier != "none" else "lexical", settings
        )
        assert scorer is not None
        report = verify_answer(
            req.answer, refs, scorer, threshold=req.threshold or settings.verify_threshold
        )
        return report.model_dump()

    # -- compare ---------------------------------------------------------------------
    @router.post(
        "/retrieval/compare",
        response_model=CompareResponse,
        tags=["query"],
        dependencies=[Depends(rate_limited)],
    )
    def compare(req: CompareRequest) -> CompareResponse:
        """Run up to four retrieval strategies on one question, side by side."""
        service = svc()
        items: list[CompareItem] = []
        ids: list[set[str]] = []
        with service.read():
            for strat in req.strategies:
                try:
                    params = service.params_with(_overrides(strat))
                    result = service.cached_retrieve(
                        service.retriever_for(params), req.question, top_k=req.top_k, filters=None
                    )
                except (ServiceUnavailableError, ValueError) as exc:
                    items.append(
                        CompareItem(
                            label=str(_overrides(strat)),
                            params_hash="-",
                            passages=[],
                            timings_ms={},
                            error=str(exc),
                        )
                    )
                    ids.append(set())
                    continue
                items.append(
                    CompareItem(
                        label=params.describe(),
                        params_hash=params.config_hash(),
                        passages=_passages(to_sources(result.results), set()),
                        timings_ms=result.timings_ms,
                    )
                )
                ids.append({r.chunk.chunk_id for r in result.results})
        overlap = [[round(len(a & b) / len(a | b), 3) if a | b else 0.0 for b in ids] for a in ids]
        return CompareResponse(request_id=rid(), results=items, overlap=overlap)

    # -- evaluation (guarded) --------------------------------------------------------
    @router.post(
        "/eval/run",
        response_model=EvalRunResponse,
        tags=["evaluation"],
        dependencies=[Depends(rate_limited)],
    )
    def eval_run(
        req: EvalRunRequest, x_eval_token: str | None = Header(default=None)
    ) -> EvalRunResponse:
        """Retrieval metrics of up to four strategies on the dev (or test) split, against
        the loaded index. Disabled unless RBIRAG_API_EVAL_TOKEN is set; the request must
        send it in ``X-Eval-Token``."""
        expected = settings.api_eval_token
        if not expected:
            raise HTTPException(status_code=403, detail="evaluation endpoint is disabled")
        if not x_eval_token or not hmac.compare_digest(x_eval_token, expected):
            raise HTTPException(status_code=401, detail="invalid evaluation token")
        return _run_eval(svc(), req, settings, rid())

    # -- documents -------------------------------------------------------------------
    def graph() -> Any:
        g = svc().graph
        if g is None:
            raise HTTPException(
                status_code=503, detail="relation graph not built; run `rbirag intel build`"
            )
        return g

    @router.get("/documents/{doc_id}", response_model=DocumentOut, tags=["documents"])
    def document(doc_id: str) -> DocumentOut:
        """Metadata, versions, original, the latest document *in the indexed corpus* (with
        its disclaimer), amendments and related documents."""
        g = graph()
        if doc_id not in g.intel:
            raise HTTPException(status_code=404, detail=f"unknown document {doc_id!r}")
        return DocumentOut(
            metadata=g.intel[doc_id],
            versions=g.versions(doc_id),
            original=g.original(doc_id),
            latest_in_corpus=g.latest_in_corpus(doc_id),
            changed_by=g.changed_by(doc_id),
            changes=g.changes(doc_id),
            consolidated_into=g.consolidated_into(doc_id),
            related=g.related(doc_id),
            unresolved_references=len(g.unresolved(doc_id)),
        )

    @router.get("/documents/{doc_id}/relations", response_model=RelationsOut, tags=["documents"])
    def relations(doc_id: str) -> RelationsOut:
        """Every resolved edge in and out of the document, and its unresolved references
        (cited documents outside the corpus or ambiguous: reported as unknown)."""
        g = graph()
        if doc_id not in g.intel:
            raise HTTPException(status_code=404, detail=f"unknown document {doc_id!r}")
        return RelationsOut(
            doc_id=doc_id,
            outgoing=g.out.get(doc_id, []),
            incoming=g.inc.get(doc_id, []),
            unresolved=[
                r.model_dump(include={"mention", "type", "resolution", "candidates"})
                for r in g.unresolved(doc_id)
            ][:100],
        )

    return router


# ----------------------------------------------------------------------------- helpers


def _early_refusal(req: AskRequest, a: QueryAnalysis) -> tuple[str, str] | None:
    vague = [r for r in a.ambiguity if r.startswith(("no specific subject", "too short"))]
    if vague:
        return "needs_clarification", NEEDS_CLARIFICATION.format(reasons="; ".join(vague))
    if req.refuse_out_of_scope and a.out_of_scope:
        return "out_of_scope", OUT_OF_SCOPE.format(reason=a.scope_reason)
    return None


def _refusal_response(
    req: AskRequest, a: QueryAnalysis, refusal: tuple[str, str], started: float
) -> AskResponse:
    return AskResponse(
        request_id=request_id_var.get() or "-",
        question=req.question,
        answer=refusal[1],
        refused=True,
        refusal_reason=refusal[0],
        citations=[],
        passages=[],
        query=a,
        confidence=ConfidenceOut(),
        mode="none",
        timings_ms={"total": round((time.perf_counter() - started) * 1000, 3)},
        usage=Usage(input_tokens=0, output_tokens=0),
        cost=CostOut(usd=0.0, inr=0.0),
    )


def _answer_response(
    req: AskRequest, a: QueryAnalysis, answer: RAGAnswer, backend: str
) -> AskResponse:
    cited = set(answer.citations.cited_ids)
    passages = _passages(answer.sources, cited)
    text = answer.answer
    if answer.refused and answer.refusal_reason == "unsupported":
        text = NOT_ESTABLISHED
    elif answer.refused and not text:
        text = REFUSAL_TEXT
    return AskResponse(
        request_id=request_id_var.get() or "-",
        question=req.question,
        answer=text,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citations=[p for p in passages if p.cited],
        passages=passages,
        verification=answer.verification,
        query=a,
        context=answer.context,
        retrieval=None,
        confidence=_confidence(answer.confidence),
        mode=answer.mode,
        backend=answer.backend or (None if answer.model is None else backend),
        model=answer.model,
        timings_ms=answer.timings_ms,
        usage=Usage(
            input_tokens=answer.usage.input_tokens + answer.usage.cache_read_input_tokens,
            output_tokens=answer.usage.output_tokens,
        ),
        cost=CostOut(usd=answer.cost.usd, inr=answer.cost.inr),
        cached=answer.cached,
    )


def _log(log: RequestLog, endpoint: str, resp: AskResponse) -> None:
    ver = resp.verification
    log.add(
        RequestLogEntry(
            request_id=resp.request_id,
            endpoint=endpoint,
            status=200,
            mode=resp.mode,
            backend=resp.backend,
            model=resp.model,
            refused=resp.refused,
            refusal_reason=resp.refusal_reason,
            timings_ms=resp.timings_ms,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=resp.cost.usd,
            cached=resp.cached,
            passages=len(resp.passages),
            distinct_documents=len({p.doc_id for p in resp.passages}),
            citations_accepted=sum(1 for c in ver.citations if c.relevant) if ver else None,
            verdict=ver.verdict if ver else None,
            retrieval_failed=(not resp.passages) if resp.mode != "none" else None,
        )
    )
    logger.info(
        "ask",
        extra={
            "request_id": resp.request_id,
            "refused": resp.refused,
            "refusal_reason": resp.refusal_reason,
            "mode": resp.mode,
            "verdict": ver.verdict if ver else None,
            "timings_ms": resp.timings_ms,
        },
    )


def _run_eval(
    service: RAGService, req: EvalRunRequest, settings: Settings, request_id: str
) -> EvalRunResponse:
    import numpy as np

    from rbirag.eval.cli import dev_test, load_qa
    from rbirag.eval.retrieval_metrics import query_metrics

    if service.bundle is None:
        raise HTTPException(status_code=503, detail="no index loaded")
    try:
        pairs = load_qa(settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail="no evaluation set available") from exc
    dev, test = dev_test(settings, pairs)
    docs = {c.doc_id for c in service.bundle.store}
    split = dev if req.split == "dev" else test
    qa = [p for p in split if p.answerable and p.gold and p.gold_doc_ids <= docs][: req.limit]
    rows: list[dict[str, Any]] = []
    with service.read():
        for strat in req.strategies:
            params = service.params_with(_overrides(strat))
            retriever = service.retriever_for(params)
            ndcg, recall, lat = [], [], []
            for p in qa:
                r = retriever.retrieve(p.question, top_k=10)
                chunks = [x.chunk for x in r.results]
                ndcg.append(query_metrics(chunks, p, level="chunk").ndcg)
                recall.append(query_metrics(chunks, p, level="doc").recall[5])
                lat.append(r.timings_ms.get("total", 0.0))
            rows.append(
                {
                    "strategy": params.describe(),
                    "params_hash": params.config_hash(),
                    "chunk_ndcg@10": float(np.mean(ndcg)) if ndcg else None,
                    "doc_recall@5": float(np.mean(recall)) if recall else None,
                    "latency_ms_p50": float(np.median(lat)) if lat else None,
                }
            )
    return EvalRunResponse(
        request_id=request_id,
        split=req.split,
        questions=len(qa),
        rows=rows,
        note="computed on the loaded index; the reported benchmark is results/benchmark",
    )
