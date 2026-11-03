"""FastAPI application: ``uvicorn --factory rbirag.api.main:create_app``."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from rbirag import __version__
from rbirag.api.logging import request_id_var
from rbirag.api.metrics import MetricsSummary, RequestLog, RequestLogEntry
from rbirag.api.schemas import Health, IngestRequest, Job, QueryRequest, QueryResponse
from rbirag.api.service import (
    RAGService,
    ServiceUnavailableError,
    response_log_fields,
    to_response,
)
from rbirag.config import Settings, get_settings

logger = logging.getLogger("rbirag.api")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], state="queued", created_at=_now())
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


def run_ingest(service: RAGService, jobs: JobRegistry, job_id: str, req: IngestRequest) -> None:
    jobs.update(job_id, state="running", started_at=_now())
    try:
        new, failed = service.collect(
            documents_path=req.documents_path, documents=req.documents, urls=req.urls
        )
        if not new:
            raise ValueError("no documents could be ingested from the request")
        corpus = service.merged(new, replace=req.replace)
        chunks = service.rebuild(corpus)
        jobs.update(
            job_id,
            state="succeeded",
            finished_at=_now(),
            documents_added=len(new),
            documents_total=len(corpus),
            chunks=chunks,
            failed_urls=failed,
        )
        logger.info(
            "ingest job finished", extra={"job_id": job_id, "documents": len(new), "chunks": chunks}
        )
    except Exception as exc:
        logger.exception("ingest job failed", extra={"job_id": job_id})
        jobs.update(job_id, state="failed", finished_at=_now(), error=str(exc)[:500])


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def create_app(
    settings: Settings | None = None,
    service: RAGService | None = None,
    request_log: RequestLog | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, RAGService] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            state["service"] = service
        else:
            svc = RAGService(settings)
            svc.load()
            state["service"] = svc
        yield
        state["service"].close()

    app = FastAPI(
        title="rbi-circulars-rag",
        version=__version__,
        description=(
            "Hybrid-retrieval RAG over Reserve Bank of India notifications. Answers cite "
            'numbered sources and say "I don\'t know" when retrieval confidence is low.'
        ),
        lifespan=lifespan,
    )
    log = request_log or RequestLog(settings.request_log_path)
    jobs = JobRegistry()

    def svc() -> RAGService:
        return state["service"]

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        logger.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    def _rid(request: Request) -> str:
        return request.headers.get("x-request-id") or request_id_var.get() or uuid.uuid4().hex[:16]

    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        """Index and model readiness (no LLM call is made)."""
        return svc().health()

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    def query(req: QueryRequest, request: Request) -> QueryResponse:
        """Answer a question with cited sources, or refuse with a reason."""
        rid = request_id_var.get() or _rid(request)
        try:
            answer, backend = svc().answer(req)
        except ServiceUnavailableError as exc:
            log.add(RequestLogEntry(request_id=rid, endpoint="/query", status=503, error=str(exc)))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            log.add(
                RequestLogEntry(request_id=rid, endpoint="/query", status=502, error=str(exc)[:300])
            )
            logger.exception("query failed")
            raise HTTPException(
                status_code=502, detail=f"generation failed: {str(exc)[:300]}"
            ) from exc
        resp = to_response(answer, rid, backend)
        fields = response_log_fields(resp)
        log.add(RequestLogEntry(request_id=rid, endpoint="/query", status=200, **fields))
        logger.info("query", extra={"request_id": rid, **fields})
        return resp

    @app.post(
        "/query/stream",
        tags=["query"],
        response_class=StreamingResponse,
        responses={
            200: {"content": {"text/event-stream": {}}, "description": "Server-sent events"}
        },
    )
    def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
        """Server-sent events: ``sources``, then ``token`` events, then ``done`` with the full
        response (same shape as ``/query``); ``error`` if generation fails mid-stream."""
        rid = request_id_var.get() or _rid(request)
        try:
            events, backend = svc().stream(req)
        except ServiceUnavailableError as exc:
            log.add(
                RequestLogEntry(
                    request_id=rid, endpoint="/query/stream", status=503, error=str(exc)
                )
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        def body() -> Iterator[str]:
            try:
                for event in events:
                    if event.type == "sources":
                        payload = [s.model_dump() for s in event.sources]
                        yield _sse("sources", {"request_id": rid, "sources": payload})
                    elif event.type == "delta":
                        yield _sse("token", {"text": event.text})
                    elif event.type == "done" and event.answer is not None:
                        resp = to_response(event.answer, rid, backend)
                        fields = response_log_fields(resp)
                        log.add(
                            RequestLogEntry(
                                request_id=rid, endpoint="/query/stream", status=200, **fields
                            )
                        )
                        logger.info("query", extra={"request_id": rid, "stream": True, **fields})
                        yield _sse("done", resp.model_dump())
            except Exception as exc:
                logger.exception("stream failed", extra={"request_id": rid})
                log.add(
                    RequestLogEntry(
                        request_id=rid, endpoint="/query/stream", status=502, error=str(exc)[:300]
                    )
                )
                yield _sse("error", {"request_id": rid, "detail": str(exc)[:300]})

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "x-request-id": rid},
        )

    @app.post("/ingest", response_model=Job, status_code=202, tags=["ingest"])
    def ingest(req: IngestRequest, background: BackgroundTasks) -> Job:
        """Add documents (a server-side JSONL, inline documents or RBI URLs) and rebuild the
        indexes in the background. Poll ``GET /ingest/{job_id}``."""
        if not (req.documents_path or req.documents or req.urls):
            raise HTTPException(status_code=422, detail="give documents_path, documents or urls")
        job = jobs.create()
        background.add_task(run_ingest, svc(), jobs, job.job_id, req)
        return job

    @app.get("/ingest/{job_id}", response_model=Job, tags=["ingest"])
    def ingest_status(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job

    @app.get("/metrics", response_model=MetricsSummary, tags=["ops"])
    def metrics() -> MetricsSummary:
        """Aggregate latency per stage, tokens and cost over the request log."""
        return log.summary()

    return app
