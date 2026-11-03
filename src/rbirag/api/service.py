"""Application state for the API: indexes, models, LLM backends and ingestion."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from rbirag import __version__
from rbirag.api.schemas import (
    Backend,
    Citation,
    CostOut,
    Health,
    QueryRequest,
    QueryResponse,
    Usage,
)
from rbirag.chunking import build_chunker
from rbirag.config import RetrievalMode, Settings
from rbirag.document import Document, read_documents_jsonl, write_documents_jsonl
from rbirag.generation.llm import LLM
from rbirag.generation.rag import RAGAnswer, RAGPipeline, StreamEvent, build_llm
from rbirag.index.build import IndexBundle, build_index, chunk_documents, load_bundle, save_bundle
from rbirag.index.embedder import Embedder, build_embedder
from rbirag.providers import get_preset
from rbirag.retrieval.pipeline import Retriever
from rbirag.retrieval.rerank import Reranker, build_reranker

logger = logging.getLogger(__name__)

UrlFetcher = Callable[[Sequence[str]], tuple[list[Document], list[str]]]


class ServiceUnavailableError(RuntimeError):
    """The index is not loaded or the requested backend is not configured."""


def default_backend(settings: Settings) -> Backend:
    if settings.llm_backend == "openai_compatible":
        return "gemini" if settings.llm_provider == "gemini" else "groq"
    return settings.llm_backend


def backend_configured(name: str, settings: Settings) -> bool:
    """Whether a backend can be used: local ones always, API ones when their key is set."""
    if name in {"fake", "ollama"}:
        return True
    if name == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    preset = get_preset(name)
    env = (
        settings.oai_api_key_env
        if settings.llm_provider == name
        else (preset.api_key_env if preset else "")
    )
    return bool(env and os.environ.get(env, "").strip())


def settings_for_provider(settings: Settings, provider: str) -> Settings:
    """Settings re-resolved for another OpenAI-compatible provider preset."""
    if settings.llm_provider == provider:
        return settings
    data = settings.model_dump()
    data.update(
        llm_provider=provider,
        oai_base_url="",
        oai_api_key_env="",
        oai_model="",
        oai_model_fallbacks=["auto"],
        llm_rpm=0.0,
    )
    return Settings(**data)


def make_llm(name: str, settings: Settings) -> LLM:
    if name in {"groq", "gemini"}:
        return build_llm(settings_for_provider(settings, name), "openai_compatible")
    return build_llm(settings, name)


def fetch_rbi_urls(
    urls: Sequence[str], *, user_agent: str, rate_limit_seconds: float
) -> tuple[list[Document], list[str]]:
    """Fetch and parse RBI notification pages (politely: one request per interval)."""
    import httpx

    from rbirag.ingest.corpus import RawRecord, build_document

    docs: list[Document] = []
    failed: list[str] = []
    with httpx.Client(
        headers={"User-Agent": user_agent}, timeout=30, follow_redirects=True
    ) as client:
        for i, url in enumerate(urls):
            if i:
                time.sleep(rate_limit_seconds)
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("fetch failed for %s: %s", url, exc)
                failed.append(url)
                continue
            ids = parse_qs(urlsplit(url).query).get("Id") or parse_qs(urlsplit(url).query).get("id")
            source_id = int(ids[0]) if ids and ids[0].isdigit() else abs(hash(url)) % 10**9
            doc = build_document(RawRecord(source_id=source_id, url=url, html=resp.content))
            if doc is None:
                failed.append(url)
            else:
                docs.append(doc)
    return docs, failed


class ReadWriteGate:
    """Many concurrent readers (queries) or one writer (index swap).

    Unlike an ``RLock``, holding a read slot is not tied to a thread, which matters for
    streamed responses whose iterator is advanced from different worker threads.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writing = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writing:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            while self._writing:
                self._cond.wait()
            self._writing = True
            while self._readers:
                self._cond.wait()

    def release_write(self) -> None:
        with self._cond:
            self._writing = False
            self._cond.notify_all()


class RAGService:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        llm_factory: Callable[[str, Settings], LLM] = make_llm,
        url_fetcher: UrlFetcher | None = None,
        persist: bool = True,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(
            settings.embedder,
            model_name=settings.embedding_model,
            query_instruction=settings.embedding_query_instruction,
            batch_size=settings.embedding_batch_size,
            device=settings.device,
            fake_dim=settings.fake_embedding_dim,
        )
        self.reranker = (
            reranker
            if reranker is not None
            else build_reranker(
                settings.reranker, model_name=settings.reranker_model, device=settings.device
            )
        )
        self.llm_factory = llm_factory
        self.url_fetcher = url_fetcher
        self.persist = persist
        self.bundle: IndexBundle | None = None
        self.documents: dict[str, Document] = {}
        self._llms: dict[str, LLM] = {}
        self._llm_lock = threading.Lock()
        self._gate = ReadWriteGate()

    # -- index lifecycle ---------------------------------------------------------------
    @property
    def index_dir(self) -> Path:
        return self.settings.index_dir / self.settings.chunker

    @property
    def collection(self) -> str:
        return f"{self.settings.qdrant_collection}_{self.settings.chunker}"

    def _qdrant_path(self) -> Path | None:
        if self.settings.qdrant_url:
            return None
        return self.settings.qdrant_path or (self.index_dir / "qdrant" if self.persist else None)

    def load(self) -> bool:
        """Load a persisted index (and the documents it was built from) if present."""
        if self.settings.documents_path.exists():
            self.documents = {
                d.doc_id: d for d in read_documents_jsonl(self.settings.documents_path)
            }
        if not (self.index_dir / "chunks.jsonl").exists():
            logger.warning("no index at %s; POST /ingest to build one", self.index_dir)
            return False
        self._gate.acquire_write()
        try:
            self.bundle = load_bundle(
                self.index_dir,
                embedder=self.embedder,
                collection=self.collection,
                qdrant_url=self.settings.qdrant_url,
                qdrant_path=self._qdrant_path(),
            )
        finally:
            self._gate.release_write()
        logger.info("loaded index from %s (%d chunks)", self.index_dir, len(self.bundle.store))
        return True

    def rebuild(self, documents: Sequence[Document]) -> int:
        """Chunk + index ``documents`` and swap them in. Returns the chunk count."""
        chunker = build_chunker(self.settings.chunker, self.settings, embedder=self.embedder)
        chunks = chunk_documents(documents, chunker)
        self._gate.acquire_write()  # in-flight queries finish first; new ones wait
        try:
            if self.bundle is not None and self.bundle.dense is not None:
                self.bundle.dense.close()
            bundle = build_index(
                chunks,
                embedder=self.embedder,
                collection=self.collection,
                qdrant_url=self.settings.qdrant_url,
                qdrant_path=self._qdrant_path(),
                batch_size=self.settings.qdrant_upsert_batch_size,
            )
            self.bundle = bundle
            self.documents = {d.doc_id: d for d in documents}
            if self.persist:
                save_bundle(
                    bundle,
                    self.index_dir,
                    info={
                        "chunker": self.settings.chunker,
                        "documents": len(documents),
                        "chunks": len(chunks),
                    },
                )
                write_documents_jsonl(documents, self.settings.documents_path)
        finally:
            self._gate.release_write()
        return len(chunks)

    def close(self) -> None:
        self._gate.acquire_write()
        try:
            if self.bundle is not None and self.bundle.dense is not None:
                self.bundle.dense.close()
        finally:
            self._gate.release_write()

    # -- ingestion ---------------------------------------------------------------------
    def collect(
        self,
        *,
        documents_path: str | None,
        documents: Sequence[Document] | None,
        urls: Sequence[str] | None,
    ) -> tuple[list[Document], list[str]]:
        new: list[Document] = []
        failed: list[str] = []
        if documents_path:
            new.extend(read_documents_jsonl(Path(documents_path)))
        if documents:
            new.extend(documents)
        if urls:
            fetcher = self.url_fetcher or (
                lambda u: fetch_rbi_urls(
                    u,
                    user_agent=self.settings.user_agent,
                    rate_limit_seconds=self.settings.fetch_rate_limit_seconds,
                )
            )
            docs, failed = fetcher(urls)
            new.extend(docs)
        return new, failed

    def merged(self, new: Sequence[Document], *, replace: bool) -> list[Document]:
        corpus = {} if replace else dict(self.documents)
        for d in new:
            corpus[d.doc_id] = d
        return list(corpus.values())

    # -- querying ----------------------------------------------------------------------
    def llm(self, name: str) -> LLM:
        with self._llm_lock:
            if name not in self._llms:
                try:
                    self._llms[name] = self.llm_factory(name, self.settings)
                except Exception as exc:  # missing key, missing SDK, bad config
                    raise ServiceUnavailableError(
                        f"LLM backend {name!r} is not available: {exc}"
                    ) from exc
            return self._llms[name]

    def pipeline(self, req: QueryRequest) -> tuple[RAGPipeline, str]:
        if self.bundle is None:
            raise ServiceUnavailableError("no index loaded; POST /ingest first")
        mode: RetrievalMode = req.mode or self.settings.retrieval_mode
        bundle = self.bundle
        if mode != "bm25" and bundle.dense is None:
            raise ServiceUnavailableError("dense index unavailable; use mode=bm25")
        retriever = Retriever(
            bundle.store,
            bm25=bundle.bm25,
            dense=bundle.dense,
            reranker=self.reranker if mode == "hybrid_rerank" else None,
            mode=mode,
            candidate_k=self.settings.candidate_k,
            rerank_k=self.settings.rerank_k,
            rrf_k=self.settings.rrf_k,
            bm25_weight=self.settings.bm25_weight,
            dense_weight=self.settings.dense_weight,
            confidence_threshold=req.threshold
            if req.threshold is not None
            else self.settings.confidence_threshold,
            confidence_temperature=self.settings.confidence_temperature,
        )
        backend = req.backend or default_backend(self.settings)
        pipeline = RAGPipeline.from_settings(retriever, self.llm(backend), self.settings)
        pipeline.gate = req.gate and self.settings.confidence_gate
        if req.top_k:
            pipeline.top_k = req.top_k
        return pipeline, backend

    def answer(self, req: QueryRequest) -> tuple[RAGAnswer, str]:
        self._gate.acquire_read()
        try:
            pipeline, backend = self.pipeline(req)
            return pipeline.answer(req.question), backend
        finally:
            self._gate.release_read()

    def stream(self, req: QueryRequest) -> tuple[Iterator[StreamEvent], str]:
        """Validates (and fails) eagerly; the read slot is held while events are produced."""
        self._gate.acquire_read()
        try:
            pipeline, backend = self.pipeline(req)
        except Exception:
            self._gate.release_read()
            raise

        def events() -> Iterator[StreamEvent]:
            try:
                yield from pipeline.stream(req.question)
            finally:
                self._gate.release_read()

        return events(), backend

    # -- health ------------------------------------------------------------------------
    def health(self) -> Health:
        bundle = self.bundle
        backends = {
            b: backend_configured(b, self.settings)
            for b in ("groq", "gemini", "ollama", "anthropic", "fake")
        }
        loaded = bundle is not None
        return Health(
            status="ok" if loaded else "degraded",
            index_loaded=loaded,
            documents=len(self.documents)
            or (len({c.doc_id for c in bundle.store}) if bundle else 0),
            chunks=len(bundle.store) if bundle else 0,
            chunker=self.settings.chunker,
            dense_index=bool(bundle and bundle.dense is not None),
            embedder=self.embedder.name,
            reranker=getattr(self.reranker, "name", None),
            default_backend=default_backend(self.settings),
            backends=backends,
            version=__version__,
        )


def to_response(answer: RAGAnswer, request_id: str, backend: str) -> QueryResponse:
    cited = set(answer.citations.cited_ids)
    sources = [
        Citation(
            n=s.n,
            chunk_id=s.chunk_id,
            doc_id=s.doc_id,
            title=s.title,
            date=s.date,
            circular_number=s.circular_number,
            url=s.url,
            pdf_url=s.pdf_url,
            section=s.section,
            score=s.score,
            text=s.text,
            cited=s.n in cited,
        )
        for s in answer.sources
    ]
    return QueryResponse(
        request_id=request_id,
        question=answer.question,
        answer=answer.answer,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citations=[c for c in sources if c.cited],
        sources=sources,
        invalid_citations=answer.citations.invalid_ids,
        uncited_sentences=answer.citations.uncited_sentences,
        confidence=answer.confidence.probability if answer.confidence else None,
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


def response_log_fields(resp: QueryResponse) -> dict[str, Any]:
    return {
        "mode": resp.mode,
        "backend": resp.backend,
        "model": resp.model,
        "refused": resp.refused,
        "refusal_reason": resp.refusal_reason,
        "timings_ms": resp.timings_ms,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cost_usd": resp.cost.usd,
        "cached": resp.cached,
    }
