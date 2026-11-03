"""End-to-end RAG: retrieve -> confidence gate -> prompt -> LLM -> citation validation."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from typing import Literal

from pydantic import BaseModel, Field

from rbirag.config import ModelPrice, RetrievalMode, Settings
from rbirag.generation.citations import CitationReport, parse_citations
from rbirag.generation.cost import Cost, compute_cost
from rbirag.generation.limits import RequestLimiter
from rbirag.generation.llm import LLM, LLMResponse, TokenUsage
from rbirag.generation.prompts import REFUSAL_TEXT, SYSTEM_PROMPT, build_messages
from rbirag.retrieval.confidence import Confidence
from rbirag.retrieval.pipeline import RetrievalResult, Retriever, ScoredChunk

logger = logging.getLogger(__name__)

RefusalReason = Literal["low_confidence", "no_sources", "model_refusal", "model_declined"]


class Source(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    title: str | None
    url: str | None
    date: str | None
    circular_number: str | None
    score: float
    text: str


class RAGAnswer(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: RefusalReason | None = None
    citations: CitationReport
    sources: list[Source]
    confidence: Confidence | None
    mode: str
    model: str | None
    backend: str | None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost: Cost = Field(default_factory=lambda: Cost(usd=0.0, inr=0.0, priced=True))
    timings_ms: dict[str, float] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    type: Literal["sources", "delta", "done"]
    text: str = ""
    sources: list[Source] = Field(default_factory=list)
    answer: RAGAnswer | None = None


def to_sources(results: list[ScoredChunk]) -> list[Source]:
    out: list[Source] = []
    for i, r in enumerate(results, start=1):
        md = r.chunk.metadata
        out.append(
            Source(
                n=i,
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                title=_s(md.get("title")),
                url=_s(md.get("url")),
                date=_s(md.get("date")),
                circular_number=_s(md.get("circular_number")),
                score=r.score,
                text=r.chunk.text,
            )
        )
    return out


def _s(value: object) -> str | None:
    return None if value is None else str(value)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        *,
        top_k: int = 5,
        mode: RetrievalMode | None = None,
        gate: bool = True,
        max_tokens: int = 1024,
        prices: Mapping[str, ModelPrice] | None = None,
        usd_to_inr: float = 88.0,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.mode: RetrievalMode = mode or retriever.mode
        self.gate = gate
        self.max_tokens = max_tokens
        self.prices = dict(prices or {})
        self.usd_to_inr = usd_to_inr
        self.system_prompt = system_prompt

    @classmethod
    def from_settings(cls, retriever: Retriever, llm: LLM, settings: Settings) -> RAGPipeline:
        return cls(
            retriever,
            llm,
            top_k=settings.answer_top_k,
            gate=settings.confidence_gate,
            max_tokens=settings.max_answer_tokens,
            prices=settings.prices,
            usd_to_inr=settings.usd_to_inr,
        )

    # -- helpers -----------------------------------------------------------------------
    def _retrieve(self, question: str) -> RetrievalResult:
        return self.retriever.retrieve(question, top_k=self.top_k, mode=self.mode)

    def _gate_reason(self, retrieval: RetrievalResult) -> RefusalReason | None:
        if not retrieval.results:
            return "no_sources"
        conf = retrieval.confidence
        if self.gate and conf is not None and not conf.is_confident:
            return "low_confidence"
        return None

    def _refusal(
        self,
        question: str,
        retrieval: RetrievalResult,
        reason: RefusalReason,
        sources: list[Source],
    ) -> RAGAnswer:
        return RAGAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            refused=True,
            refusal_reason=reason,
            citations=parse_citations(REFUSAL_TEXT, len(sources)),
            sources=sources,
            confidence=retrieval.confidence,
            mode=retrieval.mode,
            model=None,
            backend=None,
            timings_ms=self._stage_timings(retrieval, None),
        )

    @staticmethod
    def _stage_timings(retrieval: RetrievalResult, generate_ms: float | None) -> dict[str, float]:
        t = retrieval.timings_ms
        rerank = t.get("rerank", 0.0)
        timings = {"retrieve": round(t.get("total", 0.0) - rerank, 3), "rerank": rerank}
        if generate_ms is not None:
            timings["generate"] = generate_ms
        return timings

    def _finish(
        self,
        question: str,
        retrieval: RetrievalResult,
        sources: list[Source],
        response: LLMResponse,
        generate_ms: float,
    ) -> RAGAnswer:
        timings = self._stage_timings(retrieval, generate_ms)
        timings["total"] = round(sum(timings.values()), 3)
        cost = compute_cost(
            response.usage,
            model=response.model,
            backend=response.backend,
            prices=self.prices,
            usd_to_inr=self.usd_to_inr,
        )
        if not cost.priced:
            logger.warning("no price for model %s; cost reported as 0", response.model)
        text = response.text.strip()
        reason: RefusalReason | None = None
        if response.refused:
            text, reason = REFUSAL_TEXT, "model_declined"
        report = parse_citations(text, len(sources))
        if report.is_refusal and reason is None:
            reason = "model_refusal"
        return RAGAnswer(
            question=question,
            answer=text,
            refused=report.is_refusal or response.refused,
            refusal_reason=reason,
            citations=report,
            sources=sources,
            confidence=retrieval.confidence,
            mode=retrieval.mode,
            model=response.model,
            backend=response.backend,
            usage=response.usage,
            cost=cost,
            timings_ms=timings,
        )

    # -- public API ----------------------------------------------------------------------
    def answer(self, question: str) -> RAGAnswer:
        retrieval = self._retrieve(question)
        sources = to_sources(retrieval.results)
        reason = self._gate_reason(retrieval)
        if reason is not None:
            return self._refusal(question, retrieval, reason, sources)
        started = time.perf_counter()
        response = self.llm.generate(
            build_messages(question, [r.chunk for r in retrieval.results]),
            system=self.system_prompt,
            max_tokens=self.max_tokens,
        )
        generate_ms = round((time.perf_counter() - started) * 1000, 3)
        return self._finish(question, retrieval, sources, response, generate_ms)

    def stream(self, question: str) -> Iterator[StreamEvent]:
        """Yields ``sources`` first, then ``delta`` text events, then ``done`` with the answer."""
        retrieval = self._retrieve(question)
        sources = to_sources(retrieval.results)
        yield StreamEvent(type="sources", sources=sources)
        reason = self._gate_reason(retrieval)
        if reason is not None:
            answer = self._refusal(question, retrieval, reason, sources)
            yield StreamEvent(type="delta", text=answer.answer)
            yield StreamEvent(type="done", answer=answer)
            return
        started = time.perf_counter()
        final: LLMResponse | None = None
        for chunk in self.llm.stream(
            build_messages(question, [r.chunk for r in retrieval.results]),
            system=self.system_prompt,
            max_tokens=self.max_tokens,
        ):
            if chunk.text:
                yield StreamEvent(type="delta", text=chunk.text)
            if chunk.final is not None:
                final = chunk.final
        if final is None:
            raise RuntimeError("LLM stream ended without a final response")
        generate_ms = round((time.perf_counter() - started) * 1000, 3)
        yield StreamEvent(
            type="done", answer=self._finish(question, retrieval, sources, final, generate_ms)
        )


_LIMITERS: dict[str, RequestLimiter] = {}


def shared_limiter(settings: Settings) -> RequestLimiter:
    """One limiter per state directory, so all LLMs in a process share the budgets."""
    state_dir = settings.llm_cache_dir / "quota"
    key = str(state_dir.resolve())
    if key not in _LIMITERS:
        _LIMITERS[key] = RequestLimiter(
            rpm=settings.llm_rpm, rpd=settings.llm_rpd, state_dir=state_dir
        )
    return _LIMITERS[key]


def build_llm(
    settings: Settings,
    backend: str | None = None,
    model: str | None = None,
    *,
    cache: bool | None = None,
) -> LLM:
    """LLM for ``backend`` (default ``settings.llm_backend``), wrapped in the disk cache."""
    backend = backend or settings.llm_backend
    llm: LLM
    if backend == "openai_compatible":
        from rbirag.generation.openai_compat import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(
            model or settings.oai_model,
            base_url=settings.oai_base_url,
            api_key_env=settings.oai_api_key_env,
            reasoning_effort=settings.oai_reasoning_effort,
            json_mode=settings.oai_json_mode,
            stream_usage=settings.oai_stream_usage,
            timeout_seconds=settings.oai_timeout_seconds,
            limiter=shared_limiter(settings),
            max_retries=settings.llm_max_retries,
            backoff_base_seconds=settings.llm_backoff_base_seconds,
        )
    elif backend == "anthropic":
        from rbirag.generation.anthropic import AnthropicLLM

        llm = AnthropicLLM(
            model or settings.anthropic_model,
            effort=settings.anthropic_effort,
            timeout_seconds=settings.anthropic_timeout_seconds,
        )
    elif backend == "ollama":
        from rbirag.generation.ollama import OllamaLLM

        llm = OllamaLLM(
            model or settings.ollama_model,
            base_url=settings.ollama_url,
            num_ctx=settings.ollama_num_ctx,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    elif backend == "fake":
        from rbirag.generation.llm import FakeLLM

        llm = FakeLLM()
    else:
        raise ValueError(f"unknown LLM backend: {backend!r}")
    use_cache = settings.llm_cache if cache is None else cache
    if use_cache and backend != "fake":
        from rbirag.generation.cache import CachedLLM

        return CachedLLM(llm, settings.llm_cache_dir)
    return llm
