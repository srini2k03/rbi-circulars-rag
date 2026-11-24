"""End-to-end RAG: retrieve -> confidence gate -> prompt -> LLM -> citation validation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from rbirag.config import ModelPrice, RetrievalMode, Settings
from rbirag.generation.citations import CitationReport, parse_citations
from rbirag.generation.context import ContextReport, assemble_context
from rbirag.generation.cost import Cost, compute_cost
from rbirag.generation.limits import RequestLimiter
from rbirag.generation.llm import LLM, LLMResponse, TokenUsage
from rbirag.generation.prompts import REFUSAL_TEXT, SYSTEM_PROMPT, build_messages
from rbirag.generation.verify import (
    NOT_ESTABLISHED,
    LexicalSupport,
    SourceRef,
    SupportScorer,
    VerificationReport,
    remove_invalid_citations,
    verify_answer,
)
from rbirag.retrieval.confidence import Confidence
from rbirag.retrieval.pipeline import RetrievalResult, Retriever, ScoredChunk

logger = logging.getLogger(__name__)

VerifyPolicy = Literal["annotate", "abstain"]


def build_verifier(kind: str, settings: Settings) -> SupportScorer | None:
    """``lexical`` (default, no models), ``nli`` (the NLI cross-encoder; needs the ``ml``
    extra) or ``none``."""
    if kind == "none":
        return None
    if kind == "lexical":
        return LexicalSupport()
    if kind == "nli":
        from rbirag.eval.nli import CrossEncoderNLI

        return CrossEncoderNLI(settings.nli_model, device=settings.device)
    raise ValueError(f"unknown verifier: {kind!r}")


RefusalReason = Literal[
    "low_confidence", "no_sources", "model_refusal", "model_declined", "unsupported"
]


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
    pdf_url: str | None = None
    section: str | None = None
    ref_number: str | None = None
    department: str | None = None
    stage_scores: dict[str, float] = Field(default_factory=dict)
    stage_ranks: dict[str, int] = Field(default_factory=dict)


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
    cached: bool = False  # the LLM response came from the on-disk cache
    verification: VerificationReport | None = None
    raw_answer: str | None = None  # the model's text when the shown answer differs from it
    context: ContextReport | None = None  # dedup / budget / diversity / conflicts


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
                pdf_url=_s(md.get("pdf_url")),
                section=r.chunk.section,
                ref_number=_s(md.get("ref_number")),
                department=_s(md.get("department")),
                stage_scores=r.stage_scores,
                stage_ranks=r.stage_ranks,
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
        verifier: SupportScorer | None = None,
        verify_threshold: float = 0.6,
        verify_policy: VerifyPolicy = "annotate",
        dedup_threshold: float = 0.9,
        context_token_budget: int = 6000,
        relation: Callable[[str, str], str | None] | None = None,
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
        self.verifier = verifier
        self.verify_threshold = verify_threshold
        self.verify_policy: VerifyPolicy = verify_policy
        self.dedup_threshold = dedup_threshold
        self.context_token_budget = context_token_budget
        self.relation = relation

    @classmethod
    def from_settings(cls, retriever: Retriever, llm: LLM, settings: Settings) -> RAGPipeline:
        return cls(
            retriever,
            llm,
            top_k=settings.answer_top_k,
            gate=settings.confidence_gate,
            verifier=build_verifier(settings.verifier, settings),
            verify_threshold=settings.verify_threshold,
            verify_policy=settings.verify_policy,
            dedup_threshold=settings.context_dedup_threshold,
            context_token_budget=settings.context_token_budget,
            max_tokens=settings.max_answer_tokens,
            prices=settings.prices,
            usd_to_inr=settings.usd_to_inr,
        )

    # -- helpers -----------------------------------------------------------------------
    def retrieve_context(self, question: str) -> tuple[RetrievalResult, ContextReport]:
        """Retrieval plus context assembly: exactly the passages the generator sees."""
        result = self.retriever.retrieve(question, top_k=self.top_k, mode=self.mode)
        kept, report = assemble_context(
            result.results,
            dedup_threshold=self.dedup_threshold,
            token_budget=self.context_token_budget,
            relation=self.relation,
        )
        if len(kept) != len(result.results):
            kept = [r.model_copy(update={"rank": i}) for i, r in enumerate(kept, start=1)]
            result = result.model_copy(update={"results": kept})
        return result, report

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
        context: ContextReport | None = None,
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
            context=context,
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
        context: ContextReport | None = None,
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
        verification: VerificationReport | None = None
        raw: str | None = None
        refused = report.is_refusal or response.refused
        if self.verifier is not None and not refused:
            started = time.perf_counter()
            verification = verify_answer(
                text,
                [
                    SourceRef(n=s.n, doc_id=s.doc_id, text=s.text,
                              circular_number=s.circular_number, ref_number=s.ref_number)
                    for s in sources
                ],
                self.verifier,
                threshold=self.verify_threshold,
                report=report,
            )  # fmt: skip
            timings["verify"] = round((time.perf_counter() - started) * 1000, 3)
            timings["total"] = round(timings["total"] + timings["verify"], 3)
            if verification.invalid_citations:  # never show a citation that points nowhere
                raw = text
                # (``citations`` keeps reporting the model's original, invalid ids)
                text = remove_invalid_citations(text, verification.invalid_citations)
            if verification.verdict == "unsupported" and self.verify_policy == "abstain":
                raw = raw or text
                text, reason, refused = NOT_ESTABLISHED, "unsupported", True
        return RAGAnswer(
            question=question,
            answer=text,
            refused=refused,
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
            cached=response.cached,
            verification=verification,
            raw_answer=raw,
            context=context,
        )

    # -- public API ----------------------------------------------------------------------
    def answer(self, question: str) -> RAGAnswer:
        retrieval, context = self.retrieve_context(question)
        sources = to_sources(retrieval.results)
        reason = self._gate_reason(retrieval)
        if reason is not None:
            return self._refusal(question, retrieval, reason, sources, context)
        started = time.perf_counter()
        response = self.llm.generate(
            build_messages(question, [r.chunk for r in retrieval.results]),
            system=self.system_prompt,
            max_tokens=self.max_tokens,
        )
        generate_ms = round((time.perf_counter() - started) * 1000, 3)
        return self._finish(question, retrieval, sources, response, generate_ms, context)

    def stream(self, question: str) -> Iterator[StreamEvent]:
        """Yields ``sources`` first, then ``delta`` text events, then ``done`` with the answer."""
        retrieval, context = self.retrieve_context(question)
        sources = to_sources(retrieval.results)
        yield StreamEvent(type="sources", sources=sources)
        reason = self._gate_reason(retrieval)
        if reason is not None:
            answer = self._refusal(question, retrieval, reason, sources, context)
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
            type="done",
            answer=self._finish(question, retrieval, sources, final, generate_ms, context),
        )


_LIMITERS: dict[str, RequestLimiter] = {}


def shared_limiter(settings: Settings) -> RequestLimiter:
    """One limiter per state directory, so all LLMs in a process share the budgets."""
    state_dir = settings.llm_cache_dir / "quota"
    key = str(state_dir.resolve())
    if key not in _LIMITERS:
        _LIMITERS[key] = RequestLimiter(
            rpm=settings.llm_rpm,
            rpd=settings.rpd_for,
            tpm=settings.tpm_for,
            state_dir=state_dir,
        )
    return _LIMITERS[key]


_HEALTH: dict[float, Any] = {}


def shared_health(settings: Settings) -> Any:
    """One :class:`ModelHealth` per process, so cooldowns apply across all roles."""
    from rbirag.generation.openai_compat import ModelHealth

    key = settings.llm_model_cooldown_seconds
    if key not in _HEALTH:
        _HEALTH[key] = ModelHealth(cooldown_seconds=key)
    return _HEALTH[key]


def build_llm(
    settings: Settings,
    backend: str | None = None,
    model: str | None = None,
    *,
    fallbacks: Sequence[str] | None = None,
    cache: bool | None = None,
) -> LLM:
    """LLM for ``backend`` (default ``settings.llm_backend``), wrapped in the disk cache.

    For the OpenAI-compatible backend, ``fallbacks`` is the model chain after ``model``;
    by default the generator chain applies when no explicit model is given.
    """
    backend = backend or settings.llm_backend
    llm: LLM
    if backend == "openai_compatible":
        from rbirag.generation.openai_compat import OpenAICompatibleLLM

        chain = (
            list(fallbacks)
            if fallbacks is not None
            else (settings.oai_model_fallbacks if model is None else [])
        )
        llm = OpenAICompatibleLLM(
            model or settings.oai_model,
            fallbacks=chain,
            health=shared_health(settings),
            switch_after_503=settings.llm_switch_after_503,
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
            num_thread=settings.ollama_num_thread,
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
