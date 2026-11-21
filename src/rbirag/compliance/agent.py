"""The compliance agent: a bounded tool-calling loop per clause.

Per clause the planner model sees the clause and three tools. It may answer
``not_applicable`` straight away, or search, read and call ``compare_clause`` (a separate
judge call), then reply with a JSON final answer. The loop stops at the final answer or
when the tool budget is spent (the planner is then asked for its answer with tools
disabled). :func:`rbirag.compliance.guards.decide` turns the proposal into the decision.

Two safety nets found necessary on the dev split:

* if the planner proposes compliant / conflict / gap without having called
  ``compare_clause``, the agent calls it on the notifications the planner cited (or read)
  and adopts the judge's answer: the verdict always comes from a comparison against
  passages, with quotes that can be verified;
* before each planner call the oldest tool results are shortened to keep the request
  inside ``planner_context_tokens`` (small free-tier models reject oversized requests);
  a request rejected as too large is retried once with the history halved.

Latency is the sum of the recorded LLM and tool latencies, so it is the same whether a
run is live or replayed from the response cache.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from rbirag.compliance.context import fit_history, history_tokens
from rbirag.compliance.guards import Proposal, decide
from rbirag.compliance.judge import (
    ClauseInput,
    ClauseJudge,
    Judgement,
    clip_confidence,
    normalise_verdict,
    parse_json_object,
)
from rbirag.compliance.llm import (
    CacheMissError,
    ChatMessage,
    ChatModel,
    ChatResult,
    ToolCallRequest,
)
from rbirag.compliance.prompts import FINALIZE_USER, PLANNER_USER, planner_system
from rbirag.compliance.records import SUBSTANTIVE_VERDICTS, ClausePrediction
from rbirag.compliance.regulations import RegulationStore
from rbirag.compliance.tools import TOOL_SCHEMAS, ToolBox
from rbirag.config import ModelPrice, Settings
from rbirag.generation.cost import compute_cost
from rbirag.generation.llm import ConfigurationError, QuotaExhaustedError, RequestRejectedError

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 8000
# Failures that must stop a whole run (resume later / fix config / refresh the cache)
# rather than be recorded as one clause's error.
FATAL_ERRORS: tuple[type[BaseException], ...] = (
    QuotaExhaustedError,
    ConfigurationError,
    CacheMissError,
)
REPAIR_PROMPT = (
    "Reply with only the final JSON object: "
    '{"verdict": ..., "citations": [...], "confidence": ..., "rationale": ...}'
)


class Meter:
    """LLM calls, tokens, cost and latency for one clause."""

    def __init__(self, prices: Mapping[str, ModelPrice], usd_to_inr: float) -> None:
        self.prices = prices
        self.usd_to_inr = usd_to_inr
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.latency_ms = 0.0
        self.models: list[str] = []

    def add(self, result: ChatResult | None) -> None:
        if result is None:
            return
        self.llm_calls += 1
        self.input_tokens += result.usage.input_tokens + result.usage.cache_read_input_tokens
        self.output_tokens += result.usage.output_tokens
        cost = compute_cost(
            result.usage,
            model=result.model,
            backend=result.backend,
            prices=self.prices,
            usd_to_inr=self.usd_to_inr,
        )
        self.cost_usd += cost.usd
        self.latency_ms += result.latency_ms
        self.models.append(result.model)

    @property
    def model(self) -> str | None:
        return max(set(self.models), key=self.models.count) if self.models else None


def _text_tool_call(content: str | None, turn: int) -> ToolCallRequest | None:
    """Fallback for models that write a tool call as JSON text instead of using the
    tools API: ``{"tool": "search_regulations", "arguments": {...}}``."""
    import json

    data = parse_json_object(content)
    if not data:
        return None
    name = data.get("tool") or data.get("name") or data.get("function")
    args = data.get("arguments", data.get("args", data.get("parameters")))
    if isinstance(name, str) and isinstance(args, dict) and "verdict" not in data:
        return ToolCallRequest(id=f"text_{turn}", name=name, arguments=json.dumps(args))
    return None


class ComplianceAgent:
    def __init__(
        self,
        store: RegulationStore,
        chat: ChatModel,
        *,
        judge_chat: ChatModel | None = None,
        as_of: str = "2026-11-11",
        entity: str = "Commercial Banks",
        max_steps: int = 8,
        planner_max_tokens: int = 1024,
        planner_context_tokens: int = 4500,
        judge_max_tokens: int = 1200,
        judge_context_tokens: int = 3000,
        abstain_threshold: float = 0.0,
        require_verified_quote: bool = True,
        prices: Mapping[str, ModelPrice] | None = None,
        usd_to_inr: float = 88.0,
    ) -> None:
        self.store = store
        self.chat = chat
        self.judge = ClauseJudge(
            store,
            judge_chat or chat,
            as_of=as_of,
            entity=entity,
            context_tokens=judge_context_tokens,
            max_tokens=judge_max_tokens,
        )
        self.as_of = as_of
        self.entity = entity
        self.max_steps = max_steps
        self.planner_max_tokens = planner_max_tokens
        self.planner_context_tokens = planner_context_tokens
        self.abstain_threshold = abstain_threshold
        self.require_verified_quote = require_verified_quote
        self.prices = prices or {}
        self.usd_to_inr = usd_to_inr
        self.titles = {d.doc_id: d.title for d in store.documents.values()}
        self.system_prompt = planner_system(as_of=as_of, entity=entity, max_steps=max_steps)

    @classmethod
    def from_settings(
        cls, settings: Settings, store: RegulationStore, chat: ChatModel
    ) -> ComplianceAgent:
        return cls(
            store,
            chat,
            as_of=settings.compliance_as_of,
            entity=settings.compliance_entity,
            max_steps=settings.compliance_max_steps,
            planner_max_tokens=settings.compliance_planner_max_tokens,
            planner_context_tokens=settings.compliance_planner_context_tokens,
            judge_max_tokens=settings.compliance_judge_max_tokens,
            judge_context_tokens=settings.compliance_judge_context_tokens,
            abstain_threshold=settings.compliance_abstain_threshold,
            require_verified_quote=settings.compliance_require_verified_quote,
            prices=settings.prices,
            usd_to_inr=settings.usd_to_inr,
        )

    def check(self, clause: ClauseInput, *, item_id: str | None = None) -> ClausePrediction:
        item_id = item_id or clause.label
        meter = Meter(self.prices, self.usd_to_inr)
        toolbox = ToolBox(self.store, self.judge, clause)
        started = time.perf_counter()
        try:
            proposal, judgement, agent_notes = self._loop(clause, toolbox, meter)
        except FATAL_ERRORS:
            raise
        except Exception as exc:
            logger.exception("clause %s failed", item_id)
            return ClausePrediction(
                item_id=item_id,
                verdict="insufficient_evidence",
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
                tool_calls=toolbox.calls,
                llm_calls=meter.llm_calls,
                input_tokens=meter.input_tokens,
                output_tokens=meter.output_tokens,
                cost_usd=round(meter.cost_usd, 8),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                system="agent",
                model=meter.model,
            )
        decision = decide(
            proposal,
            judgement,
            seen_doc_ids=toolbox.seen_doc_ids,
            titles=self.titles,
            replaced_by=self.store.graph.replaced_by,
            entity=self.entity,
            require_verified_quote=self.require_verified_quote,
            abstain_threshold=self.abstain_threshold,
        )
        tool_ms = sum(c.latency_ms for c in toolbox.calls)
        return ClausePrediction(
            item_id=item_id,
            verdict=decision.verdict,
            citations=decision.citations,
            confidence=decision.confidence,
            rationale=decision.rationale,
            evidence=decision.evidence,
            notes=agent_notes + decision.notes,
            tool_calls=toolbox.calls,
            llm_calls=meter.llm_calls,
            input_tokens=meter.input_tokens,
            output_tokens=meter.output_tokens,
            cost_usd=round(meter.cost_usd, 8),
            latency_ms=round(meter.latency_ms + tool_ms, 2),
            system="agent",
            model=meter.model,
        )

    def _loop(
        self, clause: ClauseInput, toolbox: ToolBox, meter: Meter
    ) -> tuple[Proposal | None, Judgement | None, list[str]]:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=PLANNER_USER.format(
                    policy_title=clause.policy_title or "-",
                    context=clause.context or "-",
                    label=clause.label,
                    clause=clause.text,
                ),
            ),
        ]
        judgement: Judgement | None = None
        repaired = finalize_sent = False
        for turn in range(self.max_steps + 3):
            budget_left = len(toolbox.calls) < self.max_steps
            if not budget_left and not finalize_sent:
                messages.append(ChatMessage(role="user", content=FINALIZE_USER))
                finalize_sent = True
            result = self._plan(messages, tools=TOOL_SCHEMAS if budget_left else None)
            meter.add(result)
            msg = result.message
            calls = list(msg.tool_calls)
            if not calls and budget_left:
                text_call = _text_tool_call(msg.content, turn)
                if text_call is not None:
                    calls = [text_call]
            if calls and budget_left:
                messages.append(
                    ChatMessage(
                        role="assistant", content=msg.content, tool_calls=calls, raw=msg.raw
                    )
                )
                for call in calls:
                    if len(toolbox.calls) >= self.max_steps:
                        content = "ERROR: tool budget exhausted; give your final answer"
                    else:
                        outcome = toolbox.run(call.name, call.arguments)
                        content = outcome.content
                        if outcome.judgement is not None:
                            judgement = outcome.judgement
                            meter.add(outcome.judge_result)
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=content[:MAX_TOOL_RESULT_CHARS],
                        )
                    )
                continue
            proposal = self._proposal(msg.content)
            if proposal is not None:
                if proposal.verdict in SUBSTANTIVE_VERDICTS and judgement is None:
                    return self._auto_compare(proposal, toolbox, meter)
                return proposal, judgement, []
            if repaired:
                break
            repaired = True  # one chance to fix a malformed final answer
            messages.append(ChatMessage(role="assistant", content=msg.content or "", raw=msg.raw))
            messages.append(ChatMessage(role="user", content=REPAIR_PROMPT))
        return None, judgement, []

    def _plan(self, messages: list[ChatMessage], *, tools: Any) -> ChatResult:
        """One planner call with the history fitted to the context budget."""
        fit_history(messages, self.planner_context_tokens, tools or ())
        try:
            return self.chat.chat(
                messages,
                tools=tools,
                max_tokens=self.planner_max_tokens,
                json_object=tools is None,
            )
        except RequestRejectedError as exc:
            text = str(exc).lower()
            if "413" not in text and "too large" not in text:
                raise
            logger.warning("planner request too large; retrying with a smaller history")
            current = history_tokens(messages, tools or ())
            target = min(self.planner_context_tokens, current) // 2
            if not fit_history(messages, target, tools or ()):
                raise
            return self.chat.chat(
                messages,
                tools=tools,
                max_tokens=self.planner_max_tokens,
                json_object=tools is None,
            )

    def _auto_compare(
        self, proposal: Proposal, toolbox: ToolBox, meter: Meter
    ) -> tuple[Proposal | None, Judgement | None, list[str]]:
        """The planner skipped compare_clause: run it and adopt the judge's answer."""
        fetched = [
            d
            for c in toolbox.calls
            if c.name == "fetch_notification" and c.valid
            for d in c.result_doc_ids
        ]
        cited = [c for c in proposal.citations if c in toolbox.seen_doc_ids]
        doc_ids = list(dict.fromkeys(cited or fetched))[:4]
        if not doc_ids:
            return proposal, None, []
        outcome = toolbox.run("compare_clause", json.dumps({"doc_ids": doc_ids}))
        judgement = outcome.judgement
        if judgement is None:
            return proposal, None, []
        meter.add(outcome.judge_result)
        note = (
            f"planner proposed {proposal.verdict} without compare_clause; compared "
            f"{', '.join(doc_ids)} automatically and adopted the result ({judgement.verdict})"
        )
        adopted = Proposal(
            verdict=judgement.verdict,
            citations=judgement.citations or doc_ids,
            confidence=judgement.confidence,
            rationale=judgement.rationale or proposal.rationale,
        )
        return adopted, judgement, [note]

    @staticmethod
    def _proposal(content: str | None) -> Proposal | None:
        data: dict[str, Any] | None = parse_json_object(content)
        if not data:
            return None
        verdict = normalise_verdict(data.get("verdict"))
        if verdict is None:
            return None
        raw = data.get("citations")
        citations = [str(c) for c in raw] if isinstance(raw, list) else []
        return Proposal(
            verdict=verdict,
            citations=citations,
            confidence=clip_confidence(data["confidence"]) if "confidence" in data else None,
            rationale=str(data.get("rationale", ""))[:1000],
        )
