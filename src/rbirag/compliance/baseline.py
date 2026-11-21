"""A fixed retrieve-then-judge pipeline: the baseline the agent has to beat.

Search with the clause text, give the top notifications to the same ``compare_clause``
judge, apply the same guardrails. One LLM call per clause and no decisions about which
tools to use, so the comparison isolates what the agent's tool choices add (rephrased
searches, reading, choosing sources, answering out-of-scope clauses without retrieval).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

from rbirag.compliance.agent import FATAL_ERRORS, Meter
from rbirag.compliance.guards import Proposal, decide
from rbirag.compliance.judge import ClauseInput, ClauseJudge
from rbirag.compliance.llm import ChatModel
from rbirag.compliance.records import ClausePrediction
from rbirag.compliance.regulations import RegulationStore
from rbirag.compliance.tools import ToolBox
from rbirag.config import ModelPrice, Settings

QUERY_CHARS = 1000


class FixedPipeline:
    def __init__(
        self,
        store: RegulationStore,
        chat: ChatModel,
        *,
        as_of: str = "2026-11-11",
        entity: str = "Commercial Banks",
        judge_docs: int = 3,
        judge_max_tokens: int = 1200,
        judge_context_tokens: int = 3000,
        abstain_threshold: float = 0.0,
        require_verified_quote: bool = True,
        prices: Mapping[str, ModelPrice] | None = None,
        usd_to_inr: float = 88.0,
    ) -> None:
        self.store = store
        self.judge = ClauseJudge(
            store,
            chat,
            as_of=as_of,
            entity=entity,
            context_tokens=judge_context_tokens,
            max_tokens=judge_max_tokens,
        )
        self.entity = entity
        self.judge_docs = judge_docs
        self.abstain_threshold = abstain_threshold
        self.require_verified_quote = require_verified_quote
        self.prices = prices or {}
        self.usd_to_inr = usd_to_inr
        self.titles = {d.doc_id: d.title for d in store.documents.values()}

    @classmethod
    def from_settings(
        cls, settings: Settings, store: RegulationStore, chat: ChatModel
    ) -> FixedPipeline:
        return cls(
            store,
            chat,
            as_of=settings.compliance_as_of,
            entity=settings.compliance_entity,
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
        judgement = None
        try:
            search = toolbox.run(
                "search_regulations", json.dumps({"query": clause.text[:QUERY_CHARS]})
            )
            top = search.record.result_doc_ids[: self.judge_docs]
            if top:
                compare = toolbox.run("compare_clause", json.dumps({"doc_ids": top}))
                judgement = compare.judgement
                meter.add(compare.judge_result)
        except FATAL_ERRORS:
            raise
        except Exception as exc:
            return ClausePrediction(
                item_id=item_id,
                verdict="insufficient_evidence",
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
                tool_calls=toolbox.calls,
                llm_calls=meter.llm_calls,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                system="baseline",
            )
        proposal = (
            Proposal(
                verdict=judgement.verdict,
                citations=judgement.citations,
                confidence=judgement.confidence,
                rationale=judgement.rationale,
            )
            if judgement is not None
            else None
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
        return ClausePrediction(
            item_id=item_id,
            verdict=decision.verdict,
            citations=decision.citations,
            confidence=decision.confidence,
            rationale=decision.rationale,
            evidence=decision.evidence,
            notes=decision.notes,
            tool_calls=toolbox.calls,
            llm_calls=meter.llm_calls,
            input_tokens=meter.input_tokens,
            output_tokens=meter.output_tokens,
            cost_usd=round(meter.cost_usd, 8),
            latency_ms=round(meter.latency_ms + sum(c.latency_ms for c in toolbox.calls), 2),
            system="baseline",
            model=meter.model,
        )
