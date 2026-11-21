"""The agent's three tools: schemas, strict argument validation, execution.

An invalid call (unknown tool, malformed JSON, unexpected or missing arguments, a
notification id that does not resolve) is not executed: the model gets an error message
so it can correct itself, and the call is recorded as invalid for the tool-call
metrics.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rbirag.compliance.judge import ClauseInput, ClauseJudge, Judgement
from rbirag.compliance.llm import ChatResult
from rbirag.compliance.records import ToolCall
from rbirag.compliance.regulations import RegulationStore, UnknownNotificationError

MAX_COMPARE_DOCS = 4

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": (
                "Search RBI notifications. Returns one hit per notification (doc_id, title, "
                "date, entity addressed, snippet, replaced_by) and a retrieval confidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD (optional)."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_notification",
            "description": (
                "Read one notification: metadata, table of contents, later notifications "
                "citing it, and the passages most relevant to `focus`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "e.g. rbi-13141"},
                    "focus": {"type": "string", "description": "What to read about."},
                },
                "required": ["doc_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_clause",
            "description": (
                "Judge the clause under review against the relevant passages of up to four "
                "notifications. Returns verdict, citations, evidence quotes, confidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_COMPARE_DOCS,
                    },
                    "focus": {"type": "string", "description": "Aspect to focus on (optional)."},
                },
                "required": ["doc_ids"],
                "additionalProperties": False,
            },
        },
    },
]
_ALLOWED: dict[str, tuple[set[str], set[str]]] = {
    "search_regulations": ({"query"}, {"query", "top_k", "date_from"}),
    "fetch_notification": ({"doc_id"}, {"doc_id", "focus"}),
    "compare_clause": ({"doc_ids"}, {"doc_ids", "focus"}),
}


class ToolArgumentError(ValueError):
    pass


@dataclass
class ToolOutcome:
    record: ToolCall
    content: str  # what the model sees
    judgement: Judgement | None = None
    judge_result: ChatResult | None = None
    retrieval_confidence: float | None = None
    texts_read: list[str] = field(default_factory=list)


def _parse_args(name: str, raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ToolArgumentError(f"arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(args, dict):
        raise ToolArgumentError("arguments must be a JSON object")
    required, allowed = _ALLOWED[name]
    missing = required - set(args)
    extra = set(args) - allowed
    if missing:
        raise ToolArgumentError(f"missing argument(s): {', '.join(sorted(missing))}")
    if extra:
        raise ToolArgumentError(f"unexpected argument(s): {', '.join(sorted(extra))}")
    return args


def _text_arg(
    args: dict[str, Any], key: str, *, min_len: int = 0, max_len: int = 2000
) -> str | None:
    if key not in args or args[key] is None:
        return None
    value = args[key]
    if not isinstance(value, str) or not (min_len <= len(value.strip()) <= max_len):
        raise ToolArgumentError(f"{key} must be a string of {min_len}-{max_len} characters")
    return value.strip()


def _compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class ToolBox:
    """Executes tool calls for one clause and keeps its trajectory."""

    def __init__(self, store: RegulationStore, judge: ClauseJudge, clause: ClauseInput) -> None:
        self.store = store
        self.judge = judge
        self.clause = clause
        self.seen_doc_ids: set[str] = set()  # every notification a result showed the model
        self.calls: list[ToolCall] = []
        self._memo: dict[tuple[str, str], ToolOutcome] = {}

    def run(self, name: str, raw_arguments: str) -> ToolOutcome:
        started = time.perf_counter()
        key = (name, raw_arguments)
        if key in self._memo:  # an identical repeat: same answer, still counts as a step
            prev = self._memo[key]
            record = prev.record.model_copy(update={"latency_ms": 0.0})
            self.calls.append(record)
            return ToolOutcome(record=record, content=prev.content + " (repeated call)")
        try:
            if name not in _ALLOWED:
                raise ToolArgumentError(f"unknown tool {name!r}; use one of {sorted(_ALLOWED)}")
            args = _parse_args(name, raw_arguments)
            handler: Callable[[dict[str, Any]], ToolOutcome] = getattr(self, f"_{name}")
            outcome = handler(args)
        except (ToolArgumentError, UnknownNotificationError) as exc:
            try:
                parsed = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError:
                parsed = {"_raw": raw_arguments[:500]}
            record = ToolCall(
                name=name,
                arguments=parsed if isinstance(parsed, dict) else {"_raw": str(parsed)[:500]},
                valid=False,
                error=str(exc),
            )
            outcome = ToolOutcome(record=record, content=f"ERROR: {exc}")
        outcome.record.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.calls.append(outcome.record)
        if outcome.record.name != "compare_clause":
            # only search/fetch results count as "seen": passing a guessed id to the judge
            # must not make it citable
            self.seen_doc_ids.update(outcome.record.result_doc_ids)
        if outcome.record.valid:
            self._memo[key] = outcome
        return outcome

    # -- tools ---------------------------------------------------------------------------
    def _search_regulations(self, args: dict[str, Any]) -> ToolOutcome:
        query = _text_arg(args, "query", min_len=3, max_len=1000)
        assert query is not None
        top_k = args.get("top_k")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10
        ):
            raise ToolArgumentError("top_k must be an integer from 1 to 10")
        date_from = _text_arg(args, "date_from", max_len=10)
        try:
            day = dt.date.fromisoformat(date_from) if date_from else None
        except ValueError as exc:
            raise ToolArgumentError("date_from must be YYYY-MM-DD") from exc
        result = self.store.search(query, top_k=top_k, date_from=day)
        hits = [
            {
                "doc_id": h.doc_id,
                "title": h.title,
                "date": h.date,
                "entity": h.entity,
                "section": h.section,
                "snippet": h.snippet,
                **({"replaced_by": h.replaced_by} if h.replaced_by else {}),
            }
            for h in result.hits
        ]
        payload: dict[str, Any] = {"hits": hits, "confidence": result.confidence}
        if result.excluded_other_entity:
            payload["note"] = (
                f"{result.excluded_other_entity} hit(s) addressed to other entity types omitted"
            )
        record = ToolCall(
            name="search_regulations",
            arguments=args,
            valid=True,
            result_doc_ids=[h.doc_id for h in result.hits],
        )
        return ToolOutcome(
            record=record,
            content=_compact(payload),
            retrieval_confidence=result.confidence,
            texts_read=[h.snippet for h in result.hits],
        )

    def _fetch_notification(self, args: dict[str, Any]) -> ToolOutcome:
        doc_arg = _text_arg(args, "doc_id", min_len=1, max_len=100)
        assert doc_arg is not None
        focus = _text_arg(args, "focus", max_len=1000) or self.clause.text
        view = self.store.fetch(doc_arg, query=focus)
        payload = {
            "doc_id": view.doc_id,
            "title": view.title,
            "date": view.date,
            "entity": view.entity,
            "replaced_by": view.replaced_by,
            "cited_by": [r.model_dump() for r in view.cited_by[:5]],
            "passages": [
                {"para": p.number, "heading": p.heading, "text": p.text} for p in view.passages
            ],
            "truncated": view.truncated,
        }
        record = ToolCall(
            name="fetch_notification", arguments=args, valid=True, result_doc_ids=[view.doc_id]
        )
        return ToolOutcome(
            record=record, content=_compact(payload), texts_read=[p.text for p in view.passages]
        )

    def _compare_clause(self, args: dict[str, Any]) -> ToolOutcome:
        doc_ids = args["doc_ids"]
        if (
            not isinstance(doc_ids, list)
            or not 1 <= len(doc_ids) <= MAX_COMPARE_DOCS
            or not all(isinstance(d, str) for d in doc_ids)
        ):
            raise ToolArgumentError(
                f"doc_ids must be a list of 1-{MAX_COMPARE_DOCS} doc_id strings"
            )
        resolved = [self.store.resolve(d) for d in doc_ids]  # raises on unknown ids
        focus = _text_arg(args, "focus", max_len=1000)
        judgement, result, shown = self.judge.judge(self.clause, resolved, focus=focus)
        payload = {
            "verdict": judgement.verdict,
            "citations": judgement.citations,
            "evidence": [
                {"doc_id": e.doc_id, "quote": e.quote, "verified": e.verified}
                for e in judgement.evidence
            ],
            "confidence": judgement.confidence,
            "rationale": judgement.rationale,
        }
        record = ToolCall(
            name="compare_clause", arguments=args, valid=True, result_doc_ids=resolved
        )
        return ToolOutcome(
            record=record,
            content=_compact(payload),
            judgement=judgement,
            judge_result=result,
            texts_read=shown,
        )
