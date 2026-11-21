"""``compare_clause``: judge one clause against passages of chosen notifications.

A separate, tool-free LLM call with a fixed JSON contract. The passages are selected by
the clause text itself (BM25 within each notification), so the judge sees the
paragraphs that matter even in a 60K-token Master Direction. Every evidence quote is
checked against the passages actually shown; a quote that is not there is marked
unverified, which the guardrails use.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from rbirag.compliance.llm import ChatMessage, ChatModel, ChatResult
from rbirag.compliance.prompts import JUDGE_USER, judge_system
from rbirag.compliance.records import VERDICTS, EvidenceQuote, Verdict

if TYPE_CHECKING:
    from rbirag.compliance.regulations import RegulationStore

_VERDICT_SYNONYMS: dict[str, str] = {
    "non_compliant": "conflict",
    "noncompliant": "conflict",
    "contradiction": "conflict",
    "violation": "conflict",
    "partially_compliant": "gap",
    "partial": "gap",
    "partial_compliance": "gap",
    "incomplete": "gap",
    "na": "not_applicable",
    "n/a": "not_applicable",
    "out_of_scope": "not_applicable",
    "insufficient": "insufficient_evidence",
    "unknown": "insufficient_evidence",
    "no_evidence": "insufficient_evidence",
}
_QUOTE_FOLD = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


class ClauseInput(BaseModel):
    """The clause under review, as the agent sees it."""

    label: str
    text: str
    policy_title: str = ""
    context: str = ""


class Judgement(BaseModel):
    verdict: Verdict
    citations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    rationale: str = ""
    doc_ids: list[str] = Field(default_factory=list, description="Notifications shown.")
    parsed: bool = True


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """The first JSON object in ``text`` (tolerates code fences and surrounding prose)."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start >= 0:
        depth = 0
        in_string = escaped = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return value if isinstance(value, dict) else None
        start = cleaned.find("{", start + 1)
    return None


def normalise_verdict(value: object) -> Verdict | None:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-]+", "_", value.strip().lower())
    key = _VERDICT_SYNONYMS.get(key, key)
    return key if key in VERDICTS else None  # type: ignore[return-value]


def clip_confidence(value: object, default: float = 0.5) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number > 1.0 and number <= 100.0:  # "85" meaning 85 %
        number /= 100.0
    return min(1.0, max(0.0, number))


def _fold(text: str) -> str:
    return " ".join(text.translate(_QUOTE_FOLD).lower().split())


def verify_quote(quote: str, texts: Sequence[str], *, min_ratio: float = 0.9) -> bool:
    """Whether ``quote`` occurs in one of ``texts`` (case, whitespace, quote and dash
    insensitive; quotes of 30+ characters may differ by small edits)."""
    q = _fold(quote).strip(" .\"'")
    if len(q) < 12:
        return False
    for text in texts:
        t = _fold(text)
        if q in t:
            return True
        if len(q) >= 30 and _fuzzy_contains(t, q, min_ratio):
            return True
    return False


def _fuzzy_contains(text: str, quote: str, min_ratio: float) -> bool:
    """Align a window of ``text`` on the longest exact match with ``quote`` and require
    the window to be at least ``min_ratio`` similar (tolerates a few scattered edits)."""
    match = difflib.SequenceMatcher(None, text, quote, autojunk=False).find_longest_match(
        0, len(text), 0, len(quote)
    )
    if match.size < 8:
        return False
    start = max(0, match.a - match.b - 5)
    window = text[start : start + len(quote) + 10]
    matcher = difflib.SequenceMatcher(None, window, quote, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched >= min_ratio * len(quote)


def format_passages(views: Sequence[Any]) -> str:
    blocks: list[str] = []
    for v in views:
        head = f"[{v.doc_id}] {v.title} (date: {v.date or 'unknown'}"
        if v.entity:
            head += f"; addressed to: {v.entity}"
        head += ")"
        if v.replaced_by:
            later = ", ".join(v.replaced_by)
            head += f"\nNOTE: later notification(s) {later} repeal or supersede this one."
        body = "\n\n".join((f"[para {p.number}] " if p.number else "") + p.text for p in v.passages)
        blocks.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(blocks)


class ClauseJudge:
    def __init__(
        self,
        store: RegulationStore,
        chat: ChatModel,
        *,
        as_of: str,
        entity: str,
        context_tokens: int = 3000,
        max_tokens: int = 1200,
    ) -> None:
        self.store = store
        self.chat = chat
        self.as_of = as_of
        self.entity = entity
        self.context_tokens = context_tokens
        self.max_tokens = max_tokens

    def judge(
        self, clause: ClauseInput, doc_ids: Sequence[str], *, focus: str | None = None
    ) -> tuple[Judgement, ChatResult, list[str]]:
        """Returns the judgement, the raw chat result (for accounting) and the passage
        texts shown (for quote checks elsewhere)."""
        ids = list(dict.fromkeys(self.store.resolve(d) for d in doc_ids))
        budget = max(200, self.context_tokens // max(1, len(ids)))
        query = f"{clause.text} {focus or ''}".strip()
        views = [self.store.fetch(d, query=query, max_tokens=budget) for d in ids]
        shown = [p.text for v in views for p in v.passages]
        messages = [
            ChatMessage(role="system", content=judge_system(as_of=self.as_of, entity=self.entity)),
            ChatMessage(
                role="user",
                content=JUDGE_USER.format(
                    label=clause.label,
                    policy_title=clause.policy_title or "-",
                    context=clause.context or "-",
                    clause=clause.text,
                    passages=format_passages(views),
                ),
            ),
        ]
        result = self.chat.chat(messages, max_tokens=self.max_tokens, json_object=True)
        return self.parse(result.message.content, ids, shown), result, shown

    @staticmethod
    def parse(text: str | None, doc_ids: Sequence[str], shown: Sequence[str]) -> Judgement:
        data = parse_json_object(text)
        verdict = normalise_verdict(data.get("verdict")) if data else None
        if data is None or verdict is None:
            return Judgement(
                verdict="insufficient_evidence",
                confidence=0.0,
                rationale="the judge's reply could not be parsed",
                doc_ids=list(doc_ids),
                parsed=False,
            )
        allowed = set(doc_ids)
        raw_citations = data.get("citations")
        citations = [
            str(c)
            for c in (raw_citations if isinstance(raw_citations, list) else [])
            if str(c) in allowed
        ]
        evidence: list[EvidenceQuote] = []
        raw_evidence = data.get("evidence")
        for ev in raw_evidence if isinstance(raw_evidence, list) else []:
            if not isinstance(ev, Mapping):
                continue
            doc_id, quote = str(ev.get("doc_id", "")), str(ev.get("quote", "")).strip()
            if doc_id in allowed and quote:
                evidence.append(
                    EvidenceQuote(doc_id=doc_id, quote=quote, verified=verify_quote(quote, shown))
                )
        if not citations:
            citations = list(dict.fromkeys(e.doc_id for e in evidence))
        return Judgement(
            verdict=verdict,
            citations=citations,
            evidence=evidence,
            confidence=clip_confidence(data.get("confidence")),
            rationale=str(data.get("rationale", ""))[:1000],
            doc_ids=list(doc_ids),
        )
