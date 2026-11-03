"""LLM-as-judge: claim-decomposition faithfulness and answer relevance in one call.

The judge receives the question, the numbered sources the answer was generated from and
the answer. It splits the answer into atomic factual claims, marks each as supported or
not *by the sources alone*, and rates relevance 1-5. Structured outputs (JSON schema) are
requested when the backend supports them; the parser is tolerant either way.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from rbirag.generation.citations import strip_citations
from rbirag.generation.llm import LLM, Message, TokenUsage

JUDGE_SYSTEM = """You are a meticulous evaluator of answers produced by a retrieval-augmented \
system over Reserve Bank of India notifications. You judge only against the numbered \
sources you are given, never against outside knowledge."""

JUDGE_INSTRUCTIONS = """Evaluate the ANSWER to the QUESTION using only the SOURCES.

1. Decompose the answer into atomic factual claims (one fact each; ignore citation markers, \
greetings and statements that the answer is unknown).
2. For each claim decide whether the sources explicitly support it ("supported": true) or \
not ("supported": false: contradicted, missing, or only loosely implied). List the source \
numbers that support it.
3. Rate how well the answer addresses the question ("relevance") from 1 to 5:
   5 = directly and completely answers it; 4 = answers it with minor gaps; 3 = partially \
answers; 2 = mostly off-target; 1 = does not address the question.

Return only JSON matching: {"claims": [{"claim": str, "supported": bool, "source_ids": \
[int]}], "relevance": int, "relevance_reason": str}"""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "source_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["claim", "supported", "source_ids"],
                "additionalProperties": False,
            },
        },
        "relevance": {"type": "integer"},
        "relevance_reason": {"type": "string"},
    },
    "required": ["claims", "relevance", "relevance_reason"],
    "additionalProperties": False,
}


class JudgeParseError(ValueError):
    pass


class Claim(BaseModel):
    claim: str
    supported: bool
    source_ids: list[int] = Field(default_factory=list)

    @field_validator("supported", mode="before")
    @classmethod
    def _bool(cls, v: object) -> object:
        if isinstance(v, str):
            low = v.strip().lower()
            if low in {"yes", "true", "supported", "y", "1"}:
                return True
            if low in {"no", "false", "unsupported", "not supported", "n", "0"}:
                return False
        return v

    @field_validator("source_ids", mode="before")
    @classmethod
    def _ids(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, (int, str)):
            v = [v]
        if isinstance(v, list):
            out: list[int] = []
            for item in v:
                m = re.search(r"\d+", str(item))
                if m:
                    out.append(int(m.group(0)))
            return out
        return v


class JudgeVerdict(BaseModel):
    claims: list[Claim]
    relevance: int
    relevance_reason: str = ""

    @field_validator("relevance", mode="before")
    @classmethod
    def _clamp(cls, v: object) -> object:
        try:
            value = round(float(str(v).strip()))
        except ValueError:
            return v
        return min(5, max(1, value))

    @property
    def faithfulness(self) -> float | None:
        if not self.claims:
            return None
        return sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def relevance_score(self) -> float:
        """Relevance mapped to [0, 1]."""
        return (self.relevance - 1) / 4


class JudgeResult(BaseModel):
    verdict: JudgeVerdict | None
    usage: TokenUsage
    model: str
    backend: str
    error: str | None = None


def extract_json_object(text: str) -> dict[str, Any]:
    """First balanced JSON object in ``text`` (handles code fences and chatter)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
        start = text.find("{", start + 1) if not candidates else -1
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise JudgeParseError("no JSON object found in judge output")


def parse_judge_output(text: str) -> JudgeVerdict:
    obj = extract_json_object(text)
    try:
        return JudgeVerdict.model_validate(obj)
    except ValidationError as exc:
        raise JudgeParseError(
            f"judge JSON did not match schema: {exc.error_count()} errors"
        ) from exc


def build_judge_prompt(question: str, answer: str, sources: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {s.strip()}" for i, s in enumerate(sources, start=1))
    return (
        f"{JUDGE_INSTRUCTIONS}\n\nSOURCES:\n\n{numbered or '(none)'}\n\n"
        f"QUESTION: {question.strip()}\n\nANSWER: {strip_citations(answer) or '(empty)'}"
    )


def judge_answer(
    llm: LLM, question: str, answer: str, sources: Sequence[str], *, max_tokens: int = 4096
) -> JudgeResult:
    response = llm.generate(
        [Message(role="user", content=build_judge_prompt(question, answer, sources))],
        system=JUDGE_SYSTEM,
        max_tokens=max_tokens,
        json_schema=JUDGE_SCHEMA,
    )

    def result(verdict: JudgeVerdict | None, error: str | None = None) -> JudgeResult:
        return JudgeResult(
            verdict=verdict,
            usage=response.usage,
            model=response.model,
            backend=response.backend,
            error=error,
        )

    if response.refused:
        return result(None, "judge declined")
    try:
        return result(parse_judge_output(response.text))
    except JudgeParseError as exc:
        return result(None, str(exc))
