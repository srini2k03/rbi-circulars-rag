"""What the compliance checker produces per clause: the contract between the agent (or a
baseline) and the evaluation."""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

Verdict = Literal["compliant", "conflict", "gap", "not_applicable", "insufficient_evidence"]
VERDICTS: tuple[str, ...] = get_args(Verdict)
# Verdicts that assert something about the regulation and therefore need evidence.
SUBSTANTIVE_VERDICTS: frozenset[str] = frozenset({"compliant", "conflict", "gap"})

ToolName = Literal["search_regulations", "fetch_notification", "compare_clause"]
TOOL_NAMES: tuple[str, ...] = get_args(ToolName)


class ToolCall(BaseModel):
    """One tool invocation in a clause's trajectory."""

    name: str  # a ToolName, or whatever (invalid) name the model emitted
    arguments: dict[str, Any] = Field(default_factory=dict)
    valid: bool = Field(description="Known tool, schema-valid arguments, resolvable ids.")
    error: str | None = None
    result_doc_ids: list[str] = Field(
        default_factory=list, description="Notifications the result showed the model."
    )
    latency_ms: float = Field(default=0.0, ge=0)


class ClausePrediction(BaseModel):
    item_id: str
    verdict: Verdict
    citations: list[str] = Field(default_factory=list, description="Cited doc ids.")
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    error: str | None = Field(default=None, description="Set when the run failed.")
    system: str = Field(default="agent", description="agent | baseline | ...")
    model: str | None = None

    @property
    def steps(self) -> int:
        return len(self.tool_calls)
