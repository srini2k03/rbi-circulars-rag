"""Request / response schemas of the versioned API (``/v1``)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rbirag.api.schemas import Backend, CostOut, Usage
from rbirag.generation.context import ContextReport
from rbirag.generation.verify import VerificationReport
from rbirag.intel.extract import DocIntel
from rbirag.intel.graph import Edge, LatestInCorpus
from rbirag.query.understanding import QueryAnalysis
from rbirag.retrieval.params import Expansion, Fusion, ParentMode, ScoreNorm
from rbirag.retrieval.pipeline import RetrievalTrace

EXAMPLE = "How often must banks update KYC for high-risk customers?"
CONFIDENCE_DEFINITION = (
    "sigmoid(top cross-encoder score / temperature): how strongly the best passage matches "
    "the question. It is a retrieval signal used for the abstention gate, not a "
    "probability that the answer is correct."
)


class FilterIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    departments: list[str] = Field(default_factory=list, max_length=10)
    categories: list[str] = Field(default_factory=list, max_length=10)
    date_from: dt.date | None = None
    date_to: dt.date | None = None
    doc_ids: list[str] = Field(default_factory=list, max_length=50)


class StrategyIn(BaseModel):
    """Overrides on top of the served retrieval configuration (all optional)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank", "bm25_rerank"] | None = None
    fusion: Fusion | None = None
    bm25_weight: float | None = Field(default=None, ge=0, le=10)
    dense_weight: float | None = Field(default=None, ge=0, le=10)
    score_norm: ScoreNorm | None = None
    rrf_k: int | None = Field(default=None, ge=1, le=1000)
    rerank_depth: int | None = Field(default=None, ge=1, le=100)
    rerank_weight: float | None = Field(default=None, ge=0, le=1)
    expansion: Expansion | None = None
    parent: ParentMode | None = None
    rerank_min_score: float | None = None
    dense_min_similarity: float | None = Field(default=None, ge=-1, le=1)


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": EXAMPLE},
                {
                    "question": "And for low-risk customers?",
                    "history": [EXAMPLE],
                    "strategy": {"mode": "bm25", "expansion": "rm3"},
                },
            ]
        },
    )

    question: str = Field(min_length=3, max_length=1000)
    history: list[str] = Field(
        default_factory=list, max_length=5, description="Earlier questions (oldest first)."
    )
    filters: FilterIn | None = Field(default=None, description="Metadata filter to apply.")
    apply_suggested_filters: bool = Field(
        default=False, description="Apply the filters detected in the question (shown either way)."
    )
    use_rewrite: bool = Field(
        default=True, description="Retrieve with the follow-up rewrite when one is made."
    )
    strategy: StrategyIn | None = None
    top_k: int | None = Field(default=None, ge=1, le=10, description="Passages given to the LLM.")
    backend: Backend | None = None
    gate: bool = Field(default=True, description="Refuse when retrieval confidence is low.")
    refuse_out_of_scope: bool = Field(
        default=True, description="Refuse questions about other regulators / unrelated topics."
    )
    abstain_if_unsupported: bool = Field(
        default=False,
        description="Withhold an answer whose claims the sources do not support (otherwise "
        "the unsupported claims are listed).",
    )


class PassageOut(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    title: str | None = None
    date: str | None = None
    department: str | None = None
    circular_number: str | None = None
    ref_number: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    section: str | None = None
    text: str
    score: float
    stage_scores: dict[str, float] = Field(
        default_factory=dict,
        description="bm25 score, dense cosine, rrf / fused score, rerank logit (when used).",
    )
    stage_ranks: dict[str, int] = Field(default_factory=dict)
    cited: bool = False


class ConfidenceOut(BaseModel):
    probability: float | None = None
    top_score: float | None = None
    threshold: float | None = None
    definition: str = CONFIDENCE_DEFINITION


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[PassageOut]
    passages: list[PassageOut]
    verification: VerificationReport | None = None
    query: QueryAnalysis
    context: ContextReport | None = None
    retrieval: RetrievalTrace | None = None
    confidence: ConfidenceOut
    mode: str
    backend: str | None = None
    model: str | None = None
    timings_ms: dict[str, float]
    usage: Usage
    cost: CostOut
    cached: bool = False


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"question": EXAMPLE}]}
    )

    question: str = Field(min_length=3, max_length=1000)
    history: list[str] = Field(default_factory=list, max_length=5)
    filters: FilterIn | None = None
    strategy: StrategyIn | None = None
    top_k: int = Field(default=10, ge=1, le=20)


class RetrieveResponse(BaseModel):
    request_id: str
    query: QueryAnalysis
    passages: list[PassageOut]
    retrieval: RetrievalTrace | None = None
    confidence: ConfidenceOut
    timings_ms: dict[str, float]


class SourceIn(BaseModel):
    n: int = Field(ge=1, le=50)
    chunk_id: str | None = Field(default=None, description="Look the passage up in the index.")
    doc_id: str | None = None
    text: str | None = Field(default=None, max_length=20000)


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=5000)
    sources: list[SourceIn] = Field(min_length=1, max_length=20)
    threshold: float | None = Field(default=None, ge=0, le=1)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)
    strategies: list[StrategyIn] = Field(min_length=1, max_length=4)
    top_k: int = Field(default=5, ge=1, le=10)


class CompareItem(BaseModel):
    label: str
    params_hash: str
    passages: list[PassageOut]
    timings_ms: dict[str, float]
    error: str | None = None


class CompareResponse(BaseModel):
    request_id: str
    results: list[CompareItem]
    overlap: list[list[float]] = Field(description="Jaccard overlap of the top-k chunk ids.")


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies: list[StrategyIn] = Field(min_length=1, max_length=4)
    split: Literal["dev", "test"] = "dev"
    limit: int = Field(default=50, ge=1, le=200)


class EvalRunResponse(BaseModel):
    request_id: str
    split: str
    questions: int
    rows: list[dict[str, Any]]
    note: str


class DocumentOut(BaseModel):
    metadata: DocIntel
    versions: list[str]
    original: str
    latest_in_corpus: LatestInCorpus
    changed_by: list[Edge]
    changes: list[Edge]
    consolidated_into: list[str]
    related: list[tuple[str, str]]
    unresolved_references: int


class RelationsOut(BaseModel):
    doc_id: str
    outgoing: list[Edge]
    incoming: list[Edge]
    unresolved: list[dict[str, Any]]


class Ready(BaseModel):
    ready: bool
    index_loaded: bool
    relation_graph: bool
    checks: dict[str, bool]
