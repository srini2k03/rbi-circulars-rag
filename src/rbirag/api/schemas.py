"""Request/response models (with OpenAPI examples for the Swagger UI)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rbirag.config import RetrievalMode
from rbirag.document import Document

Backend = Literal["groq", "gemini", "ollama", "anthropic", "fake"]
JobState = Literal["queued", "running", "succeeded", "failed"]

EXAMPLE_QUESTION = "How often must banks update KYC for high-risk customers?"


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": EXAMPLE_QUESTION},
                {
                    "question": "What is the late submission fee for ODI reporting?",
                    "mode": "hybrid_rerank",
                    "top_k": 5,
                    "backend": "groq",
                    "threshold": 0.5,
                },
            ]
        }
    )

    question: str = Field(min_length=3, max_length=2000, description="The question to answer.")
    mode: RetrievalMode | None = Field(
        default=None, description="Retrieval mode (default from settings: hybrid_rerank)."
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Sources given to the LLM.")
    backend: Backend | None = Field(
        default=None, description="LLM backend (default from settings: groq)."
    )
    gate: bool = Field(default=True, description="Refuse when retrieval confidence is low.")
    threshold: float | None = Field(
        default=None, ge=0, le=1, description="Confidence-gate threshold override."
    )


class Citation(BaseModel):
    n: int = Field(description="Source number as used in the answer, e.g. [1].")
    chunk_id: str
    doc_id: str
    title: str | None = None
    date: str | None = None
    circular_number: str | None = None
    url: str | None = Field(default=None, description="RBI notification page.")
    pdf_url: str | None = None
    section: str | None = None
    score: float
    text: str
    cited: bool = Field(description="Whether the answer cites this source.")


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class CostOut(BaseModel):
    usd: float = 0.0
    inr: float = 0.0


class QueryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request_id": "6f1c1c5e6a8e4c39",
                    "question": EXAMPLE_QUESTION,
                    "answer": "At least once every two years for high-risk customers [1].",
                    "refused": False,
                    "refusal_reason": None,
                    "citations": [
                        {
                            "n": 1,
                            "chunk_id": "rbi-12655::section::0002",
                            "doc_id": "rbi-12655",
                            "title": "Master Direction - Know Your Customer (KYC) Directions",
                            "date": "2025-06-12",
                            "circular_number": "RBI/2025-26/41",
                            "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12655&Mode=0",
                            "score": 4.2,
                            "text": "2. Periodic updation of KYC shall be carried out ...",
                            "cited": True,
                        }
                    ],
                    "sources": [],
                    "invalid_citations": [],
                    "uncited_sentences": [],
                    "confidence": 0.93,
                    "mode": "hybrid_rerank",
                    "backend": "groq",
                    "model": "openai/gpt-oss-20b",
                    "timings_ms": {
                        "retrieve": 41.2,
                        "rerank": 96.0,
                        "generate": 812.5,
                        "total": 949.7,
                    },
                    "usage": {"input_tokens": 1874, "output_tokens": 61},
                    "cost": {"usd": 0.0, "inr": 0.0},
                }
            ]
        }
    )

    request_id: str
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation] = Field(description="Sources cited in the answer.")
    sources: list[Citation] = Field(description="All sources given to the LLM.")
    invalid_citations: list[int]
    uncited_sentences: list[str]
    confidence: float | None
    mode: str
    backend: str | None
    model: str | None
    timings_ms: dict[str, float]
    usage: Usage
    cost: CostOut
    cached: bool = Field(default=False, description="Answer served from the LLM cache.")


class IngestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"documents_path": "data/processed/documents.jsonl"},
                {"urls": ["https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12701&Mode=0"]},
            ]
        }
    )

    documents_path: str | None = Field(
        default=None, description="Server-side path of a documents JSONL (as written by ingest)."
    )
    documents: list[Document] | None = Field(default=None, description="Documents inline.")
    urls: list[str] | None = Field(
        default=None, max_length=200, description="RBI notification URLs to fetch and parse."
    )
    replace: bool = Field(default=False, description="Replace the corpus instead of adding to it.")


class Job(BaseModel):
    job_id: str
    state: JobState
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    documents_added: int = 0
    documents_total: int | None = None
    chunks: int | None = None
    failed_urls: list[str] = Field(default_factory=list)
    error: str | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    index_loaded: bool
    documents: int
    chunks: int
    chunker: str
    dense_index: bool
    embedder: str
    reranker: str | None
    default_backend: str
    backends: dict[str, bool] = Field(description="Backend -> configured (key present / local).")
    version: str
