"""Application settings, driven by environment variables and an optional ``.env`` file.

Every field can be overridden with an ``RBIRAG_``-prefixed environment variable, e.g.
``RBIRAG_FETCH_START_ID=12800`` or ``RBIRAG_RETRIEVAL_MODE=hybrid``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
ChunkerName = Literal["fixed", "semantic", "section"]
TokenizerName = Literal["regex", "whitespace", "tiktoken"]
EmbedderBackend = Literal["sentence-transformers", "fake"]
RerankerBackend = Literal["cross-encoder", "fake", "none"]

DEFAULT_USER_AGENT = (
    "rbi-circulars-rag research crawler (+https://github.com/srini2k03/rbi-circulars-rag)"
)


class Settings(BaseSettings):
    """All tunables in one place. Grouped by pipeline stage."""

    model_config = SettingsConfigDict(
        env_prefix="RBIRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("indexes")

    # ------------------------------------------------------------------ fetch
    fetch_base_url: str = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={id}&Mode=0"
    # Scan downwards from ``fetch_start_id``; IDs above the newest notification simply miss.
    fetch_start_id: int = Field(default=13050, ge=1)
    fetch_min_id: int = Field(default=9000, ge=1)
    fetch_max_docs: int = Field(default=2200, ge=1)
    fetch_rate_limit_seconds: float = Field(default=1.5, ge=1.0)
    fetch_max_retries: int = Field(default=4, ge=0)
    fetch_backoff_base_seconds: float = Field(default=2.0, gt=0)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0)
    fetch_max_consecutive_misses: int = Field(default=300, ge=1)
    fetch_pdfs: bool = True
    user_agent: str = DEFAULT_USER_AGENT

    # --------------------------------------------------------------- chunking
    chunker: ChunkerName = "section"
    tokenizer: TokenizerName = "regex"
    tiktoken_encoding: str = "cl100k_base"
    chunk_size: int = Field(default=512, ge=8)
    chunk_overlap: int = Field(default=64, ge=0)
    semantic_breakpoint_percentile: float = Field(default=10.0, gt=0, lt=100)
    semantic_min_tokens: int = Field(default=64, ge=1)
    section_min_tokens: int = Field(default=64, ge=1)

    # ----------------------------------------------------------------- models
    embedder: EmbedderBackend = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=32, ge=1)
    fake_embedding_dim: int = Field(default=256, ge=8)
    reranker: RerankerBackend = "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str | None = None

    # ----------------------------------------------------------------- qdrant
    qdrant_url: str | None = None
    qdrant_path: Path | None = None  # embedded on-disk mode; ``None`` + no URL => in-memory
    qdrant_collection: str = "rbi_chunks"
    qdrant_upsert_batch_size: int = Field(default=128, ge=1)

    # -------------------------------------------------------------- retrieval
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=50, ge=1)
    rerank_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    bm25_weight: float = Field(default=1.0, ge=0)
    dense_weight: float = Field(default=1.0, ge=0)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    confidence_temperature: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.fetch_min_id > self.fetch_start_id:
            raise ValueError("fetch_min_id must be <= fetch_start_id")
        return self

    # Convenience paths -------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / "manifest.jsonl"

    @property
    def documents_path(self) -> Path:
        return self.processed_dir / "documents.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cleared in tests via ``get_settings.cache_clear()``)."""
    return Settings()
