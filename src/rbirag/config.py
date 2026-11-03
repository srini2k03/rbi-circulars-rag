"""Application settings, driven by environment variables and an optional ``.env`` file.

Every field can be overridden with an ``RBIRAG_``-prefixed environment variable, e.g.
``RBIRAG_FETCH_START_ID=12800`` or ``RBIRAG_RETRIEVAL_MODE=hybrid``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from rbirag.providers import get_preset

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
ChunkerName = Literal["fixed", "semantic", "section"]
TokenizerName = Literal["regex", "whitespace", "tiktoken"]
EmbedderBackend = Literal["sentence-transformers", "fake"]
RerankerBackend = Literal["cross-encoder", "fake", "none"]
LLMBackend = Literal["openai_compatible", "anthropic", "ollama", "fake"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ModelPrice(BaseModel):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)


# USD per 1M tokens. Groq and Gemini models are used on their free tiers, so they cost
# $0 (tokens are still recorded). Anthropic first-party list prices as of 2026-10 are kept for the
# optional Anthropic backend; cache writes 1.25x input, cache reads 0.1x input. Claude Haiku
# 5.5's price applies to prompts up to 100K tokens. Local Ollama models cost nothing.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    # Groq free tier: $0 (tokens are still recorded).
    "llama-3.3-70b-versatile": ModelPrice(input=0.0, output=0.0),
    "llama-3.1-8b-instant": ModelPrice(input=0.0, output=0.0),
    "openai/gpt-oss-120b": ModelPrice(input=0.0, output=0.0),
    "openai/gpt-oss-20b": ModelPrice(input=0.0, output=0.0),
    "qwen/qwen3-32b": ModelPrice(input=0.0, output=0.0),
    "qwen/qwen3.8-27b": ModelPrice(input=0.0, output=0.0),
    "gemini-3.8-flash": ModelPrice(input=0.0, output=0.0),
    "gemini-3.7-flash": ModelPrice(input=0.0, output=0.0),
    "gemini-3.6-flash": ModelPrice(input=0.0, output=0.0),
    "gemini-3.5-flash": ModelPrice(input=0.0, output=0.0),
    "gemini-3.5-flash-lite": ModelPrice(input=0.0, output=0.0),
    "gemini-3.1-flash-lite": ModelPrice(input=0.0, output=0.0),
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
}

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

    # ------------------------------------------------------------- generation
    llm_backend: LLMBackend = "openai_compatible"
    # OpenAI-compatible chat-completions provider. "groq" (default) and "gemini" are
    # presets (see rbirag.providers); "custom" needs oai_base_url/oai_api_key_env/models.
    # Empty strings and ["auto"] mean "use the provider preset". Keys are read from the
    # environment variable *named* here and are never stored in settings, logged or echoed.
    llm_provider: Literal["groq", "gemini", "custom"] = "groq"
    oai_base_url: str = ""
    oai_api_key_env: str = ""
    oai_model: str = ""
    # Fallback chains (comma-separated in env vars): used when a model is overloaded
    # (503), not found / not permitted (404/403), or out of daily quota.
    oai_model_fallbacks: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["auto"])
    oai_reasoning_effort: str | None = None
    oai_json_mode: Literal["", "json_schema", "json_object", "none"] = ""
    oai_stream_usage: bool = True
    oai_timeout_seconds: float = Field(default=120.0, gt=0)
    # Client-side limits per model. 0 = the provider preset's (per-model) value.
    llm_rpm: float = Field(default=0.0, ge=0)
    llm_rpd: int = Field(default=0, ge=0)
    llm_tpm: int = Field(default=0, ge=0)
    llm_max_retries: int = Field(default=6, ge=0)
    llm_backoff_base_seconds: float = Field(default=5.0, gt=0)
    llm_switch_after_503: int = Field(default=2, ge=1)
    llm_model_cooldown_seconds: float = Field(default=300.0, ge=0)
    # Disk cache of every LLM call, keyed by a hash of the full request.
    llm_cache: bool = True
    llm_cache_dir: Path = Path(".cache/llm")
    # Optional Anthropic backend (used only when selected and ANTHROPIC_API_KEY is set).
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_effort: Effort = "low"
    anthropic_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_num_ctx: int = Field(default=8192, ge=512)
    ollama_timeout_seconds: float = Field(default=600.0, gt=0)
    max_answer_tokens: int = Field(default=1024, ge=16)
    answer_top_k: int = Field(default=5, ge=1)
    # The gate refuses without calling the LLM when retrieval confidence is below the
    # threshold. Only modes with a reranker produce a confidence.
    confidence_gate: bool = True
    prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICES))
    usd_to_inr: float = Field(default=88.0, gt=0)

    # -------------------------------------------------------------------- api
    # JSONL mirror of the API request log (None = in memory only).
    request_log_path: Path | None = None
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # ------------------------------------------------------------- evaluation
    eval_dir: Path = Path("data/eval")
    results_dir: Path = Path("results")
    qa_backend: LLMBackend = "openai_compatible"
    qa_model: str = ""
    qa_model_fallbacks: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["auto"])
    judge_backend: LLMBackend = "openai_compatible"
    judge_model: str = ""
    judge_model_fallbacks: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["auto"])
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    bootstrap_samples: int = Field(default=1000, ge=100)
    dev_fraction: float = Field(default=0.3, gt=0, lt=1)
    seed: int = 13

    @field_validator(
        "oai_model_fallbacks", "qa_model_fallbacks", "judge_model_fallbacks", mode="before"
    )
    @classmethod
    def _split_models(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                return json.loads(text)
            return [m.strip() for m in text.split(",") if m.strip()]
        return value

    @model_validator(mode="after")
    def _apply_provider_preset(self) -> Settings:
        """Fill provider-dependent settings left empty / "auto" from the preset."""
        preset = get_preset(self.llm_provider)
        if preset is None:  # custom provider: everything must be explicit
            required = ("oai_base_url", "oai_api_key_env", "oai_model", "qa_model", "judge_model")
            missing = [n for n in required if not getattr(self, n)]
            if missing:
                raise ValueError(f"llm_provider=custom requires: {', '.join(missing)}")
            for name in ("oai_model_fallbacks", "qa_model_fallbacks", "judge_model_fallbacks"):
                if getattr(self, name) == ["auto"]:
                    setattr(self, name, [])
            if not self.llm_rpm:
                self.llm_rpm = 10.0
            self.oai_json_mode = self.oai_json_mode or "json_object"
            if not self.llm_rpd:
                self.llm_rpd = 200
            return self
        self.oai_base_url = self.oai_base_url or preset.base_url
        self.oai_api_key_env = self.oai_api_key_env or preset.api_key_env
        self.oai_json_mode = self.oai_json_mode or preset.json_mode  # type: ignore[assignment]
        for model_field, chain in (
            ("oai_model", preset.generator),
            ("qa_model", preset.qa_writer),
            ("judge_model", preset.judge),
        ):
            fallbacks_field = f"{model_field}_fallbacks"
            explicit_model = bool(getattr(self, model_field))
            if not explicit_model:
                setattr(self, model_field, chain[0])
            if getattr(self, fallbacks_field) == ["auto"]:
                primary = getattr(self, model_field)
                # an explicitly chosen model keeps the preset chain minus itself
                setattr(self, fallbacks_field, [m for m in chain if m != primary])
        if not self.llm_rpm:
            self.llm_rpm = preset.rpm
        return self

    def rpd_for(self, model: str) -> int:
        """Requests/day budget for ``model`` (explicit setting, else the preset's)."""
        if self.llm_rpd:
            return self.llm_rpd
        preset = get_preset(self.llm_provider)
        return preset.rpd_for(model) if preset else 200

    def tpm_for(self, model: str) -> int:
        """Tokens/minute budget for ``model`` (0 = none)."""
        if self.llm_tpm:
            return self.llm_tpm
        preset = get_preset(self.llm_provider)
        return preset.tpm_for(model) if preset else 0

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
