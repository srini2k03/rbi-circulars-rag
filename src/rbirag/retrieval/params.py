"""Every retrieval knob in one validated, hashable object.

:class:`RetrievalParams` is what the API serves, what the benchmark tunes on the dev split
and what the experiment log records (by :meth:`RetrievalParams.config_hash`), so a reported
number can always be traced to the exact configuration that produced it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from rbirag.config import Settings

Fusion = Literal["rrf", "weighted"]
ScoreNorm = Literal["minmax", "zscore"]
Expansion = Literal["none", "acronyms", "rm3", "acronyms+rm3"]
ParentMode = Literal["none", "window"]

# Modes that need each component.
BM25_MODES = frozenset({"bm25", "bm25_rerank", "hybrid", "hybrid_rerank"})
DENSE_MODES = frozenset({"dense", "hybrid", "hybrid_rerank"})
RERANK_MODES = frozenset({"bm25_rerank", "hybrid_rerank"})


class RetrievalParams(BaseModel):
    """Retrieval configuration. Defaults reproduce the original pipeline (RRF k=60, equal
    weights, 50 candidates per retriever, rerank the top 20, no expansion, no thresholds)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank", "bm25_rerank"] = "hybrid_rerank"
    # candidates taken from each first-stage retriever before fusion / reranking
    bm25_candidates: int = Field(default=50, ge=1, le=1000)
    dense_candidates: int = Field(default=50, ge=1, le=1000)
    fusion: Fusion = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    bm25_weight: float = Field(default=1.0, ge=0)
    dense_weight: float = Field(default=1.0, ge=0)
    # weighted fusion only: how raw scores are normalised per query before weighting
    score_norm: ScoreNorm = "minmax"
    # cross-encoder: how many fused candidates are reranked, and the weight of the
    # (min-max normalised) cross-encoder score against the normalised first-stage score
    # (1.0 = pure reranking, the original behaviour)
    rerank_depth: int = Field(default=20, ge=1, le=1000)
    rerank_weight: float = Field(default=1.0, ge=0, le=1)
    # evidence thresholds (None = off): passages below them never reach the context
    rerank_min_score: float | None = None  # cross-encoder logit
    dense_min_similarity: float | None = Field(default=None, ge=-1, le=1)  # cosine
    # query expansion (BM25 arm only); the expanded terms are reported with every result
    expansion: Expansion = "none"
    acronym_weight: float = Field(default=0.5, ge=0, le=2)
    rm3_fb_docs: int = Field(default=5, ge=1, le=50)
    rm3_fb_terms: int = Field(default=20, ge=1, le=200)
    rm3_orig_weight: float = Field(default=0.6, ge=0, le=1)
    # parent-child: return a wider window of the source document around each hit
    parent: ParentMode = "none"
    parent_tokens: int = Field(default=1024, ge=16, le=8192)
    # final context size (passages passed to the generator)
    context_k: int = Field(default=5, ge=1, le=50)
    # confidence gate (reranking modes only): sigmoid(top logit / temperature) >= threshold
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    confidence_temperature: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> RetrievalParams:
        if self.fusion == "weighted" and self.bm25_weight + self.dense_weight == 0:
            raise ValueError("weighted fusion needs a positive bm25_weight or dense_weight")
        return self

    @property
    def uses_bm25(self) -> bool:
        return self.mode in BM25_MODES

    @property
    def uses_dense(self) -> bool:
        return self.mode in DENSE_MODES

    @property
    def uses_reranker(self) -> bool:
        return self.mode in RERANK_MODES

    def config_hash(self) -> str:
        """Short stable hash of every field (key order independent)."""
        blob = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        """A compact human-readable name, e.g. ``hybrid[weighted a=0.7]+rerank@30``."""
        parts = [self.mode.replace("_rerank", "")]
        if self.mode.startswith("hybrid"):
            if self.fusion == "rrf":
                parts.append(f"[rrf k={self.rrf_k} w={self.bm25_weight:g}/{self.dense_weight:g}]")
            else:
                total = self.bm25_weight + self.dense_weight
                parts.append(f"[{self.score_norm} a={self.bm25_weight / total:.2f}]")
        if self.expansion != "none" and self.uses_bm25:
            parts.append(f"+{self.expansion}")
        if self.uses_reranker:
            parts.append(f"+rerank@{self.rerank_depth}")
            if self.rerank_weight < 1:
                parts.append(f"(w={self.rerank_weight:g})")
        if self.parent != "none":
            parts.append(f"+parent{self.parent_tokens}")
        return "".join(parts)

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalParams:
        """The served configuration (every field can be set with an ``RBIRAG_`` env var)."""
        return cls(
            mode=settings.retrieval_mode,
            bm25_candidates=settings.bm25_candidates or settings.candidate_k,
            dense_candidates=settings.dense_candidates or settings.candidate_k,
            fusion=settings.fusion,
            rrf_k=settings.rrf_k,
            bm25_weight=settings.bm25_weight,
            dense_weight=settings.dense_weight,
            score_norm=settings.score_norm,
            rerank_depth=settings.rerank_k,
            rerank_weight=settings.rerank_weight,
            rerank_min_score=settings.rerank_min_score,
            dense_min_similarity=settings.dense_min_similarity,
            expansion=settings.query_expansion,
            acronym_weight=settings.acronym_weight,
            rm3_fb_docs=settings.rm3_fb_docs,
            rm3_fb_terms=settings.rm3_fb_terms,
            rm3_orig_weight=settings.rm3_orig_weight,
            parent=settings.parent_context,
            parent_tokens=settings.parent_tokens,
            context_k=settings.answer_top_k,
            confidence_threshold=settings.confidence_threshold,
            confidence_temperature=settings.confidence_temperature,
        )


class RetrievalFilter(BaseModel):
    """Metadata filter applied *before* ranking (BM25 mask / Qdrant payload filter), so a
    filtered search returns the best matches inside the filter, not a thinned top-k."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    departments: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    date_from: dt.date | None = None
    date_to: dt.date | None = None
    doc_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> RetrievalFilter:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self

    @property
    def is_empty(self) -> bool:
        return not (
            self.departments or self.categories or self.date_from or self.date_to or self.doc_ids
        )

    def matches(self, metadata: Mapping[str, object]) -> bool:
        """Whether a chunk with this document metadata passes the filter. A document
        without a date (or department, category) fails a filter on that field."""
        if self.doc_ids and metadata.get("doc_id") not in self.doc_ids:
            return False
        if self.departments:
            dept = str(metadata.get("department") or "").lower()
            if dept not in {d.lower() for d in self.departments}:
                return False
        if self.categories:
            cat = str(metadata.get("category") or "").lower()
            if cat not in {c.lower() for c in self.categories}:
                return False
        if self.date_from or self.date_to:
            raw = metadata.get("date")
            if not raw:
                return False
            try:
                day = dt.date.fromisoformat(str(raw))
            except ValueError:
                return False
            if self.date_from and day < self.date_from:
                return False
            if self.date_to and day > self.date_to:
                return False
        return True
