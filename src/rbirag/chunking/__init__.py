"""Chunkers: fixed-size, semantic and section-aware."""

from __future__ import annotations

from rbirag.chunking.base import Chunk, Chunker, get_tokenizer
from rbirag.chunking.fixed import FixedSizeChunker
from rbirag.chunking.section import SectionChunker
from rbirag.chunking.semantic import SemanticChunker
from rbirag.config import Settings
from rbirag.index.embedder import Embedder

__all__ = [
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "SectionChunker",
    "SemanticChunker",
    "build_chunker",
]


def build_chunker(name: str, settings: Settings, *, embedder: Embedder | None = None) -> Chunker:
    tokenizer = get_tokenizer(settings.tokenizer, tiktoken_encoding=settings.tiktoken_encoding)
    fixed = FixedSizeChunker(settings.chunk_size, settings.chunk_overlap, tokenizer)
    if name == "fixed":
        return fixed
    if name == "section":
        return SectionChunker(
            max_tokens=settings.chunk_size,
            min_tokens=settings.section_min_tokens,
            fixed=fixed,
        )
    if name == "semantic":
        if embedder is None:
            raise ValueError("the semantic chunker needs an embedder")
        return SemanticChunker(
            embedder,
            percentile=settings.semantic_breakpoint_percentile,
            min_tokens=settings.semantic_min_tokens,
            max_tokens=settings.chunk_size,
            fixed=fixed,
        )
    raise ValueError(f"unknown chunker: {name!r}")
