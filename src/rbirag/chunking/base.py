"""Shared chunking primitives: the :class:`Chunk` model, the :class:`Chunker` protocol,
offset-preserving tokenizers and a sentence splitter.

Invariant for every chunker: ``chunk.text == doc.text[chunk.start:chunk.end]``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rbirag.document import Document, MetadataValue

Span = tuple[int, int]


class Chunk(BaseModel):
    """A retrievable unit of text plus the metadata of the document it came from."""

    chunk_id: str
    doc_id: str
    index: int = Field(ge=0)
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    n_tokens: int = Field(ge=0)
    chunker: str
    section: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    def payload(self) -> dict[str, MetadataValue]:
        """Flat payload for vector stores: chunk fields + document metadata."""
        return {
            **self.metadata,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "section": self.section,
            "chunker": self.chunker,
            "text": self.text,
        }


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


# ----------------------------------------------------------------------------- tokenizers


class Tokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[Span]:
        """Character spans ``(start, end)`` of each token, in order, non-overlapping."""
        ...


class RegexTokenizer:
    """Words (incl. numbers like ``21.06.201``) and individual punctuation marks."""

    name = "regex"
    _pattern = re.compile(r"\w+(?:[.\-/']\w+)*|[^\w\s]")

    def spans(self, text: str) -> list[Span]:
        return [m.span() for m in self._pattern.finditer(text)]


class WhitespaceTokenizer:
    name = "whitespace"
    _pattern = re.compile(r"\S+")

    def spans(self, text: str) -> list[Span]:
        return [m.span() for m in self._pattern.finditer(text)]


class TiktokenTokenizer:
    """BPE tokens via ``tiktoken`` (optional extra; downloads its vocabulary on first use)."""

    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    def spans(self, text: str) -> list[Span]:
        tokens = self._enc.encode(text, disallowed_special=())
        if not tokens:
            return []
        decoded, offsets = self._enc.decode_with_offsets(tokens)
        if decoded != text:  # pragma: no cover - only for invalid UTF-8 round-trips
            return WhitespaceTokenizer().spans(text)
        ends = [*offsets[1:], len(text)]
        out: list[Span] = []
        for s, e in zip(offsets, ends, strict=True):
            # Trim surrounding whitespace so spans point at visible text.
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                out.append((s, e))
        return out


def get_tokenizer(name: str = "regex", *, tiktoken_encoding: str = "cl100k_base") -> Tokenizer:
    if name == "regex":
        return RegexTokenizer()
    if name == "whitespace":
        return WhitespaceTokenizer()
    if name == "tiktoken":
        return TiktokenTokenizer(tiktoken_encoding)
    raise ValueError(f"unknown tokenizer: {name!r}")


# ----------------------------------------------------------------------------- sentences

_ABBREVIATIONS = (
    "no", "nos", "rs", "re", "viz", "i.e", "e.g", "etc", "dr", "mr", "mrs", "ms", "sr", "jr",
    "vs", "cf", "para", "paras", "sec", "art", "ch", "vol", "fig", "dt", "ltd", "pvt", "co",
    "govt", "dept", "approx", "inc", "st", "a.p", "u.s", "rbi",
)  # fmt: skip
_SENTENCE_END = re.compile(r"(?<=[.?!;])\s+(?=[\"'(\[]?[A-Z0-9]|\([a-z]{1,4}\)|\([0-9]{1,3}\))|\n+")


def split_sentences(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Sentence spans within ``text[start:end]`` (offsets are absolute into ``text``).

    Splits on terminal punctuation followed by a capital/digit, and on newlines; avoids
    splitting after common abbreviations (``No.``, ``Rs.``, ``i.e.``) and decimal numbers.
    """
    end = len(text) if end is None else end
    spans: list[Span] = []
    seg_start = start
    for m in _SENTENCE_END.finditer(text, start, end):
        before = text[seg_start : m.start()]
        if "\n" not in m.group(0):
            last_word = re.search(r"(\S+)$", before)
            word = last_word.group(1).lower().rstrip(".") if last_word else ""
            if word in _ABBREVIATIONS or re.fullmatch(r"\(?[a-z0-9]{1,3}\)?", word):
                continue
        spans.append((seg_start, m.start()))
        seg_start = m.end()
    spans.append((seg_start, end))
    out: list[Span] = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


# ----------------------------------------------------------------------------- helpers


def trim_span(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def make_chunks(
    doc: Document,
    spans: Sequence[tuple[int, int, int, str | None]],
    chunker: str,
) -> list[Chunk]:
    """Build :class:`Chunk` objects from ``(start, end, n_tokens, section)`` tuples."""
    metadata = doc.metadata()
    chunks: list[Chunk] = []
    for start, end, n_tokens, section in spans:
        s, e = trim_span(doc.text, start, end)
        if e <= s:
            continue
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{chunker}::{idx:04d}",
                doc_id=doc.doc_id,
                index=idx,
                text=doc.text[s:e],
                start=s,
                end=e,
                n_tokens=n_tokens,
                chunker=chunker,
                section=section,
                metadata=metadata,
            )
        )
    return chunks


def token_windows(spans: Sequence[Span], size: int, overlap: int) -> list[tuple[int, int]]:
    """Index ranges ``[i, j)`` over ``spans`` for fixed windows with ``overlap`` tokens."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    n = len(spans)
    if n == 0:
        return []
    step = size - overlap
    windows: list[tuple[int, int]] = []
    i = 0
    while True:
        j = min(i + size, n)
        windows.append((i, j))
        if j >= n:
            break
        i += step
    return windows
