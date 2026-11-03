"""Fixed-size token windows with overlap."""

from __future__ import annotations

from rbirag.chunking.base import (
    Chunk,
    RegexTokenizer,
    Tokenizer,
    make_chunks,
    token_windows,
)
from rbirag.document import Document


class FixedSizeChunker:
    name = "fixed"

    def __init__(
        self, chunk_size: int = 512, overlap: int = 64, tokenizer: Tokenizer | None = None
    ) -> None:
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer or RegexTokenizer()

    def windows(
        self, text: str, start: int = 0, end: int | None = None
    ) -> list[tuple[int, int, int]]:
        """``(char_start, char_end, n_tokens)`` windows over ``text[start:end]``."""
        end = len(text) if end is None else end
        spans = [(s + start, e + start) for s, e in self.tokenizer.spans(text[start:end])]
        return [
            (spans[i][0], spans[j - 1][1], j - i)
            for i, j in token_windows(spans, self.chunk_size, self.overlap)
        ]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.spans(text))

    def chunk(self, doc: Document) -> list[Chunk]:
        return make_chunks(doc, [(s, e, n, None) for s, e, n in self.windows(doc.text)], self.name)
