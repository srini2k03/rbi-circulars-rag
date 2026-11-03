"""Semantic chunking: break where adjacent-sentence similarity drops below a percentile."""

from __future__ import annotations

import numpy as np

from rbirag.chunking.base import Chunk, Span, make_chunks, split_sentences
from rbirag.chunking.fixed import FixedSizeChunker
from rbirag.document import Document
from rbirag.index.embedder import Embedder


class SemanticChunker:
    """Greg Kamradt-style semantic splitting with an injected :class:`Embedder`.

    1. Split into sentences (with offsets).
    2. Embed each sentence; cosine similarity between neighbours.
    3. Break wherever similarity < ``percentile``-th percentile of the document's sims.
    4. Merge groups smaller than ``min_tokens``; split groups larger than ``max_tokens``
       with the fixed chunker.
    """

    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        *,
        percentile: float = 10.0,
        min_tokens: int = 64,
        max_tokens: int = 512,
        overlap: int = 0,
        fixed: FixedSizeChunker | None = None,
    ) -> None:
        if not 0 < percentile < 100:
            raise ValueError("percentile must be in (0, 100)")
        self.embedder = embedder
        self.percentile = percentile
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.fixed = fixed or FixedSizeChunker(max_tokens, overlap)

    def breakpoints(self, sims: np.ndarray) -> list[int]:
        """Indices ``i`` such that a new group starts at sentence ``i + 1``."""
        if sims.size == 0:
            return []
        threshold = float(np.percentile(sims, self.percentile))
        return [i for i, s in enumerate(sims.tolist()) if s < threshold]

    def _groups(self, text: str, sentences: list[Span]) -> list[Span]:
        if len(sentences) <= 1:
            return list(sentences)
        vecs = self.embedder.embed_documents([text[s:e] for s, e in sentences])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
        cuts = set(self.breakpoints(sims))
        groups: list[Span] = []
        g_start = sentences[0][0]
        for i, (_, e) in enumerate(sentences):
            if i in cuts or i == len(sentences) - 1:
                groups.append((g_start, e))
                if i + 1 < len(sentences):
                    g_start = sentences[i + 1][0]
        return groups

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        groups = self._groups(text, split_sentences(text))

        # Merge undersized groups forward (the last one backward).
        merged: list[Span] = []
        for g in groups:
            if merged and self.fixed.count_tokens(text[merged[-1][0] : merged[-1][1]]) < (
                self.min_tokens
            ):
                merged[-1] = (merged[-1][0], g[1])
            else:
                merged.append(g)
        if len(merged) > 1 and self.fixed.count_tokens(text[merged[-1][0] : merged[-1][1]]) < (
            self.min_tokens
        ):
            last = merged.pop()
            merged[-1] = (merged[-1][0], last[1])

        out: list[tuple[int, int, int, str | None]] = []
        for s, e in merged:
            n = self.fixed.count_tokens(text[s:e])
            if n > self.max_tokens:
                out.extend((ws, we, wn, None) for ws, we, wn in self.fixed.windows(text, s, e))
            else:
                out.append((s, e, n, None))
        return make_chunks(doc, self._absorb_small(text, out), self.name)

    def _absorb_small(
        self, text: str, pieces: list[tuple[int, int, int, str | None]]
    ) -> list[tuple[int, int, int, str | None]]:
        """Fold undersized pieces (e.g. the tail of a fixed split) into a neighbour if it fits."""
        result: list[tuple[int, int, int, str | None]] = []
        i = 0
        while i < len(pieces):
            s, e, n, label = pieces[i]
            if n < self.min_tokens and i + 1 < len(pieces):
                ns, ne, _, _ = pieces[i + 1]
                merged_n = self.fixed.count_tokens(text[s:ne])
                if ns >= s and merged_n <= self.max_tokens:
                    pieces[i + 1] = (s, ne, merged_n, label)
                    i += 1
                    continue
            if n < self.min_tokens and result:
                ps, _, _, plabel = result[-1]
                merged_n = self.fixed.count_tokens(text[ps:e])
                if merged_n <= self.max_tokens:
                    result[-1] = (ps, e, merged_n, plabel)
                    i += 1
                    continue
            result.append((s, e, n, label))
            i += 1
        return result
