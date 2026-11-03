"""Section-aware chunking: respect headings and numbered paragraphs."""

from __future__ import annotations

from dataclasses import dataclass

from rbirag.chunking.base import Chunk, make_chunks
from rbirag.chunking.fixed import FixedSizeChunker
from rbirag.document import Document, Section
from rbirag.ingest.metadata import detect_sections


class SectionChunker:
    """One chunk per section, merging small neighbours and splitting large sections.

    * Segments are ``[section.start, next_section.start)``; text before the first section
      (letter header) is its own segment.
    * Consecutive segments are merged while the total stays <= ``max_tokens`` and the
      running chunk is still below ``min_tokens`` (so small numbered paras group together,
      but a chunk never mixes a large section with the next one).
    * Segments larger than ``max_tokens`` are split with the fixed-size chunker.
    """

    name = "section"

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        min_tokens: int = 64,
        overlap: int = 64,
        fixed: FixedSizeChunker | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.fixed = fixed or FixedSizeChunker(max_tokens, min(overlap, max_tokens - 1))

    @staticmethod
    def segments(text: str, sections: list[Section]) -> list[tuple[int, int, str | None]]:
        ordered = sorted(sections, key=lambda s: s.start)
        bounds: list[tuple[int, int, str | None]] = []
        if not ordered:
            return [(0, len(text), None)]
        if ordered[0].start > 0:
            bounds.append((0, ordered[0].start, None))
        for i, sec in enumerate(ordered):
            end = ordered[i + 1].start if i + 1 < len(ordered) else len(text)
            label = f"{sec.number} {sec.heading}".strip() if sec.number else sec.heading
            bounds.append((sec.start, end, label))
        return [(s, e, label) for s, e, label in bounds if text[s:e].strip()]

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        sections = doc.sections or detect_sections(text)
        segs = [
            _Piece(s, e, self.fixed.count_tokens(text[s:e]), label)
            for s, e, label in self.segments(text, sections)
        ]

        out: list[_Piece] = []
        cur: _Piece | None = None
        for seg in segs:
            if seg.n_tokens > self.max_tokens:
                if cur is not None:
                    out.append(cur)
                    cur = None
                out.extend(
                    _Piece(ws, we, wn, seg.label)
                    for ws, we, wn in self.fixed.windows(text, seg.start, seg.end)
                )
            elif cur is None:
                cur = seg
            elif cur.n_tokens < self.min_tokens and cur.n_tokens + seg.n_tokens <= self.max_tokens:
                cur = _Piece(
                    cur.start, seg.end, cur.n_tokens + seg.n_tokens, cur.label or seg.label
                )
            else:
                out.append(cur)
                cur = seg
        if cur is not None:
            out.append(cur)
        return make_chunks(doc, [(p.start, p.end, p.n_tokens, p.label) for p in out], self.name)


@dataclass(frozen=True)
class _Piece:
    start: int
    end: int
    n_tokens: int
    label: str | None
