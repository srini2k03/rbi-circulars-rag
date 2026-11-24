"""Parent-child retrieval: match on small passages, hand the generator a wider window.

A small chunk ranks well because its few hundred tokens are about the query; the answer,
however, often needs the surrounding paragraphs (conditions, exceptions, the table the
sentence refers to). :class:`ParentResolver` widens each retrieved passage with its
neighbouring passages of the same document until the window reaches ``parent_tokens``,
and merges windows that overlap, so the same text is never sent twice.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rbirag.chunking.base import Chunk
from rbirag.index.store import ChunkStore

if TYPE_CHECKING:
    from rbirag.retrieval.pipeline import ScoredChunk


class ParentResolver:
    def __init__(self, store: ChunkStore) -> None:
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in store:
            by_doc[chunk.doc_id].append(chunk)
        self._docs = {d: sorted(cs, key=lambda c: c.start) for d, cs in by_doc.items()}
        self._pos = {c.chunk_id: i for cs in self._docs.values() for i, c in enumerate(cs)}

    def window(self, chunk: Chunk, parent_tokens: int) -> tuple[int, int]:
        """Indexes ``[lo, hi]`` (inclusive) of the neighbouring chunks forming the window:
        grow alternately after and before the hit while the new chunk still fits."""
        chunks = self._docs[chunk.doc_id]
        i = self._pos[chunk.chunk_id]
        lo = hi = i
        tokens = chunk.n_tokens
        grew = True
        while grew:
            grew = False
            for j, side in ((hi + 1, "hi"), (lo - 1, "lo")):
                if 0 <= j < len(chunks):
                    extra = _new_tokens(chunks, lo, hi, j)
                    if tokens + extra <= parent_tokens:
                        tokens += extra
                        lo, hi = (lo, j) if side == "hi" else (j, hi)
                        grew = True
        return lo, hi

    def stitch(self, doc_id: str, lo: int, hi: int, child: Chunk) -> Chunk:
        """One chunk covering ``chunks[lo..hi]`` of ``doc_id`` (text rebuilt from the
        chunks: ``chunk.text == doc.text[start:end]`` makes overlaps exact)."""
        chunks = self._docs[doc_id][lo : hi + 1]
        text = chunks[0].text
        end = chunks[0].end
        for c in chunks[1:]:
            if c.start >= end:
                text += " " * (c.start - end) + c.text  # whitespace between windows
            elif c.end > end:
                text += c.text[end - c.start :]
            end = max(end, c.end)
        return Chunk(
            chunk_id=f"{child.chunk_id}~p{chunks[0].index}-{chunks[-1].index}",
            doc_id=doc_id,
            index=chunks[0].index,
            text=text,
            start=chunks[0].start,
            end=end,
            n_tokens=sum(c.n_tokens for c in chunks),
            chunker=f"{child.chunker}+parent",
            section=child.section,
            metadata=child.metadata,
        )

    def expand(self, results: list[ScoredChunk], parent_tokens: int) -> list[ScoredChunk]:
        """Replace each hit by its window; a hit whose window overlaps a better-ranked
        window of the same document is merged into it (dropped), so ranks stay dense."""
        taken: dict[str, list[tuple[int, int]]] = defaultdict(list)
        out: list[ScoredChunk] = []
        for r in results:
            lo, hi = self.window(r.chunk, parent_tokens)
            spans = taken[r.chunk.doc_id]
            if any(lo <= b and a <= hi for a, b in spans):
                continue
            spans.append((lo, hi))
            parent = self.stitch(r.chunk.doc_id, lo, hi, r.chunk)
            out.append(
                r.model_copy(
                    update={
                        "chunk": parent,
                        "rank": len(out) + 1,
                        "child_chunk_id": r.chunk.chunk_id,
                    }
                )
            )
        return out


def _new_tokens(chunks: list[Chunk], lo: int, hi: int, j: int) -> int:
    """Tokens chunk ``j`` adds to the window ``[lo, hi]`` (its overlap counted once)."""
    c = chunks[j]
    if j > hi:
        prev_end = chunks[hi].end
        if c.start >= prev_end:
            return c.n_tokens
        share = (c.end - prev_end) / max(1, c.end - c.start)
    else:
        next_start = chunks[lo].start
        if c.end <= next_start:
            return c.n_tokens
        share = (next_start - c.start) / max(1, c.end - c.start)
    return max(0, round(c.n_tokens * share))
