"""Context assembly checks between retrieval and generation.

* **Deduplication** - successive versions of a direction repeat whole paragraphs; a
  passage that is near-identical to a better-ranked one adds no evidence and wastes the
  context window, so it is dropped (word 5-gram Jaccard >= threshold).
* **Length budget** - passages are kept in rank order until the token budget is spent.
* **Source diversity** - how many distinct documents the context draws on.
* **Conflicting evidence** - two passages from different documents that state the same
  sentence with different figures ("... two years ..." vs "... three years ..."), or that
  come from different versions of one direction (or from a document and one that amends
  or supersedes it, when the relation graph is available). Conflicts are *reported* with
  both sides and their dates; the system never silently picks one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from rbirag.chunking.base import split_sentences
from rbirag.generation.verify import numbers_in
from rbirag.index.bm25 import bm25_tokenize

if TYPE_CHECKING:
    from rbirag.retrieval.pipeline import ScoredChunk


class Conflict(BaseModel):
    sources: tuple[int, int]  # 1-based positions in the context
    kind: str  # "different figures" | "versions of one direction" | "amended" | ...
    detail: str


class ContextReport(BaseModel):
    kept: int
    dropped_duplicates: list[str] = Field(default_factory=list)  # chunk ids
    dropped_for_budget: list[str] = Field(default_factory=list)
    tokens: int = 0
    distinct_documents: int = 0
    conflicts: list[Conflict] = Field(default_factory=list)


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = bm25_tokenize(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def assemble_context(
    results: Sequence[ScoredChunk],
    *,
    dedup_threshold: float = 0.9,
    token_budget: int = 6000,
    relation: Callable[[str, str], str | None] | None = None,
) -> tuple[list[ScoredChunk], ContextReport]:
    """Deduplicate, apply the token budget and look for conflicts. ``relation(a, b)``
    (optional) names a known relation between two documents (e.g. from the graph)."""
    kept: list[ScoredChunk] = []
    shingles: list[set[tuple[str, ...]]] = []
    report = ContextReport(kept=0)
    tokens = 0
    for r in results:
        sh = _shingles(r.chunk.text)
        if any(jaccard(sh, other) >= dedup_threshold for other in shingles):
            report.dropped_duplicates.append(r.chunk.chunk_id)
            continue
        if kept and tokens + r.chunk.n_tokens > token_budget:
            report.dropped_for_budget.append(r.chunk.chunk_id)
            continue
        kept.append(r)
        shingles.append(sh)
        tokens += r.chunk.n_tokens
    report.kept = len(kept)
    report.tokens = tokens
    report.distinct_documents = len({r.chunk.doc_id for r in kept})
    report.conflicts = find_conflicts(kept, relation=relation)
    return kept, report


def find_conflicts(
    results: Sequence[ScoredChunk],
    *,
    relation: Callable[[str, str], str | None] | None = None,
    min_overlap: float = 0.6,
    limit: int = 5,
) -> list[Conflict]:
    out: list[Conflict] = []
    sents = [
        [(r.chunk.text[a:b], set(bm25_tokenize(r.chunk.text[a:b])))
         for a, b in split_sentences(r.chunk.text)]
        for r in results
    ]  # fmt: skip
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i].chunk, results[j].chunk
            if a.doc_id == b.doc_id:
                continue
            da, db = a.metadata.get("date"), b.metadata.get("date")
            rel = relation(a.doc_id, b.doc_id) if relation else None
            if rel:
                detail = f"[{i + 1}] dated {da}, [{j + 1}] dated {db}"
                out.append(Conflict(sources=(i + 1, j + 1), kind=rel, detail=detail))
                continue
            for sa, ta in sents[i]:
                if len(ta) < 6:
                    continue
                hit = None
                for sb, tb in sents[j]:
                    if len(tb) < 6 or len(ta & tb) / len(ta | tb) < min_overlap:
                        continue
                    na, nb = numbers_in(sa), numbers_in(sb)
                    if na and nb and na != nb:
                        hit = (sa, sb)
                        break
                if hit:
                    out.append(Conflict(
                        sources=(i + 1, j + 1), kind="different figures",
                        detail=f"[{i + 1}] ({da}): \"{_short(hit[0])}\" vs [{j + 1}] ({db}): "
                               f"\"{_short(hit[1])}\"",
                    ))  # fmt: skip
                    break
            if len(out) >= limit:
                return out
    return out


def _short(text: str, limit: int = 160) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= limit else t[: limit - 1] + "…"
