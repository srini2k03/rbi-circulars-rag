"""Retrieval metrics (Recall@k, Precision@k, MRR@k, nDCG@k, context precision and
character-level context coverage) at document and chunk level.

Gold evidence is stored as character spans in the source document, so it is independent of
the chunker: a retrieved chunk is *relevant* at chunk level if it overlaps a gold span by at
least ``min_overlap`` of that span (or of the chunk, whichever is smaller); at document
level if it comes from a gold document. Relevance is binary; for nDCG each gold item
(span or document) counts once, at the rank where it is first covered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from rbirag.chunking.base import Chunk
from rbirag.eval.dataset import GoldSpan, QAPair

Level = Literal["doc", "chunk"]
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def span_overlap(chunk: Chunk, gold: GoldSpan) -> int:
    if chunk.doc_id != gold.doc_id:
        return 0
    return max(0, min(chunk.end, gold.end) - max(chunk.start, gold.start))


def covers(chunk: Chunk, gold: GoldSpan, min_overlap: float = 0.5) -> bool:
    ov = span_overlap(chunk, gold)
    if ov <= 0:
        return False
    denom = max(1, min(gold.end - gold.start, chunk.end - chunk.start))
    return ov / denom >= min_overlap


def first_hits(
    retrieved: Sequence[Chunk], qa: QAPair, level: Level, min_overlap: float = 0.5
) -> list[int | None]:
    """For each gold item, the 1-based rank where it is first covered (``None`` if never)."""
    if level == "doc":
        golds_doc = sorted(qa.gold_doc_ids)
        ranks_doc: list[int | None] = []
        for doc_id in golds_doc:
            rank = next((i for i, c in enumerate(retrieved, 1) if c.doc_id == doc_id), None)
            ranks_doc.append(rank)
        return ranks_doc
    ranks: list[int | None] = []
    for g in qa.gold:
        rank = next((i for i, c in enumerate(retrieved, 1) if covers(c, g, min_overlap)), None)
        ranks.append(rank)
    return ranks


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of gold items found in the top-k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Reciprocal rank of the first relevant result within the top-k."""
    found = [r for r in ranks if r is not None and r <= k]
    return 1.0 / min(found) if found else 0.0


def ndcg_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Binary-gain nDCG over *distinct relevant ranks*.

    ``ranks`` holds, per gold item, the rank of the first retrieved result covering it.
    Several gold items covered by the same result share that rank and count once:
    DCG = sum of 1/log2(r+1) over the distinct ranks r <= k. The ideal ranking needs one
    result per distinct rank found plus one per gold item never found, capped at k, so
    nDCG is always in [0, 1] (e.g. two gold spans inside one rank-1 chunk give 1.0).
    """
    if not ranks:
        return 0.0
    distinct = {r for r in ranks if r is not None}
    missing = sum(1 for r in ranks if r is None)
    dcg = sum(1.0 / math.log2(r + 1) for r in distinct if r <= k)
    n_ideal = min(len(distinct) + missing, k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / ideal if ideal else 0.0


def is_relevant(chunk: Chunk, qa: QAPair, level: Level, min_overlap: float = 0.5) -> bool:
    """Binary relevance of one retrieved chunk (gold document / covers a gold span)."""
    if level == "doc":
        return chunk.doc_id in qa.gold_doc_ids
    return any(covers(chunk, g, min_overlap) for g in qa.gold)


def precision_at_k(relevant: Sequence[bool], k: int) -> float:
    """Share of the top-``k`` results that are relevant (missing results count as not)."""
    return sum(1 for r in relevant[:k] if r) / k if k else 0.0


def average_precision_at_k(relevant: Sequence[bool], k: int) -> float:
    """Rank-aware context precision: mean of P@i over the relevant ranks i <= k
    (RAGAS-style "context precision"); 0 when nothing relevant is retrieved."""
    hits = 0
    total = 0.0
    for i, rel in enumerate(relevant[:k], start=1):
        if rel:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def context_coverage(retrieved: Sequence[Chunk], qa: QAPair) -> float:
    """Share of gold-evidence characters contained in the retrieved text (character
    level, so it is fair to contexts of any size, e.g. parent windows)."""
    total = sum(max(0, g.end - g.start) for g in qa.gold)
    if not total:
        return 0.0
    covered = 0
    for g in qa.gold:
        spans = sorted(
            (max(c.start, g.start), min(c.end, g.end))
            for c in retrieved
            if c.doc_id == g.doc_id and min(c.end, g.end) > max(c.start, g.start)
        )
        end = g.start
        for a, b in spans:
            a = max(a, end)
            if b > a:
                covered += b - a
                end = b
    return covered / total


class QueryMetrics(BaseModel):
    qa_id: str
    level: Level
    recall: dict[int, float]
    mrr: float
    ndcg: float
    precision: dict[int, float] = {}
    context_precision: float | None = None  # average precision over the context (top 5)


def query_metrics(
    retrieved: Sequence[Chunk],
    qa: QAPair,
    *,
    level: Level,
    ks: Sequence[int] = DEFAULT_KS,
    mrr_k: int = 10,
    ndcg_k: int = 10,
    min_overlap: float = 0.5,
    context_k: int = 5,
) -> QueryMetrics:
    ranks = first_hits(retrieved, qa, level, min_overlap)
    relevant = [is_relevant(c, qa, level, min_overlap) for c in retrieved]
    return QueryMetrics(
        qa_id=qa.qa_id,
        level=level,
        recall={k: recall_at_k(ranks, k) for k in ks},
        mrr=mrr_at_k(ranks, mrr_k),
        ndcg=ndcg_at_k(ranks, ndcg_k),
        precision={k: precision_at_k(relevant, k) for k in ks},
        context_precision=average_precision_at_k(relevant, context_k),
    )
