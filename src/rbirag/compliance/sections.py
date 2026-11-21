"""Cut a long notification down to the passages relevant to a clause.

Master Directions run to 235K characters (~60K tokens), far beyond what a free-tier model
can take per call. A notification is split into passages along its detected sections
(numbered paragraphs and headings), over-long sections are split at paragraph breaks, and
the passages that best match the query (BM25 within the document) are returned in
document order until a token budget is spent.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rbirag.document import Document
from rbirag.generation.llm import approx_tokens
from rbirag.index.bm25 import BM25Index

DEFAULT_PASSAGE_CHARS = 2000
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


class Passage(BaseModel):
    index: int = Field(ge=0)
    heading: str | None = None
    number: str | None = None
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    score: float | None = None  # BM25 score against the query, when one was given


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` at paragraph breaks (else hard) into <= ``max_chars``."""
    if end - start <= max_chars:
        return [(start, end)]
    cuts = [start]
    breaks = [m.end() for m in _PARAGRAPH_BREAK_RE.finditer(text, start, end)]
    last = start
    for b in [*breaks, end]:
        if b - cuts[-1] > max_chars and last > cuts[-1]:
            cuts.append(last)
        last = b
    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts, [*cuts[1:], end], strict=True):
        while b - a > max_chars:  # one paragraph longer than the limit: hard split
            spans.append((a, a + max_chars))
            a += max_chars
        if b > a:
            spans.append((a, b))
    return spans


def split_passages(doc: Document, *, max_chars: int = DEFAULT_PASSAGE_CHARS) -> list[Passage]:
    """Contiguous passages covering ``doc.text``, one or more per detected section."""
    text = doc.text
    if not text:
        return []
    bounds: list[tuple[int, str | None, str | None]] = [(0, None, None)]
    for s in sorted(doc.sections, key=lambda s: s.start):
        if s.start > bounds[-1][0]:
            bounds.append((s.start, s.heading, s.number))
        elif s.start == bounds[-1][0]:
            bounds[-1] = (s.start, s.heading, s.number)
    passages: list[Passage] = []
    for k, (start, heading, number) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else len(text)
        for a, b in _split_long(text, start, end, max_chars):
            chunk = text[a:b]
            lead = len(chunk) - len(chunk.lstrip())
            a, b = a + lead, a + len(chunk.rstrip())
            if b > a:
                passages.append(
                    Passage(
                        index=len(passages),
                        heading=heading,
                        number=number,
                        start=a,
                        end=b,
                        text=text[a:b],
                    )
                )
    return passages


def table_of_contents(doc: Document, *, limit: int = 40) -> list[str]:
    """Top-level headings (``Chapter III - ...``, ``5. Customer Due Diligence``)."""
    heads = [s for s in doc.sections if s.level == 1]
    out: list[str] = []
    for s in heads[:limit]:
        label = s.heading
        if s.number and not label.startswith(s.number):
            label = f"{s.number}. {label}"
        out.append(label)
    return out


def select_passages(
    doc: Document,
    query: str | None,
    *,
    max_tokens: int,
    max_chars: int = DEFAULT_PASSAGE_CHARS,
) -> tuple[list[Passage], bool]:
    """Passages most relevant to ``query`` within ``max_tokens``, in document order.

    Without a query (or when nothing matches) the leading passages are returned. The
    second value says whether anything was left out.
    """
    passages = split_passages(doc, max_chars=max_chars)
    if not passages:
        return [], False
    order: list[int]
    scores: dict[int, float] = {}
    if query and query.strip():
        index = BM25Index().build([str(p.index) for p in passages], [p.text for p in passages])
        hits = index.search(query, k=len(passages))
        scores = {int(pid): s for pid, s in hits}
        order = [int(pid) for pid, _ in hits]
    else:
        order = []
    if not order:
        order = [p.index for p in passages]

    chosen: list[int] = []
    used = 0
    for i in order:
        cost = approx_tokens(passages[i].text)
        if used + cost > max_tokens:
            if not chosen:  # always return something: the best passage, trimmed
                p = passages[i]
                keep = min(len(p.text), max_tokens * 4)
                chosen_passage = p.model_copy(
                    update={"text": p.text[:keep], "end": p.start + keep, "score": scores.get(i)}
                )
                return [chosen_passage], True
            continue
        chosen.append(i)
        used += cost
    chosen.sort()
    selected = [passages[i].model_copy(update={"score": scores.get(i)}) for i in chosen]
    return selected, len(chosen) < len(passages)
