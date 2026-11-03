"""Parse ``[n]`` citations from an answer and validate them against the sources."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rbirag.generation.prompts import REFUSAL_TEXT

# [1] | [1, 3] | [1,2,3] | [1-3] | [1\u20133]
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*(?:,|-|\u2013)\s*\d+)*)\s*\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?:\s*\[[\d,\s\-\u2013]+\])*\s+(?=[A-Z0-9(\"'])")
_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "not contain the answer",
    "cannot answer",
    "can't answer",
    "no information",
    "not covered by the provided",
)


class SentenceCitations(BaseModel):
    sentence: str
    citations: list[int] = Field(default_factory=list)


class CitationReport(BaseModel):
    cited_ids: list[int]  # unique, in order of first appearance
    invalid_ids: list[int]  # cited ids that do not match a source
    sentences: list[SentenceCitations]
    uncited_sentences: list[str]
    is_refusal: bool

    @property
    def valid(self) -> bool:
        """Every citation resolves and (unless refusing) every sentence is cited."""
        if self.is_refusal:
            return not self.invalid_ids
        return not self.invalid_ids and not self.uncited_sentences and bool(self.cited_ids)


def expand_citation(group: str) -> list[int]:
    """``"1, 3-5"`` -> ``[1, 3, 4, 5]``. Ranges larger than 20 are not expanded."""
    ids: list[int] = []
    for part in re.split(r"\s*,\s*", group.strip()):
        rng = re.fullmatch(r"(\d+)\s*[-\u2013]\s*(\d+)", part)
        if rng:
            lo, hi = int(rng.group(1)), int(rng.group(2))
            if lo <= hi and hi - lo <= 20:
                ids.extend(range(lo, hi + 1))
            else:
                ids.extend([lo, hi])
        elif part.isdigit():
            ids.append(int(part))
    return ids


def extract_citations(text: str) -> list[int]:
    ids: list[int] = []
    for m in _CITATION_RE.finditer(text):
        ids.extend(expand_citation(m.group(1)))
    return ids


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:])", r"\1", _CITATION_RE.sub("", text)).strip()


def is_refusal(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if normalized.startswith(REFUSAL_TEXT.lower().rstrip(".")):
        return True
    return len(normalized) < 200 and any(m in normalized for m in _REFUSAL_MARKERS)


def split_sentences(text: str) -> list[str]:
    """Answer sentences; trailing citations stay attached to their sentence."""
    out: list[str] = []
    for para in re.split(r"\n+", text.strip()):
        para = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", para)  # list markers
        out.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return out


def _is_substantive(sentence: str) -> bool:
    """Headers like "Summary:" or very short fragments need no citation."""
    bare = strip_citations(sentence)
    return len(re.findall(r"[A-Za-z0-9]+", bare)) >= 4 and not bare.endswith(":")


def parse_citations(answer: str, n_sources: int) -> CitationReport:
    refusal = is_refusal(answer)
    sentences = [
        SentenceCitations(sentence=s, citations=extract_citations(s))
        for s in split_sentences(answer)
    ]
    all_ids = extract_citations(answer)
    unique = list(dict.fromkeys(all_ids))
    invalid = [i for i in unique if not 1 <= i <= n_sources]
    uncited = (
        []
        if refusal
        else [s.sentence for s in sentences if not s.citations and _is_substantive(s.sentence)]
    )
    return CitationReport(
        cited_ids=unique,
        invalid_ids=invalid,
        sentences=sentences,
        uncited_sentences=uncited,
        is_refusal=refusal,
    )
