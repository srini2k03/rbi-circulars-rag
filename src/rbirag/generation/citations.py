"""Parse ``[n]`` citations from an answer and validate them against the sources."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rbirag.generation.prompts import REFUSAL_TEXT

# [1] | [1, 3] | [1,2,3] | [1-3] | [1\u20133]; ids have 1-3 digits, so "[1949]" or "[2024]"
# (years, section numbers) are never read as citations.
_ID = r"\d{1,3}"
_CITATION_RE = re.compile(r"\[\s*(" + _ID + r"(?:\s*(?:,|-|\u2013)\s*" + _ID + r")*)\s*\]")
# A citation written after the full stop ("... two years. [2] The limit ...") belongs to the
# sentence before it: move it in front of the punctuation before splitting.
_TRAILING_CITES = re.compile(
    r"([.!?])((?:\s*\[\s*" + _ID + r"(?:\s*(?:,|-|\u2013)\s*" + _ID + r")*\s*\])+)"
)
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
    cited_ids: list[int]  # unique ids within 1..n_sources, in order of first appearance
    invalid_ids: list[int]  # 1-3 digit bracketed ids beyond the sources (flagged)
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
    """Answer sentences, using the abbreviation-aware splitter of the chunkers (so
    "Rs. 5 lakh" and "No. 45" do not split). Citations written after a sentence's full
    stop are attached to that sentence."""
    from rbirag.chunking.base import split_sentences as split_spans

    out: list[str] = []
    for para in re.split(r"\n+", text.strip()):
        para = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", para)  # list markers
        para = _TRAILING_CITES.sub(lambda m: m.group(2).rstrip() + m.group(1), para)
        out.extend(para[a:b] for a, b in split_spans(para, semicolons=False))
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
    all_ids = list(dict.fromkeys(extract_citations(answer)))
    unique = [i for i in all_ids if 1 <= i <= n_sources]  # citations of real sources
    invalid = [i for i in all_ids if not 1 <= i <= n_sources]  # [n] beyond the sources
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
