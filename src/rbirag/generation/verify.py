"""Citation verification: does every cited source exist, belong to the document it is
shown as, contain evidence, and do the sources support every claim of the answer?

Checks, per answer:

1. **Existence** - every ``[n]`` refers to one of the sources the model was shown.
2. **Attribution** - each source's passage belongs to the document whose metadata is
   displayed with it, and every circular / reference number the answer *mentions* belongs to
   a cited (or at least a retrieved) source; a number found nowhere is flagged as possibly
   fabricated.
3. **Relevance** - each cited source contains evidence for at least one claim citing it.
4. **Support** - every claim (answer sentences split into clauses) is checked against short
   windows of its cited sources (and, failing that, of the other sources) with an
   entailment scorer: an NLI cross-encoder when the ``ml`` extra is installed, otherwise a
   transparent lexical scorer that requires every number in the claim to appear in the
   evidence window.

Unsupported claims are listed; an answer with no supported claim gets an explicit
statement that the available documents do not establish the answer. Citations are never
added or invented by the verifier: an invalid ``[n]`` is removed from the displayed text and
reported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rbirag.generation.citations import CitationReport, parse_citations, strip_citations
from rbirag.generation.evidence import is_non_answer, premise_windows, split_claims
from rbirag.index.bm25 import bm25_tokenize
from rbirag.ingest.metadata import CIRCULAR_NUMBER_RE, REF_NUMBER_RE

NOT_ESTABLISHED = "The available documents do not establish an answer to this question."
ClaimStatus = Literal["supported", "supported_by_uncited_source", "unsupported"]
Verdict = Literal["supported", "partially_supported", "unsupported", "refusal", "no_claims"]

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_UNITS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
)  # fmt: skip
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(_UNITS)} | {
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "ninety": "90",
}  # fmt: skip
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)


def numbers_in(text: str) -> set[str]:
    """Figures in ``text``, with number words read as digits ("two years" -> "2") and
    thousands separators removed ("5,00,000" -> "500000")."""
    found = {n.replace(",", "") for n in _NUMBER_RE.findall(text)}
    found |= {_NUMBER_WORDS[w.lower()] for w in _NUMBER_WORD_RE.findall(text)}
    return found


@runtime_checkable
class SupportScorer(Protocol):
    name: str

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Support of each ``(evidence, claim)`` pair in [0, 1]."""
        ...


class LexicalSupport:
    """Share of the claim's content words found in the evidence, and 0 when any number in
    the claim (an amount, a date, a percentage, a paragraph) is missing from it. Cheap,
    explainable and conservative about figures; it cannot recognise paraphrases."""

    name = "lexical-support"

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for evidence, claim in pairs:
            nums = numbers_in(claim)
            if nums - numbers_in(evidence):
                out.append(0.0)
                continue
            c = set(bm25_tokenize(claim))
            e = set(bm25_tokenize(evidence))
            out.append(len(c & e) / len(c) if c else 0.0)
        return out


class SourceRef(BaseModel):
    """What the verifier needs to know about one numbered source."""

    n: int
    doc_id: str
    text: str
    circular_number: str | None = None
    ref_number: str | None = None
    # the doc_id the displayed metadata (title, circular number, URL) belongs to
    metadata_doc_id: str | None = None


class ClaimCheck(BaseModel):
    claim: str
    sentence: int
    cited: list[int]
    support: float  # best score over windows of the cited sources (all sources if uncited)
    status: ClaimStatus
    supporting_source: int | None = None
    evidence: str | None = None  # the best window (trimmed)


class CitationCheck(BaseModel):
    n: int
    exists: bool
    doc_id: str | None = None
    belongs_to_document: bool | None = None
    relevant: bool | None = None  # supports at least one claim that cites it
    max_support: float | None = None


class VerificationReport(BaseModel):
    scorer: str
    threshold: float
    verdict: Verdict
    claims: list[ClaimCheck] = Field(default_factory=list)
    citations: list[CitationCheck] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    uncited_sentences: int = 0
    unknown_references: list[str] = Field(default_factory=list)  # numbers no source has
    supported_share: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    statement: str | None = None  # shown to the user when evidence is missing
    non_answer: bool = False  # "the documents do not specify ..." (not a factual claim)


def _ref_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9/]", "", value.upper())


def mentioned_references(answer: str) -> list[str]:
    refs = [m.group(0) for m in CIRCULAR_NUMBER_RE.finditer(answer)]
    refs += [m.group("ref") for m in REF_NUMBER_RE.finditer(answer)]
    return list(dict.fromkeys(refs))


def verify_answer(
    answer: str,
    sources: Sequence[SourceRef],
    scorer: SupportScorer,
    *,
    threshold: float = 0.5,
    report: CitationReport | None = None,
) -> VerificationReport:
    report = report or parse_citations(answer, len(sources))
    out = VerificationReport(
        scorer=scorer.name, threshold=threshold, verdict="no_claims",
        invalid_citations=list(report.invalid_ids),
        uncited_sentences=len(report.uncited_sentences), non_answer=is_non_answer(answer),
    )  # fmt: skip
    by_n = {s.n: s for s in sources}
    if report.is_refusal:
        out.verdict = "refusal"
        return out
    # 2. attribution of mentioned circular / reference numbers
    known = {_ref_key(x) for s in sources for x in (s.circular_number, s.ref_number) if x} | {
        _ref_key(m) for s in sources for m in mentioned_references(s.text)
    }
    out.unknown_references = [r for r in mentioned_references(answer) if _ref_key(r) not in known]
    # 4. claims
    windows = {s.n: premise_windows(s.text) for s in sources}
    units: list[tuple[int, str, list[int]]] = []
    for i, sent in enumerate(report.sentences):
        text = strip_citations(sent.sentence)
        if len(text.split()) < 3:
            continue
        cited = [c for c in sent.citations if c in by_n]
        units += [(i, claim, cited) for claim in split_claims(text)]
    if not units:
        return out
    pairs: list[tuple[str, str]] = []
    owner: list[tuple[int, int, str]] = []  # (unit, source n, window text)
    for u, (_, claim, _) in enumerate(units):
        for n, ws in windows.items():
            for w in ws:
                pairs.append((w.text, claim))
                owner.append((u, n, w.text))
    scores = scorer.entailment(pairs) if pairs else []
    best: dict[tuple[int, int], tuple[float, str]] = {}
    for (u, n, wtext), score in zip(owner, scores, strict=True):
        if score > best.get((u, n), (-1.0, ""))[0]:
            best[(u, n)] = (score, wtext)
    per_source: dict[int, float] = {}
    for u, (i, claim, cited) in enumerate(units):
        pool = cited or list(by_n)
        cand = [(best.get((u, n), (0.0, ""))[0], n) for n in pool]
        score, n_best = max(cand, default=(0.0, None))
        for n in cited:
            per_source[n] = max(per_source.get(n, 0.0), best.get((u, n), (0.0, ""))[0])
        status: ClaimStatus = "supported" if score >= threshold else "unsupported"
        if status == "unsupported" and cited:
            other = [(best.get((u, n), (0.0, ""))[0], n) for n in by_n if n not in cited]
            alt, n_alt = max(other, default=(0.0, None))
            if alt >= threshold:
                status, score, n_best = "supported_by_uncited_source", alt, n_alt
        evidence = best.get((u, n_best), (0.0, ""))[1] if n_best is not None else None
        out.claims.append(
            ClaimCheck(
                claim=claim, sentence=i, cited=cited, support=round(score, 4), status=status,
                supporting_source=n_best if status != "unsupported" else None,
                evidence=(evidence[:300] if evidence and status != "unsupported" else None),
            )
        )  # fmt: skip
    # 1-3. per citation
    cited_ids = sorted({c for s in report.sentences for c in s.citations})
    for n in cited_ids:
        src = by_n.get(n)
        if src is None:
            out.citations.append(CitationCheck(n=n, exists=False))
            continue
        support = per_source.get(n)
        out.citations.append(
            CitationCheck(
                n=n, exists=True, doc_id=src.doc_id,
                belongs_to_document=(src.metadata_doc_id or src.doc_id) == src.doc_id,
                relevant=None if support is None else support >= threshold,
                max_support=None if support is None else round(support, 4),
            )
        )  # fmt: skip
    ok = [c for c in out.claims if c.status != "unsupported"]
    out.supported_share = len(ok) / len(out.claims)
    out.unsupported_claims = [c.claim for c in out.claims if c.status == "unsupported"]
    if not ok:
        out.verdict = "unsupported"
        out.statement = NOT_ESTABLISHED
    elif out.unsupported_claims:
        out.verdict = "partially_supported"
        out.statement = "Not established by the retrieved documents: " + "; ".join(
            f'"{c}"' for c in out.unsupported_claims[:5]
        )
    else:
        out.verdict = "supported"
    return out


def remove_invalid_citations(answer: str, invalid: Sequence[int]) -> str:
    """Drop ``[n]`` markers that point at no source (never shown to the user)."""
    if not invalid:
        return answer
    bad = {str(i) for i in invalid}

    def fix(m: re.Match[str]) -> str:
        ids = [x.strip() for x in m.group(1).split(",")]
        keep = [x for x in ids if x not in bad]
        return f"[{', '.join(keep)}]" if keep else ""

    return re.sub(r"\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]", fix, answer).replace("  ", " ")
