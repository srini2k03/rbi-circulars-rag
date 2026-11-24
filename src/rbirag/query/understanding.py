"""Query understanding: normalise, classify, and flag questions the system should not
answer as asked (ambiguous, multi-part, follow-ups without context, out of scope).

Nothing here silently changes what the user asked. The analysis is returned with every
answer; a rewrite (a follow-up resolved against the previous question) is shown to the
user, and it is used for retrieval only when :func:`analyse_query` is told to.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from rbirag.index.bm25 import bm25_tokenize
from rbirag.ingest.metadata import CIRCULAR_NUMBER_RE, REF_NUMBER_RE
from rbirag.query.filters import detect_filters

QuestionType = Literal[
    "relation_lookup", "comparison", "temporal", "numeric", "procedural", "definition",
    "yes_no", "factual",
]  # fmt: skip

# Checked in order; the first match wins.
_TYPE_PATTERNS: tuple[tuple[QuestionType, re.Pattern[str]], ...] = (
    ("relation_lookup", re.compile(
        r"\b(?:amend(?:s|ed|ment)?|supersed\w*|replac\w+|withdraw\w*|repeal\w*|latest "
        r"(?:version|circular|direction)|which (?:circular|notification|direction)|"
        r"current version|still (?:valid|applicable|in force))\b", re.I)),
    ("comparison", re.compile(
        r"\b(?:differ\w*|compar\w*|versus|vs\.?|what changed|changes? (?:between|from|in)|"
        r"how (?:is|are|does) .{1,40} different)\b", re.I)),
    ("temporal", re.compile(
        r"\b(?:when|by when|from which date|effective (?:date|from)|deadline|due date|"
        r"timeline|time limit|within how (?:many|long))\b", re.I)),
    ("numeric", re.compile(
        r"\b(?:how (?:much|many|long|often)|what (?:is|are) the (?:limit|cap|ceiling|rate|"
        r"amount|threshold|percentage|minimum|maximum|ratio)|limit|percentage|per cent|"
        r"minimum|maximum|ceiling|threshold|ratio|rs\.?|₹|crore|lakh)\b", re.I)),
    ("procedural", re.compile(
        r"\b(?:how (?:to|do|does|should|can|must)|procedure|process|steps|manner|format|"
        r"report(?:ing)? (?:to|of)|submit\w*|apply|application)\b", re.I)),
    ("definition", re.compile(
        r"^\s*(?:what (?:is|are|does)|define|definition of|meaning of|who (?:is|are))\b", re.I)),
    ("yes_no", re.compile(
        r"^\s*(?:is|are|can|does|do|should|must|may|will|has|have|was|were)\b", re.I)),
)  # fmt: skip

# Regulators and authorities whose rules the RBI corpus does not contain.
OTHER_AUTHORITIES: dict[str, str] = {
    r"\bSEBI\b|\bSecurities and Exchange Board\b": "SEBI",
    r"\bIRDAI?\b|\bInsurance Regulatory\b": "IRDAI",
    r"\bPFRDA\b|\bPension Fund Regulatory\b": "PFRDA",
    r"\bIFSCA\b": "IFSCA",
    r"\bCBDT\b|\bincome[- ]tax (?:act|rate|return|department)\b|\bcapital gains tax\b": "tax law",
    r"\bGST (?:council|rate)\b": "GST",
    r"\bMCA\b|\bCompanies Act\b": "company law",
}
# Words that make a question about Indian banking / RBI regulation in-domain.
DOMAIN_WORDS = frozenset(
    {
        "rbi", "reserve", "bank", "banks", "banking", "nbfc", "nbfcs", "kyc", "aml", "loan",
        "loans", "deposit", "deposits", "credit", "interest", "rate", "rates", "payment",
        "payments", "upi", "neft", "rtgs", "remittance", "fema", "foreign", "exchange", "currency",
        "notes", "coins", "cheque", "account", "accounts", "borrower", "borrowers", "lending",
        "priority", "sector", "capital", "basel", "liquidity", "npa", "provisioning", "circular",
        "circulars", "direction", "directions", "master", "ombudsman", "fraud", "cooperative",
        "ucb", "rrb", "msme", "export", "import", "ecb", "odi", "fdi", "lrs", "gold", "securities",
        "government", "bonds", "treasury", "repo", "crr", "slr", "regulated", "entities",
    }
)  # fmt: skip
_PRONOUN_START = re.compile(
    r"^\s*(?:and|also|what about|how about|then|so|but)\b|\b(?:it|its|this|that|these|those|"
    r"they|them|the same|such|above|said)\b",
    re.I,
)
_GENERIC = frozenset(
    {
        "limit", "rate", "rule", "rules", "amount", "time", "period", "date", "norms", "guidelines",
        "requirement", "requirements", "applicable", "apply", "what", "which", "how", "much",
        "many", "is", "are", "the", "for", "of",
    }
)  # fmt: skip


class QueryAnalysis(BaseModel):
    original: str
    normalized: str
    qtype: QuestionType
    parts: list[str] = Field(default_factory=list)  # >1: a multi-part question
    ambiguous: bool = False
    ambiguity: list[str] = Field(default_factory=list)  # reasons
    follow_up: bool = False
    rewritten: str | None = None  # what is retrieved for, when it differs from the question
    rewrite_reason: str | None = None
    out_of_scope: bool = False
    scope_reason: str | None = None
    in_vocabulary: float | None = None  # share of content words the index knows
    references: list[str] = Field(default_factory=list)  # circular / ref numbers mentioned
    filters: dict[str, object] | None = None  # suggested (not applied) metadata filters
    filter_notes: list[str] = Field(default_factory=list)

    @property
    def retrieval_query(self) -> str:
        return self.rewritten or self.normalized


def normalize(question: str) -> str:
    """NFKC, straight quotes, single spaces; the wording is otherwise left untouched."""
    text = unicodedata.normalize("NFKC", question)
    text = text.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
    return re.sub(r"\s+", " ", text).strip()


def classify(question: str) -> QuestionType:
    for qtype, pattern in _TYPE_PATTERNS:
        if pattern.search(question):
            return qtype
    return "factual"


def split_parts(question: str) -> list[str]:
    """Several questions in one: more than one '?', or 'and also' / ';' joining clauses that
    each start with a question word."""
    pieces = [p.strip() for p in re.split(r"\?\s+(?=\S)", question) if p.strip()]
    if len(pieces) <= 1:
        pieces = [
            p.strip() for p in re.split(r";\s+|\s+and also\s+", question) if p.strip()
        ]  # fmt: skip
        qword = re.compile(r"^(?:what|which|how|when|who|whether|is|are|can|does|do)\b", re.I)
        if len(pieces) > 1 and not all(qword.match(p) for p in pieces[1:]):
            return [question]
    return [p if p.endswith("?") else p + "?" for p in pieces] if len(pieces) > 1 else [question]


def analyse_query(
    question: str,
    *,
    history: Sequence[str] = (),
    vocabulary: Callable[[str], bool] | None = None,
) -> QueryAnalysis:
    """``vocabulary(token) -> bool`` tells whether the index knows a (BM25-normalised)
    token; it sharpens the out-of-scope check."""
    text = normalize(question)
    out = QueryAnalysis(original=question, normalized=text, qtype=classify(text))
    out.parts = split_parts(text)
    tokens = bm25_tokenize(text)
    content = [t for t in tokens if t not in _GENERIC]
    out.references = [m.group(0) for m in CIRCULAR_NUMBER_RE.finditer(text)] + [
        m.group("ref") for m in REF_NUMBER_RE.finditer(text)
    ]
    # follow-ups: elliptical or pronoun-led questions after an earlier turn
    if history:
        short = len(content) <= 3
        if _PRONOUN_START.search(text) or short:
            prev = normalize(history[-1])
            out.follow_up = True
            out.rewritten = f"{text} (follow-up to: {prev})"
            out.rewrite_reason = (
                "the question refers back to the previous one ("
                + ("pronoun / connective" if _PRONOUN_START.search(text) else "very short")
                + "), so the previous question is added for retrieval"
            )
    # ambiguity (only when no conversation context resolves it)
    if not out.follow_up:
        if len(content) == 0:
            out.ambiguity.append("no specific subject (only generic words)")
        elif len(tokens) <= 2:
            out.ambiguity.append("too short to identify the regulation or entity asked about")
        if re.search(r"\b(?:this|that|the above|the said|such) (?:circular|direction|"
                     r"notification|scheme|limit|rule)\b", text, re.I):  # fmt: skip
            out.ambiguity.append("refers to 'this/that' document without naming it")
    if len(out.parts) > 1:
        out.ambiguity.append(f"{len(out.parts)} questions in one; each is answered separately")
    out.ambiguous = bool(out.ambiguity)
    # scope
    for pattern, name in OTHER_AUTHORITIES.items():
        if re.search(pattern, text) and not re.search(r"\bRBI\b|\bReserve Bank\b", text):
            out.out_of_scope = True
            out.scope_reason = f"asks about {name} rules, which this RBI corpus does not cover"
            break
    if vocabulary is not None and content:
        known = sum(1 for t in content if vocabulary(t))
        out.in_vocabulary = round(known / len(content), 3)
    domain = any(t in DOMAIN_WORDS for t in tokens) or bool(out.references)
    if not out.out_of_scope and not domain:
        if out.in_vocabulary is not None and out.in_vocabulary < 0.5:
            out.out_of_scope = True
            out.scope_reason = "most of its words never occur in the RBI corpus"
        elif out.in_vocabulary is None or out.in_vocabulary < 0.8:
            out.out_of_scope = True
            out.scope_reason = "no banking or RBI-regulation terms"
    detected = detect_filters(text)
    if detected.filter is not None:
        out.filters = detected.filter.model_dump(mode="json", exclude_defaults=True)
        out.filter_notes = detected.notes
    return out
