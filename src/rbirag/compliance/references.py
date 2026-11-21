"""Which notifications amend, supersede or repeal which: a citation graph over the corpus.

The corpus has no "withdrawn" flag, so a 2019 circular and the 2025 Direction that
repealed it look equally authoritative to retrieval. RBI notifications do, however, cite
what they change: by circular number (``RBI/2024-25/87``), by department reference
number (``DOR.AML.REC.49/14.01.001/2024-25``) or by the full title of a Master
Direction. This module finds those citations and labels each with the strongest relation
its surrounding text states (``repeals`` > ``supersedes`` > ``amends`` > ``mentions``).

It is a heuristic, and the relation labels are only as good as the keyword window; the
agent sees them as hints ("a later notification repeals this"), never as verdicts.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel

from rbirag.document import Document
from rbirag.ingest.metadata import CIRCULAR_NUMBER_RE, REF_NUMBER_RE

Relation = Literal["repeals", "supersedes", "amends", "mentions"]
RELATION_STRENGTH: dict[str, int] = {"mentions": 0, "amends": 1, "supersedes": 2, "repeals": 3}
REPLACING: frozenset[str] = frozenset({"repeals", "supersedes"})

CONTEXT_CHARS = 150  # keyword window on each side of a citation
EVIDENCE_CHARS = 250
HEADING_LOOKBACK = 3  # enclosing sections examined (a list row may be its own "section")
HEADING_REACH_CHARS = 8000  # ...but only sections that start this close to the citation
_REPEAL_RE = re.compile(r"\brepeal", re.IGNORECASE)
_SUPERSEDE_RE = re.compile(
    r"\b(?:supersed|(?:stands?|are|is|been|hereby|being) withdrawn|rescind|stands? cancelled)",
    re.IGNORECASE,
)
# Headings/titles of withdrawal lists ("Review of Guidelines - Withdrawal of Circulars").
_WITHDRAWAL_HEADING_RE = re.compile(
    r"withdrawal of (?:the )?(?:\w+ ){0,2}(?:circulars?|instructions|directions)"
    r"|circulars? (?:being |to be )?withdrawn",
    re.IGNORECASE,
)
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.;])\s+(?=[A-Za-z(])|\n")
_AMEND_RE = re.compile(r"\bamend", re.IGNORECASE)
_BOILERPLATE_AMEND_RE = re.compile(
    r"as (?:amended|modified|updated) (?:from time to time|up ?to date|thereafter)", re.IGNORECASE
)
_UPDATED_AS_ON_RE = re.compile(r"\s*\((?:updated|as updated)[^)]*\)\s*$", re.IGNORECASE)
_TITLE_PREFIXES = ("master direction", "reserve bank of india (")
MIN_TITLE_CHARS = 30


class Reference(BaseModel):
    source: str  # the citing notification
    target: str  # the cited notification
    relation: Relation
    evidence: str  # the text around the citation


def normalise_circular_number(value: str) -> str:
    """``RBI/2024-2025/87`` and ``RBI / 2024-25 / 87`` -> ``RBI/2024-25/87``."""
    value = re.sub(r"\s+", "", value).upper()
    return re.sub(r"(\d{4})-\d{2}(\d{2})(?=/)", r"\1-\2", value)


def normalise_ref_number(value: str) -> str:
    value = re.sub(r"\s+", "", value).upper()
    return re.sub(r"(\d{4})-\d{2}(\d{2})$", r"\1-\2", value)


def normalise_title(value: str) -> str:
    value = _UPDATED_AS_ON_RE.sub("", value)
    value = re.sub(r"[\u2013\u2014-]+", "-", value.lower())
    value = re.sub(r"[\"'\u2018\u2019\u201c\u201d]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def sentence_around(text: str, start: int, end: int, reach: int = CONTEXT_CHARS) -> str:
    """The sentence (or table cell/line) containing ``text[start:end]``, at most ``reach``
    characters either side."""
    lo, hi = max(0, start - reach), min(len(text), end + reach)
    left = text[lo:start]
    breaks = list(_SENTENCE_BREAK_RE.finditer(left))
    if breaks:
        lo += breaks[-1].end()
    right = _SENTENCE_BREAK_RE.search(text, end, hi)
    if right:
        hi = right.start()
    return text[lo:hi]


def classify_relation(context: str, heading: str = "") -> Relation:
    """The relation stated by an enclosing heading (``8. Repeal Provisions``), else by the
    text right around the citation."""
    if heading:
        if _REPEAL_RE.search(heading):
            return "repeals"
        if _SUPERSEDE_RE.search(heading) or _WITHDRAWAL_HEADING_RE.search(heading):
            return "supersedes"
        if _AMEND_RE.search(_BOILERPLATE_AMEND_RE.sub(" ", heading)):
            return "amends"
    context = _BOILERPLATE_AMEND_RE.sub(" ", context)
    if _REPEAL_RE.search(context):
        return "repeals"
    if _SUPERSEDE_RE.search(context):
        return "supersedes"
    if _AMEND_RE.search(context):
        return "amends"
    return "mentions"


class ReferenceGraph:
    """Directed citation graph: ``source`` (newer) cites ``target`` (older or same date)."""

    def __init__(self, references: Iterable[Reference], documents: Mapping[str, Document]):
        self._docs = documents
        self._incoming: dict[str, list[Reference]] = defaultdict(list)
        self._outgoing: dict[str, list[Reference]] = defaultdict(list)
        for ref in references:
            self._incoming[ref.target].append(ref)
            self._outgoing[ref.source].append(ref)
        for refs in (*self._incoming.values(), *self._outgoing.values()):
            refs.sort(key=self._order_key)

    def _order_key(self, ref: Reference) -> tuple[int, str, str]:
        doc = self._docs.get(ref.source)
        date = doc.date.isoformat() if doc is not None and doc.date else ""
        return (-RELATION_STRENGTH[ref.relation], date, ref.source)

    def __len__(self) -> int:
        return sum(len(v) for v in self._incoming.values())

    def cited_by(self, doc_id: str) -> list[Reference]:
        """Later notifications citing ``doc_id``, strongest relation first."""
        return list(self._incoming.get(doc_id, ()))

    def cites(self, doc_id: str) -> list[Reference]:
        return list(self._outgoing.get(doc_id, ()))

    def replaced_by(self, doc_id: str) -> list[str]:
        """Notifications that (per their text) repeal or supersede ``doc_id``."""
        return [r.source for r in self.cited_by(doc_id) if r.relation in REPLACING]

    @classmethod
    def build(cls, documents: Iterable[Document]) -> ReferenceGraph:
        docs = {d.doc_id: d for d in documents}
        by_circular: dict[str, str] = {}
        by_ref: dict[str, str] = {}
        titles: list[tuple[str, str]] = []
        for d in docs.values():
            if d.circular_number:
                by_circular.setdefault(normalise_circular_number(d.circular_number), d.doc_id)
            if d.ref_number:
                by_ref.setdefault(normalise_ref_number(d.ref_number), d.doc_id)
            title = normalise_title(d.title)
            if len(title) >= MIN_TITLE_CHARS and title.startswith(_TITLE_PREFIXES):
                titles.append((title, d.doc_id))

        best: dict[tuple[str, str], Reference] = {}

        def enclosing_headings(source: Document, starts: list[int], pos: int) -> str:
            """Nearby enclosing section headings, plus the title of a withdrawal list."""
            i = bisect.bisect_right(starts, pos)
            nearby = source.sections[max(0, i - HEADING_LOOKBACK) : i]
            heads = [s.heading for s in nearby if pos - s.start <= HEADING_REACH_CHARS]
            if _WITHDRAWAL_HEADING_RE.search(source.title):
                heads.append(source.title)
            return " | ".join(heads)

        def add(
            source: Document, target_id: str, text: str, start: int, end: int, heading: str = ""
        ) -> None:
            if target_id == source.doc_id:
                return
            target = docs[target_id]
            if source.date and target.date and source.date < target.date:
                return  # an older notification cannot amend or repeal a newer one
            target_title = normalise_title(target.title)
            if len(target_title) >= MIN_TITLE_CHARS and target_title in normalise_title(
                source.title
            ):
                # "Amendment to the Master Direction - KYC ...": the title states the relation
                heading = f"{heading} | {source.title}" if heading else source.title
            context = sentence_around(text, start, end)
            evidence = text[max(0, start - EVIDENCE_CHARS) : end + EVIDENCE_CHARS]
            ref = Reference(
                source=source.doc_id,
                target=target_id,
                relation=classify_relation(context, heading),
                evidence=" ".join(evidence.split()),
            )
            key = (source.doc_id, target_id)
            prev = best.get(key)
            if prev is None or RELATION_STRENGTH[ref.relation] > RELATION_STRENGTH[prev.relation]:
                best[key] = ref

        for d in docs.values():
            starts = [s.start for s in d.sections]
            for m in CIRCULAR_NUMBER_RE.finditer(d.text):
                target_id = by_circular.get(normalise_circular_number(m.group(0)))
                if target_id:
                    head = enclosing_headings(d, starts, m.start())
                    add(d, target_id, d.text, m.start(), m.end(), head)
            for m in REF_NUMBER_RE.finditer(d.text):
                target_id = by_ref.get(normalise_ref_number(m.group("ref")))
                if target_id:
                    head = enclosing_headings(d, starts, m.start())
                    add(d, target_id, d.text, m.start(), m.end(), head)
            lowered = normalise_title(d.text) if titles else ""
            for title, target_id in titles:
                pos = lowered.find(title)
                if pos >= 0:
                    add(d, target_id, lowered, pos, pos + len(title))
        return cls(best.values(), docs)
