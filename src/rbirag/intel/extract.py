"""Regulatory metadata and cross-references extracted from RBI notifications.

Per document: circular / reference / notification numbers, publication date, effective
date (with the sentence it came from), department, subject, category and an amendment
flag. Per reference: every circular number, department reference number, A.P. (DIR
Series) number or Master Direction title cited in the body, the relation the citing
sentence states (supersedes / partially modifies / amends / refers to) and the corpus
document it resolves to. A reference that does not resolve to exactly one document is
kept with ``target=None`` and its candidates: unknown, never guessed.

Everything here is pattern-based and auditable; the precision of the patterns is
measured on a checked sample (``rbirag intel sample``), not assumed.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from rbirag.chunking.base import split_sentences
from rbirag.document import Document
from rbirag.ingest.metadata import (
    AP_DIR_RE,
    CIRCULAR_NUMBER_RE,
    REF_NUMBER_RE,
    _squash,
    find_dates,
)

Category = Literal[
    "master_direction",
    "master_circular",
    "directions",
    "regulations",
    "ap_dir_circular",
    "notification",
    "circular",
]
# From the citing document's point of view. "amended_by" is the consolidated-direction
# footnote pattern ("Amended vide circular X dated ...", "Repealed vide ..."): the *cited*
# document changed the citing one. "consolidates": a row of the list of circulars a master
# circular / direction consolidates.
RelationType = Literal[
    "supersedes", "partially_modifies", "amends", "amended_by", "consolidates", "refers_to"
]
EffectiveKind = Literal["explicit", "immediate", "gazette", "none"]


class EffectiveDate(BaseModel):
    kind: EffectiveKind
    date: dt.date | None = None
    evidence: str | None = None  # the sentence the date was read from


class DocIntel(BaseModel):
    doc_id: str
    title: str
    subject: str
    category: Category
    is_amendment: bool
    circular_number: str | None
    ref_number: str | None
    notification_number: str | None
    publication_date: dt.date | None
    effective: EffectiveDate
    department: str | None
    version_note: str | None = None  # "updated as on July 01, 2026"
    family: str  # title with dates/versions removed (groups versions of one direction)
    url: str


class Relation(BaseModel):
    citing: str  # the document the reference was found in
    cited: str | None  # the corpus document it resolves to; None = unknown
    type: RelationType
    mention: str  # the reference as written
    mention_kind: Literal["circular_number", "ref_number", "ap_dir", "master_direction"]
    cue: str | None  # the words that set the relation type
    confidence: Literal["high", "medium", "low"]
    resolution: Literal["unique", "by_date", "version_at_date", "ambiguous", "not_in_corpus"]
    candidates: list[str] = Field(default_factory=list)
    sentence: str  # the citing sentence (trimmed)
    start: int  # offset of the mention in the citing document


# ----------------------------------------------------------------------------- metadata

_VERSION_RE = re.compile(r"\(\s*updated as on ([^)]*)\)", re.IGNORECASE)
_NOTIFICATION_NO_RE = re.compile(
    r"\bNotification\s+No\.?\s*(?P<num>(?:FEMA|DOR|DBR|DNBR|DBOD|DPSS|IDMD|FMRD)[\w.()/\-\s]{2,40}?"
    r"\d{4}(?:-\d{2,4})?(?:-RB)?)",
    re.IGNORECASE,
)
_EFFECTIVE_RE = re.compile(
    r"\b(?:with effect from|w\.\s?e\.\s?f\.?|effective (?:from|on)|"
    r"(?:shall|will) (?:come into (?:force|effect)|be effective|become effective)"
    r"(?: (?:from|on))?)\s+"
    r"(?:the\s+)?",
    re.IGNORECASE,
)
_IMMEDIATE_RE = re.compile(r"\bwith immediate effect\b|\bcome into force at once\b", re.I)
_GAZETTE_RE = re.compile(
    r"\bdate of (?:their|its) publication in the (?:Official )?Gazette\b", re.I
)


# Words every direction title shares; they say nothing about *which* direction is meant.
GENERIC_TITLE_WORDS = frozenset(
    {
        "master", "direction", "directions", "circular", "reserve", "bank", "india", "non",
        "banking", "financial", "company", "companies", "nbfc", "nbfcs", "regulations",
        "updated", "as",
    }
)  # fmt: skip


def title_family(title: str) -> str:
    """Lower-case title without version notes, years, dates and punctuation."""
    t = _VERSION_RE.sub(" ", title)
    t = re.sub(r"\b(?:19|20)\d{2}(?:-\d{2,4})?\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return " ".join(w for w in t.split() if w not in {"the", "of", "and", "on", "for", "rbi"})


def classify_category(title: str, ref_number: str | None, head: str) -> Category:
    if re.search(r"\bmaster direction", title, re.I):
        return "master_direction"
    if re.search(r"\bmaster circular", title, re.I):
        return "master_circular"
    if re.search(r"\bdirections?\b[^a-z]*(?:\(|,)?\s*(?:19|20)\d{2}\b", title, re.I):
        return "directions"
    if re.search(r"\bregulations\b", title, re.I):
        return "regulations"
    if ref_number and ref_number.upper().startswith("A.P. (DIR"):
        return "ap_dir_circular"
    if re.search(r"^\s*NOTIFICATION\b", head, re.M):
        return "notification"
    return "circular"


_SUBJECT_RE = re.compile(
    r"\b(?:these|this|the (?:revised|above|said|amended)?\s*)(?:directions?|instructions|"
    r"guidelines|circular|scheme|amendments?|changes|provisions|regulations|rules|norms|"
    r"framework|agreement|notification|facility|limits?)\b|\bbanks shall\b",
    re.IGNORECASE,
)


def extract_effective(
    text: str, published: dt.date | None, category: str = "circular"
) -> EffectiveDate:
    """When the document's instructions take effect: the first effective-date sentence
    about the document itself ("These Directions shall come into force ..."), else the
    first effective-date sentence. Master circulars consolidate provisions with many
    different dates, so none is assigned to them."""
    if category == "master_circular":
        return EffectiveDate(kind="none")
    found = _effective_candidates(text, published)
    for cand, subject in found:
        if subject:
            return cand
    return found[0][0] if found else EffectiveDate(kind="none")


def _effective_candidates(text: str, published: dt.date | None) -> list[tuple[EffectiveDate, bool]]:
    out: list[tuple[EffectiveDate, bool]] = []
    for a, b in split_sentences(text):
        sentence = text[a:b]
        subject = bool(_SUBJECT_RE.search(sentence))
        if _IMMEDIATE_RE.search(sentence):
            out.append(
                (EffectiveDate(kind="immediate", date=published, evidence=_trim(sentence)), subject)
            )
            continue
        m = _EFFECTIVE_RE.search(sentence)
        if m:
            dates = [d for d in find_dates(sentence[m.end() : m.end() + 60]) if d.start < 25]
            if dates:
                eff = EffectiveDate(kind="explicit", date=dates[0].value, evidence=_trim(sentence))
                out.append((eff, subject))
            elif _GAZETTE_RE.search(sentence):
                out.append((EffectiveDate(kind="gazette", evidence=_trim(sentence)), subject))
        if len(out) >= 20:
            break
    return out


def doc_intel(doc: Document) -> DocIntel:
    head = "\n".join(doc.text.split("\n")[:30])
    version = _VERSION_RE.search(doc.title)
    notif = _NOTIFICATION_NO_RE.search(head)
    subject = _VERSION_RE.sub("", doc.title).strip(" -")
    category = classify_category(doc.title, doc.ref_number, head)
    return DocIntel(
        doc_id=doc.doc_id,
        title=doc.title,
        subject=subject,
        category=category,
        is_amendment=bool(re.search(r"\bamendment", doc.title, re.I)),
        circular_number=doc.circular_number,
        ref_number=doc.ref_number,
        notification_number=_squash(notif.group("num")) if notif else None,
        publication_date=doc.date,
        effective=extract_effective(doc.text, doc.date, category),
        department=doc.department,
        version_note=f"updated as on {version.group(1).strip()}" if version else None,
        family=title_family(doc.title),
        url=doc.url,
    )


def _trim(sentence: str, limit: int = 400) -> str:
    s = " ".join(sentence.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ----------------------------------------------------------------------------- references

_SUPERSEDE_RE = re.compile(
    r"\bin supersession of\b|\bsupersed(?:e|es|ed|ing)\b|\brepeal(?:s|ed)?\b|"
    r"\b(?:stand|stands|is|are|has been|have been|hereby|shall stand) (?:hereby )?withdrawn\b",
    re.IGNORECASE,
)
_PARTIAL_RE = re.compile(r"\bin partial modification of\b|\bpartially modif", re.I)
_AS_AMENDED_RE = re.compile(
    r"\bas amended(?: from time to time| up to [^,;]{0,40}| on [^,;]{0,40}| vide [^,;]{0,60})?|"
    r"\bamended from time to time\b|\band amendments thereto\b",
    re.IGNORECASE,
)
_AMEND_RE = re.compile(
    r"\b(?:stands?|is|are|has been|have been|being|shall be|will be|hereby|accordingly)\s+"
    r"(?:further\s+|suitably\s+|partially\s+|also\s+)?(?:amended|modified|substituted)\b|"
    r"\bamendments? (?:to|in) (?:the |our |its |RBI )?(?:master|circular|notification|"
    r"direction|para|paragraph|regulation|scheme|guidelines|instructions)|"
    r"\b(?:amend|amends|modify|modifies) (?:the|our|its|para|paragraph|clause)\b|"
    r"\bin modification of\b",
    re.IGNORECASE,
)
_REPEAL_HEADING_RE = re.compile(r"\brepeal|\bwithdrawn\b", re.I)
_LIST_HEADING_RE = re.compile(
    r"\blist of (?:circulars|notifications)|\bcirculars? consolidated|\bconsolidat|"
    r"\b(?:circular|reference|notification)s?\s*(?:/\s*notification\s*)?(?:ref\.?\s*)?no\.?\s+date\b",
    re.I,
)
_MD_MENTION_RE = re.compile(
    r"\bMaster Direction\s*(?:[-\u2013\u2014:]\s*|on\s+|\(MD\)\s*(?:on\s+)?)?"
    r"(?P<name>[A-Z][^.;:\n]{3,160}?)"
    r"(?=\s+dated\b|\s+issued\b|\s+as amended\b|"
    r"\s+(?:is|are|was|has|have|shall|stands?|which|and|vide|for|to)\b|[.;:\n]|$)"
)
_DATED_RE = re.compile(r"^\W{0,3}dated\s+", re.I)


_FOOTNOTE_RE = re.compile(
    r"\b(?:amended|inserted|substituted|deleted|added|modified|updated|revised|omitted|"
    r"repealed|withdrawn|superseded)\s*,?\s+(?:vide|by|through|as per|in terms of)\b",
    re.IGNORECASE,
)


def relation_type(sentence: str, *, list_row: str | None) -> tuple[RelationType, str | None]:
    """The relation the citing sentence states (strongest cue wins). In a row of a list of
    circulars the row text is the *cited* circular's title, so its words ("Amendments to
    ...") say nothing about the relation; the list's heading does (``list_row`` is
    "repealed" or "consolidated" for such rows)."""
    m = _FOOTNOTE_RE.search(sentence)
    if m:
        return "amended_by", m.group(0)
    if list_row == "repealed":
        return "supersedes", "row of a list of repealed circulars"
    if list_row == "consolidated":
        return "consolidates", "row of a list of consolidated circulars"
    m = _SUPERSEDE_RE.search(sentence)
    if m:
        return "supersedes", m.group(0)
    m = _PARTIAL_RE.search(sentence)
    if m:
        return "partially_modifies", m.group(0)
    m = _AMEND_RE.search(_AS_AMENDED_RE.sub(" ", sentence))
    if m:
        return "amends", m.group(0)
    return "refers_to", None


def _key(ref: str) -> str:
    """Normalised reference key: upper case, no prefixes, spaces or 'No.' tokens."""
    r = re.sub(
        r"^(?:notification|circular|master circular|master direction)?\s*"
        r"(?:ref\.?\s*)?(?:no\.?\s*)?",
        "",
        ref.strip(),
        flags=re.I,
    )
    r = re.sub(r"\s+", "", r.upper())
    r = re.sub(r"(?<=[.\s])NO\.", "", r)
    return r.rstrip(".")


def _fy(day: dt.date) -> str:
    start = day.year if day.month >= 4 else day.year - 1
    return f"{start}-{str(start + 1)[2:]}"


class Resolver:
    """Look-up tables from reference strings to corpus documents."""

    def __init__(self, docs: Sequence[Document], intel: dict[str, DocIntel]) -> None:
        self.by_circular: dict[str, list[str]] = defaultdict(list)
        self.by_ref: dict[str, list[str]] = defaultdict(list)
        self.by_apdir: dict[tuple[int, str], list[str]] = defaultdict(list)
        self.md_families: dict[str, list[str]] = defaultdict(list)
        self.dates = {d.doc_id: d.date for d in docs}
        for d in docs:
            if d.circular_number:
                self.by_circular[_key(d.circular_number)].append(d.doc_id)
            if d.ref_number:
                ap = re.search(r"A\.P\. \(DIR Series\) Circular No\. (\d+)", d.ref_number)
                if ap and d.date:
                    self.by_apdir[(int(ap.group(1)), _fy(d.date))].append(d.doc_id)
                else:
                    self.by_ref[_key(d.ref_number)].append(d.doc_id)
            info = intel[d.doc_id]
            if info.category == "master_direction":
                self.md_families[info.family].append(d.doc_id)

    def resolve(
        self, kind: str, key: str, *, dated: dt.date | None, citing: Document
    ) -> tuple[str | None, str, list[str]]:
        """``(target, resolution, candidates)`` for one mention."""
        if kind == "circular_number":
            cands = self.by_circular.get(key, [])
        elif kind == "ref_number":
            cands = self.by_ref.get(key, [])
        elif kind == "ap_dir":
            num = int(key)
            if dated is None:
                cands = sorted(
                    {c for (n, _), ids in self.by_apdir.items() if n == num for c in ids}
                )
            else:
                cands = self.by_apdir.get((num, _fy(dated)), [])
        else:
            cands = self._md_candidates(key)
        cands = [c for c in dict.fromkeys(cands) if c != citing.doc_id]
        if not cands:
            return None, "not_in_corpus", []
        if len(cands) == 1:
            return cands[0], "unique", cands
        if dated is not None:
            same = [c for c in cands if self.dates.get(c) == dated]
            if len(same) == 1:
                return same[0], "by_date", cands
        if kind in {"master_direction", "ref_number"} and citing.date is not None:
            # versions of one direction: the latest version dated on or before the citation
            dated_c = sorted(
                (c for c in cands if self.dates.get(c) and self.dates[c] <= citing.date),  # type: ignore[operator]
                key=lambda c: (self.dates[c], c),
            )
            if dated_c:
                return dated_c[-1], "version_at_date", cands
        return None, "ambiguous", cands

    def _md_candidates(self, name: str) -> list[str]:
        words = set(title_family(name).split()) - GENERIC_TITLE_WORDS
        if len(words) < 2:
            return []
        best: list[str] = []
        best_score = 0.0
        for family, ids in self.md_families.items():
            other = set(family.split()) - GENERIC_TITLE_WORDS
            score = len(words & other) / len(words | other)
            if score > best_score + 1e-9:
                best, best_score = list(ids), score
            elif abs(score - best_score) <= 1e-9:
                best += ids
        return best if best_score >= 0.7 else []


def _mentions(text: str) -> Iterable[tuple[str, str, str, int, int]]:
    """``(kind, raw, key, start, end)`` for every reference mention in ``text``."""
    seen: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return not any(a < e and s < b for s, e in seen)

    for m in CIRCULAR_NUMBER_RE.finditer(text):
        seen.append(m.span())
        yield "circular_number", m.group(0), _key(m.group(0)), m.start(), m.end()
    for m in AP_DIR_RE.finditer(text):
        if free(*m.span()):
            seen.append(m.span())
            yield "ap_dir", m.group(0), m.group("num"), m.start(), m.end()
    for line_start, line in _lines(text):
        for m in REF_NUMBER_RE.finditer(line):
            a, b = line_start + m.start("ref"), line_start + m.end("ref")
            raw = _squash(m.group("ref"))
            if raw.upper().startswith("RBI/") or not free(a, b):
                continue
            seen.append((a, b))
            yield "ref_number", raw, _key(raw), a, b
    for m in _MD_MENTION_RE.finditer(text):
        if free(*m.span()):
            seen.append(m.span())
            yield "master_direction", m.group(0), m.group("name"), m.start(), m.end()


def _lines(text: str) -> Iterable[tuple[int, str]]:
    pos = 0
    for line in text.split("\n"):
        yield pos, line
        pos += len(line) + 1


def _list_row(doc: Document, category: str, pos: int) -> str | None:
    """ "repealed" / "consolidated" when ``pos`` is in a short numbered row of a list of
    circulars (under a heading such as "List of circulars repealed" or a table header
    "Sr. No. Circular No. Date Subject"), else None."""
    heads = [s for s in doc.sections if s.kind == "heading" and s.start <= pos]
    if not heads:
        return None
    heading = heads[-1].heading
    line_start = doc.text.rfind("\n", 0, pos) + 1
    line_end = doc.text.find("\n", pos)
    line = doc.text[line_start : line_end if line_end >= 0 else len(doc.text)]
    if len(line) >= 300 or not re.match(r"\s*(?:\d{1,3}[.)]?\s|[•\-*]\s|[ivx]{1,4}[.)]\s)", line):
        return None
    if _REPEAL_HEADING_RE.search(heading):
        return "repealed"
    if _LIST_HEADING_RE.search(heading) and category in {"master_circular", "master_direction"}:
        return "consolidated"
    return None


def extract_relations(
    doc: Document, resolver: Resolver, category: str = "circular"
) -> list[Relation]:
    text = doc.text
    header_end = min(len(text), 400)  # the document's own numbers live in the header
    own = {_key(x) for x in (doc.circular_number, doc.ref_number) if x}
    sentences = split_sentences(text)
    out: list[Relation] = []
    for kind, raw, key, a, b in _mentions(text):
        if a < header_end and key in own:
            continue
        if kind in {"circular_number", "ref_number"} and key in own:
            continue  # the document citing itself (footers, annex headers)
        sent = next(((s, e) for s, e in sentences if s <= a < e), (a, b))
        sentence = text[sent[0] : sent[1]]
        after = text[b : b + 60]
        dated = None
        if _DATED_RE.search(after):
            found = find_dates(after[:50])
            dated = found[0].value if found else None
        rtype, cue = relation_type(sentence, list_row=_list_row(doc, category, a))
        target, resolution, cands = resolver.resolve(kind, key, dated=dated, citing=doc)
        exact = kind in {"circular_number", "ref_number", "ap_dir"}
        if target is None:
            confidence = "low"
        elif resolution in {"unique", "by_date"} and exact:
            confidence = "high"
        else:
            confidence = "medium"
        out.append(
            Relation(
                citing=doc.doc_id, cited=target, type=rtype, mention=raw, mention_kind=kind,  # type: ignore[arg-type]
                cue=cue, confidence=confidence, resolution=resolution,  # type: ignore[arg-type]
                candidates=cands[:10], sentence=_trim(sentence), start=a,
            )
        )  # fmt: skip
    return out


def build_intel(docs: Sequence[Document]) -> tuple[dict[str, DocIntel], list[Relation]]:
    intel = {d.doc_id: doc_intel(d) for d in docs}
    resolver = Resolver(docs, intel)
    relations = [r for d in docs for r in extract_relations(d, resolver, intel[d.doc_id].category)]
    return intel, relations
