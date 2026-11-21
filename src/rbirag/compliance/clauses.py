"""Split a policy document into individually checkable clauses.

Bank policies are numbered documents (``3. Customer Due Diligence`` / ``3.1 The Bank
shall ...`` / ``(a) ...``), sometimes Markdown, sometimes unnumbered prose. The splitter:

* treats decimal-numbered lines (``3.1``, ``Clause 4``, ``7.``) as clause starts;
* treats a numbered line as a *heading* when it is short, has no sentence punctuation and
  is followed by its own sub-clauses (``3.`` followed by ``3.1``);
* keeps lettered/roman/bulleted sub-items (``(a)``, ``ii)``, ``-``) inside their parent
  clause, since they usually complete its sentence;
* uses Markdown and title-case headings for the heading path only;
* falls back to blank-line paragraphs for unnumbered text;
* refuses to start a clause at a number that merely continues the previous line
  (``... under section`` / ``5.2 of the Act``), which happens in PDF text.

Invariant: ``clause.text == source[clause.start:clause.end]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

ClauseKind = Literal["numbered", "paragraph"]
MarkerKind = Literal["md_heading", "heading", "numbered"]

_KEYWORD = r"(?:clause|section|article|para(?:graph)?)"
DECIMAL_RE = re.compile(
    rf"^(?:{_KEYWORD}\s+)?(?P<num>\d{{1,2}}(?:\.\d{{1,2}}){{1,4}})\.?(?:\s+|$)(?P<rest>.*)$",
    re.IGNORECASE,
)
TOP_LEVEL_RE = re.compile(
    rf"^(?:{_KEYWORD}\s+(?P<kw>\d{{1,2}})[.:)]?|(?P<num>\d{{1,2}})[.)])(?:\s+|$)(?P<rest>.*)$",
    re.IGNORECASE,
)
MD_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<rest>.+?)\s*#*$")
SUB_ITEM_RE = re.compile(r"^(?:\(?(?:[a-z]|[ivx]{1,5})\)|[a-z]\.|[-*•▪●◦])\s+")
HEADING_PUNCT = (".", ",", ";", ":", "?", "!")
CONTINUATION_END_RE = re.compile(r"[A-Za-z0-9,(\-]$")

MIN_PARAGRAPH_CHARS = 40
MAX_HEADING_CHARS = 90
MAX_HEADING_WORDS = 12


class Clause(BaseModel):
    clause_id: str = Field(description="Sequential id within the document, e.g. c004.")
    number: str | None = Field(default=None, description="The policy's own number, e.g. 3.1.")
    heading_path: list[str] = Field(default_factory=list, description="Outermost first.")
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    kind: ClauseKind

    @property
    def label(self) -> str:
        return self.number or self.clause_id

    @property
    def context(self) -> str:
        return " > ".join(self.heading_path)


@dataclass
class _Line:
    idx: int
    start: int  # offset of the first non-space character
    end: int  # offset just past the last non-space character
    text: str  # stripped


@dataclass
class _Marker:
    line: int  # index into the non-empty line list
    kind: MarkerKind
    num: str | None
    rank: int  # heading-stack rank: smaller = outer
    title: str  # heading text (for headings) or the text after the number


def _lines(text: str) -> list[_Line]:
    out: list[_Line] = []
    pos = 0
    for i, raw in enumerate(text.split("\n")):
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            out.append(_Line(i, pos + lead, pos + lead + len(stripped), stripped))
        pos += len(raw) + 1
    return out


def _heading_like(text: str) -> bool:
    """Short, unpunctuated, and title- or upper-case."""
    if not text or len(text) > MAX_HEADING_CHARS or text.endswith(HEADING_PUNCT):
        return False
    words = text.split()
    if len(words) > MAX_HEADING_WORDS or not text[0].isupper():
        return False
    if text.isupper():
        return True
    long_words = [w for w in words if len(w) > 3 and w[0].isalpha()]
    if not long_words:
        return True
    capitalised = sum(1 for w in long_words if w[0].isupper())
    return capitalised / len(long_words) >= 0.6


def _numbered(text: str) -> tuple[str, str] | None:
    m = DECIMAL_RE.match(text)
    if m:
        return m.group("num"), m.group("rest").strip()
    m = TOP_LEVEL_RE.match(text)
    if m:
        return (m.group("kw") or m.group("num")), m.group("rest").strip()
    return None


def _is_child(parent: str, child: str | None) -> bool:
    return child is not None and child.startswith(parent + ".")


def _find_markers(lines: list[_Line]) -> list[_Marker]:
    markers: list[_Marker] = []
    for k, line in enumerate(lines):
        md = MD_HEADING_RE.match(line.text)
        if md:
            rest = md.group("rest").strip()
            numbered = _numbered(rest)
            num, title = numbered if numbered else (None, rest)
            markers.append(_Marker(k, "md_heading", num, len(md.group("hashes")), title or rest))
            continue
        if SUB_ITEM_RE.match(line.text):
            continue
        numbered = _numbered(line.text)
        if numbered:
            prev = lines[k - 1] if k else None
            directly_after = prev is not None and prev.idx == line.idx - 1
            prev_is_heading = (
                bool(markers)
                and markers[-1].line == k - 1
                and (markers[-1].kind != "numbered" or _heading_like(markers[-1].title))
            )
            if (
                directly_after
                and prev is not None
                and not prev_is_heading
                and CONTINUATION_END_RE.search(prev.text)
            ):
                continue  # "... under section\n5.2 of the Act": a wrapped line, not a clause
            num, rest = numbered
            markers.append(_Marker(k, "numbered", num, 10 + num.count("."), rest))
            continue
        nxt = lines[k + 1] if k + 1 < len(lines) else None
        if nxt is not None and _heading_like(line.text) and not _heading_like(nxt.text):
            markers.append(_Marker(k, "heading", None, 5, line.text))
    return markers


def _paragraphs(text: str, lines: list[_Line], first: int, last: int) -> list[tuple[int, int]]:
    """Blank-line separated paragraphs over ``lines[first:last]`` as (start, end) offsets."""
    spans: list[tuple[int, int]] = []
    k = first
    while k < last:
        j = k
        while j + 1 < last and lines[j + 1].idx == lines[j].idx + 1:
            j += 1
        start, end = lines[k].start, lines[j].end
        if len(text[start:end]) >= MIN_PARAGRAPH_CHARS:
            spans.append((start, end))
        k = j + 1
    return spans


def split_clauses(text: str) -> list[Clause]:
    """Split ``text`` (as produced by :func:`~rbirag.compliance.policy.load_policy`)."""
    lines = _lines(text)
    if not lines:
        return []
    markers = _find_markers(lines)
    found: list[tuple[int, int, str | None, list[str], ClauseKind]] = []
    stack: list[_Marker] = []

    def path() -> list[str]:
        return [m.title for m in stack]

    def push(marker: _Marker) -> None:
        while stack and stack[-1].rank >= marker.rank:
            stack.pop()
        stack.append(marker)

    def add_paragraphs(first: int, last: int) -> None:
        for start, end in _paragraphs(text, lines, first, last):
            found.append((start, end, None, path(), "paragraph"))

    first_marker = markers[0].line if markers else len(lines)
    # Preamble: skip the title line(s); keep substantive paragraphs.
    add_paragraphs(0, first_marker)

    for i, marker in enumerate(markers):
        body_first = marker.line + 1
        body_last = markers[i + 1].line if i + 1 < len(markers) else len(lines)
        nxt = markers[i + 1] if i + 1 < len(markers) else None
        if marker.kind == "numbered":
            assert marker.num is not None
            num = marker.num
            # numbered headings close sibling/unrelated numbered headings
            while stack and stack[-1].num is not None and not _is_child(stack[-1].num, num):
                stack.pop()
            is_heading = (
                _heading_like(marker.title)
                and nxt is not None
                and _is_child(num, nxt.num)
                and nxt.kind == "numbered"
            )
            if is_heading:
                push(marker)
                if body_first < body_last:
                    start, end = lines[body_first].start, lines[body_last - 1].end
                    if end - start >= MIN_PARAGRAPH_CHARS:
                        found.append((start, end, num, path(), "numbered"))
                continue
            start, end = lines[marker.line].start, lines[body_last - 1].end
            found.append((start, end, num, path(), "numbered"))
        else:
            if marker.num is not None:
                while (
                    stack and stack[-1].num is not None and not _is_child(stack[-1].num, marker.num)
                ):
                    stack.pop()
            push(marker)
            add_paragraphs(body_first, body_last)

    found.sort(key=lambda f: f[0])
    return [
        Clause(
            clause_id=f"c{n:03d}",
            number=num,
            heading_path=heading_path,
            text=text[start:end],
            start=start,
            end=end,
            kind=kind,
        )
        for n, (start, end, num, heading_path, kind) in enumerate(found, start=1)
    ]
