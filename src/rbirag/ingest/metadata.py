"""Metadata extraction for RBI notifications.

RBI letters follow a stable template, which the extractors lean on::

    RBI/2024-25/12                                <- circular number
    DOR.STR.REC.5/21.06.201/2024-25               <- department reference number
    April 1, 2024                                 <- date
    All Scheduled Commercial Banks                <- addressee (one or more lines)
    Madam / Dear Sir,                             <- salutation
    <Subject line>                                <- title (if the page header lacks one)
    1. Please refer to ...                        <- numbered paragraphs
    Yours faithfully,                             <- closing / signature block
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from datetime import date

from rbirag.document import Section

HEADER_LINES = 30  # how far into the text we look for header metadata

# ----------------------------------------------------------------------------- dates

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4,
    "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}  # fmt: skip
_MONTH_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_ORD = r"(?:st|nd|rd|th)?"
DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # April 1, 2024 | Apr 01, 2024 | Apr. 1st 2024
    re.compile(
        rf"\b(?P<month>{_MONTH_RE})\.?\s+(?P<day>\d{{1,2}}){_ORD},?\s+(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    ),
    # 1 April 2024 | 1st April, 2024 | 01-Apr-2024 | 1st day of April 2024
    re.compile(
        rf"\b(?P<day>\d{{1,2}}){_ORD}(?:\s+day\s+of)?[\s\-]+(?P<month>{_MONTH_RE})\.?[,\s\-]+"
        rf"(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    ),
    # 2024-04-01 (ISO)
    re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b"),
    # 01.04.2024 | 01/04/2024 | 01-04-2024 (Indian day-first)
    re.compile(r"(?<![\d./])(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>\d{4})\b"),
)
MIN_YEAR, MAX_YEAR = 1935, 2100  # RBI was established in 1935


@dataclass(frozen=True)
class DateMatch:
    value: date
    start: int
    end: int


def _to_date(day: str, month: str, year: str) -> date | None:
    m = int(month) if month.isdigit() else _MONTHS.get(month.lower().rstrip("."))
    if m is None:
        return None
    y = int(year)
    if not MIN_YEAR <= y <= MAX_YEAR:
        return None
    try:
        return date(y, m, int(day))
    except ValueError:
        return None


def find_dates(text: str) -> list[DateMatch]:
    """All dates in ``text``, in order of appearance, with overlapping matches removed."""
    found: list[DateMatch] = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            d = _to_date(m.group("day"), m.group("month"), m.group("year"))
            if d is None:
                continue
            if any(m.start() < f.end and f.start < m.end() for f in found):
                continue
            found.append(DateMatch(d, m.start(), m.end()))
    return sorted(found, key=lambda f: f.start)


def parse_date(value: str) -> date | None:
    """Parse the first date found in ``value`` (any supported format)."""
    matches = find_dates(value)
    return matches[0].value if matches else None


_DATE_LINE_RE = re.compile(r"^(?:date[d]?\s*:?\s*)?(?P<rest>.+?)[.\s]*$", re.IGNORECASE)


def extract_date(text: str, *, date_hint: str | None = None) -> date | None:
    """Document date: header hint, else a header line that *is* a date, else the first date."""
    if date_hint:
        hinted = parse_date(date_hint)
        if hinted:
            return hinted
    lines = [ln.strip() for ln in text.split("\n")[:HEADER_LINES] if ln.strip()]
    for line in lines:
        m = _DATE_LINE_RE.match(line)
        rest = m.group("rest") if m else line
        matches = find_dates(rest)
        if matches and matches[0].start == 0 and len(rest) - matches[0].end <= 2:
            return matches[0].value
    head = "\n".join(lines)
    return parse_date(head)


# ----------------------------------------------------------------------------- numbers

CIRCULAR_NUMBER_RE = re.compile(
    r"(?<![\w/])RBI\s?/\s?(?:(?P<dept>[A-Z]{2,10})\s?/\s?)?(?P<fy>\d{4}\s?-\s?\d{2,4})\s?/\s?"
    r"(?P<num>\d{1,4})\b"
)
REF_NUMBER_RE = re.compile(
    r"(?<![\w/.])(?P<ref>[A-Z][A-Za-z&]*\.?\s?(?:\([A-Za-z\s]+\)\s?)?"
    r"(?:[A-Za-z0-9&()\-]+\.\s?){0,8}(?:No\.?\s?)?[A-Z]{0,3}[-.]?\s?\d+[A-Za-z]?"
    r"\s?/\s?[\d.\-]+\s?/\s?\d{4}-\d{2,4})"
)
AP_DIR_RE = re.compile(
    r"A\.\s?P\.\s?\(\s?DIR\s+Series\s?\)\s+Circular\s+No\.?\s?(?P<num>\d+)", re.IGNORECASE
)
MASTER_DIRECTION_NO_RE = re.compile(
    r"(?<![\w/.])(?P<ref>(?:[A-Z][A-Za-z]*\.){0,4}\s?(?:FED\s+)?Master\s+Direction\s+No\.?\s?\d+"
    r"(?:/\d{4}-\d{2,4})?)",
)


def _squash(value: str) -> str:
    value = re.sub(r"\s*([/.\-])\s*", r"\1", value.strip())
    return re.sub(r"\s+", " ", value)


def extract_circular_number(text: str) -> str | None:
    """``RBI/2024-25/12`` style number (``RBI/FED/2017-18/60`` too), whitespace-normalised."""
    m = CIRCULAR_NUMBER_RE.search(text)
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(0))


def extract_ref_number(text: str) -> str | None:
    """Department reference number, e.g. ``DOR.STR.REC.5/21.06.201/2024-25``.

    Falls back to ``A.P. (DIR Series) Circular No. N`` and ``Master Direction No. N`` forms.
    """
    head = "\n".join(text.split("\n")[:HEADER_LINES])
    for line in head.split("\n"):
        for m in REF_NUMBER_RE.finditer(line):
            ref = m.group("ref")
            if ref.upper().startswith("RBI/"):
                continue
            return _squash(ref)
    ap = AP_DIR_RE.search(head)
    if ap:
        return f"A.P. (DIR Series) Circular No. {ap.group('num')}"
    md = MASTER_DIRECTION_NO_RE.search(head)
    if md:
        return re.sub(r"\s+", " ", md.group("ref").strip())
    return None


# ----------------------------------------------------------------------------- department

DEPARTMENT_CODES: dict[str, str] = {
    "DOR": "Department of Regulation",
    "DBR": "Department of Banking Regulation",
    "DBOD": "Department of Banking Operations and Development",
    "DOS": "Department of Supervision",
    "DBS": "Department of Banking Supervision",
    "DNBR": "Department of Non-Banking Regulation",
    "DNBS": "Department of Non-Banking Supervision",
    "DPSS": "Department of Payment and Settlement Systems",
    "FIDD": "Financial Inclusion and Development Department",
    "RPCD": "Rural Planning and Credit Department",
    "FED": "Foreign Exchange Department",
    "FMRD": "Financial Markets Regulation Department",
    "FMOD": "Financial Markets Operations Department",
    "IDMD": "Internal Debt Management Department",
    "DCM": "Department of Currency Management",
    "DGBA": "Department of Government and Bank Accounts",
    "CEPD": "Consumer Education and Protection Department",
    "DEPR": "Department of Economic and Policy Research",
    "DSIM": "Department of Statistics and Information Management",
    "MPD": "Monetary Policy Department",
    "UBD": "Urban Banks Department",
    "DCBR": "Department of Co-operative Bank Regulation",
    "DCBS": "Department of Co-operative Bank Supervision",
    "FSD": "Financial Stability Department",
    "DIT": "Department of Information Technology",
    "HRMD": "Human Resource Management Department",
    "DEIO": "Department of External Investments and Operations",
}
_DEPARTMENT_NAME_RE = re.compile(
    "|".join(re.escape(v) for v in sorted(set(DEPARTMENT_CODES.values()), key=len, reverse=True)),
    re.IGNORECASE,
)
_CANONICAL = {v.lower(): v for v in DEPARTMENT_CODES.values()}


def department_from_code(code: str) -> str | None:
    for token in re.split(r"[.\s()/,\-]+", code):
        name = DEPARTMENT_CODES.get(token.upper())
        if name:
            return name
    return None


def extract_department(
    text: str, *, ref_number: str | None = None, circular_number: str | None = None
) -> str | None:
    """Canonical department name from the reference code, else from an explicit mention."""
    if ref_number:
        if ref_number.upper().startswith("A.P. (DIR"):
            return DEPARTMENT_CODES["FED"]
        name = department_from_code(ref_number)
        if name:
            return name
    if circular_number:
        m = CIRCULAR_NUMBER_RE.search(circular_number)
        if m and m.group("dept"):
            name = DEPARTMENT_CODES.get(m.group("dept").upper())
            if name:
                return name
    # Explicit mention: letterhead (top) or signature block (bottom).
    for region in (text[:2000], text[-2000:]):
        m = _DEPARTMENT_NAME_RE.search(region)
        if m:
            return _CANONICAL[m.group(0).lower()]
    return None


# ----------------------------------------------------------------------------- addressee, title

SALUTATION_RE = re.compile(
    r"^(?:dear\s+)?(?:madam|sirs?)(?:\s*/\s*(?:dear\s+)?(?:madam|sirs?))?\s*[,.]?$",
    re.IGNORECASE,
)
CLOSING_RE = re.compile(r"^yours\s+(?:faithfully|sincerely|truly)\b", re.IGNORECASE)


def _is_header_meta(line: str) -> bool:
    if CIRCULAR_NUMBER_RE.search(line) or REF_NUMBER_RE.search(line) or AP_DIR_RE.search(line):
        return True
    m = _DATE_LINE_RE.match(line)
    rest = m.group("rest") if m else line
    dm = find_dates(rest)
    return bool(dm) and dm[0].start == 0 and len(rest) - dm[0].end <= 2


def _salutation_index(lines: list[str]) -> int | None:
    for i, line in enumerate(lines[:HEADER_LINES]):
        if SALUTATION_RE.match(line.strip()):
            return i
    return None


def extract_addressee(text: str) -> str | None:
    """Lines between the last header-meta line (number/date) and the salutation."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sal = _salutation_index(lines)
    if sal is None:
        return None
    start = 0
    for i in range(sal):
        if _is_header_meta(lines[i]):
            start = i + 1
    block = [
        ln for ln in lines[start:sal] if not _is_header_meta(ln) and ln.lower() not in {"to", "to,"}
    ]
    block = block[-8:]
    if not block:
        return None
    return "; ".join(block)


def extract_title(text: str, *, html_title: str | None = None) -> str:
    """Page-header title when present, else the subject line after the salutation."""
    if html_title and len(html_title.strip()) >= 4:
        return html_title.strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sal = _salutation_index(lines)
    if sal is not None and sal + 1 < len(lines):
        candidate = lines[sal + 1]
        if len(candidate) <= 300 and not NUMBERED_RE.match(candidate):
            return candidate
    for line in lines:
        if not _is_header_meta(line) and len(line) >= 4:
            return line[:300]
    return "Untitled notification"


# ----------------------------------------------------------------------------- sections

NUMBERED_RE = re.compile(r"^(?P<num>\d{1,3}(?:\.\d{1,3}){0,3})(?:\.|\))?\s+(?P<rest>\S.*)$")
EXPLICIT_HEADING_RE = re.compile(
    r"^(?P<kw>chapter|part|section|annex(?:ure)?|appendix|schedule)\b"
    r"(?:[\s\-:]*(?P<num>[IVXLC]+|\d+|[A-Z])\b)?[\s\-:.]*(?P<rest>.*)$",
    re.IGNORECASE,
)
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to",
     "under", "with", "into", "vis-a-vis"}
)  # fmt: skip


def _looks_like_heading(line: str) -> bool:
    if not 3 <= len(line) <= 100 or line[-1] in ".,;:":
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", line)
    if not words or len(words) > 14:
        return False
    if line.isupper() and len(words) >= 1 and sum(len(w) for w in words) >= 4:
        return True
    significant = [w for w in words if w.lower() not in _SMALL_WORDS]
    if len(significant) < 2:
        return False
    capped = sum(1 for w in significant if w[0].isupper())
    return capped / len(significant) >= 0.8


def _numbered_heading(rest: str) -> str:
    m = re.match(r"^(.{3,100}?)(?:[.:;](?:\s|$)|$)", rest)
    head = m.group(1) if m else rest[:100]
    return head.strip()


def detect_sections(text: str) -> list[Section]:
    """Headings and numbered paragraphs with character offsets into ``text``.

    The title-case heuristic is disabled in the letter header (before the salutation) and
    in the signature block, where names, designations and addressees look like headings.
    """
    raw_lines = text.split("\n")
    offsets: list[int] = []
    pos = 0
    for line in raw_lines:
        offsets.append(pos)
        pos += len(line) + 1

    stripped = [ln.strip() for ln in raw_lines]
    non_empty = [i for i, ln in enumerate(stripped) if ln]
    sal_line: int | None = None
    for i in non_empty[:HEADER_LINES]:
        if SALUTATION_RE.match(stripped[i]):
            sal_line = i
            break
    header_end = sal_line + 1 if sal_line is not None else 0
    closing_lines: set[int] = set()
    for i in non_empty:
        if CLOSING_RE.match(stripped[i]):
            closing_lines.add(i)
            for j in [j for j in non_empty if j > i][:6]:
                if EXPLICIT_HEADING_RE.match(stripped[j]):
                    break  # signature block ends where an Annex/Appendix starts
                closing_lines.add(j)

    found: list[tuple[int, str, str | None, int, str]] = []  # (line_idx, heading, num, lvl, kind)
    for i in non_empty:
        line = stripped[i]
        if i < header_end and sal_line is not None:
            continue
        explicit = EXPLICIT_HEADING_RE.match(line)
        if explicit and len(line) <= 120:
            found.append((i, line, explicit.group("num"), 1, "heading"))
            continue
        numbered = NUMBERED_RE.match(line)
        if numbered:
            num = numbered.group("num")
            level = num.count(".") + 1
            found.append((i, _numbered_heading(numbered.group("rest")), num, level, "numbered"))
            continue
        if i in closing_lines or (sal_line is None and i < 3):
            continue
        if _looks_like_heading(line):
            found.append((i, line, None, 1, "heading"))

    sections: list[Section] = []
    for k, (i, heading, num, level, kind) in enumerate(found):
        start = offsets[i] + (len(raw_lines[i]) - len(raw_lines[i].lstrip()))
        end = offsets[found[k + 1][0]] if k + 1 < len(found) else len(text)
        end = start + len(text[start:end].rstrip())
        sections.append(
            Section(
                heading=heading,
                number=num,
                level=level,
                start=start,
                end=max(end, start),
                kind="numbered" if kind == "numbered" else "heading",
            )
        )
    return sections


# ----------------------------------------------------------------------------- facade


@dataclass
class Metadata:
    title: str
    date: dt.date | None
    department: str | None
    circular_number: str | None
    ref_number: str | None
    addressee: str | None
    sections: list[Section] = field(default_factory=list)


def extract_metadata(
    text: str, *, html_title: str | None = None, date_hint: str | None = None
) -> Metadata:
    circular = extract_circular_number("\n".join(text.split("\n")[:HEADER_LINES]))
    ref = extract_ref_number(text)
    return Metadata(
        title=extract_title(text, html_title=html_title),
        date=extract_date(text, date_hint=date_hint),
        department=extract_department(text, ref_number=ref, circular_number=circular),
        circular_number=circular,
        ref_number=ref,
        addressee=extract_addressee(text),
        sections=detect_sections(text),
    )
