"""Metadata filters inferred from the wording of a question (shown to the user, never
applied silently).

Only explicit signals are used: a year ("in 2023", "FY 2023-24"), a department name or
code ("Foreign Exchange Department", "DPSS"), or a strongly department-specific term
("FEMA" and "authorised dealer" belong to the Foreign Exchange Department). Everything
else is left unfiltered. The benchmark measures whether these filters help.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from rbirag.ingest.metadata import DEPARTMENT_CODES
from rbirag.retrieval.params import RetrievalFilter

_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})(?:\s*-\s*(\d{2,4}))?\b")
_CODE_RE = re.compile(r"\b(" + "|".join(sorted(DEPARTMENT_CODES, key=len, reverse=True)) + r")\b")
# Terms specific enough to imply a department (checked case-insensitively).
TOPIC_DEPARTMENTS: dict[str, str] = {
    r"\bFEMA\b": "Foreign Exchange Department",
    r"\bforeign exchange management\b": "Foreign Exchange Department",
    r"\bA\.?\s?P\.?\s?\(DIR Series\)": "Foreign Exchange Department",
    r"\bauthori[sz]ed dealers?\b": "Foreign Exchange Department",
    r"\bpayment (?:and settlement )?systems?\b": "Department of Payment and Settlement Systems",
    r"\bcurrency chests?\b": "Department of Currency Management",
    r"\bbank ?notes?\b": "Department of Currency Management",
}
# Words that look like department codes but are ordinary words in questions.
_CODE_STOP = frozenset({"FED"})


@dataclass
class DetectedFilters:
    filter: RetrievalFilter | None
    notes: list[str] = field(default_factory=list)


def detect_filters(
    question: str, *, year_tolerance: int = 1, today: dt.date | None = None
) -> DetectedFilters:
    """Department and date filters stated in ``question`` (``filter=None`` when none).

    A year Y becomes the range Y-``year_tolerance`` .. Y+``year_tolerance`` (a direction
    named "2025" may be issued, or last amended, in a neighbouring year); "FY 2023-24"
    becomes April 2023 to March 2024 widened the same way.
    """
    notes: list[str] = []
    departments: set[str] = set()
    for m in _CODE_RE.finditer(question):
        if m.group(1) in _CODE_STOP:
            continue
        departments.add(DEPARTMENT_CODES[m.group(1)])
        notes.append(f"department code {m.group(1)}")
    for name in set(DEPARTMENT_CODES.values()):
        if name.lower() in question.lower():
            departments.add(name)
            notes.append(f"department named: {name}")
    for pattern, name in TOPIC_DEPARTMENTS.items():
        hit = re.search(pattern, question, flags=re.IGNORECASE)
        if hit:
            departments.add(name)
            notes.append(f"'{hit.group(0)}' -> {name}")
    date_from = date_to = None
    this_year = (today or dt.date.today()).year
    years = [
        (int(m.group(1)), m.group(2)) for m in _YEAR_RE.finditer(question)
        if 1990 <= int(m.group(1)) <= this_year + 1
    ]  # fmt: skip
    if years:
        start_year, fy_end = min(years, key=lambda y: y[0])
        end_year = max(y for y, _ in years)
        if fy_end:
            end_year = max(end_year, start_year + 1)
        date_from = dt.date(start_year - year_tolerance, 1, 1)
        date_to = dt.date(min(end_year + year_tolerance, this_year + 1), 12, 31)
        notes.append(f"year(s) {sorted({y for y, _ in years})} -> {date_from}..{date_to}")
    if not departments and date_from is None:
        return DetectedFilters(filter=None)
    return DetectedFilters(
        filter=RetrievalFilter(
            departments=tuple(sorted(departments)), date_from=date_from, date_to=date_to
        ),
        notes=notes,
    )
