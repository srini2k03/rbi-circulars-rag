from __future__ import annotations

from datetime import date

import pytest

from rbirag.ingest.metadata import (
    detect_sections,
    extract_addressee,
    extract_circular_number,
    extract_date,
    extract_department,
    extract_metadata,
    extract_ref_number,
    extract_title,
    find_dates,
    parse_date,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("April 1, 2024", date(2024, 4, 1)),
        ("Apr 01, 2024", date(2024, 4, 1)),
        ("Sept. 3 2023", date(2023, 9, 3)),
        ("1 April 2024", date(2024, 4, 1)),
        ("1st April, 2024", date(2024, 4, 1)),
        ("22nd day of March 2021", date(2021, 3, 22)),
        ("01-Apr-2024", date(2024, 4, 1)),
        ("01.04.2024", date(2024, 4, 1)),  # day-first (Indian convention)
        ("12/06/2025", date(2025, 6, 12)),
        ("5-11-2019", date(2019, 11, 5)),
        ("2024-04-01", date(2024, 4, 1)),
        ("dated December 31, 1999", date(1999, 12, 31)),
    ],
)
def test_parse_date_formats(text: str, expected: date) -> None:
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text",
    ["Chaitra 12, 1946 (Saka)", "31/02/2024", "13.13.2024", "May 5, 1800", "RBI/2024-25/12"],
)
def test_parse_date_rejects_invalid(text: str) -> None:
    assert parse_date(text) is None


def test_find_dates_in_order_without_overlaps() -> None:
    found = find_dates("from April 1, 2024 to 31.03.2025 (2025-03-31)")
    assert [f.value for f in found] == [date(2024, 4, 1), date(2025, 3, 31), date(2025, 3, 31)]


def test_extract_date_prefers_hint_then_date_line() -> None:
    text = "RBI/2024-25/7\nRe: circular dated March 3, 2016\nApril 9, 2024\nAll Banks"
    assert extract_date(text) == date(2024, 4, 9)
    assert extract_date(text, date_hint="Apr 10, 2024") == date(2024, 4, 10)
    assert extract_date("no dates here") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("RBI/2024-25/123 something", "RBI/2024-25/123"),
        ("ref RBI/FED/2017-18/60 here", "RBI/FED/2017-18/60"),
        ("RBI / 2023-24 / 7", "RBI/2023-24/7"),
        ("RBI/2004-2005/9", "RBI/2004-2005/9"),
        ("XRBI/2024-25/1", None),
        ("no number", None),
    ],
)
def test_extract_circular_number(text: str, expected: str | None) -> None:
    assert extract_circular_number(text) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("DOR.STR.REC.5/21.06.201/2024-25", "DOR.STR.REC.5/21.06.201/2024-25"),
        ("DoR.FIN.REC.No.45/03.10.123/2022-23", "DoR.FIN.REC.No.45/03.10.123/2022-23"),
        ("CO.DPSS.POLC.No.S-1234/02-14-003/2023-24", "CO.DPSS.POLC.No.S-1234/02-14-003/2023-24"),
        ("FIDD.CO.Plan.BC.No.5/04.09.01/2024-25", "FIDD.CO.Plan.BC.No.5/04.09.01/2024-25"),
        ("DBR.No.BP.BC.1/21.06.201/2015-16", "DBR.No.BP.BC.1/21.06.201/2015-16"),
        ("FMRD.DIRD.01/14.03.059/2023-24", "FMRD.DIRD.01/14.03.059/2023-24"),
        ("DCM (Plg) No.S1234/10.27.00/2023-24", "DCM (Plg) No.S1234/10.27.00/2023-24"),
        ("DOR. STR. REC. 5/21.06.201/2024-25", "DOR.STR.REC.5/21.06.201/2024-25"),
        ("A.P. (DIR Series) Circular No. 07", "A.P. (DIR Series) Circular No. 07"),
        ("RBI/2024-25/12", None),
    ],
)
def test_extract_ref_number(line: str, expected: str | None) -> None:
    assert extract_ref_number(f"RBI/2024-25/12\n{line}\nApril 1, 2024") == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("DOR.STR.REC.5/21.06.201/2024-25", "Department of Regulation"),
        (
            "CO.DPSS.POLC.No.S-1234/02-14-003/2023-24",
            "Department of Payment and Settlement Systems",
        ),
        ("FIDD.CO.Plan.BC.No.5/04.09.01/2024-25", "Financial Inclusion and Development Department"),
        ("A.P. (DIR Series) Circular No. 07", "Foreign Exchange Department"),
        ("DCM (Plg) No.S1234/10.27.00/2023-24", "Department of Currency Management"),
    ],
)
def test_department_from_ref(ref: str, expected: str) -> None:
    assert extract_department("", ref_number=ref) == expected


def test_department_from_circular_number_and_text() -> None:
    assert extract_department("", circular_number="RBI/FED/2017-18/60") == (
        "Foreign Exchange Department"
    )
    text = "body\n(Name)\nChief General Manager\nDepartment of Supervision"
    assert extract_department(text) == "Department of Supervision"
    assert extract_department("nothing") is None


LETTER = """RBI/2025-26/58
DOR.SFB.REC.31/16.13.218/2025-26
July 15, 2025
All Small Finance Banks
All Payments Banks
Madam / Dear Sir,
Interest Rate on Deposits
Please refer to the Master Direction dated March 3, 2016.
2. On a review, it has been decided that banks may offer differential rates.
2.1 Banks shall have a Board-approved policy.
Chapter II - Customer Due Diligence
3. These directions shall come into force with immediate effect.
Yours faithfully,
(Sunil Kumar Rao)
Chief General Manager
Annex
Conditions for Bulk Deposits
(i) Schedule of rates."""


def test_addressee_and_title() -> None:
    assert extract_addressee(LETTER) == "All Small Finance Banks; All Payments Banks"
    assert extract_title(LETTER) == "Interest Rate on Deposits"
    assert extract_title(LETTER, html_title="From Header") == "From Header"
    assert extract_addressee("no salutation here") is None


def test_salutation_variants() -> None:
    for sal in ("Madam / Sir,", "Dear Sir / Madam,", "Dear Sir,", "Madam/Dear Sir"):
        text = f"April 1, 2024\nTo\nAll Banks\n{sal}\nSubject"
        assert extract_addressee(text) == "All Banks", sal


def test_detect_sections_offsets_and_kinds() -> None:
    sections = detect_sections(LETTER)
    by_number = {s.number: s for s in sections if s.number}
    assert {"2", "2.1", "3", "II"} <= set(by_number)
    assert by_number["2.1"].level == 2
    assert by_number["2"].kind == "numbered"
    assert by_number["II"].kind == "heading"
    for s in sections:
        assert LETTER[s.start : s.end].strip() == LETTER[s.start : s.end]
        assert LETTER[s.start :].startswith(LETTER[s.start : s.start + 5])
    starts = [s.start for s in sections]
    assert starts == sorted(starts)
    headings = [s.heading for s in sections]
    # header block, addressee and signature lines are never headings
    for not_heading in ("All Small Finance Banks", "Chief General Manager", "(Sunil Kumar Rao)"):
        assert not_heading not in headings
    assert "Annex" in headings
    assert "Conditions for Bulk Deposits" in headings
    assert by_number["2"].heading.startswith("On a review")


def test_extract_metadata_facade() -> None:
    md = extract_metadata(LETTER)
    assert md.date == date(2025, 7, 15)
    assert md.circular_number == "RBI/2025-26/58"
    assert md.ref_number == "DOR.SFB.REC.31/16.13.218/2025-26"
    assert md.department == "Department of Regulation"
    assert md.title == "Interest Rate on Deposits"
    assert md.sections


def test_fixture_documents_metadata(fixture_docs: list) -> None:  # type: ignore[type-arg]
    by_id = {d.source_id: d for d in fixture_docs}
    d = by_id[12701]
    assert (d.date, d.circular_number, d.department) == (
        date(2025, 7, 15), "RBI/2025-26/58", "Department of Regulation",
    )  # fmt: skip
    assert d.addressee == "All Small Finance Banks; All Payments Banks"
    d = by_id[12655]
    assert d.date == date(2025, 6, 12)
    assert d.ref_number == "DOR.AML.REC.19/14.01.001/2025-26"
    d = by_id[12590]
    assert d.title.startswith("Foreign Exchange Management (Overseas Investment)")
    assert d.ref_number == "A.P. (DIR Series) Circular No. 07"
    assert d.department == "Foreign Exchange Department"
    assert d.addressee == "All Category-I Authorised Dealer Banks"
    assert d.date == date(2025, 4, 22)
