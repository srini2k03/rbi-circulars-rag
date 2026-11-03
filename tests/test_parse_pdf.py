from __future__ import annotations

import pymupdf

from rbirag.ingest.parse_pdf import parse_pdf


def _make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for i, body in enumerate(pages, 1):
        page = doc.new_page()
        page.insert_text((72, 50), "RESERVE BANK OF INDIA", fontsize=9)
        y = 100.0
        for line in body.split("\n"):
            page.insert_text((72, y), line, fontsize=11)
            y += 16
        page.insert_text((290, 800), str(i), fontsize=9)
    doc.set_metadata({"title": "Test Circular"})
    data: bytes = doc.tobytes()
    doc.close()
    return data


def test_parse_pdf_strips_running_header_and_page_numbers() -> None:
    pdf = _make_pdf(
        [
            "RBI/2024-25/99\nDOR.STR.REC.1/21.06.201/2024-25\nApril 2, 2024",
            "1. First paragraph on page two.",
            "2. Second paragraph on page three.",
        ]
    )
    parsed = parse_pdf(pdf)
    assert parsed.page_count == 3
    assert parsed.title == "Test Circular"
    assert "RESERVE BANK OF INDIA" not in parsed.text
    assert "RBI/2024-25/99" in parsed.text
    assert "1. First paragraph on page two." in parsed.text
    lines = parsed.text.split("\n")
    assert "2" not in lines and "3" not in lines


def test_parse_pdf_corrupt_bytes_returns_empty() -> None:
    parsed = parse_pdf(b"not a pdf at all")
    assert parsed.is_empty and parsed.page_count == 0
