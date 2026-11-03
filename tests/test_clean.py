from __future__ import annotations

from rbirag.ingest.clean import (
    clean_text,
    is_boilerplate_line,
    normalize_unicode,
    strip_page_numbers,
    strip_repeated_page_lines,
)


def test_normalize_unicode_folds_quotes_dashes_spaces_and_zero_width() -> None:
    raw = "Bank\u2019s \u201cKYC\u201d \u2013 Rs.\u00a05\u200b lakh\ufb01"
    assert normalize_unicode(raw) == 'Bank\'s "KYC" - Rs. 5 lakhfi'


def test_clean_text_collapses_whitespace_and_drops_boilerplate() -> None:
    raw = (
        "  Top \n\n\n\nRBI/2025-26/1   \n Print\n(245 kb)\n"
        "Page Last Updated on: May 1, 2025\nBody  text\n"
    )
    assert clean_text(raw) == "RBI/2025-26/1\nBody text"


def test_boilerplate_does_not_eat_content_numbers() -> None:
    assert not is_boilerplate_line("3")  # bare numbers are only stripped at PDF page edges
    assert not is_boilerplate_line("3. Banks shall comply.")
    assert is_boilerplate_line("Page 2 of 7")


def test_strip_page_numbers_only_at_edges() -> None:
    page = "7\nHeading\n25\nmiddle\nend\n- 7 -"
    assert strip_page_numbers(page) == "Heading\n25\nmiddle\nend"


def test_strip_repeated_page_lines_removes_running_headers() -> None:
    pages = [f"Reserve Bank of India\nbody {i}\nmore {i}\nPage {i}" for i in range(1, 5)]
    cleaned = strip_repeated_page_lines(pages)
    assert all("Reserve Bank of India" not in p for p in cleaned)
    assert all(f"body {i}" in p for i, p in enumerate(cleaned, 1))
    assert all("Page" not in p for p in cleaned)
