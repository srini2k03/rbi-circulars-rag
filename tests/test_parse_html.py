from __future__ import annotations

import logging

import pytest

from rbirag.ingest.parse_html import (
    diagnose_html,
    looks_not_found,
    node_to_text,
    parse_notification_html,
)
from tests.conftest import fixture_html


def test_classic_table_layout_uses_tableheader_strategy(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="rbirag.ingest.parse_html"):
        parsed = parse_notification_html(fixture_html("notification_12701.html"), source="12701")
    assert parsed.strategy == "rbi_tableheader"
    assert "strategy=rbi_tableheader" in caplog.text
    assert parsed.title == "Interest Rate on Deposits - Revised Directions for Small Finance Banks"
    assert parsed.date_hint == "Jul 15, 2025"
    assert parsed.pdf_links == [
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT41A7F3C2E1B04D7EA8F2.PDF"
    ]
    assert parsed.layout_detected
    text = parsed.text
    assert text.startswith("RBI/2025-26/58\nDOR.SFB.REC.31/16.13.218/2025-26")
    assert "2.1 Banks shall have a Board-approved policy" in text
    # chrome, scripts and boilerplate are gone
    for junk in ("About Us", "pageId", "Page Last Updated", "All Rights Reserved", "(245 kb)"):
        assert junk not in text
    # table rows are kept on their own lines
    assert "Above \u20b910 crore 75" in text


def test_container_layout(caplog: pytest.LogCaptureFixture) -> None:
    parsed = parse_notification_html(
        fixture_html("notification_12655.html"), base_url="https://www.rbi.org.in/Scripts/X.aspx"
    )
    assert parsed.strategy == "rbi_container"
    assert parsed.title is not None and parsed.title.startswith("Master Direction - Know Your")
    # relative PDF link resolved against the page URL
    assert parsed.pdf_links == [
        "https://www.rbi.org.in/rdocs/notification/PDFs/MD18KYC12062025.PDF"
    ]
    assert "Home" not in parsed.text.split("\n")[0]
    assert "Copyright" not in parsed.text


def test_unknown_layout_falls_back_to_text_density() -> None:
    parsed = parse_notification_html(fixture_html("notification_12590.html"))
    assert parsed.strategy == "main_text_density"
    assert parsed.text.startswith("RBI/2025-26/12")
    assert "Previous" not in parsed.text and "Circular Index" not in parsed.text
    assert "FIRMS portal" in parsed.text
    assert parsed.tried == ["rbi_tableheader", "rbi_container", "main_text_density"]


def test_empty_rbi_template_is_not_found() -> None:
    assert looks_not_found(fixture_html("notification_13050_empty.html"))


def test_error_page_is_not_found() -> None:
    html = fixture_html("notification_error.html")
    parsed = parse_notification_html(html)
    assert parsed.strategy == "none" and parsed.is_empty
    assert looks_not_found(html)


@pytest.mark.parametrize("name", ["notification_12701.html", "notification_12655.html"])
def test_real_fixtures_are_not_flagged_not_found(name: str) -> None:
    assert not looks_not_found(fixture_html(name))


def test_body_text_last_resort() -> None:
    body = " ".join(["Plain paragraph text about payment systems."] * 10)
    parsed = parse_notification_html(f"<html><body>{body}</body></html>")
    assert parsed.strategy == "body_text"
    assert parsed.text.startswith("Plain paragraph")


def test_decodes_cp1252_bytes() -> None:
    html = (
        "<html><body><div><p>" + "Caf\xe9 rules apply. " * 20 + "</p></div></body></html>"
    ).encode("cp1252")
    assert "Caf\u00e9" in parse_notification_html(html).text


def test_node_to_text_block_structure() -> None:
    from selectolax.lexbor import LexborHTMLParser

    tree = LexborHTMLParser("<div><p>a <b>b</b>\n  c</p><p>d<br>e</p><pre>x\n  y</pre></div>")
    node = tree.css_first("div")
    assert node is not None
    lines = [ln.strip() for ln in node_to_text(node).split("\n") if ln.strip()]
    assert lines == ["a b c", "d", "e", "x", "y"]


def test_diagnose_reports_every_strategy() -> None:
    report = diagnose_html(fixture_html("notification_12701.html"))
    strategies = report["strategies"]
    assert isinstance(strategies, dict)
    assert set(strategies) == {"rbi_tableheader", "rbi_container", "main_text_density", "body_text"}
    assert strategies["rbi_tableheader"]["matched"] is True
    assert report["rbi_layout_detected"] is True
