from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from rbirag.document import read_documents_jsonl, write_documents_jsonl
from rbirag.ingest.corpus import (
    RawRecord,
    build_document,
    build_documents,
    corpus_stats,
    iter_raw_records,
)
from rbirag.ingest.fetch_rbi import Manifest, ManifestEntry
from tests.conftest import fixture_html

STUB_HTML = b"""<html><body><table class="tablebg">
<tr><td class="tableheader"><b>Date : </b>Mar 28, 2024</td></tr>
<tr><td class="tableheader"><b>Master Direction - Stub Page</b></td></tr>
<tr><td class="tablecontent2"><a href="https://rbidocs.rbi.org.in/rdocs/x.PDF">PDF</a>
Please see the attached PDF for the full text of this notification issued today.
The direction is applicable to all regulated entities as specified in the attachment.
</td></tr></table></body></html>"""


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    for line in text.split("\n"):
        page.insert_text((72, y), line, fontsize=10)
        y += 14
    data: bytes = doc.tobytes()
    doc.close()
    return data


PDF_TEXT = "\n".join(
    [
        "RBI/2023-24/140",
        "DOR.CRE.REC.85/21.01.003/2023-24",
        "March 28, 2024",
        "All Commercial Banks",
        "Madam / Dear Sir,",
        "Stub Direction Subject",
        *[
            f"{i}. Paragraph number {i} sets out a substantive requirement for banks."
            for i in range(1, 12)
        ],
    ]
)


def test_stub_html_uses_pdf_text_and_html_title() -> None:
    doc = build_document(
        RawRecord(source_id=9, url="u", html=STUB_HTML, pdf=make_pdf(PDF_TEXT), pdf_url="p")
    )
    assert doc is not None
    assert doc.source_type == "pdf"
    assert doc.parse_strategy == "pdf(html:rbi_tableheader)"
    assert doc.title == "Master Direction - Stub Page"
    assert doc.circular_number == "RBI/2023-24/140"
    assert doc.department == "Department of Regulation"
    assert doc.date is not None and doc.date.isoformat() == "2024-03-28"
    assert len(doc.sections) >= 11


def test_empty_page_yields_no_document() -> None:
    raw = RawRecord(source_id=1, url="u", html=fixture_html("notification_13050_empty.html"))
    assert build_document(raw) is None


def test_iter_raw_records_from_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "html").mkdir(parents=True)
    (raw / "html" / "12701.html").write_bytes(fixture_html("notification_12701.html"))
    m = Manifest(raw / "manifest.jsonl")
    now = datetime(2025, 7, 16, tzinfo=UTC)
    m.record(
        ManifestEntry(id=12701, url="u1", status="ok", fetched_at=now, html_path="html/12701.html")
    )
    m.record(ManifestEntry(id=12700, url="u0", status="not_found", fetched_at=now))
    records = list(iter_raw_records(raw))
    assert [r.source_id for r in records] == [12701]
    assert records[0].fetched_at == now
    docs = list(build_documents(records))
    assert docs[0].fetched_at == now and docs[0].doc_id == "rbi-12701"


def test_documents_jsonl_roundtrip_and_stats(tmp_path: Path, fixture_docs: list) -> None:  # type: ignore[type-arg]
    path = tmp_path / "docs.jsonl"
    assert write_documents_jsonl(fixture_docs, path) == 3
    loaded = read_documents_jsonl(path)
    assert loaded == fixture_docs
    stats = corpus_stats(loaded)
    assert stats["documents"] == 3
    assert stats["parse_strategies"] == {
        "rbi_tableheader": 1, "rbi_container": 1, "main_text_density": 1,
    }  # fmt: skip
    cov = stats["metadata_coverage_pct"]
    assert cov["date"] == 100.0 and cov["circular_number"] == 100.0 and cov["title"] == 100.0
    assert stats["date_range"] == ["2025-04-22", "2025-07-15"]
    json.dumps(stats)  # serialisable
    assert corpus_stats([])["documents"] == 0


def test_pdf_stub_page_is_not_treated_as_missing() -> None:
    from rbirag.ingest.parse_html import looks_not_found, parse_notification_html

    parsed = parse_notification_html(STUB_HTML)
    assert parsed.strategy == "rbi_tableheader"
    assert parsed.pdf_links == ["https://rbidocs.rbi.org.in/rdocs/x.PDF"]
    assert not looks_not_found(STUB_HTML)
