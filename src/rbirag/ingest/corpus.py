"""Turn raw fetched pages (+ PDFs) into processed :class:`~rbirag.document.Document` JSONL."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rbirag.document import Document
from rbirag.ingest.fetch_rbi import DEFAULT_URL_TEMPLATE, Manifest, sha256_bytes
from rbirag.ingest.metadata import extract_metadata
from rbirag.ingest.parse_html import parse_notification_html, parsed_looks_not_found
from rbirag.ingest.parse_pdf import parse_pdf

logger = logging.getLogger(__name__)

MIN_HTML_CHARS = 400  # below this the HTML is treated as a stub pointing at the PDF
METADATA_FIELDS = ("title", "date", "department", "circular_number", "ref_number", "addressee")


@dataclass(frozen=True)
class RawRecord:
    source_id: int
    url: str
    html: bytes
    pdf: bytes | None = None
    pdf_url: str | None = None
    fetched_at: datetime | None = None


def build_document(record: RawRecord, *, min_html_chars: int = MIN_HTML_CHARS) -> Document | None:
    """Parse, clean and annotate one notification. Returns ``None`` if no text is found."""
    parsed = parse_notification_html(record.html, base_url=record.url, source=str(record.source_id))
    pdf_text = ""
    pdf_title: str | None = None
    if record.pdf:
        parsed_pdf = parse_pdf(record.pdf, source=record.pdf_url)
        pdf_text, pdf_title = parsed_pdf.text, parsed_pdf.title

    if parsed_looks_not_found(parsed) and not pdf_text:
        logger.info("notification %s looks like an empty/unused id; skipped", record.source_id)
        return None
    if len(parsed.text) >= min_html_chars or len(parsed.text) >= len(pdf_text):
        text, source_type, strategy = parsed.text, "html", parsed.strategy
    else:
        text, source_type, strategy = pdf_text, "pdf", f"pdf(html:{parsed.strategy})"
    if not text.strip():
        logger.warning("no text for notification %s", record.source_id)
        return None

    md = extract_metadata(text, html_title=parsed.title, date_hint=parsed.date_hint)
    if pdf_text and source_type == "html":
        # HTML stubs sometimes omit the letter header; the PDF usually has it.
        pdf_md = extract_metadata(pdf_text)
        md.circular_number = md.circular_number or pdf_md.circular_number
        md.ref_number = md.ref_number or pdf_md.ref_number
        md.date = md.date or pdf_md.date
        md.department = md.department or pdf_md.department
        md.addressee = md.addressee or pdf_md.addressee
    title = md.title
    if title == "Untitled notification" and pdf_title:
        title = pdf_title

    return Document(
        doc_id=f"rbi-{record.source_id}",
        source_id=record.source_id,
        url=record.url,
        pdf_url=record.pdf_url or (parsed.pdf_links[0] if parsed.pdf_links else None),
        title=title,
        date=md.date,
        department=md.department,
        circular_number=md.circular_number,
        ref_number=md.ref_number,
        addressee=md.addressee,
        text=text,
        sections=md.sections,
        source_type="pdf" if source_type == "pdf" else "html",
        parse_strategy=strategy,
        fetched_at=record.fetched_at,
        sha256=sha256_bytes(record.html),
    )


def iter_raw_records(
    raw_dir: Path, url_template: str = DEFAULT_URL_TEMPLATE
) -> Iterator[RawRecord]:
    """Records from the crawler manifest, or - if there is none - every ``html/<id>.html``."""
    manifest_path = raw_dir / "manifest.jsonl"
    if manifest_path.exists():
        for entry in Manifest(manifest_path).ok_entries():
            if not entry.html_path:
                continue
            html_file = raw_dir / entry.html_path
            if not html_file.exists():
                logger.warning("manifest points at missing file %s", html_file)
                continue
            pdf_file = raw_dir / entry.pdf_path if entry.pdf_path else None
            yield RawRecord(
                source_id=entry.id,
                url=entry.url,
                html=html_file.read_bytes(),
                pdf=pdf_file.read_bytes() if pdf_file and pdf_file.exists() else None,
                pdf_url=entry.pdf_url,
                fetched_at=entry.fetched_at,
            )
        return

    for html_file in sorted((raw_dir / "html").glob("*.html")):
        m = re.search(r"(\d+)", html_file.stem)
        if not m:
            continue
        sid = int(m.group(1))
        pdf_file = raw_dir / "pdf" / f"{sid}.pdf"
        yield RawRecord(
            source_id=sid,
            url=url_template.format(id=sid),
            html=html_file.read_bytes(),
            pdf=pdf_file.read_bytes() if pdf_file.exists() else None,
        )


def build_documents(records: Iterable[RawRecord]) -> Iterator[Document]:
    for record in records:
        try:
            doc = build_document(record)
        except Exception:  # one malformed page must not kill a 2,000-doc run
            logger.exception("failed to build document %s", record.source_id)
            continue
        if doc is not None:
            yield doc


def corpus_stats(docs: Iterable[Document]) -> dict[str, Any]:
    """Doc count, parse-strategy counts, metadata coverage (%), date range, size stats."""
    docs = list(docs)
    n = len(docs)
    coverage: dict[str, float] = {}
    for name in METADATA_FIELDS:
        present = 0
        for d in docs:
            value = getattr(d, name)
            if name == "title":
                present += int(bool(value) and value != "Untitled notification")
            else:
                present += int(value is not None and value != "")
        coverage[name] = round(100.0 * present / n, 1) if n else 0.0
    coverage["sections"] = round(100.0 * sum(1 for d in docs if d.sections) / n, 1) if n else 0.0
    dates = sorted(d.date for d in docs if d.date)
    chars = sorted(len(d.text) for d in docs)
    return {
        "documents": n,
        "parse_strategies": dict(Counter(d.parse_strategy for d in docs).most_common()),
        "source_types": dict(Counter(d.source_type for d in docs)),
        "metadata_coverage_pct": coverage,
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()] if dates else None,
        "docs_per_year": dict(sorted(Counter(d.year for d in dates).items())),
        "departments": dict(Counter(d.department or "unknown" for d in docs).most_common(15)),
        "chars": {
            "total": sum(chars),
            "median": chars[n // 2] if n else 0,
            "min": chars[0] if n else 0,
            "max": chars[-1] if n else 0,
        },
    }
