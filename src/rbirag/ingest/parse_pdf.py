"""PDF text extraction (rbidocs.rbi.org.in attachments) with PyMuPDF."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rbirag.ingest.clean import clean_text, strip_page_numbers, strip_repeated_page_lines

logger = logging.getLogger(__name__)


@dataclass
class ParsedPDF:
    text: str
    page_count: int
    title: str | None = None
    pages: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def parse_pdf(data: bytes | Path, *, source: str | None = None) -> ParsedPDF:
    """Extract text page by page, drop running headers/footers and page numbers, clean.

    Scanned (image-only) PDFs yield empty text; OCR is intentionally out of scope.
    """
    import importlib

    pymupdf: Any = importlib.import_module("pymupdf")  # typed as Any: stubs vary by version
    try:
        if isinstance(data, Path):
            doc = pymupdf.open(data)
        else:
            doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # pymupdf raises several exception types for corrupt files
        logger.warning("parse_pdf source=%s failed to open: %s", source, exc)
        return ParsedPDF(text="", page_count=0)

    with doc:
        raw_pages: list[str] = [page.get_text("text", sort=True) for page in doc]
        meta_title = (doc.metadata or {}).get("title") or None

    pages = [strip_page_numbers(p) for p in raw_pages]
    pages = strip_repeated_page_lines(pages)
    text = clean_text("\n\n".join(pages))
    logger.info("parse_pdf source=%s pages=%d chars=%d", source, len(raw_pages), len(text))
    return ParsedPDF(
        text=text,
        page_count=len(raw_pages),
        title=meta_title.strip() if isinstance(meta_title, str) and meta_title.strip() else None,
        pages=pages,
    )
