"""Load a bank's internal policy document (PDF, plain text or Markdown) as clean text."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rbirag.ingest.clean import normalize_unicode, normalize_whitespace

PolicySourceType = Literal["pdf", "text"]
TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES
MAX_POLICY_BYTES = 10 * 1024 * 1024  # 10 MB: far above any real policy document


class PolicyLoadError(ValueError):
    """The input is not a usable policy document (format, size, or no extractable text)."""


class PolicyDocument(BaseModel):
    name: str
    text: str
    source_type: PolicySourceType
    pages: int | None = Field(default=None, ge=0)


def clean_policy_text(text: str) -> str:
    """Unicode folding and whitespace normalisation; line structure is kept (the clause
    splitter relies on it)."""
    return normalize_whitespace(normalize_unicode(text.replace("\t", " ")))


def policy_from_text(text: str, *, name: str = "policy") -> PolicyDocument:
    cleaned = clean_policy_text(text)
    if not cleaned:
        raise PolicyLoadError("the policy document is empty")
    return PolicyDocument(name=name, text=cleaned, source_type="text")


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_policy(source: Path | bytes, *, filename: str | None = None) -> PolicyDocument:
    """Load a policy from a path, or from raw bytes plus the original ``filename``.

    The format is chosen by file extension. Scanned (image-only) PDFs have no text layer
    and are rejected with a clear message; OCR is out of scope.
    """
    if isinstance(source, Path):
        name = filename or source.name
        if not source.is_file():
            raise PolicyLoadError(f"no such file: {source}")
        if source.stat().st_size > MAX_POLICY_BYTES:
            raise PolicyLoadError(f"{name} is larger than {MAX_POLICY_BYTES // 2**20} MB")
        data = source.read_bytes()
    else:
        if filename is None:
            raise PolicyLoadError("filename is required when loading from bytes")
        name, data = filename, source
        if len(data) > MAX_POLICY_BYTES:
            raise PolicyLoadError(f"{name} is larger than {MAX_POLICY_BYTES // 2**20} MB")

    suffix = Path(name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return policy_from_text(_decode(data), name=name)
    if suffix in PDF_SUFFIXES:
        from rbirag.ingest.parse_pdf import parse_pdf

        parsed = parse_pdf(data, source=name)
        if parsed.page_count == 0:
            raise PolicyLoadError(f"{name} could not be opened as a PDF")
        if parsed.is_empty:
            raise PolicyLoadError(f"{name} has no extractable text (scanned PDF? OCR unsupported)")
        return PolicyDocument(
            name=name,
            text=clean_policy_text(parsed.text),
            source_type="pdf",
            pages=parsed.page_count,
        )
    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    raise PolicyLoadError(f"unsupported file type {suffix or '(none)'!r}; use one of {supported}")
