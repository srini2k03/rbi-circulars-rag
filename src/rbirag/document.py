"""Typed corpus model shared by ingestion, chunking and indexing."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["html", "pdf"]
MetadataValue = str | int | None


class Section(BaseModel):
    """A heading or numbered paragraph, located by character offsets into ``Document.text``."""

    model_config = ConfigDict(frozen=True)

    heading: str
    number: str | None = None
    level: int = Field(default=1, ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    kind: Literal["heading", "numbered"] = "heading"


class Document(BaseModel):
    """One processed RBI notification."""

    doc_id: str
    source_id: int | None = None
    url: str
    pdf_url: str | None = None
    title: str
    date: dt.date | None = None
    department: str | None = None
    circular_number: str | None = None  # e.g. "RBI/2024-25/12"
    ref_number: str | None = None  # e.g. "DOR.STR.REC.5/21.06.201/2024-25"
    addressee: str | None = None
    text: str
    sections: list[Section] = Field(default_factory=list)
    source_type: SourceType = "html"
    parse_strategy: str = "unknown"
    fetched_at: dt.datetime | None = None
    sha256: str | None = None

    def metadata(self) -> dict[str, MetadataValue]:
        """Flat, JSON-friendly metadata copied onto every chunk (and into Qdrant payloads)."""
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "title": self.title,
            "date": self.date.isoformat() if self.date else None,
            "year": self.date.year if self.date else None,
            "department": self.department,
            "circular_number": self.circular_number,
            "ref_number": self.ref_number,
        }


def write_documents_jsonl(docs: Iterable[Document], path: Path) -> int:
    """Write documents as JSON Lines; returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json() + "\n")
            n += 1
    tmp.replace(path)
    return n


def iter_documents_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Document.model_validate(json.loads(line))


def read_documents_jsonl(path: Path) -> list[Document]:
    return list(iter_documents_jsonl(path))
