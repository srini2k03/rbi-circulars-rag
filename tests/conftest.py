from __future__ import annotations

from pathlib import Path

import pytest

from rbirag.config import get_settings
from rbirag.document import Document
from rbirag.ingest.corpus import RawRecord, build_document
from rbirag.ingest.metadata import detect_sections

FIXTURES = Path(__file__).parent / "fixtures"
HTML_DIR = FIXTURES / "html"
REAL_FIXTURES = ("notification_12701.html", "notification_12655.html", "notification_12590.html")


def fixture_html(name: str) -> bytes:
    return (HTML_DIR / name).read_bytes()


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never pick up a developer's .env or RBIRAG_* variables in tests."""
    import os

    for key in list(os.environ):
        if key.startswith("RBIRAG_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def fixture_docs() -> list[Document]:
    docs = []
    for name in REAL_FIXTURES:
        sid = int(name.split("_")[1].split(".")[0])
        doc = build_document(
            RawRecord(
                source_id=sid,
                url=f"https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={sid}&Mode=0",
                html=fixture_html(name),
            )
        )
        assert doc is not None
        docs.append(doc)
    return docs


def make_doc(text: str, doc_id: str = "d1", **kw: object) -> Document:
    return Document(
        doc_id=doc_id,
        url=f"https://example.test/{doc_id}",
        title=str(kw.pop("title", "T")),
        text=text,
        sections=detect_sections(text),
        **kw,  # type: ignore[arg-type]
    )
