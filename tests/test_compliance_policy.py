from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from rbirag.compliance.policy import PolicyLoadError, load_policy, policy_from_text


def make_pdf(path: Path, lines: list[str]) -> Path:
    pymupdf: Any = importlib.import_module("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    doc.save(path)
    doc.close()
    return path


def test_text_and_markdown_are_cleaned(tmp_path: Path) -> None:
    src = tmp_path / "policy.md"
    src.write_text(
        "﻿# Policy\r\n\r\n\r\n\r\n1.1  The\tBank shall “not” act.\n",
        encoding="utf-8",
    )
    doc = load_policy(src)
    assert doc.source_type == "text" and doc.name == "policy.md"
    assert doc.text == '# Policy\n\n1.1 The Bank shall "not" act.'


def test_bytes_need_a_filename() -> None:
    with pytest.raises(PolicyLoadError, match="filename"):
        load_policy(b"1.1 text")
    assert load_policy(b"1.1 The Bank shall comply.", filename="p.txt").text.startswith("1.1")


def test_cp1252_text_is_decoded() -> None:
    doc = load_policy("Rs. 10 lakh \u2013 limit".encode("cp1252"), filename="p.txt")
    assert "lakh - limit" in doc.text


def test_pdf_is_parsed(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "policy.pdf",
        ["1. Governance", "1.1 The Board shall review this Policy annually.", "1.2 Second."],
    )
    doc = load_policy(pdf)
    assert doc.source_type == "pdf" and doc.pages == 1
    assert "1.1 The Board shall review this Policy annually." in doc.text


def test_empty_pdf_and_bad_inputs_are_rejected(tmp_path: Path) -> None:
    blank = make_pdf(tmp_path / "blank.pdf", [])
    with pytest.raises(PolicyLoadError, match="no extractable text"):
        load_policy(blank)
    with pytest.raises(PolicyLoadError, match="could not be opened"):
        load_policy(b"not a pdf", filename="x.pdf")
    with pytest.raises(PolicyLoadError, match="unsupported file type"):
        load_policy(b"x", filename="policy.docx")
    with pytest.raises(PolicyLoadError, match="no such file"):
        load_policy(tmp_path / "missing.txt")
    with pytest.raises(PolicyLoadError, match="empty"):
        policy_from_text("  \n ")


def test_oversized_input_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import rbirag.compliance.policy as policy

    monkeypatch.setattr(policy, "MAX_POLICY_BYTES", 10)
    with pytest.raises(PolicyLoadError, match="larger than"):
        load_policy(b"x" * 11, filename="p.txt")
