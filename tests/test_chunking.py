from __future__ import annotations

import itertools
import re

import numpy as np
import pytest

from rbirag.chunking import FixedSizeChunker, SectionChunker, SemanticChunker, build_chunker
from rbirag.chunking.base import (
    Chunk,
    Chunker,
    RegexTokenizer,
    WhitespaceTokenizer,
    get_tokenizer,
    split_sentences,
    token_windows,
)
from rbirag.config import Settings
from rbirag.document import Document
from rbirag.index.embedder import FakeEmbedder, FloatArray
from tests.conftest import make_doc


def assert_offsets(doc: Document, chunks: list[Chunk]) -> None:
    for i, c in enumerate(chunks):
        assert c.text == doc.text[c.start : c.end]
        assert c.index == i
        assert c.doc_id == doc.doc_id
        assert c.chunk_id == f"{doc.doc_id}::{c.chunker}::{i:04d}"
        assert c.text == c.text.strip() and c.text


def assert_no_lost_text(doc: Document, chunks: list[Chunk]) -> None:
    """Every non-whitespace character of the document is inside some chunk."""
    covered = bytearray(len(doc.text))
    for c in chunks:
        covered[c.start : c.end] = b"\x01" * (c.end - c.start)
    lost = [ch for ch, cov in zip(doc.text, covered, strict=True) if not cov and not ch.isspace()]
    assert lost == []


# ----------------------------------------------------------------------------- tokenizers


def test_regex_tokenizer_spans() -> None:
    text = "Rs. 5,000 under DOR.STR.REC.5/21.06.201 isn't"
    toks = [text[s:e] for s, e in RegexTokenizer().spans(text)]
    assert toks == ["Rs", ".", "5", ",", "000", "under", "DOR.STR.REC.5/21.06.201", "isn't"]


def test_whitespace_tokenizer_spans() -> None:
    assert WhitespaceTokenizer().spans(" a  bc\nd ") == [(1, 2), (4, 6), (7, 8)]


def test_get_tokenizer_unknown() -> None:
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_tokenizer("nope")


def test_token_windows_overlap() -> None:
    spans = [(i, i + 1) for i in range(10)]
    assert token_windows(spans, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    assert token_windows(spans, 4, 0) == [(0, 4), (4, 8), (8, 10)]
    assert token_windows(spans, 20, 5) == [(0, 10)]
    assert token_windows([], 4, 1) == []
    with pytest.raises(ValueError, match="overlap"):
        token_windows(spans, 4, 4)


def test_split_sentences_handles_abbreviations() -> None:
    text = (
        "Please refer to Circular No. 12 dated 1.4.2024. Banks shall comply, i.e. at once. "
        "(ii) Next item.\nNew line"
    )
    sents = [text[s:e] for s, e in split_sentences(text)]
    assert sents == [
        "Please refer to Circular No. 12 dated 1.4.2024.",
        "Banks shall comply, i.e. at once.",
        "(ii) Next item.",
        "New line",
    ]


# ----------------------------------------------------------------------------- fixed


def test_fixed_chunker_sizes_offsets_and_overlap() -> None:
    words = [f"w{i}" for i in range(100)]
    doc = make_doc(" ".join(words))
    chunks = FixedSizeChunker(chunk_size=30, overlap=10, tokenizer=WhitespaceTokenizer()).chunk(doc)
    assert [c.n_tokens for c in chunks] == [30, 30, 30, 30, 20]
    assert_offsets(doc, chunks)
    assert_no_lost_text(doc, chunks)
    for a, b in itertools.pairwise(chunks):
        assert a.text.split()[-10:] == b.text.split()[:10]  # exactly `overlap` shared tokens
    assert chunks[0].metadata["doc_id"] == "d1"


def test_fixed_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker(chunk_size=10, overlap=10)


def test_fixed_chunker_empty_doc() -> None:
    assert FixedSizeChunker().chunk(make_doc("   ")) == []


# ----------------------------------------------------------------------------- section


SECTIONED = "\n".join(
    [
        "RBI/2025-26/1",
        "April 1, 2025",
        "All Banks",
        "Madam / Dear Sir,",
        "Subject Line Here",
        "1. " + "alpha " * 5,
        "2. " + "beta " * 5,
        "3. " + "gamma " * 120,
        "4. " + "delta " * 5,
        "Chapter II - Big Section",
        "5. " + "epsilon " * 30,
    ]
)


def test_section_chunker_respects_boundaries_merges_and_splits() -> None:
    doc = make_doc(SECTIONED)
    chunker = SectionChunker(
        max_tokens=50,
        min_tokens=15,
        fixed=FixedSizeChunker(50, 5, WhitespaceTokenizer()),
    )
    chunks = chunker.chunk(doc)
    assert_offsets(doc, chunks)
    assert_no_lost_text(doc, chunks)
    assert all(c.n_tokens <= 50 for c in chunks)
    texts = [c.text for c in chunks]
    # small consecutive segments are merged (subject heading + para 1)
    assert any("Subject Line Here" in t and "alpha" in t for t in texts)
    # the large para 3 is split, and never shares a chunk with para 2 or 4
    gamma = [c for c in chunks if "gamma" in c.text]
    assert len(gamma) >= 3
    assert all("beta" not in c.text and "delta" not in c.text for c in gamma)
    assert all(c.section and c.section.startswith("3 ") for c in gamma)
    # each chunk starts on a section/line boundary or inside a split section
    boundaries = {0} | {s.start for s in doc.sections}
    for c in chunks:
        if "gamma" not in c.text or c is gamma[0]:
            assert c.start in boundaries or doc.text[c.start - 1] == "\n"


def test_section_chunker_without_sections_falls_back_to_one_segment() -> None:
    doc = make_doc("just some words here " * 3)
    chunks = SectionChunker(max_tokens=100, min_tokens=5).chunk(doc)
    assert len(chunks) == 1 and chunks[0].section is None


def test_section_chunker_on_fixtures(fixture_docs: list[Document]) -> None:
    chunker = SectionChunker(max_tokens=80, min_tokens=20)
    for doc in fixture_docs:
        chunks = chunker.chunk(doc)
        assert chunks
        assert_offsets(doc, chunks)
        assert_no_lost_text(doc, chunks)


# ----------------------------------------------------------------------------- semantic


class TopicEmbedder:
    """Sentences mentioning 'cat' vs anything else land on orthogonal axes."""

    name = "topic"
    dim = 2

    def embed_documents(self, texts: list[str]) -> FloatArray:
        rows = [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> FloatArray:
        return self.embed_documents([text])[0]


def test_semantic_chunker_splits_on_topic_shift() -> None:
    text = (
        "The cat sat. The cat ate. A cat slept. "
        "Banks report data. Banks file returns. Banks keep records."
    )
    doc = make_doc(text)
    chunker = SemanticChunker(TopicEmbedder(), percentile=50, min_tokens=1, max_tokens=100)
    chunks = chunker.chunk(doc)
    assert [c.text for c in chunks] == [
        "The cat sat. The cat ate. A cat slept.",
        "Banks report data. Banks file returns. Banks keep records.",
    ]
    assert_offsets(doc, chunks)
    assert_no_lost_text(doc, chunks)


def test_semantic_breakpoints_percentile() -> None:
    chunker = SemanticChunker(FakeEmbedder(8), percentile=30)
    sims = np.array([0.9, 0.1, 0.8, 0.7, 0.05])
    assert chunker.breakpoints(sims) == [1, 4]


def test_semantic_chunker_merges_small_and_splits_large(fixture_docs: list[Document]) -> None:
    chunker = SemanticChunker(FakeEmbedder(32), percentile=30, min_tokens=20, max_tokens=60)
    for doc in fixture_docs:
        chunks = chunker.chunk(doc)
        assert_offsets(doc, chunks)
        assert_no_lost_text(doc, chunks)
        assert all(c.n_tokens <= 60 for c in chunks)
        # undersized pieces only survive when no neighbour has room for them
        for i, c in enumerate(chunks):
            if c.n_tokens < 20:
                neighbours = chunks[max(0, i - 1) : i] + chunks[i + 1 : i + 2]
                assert all(nb.n_tokens + c.n_tokens > 60 for nb in neighbours)


def test_build_chunker_factory() -> None:
    s = Settings(chunk_size=128, chunk_overlap=16)
    for name in ("fixed", "section"):
        assert isinstance(build_chunker(name, s), Chunker)
    assert isinstance(build_chunker("semantic", s, embedder=FakeEmbedder()), SemanticChunker)
    with pytest.raises(ValueError, match="embedder"):
        build_chunker("semantic", s)
    with pytest.raises(ValueError, match="unknown chunker"):
        build_chunker("magic", s)


def test_tiktoken_tokenizer_offsets() -> None:
    pytest.importorskip("tiktoken")
    try:
        tok = get_tokenizer("tiktoken")
    except Exception as exc:  # vocabulary download needs network
        pytest.skip(f"tiktoken encoding unavailable offline: {exc}")
    text = "Interest Rate on Deposits"
    spans = tok.spans(text)
    assert spans and all(text[s:e].strip() == text[s:e] for s, e in spans)
    assert re.sub(r"\s", "", "".join(text[s:e] for s, e in spans)) == re.sub(r"\s", "", text)
