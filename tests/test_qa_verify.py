from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from rich.console import Console

from rbirag.chunking import SectionChunker
from rbirag.chunking.base import Chunk
from rbirag.document import Document
from rbirag.eval.dataset import QAPair, read_qa_jsonl, write_qa_jsonl
from rbirag.eval.qa_generate import (
    Sample,
    allocate,
    generate_candidates,
    generate_pair,
    locate,
    stratified_samples,
)
from rbirag.eval.verify_cli import run_review
from rbirag.generation.llm import Message, QuotaExhaustedError, ScriptedLLM
from rbirag.index.build import chunk_documents


@pytest.fixture
def chunks(fixture_docs: list[Document]) -> list[Chunk]:
    return chunk_documents(fixture_docs, SectionChunker(max_tokens=80, min_tokens=20))


def test_allocate_proportional_min_one_and_capped() -> None:
    assert allocate({"a": 10, "b": 10}, 4) == {"a": 2, "b": 2}
    assert allocate({"a": 90, "b": 10}, 10) == {"a": 9, "b": 1}
    assert allocate({"a": 1, "b": 100, "c": 3}, 6) == {"a": 1, "b": 4, "c": 1}
    assert allocate({"a": 2, "b": 2}, 10) == {"a": 2, "b": 2}  # capped at stratum size
    assert allocate({"a": 5, "b": 1, "c": 1}, 2) == {
        "a": 2,
        "b": 0,
        "c": 0,
    }  # n < strata: proportional
    assert sum(allocate({"x": 7, "y": 3, "z": 1}, 5).values()) == 5


def test_stratified_samples_cover_strata_and_types(chunks: list[Chunk]) -> None:
    samples = stratified_samples(chunks, 8, seed=1)
    assert samples
    assert len({s.chunks[0].metadata["department"] for s in samples}) >= 2
    types = {s.qtype for s in samples}
    assert "factual" in types
    for s in samples:
        if s.qtype == "multi_chunk":
            assert len(s.chunks) == 2
            assert s.chunks[0].doc_id == s.chunks[1].doc_id
    assert [s.key for s in samples] == [s.key for s in stratified_samples(chunks, 8, seed=1)]


def test_locate_evidence(chunks: list[Chunk]) -> None:
    c = next(c for c in chunks if "two years" in c.text)
    span = locate("at least once in every two years", c)
    assert span is not None
    assert span.doc_id == c.doc_id
    fuzzy = locate("AT LEAST   once in every two\nyears", c)
    assert fuzzy is not None
    assert (fuzzy.start, fuzzy.end) == (span.start, span.end)
    assert locate("text that is not there at all", c) is None


def test_generate_pair_parses_and_checks(chunks: list[Chunk], fixture_docs: list[Document]) -> None:
    c = next(c for c in chunks if "two years" in c.text)
    reply = json.dumps({
        "question": "How often must regulated entities update KYC for high risk customers?",
        "answer": "At least once every two years.",
        "evidence": "at least once in every two years for high risk customers",
    })  # fmt: skip
    llm = ScriptedLLM([reply], model="gemini-2.5-flash")
    pair = generate_pair(llm, Sample((c,), "numeric"))
    assert pair is not None
    assert pair.auto_check is True
    assert pair.gold[0].doc_id == c.doc_id
    doc = next(d for d in fixture_docs if d.doc_id == c.doc_id)
    assert doc.text[pair.gold[0].start : pair.gold[0].end].lower().startswith("at least once")
    assert pair.generator_model == "gemini-2.5-flash"
    vague = json.dumps(
        {"question": "What does this circular say about KYC?", "answer": "x", "evidence": "nope"}
    )
    bad = generate_pair(ScriptedLLM([vague]), Sample((c,), "factual"))
    assert bad is not None
    assert bad.auto_check is False
    assert bad.gold  # falls back to the chunk span
    assert generate_pair(ScriptedLLM(["not json"]), Sample((c,), "factual")) is None


def scripted_qa(messages: Sequence[Message], system: str | None) -> str:
    prompt = messages[-1].content
    if "RBI notifications cannot answer" in prompt:
        return json.dumps({"questions": ["What is the SEBI limit for mutual fund expense ratios in 2025?",
                                         "What is the GST rate on gold jewellery making charges?"]})  # fmt: skip
    passage = prompt.split("PASSAGE:\n", 1)[1]
    words = passage.split()
    return json.dumps({"question": "What does the RBI direction require from regulated entities here?",
                       "answer": "As stated.", "evidence": " ".join(words[:8])})  # fmt: skip


def test_generate_candidates_resumes_after_quota(chunks: list[Chunk]) -> None:
    calls = {"n": 0}

    def limited(messages: Sequence[Message], system: str | None) -> str:
        calls["n"] += 1
        if calls["n"] > 2:
            raise QuotaExhaustedError("daily budget")
        return scripted_qa(messages, system)

    first = generate_candidates(ScriptedLLM(limited), chunks, n_answerable=5, n_unanswerable=2)
    assert not first.complete
    assert len(first.pairs) == 2
    second = generate_candidates(
        ScriptedLLM(scripted_qa), chunks, n_answerable=5, n_unanswerable=2, existing=first.pairs
    )
    assert second.complete
    assert second.attempted == len(stratified_samples(chunks, 5)) - 2  # finished items skipped
    assert sum(not p.answerable for p in second.pairs) == 2
    assert all(
        p.qtype == "unanswerable" and p.answer == "" for p in second.pairs if not p.answerable
    )


def _pairs(n: int) -> list[QAPair]:
    return [
        QAPair(
            qa_id=f"q{i}",
            question=f"Question number {i}?",
            answer=f"A{i}",
            qtype="factual",
            source_text="src",
        )
        for i in range(n)
    ]


def test_review_accept_edit_reject_and_resume(tmp_path: Path) -> None:
    cand = tmp_path / "cand.jsonl"
    out = tmp_path / "qa_v1.jsonl"
    write_qa_jsonl(_pairs(4), cand)
    choices = iter(["a", "e", "r", "q"])
    texts = iter(["Edited question?", "Edited answer", "duplicate"])
    log = (tmp_path / "log.txt").open("w")
    console = Console(file=log, width=100)
    summary = run_review(
        cand, out, seed=0, console=console,
        ask_choice=lambda p, c, d: next(choices), ask_text=lambda p, d: next(texts),
    )  # fmt: skip
    assert summary["accepted"] == 1
    assert summary["edited"] == 1
    assert summary["rejected"] == 1
    assert summary["pending"] == 1
    assert summary["human_verified"] == 2
    saved = {p.qa_id: p for p in read_qa_jsonl(out)}
    edited = next(p for p in saved.values() if p.review_status == "edited")
    assert edited.question == "Edited question?"
    assert edited.verified
    rejected = next(p for p in saved.values() if p.review_status == "rejected")
    assert rejected.reviewer_note == "duplicate"
    assert not rejected.verified
    # resume: only the remaining pending pair is shown
    shown: list[str] = []
    run_review(
        cand, out, console=console,
        ask_choice=lambda p, c, d: shown.append("x") or "a", ask_text=lambda p, d: d,
    )  # fmt: skip
    assert len(shown) == 1
    assert sum(p.verified for p in read_qa_jsonl(out)) == 3
