from __future__ import annotations

import pytest

from rbirag.chunking import SectionChunker
from rbirag.config import ModelPrice
from rbirag.document import Document
from rbirag.generation.llm import FakeLLM, LLMResponse, Message, ScriptedLLM
from rbirag.generation.prompts import REFUSAL_TEXT, build_user_prompt
from rbirag.generation.rag import RAGPipeline
from rbirag.index.build import build_index, chunk_documents
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.pipeline import Retriever
from rbirag.retrieval.rerank import FakeReranker

Q = "How often is periodic updation of KYC for high risk customers?"


@pytest.fixture
def retriever(fixture_docs: list[Document]) -> Retriever:
    chunks = chunk_documents(fixture_docs, SectionChunker(max_tokens=120, min_tokens=30))
    b = build_index(chunks, embedder=FakeEmbedder(128))
    return Retriever(b.store, bm25=b.bm25, dense=b.dense, reranker=FakeReranker(), rerank_k=10)


def test_prompt_numbers_sources(retriever: Retriever) -> None:
    chunks = [r.chunk for r in retriever.retrieve(Q, top_k=2).results]
    prompt = build_user_prompt(Q, chunks)
    assert prompt.startswith("Sources:\n\n[1] ")
    assert "\n\n[2] " in prompt
    assert "RBI/2025-26/41" in prompt  # metadata in the source header
    assert prompt.endswith(f"Question: {Q}")


def test_answer_with_valid_citations_and_cost(retriever: Retriever) -> None:
    llm = ScriptedLLM(
        ["Every two years for high-risk customers [1]. Medium risk: eight years [1]."],
        model="claude-haiku-5-5",
        backend="anthropic",
    )
    rag = RAGPipeline(
        retriever, llm, top_k=3, prices={"claude-haiku-5-5": ModelPrice(input=1.0, output=1.0)}
    )
    ans = rag.answer(Q)
    assert not ans.refused
    assert ans.citations.valid
    assert ans.citations.cited_ids == [1]
    assert ans.sources[0].doc_id == "rbi-12655"
    assert len(ans.sources) == 3
    expected = (ans.usage.input_tokens + ans.usage.output_tokens) / 1_000_000
    assert ans.cost.usd == pytest.approx(expected)
    assert set(ans.timings_ms) == {"retrieve", "rerank", "generate", "total"}
    system, messages = llm.calls[0]
    assert system is not None and "square brackets" in system
    assert messages[0].content.startswith("Sources:")


def test_invalid_citations_flagged(retriever: Retriever) -> None:
    rag = RAGPipeline(retriever, ScriptedLLM(["It is every two years [9]."]), top_k=3)
    ans = rag.answer(Q)
    assert ans.citations.invalid_ids == [9]
    assert not ans.citations.valid


def test_gate_refuses_without_calling_llm(retriever: Retriever) -> None:
    llm = ScriptedLLM([])
    ans = RAGPipeline(retriever, llm).answer("monsoon cricket score in Chennai")
    assert ans.refused
    assert ans.refusal_reason == "low_confidence"
    assert ans.answer == REFUSAL_TEXT
    assert llm.calls == []
    assert "generate" not in ans.timings_ms
    # with the gate off the LLM is asked (and may refuse itself)
    llm2 = ScriptedLLM([REFUSAL_TEXT])
    ans2 = RAGPipeline(retriever, llm2, gate=False).answer("monsoon cricket score in Chennai")
    assert ans2.refused
    assert ans2.refusal_reason == "model_refusal"
    assert len(llm2.calls) == 1


def test_modes_without_reranker_have_no_gate(retriever: Retriever) -> None:
    ans = RAGPipeline(
        retriever, ScriptedLLM(["Some answer here is given [1]."]), mode="hybrid"
    ).answer("monsoon cricket")
    assert ans.confidence is None
    assert not ans.refused


def test_provider_refusal(retriever: Retriever) -> None:
    class Declines(ScriptedLLM):
        def generate(self, messages, **kw):  # type: ignore[no-untyped-def,override]
            return LLMResponse(text="", model="m", backend="anthropic", refused=True)

    ans = RAGPipeline(retriever, Declines(replies=[])).answer(Q)
    assert ans.refused
    assert ans.refusal_reason == "model_declined"


def test_stream_events(retriever: Retriever) -> None:
    events = list(RAGPipeline(retriever, FakeLLM(), top_k=3).stream(Q))
    assert events[0].type == "sources"
    assert len(events[0].sources) == 3
    assert {e.type for e in events[1:-1]} == {"delta"}
    done = events[-1]
    assert done.type == "done"
    assert done.answer is not None
    assert "".join(e.text for e in events[1:-1]).strip() == done.answer.answer
    assert done.answer.citations.cited_ids == [1]
    # gated stream: sources, one delta with the refusal, done
    gated = list(RAGPipeline(retriever, FakeLLM()).stream("monsoon cricket"))
    assert [e.type for e in gated] == ["sources", "delta", "done"]


def test_fake_llm_refuses_without_sources() -> None:
    assert (
        FakeLLM()
        .generate([Message(role="user", content="Sources:\n\n(no sources)\n\nQuestion: x")])
        .text
        == REFUSAL_TEXT
    )
