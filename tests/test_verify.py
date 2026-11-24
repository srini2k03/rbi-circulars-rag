"""Citation verification: existence, attribution, relevance and claim support."""

from __future__ import annotations

from rbirag.generation.verify import (
    NOT_ESTABLISHED,
    LexicalSupport,
    SourceRef,
    remove_invalid_citations,
    verify_answer,
)

SOURCES = [
    SourceRef(n=1, doc_id="rbi-1", circular_number="RBI/2024-25/12",
              text="Periodic updation of KYC shall be carried out at least once in every two "
                   "years for high risk customers. Low risk customers are updated every ten "
                   "years."),
    SourceRef(n=2, doc_id="rbi-2", ref_number="DOR.STR.REC.5/21.06.201/2024-25",
              text="Small finance banks may offer differential interest on bulk deposits of "
                   "Rs. 3 crore and above."),
]  # fmt: skip
SCORER = LexicalSupport()


def test_supported_answer() -> None:
    rep = verify_answer(
        "KYC updation is carried out every two years for high risk customers [1].",
        SOURCES, SCORER,
    )  # fmt: skip
    assert rep.verdict == "supported" and rep.supported_share == 1.0
    assert rep.claims[0].supporting_source == 1 and rep.claims[0].evidence
    assert rep.citations[0].exists and rep.citations[0].relevant
    assert rep.statement is None


def test_wrong_number_is_unsupported() -> None:
    rep = verify_answer(
        "KYC updation is carried out every three years for high risk customers [1]. "
        "The deposit threshold is Rs. 5 crore [2].",
        SOURCES, SCORER,
    )  # fmt: skip
    assert rep.verdict == "unsupported"
    assert rep.statement == NOT_ESTABLISHED
    assert len(rep.unsupported_claims) == 2  # "5 crore" is not in source 2
    assert rep.citations[1].relevant is False


def test_partial_support_and_uncited_source() -> None:
    rep = verify_answer(
        "Low risk customers are updated every ten years [2]. "
        "High risk customers must be updated every 5 years [1].",
        SOURCES, SCORER,
    )  # fmt: skip
    assert rep.verdict == "partially_supported"
    first = rep.claims[0]
    assert first.status == "supported_by_uncited_source" and first.supporting_source == 1
    assert rep.statement and "every 5 years" in rep.statement


def test_invalid_citations_and_unknown_references() -> None:
    answer = "Banks follow circular RBI/2019-20/99 for KYC updation every two years [1, 7]."
    rep = verify_answer(answer, SOURCES, SCORER)
    assert rep.invalid_citations == [7]
    assert rep.unknown_references == ["RBI/2019-20/99"]
    assert [c.exists for c in rep.citations] == [True, False]
    assert remove_invalid_citations(answer, [7]).endswith("two years [1].")
    assert remove_invalid_citations("A [7]. B [1].", [7]) == "A . B [1]."


def test_numbers_in() -> None:
    from rbirag.generation.verify import numbers_in

    assert numbers_in("every two years, Rs. 5,00,000 and 3.5 per cent") == {"2", "500000", "3.5"}


def test_refusal_and_non_answer() -> None:
    assert verify_answer("I don't know based on the provided RBI notifications.", SOURCES,
                         SCORER).verdict == "refusal"  # fmt: skip
    rep = verify_answer("The documents do not specify the penalty for late updation.", SOURCES,
                        SCORER)  # fmt: skip
    assert rep.non_answer and rep.verdict == "unsupported"


def test_pipeline_abstains_on_unsupported(fixture_docs) -> None:  # type: ignore[no-untyped-def]
    from rbirag.chunking import SectionChunker
    from rbirag.generation.llm import FakeLLM
    from rbirag.generation.rag import RAGPipeline
    from rbirag.index.build import build_index, chunk_documents
    from rbirag.retrieval.params import RetrievalParams
    from rbirag.retrieval.pipeline import Retriever

    chunks = chunk_documents(fixture_docs, SectionChunker(max_tokens=120, min_tokens=30))
    b = build_index(chunks, embedder=None)
    retr = Retriever(b.store, bm25=b.bm25, params=RetrievalParams(mode="bm25"))

    class Liar(FakeLLM):
        def generate(self, messages, **kw):  # type: ignore[no-untyped-def]
            resp = super().generate(messages, **kw)
            return resp.model_copy(update={"text": "The limit is Rs. 987654 crore [1] [9]."})

    q = "How often is periodic updation of KYC for high risk customers?"
    annotate = RAGPipeline(retr, Liar(), verifier=LexicalSupport()).answer(q)
    assert annotate.verification is not None and annotate.verification.verdict == "unsupported"
    assert "[9]" not in annotate.answer and annotate.raw_answer and "[9]" in annotate.raw_answer
    assert annotate.citations.invalid_ids == [9]  # evaluation still sees the model's error
    assert not annotate.refused and "verify" in annotate.timings_ms
    abstain = RAGPipeline(retr, Liar(), verifier=LexicalSupport(), verify_policy="abstain")
    out = abstain.answer(q)
    assert out.refused and out.refusal_reason == "unsupported" and out.answer == NOT_ESTABLISHED
