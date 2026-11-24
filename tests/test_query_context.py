"""Query understanding and context assembly (dedup, budget, conflicts)."""

from __future__ import annotations

from rbirag.chunking.base import Chunk
from rbirag.generation.context import assemble_context
from rbirag.query.understanding import analyse_query, classify, normalize, split_parts
from rbirag.retrieval.pipeline import ScoredChunk


def test_classify_and_normalize() -> None:
    assert classify("Which circular amended the KYC direction?") == "relation_lookup"
    assert classify("What is the difference between CRR and SLR?") == "comparison"
    assert classify("When do the new norms take effect?") == "temporal"
    assert classify("How much can a resident remit under LRS?") == "numeric"
    assert classify("How to report a fraud to RBI?") == "procedural"
    assert classify("What is a Key Facts Statement?") == "definition"
    assert classify("Can NBFCs accept demand deposits?") == "yes_no"
    assert normalize("  KYC\u2019s   rules ") == "KYC's rules"


def test_ambiguity_parts_and_scope() -> None:
    a = analyse_query("What is the limit?")
    assert a.ambiguous and a.ambiguity
    b = analyse_query("What does this circular require for KYC updation?")
    assert any("this/that" in r for r in b.ambiguity)
    assert split_parts("What is the CRR? What is the SLR?") == [
        "What is the CRR?", "What is the SLR?",
    ]  # fmt: skip
    c = analyse_query("What is the LRS limit and also which transactions are prohibited?")
    assert len(c.parts) == 2 and c.ambiguous
    sebi = analyse_query("What are SEBI's rules for IPO lock-in periods?")
    assert sebi.out_of_scope and "SEBI" in (sebi.scope_reason or "")
    assert not analyse_query("How do SEBI-registered entities open accounts with RBI?").out_of_scope
    cricket = analyse_query("Who won the cricket world cup?", vocabulary=lambda t: False)
    assert cricket.out_of_scope
    kyc = analyse_query("What is the KYC updation period for high risk customers?",
                        vocabulary=lambda t: True)  # fmt: skip
    assert not kyc.out_of_scope and not kyc.ambiguous and kyc.in_vocabulary == 1.0
    ref = analyse_query("What does RBI/2024-25/12 say?")
    assert ref.references == ["RBI/2024-25/12"]
    fema = analyse_query("FEMA rules on ECB in 2023")
    assert fema.filters and fema.filter_notes  # suggested, not applied


def test_follow_up_rewrite_is_explicit() -> None:
    a = analyse_query("And for low risk ones?", history=["What is the KYC updation period?"])
    assert a.follow_up and a.rewritten and "KYC updation period" in a.rewritten
    assert a.retrieval_query == a.rewritten and a.rewrite_reason
    plain = analyse_query("What is the KYC updation period?")
    assert plain.retrieval_query == plain.normalized and not plain.follow_up


def _sc(cid: str, doc: str, text: str, rank: int, date: str = "2020-01-01") -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, doc_id=doc, index=0, text=text, start=0, end=len(text),
                  n_tokens=len(text.split()), chunker="fixed", metadata={"date": date})  # fmt: skip
    return ScoredChunk(chunk=chunk, score=1.0, rank=rank)


def test_context_dedup_budget_and_conflicts() -> None:
    base = "Periodic updation of KYC shall be carried out at least once in every two years."
    results = [
        _sc("a", "d1", base, 1, "2016-02-25"),
        _sc("b", "d2", base, 2, "2017-01-01"),  # duplicate text from another version
        _sc("c", "d3", base.replace("two years", "three years"), 3, "2023-05-10"),
        _sc("d", "d4", "Bulk deposits are deposits of Rs. 3 crore and above. " * 3, 4),
    ]
    kept, rep = assemble_context(results, token_budget=40)
    assert rep.dropped_duplicates == ["b"]
    assert [r.chunk.chunk_id for r in kept] == ["a", "c"]
    assert rep.dropped_for_budget == ["d"] and rep.distinct_documents == 2
    assert rep.conflicts and rep.conflicts[0].kind == "different figures"
    assert "2016-02-25" in rep.conflicts[0].detail and "three years" in rep.conflicts[0].detail
    _, rep2 = assemble_context(results[:1] + results[2:3],
                                   relation=lambda a, b: "versions of one direction")  # fmt: skip
    assert rep2.conflicts[0].kind == "versions of one direction"
