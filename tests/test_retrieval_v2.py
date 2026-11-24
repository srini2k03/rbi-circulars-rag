"""Retrieval architecture: weighted BM25, fusion, expansion, filters, parents, tracing."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from rbirag.chunking import FixedSizeChunker, SectionChunker
from rbirag.document import Document
from rbirag.index.bm25 import BM25Index, bm25_tokenize
from rbirag.index.build import build_index, chunk_documents
from rbirag.index.embedder import FakeEmbedder
from rbirag.query.filters import detect_filters
from rbirag.retrieval.expansion import (
    AcronymEntry,
    AcronymExpander,
    expand_query,
    load_acronyms,
    mine_acronyms,
    rm3,
)
from rbirag.retrieval.fusion import interpolate, normalise, weighted_fusion
from rbirag.retrieval.params import RetrievalFilter, RetrievalParams
from rbirag.retrieval.pipeline import Retriever
from rbirag.retrieval.rerank import FakeReranker

TEXTS = [
    "Know Your Customer (KYC) norms apply to all regulated entities.",
    "Periodic updation of KYC for high risk customers is every two years.",
    "Priority sector lending (PSL) targets for small finance banks.",
    "Interest rates on bulk deposits may be differential.",
    "",  # empty chunk: rank_bm25 needs a placeholder token
]


@pytest.fixture
def bm25() -> BM25Index:
    return BM25Index().build([f"c{i}" for i in range(len(TEXTS))], TEXTS)


def test_weighted_scores_equal_rank_bm25(bm25: BM25Index) -> None:
    ref = BM25Okapi([bm25_tokenize(t) or ["\x00"] for t in TEXTS])
    for query in ["KYC updation high risk", "bulk deposits interest", "kyc kyc customers"]:
        toks = bm25_tokenize(query)
        expected = ref.get_scores(toks)
        got = bm25.score_terms([(t, 1.0) for t in toks])
        assert np.array_equal(expected, got)  # bit-identical, not just close


def test_search_mask_and_weights(bm25: BM25Index) -> None:
    hits = bm25.search("KYC customers", 5)
    assert hits[0][0] in {"c0", "c1"}
    mask = bm25.mask(lambda cid: cid != hits[0][0])
    assert hits[0][0] not in [c for c, _ in bm25.search("KYC customers", 5, allowed=mask)]
    with pytest.raises(ValueError, match="one entry per indexed chunk"):
        bm25.search("KYC", 5, allowed=np.ones(2, dtype=bool))
    double = bm25.search_terms([("kyc", 2.0)], 5)
    single = bm25.search_terms([("kyc", 1.0)], 5)
    assert [c for c, _ in double] == [c for c, _ in single]
    assert double[0][1] == pytest.approx(2 * single[0][1])
    assert bm25.search_terms([("unknownterm", 1.0)], 5) == []


def test_normalise_and_weighted_fusion() -> None:
    assert normalise([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]
    assert normalise([2.0, 2.0]) == [1.0, 1.0]
    z = normalise([1.0, 2.0, 3.0], "zscore")
    assert z[1] == pytest.approx(0.0) and z[2] == pytest.approx(-z[0])
    with pytest.raises(ValueError, match="normalisation"):
        normalise([1.0], "rank")  # type: ignore[arg-type]
    sparse = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
    dense = [("c", 0.9), ("b", 0.8), ("d", 0.1)]
    fused = dict(weighted_fusion([sparse, dense], [0.5, 0.5]))
    assert fused["b"] == pytest.approx(0.5 * 0.5 + 0.5 * 0.875)
    assert fused["d"] == pytest.approx(0.0)  # missing from BM25 -> floor 0, dense min -> 0
    only_bm25 = weighted_fusion([sparse, dense], [1.0, 0.0])
    assert [c for c, _ in only_bm25][:2] == ["a", "b"]
    with pytest.raises(ValueError, match="one weight"):
        weighted_fusion([sparse], [1.0, 1.0])


def test_interpolate_rerank() -> None:
    first = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    rr = {"a": -1.0, "b": 2.0, "c": 0.0}
    assert [c for c, _ in interpolate(first, rr, rerank_weight=1.0)] == ["b", "c", "a"]
    assert [c for c, _ in interpolate(first, rr, rerank_weight=0.0)] == ["a", "b", "c"]
    mixed = dict(interpolate(first, rr, rerank_weight=0.5))
    assert mixed["b"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)


def test_params_hash_describe_and_filter() -> None:
    p = RetrievalParams(mode="hybrid", fusion="weighted", bm25_weight=0.7, dense_weight=0.3)
    assert p.config_hash() == RetrievalParams(**p.model_dump()).config_hash()
    assert p.config_hash() != RetrievalParams(mode="hybrid").config_hash()
    assert "a=0.70" in p.describe()
    assert "+rerank@30" in RetrievalParams(mode="bm25_rerank", rerank_depth=30).describe()
    with pytest.raises(ValueError, match="positive bm25_weight"):
        RetrievalParams(fusion="weighted", bm25_weight=0, dense_weight=0)
    flt = RetrievalFilter(departments=("Foreign Exchange Department",),
                          date_from=dt.date(2020, 1, 1))  # fmt: skip
    assert flt.matches({"department": "foreign exchange department", "date": "2021-05-01"})
    assert not flt.matches({"department": "Foreign Exchange Department", "date": "2019-05-01"})
    assert not flt.matches({"department": "Foreign Exchange Department", "date": None})
    assert not flt.matches({"department": "Department of Regulation", "date": "2021-05-01"})
    assert RetrievalFilter().is_empty
    with pytest.raises(ValueError, match="on or before"):
        RetrievalFilter(date_from=dt.date(2021, 1, 1), date_to=dt.date(2020, 1, 1))


def test_mine_acronyms_and_expand() -> None:
    texts = [
        "The Know Your Customer (KYC) norms. Know Your Customer (KYC) again.",
        "Know Your Customer (KYC) and Non-Banking Financial Companies (NBFCs) rules.",
        "Non-Banking Financial Companies (NBFC) ... Non Banking Financial Companies (NBFC).",
        "Information Technology (IT) is too short to keep.",
    ]
    entries = {e.acronym: e for e in mine_acronyms(texts, min_count=2)}
    assert entries["KYC"].expansion == "know your customer"
    assert entries["NBFC"].expansion == "non banking financial companies"
    assert "IT" not in entries
    ex = AcronymExpander(list(entries.values()))
    eq = ex.expand("KYC rules for NBFCs", weight=0.5)
    assert eq.added["customer"] == 0.5 and "company" in eq.added
    assert "KYC = know your customer" in eq.notes
    back = ex.expand("know your customer rules", weight=0.5)
    assert back.added == {"kyc": 0.5}
    guarded = AcronymExpander([AcronymEntry("PAN", "permanent account number", 9, 1.0, False)])
    assert guarded.expand("open a pan account", weight=0.5).added == {}
    assert guarded.expand("open a PAN account", weight=0.5).added


def test_committed_acronym_dictionary_loads() -> None:
    entries = load_acronyms()
    by = {e.acronym: e.expansion for e in entries}
    assert by["KYC"] == "know your customer"
    assert len(entries) > 100


def test_rm3_adds_feedback_terms(bm25: BM25Index) -> None:
    terms, added = rm3(bm25, [("kyc", 1.0)], fb_docs=2, fb_terms=5, orig_weight=0.5)
    weights = dict(terms)
    assert weights["kyc"] > 0.5  # original term keeps (at least) its interpolated weight
    assert added and all(t not in {"kyc"} for t in added)
    assert sum(weights.values()) == pytest.approx(1.0)
    same, none = rm3(bm25, [("zzz", 1.0)], fb_docs=2, fb_terms=5, orig_weight=0.5)
    assert same == [("zzz", 1.0)] and none == {}
    eq = expand_query("KYC", bm25, expansion="none", acronyms=None, acronym_weight=0.5,
                      fb_docs=2, fb_terms=5, orig_weight=0.5)  # fmt: skip
    assert eq.terms == [("kyc", 1.0)] and eq.summary() == "no expansion"
    with pytest.raises(ValueError, match="AcronymExpander"):
        expand_query("KYC", bm25, expansion="acronyms", acronyms=None, acronym_weight=0.5,
                     fb_docs=2, fb_terms=5, orig_weight=0.5)  # fmt: skip


def test_detect_filters() -> None:
    d = detect_filters("FEMA rules for ECB in 2023?", today=dt.date(2026, 1, 1))
    assert d.filter is not None
    assert d.filter.departments == ("Foreign Exchange Department",)
    assert (d.filter.date_from, d.filter.date_to) == (dt.date(2022, 1, 1), dt.date(2024, 12, 31))
    fy = detect_filters("DPSS circulars in FY 2022-23", today=dt.date(2026, 1, 1)).filter
    assert fy is not None and fy.date_to == dt.date(2024, 12, 31)
    assert "Department of Payment and Settlement Systems" in fy.departments
    assert detect_filters("How do I open an account?").filter is None
    assert detect_filters("rules from 1850").filter is None  # not a plausible year


@pytest.fixture
def full(fixture_docs: list[Document]) -> Retriever:
    chunks = chunk_documents(fixture_docs, SectionChunker(max_tokens=120, min_tokens=30))
    b = build_index(chunks, embedder=FakeEmbedder(64))
    return Retriever(b.store, bm25=b.bm25, dense=b.dense, reranker=FakeReranker())


Q = "How often is periodic updation of KYC for high risk customers?"


def test_modes_trace_and_ranks(full: Retriever) -> None:
    for params in [
        RetrievalParams(mode="hybrid", fusion="weighted", bm25_weight=0.7, dense_weight=0.3),
        RetrievalParams(mode="hybrid", fusion="weighted", score_norm="zscore"),
        RetrievalParams(mode="bm25_rerank", rerank_depth=5, rerank_weight=0.5),
        RetrievalParams(mode="hybrid_rerank", expansion="rm3"),
        RetrievalParams(mode="bm25", expansion="acronyms"),
    ]:
        r = full.retrieve(Q, top_k=3, params=params)
        assert r.results and r.results[0].chunk.doc_id == "rbi-12655", params.describe()
        assert r.trace is not None and r.trace.params_hash == params.config_hash()
        assert all(x.rank == i for i, x in enumerate(r.results, 1))
    r = full.retrieve(Q, top_k=3, params=RetrievalParams(mode="hybrid_rerank"))
    top = r.results[0]
    assert {"bm25", "dense", "rerank"} <= set(top.stage_scores)
    assert {"fusion", "rerank"} <= set(top.stage_ranks)
    assert r.confidence is not None
    assert r.trace is not None and r.trace.candidates["rerank"] >= 3
    acr = full.retrieve("KYC updation", top_k=3, params=RetrievalParams(mode="bm25",
                        expansion="acronyms"))  # fmt: skip
    assert acr.trace is not None and "customer" in acr.trace.expansion


def test_thresholds_drop_passages(full: Retriever) -> None:
    strict = RetrievalParams(mode="bm25_rerank", rerank_min_score=4.99)
    r = full.retrieve(Q, top_k=5, params=strict)
    assert r.trace is not None and r.trace.dropped["rerank_min_score"] > 0
    assert all(x.stage_scores["rerank"] >= 4.99 for x in r.results)
    dense = full.retrieve(Q, top_k=5, params=RetrievalParams(mode="hybrid",
                          dense_min_similarity=0.99))  # fmt: skip
    assert dense.trace is not None and dense.trace.dropped["dense_min_similarity"] > 0


def test_filters_restrict_both_arms(full: Retriever) -> None:
    flt = RetrievalFilter(doc_ids=("rbi-12590",))
    r = full.retrieve(Q, top_k=3, filters=flt, params=RetrievalParams(mode="hybrid"))
    assert r.results and {x.chunk.doc_id for x in r.results} == {"rbi-12590"}
    assert r.trace is not None and r.trace.filters == {"doc_ids": ["rbi-12590"]}
    dept = full.store.get(r.results[0].chunk.chunk_id).metadata.get("department")
    if dept:
        r2 = full.retrieve(Q, top_k=3, mode="dense",
                           filters=RetrievalFilter(departments=(str(dept),)))  # fmt: skip
        assert all(x.chunk.metadata.get("department") == dept for x in r2.results)


def test_parent_windows(fixture_docs: list[Document]) -> None:
    chunks = chunk_documents(fixture_docs, FixedSizeChunker(40, 8))
    b = build_index(chunks, embedder=None)
    doc_text = {d.doc_id: d.text for d in fixture_docs}
    ret = Retriever(b.store, bm25=b.bm25, params=RetrievalParams(mode="bm25"))
    plain = ret.retrieve(Q, top_k=3)
    parents = ret.retrieve(Q, top_k=3, params=RetrievalParams(mode="bm25", parent="window",
                           parent_tokens=120))  # fmt: skip
    assert parents.results[0].child_chunk_id == plain.results[0].chunk.chunk_id
    for r in parents.results:
        c = r.chunk
        assert c.text == doc_text[c.doc_id][c.start : c.end]  # the stitched window is exact
        assert c.n_tokens <= 120 + 40
    spans = [(r.chunk.doc_id, r.chunk.start, r.chunk.end) for r in parents.results]
    for i, (d, a, e) in enumerate(spans):
        for d2, a2, e2 in spans[i + 1 :]:
            assert d != d2 or e <= a2 or e2 <= a  # merged: no overlapping windows
