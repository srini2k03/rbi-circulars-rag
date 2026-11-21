from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rbirag.chunking import build_chunker
from rbirag.compliance.regulations import RegulationStore, UnknownNotificationError
from rbirag.config import Settings
from rbirag.document import Document, write_documents_jsonl
from rbirag.index.build import build_index, chunk_documents, save_bundle
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.pipeline import make_retriever
from rbirag.retrieval.rerank import FakeReranker
from tests.conftest import make_doc

LATER = make_doc(
    "RBI/2025-26/99\nDOR.AML.REC.70/14.01.001/2025-26\nOctober 1, 2025\n\n"
    "1. The following circulars are hereby withdrawn: DOR.AML.REC.19/14.01.001/2025-26.\n\n"
    "2. Periodic updation of KYC for high risk customers shall continue as per the KYC "
    "Directions applicable to banks.",
    doc_id="rbi-12999",
    title="Withdrawal of KYC circulars",
    date=dt.date(2025, 10, 1),
    circular_number="RBI/2025-26/99",
    ref_number="DOR.AML.REC.70/14.01.001/2025-26",
)


def settings(**kw: object) -> Settings:
    return Settings(chunker="fixed", chunk_size=64, chunk_overlap=8, **kw)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def corpus(fixture_docs: list[Document]) -> list[Document]:
    return [*fixture_docs, LATER]


@pytest.fixture(scope="module")
def store(corpus: list[Document]) -> RegulationStore:
    s = settings()
    chunks = chunk_documents(corpus, build_chunker("fixed", s))
    bundle = build_index(chunks, embedder=FakeEmbedder(64))
    retriever = make_retriever(
        s, bundle.store, bm25=bundle.bm25, dense=bundle.dense, reranker=FakeReranker()
    )
    return RegulationStore(retriever, {d.doc_id: d for d in corpus}, top_k=3, fetch_max_tokens=200)


def test_search_returns_one_hit_per_notification(store: RegulationStore) -> None:
    result = store.search("periodic updation of KYC for high risk customers")
    ids = [h.doc_id for h in result.hits]
    assert len(ids) == len(set(ids)) and 1 <= len(ids) <= 3
    assert ids[0] == "rbi-12655"
    assert result.mode == "hybrid_rerank" and result.confidence is not None
    top = result.hits[0]
    assert top.rank == 1 and top.date == "2025-06-12" and top.circular_number == "RBI/2025-26/41"
    assert top.url and top.url.startswith("https://www.rbi.org.in/")
    assert len(top.snippet) <= 504


def test_search_flags_replaced_notifications(store: RegulationStore) -> None:
    result = store.search("periodic updation of KYC for high risk customers")
    top = result.hits[0]
    assert top.doc_id == "rbi-12655" and top.replaced_by == ["rbi-12999"]


def test_search_date_filter(store: RegulationStore) -> None:
    result = store.search("deposits interest rate", date_from=dt.date(2025, 7, 1))
    assert result.hits and all(h.date and h.date >= "2025-07-01" for h in result.hits)
    result = store.search("deposits interest rate", date_to=dt.date(2025, 5, 1))
    assert all(h.date and h.date <= "2025-05-01" for h in result.hits)


def test_fetch_resolves_identifiers_and_selects_passages(store: RegulationStore) -> None:
    for ident in ("rbi-12655", "RBI-12655", "12655", "RBI/2025-26/41", "RBI/2025-2026/41"):
        assert store.resolve(ident) == "rbi-12655"
    assert "DOR.AML.REC.19/14.01.001/2025-26" in store
    view = store.fetch("12655", query="periodic updation high risk")
    assert view.doc_id == "rbi-12655" and view.passages
    assert view.total_chars == len(store.documents["rbi-12655"].text)
    assert [r.doc_id for r in view.cited_by] == ["rbi-12999"]
    assert view.cited_by[0].relation == "supersedes"
    assert view.replaced_by == ["rbi-12999"]


def test_unknown_notification(store: RegulationStore) -> None:
    assert "rbi-1" not in store and 42 not in store
    with pytest.raises(UnknownNotificationError, match="no notification 'rbi-1'"):
        store.fetch("rbi-1")


def test_from_settings_and_cli(tmp_path: Path, corpus: list[Document]) -> None:
    s = settings(
        processed_dir=tmp_path / "processed",
        index_dir=tmp_path / "indexes",
        embedder="fake",
        reranker="fake",
        retrieval_mode="hybrid_rerank",
        qdrant_path=tmp_path / "qdrant",
    )
    with pytest.raises(FileNotFoundError, match="no documents"):
        RegulationStore.from_settings(s)
    write_documents_jsonl(corpus, s.documents_path)
    with pytest.raises(FileNotFoundError, match="no index"):
        RegulationStore.from_settings(s)
    chunks = chunk_documents(corpus, build_chunker("fixed", s))
    emb = FakeEmbedder(s.fake_embedding_dim)
    bundle = build_index(
        chunks, embedder=emb, collection="rbi_chunks_fixed", qdrant_path=s.qdrant_path
    )
    save_bundle(bundle, s.index_dir / "fixed", info={})
    assert bundle.dense is not None
    bundle.dense.close()

    bm25_store = RegulationStore.from_settings(
        s.model_copy(update={"compliance_search_mode": "bm25"})
    )
    assert bm25_store.search("KYC updation").mode == "bm25"
    full = RegulationStore.from_settings(s)
    assert full.search("KYC updation").mode == "hybrid_rerank"
    assert full.retriever.dense is not None
    full.retriever.dense.close()

    from rbirag.cli import app

    env = {
        "RBIRAG_PROCESSED_DIR": str(s.processed_dir),
        "RBIRAG_INDEX_DIR": str(s.index_dir),
        "RBIRAG_CHUNKER": "fixed",
    }
    runner = CliRunner()
    out = runner.invoke(app, ["comply", "search", "KYC updation", "--mode", "bm25"], env=env)
    assert out.exit_code == 0, out.output
    assert "rbi-12655" in out.output
    out = runner.invoke(app, ["comply", "fetch", "12655", "--query", "updation"], env=env)
    assert out.exit_code == 0 and "rbi-12655" in out.output and "replaced by" in out.output
    out = runner.invoke(app, ["comply", "fetch", "rbi-1"], env=env)
    assert out.exit_code == 1 and "no notification" in out.output


def test_cli_clauses(tmp_path: Path) -> None:
    from rbirag.cli import app

    policy = tmp_path / "p.md"
    policy.write_text("# P\n\n1.1 The Bank shall review the policy annually.\n", encoding="utf-8")
    runner = CliRunner()
    out = runner.invoke(app, ["comply", "clauses", str(policy)])
    assert out.exit_code == 0 and "[1.1]" in out.output
    out = runner.invoke(app, ["comply", "clauses", str(policy), "--json"])
    assert '"number": "1.1"' in out.output
    out = runner.invoke(app, ["comply", "clauses", str(tmp_path / "x.docx")])
    assert out.exit_code == 1
