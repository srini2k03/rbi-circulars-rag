from __future__ import annotations

from pathlib import Path

import numpy as np

from rbirag.chunking.base import Chunk
from rbirag.index.dense import QdrantIndex, point_id
from rbirag.index.embedder import FakeEmbedder, build_embedder

TEXTS = [
    "periodic updation of kyc for high risk customers",
    "interest rate on bulk deposits of small finance banks",
    "overseas investment reporting on the firms portal",
]


def chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc{i}::fixed::0000",
            doc_id=f"doc{i}",
            index=0,
            text=t,
            start=0,
            end=len(t),
            n_tokens=len(t.split()),
            chunker="fixed",
            metadata={"title": f"Title {i}", "year": 2025},
        )
        for i, t in enumerate(TEXTS)
    ]


def test_fake_embedder_is_deterministic_normalised_and_similarity_aware() -> None:
    emb = FakeEmbedder(64)
    a = emb.embed_documents(TEXTS)
    b = FakeEmbedder(64).embed_documents(TEXTS)
    assert a.shape == (3, 64) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    q = emb.embed_query("kyc updation for high risk customers")
    sims = a @ q
    assert int(np.argmax(sims)) == 0
    assert emb.embed_documents([]).shape == (0, 64)


def test_build_embedder_is_lazy() -> None:
    # Constructing the real embedder must not import torch / download anything.
    emb = build_embedder("sentence-transformers")
    assert emb.name == "BAAI/bge-small-en-v1.5"
    assert build_embedder("fake", fake_dim=16).dim == 16


def test_qdrant_in_memory_upsert_and_search() -> None:
    idx = QdrantIndex(FakeEmbedder(64), collection="t")
    idx.recreate()
    assert idx.upsert(chunks(), batch_size=2) == 3  # two batches
    assert idx.count() == 3
    results = idx.search("bulk deposits interest rate", k=2)
    assert results[0][0] == "doc1::fixed::0000"
    assert len(results) == 2 and results[0][1] >= results[1][1]
    # payload carries metadata and text
    rec = idx.client.retrieve("t", ids=[point_id("doc1::fixed::0000")], with_payload=True)[0]
    assert rec.payload["title"] == "Title 1" and rec.payload["text"] == TEXTS[1]
    # upsert is idempotent (stable uuid5 ids)
    idx.upsert(chunks())
    assert idx.count() == 3
    idx.close()


def test_qdrant_search_without_collection_returns_empty() -> None:
    assert QdrantIndex(FakeEmbedder(8), collection="missing").search("x") == []


def test_qdrant_embedded_path_persists(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    idx = QdrantIndex(FakeEmbedder(32), collection="p", path=path)
    idx.recreate()
    idx.upsert(chunks())
    idx.close()
    reopened = QdrantIndex(FakeEmbedder(32), collection="p", path=path)
    assert reopened.count() == 3
    assert reopened.search("firms portal", k=1)[0][0] == "doc2::fixed::0000"
    reopened.close()


def test_cached_embedder_shares_vectors_across_processes(tmp_path: Path) -> None:
    import numpy as np

    from rbirag.index.embedder import CachedEmbedder, FakeEmbedder

    class Counting(FakeEmbedder):
        calls = 0

        def embed_documents(self, texts):  # type: ignore[no-untyped-def,override]
            Counting.calls += len(texts)
            return super().embed_documents(texts)

    texts = ["kyc updation", "bulk deposits", "kyc updation", "form odi"]
    first = CachedEmbedder(Counting(32), tmp_path)
    a = first.embed_documents(texts)
    assert Counting.calls == 3  # duplicates embedded once
    second = CachedEmbedder(Counting(32), tmp_path)  # e.g. another CI job
    assert len(second) == 3 and second.dim == 32
    b = second.embed_documents(texts)
    assert Counting.calls == 3  # served from disk
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(b, FakeEmbedder(32).embed_documents(texts))
    second.embed_documents(["new text"])
    assert Counting.calls == 4 and len(CachedEmbedder(Counting(32), tmp_path)) == 4
