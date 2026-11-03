from __future__ import annotations

import datetime as dt
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rbirag.config import Settings
from rbirag.document import Document
from rbirag.index.embedder import FakeEmbedder
from rbirag.web_export import best_config, dequantize, export_web_index, quantize_int8
from tests.webutil import ROOT, build_site, serve


def test_quantize_roundtrip() -> None:
    vecs = np.random.default_rng(1).normal(size=(50, 32)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q, s = quantize_int8(vecs)
    assert q.dtype == np.int8
    assert np.abs(q).max() == 127
    np.testing.assert_allclose(dequantize(q, s), vecs, atol=0.01)
    zq, zs = quantize_int8(np.zeros((1, 4), dtype=np.float32))
    assert zs[0] == 1.0
    assert not zq.any()


def test_export_files_and_manifest(fixture_docs: list[Document], tmp_path: Path) -> None:
    out = tmp_path / "idx"
    m = export_web_index(
        fixture_docs,
        out,
        settings=Settings(),
        embedder=FakeEmbedder(64),
        chunker="section",
        size=128,
    )
    assert m["n_documents"] == 3
    assert m["note"] is None
    assert m["embedding"]["dim"] == 64
    assert m["prompt"]["refusal"].startswith("I don't know")
    assert m["generators"]["groq"] == "openai/gpt-oss-20b"
    n = m["n_chunks"]
    assert (out / "embeddings.i8").stat().st_size == n * 64
    assert (out / "scales.f32").stat().st_size == n * 4
    chunks = json.loads(gzip.decompress((out / "chunks.json.gz").read_bytes()))
    assert chunks["fields"][0] == "id"
    assert len(chunks["rows"]) == n
    bm25 = json.loads(gzip.decompress((out / "bm25.json.gz").read_bytes()))
    assert len(bm25["doc_len"]) == n
    assert bm25["df"]["kyc"] >= 1
    assert m["total_bytes"] == sum(m["files"].values())


def test_export_trims_oldest_years_over_budget(
    fixture_docs: list[Document], tmp_path: Path
) -> None:
    docs = [
        d.model_copy(update={"doc_id": f"{d.doc_id}-{y}", "date": dt.date(y, 1, 1)})
        for y in (2019, 2022, 2025) for d in fixture_docs
    ]  # fmt: skip
    full = export_web_index(
        docs,
        tmp_path / "a",
        settings=Settings(),
        embedder=FakeEmbedder(64),
        chunker="section",
        size=128,
    )
    cap_mb = full["total_bytes"] * 0.5 / 2**20
    m = export_web_index(
        docs,
        tmp_path / "b",
        settings=Settings(),
        embedder=FakeEmbedder(64),
        chunker="section",
        size=128,
        cap_mb=cap_mb,
    )
    assert m["note"] is not None
    assert "onwards" in m["note"]
    assert m["years"][1] == 2025
    assert m["years"][0] > 2019
    assert m["n_documents"] < len(docs)


def test_best_config(tmp_path: Path) -> None:
    assert best_config(tmp_path)[:2] == ("section", 512)
    (tmp_path / "retrieval").mkdir()
    (tmp_path / "retrieval" / "rows-x.json").write_text(json.dumps([
        {"config": "fixed-256-hybrid", "chunker": "fixed", "size": 256, "chunk_ndcg@10": 0.4},
        {"config": "semantic-1024-hybrid_rerank", "chunker": "semantic", "size": 1024, "chunk_ndcg@10": 0.6},
    ]))  # fmt: skip
    assert best_config(tmp_path)[:2] == ("semantic", 1024)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_loader_reads_the_export(fixture_docs: list[Document], tmp_path: Path) -> None:
    site = tmp_path / "site"
    m = build_site(site, fixture_docs)
    with serve(site) as base:
        out = subprocess.run(
            ["node", str(ROOT / "web/tests/search-index.mjs"), f"{base}/data/index", "periodic updation of KYC high risk customers"],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
    res = json.loads(out.stdout)
    assert res["n"] == m["n_chunks"]
    assert res["dim"] == 64
    assert res["scales"] == m["n_chunks"]
    assert res["hits"][0]["doc_id"] == "rbi-12655"
