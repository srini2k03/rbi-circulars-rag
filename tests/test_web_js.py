"""The browser search must match the Python pipeline: run the JS modules under Node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rbirag.chunking.base import Chunk
from rbirag.generation.prompts import build_user_prompt
from rbirag.index.bm25 import STOPWORDS, BM25Index, bm25_tokenize
from rbirag.retrieval.confidence import sigmoid
from rbirag.retrieval.hybrid import reciprocal_rank_fusion
from rbirag.web_export import quantize_int8

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

DOCS = [
    "Banks shall update KYC periodically for high-risk customers every two years.",
    "The policies of Small Finance Banks on bulk deposits; interest rates apply.",
    "Overseas investment reporting through the FIRMS portal within 30 days.",
    "",
    "Customers' accounts shall not be frozen; KYC KYC updation is pending.",
]
QUERIES = ["KYC updation for high risk customers", "bulk deposits interest", "unknown words zzz"]


def run_node(spec: dict, tmp_path: Path) -> dict:  # type: ignore[type-arg]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    out = subprocess.run(
        [NODE, str(ROOT / "web/tests/parity.mjs"), str(path)],
        capture_output=True,
        text=True,
        check=True,
    )  # type: ignore[list-item]
    return json.loads(out.stdout)  # type: ignore[no-any-return]


def test_js_matches_python(tmp_path: Path) -> None:
    texts = [
        "The Banks' POLICIES are for customers",
        "".join(map(chr, (0xFF2B, 0xFF39, 0xFF23))) + " norms",
        "processes classes analysis 2024-25",
    ]
    chunk = Chunk(chunk_id="c1", doc_id="rbi-1", index=0, text="Banks must comply.", start=0, end=18, n_tokens=3,
                  chunker="section", section="2 Rules",
                  metadata={"title": "T", "circular_number": "RBI/2025-26/1", "date": "2025-01-02", "department": "Department of Regulation"})  # fmt: skip
    vecs = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q, scales = quantize_int8(vecs)
    spec = {
        "stopwords": sorted(STOPWORDS),
        "docs": DOCS,
        "bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
        "texts": texts,
        "queries": QUERIES,
        "rrf": {"rankings": [["a", "b", "c"], ["b", "d", "a"]], "k": 60},
        "sigmoid": [-3.0, 0.0, 2.5],
        "prompt": {
            "question": " What applies? ",
            "chunks": [
                {
                    "title": "T",
                    "circular_number": "RBI/2025-26/1",
                    "date": "2025-01-02",
                    "department": "Department of Regulation",
                    "section": "2 Rules",
                    "text": "Banks must comply.",
                }
            ],
        },
        "cited": "Yes [1]. Also [2, 4] and [5-6] in [1949].",
        "dense": {
            "query": vecs[0].tolist(),
            "q": q.flatten().tolist(),
            "scales": scales.tolist(),
            "dim": 8,
        },
    }
    js = run_node(spec, tmp_path)
    assert js["tokens"] == [bm25_tokenize(t) for t in texts]
    index = BM25Index().build([str(i) for i in range(len(DOCS))], DOCS)
    for query, js_scores in zip(QUERIES, js["scores"], strict=True):
        py = index._bm25.get_scores(bm25_tokenize(query))
        np.testing.assert_allclose(js_scores, py, rtol=1e-9, atol=1e-12)
    py_rrf = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [r[0] for r in js["rrf"]] == [d for d, _ in py_rrf]
    np.testing.assert_allclose([r[1] for r in js["rrf"]], [s for _, s in py_rrf])
    np.testing.assert_allclose(js["sigmoid"], [sigmoid(x) for x in spec["sigmoid"]])
    assert js["prompt"] == build_user_prompt(" What applies? ", [chunk])
    assert js["cited"] == [1, 2, 4, 5, 6]
    exact = vecs @ vecs[0]
    np.testing.assert_allclose(js["dense"], exact, atol=0.02)  # int8 quantisation error
