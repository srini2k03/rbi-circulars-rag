"""Helpers for the static-site tests: build a site dir with a fixture index and serve it."""

from __future__ import annotations

import functools
import json
import shutil
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rbirag.config import Settings
from rbirag.document import Document
from rbirag.index.embedder import FakeEmbedder
from rbirag.web_export import export_web_index

ROOT = Path(__file__).resolve().parents[1]


def build_site(site: Path, docs: Sequence[Document], *, results: bool = True) -> dict[str, object]:
    shutil.copytree(ROOT / "web", site, ignore=shutil.ignore_patterns("tests", "data"))
    manifest = export_web_index(
        docs, site / "data" / "index", settings=Settings(), embedder=FakeEmbedder(64),
        chunker="section", size=128,
    )  # fmt: skip
    if results:
        res = site / "results"
        (res / "charts").mkdir(parents=True)
        (res / "run_metadata.json").write_text(json.dumps({
            "test_set": "model-generated test set (`data/eval/qa_candidates.jsonl`), 6 questions, 0 human-verified",
            "generated_at": "2026-11-03T00:00:00+00:00", "commit": "abcdef1234",
            "model_mix": {"judge": "openai/gpt-oss-120b 100%"},
        }))  # fmt: skip
        (res / "retrieval_ablation.csv").write_text(
            "config,chunk_recall@1,chunk_recall@5,chunk_recall@10,chunk_mrr@10,chunk_ndcg@10,doc_recall@5,latency_ms_p50\n"
            "section-512-hybrid_rerank,0.5,0.8,0.9,0.6,0.7,0.95,120\n"
            "fixed-256-bm25,0.3,0.6,0.7,0.4,0.5,0.9,5\n"
        )
        (res / "generation.csv").write_text(
            "config,backend,model,status,n,judge_faithfulness,nli_faithfulness,judge_relevance,citation_accuracy,refusal_precision,refusal_recall\n"
            "section-512-hybrid_rerank,openai_compatible,openai/gpt-oss-20b,measured,50,0.9,0.8,0.85,0.75,1.0,0.5\n"
            'fixed-256-bm25,ollama,qwen2.5:3b-instruct,"incomplete (3/10)",3,,,,,,\n'
        )
    return manifest


@contextmanager
def serve(directory: Path) -> Iterator[str]:
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(directory))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
