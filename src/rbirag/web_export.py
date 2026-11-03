"""Export a static, browser-searchable index for the GitHub Pages demo.

Files written to ``<out>/``:

* ``manifest.json``     config, models, sizes, BM25 params + stopwords, prompts, notes
* ``chunks.json.gz``    ``{"fields": [...], "rows": [[...], ...]}`` chunk texts + metadata
* ``embeddings.i8``     int8 embeddings, row-major ``n x dim``
* ``scales.f32``        float32 per-row scale (``vector ~= int8 * scale``)
* ``bm25.json.gz``      document frequencies and lengths (same tokenizer as Python)

The browser embeds queries with the same model (transformers.js port of
``BAAI/bge-small-en-v1.5`` with the same query instruction), so cosine scores match the
Python pipeline up to int8 quantisation. If the export exceeds ``cap_mb`` the oldest years
are dropped until it fits, and the manifest says so.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from rbirag.chunking import build_chunker
from rbirag.chunking.base import Chunk
from rbirag.config import Settings
from rbirag.document import Document
from rbirag.generation.prompts import REFUSAL_TEXT, SYSTEM_PROMPT
from rbirag.index.bm25 import STOPWORDS, bm25_tokenize
from rbirag.index.embedder import Embedder
from rbirag.providers import GEMINI, GROQ

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
CHUNK_FIELDS = (
    "id", "doc_id", "text", "title", "date", "circular_number", "department", "url", "section",
)  # fmt: skip


def quantize_int8(
    vecs: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float32]]:
    """Symmetric per-row int8 quantisation: ``q = round(v / s)``, ``s = max|v| / 127``."""
    scales = np.abs(vecs).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(vecs / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def dequantize(q: npt.NDArray[np.int8], scales: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (q.astype(np.float32) * scales[:, None]).astype(np.float32)


def _chunk_rows(chunks: Sequence[Chunk]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for c in chunks:
        md = c.metadata
        rows.append([c.chunk_id, c.doc_id, c.text, md.get("title"), md.get("date"),
                     md.get("circular_number"), md.get("department"), md.get("url"),
                     c.section])  # fmt: skip
    return rows


def bm25_stats(texts: Sequence[str]) -> dict[str, Any]:
    df: Counter[str] = Counter()
    lengths: list[int] = []
    for t in texts:
        toks = bm25_tokenize(t) or ["\x00"]  # same placeholder as BM25Index for empty texts
        lengths.append(len(toks))
        df.update(set(toks))
    return {
        "k1": 1.5,
        "b": 0.75,
        "epsilon": 0.25,
        "avgdl": float(np.mean(lengths)) if lengths else 0.0,
        "doc_len": lengths,
        "df": dict(sorted(df.items())),
    }


def _gz_json(obj: Any) -> bytes:
    return gzip.compress(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode(), 9)


def estimate_bytes(chunks: Sequence[Chunk], dim: int) -> int:
    """Rough size: gzip ~ 0.33 of chunk text/metadata, plus embeddings and BM25 stats."""
    text = sum(len(c.text) + 120 for c in chunks)
    return int(text * 0.33) + len(chunks) * (dim + 4 + 6)


def select_years(
    docs: Sequence[Document], chunks_per_doc: dict[str, int], bytes_per_chunk: float, cap: int
) -> int | None:
    """Earliest year to keep so the estimate fits ``cap`` (``None`` = keep everything)."""
    total = sum(chunks_per_doc.values()) * bytes_per_chunk
    if total <= cap:
        return None
    by_year: dict[int, int] = Counter()
    for d in docs:
        if d.date:
            by_year[d.date.year] += chunks_per_doc.get(d.doc_id, 0)
    running = 0.0
    keep_from: int | None = None
    for year in sorted(by_year, reverse=True):
        if running + by_year[year] * bytes_per_chunk > cap and keep_from is not None:
            break
        running += by_year[year] * bytes_per_chunk
        keep_from = year
    return keep_from


def export_web_index(
    docs: Sequence[Document],
    out_dir: Path,
    *,
    settings: Settings,
    embedder: Embedder,
    chunker: str,
    size: int,
    cap_mb: float = 25.0,
    config_source: str = "default",
    batch_size: int = 256,
) -> dict[str, Any]:
    s = settings.model_copy(
        update={
            "chunk_size": size,
            "chunk_overlap": max(0, size // 8),
            "section_min_tokens": max(16, size // 8),
        }
    )
    chunk = build_chunker(chunker, s, embedder=embedder)
    all_chunks = {d.doc_id: chunk.chunk(d) for d in docs}
    flat = [c for cs in all_chunks.values() for c in cs]
    cap = int(cap_mb * 1024 * 1024)
    note = None
    kept_docs = list(docs)
    per_chunk = estimate_bytes(flat, embedder.dim) / max(1, len(flat))
    keep_from = select_years(docs, {k: len(v) for k, v in all_chunks.items()}, per_chunk, cap)
    if keep_from is not None:
        kept_docs = [d for d in docs if d.date and d.date.year >= keep_from]
        note = (
            f"The full corpus exceeds the {cap_mb:.0f} MB browser budget, so this demo index "
            f"covers notifications from {keep_from} onwards ({len(kept_docs)} of {len(docs)} "
            "documents). The API and the evaluation use the full corpus."
        )
    chunks = [c for d in kept_docs for c in all_chunks[d.doc_id]]

    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [c.text for c in chunks]
    parts = [
        embedder.embed_documents(texts[i : i + batch_size])
        for i in range(0, len(texts), batch_size)
    ]
    vecs = np.concatenate(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)
    q, scales = quantize_int8(vecs.astype(np.float32))
    (out_dir / "embeddings.i8").write_bytes(q.tobytes())
    (out_dir / "scales.f32").write_bytes(scales.astype("<f4").tobytes())
    (out_dir / "chunks.json.gz").write_bytes(
        _gz_json({"fields": CHUNK_FIELDS, "rows": _chunk_rows(chunks)})
    )
    stats = bm25_stats(texts)
    (out_dir / "bm25.json.gz").write_bytes(_gz_json(stats))

    files = {
        p.name: p.stat().st_size
        for p in out_dir.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    years = sorted(d.date.year for d in kept_docs if d.date)
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": f"{chunker}-{size}-hybrid_rerank",
        "config_source": config_source,
        "chunker": chunker,
        "chunk_size": size,
        "n_chunks": len(chunks),
        "n_documents": len(kept_docs),
        "n_documents_corpus": len(docs),
        "years": [years[0], years[-1]] if years else None,
        "note": note,
        "embedding": {
            "model": embedder.name,
            "browser_model": "Xenova/bge-small-en-v1.5",
            "dim": embedder.dim,
            "query_instruction": settings.embedding_query_instruction,
            "pooling": "cls",
            "quantization": "int8, symmetric, per-row float32 scale",
        },
        "reranker": {
            "model": settings.reranker_model,
            "browser_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        },
        "retrieval": {
            "candidate_k": settings.candidate_k,
            "rerank_k": settings.rerank_k,
            "rrf_k": settings.rrf_k,
            "top_k": settings.answer_top_k,
            "confidence_threshold": settings.confidence_threshold,
        },
        "bm25": {
            "k1": stats["k1"],
            "b": stats["b"],
            "epsilon": stats["epsilon"],
            "stopwords": sorted(STOPWORDS),
        },
        "prompt": {"system": SYSTEM_PROMPT, "refusal": REFUSAL_TEXT},
        "generators": {"groq": GROQ.generator[0], "gemini": GEMINI.generator[0]},
        "files": files,
        "total_bytes": sum(files.values()),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "exported %d chunks (%.1f MB) to %s", len(chunks), manifest["total_bytes"] / 2**20, out_dir
    )
    return manifest


def best_config(results_dir: Path) -> tuple[str, int, str]:
    """Chunker and size of the best retrieval config by chunk-level nDCG@10, if known."""
    rows: list[dict[str, Any]] = []
    for f in sorted((results_dir / "retrieval").glob("rows-*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8")))
    rows = [r for r in rows if r.get("chunk_ndcg@10") is not None]
    if not rows:
        return "section", 512, "default (no retrieval results yet)"
    best = max(rows, key=lambda r: r["chunk_ndcg@10"])
    return str(best["chunker"]), int(best["size"]), f"best retrieval config {best['config']}"
