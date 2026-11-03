"""Build and load the on-disk index bundle (chunks + BM25 + Qdrant) for one chunker."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rbirag.chunking.base import Chunk, Chunker
from rbirag.document import Document
from rbirag.index.bm25 import BM25Index
from rbirag.index.dense import QdrantIndex
from rbirag.index.embedder import Embedder
from rbirag.index.store import ChunkStore

logger = logging.getLogger(__name__)


@dataclass
class IndexBundle:
    store: ChunkStore
    bm25: BM25Index
    dense: QdrantIndex | None


def chunk_documents(docs: Iterable[Document], chunker: Chunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc))
    return chunks


def build_index(
    chunks: list[Chunk],
    *,
    embedder: Embedder | None,
    collection: str = "rbi_chunks",
    qdrant_url: str | None = None,
    qdrant_path: Path | None = None,
    batch_size: int = 128,
) -> IndexBundle:
    """Build BM25 + (optionally) dense indexes over ``chunks``. Dense is skipped if no embedder."""
    store = ChunkStore(chunks)
    bm25 = BM25Index().build([c.chunk_id for c in chunks], [c.text for c in chunks])
    dense: QdrantIndex | None = None
    if embedder is not None:
        dense = QdrantIndex(embedder, collection=collection, url=qdrant_url, path=qdrant_path)
        dense.recreate()
        dense.upsert(chunks, batch_size=batch_size)
    logger.info("indexed %d chunks (dense=%s)", len(chunks), dense is not None)
    return IndexBundle(store=store, bm25=bm25, dense=dense)


def save_bundle(bundle: IndexBundle, index_dir: Path, *, info: dict[str, object]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    bundle.store.save(index_dir / "chunks.jsonl")
    bundle.bm25.save(index_dir / "bm25.pkl")
    (index_dir / "index_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_bundle(
    index_dir: Path,
    *,
    embedder: Embedder | None,
    collection: str = "rbi_chunks",
    qdrant_url: str | None = None,
    qdrant_path: Path | None = None,
) -> IndexBundle:
    store = ChunkStore.load(index_dir / "chunks.jsonl")
    bm25 = BM25Index.load(index_dir / "bm25.pkl")
    dense = None
    if embedder is not None and (qdrant_url or qdrant_path):
        dense = QdrantIndex(embedder, collection=collection, url=qdrant_url, path=qdrant_path)
    return IndexBundle(store=store, bm25=bm25, dense=dense)
