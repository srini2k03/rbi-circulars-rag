"""Dense index on Qdrant (embedded ``:memory:`` / on-disk ``path=`` mode, or a server URL)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rbirag.chunking.base import Chunk
from rbirag.index.embedder import Embedder
from rbirag.retrieval.params import RetrievalFilter

logger = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.UUID("6f1c1c5e-6a8e-4c39-9d43-2f1f3a0b8e11")


def point_id(chunk_id: str) -> str:
    """Stable UUID for a chunk id (Qdrant only accepts ints/UUIDs as point ids)."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantIndex:
    def __init__(
        self,
        embedder: Embedder,
        *,
        collection: str = "rbi_chunks",
        url: str | None = None,
        path: Path | str | None = None,
        location: str | None = None,
        client: Any = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.collection = collection
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url)
        elif path is not None:
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(location=location or ":memory:")

    def close(self) -> None:
        self.client.close()

    def recreate(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE
            ),
        )

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.recreate()

    def upsert(self, chunks: Sequence[Chunk], *, batch_size: int = 128) -> int:
        """Embed and upsert in batches. Payload = chunk metadata (incl. text)."""
        from qdrant_client import models

        self.ensure_collection()
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            points = [
                models.PointStruct(id=point_id(c.chunk_id), vector=v.tolist(), payload=c.payload())
                for c, v in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            total += len(points)
            logger.debug("upserted %d/%d", total, len(chunks))
        return total

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def search(
        self, query: str, k: int = 10, *, flt: RetrievalFilter | None = None
    ) -> list[tuple[str, float]]:
        """Top-``k`` ``(chunk_id, cosine)`` best first, optionally inside a metadata filter
        (department / category / doc id / year as a Qdrant payload filter; exact dates are
        checked on the returned payloads, over-fetching to keep ``k`` results)."""
        if not self.client.collection_exists(self.collection):
            return []
        vector = self.embedder.embed_query(query)
        if flt is None or flt.is_empty:
            response = self.client.query_points(
                self.collection, query=vector.tolist(), limit=k, with_payload=["chunk_id"]
            )
            return [
                (str((p.payload or {}).get("chunk_id")), float(p.score)) for p in response.points
            ]
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k * 4 if (flt.date_from or flt.date_to) else k,
            query_filter=qdrant_filter(flt),
            with_payload=True,
        )
        out: list[tuple[str, float]] = []
        for point in response.points:
            payload = point.payload or {}
            if flt.matches(payload):
                out.append((str(payload.get("chunk_id")), float(point.score)))
        return out[:k]


def qdrant_filter(flt: RetrievalFilter) -> Any:
    """The Qdrant payload filter for a :class:`RetrievalFilter` (year granularity)."""
    from qdrant_client import models

    must: list[Any] = []
    if flt.doc_ids:
        must.append(
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(flt.doc_ids)))
        )
    if flt.departments:
        must.append(
            models.FieldCondition(
                key="department", match=models.MatchAny(any=list(flt.departments))
            )
        )
    if flt.categories:
        must.append(
            models.FieldCondition(key="category", match=models.MatchAny(any=list(flt.categories)))
        )
    if flt.date_from or flt.date_to:
        must.append(
            models.FieldCondition(
                key="year",
                range=models.Range(
                    gte=flt.date_from.year if flt.date_from else None,
                    lte=flt.date_to.year if flt.date_to else None,
                ),
            )
        )
    return models.Filter(must=must)
