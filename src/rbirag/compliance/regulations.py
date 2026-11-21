"""Search and fetch over the RBI corpus: what the compliance agent's tools call.

``search`` reuses the RAG retriever unchanged (same modes, fusion, reranker and
confidence as the reported retrieval results) and groups chunks by notification, so the
agent sees distinct notifications with their date and status rather than five chunks of
one Master Direction. ``fetch`` returns a notification's metadata, table of contents,
the passages most relevant to the clause under review, and which later notifications cite
it (and whether they say they repeal or supersede it).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from rbirag.compliance.entities import TARGET_ENTITY, entity_of, is_other_entity
from rbirag.compliance.references import (
    ReferenceGraph,
    Relation,
    normalise_circular_number,
    normalise_ref_number,
)
from rbirag.compliance.sections import Passage, select_passages, table_of_contents
from rbirag.document import Document, iter_documents_jsonl

if TYPE_CHECKING:
    from rbirag.config import Settings
    from rbirag.index.embedder import Embedder
    from rbirag.retrieval.pipeline import Retriever
    from rbirag.retrieval.rerank import Reranker

MAX_CITED_BY = 8


class UnknownNotificationError(KeyError):
    """No notification in the corpus matches the given identifier."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown notification"


class ReferenceOut(BaseModel):
    doc_id: str
    date: str | None
    title: str
    relation: Relation


class SearchHit(BaseModel):
    rank: int = Field(ge=1)
    doc_id: str
    title: str | None
    date: str | None
    entity: str | None = Field(default=None, description="Entity type the Direction addresses.")
    circular_number: str | None
    department: str | None
    url: str | None
    section: str | None
    snippet: str
    score: float
    replaced_by: list[str] = Field(
        default_factory=list,
        description="Later notifications whose text says they repeal or supersede this one.",
    )


class SearchResult(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
    confidence: float | None = Field(
        default=None, description="Reranker confidence that the top hit is relevant (0-1)."
    )
    timings_ms: dict[str, float] = Field(default_factory=dict)
    excluded_other_entity: int = Field(
        default=0, description="Hits dropped because they address another entity type."
    )


class NotificationView(BaseModel):
    doc_id: str
    title: str
    date: str | None
    entity: str | None = None
    department: str | None
    circular_number: str | None
    ref_number: str | None
    url: str
    total_chars: int
    table_of_contents: list[str]
    passages: list[Passage]
    truncated: bool = Field(description="Whether passages were left out to fit the budget.")
    cited_by: list[ReferenceOut] = Field(
        default_factory=list, description="Later notifications citing this one."
    )
    replaced_by: list[str] = Field(default_factory=list)


def _snippet(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit].rstrip(" ,;") + " ..."


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)


class RegulationStore:
    def __init__(
        self,
        retriever: Retriever,
        documents: Mapping[str, Document],
        *,
        graph: ReferenceGraph | None = None,
        top_k: int = 5,
        depth_factor: int = 4,
        snippet_chars: int = 500,
        fetch_max_tokens: int = 1500,
        passage_chars: int = 2000,
        entity: str = TARGET_ENTITY,
        entity_filter: bool = True,
    ) -> None:
        self.retriever = retriever
        self.documents = dict(documents)
        self.graph = graph if graph is not None else ReferenceGraph.build(self.documents.values())
        self.top_k = top_k
        self.depth_factor = depth_factor
        self.snippet_chars = snippet_chars
        self.fetch_max_tokens = fetch_max_tokens
        self.passage_chars = passage_chars
        self.entity = entity
        self.entity_filter = entity_filter
        self._aliases: dict[str, str] = {}
        for d in self.documents.values():
            if d.circular_number:
                self._aliases.setdefault(normalise_circular_number(d.circular_number), d.doc_id)
            if d.ref_number:
                self._aliases.setdefault(normalise_ref_number(d.ref_number), d.doc_id)

    # -- identifiers ---------------------------------------------------------------------
    def resolve(self, identifier: str) -> str:
        """A doc id (``rbi-12655``), bare RBI id (``12655``), circular number or reference
        number -> doc id. Raises :class:`UnknownNotificationError`."""
        key = identifier.strip()
        if key in self.documents:
            return key
        lowered = key.lower()
        if lowered in self.documents:
            return lowered
        if key.isdigit() and f"rbi-{key}" in self.documents:
            return f"rbi-{key}"
        for alias in (normalise_circular_number(key), normalise_ref_number(key)):
            if alias in self._aliases:
                return self._aliases[alias]
        raise UnknownNotificationError(f"no notification {identifier!r} in the corpus")

    def __contains__(self, identifier: object) -> bool:
        if not isinstance(identifier, str):
            return False
        try:
            self.resolve(identifier)
        except UnknownNotificationError:
            return False
        return True

    # -- search --------------------------------------------------------------------------
    def close(self) -> None:
        """Release the dense index (local Qdrant holds an exclusive lock on its folder,
        so a second store over the same index can only open after this)."""
        dense = getattr(self.retriever, "dense", None)
        if dense is not None:
            dense.close()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        date_from: dt.date | None = None,
        date_to: dt.date | None = None,
        entity_filter: bool | None = None,
    ) -> SearchResult:
        """Top notifications for ``query``: one hit per notification, best chunk shown.

        With the entity filter on, notifications addressed to another entity type (the
        NBFC, UCB, RRB, ... Directions) are skipped; the search goes deeper to compensate.
        """
        k = top_k or self.top_k
        drop_other = self.entity_filter if entity_filter is None else entity_filter
        filtered = date_from is not None or date_to is not None
        depth = k * self.depth_factor * (2 if filtered else 1) * (2 if drop_other else 1)
        excluded = 0
        retrieval = self.retriever.retrieve(query, top_k=depth)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for scored in retrieval.results:
            chunk = scored.chunk
            if chunk.doc_id in seen:
                continue
            meta = chunk.metadata
            doc = self.documents.get(chunk.doc_id)
            title = _as_str(meta.get("title")) or (doc.title if doc else None)
            if drop_other and is_other_entity(title, self.entity):
                seen.add(chunk.doc_id)
                excluded += 1
                continue
            date = _as_str(meta.get("date")) or (
                doc.date.isoformat() if doc is not None and doc.date else None
            )
            if filtered:
                if date is None:
                    continue
                day = dt.date.fromisoformat(date)
                if (date_from and day < date_from) or (date_to and day > date_to):
                    continue
            seen.add(chunk.doc_id)
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    doc_id=chunk.doc_id,
                    title=title,
                    date=date,
                    entity=entity_of(title),
                    circular_number=_as_str(meta.get("circular_number")),
                    department=(doc.department if doc else None),
                    url=_as_str(meta.get("url")) or (doc.url if doc else None),
                    section=chunk.section,
                    snippet=_snippet(chunk.text, self.snippet_chars),
                    score=round(float(scored.score), 4),
                    replaced_by=self.graph.replaced_by(chunk.doc_id),
                )
            )
            if len(hits) >= k:
                break
        return SearchResult(
            query=query,
            mode=retrieval.mode,
            hits=hits,
            confidence=(
                round(retrieval.confidence.probability, 4) if retrieval.confidence else None
            ),
            timings_ms=retrieval.timings_ms,
            excluded_other_entity=excluded,
        )

    # -- fetch ---------------------------------------------------------------------------
    def fetch(
        self, identifier: str, *, query: str | None = None, max_tokens: int | None = None
    ) -> NotificationView:
        """A notification's metadata plus the passages most relevant to ``query``."""
        doc = self.documents[self.resolve(identifier)]
        passages, truncated = select_passages(
            doc,
            query,
            max_tokens=max_tokens or self.fetch_max_tokens,
            max_chars=self.passage_chars,
        )
        cited_by: list[ReferenceOut] = []
        for ref in self.graph.cited_by(doc.doc_id)[:MAX_CITED_BY]:
            src = self.documents[ref.source]
            cited_by.append(
                ReferenceOut(
                    doc_id=src.doc_id,
                    date=src.date.isoformat() if src.date else None,
                    title=src.title,
                    relation=ref.relation,
                )
            )
        return NotificationView(
            doc_id=doc.doc_id,
            title=doc.title,
            date=doc.date.isoformat() if doc.date else None,
            entity=entity_of(doc.title),
            department=doc.department,
            circular_number=doc.circular_number,
            ref_number=doc.ref_number,
            url=doc.url,
            total_chars=len(doc.text),
            table_of_contents=table_of_contents(doc),
            passages=passages,
            truncated=truncated,
            cited_by=cited_by,
            replaced_by=self.graph.replaced_by(doc.doc_id),
        )

    # -- construction --------------------------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        graph: ReferenceGraph | None = None,
        documents: Mapping[str, Document] | None = None,
    ) -> RegulationStore:
        """Load the persisted index and documents exactly as the RAG API does.

        ``graph`` and ``documents`` can be passed in to share them between stores (the
        reference graph takes ~15 s to build on the full corpus)."""
        from rbirag.index.build import load_bundle
        from rbirag.index.embedder import build_embedder
        from rbirag.retrieval.pipeline import make_retriever
        from rbirag.retrieval.rerank import build_reranker

        if not settings.documents_path.exists():
            raise FileNotFoundError(
                f"no documents at {settings.documents_path}; run `rbirag ingest` or download "
                "the corpus release"
            )
        chunker = settings.compliance_chunker or settings.chunker
        index_dir = settings.index_dir / chunker
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError(f"no index at {index_dir}; run `rbirag index` first")
        mode = settings.compliance_search_mode or settings.retrieval_mode
        emb = None
        if mode != "bm25":
            emb = embedder or build_embedder(
                settings.embedder,
                model_name=settings.embedding_model,
                query_instruction=settings.embedding_query_instruction,
                batch_size=settings.embedding_batch_size,
                device=settings.device,
                fake_dim=settings.fake_embedding_dim,
            )
        bundle = load_bundle(
            index_dir,
            embedder=emb,
            collection=f"{settings.qdrant_collection}_{chunker}",
            qdrant_url=settings.qdrant_url,
            qdrant_path=None
            if settings.qdrant_url
            else (settings.qdrant_path or index_dir / "qdrant"),
        )
        rr = None
        if mode == "hybrid_rerank":
            rr = reranker or build_reranker(
                settings.reranker, model_name=settings.reranker_model, device=settings.device
            )
        retriever = make_retriever(
            settings, bundle.store, bm25=bundle.bm25, dense=bundle.dense, reranker=rr, mode=mode
        )
        if documents is None:
            documents = {d.doc_id: d for d in iter_documents_jsonl(settings.documents_path)}
        return cls(
            retriever,
            documents,
            graph=graph,
            top_k=settings.compliance_search_top_k,
            depth_factor=settings.compliance_search_depth_factor,
            snippet_chars=settings.compliance_snippet_chars,
            fetch_max_tokens=settings.compliance_fetch_max_tokens,
            passage_chars=settings.compliance_passage_chars,
            entity=settings.compliance_entity,
            entity_filter=settings.compliance_entity_filter,
        )
