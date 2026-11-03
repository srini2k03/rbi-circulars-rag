"""Rerankers: protocol, cross-encoder (lazy), and a deterministic fake for tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from rbirag.index.bm25 import bm25_tokenize


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance logits (higher is better), one per text."""
        ...


class CrossEncoderReranker:
    """``cross-encoder/ms-marco-MiniLM-L-6-v2`` via sentence-transformers. Outputs logits."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name, device=self.device, max_length=self.max_length)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


class FakeReranker:
    """Query-term coverage mapped to a logit-like range ``[-5, 5]``. Deterministic."""

    name = "fake-overlap"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(bm25_tokenize(query))
        out: list[float] = []
        for t in texts:
            if not q:
                out.append(-5.0)
                continue
            coverage = len(q & set(bm25_tokenize(t))) / len(q)
            out.append(round(10.0 * coverage - 5.0, 6))
        return out


def build_reranker(
    backend: str, *, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str | None = None,
) -> Reranker | None:  # fmt: skip
    if backend == "none":
        return None
    if backend == "fake":
        return FakeReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker(model_name, device=device)
    raise ValueError(f"unknown reranker backend: {backend!r}")
