"""Model-score caches that let the retrieval benchmark run through the *real*
:class:`~rbirag.retrieval.pipeline.Retriever` without re-running the models.

The embedder and the cross-encoder only run inside CI (they are downloaded from Hugging
Face). The benchmark workflow records every dense top-k list and every cross-encoder
score it computes into a cache that is committed with the results; re-running the
benchmark anywhere else replays those scores, so the strategy comparison (fusion
weights, expansion, rerank depth, thresholds) is reproducible on a laptop with no
models and no network. A cache miss falls back to the live model when one is given, and
raises otherwise, so a replay can never silently use different scores.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rbirag.retrieval.params import RetrievalFilter
from rbirag.retrieval.pipeline import DenseSearcher
from rbirag.retrieval.rerank import Reranker

DENSE_DEPTH = 100  # dense candidates recorded per query (every strategy uses <= this)
DENSE_DIGITS = 6  # cosine similarities are stored (and used) rounded to this many digits
RERANK_DIGITS = 5  # cross-encoder logits likewise


class CacheMissError(LookupError):
    """A score the cache does not hold and no live model can compute."""


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class ScoreCache:
    """``query -> dense top-k`` and ``(query, passage) -> cross-encoder logit``."""

    def __init__(self) -> None:
        self.dense: dict[str, list[tuple[str, float]]] = {}
        self.rerank: dict[str, dict[str, float]] = {}
        self.misses = 0

    # -- persistence ---------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        queries = sorted(set(self.dense) | set(self.rerank))
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for q in queries:
                row: dict[str, Any] = {"q": q}
                if q in self.dense:
                    row["dense"] = [[c, s] for c, s in self.dense[q]]
                if q in self.rerank:
                    row["rerank"] = dict(sorted(self.rerank[q].items()))
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> ScoreCache:
        cache = cls()
        if not path.exists():
            return cache
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "dense" in row:
                    cache.dense[row["q"]] = [(c, float(s)) for c, s in row["dense"]]
                if "rerank" in row:
                    cache.rerank[row["q"]] = {k: float(v) for k, v in row["rerank"].items()}
        return cache


class CachedDense:
    """Dense search answered from the cache (filling it from ``live`` on a miss)."""

    def __init__(self, cache: ScoreCache, live: DenseSearcher | None = None) -> None:
        self.cache = cache
        self.live = live

    def search(
        self, query: str, k: int = 10, *, flt: RetrievalFilter | None = None
    ) -> list[tuple[str, float]]:
        if flt is not None and not flt.is_empty:
            if self.live is None:
                raise CacheMissError("filtered dense search needs the live index")
            return self.live.search(query, k, flt=flt)
        if k > DENSE_DEPTH:
            raise ValueError(f"the cache holds the dense top-{DENSE_DEPTH} only")
        hit = self.cache.dense.get(query)
        if hit is None:
            if self.live is None:
                raise CacheMissError(f"no cached dense results for query {query[:60]!r}")
            self.cache.misses += 1
            # rounded exactly as saved, so a live run and its replay rank identically
            hit = [(c, round(s, DENSE_DIGITS)) for c, s in self.live.search(query, DENSE_DEPTH)]
            self.cache.dense[query] = hit
        return hit[:k]


class CachedReranker:
    """Cross-encoder scores from the cache (computing and storing misses with ``live``)."""

    def __init__(self, cache: ScoreCache, live: Reranker | None = None) -> None:
        self.cache = cache
        self.live = live
        self.name = live.name if live is not None else "cached-cross-encoder"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        known = self.cache.rerank.setdefault(query, {})
        keys = [text_key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in known]
        if missing:
            if self.live is None:
                raise CacheMissError(f"{len(missing)} reranker scores missing for {query[:60]!r}")
            self.cache.misses += len(missing)
            unique = dict(missing)
            scores = self.live.score(query, list(unique.values()))
            known.update(
                (k, round(s, RERANK_DIGITS)) for k, s in zip(unique.keys(), scores, strict=True)
            )
        return [known[k] for k in keys]
