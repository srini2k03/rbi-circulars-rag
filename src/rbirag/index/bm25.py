"""BM25 sparse index (rank_bm25) with an RBI-aware tokenizer and versioned persistence."""

from __future__ import annotations

import pickle
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1

_STOPWORDS_TEXT = """
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves shall may also thereof
therein hereby herein vide viz ie eg etc
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_TEXT.split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Very light plural folding: ``banks -> bank``, ``policies -> policy``."""
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def bm25_tokenize(text: str, *, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Lower-case, NFKC, alphanumeric runs, stopword removal, light stemming."""
    text = unicodedata.normalize("NFKC", text).lower()
    return [_stem(t) for t in _TOKEN_RE.findall(text) if t not in stopwords]


class BM25Index:
    """Okapi BM25 over chunk texts. Persisted as tokens (the model is rebuilt on load)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: Any = None

    def __len__(self) -> int:
        return len(self.ids)

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        self.ids = list(ids)
        self._tokens = [bm25_tokenize(t) for t in texts]
        self._fit()
        return self

    def _fit(self) -> None:
        from rank_bm25 import BM25Okapi

        if not self._tokens:
            self._bm25 = None
            return
        # rank_bm25 divides by zero on an all-empty corpus; give empty docs a placeholder.
        corpus = [toks or ["\x00"] for toks in self._tokens]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-``k`` ``(chunk_id, score)`` with score > 0, best first (ties by id)."""
        if self._bm25 is None:
            return []
        q = bm25_tokenize(query)
        if not q:
            return []
        scores = np.asarray(self._bm25.get_scores(q), dtype=np.float64)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in order[:k] if scores[i] > 0]

    # -- persistence ---------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "params": {"k1": self.k1, "b": self.b, "epsilon": self.epsilon},
            "ids": self.ids,
            "tokens": self._tokens,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load an index written by :meth:`save`. Only load files you produced yourself."""
        with path.open("rb") as fh:
            payload = pickle.load(fh)  # trusted, locally produced artifact
        version = payload.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"BM25 index format {version} != supported {FORMAT_VERSION}")
        idx = cls(**payload["params"])
        idx.ids = list(payload["ids"])
        idx._tokens = [list(t) for t in payload["tokens"]]
        idx._fit()
        return idx
