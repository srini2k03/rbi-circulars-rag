"""BM25 sparse index (rank_bm25) with an RBI-aware tokenizer and versioned persistence."""

from __future__ import annotations

import pickle
import re
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT_VERSION = 1

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

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
        self._post: dict[str, tuple[IntArray, FloatArray]] | None = None
        self._doc_len: FloatArray = np.zeros(0, dtype=np.float64)
        self._pos: dict[str, int] | None = None

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

        self._post, self._pos = None, None
        if not self._tokens:
            self._bm25 = None
            return
        # rank_bm25 divides by zero on an all-empty corpus; give empty docs a placeholder.
        corpus = [toks or ["\x00"] for toks in self._tokens]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def search(
        self, query: str, k: int = 10, *, allowed: BoolArray | None = None
    ) -> list[tuple[str, float]]:
        """Top-``k`` ``(chunk_id, score)`` with score > 0, best first (ties by id).

        ``allowed`` (a boolean mask over the indexed chunks) restricts the search, e.g. to
        the chunks that pass a metadata filter; masked-out chunks are never returned.
        """
        return self.search_terms([(t, 1.0) for t in bm25_tokenize(query)], k, allowed=allowed)

    def search_terms(
        self,
        terms: Sequence[tuple[str, float]],
        k: int = 10,
        *,
        allowed: BoolArray | None = None,
    ) -> list[tuple[str, float]]:
        """Top-``k`` for a weighted query: ``score(d) = sum_t w_t * bm25(t, d)``.

        With unit weights this is exactly ``BM25Okapi.get_scores`` (same IDF, same
        per-term arithmetic, same summation order); weights let query expansion (RM3,
        acronyms) down-weight added terms.
        """
        if self._bm25 is None or not terms:
            return []
        scores = self.score_terms(terms)
        if allowed is not None:
            if allowed.shape != scores.shape:
                raise ValueError("allowed mask must have one entry per indexed chunk")
            scores = np.where(allowed, scores, 0.0)
        return self._top(scores, k)

    def _top(self, scores: FloatArray, k: int) -> list[tuple[str, float]]:
        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []
        order = sorted(positive.tolist(), key=lambda i: (-scores[i], self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in order[:k]]

    def score_terms(self, terms: Sequence[tuple[str, float]]) -> FloatArray:
        """BM25 scores of every chunk for weighted ``(term, weight)`` pairs (postings-based,
        so cost grows with the matching postings, not with the corpus size per term)."""
        scores = np.zeros(len(self.ids), dtype=np.float64)
        if self._bm25 is None:
            return scores
        postings = self._postings()
        k1, b = self.k1, self.b
        for term, weight in terms:
            hit = postings.get(term)
            idf = self._bm25.idf.get(term) or 0.0
            if hit is None or not idf or not weight:
                continue
            idx, tf = hit
            denom = tf + k1 * (1 - b + b * self._doc_len[idx] / self._bm25.avgdl)
            scores[idx] += weight * (idf * (tf * (k1 + 1) / denom))
        return scores

    def _postings(self) -> dict[str, tuple[IntArray, FloatArray]]:
        if self._post is None:
            acc: dict[str, tuple[list[int], list[float]]] = {}
            for i, freqs in enumerate(self._bm25.doc_freqs):
                for term, tf in freqs.items():
                    entry = acc.setdefault(term, ([], []))
                    entry[0].append(i)
                    entry[1].append(float(tf))
            self._post = {
                t: (np.asarray(ix, dtype=np.int64), np.asarray(tf, dtype=np.float64))
                for t, (ix, tf) in acc.items()
            }
            self._doc_len = np.asarray(self._bm25.doc_len, dtype=np.float64)
        return self._post

    def warm(self) -> None:
        """Build the postings now (otherwise the first query pays for it)."""
        if self._bm25 is not None:
            self._postings()

    # -- statistics used by query expansion ----------------------------------------------
    def tokens_of(self, chunk_id: str) -> list[str]:
        """The indexed tokens of one chunk (for pseudo-relevance feedback)."""
        return self._tokens[self._position()[chunk_id]]

    def position(self, chunk_id: str) -> int:
        return self._position()[chunk_id]

    def _position(self) -> dict[str, int]:
        if self._pos is None:
            self._pos = {cid: i for i, cid in enumerate(self.ids)}
        return self._pos

    def idf(self, term: str) -> float:
        return float(self._bm25.idf.get(term) or 0.0) if self._bm25 is not None else 0.0

    def mask(self, keep: Callable[[str], bool]) -> BoolArray:
        """Boolean mask over the indexed chunks for which ``keep(chunk_id)`` is true."""
        return np.fromiter((keep(cid) for cid in self.ids), dtype=bool, count=len(self.ids))

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
