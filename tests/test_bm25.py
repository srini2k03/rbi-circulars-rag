from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from rbirag.index.bm25 import BM25Index, bm25_tokenize

CORPUS = {
    "kyc": "Periodic updation of KYC for high risk customers every two years.",
    "deposit": "Interest rates on bulk term deposits of small finance banks.",
    "forex": "Overseas investment reporting through the FIRMS portal by AD banks.",
    "cards": "Credit card and debit card issuance and conduct directions.",
}


@pytest.fixture
def index() -> BM25Index:
    return BM25Index().build(list(CORPUS), list(CORPUS.values()))


def test_tokenizer_normalises_and_drops_stopwords() -> None:
    assert bm25_tokenize("The Banks' POLICIES are for customers") == ["bank", "policy", "customer"]
    assert bm25_tokenize("\uff2b\uff39\uff23 norms") == ["kyc", "norm"]  # NFKC full-width


def test_bm25_ranks_the_obvious_document_first(index: BM25Index) -> None:
    assert index.search("KYC updation for high risk customer")[0][0] == "kyc"
    assert index.search("bulk deposit interest rate")[0][0] == "deposit"
    assert index.search("FIRMS overseas investment")[0][0] == "forex"


def test_bm25_scores_descending_and_positive(index: BM25Index) -> None:
    results = index.search("banks", k=10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
    assert index.search("zzz unknown") == []
    assert index.search("the of and") == []  # stopwords only


def test_bm25_k_limits_results(index: BM25Index) -> None:
    assert len(index.search("banks card kyc deposit", k=2)) == 2


def test_bm25_roundtrip(tmp_path: Path, index: BM25Index) -> None:
    path = tmp_path / "bm25.pkl"
    index.save(path)
    loaded = BM25Index.load(path)
    assert loaded.ids == index.ids
    assert loaded.search("credit card") == index.search("credit card")


def test_bm25_rejects_unknown_version(tmp_path: Path, index: BM25Index) -> None:
    path = tmp_path / "bm25.pkl"
    index.save(path)
    payload = pickle.loads(path.read_bytes())
    payload["format_version"] = 999
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="format 999"):
        BM25Index.load(path)


def test_bm25_empty_and_mismatch() -> None:
    assert BM25Index().build([], []).search("x") == []
    with pytest.raises(ValueError, match="same length"):
        BM25Index().build(["a"], [])
