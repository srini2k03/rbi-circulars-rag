"""The /v1 API: ask, retrieve, verify, compare, eval (guarded), documents, ops."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from rbirag.api.main import create_app
from rbirag.api.metrics import RequestLog
from rbirag.api.ratelimit import RateLimiter
from rbirag.api.service import RAGService
from rbirag.config import Settings
from rbirag.document import Document
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.rerank import FakeReranker
from tests.test_api import Q, fake_llm_factory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedder="fake", reranker="fake", llm_backend="fake", chunker="section",
        chunk_size=128, chunk_overlap=16, section_min_tokens=24, index_dir=tmp_path / "idx",
        processed_dir=tmp_path / "processed", api_rate_limit_per_minute=1000,
        api_eval_token="s3cret", eval_dir=Path(__file__).resolve().parents[1] / "data" / "eval",
    )  # fmt: skip


@pytest.fixture
def log() -> RequestLog:
    return RequestLog()


@pytest.fixture
def client(
    settings: Settings, log: RequestLog, fixture_docs: list[Document]
) -> Iterator[TestClient]:
    from rbirag.intel.cli import save_graph
    from rbirag.intel.extract import build_intel

    intel, relations = build_intel(fixture_docs)
    save_graph(intel, relations, settings.processed_dir / "graph.json.gz")
    service = RAGService(
        settings, embedder=FakeEmbedder(64), reranker=FakeReranker(),
        llm_factory=fake_llm_factory, persist=False, url_fetcher=lambda urls: ([], list(urls)),
    )  # fmt: skip
    with TestClient(create_app(settings, service=service, request_log=log)) as c:
        r = c.post("/ingest", json={"documents": [json.loads(d.model_dump_json())
                                                  for d in fixture_docs]})  # fmt: skip
        assert c.get(f"/ingest/{r.json()['job_id']}").json()["state"] == "succeeded"
        yield c


def test_ask_returns_verified_citations(client: TestClient, log: RequestLog) -> None:
    r = client.post("/v1/ask", json={"question": Q, "backend": "fake"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert not body["refused"] and body["citations"]
    assert body["citations"][0]["doc_id"] == "rbi-12655"
    assert body["verification"]["verdict"] in {"supported", "partially_supported", "unsupported"}
    assert body["query"]["qtype"] in {"numeric", "factual", "temporal"}
    assert body["confidence"]["definition"].startswith("sigmoid")
    assert {"retrieve", "generate", "verify", "total"} <= set(body["timings_ms"])
    assert body["context"]["distinct_documents"] >= 1
    p = body["passages"][0]
    assert "rerank" in p["stage_scores"] and {"bm25", "rerank"} <= set(p["stage_ranks"])
    summary = client.get("/metrics").json()
    assert summary["by_endpoint"]["/v1/ask"] == 1 and summary["verdicts"]
    assert summary["failed_retrieval_rate"] == 0.0


def test_ask_refuses_out_of_scope_and_vague(client: TestClient) -> None:
    r = client.post("/v1/ask", json={"question": "What are SEBI's rules for IPO lock-in?"})
    assert r.json()["refused"] and r.json()["refusal_reason"] == "out_of_scope"
    r = client.post("/v1/ask", json={"question": "What is the limit?"})
    assert r.json()["refusal_reason"] == "needs_clarification"
    r = client.post("/v1/ask", json={"question": "And for low risk?", "history": [Q],
                                     "backend": "fake"})  # fmt: skip
    body = r.json()
    assert body["query"]["follow_up"] and "follow-up to" in body["query"]["rewritten"]


def test_ask_with_strategy_filters_and_validation(client: TestClient) -> None:
    r = client.post("/v1/ask", json={"question": Q, "backend": "fake",
                                     "strategy": {"mode": "bm25_rerank", "rerank_depth": 5},
                                     "filters": {"doc_ids": ["rbi-12701"]}})  # fmt: skip
    assert r.status_code == 200
    assert {p["doc_id"] for p in r.json()["passages"]} <= {"rbi-12701"}
    assert client.post("/v1/ask", json={"question": "x"}).status_code == 422
    assert client.post("/v1/ask", json={"question": Q, "nope": 1}).status_code == 422
    bad = client.post("/v1/ask", json={"question": Q, "strategy": {"rerank_weight": 3}})
    assert bad.status_code == 422


def test_retrieve_compare_and_verify(client: TestClient) -> None:
    r = client.post("/v1/retrieve", json={"question": Q, "top_k": 3,
                                          "strategy": {"mode": "bm25", "expansion": "rm3"}})  # fmt: skip
    body = r.json()
    assert r.status_code == 200 and len(body["passages"]) == 3
    assert body["retrieval"]["expansion"] and body["passages"][0]["stage_scores"]["bm25"] > 0
    again = client.post("/v1/retrieve", json={"question": Q, "top_k": 3,
                                              "strategy": {"mode": "bm25", "expansion": "rm3"}})  # fmt: skip
    assert again.json()["passages"] == body["passages"]  # served from the cache
    cmp = client.post(
        "/v1/retrieval/compare",
        json={
            "question": Q,
            "strategies": [
                {"mode": "bm25"},
                {"mode": "hybrid", "fusion": "weighted"},
                {"mode": "dense"},
            ],
        },
    )
    out = cmp.json()
    assert cmp.status_code == 200 and len(out["results"]) == 3
    assert out["overlap"][0][0] == 1.0
    chunk = body["passages"][0]["chunk_id"]
    v = client.post(
        "/v1/citations/verify",
        json={
            "answer": "Periodic updation of KYC is done every two years [1]. Rs. 99 crore [7].",
            "sources": [{"n": 1, "chunk_id": chunk}],
        },
    )
    rep = v.json()
    assert v.status_code == 200 and rep["invalid_citations"] == [7]
    assert (
        client.post(
            "/v1/citations/verify",
            json={"answer": "x [1]", "sources": [{"n": 1, "chunk_id": "missing"}]},
        ).status_code
        == 422
    )


def test_eval_run_is_guarded(client: TestClient) -> None:
    payload = {"strategies": [{"mode": "bm25"}], "limit": 5}
    assert client.post("/v1/eval/run", json=payload).status_code == 401
    r = client.post("/v1/eval/run", json=payload, headers={"X-Eval-Token": "wrong"})
    assert r.status_code == 401
    r = client.post("/v1/eval/run", json=payload, headers={"X-Eval-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["rows"][0]["strategy"].startswith("bm25")


def test_documents_relations_ready_and_limits(client: TestClient) -> None:
    r = client.get("/v1/documents/rbi-12655")
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["metadata"]["doc_id"] == "rbi-12655"
    assert "not a verified statement" in doc["latest_in_corpus"]["disclaimer"]
    assert client.get("/v1/documents/nope").status_code == 404
    rel = client.get("/v1/documents/rbi-12655/relations").json()
    assert set(rel) == {"doc_id", "outgoing", "incoming", "unresolved"}
    ready = client.get("/v1/ready").json()
    assert ready["ready"] and ready["relation_graph"]
    assert client.get("/v1/health").json()["index_loaded"]
    big = client.post("/v1/ask", content=b"{" + b" " * 70_000 + b"}",
                      headers={"content-type": "application/json"})  # fmt: skip
    assert big.status_code == 413
    spec = client.get("/openapi.json").json()
    assert "/v1/ask" in spec["paths"] and "/v1/documents/{doc_id}" in spec["paths"]


def test_rate_limiter() -> None:
    lim = RateLimiter(2)
    assert lim.check("a", now=0.0) == 0.0 and lim.check("a", now=0.0) == 0.0
    assert lim.check("a", now=0.0) > 0  # bucket empty
    assert lim.check("b", now=0.0) == 0.0  # per client
    assert lim.check("a", now=31.0) == 0.0  # refilled
    assert not RateLimiter(0).enabled and RateLimiter(0).check("a") == 0.0


def test_rate_limit_returns_429(tmp_path: Path) -> None:
    s = Settings(embedder="fake", reranker="fake", llm_backend="fake",
                 index_dir=tmp_path / "i", processed_dir=tmp_path / "p",
                 api_rate_limit_per_minute=1)  # fmt: skip
    service = RAGService(s, embedder=FakeEmbedder(16), reranker=FakeReranker(),
                         llm_factory=fake_llm_factory, persist=False)  # fmt: skip
    with TestClient(create_app(s, service=service)) as c:
        first = c.post("/v1/retrieve", json={"question": Q})
        assert first.status_code == 503  # no index, but it counted
        second = c.post("/v1/retrieve", json={"question": Q})
        assert second.status_code == 429 and "Retry-After" in second.headers
        assert c.get("/v1/ready").status_code == 503
