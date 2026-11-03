"""API tests: FakeLLM, FakeEmbedder/FakeReranker and in-memory Qdrant; no network."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from rbirag.api.logging import JSONFormatter, request_id_var
from rbirag.api.main import create_app
from rbirag.api.metrics import RequestLog, RequestLogEntry
from rbirag.api.service import RAGService
from rbirag.config import Settings
from rbirag.document import Document, write_documents_jsonl
from rbirag.generation.llm import LLM, FakeLLM
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.rerank import FakeReranker

Q = "How often is periodic updation of KYC for high risk customers?"


def fake_llm_factory(name: str, settings: Settings) -> LLM:
    if name == "groq":
        raise RuntimeError("GROQ_API_KEY is not set")
    return FakeLLM()


def parse_sse(text: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedder="fake", reranker="fake", llm_backend="fake", chunker="section",
        chunk_size=128, chunk_overlap=16, section_min_tokens=24, index_dir=tmp_path / "idx",
        processed_dir=tmp_path / "processed",
    )  # fmt: skip


@pytest.fixture
def service(settings: Settings) -> RAGService:
    return RAGService(
        settings, embedder=FakeEmbedder(64), reranker=FakeReranker(),
        llm_factory=fake_llm_factory, persist=False,
        url_fetcher=lambda urls: ([], list(urls)),
    )  # fmt: skip


@pytest.fixture
def log() -> RequestLog:
    return RequestLog()


@pytest.fixture
def client(settings: Settings, service: RAGService, log: RequestLog) -> Iterator[TestClient]:
    with TestClient(create_app(settings, service=service, request_log=log)) as c:
        yield c


@pytest.fixture
def ingested(client: TestClient, fixture_docs: list[Document]) -> TestClient:
    r = client.post(
        "/ingest", json={"documents": [json.loads(d.model_dump_json()) for d in fixture_docs]}
    )
    assert r.status_code == 202
    job = client.get(f"/ingest/{r.json()['job_id']}").json()
    assert job["state"] == "succeeded", job
    assert job["documents_added"] == 3
    assert job["chunks"] > 0
    return client


def test_health_and_query_before_ingest(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "degraded"
    assert h["index_loaded"] is False
    assert h["backends"]["fake"] is True
    r = client.post("/query", json={"question": Q})
    assert r.status_code == 503
    assert "POST /ingest" in r.json()["detail"]


def test_ingest_then_health(ingested: TestClient) -> None:
    h = ingested.get("/health").json()
    assert h["status"] == "ok"
    assert h["documents"] == 3
    assert h["chunks"] > 0
    assert h["dense_index"] is True
    assert h["default_backend"] == "fake"


def test_query_returns_cited_answer(ingested: TestClient) -> None:
    r = ingested.post(
        "/query", json={"question": Q, "top_k": 3}, headers={"x-request-id": "req-123"}
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-123"
    body = r.json()
    assert body["request_id"] == "req-123"
    assert body["refused"] is False
    assert body["backend"] == "fake"
    assert len(body["sources"]) == 3
    cit = body["citations"][0]
    assert cit["n"] == 1
    assert cit["doc_id"] == "rbi-12655"
    assert cit["circular_number"] == "RBI/2025-26/41"
    assert cit["date"] == "2025-06-12"
    assert cit["url"] == "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12655&Mode=0"
    assert cit["cited"] is True
    assert set(body["timings_ms"]) == {"retrieve", "rerank", "generate", "total"}
    assert body["usage"]["input_tokens"] > 0
    assert body["cost"] == {"usd": 0.0, "inr": 0.0}
    assert 0 <= body["confidence"] <= 1


def test_query_gate_refuses_with_reason(ingested: TestClient) -> None:
    body = ingested.post("/query", json={"question": "monsoon cricket score in Chennai"}).json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "low_confidence"
    assert body["model"] is None
    assert body["citations"] == []
    # threshold 0 lets it through to the LLM
    body = ingested.post(
        "/query", json={"question": "monsoon cricket score", "threshold": 0.0}
    ).json()
    assert body["refusal_reason"] != "low_confidence"


def test_query_stream_sse(ingested: TestClient) -> None:
    with ingested.stream("POST", "/query/stream", json={"question": Q, "top_k": 3}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = parse_sse(text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources"
    assert kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"token"}
    assert len(events[0][1]["sources"]) == 3
    done = events[-1][1]
    assert "".join(d["text"] for k, d in events if k == "token").strip() == done["answer"]
    assert done["citations"][0]["doc_id"] == "rbi-12655"
    assert done["request_id"] == events[0][1]["request_id"]


def test_unconfigured_backend_is_503(ingested: TestClient) -> None:
    r = ingested.post("/query", json={"question": Q, "backend": "groq"})
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]
    r = ingested.post("/query/stream", json={"question": Q, "backend": "groq"})
    assert r.status_code == 503


def test_metrics_aggregate(ingested: TestClient, log: RequestLog) -> None:
    ingested.post("/query", json={"question": Q})
    ingested.post("/query", json={"question": "monsoon cricket score in Chennai"})
    ingested.post("/query", json={"question": Q, "backend": "groq"})
    m = ingested.get("/metrics").json()
    assert m["requests"] == 3
    assert m["errors"] == 1
    assert m["refusals"] == 1
    assert m["refusal_reasons"] == {"low_confidence": 1}
    assert set(m["latency"]) >= {"retrieve", "rerank", "total", "generate"}
    assert m["latency"]["generate"]["count"] == 1
    assert m["input_tokens"] > 0
    assert m["cost_usd"] == 0.0
    assert len(log.entries()) == 3


def test_ingest_urls_and_validation(
    client: TestClient, service: RAGService, fixture_docs: list[Document]
) -> None:
    service.url_fetcher = lambda urls: ([fixture_docs[0]], [u for u in urls if "bad" in u])
    r = client.post(
        "/ingest", json={"urls": ["https://www.rbi.org.in/x?Id=12701", "https://bad.example/y"]}
    )
    job = client.get(f"/ingest/{r.json()['job_id']}").json()
    assert job["state"] == "succeeded"
    assert job["failed_urls"] == ["https://bad.example/y"]
    assert job["documents_total"] == 1
    assert client.post("/ingest", json={}).status_code == 422
    assert client.get("/ingest/nope").status_code == 404
    service.url_fetcher = lambda urls: ([], list(urls))
    r = client.post("/ingest", json={"urls": ["https://bad.example/z"], "replace": True})
    job = client.get(f"/ingest/{r.json()['job_id']}").json()
    assert job["state"] == "failed"
    assert "no documents" in job["error"]


def test_ingest_from_server_path(
    client: TestClient, fixture_docs: list[Document], tmp_path: Path
) -> None:
    path = tmp_path / "docs.jsonl"
    write_documents_jsonl(fixture_docs[:2], path)
    r = client.post("/ingest", json={"documents_path": str(path)})
    assert client.get(f"/ingest/{r.json()['job_id']}").json()["documents_added"] == 2


def test_persisted_index_reloads(settings: Settings, fixture_docs: list[Document]) -> None:
    svc = RAGService(
        settings, embedder=FakeEmbedder(64), reranker=FakeReranker(), llm_factory=fake_llm_factory
    )
    n = svc.rebuild(fixture_docs)
    svc.close()
    again = RAGService(
        settings, embedder=FakeEmbedder(64), reranker=FakeReranker(), llm_factory=fake_llm_factory
    )
    assert again.load()
    assert again.health().chunks == n
    assert again.health().documents == 3
    again.close()


def test_openapi_has_examples(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    assert schemas["QueryRequest"]["examples"][0]["question"]
    assert schemas["QueryResponse"]["examples"][0]["citations"][0]["circular_number"]
    assert "/query/stream" in spec["paths"]


def test_json_log_formatter_includes_request_id() -> None:
    token = request_id_var.set("abc")
    try:
        record = logging.makeLogRecord(
            {"msg": "query", "levelname": "INFO", "name": "x", "mode": "bm25"}
        )
        out = json.loads(JSONFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert out["request_id"] == "abc"
    assert out["message"] == "query"
    assert out["mode"] == "bm25"


def test_request_log_mirrors_to_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = RequestLog(path)
    for i, ms in enumerate([10.0, 20.0, 30.0, 40.0]):
        log.add(RequestLogEntry(request_id=str(i), endpoint="/query", status=200, model="m",
                                timings_ms={"total": ms}, input_tokens=10, output_tokens=5))  # fmt: skip
    s = log.summary()
    assert s.latency["total"].mean_ms == 25.0
    assert s.latency["total"].p50_ms == 25.0
    assert s.tokens_per_answer == 15.0
    assert len(path.read_text().splitlines()) == 4


def test_metrics_exclude_cached_from_latency_and_tokens() -> None:
    log = RequestLog()
    log.add(RequestLogEntry(request_id="1", endpoint="/query", status=200, model="m",
                            timings_ms={"total": 100.0}, input_tokens=10, output_tokens=5))  # fmt: skip
    log.add(RequestLogEntry(request_id="2", endpoint="/query", status=200, model="m", cached=True,
                            timings_ms={"total": 1.0}, input_tokens=10, output_tokens=5))  # fmt: skip
    s = log.summary()
    assert s.latency["total"].count == 1
    assert s.latency["total"].mean_ms == 100.0
    assert s.input_tokens == 10
    assert s.cached == 1
