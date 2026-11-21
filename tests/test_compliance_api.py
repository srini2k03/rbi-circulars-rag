from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rbirag.api.main import create_app
from rbirag.api.service import RAGService
from rbirag.compliance.agent import ComplianceAgent
from rbirag.compliance.judge import ClauseInput
from rbirag.compliance.llm import DEMO_MODEL, build_chat_model, final, tool_call
from rbirag.compliance.records import ClausePrediction
from rbirag.compliance.regulations import RegulationStore
from rbirag.compliance.service import ComplianceService, default_checker
from rbirag.config import Settings
from rbirag.index.embedder import FakeEmbedder
from rbirag.retrieval.rerank import FakeReranker
from tests.test_compliance_agent import (
    scripted_agent_model,
    store,  # noqa: F401  (module fixture)
)

POLICY = """# Sample Bank KYC Policy

## 1. Customer due diligence

1.1 Periodic updation of KYC shall be carried out at least once every three years for high
risk customers.

1.2 Records of transactions shall be kept for at least five years.

## 2. Administration

2.1 The Compliance Department shall hold its quarterly review in the first week of the
quarter.
"""


def _settings(tmp_path: Path, **kw: Any) -> Settings:
    return Settings(
        embedder="fake",
        reranker="fake",
        llm_backend="fake",
        index_dir=tmp_path / "idx",
        processed_dir=tmp_path / "processed",
        compliance_llm_cache_mode="off",
        compliance_api_preload=False,
        **kw,
    )


def _client(settings: Settings, comply: ComplianceService) -> TestClient:
    rag = RAGService(settings, embedder=FakeEmbedder(64), reranker=FakeReranker(), persist=False)
    return TestClient(create_app(settings, service=rag, compliance=comply))


def _wait(client: TestClient, job_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(f"/comply/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def demo_client(tmp_path: Path, store: RegulationStore) -> Iterator[TestClient]:  # noqa: F811
    settings = _settings(tmp_path, compliance_api_max_clauses=2)
    with _client(settings, ComplianceService(settings, store=store)) as c:
        yield c


def test_status_and_clauses(demo_client: TestClient) -> None:
    status = demo_client.get("/comply/status").json()
    assert status["loaded"] and status["notifications"] == 3
    assert status["backends"]["fake"] is True and status["max_clauses_per_check"] == 2
    r = demo_client.post("/comply/clauses", json={"name": "kyc.md", "text": POLICY})
    assert r.status_code == 200
    body = r.json()
    assert [c["label"] for c in body["clauses"]] == ["1.1", "1.2", "2.1"]
    assert body["clauses"][0]["heading_path"][-1] == "Customer due diligence"

    up = demo_client.post(
        "/comply/clauses/upload", files={"file": ("kyc.md", POLICY.encode(), "text/markdown")}
    )
    assert up.status_code == 200 and len(up.json()["clauses"]) == 3
    bad = demo_client.post(
        "/comply/clauses/upload", files={"file": ("kyc.docx", b"PK\x03\x04", "application/zip")}
    )
    assert bad.status_code == 422


def test_check_with_the_offline_demo_backend(demo_client: TestClient) -> None:
    r = demo_client.post(
        "/comply/check", json={"name": "kyc.md", "text": POLICY, "backend": "fake"}
    )
    assert r.status_code == 202, r.text
    job = _wait(demo_client, r.json()["job_id"])
    assert job["state"] == "succeeded", job.get("error")
    assert job["clauses_total"] == 2 and job["clauses_done"] == 2  # capped by the demo limit
    report = job["report"]
    assert report["model"] == DEMO_MODEL and report["summary"]["insufficient_evidence"] == 2
    tools = [c["name"] for c in report["findings"][0]["prediction"]["tool_calls"]]
    assert tools == ["search_regulations", "compare_clause"]
    md = demo_client.get(f"/comply/jobs/{job['job_id']}/report.md")
    assert md.status_code == 200 and md.text.startswith("# Compliance check: kyc.md")

    upload = demo_client.post(
        "/comply/check/upload",
        files={"file": ("kyc.md", POLICY.encode(), "text/markdown")},
        data={"system": "baseline", "backend": "fake", "max_clauses": "1"},
    )
    assert upload.status_code == 202
    done = _wait(demo_client, upload.json()["job_id"])
    assert done["state"] == "succeeded" and done["system"] == "baseline"
    assert done["clauses_total"] == 1


def test_check_errors(demo_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = demo_client.post("/comply/check", json={"text": POLICY, "backend": "gemini"})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]
    r = demo_client.post(
        "/comply/check/upload",
        files={"file": ("kyc.md", POLICY.encode(), "text/markdown")},
        data={"system": "oracle", "backend": "fake"},
    )
    assert r.status_code == 422
    assert demo_client.get("/comply/jobs/nope").status_code == 404
    assert demo_client.post("/comply/check", json={"text": "  \n\n  "}).status_code == 422


def test_check_reports_progress_and_full_findings(
    tmp_path: Path,
    store: RegulationStore,  # noqa: F811
) -> None:
    release = threading.Event()
    settings = _settings(tmp_path)

    class Gated:
        """The scripted agent, held back after the first clause until released."""

        def __init__(self) -> None:
            self.n = 0

        def check(self, clause: ClauseInput, *, item_id: str | None = None) -> ClausePrediction:
            self.n += 1
            if self.n == 2:
                release.wait(10)
            if "three years" in clause.text:
                chat = scripted_agent_model(
                    [
                        tool_call("search_regulations", {"query": "periodic updation KYC"}),
                        tool_call("compare_clause", {"doc_ids": ["rbi-900"]}, call_id="c2"),
                        final({"verdict": "conflict", "citations": ["rbi-900"], "confidence": 0.8}),
                    ]
                )
            else:
                chat = scripted_agent_model(
                    [final({"verdict": "not_applicable", "citations": [], "confidence": 0.9})]
                )
            return ComplianceAgent(store, chat, judge_context_tokens=600).check(
                clause, item_id=item_id
            )

    gated = Gated()
    comply = ComplianceService(settings, store=store, checker_factory=lambda *_: gated)
    with _client(settings, comply) as client:
        job_id = client.post(
            "/comply/check", json={"name": "kyc.md", "text": POLICY, "backend": "fake"}
        ).json()["job_id"]
        deadline = time.monotonic() + 10
        while client.get(f"/comply/jobs/{job_id}").json()["clauses_done"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        partial = client.get(f"/comply/jobs/{job_id}").json()
        assert partial["state"] == "running" and len(partial["findings"]) == 1
        assert client.get(f"/comply/jobs/{job_id}/report.md").status_code == 409
        release.set()
        job = _wait(client, job_id)
    assert job["state"] == "succeeded"
    first = job["report"]["findings"][0]
    assert first["prediction"]["verdict"] == "conflict"
    assert first["sources"][0]["doc_id"] == "rbi-900"
    assert first["sources"][0]["quotes"]  # the verified evidence quote is in the report
    assert job["report"]["summary"]["conflict"] == 1


def test_default_checker_and_demo_backend(tmp_path: Path, store: RegulationStore) -> None:  # noqa: F811
    settings = _settings(tmp_path)
    agent = default_checker("agent", "fake", settings, store)
    pred = agent.check(
        ClauseInput(label="3.2", text="Periodic updation of KYC every three years for high risk.")
    )
    assert pred.verdict == "insufficient_evidence" and pred.model == DEMO_MODEL
    assert [c.name for c in pred.tool_calls] == ["search_regulations", "compare_clause"]
    assert build_chat_model(settings, "fake").model == DEMO_MODEL


def test_service_reports_a_missing_corpus(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    comply = ComplianceService(settings)
    with _client(settings, comply) as client:
        r = client.post("/comply/check", json={"text": POLICY, "backend": "fake"})
        assert r.status_code == 503 and "no documents" in r.json()["detail"]
        status = client.get("/comply/status").json()
        assert not status["loaded"] and "no documents" in (status["error"] or "")
