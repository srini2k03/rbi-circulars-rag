from __future__ import annotations

import json
from typing import Any

import httpx

from rbirag.ui_client import get_json, iter_sse, linkify_citations, source_caption, stream_query


def test_iter_sse_parses_events_and_multiline_data() -> None:
    lines = [
        "event: sources", 'data: {"sources": []}', "",
        ": keep-alive", "",
        "event: token", 'data: {"text": "Hi"}', "",
        'data: {"a":', "data: 1}", "",
    ]  # fmt: skip
    assert list(iter_sse(lines)) == [
        ("sources", {"sources": []}),
        ("token", {"text": "Hi"}),
        ("message", {"a": 1}),
    ]


def test_stream_query_and_errors() -> None:
    body = 'event: token\ndata: {"text": "ok"}\n\nevent: done\ndata: {"answer": "ok"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["question"] == "bad":
            return httpx.Response(503, json={"detail": "no index loaded"})
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert list(stream_query("http://api", {"question": "q"}, client=client)) == [
        ("token", {"text": "ok"}),
        ("done", {"answer": "ok"}),
    ]
    assert list(stream_query("http://api/", {"question": "bad"}, client=client)) == [
        ("error", {"detail": "HTTP 503: no index loaded"})
    ]
    health = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok"}))
    )
    assert get_json("http://api", "/health", client=health) == {"status": "ok"}


def test_linkify_citations() -> None:
    out = linkify_citations("Rule applies [1]. Also [2, 3]. Bad [9].", [1, 2, 3])
    assert out == (
        "Rule applies [\\[1\\]](#source-1). Also [\\[2\\]](#source-2)[\\[3\\]](#source-3). Bad [9]."
    )
    assert source_caption({"title": "T", "date": "2025-01-01", "circular_number": "RBI/1"}) == (
        "T | RBI/1 | 2025-01-01"
    )


def test_compliance_helpers() -> None:
    from rbirag.ui_client import findings_rows, post_file, post_json, verdict_badge, verdict_counts

    assert verdict_badge("conflict") == ":red[**CONFLICT**]"
    assert verdict_badge("weird") == ":gray[**WEIRD**]"
    findings = [
        {
            "clause": {"number": "1.1", "clause_id": "c001"},
            "prediction": {
                "verdict": "conflict",
                "confidence": 0.8123,
                "tool_calls": [{}, {}],
                "error": None,
            },
            "sources": [{"doc_id": "rbi-900"}],
        },
        {
            "clause": {"number": None, "clause_id": "c002"},
            "prediction": {"verdict": "not_applicable", "confidence": None, "tool_calls": []},
        },
    ]
    rows = findings_rows(findings)
    assert rows[0] == {
        "clause": "1.1",
        "verdict": "CONFLICT",
        "confidence": 0.81,
        "sources": "rbi-900",
        "steps": 2,
        "error": "",
    }
    assert rows[1]["clause"] == "c002" and rows[1]["sources"] == "-"
    counts = verdict_counts(findings)
    assert counts["conflict"] == 1 and counts["not_applicable"] == 1 and counts["gap"] == 0

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["type"] = request.headers.get("content-type", "")
        if request.url.path == "/bad":
            return httpx.Response(500, text="boom")
        return httpx.Response(202, json={"job_id": "j1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert post_json("http://api/", "/comply/check", {"text": "x"}, client=client) == (
        202,
        {"job_id": "j1"},
    )
    status, _ = post_file(
        "http://api",
        "/comply/check/upload",
        filename="p.md",
        content=b"1. x",
        data={"system": "agent", "backend": ""},
        client=client,
    )
    assert status == 202 and seen["type"].startswith("multipart/form-data")
    assert post_json("http://api", "/bad", {}, client=client) == (500, {"detail": "boom"})
