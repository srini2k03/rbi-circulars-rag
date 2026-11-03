from __future__ import annotations

import json

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
