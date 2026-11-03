"""Small, testable client helpers used by the Streamlit UI (and handy in notebooks)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

_CITE = re.compile(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]")


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse ``event:``/``data:`` server-sent events into ``(event, json_data)`` pairs."""
    event = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, json.loads("\n".join(data))


def stream_query(
    base_url: str,
    payload: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    timeout: float = 300,
) -> Iterator[tuple[str, dict[str, Any]]]:
    own = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        with http.stream("POST", base_url.rstrip("/") + "/query/stream", json=payload) as resp:
            if resp.status_code != 200:
                resp.read()
                detail = (
                    resp.json().get("detail")
                    if resp.headers.get("content-type", "").startswith("application/json")
                    else resp.text
                )
                yield "error", {"detail": f"HTTP {resp.status_code}: {detail}"}
                return
            yield from iter_sse(resp.iter_lines())
    finally:
        if own:
            http.close()


def get_json(base_url: str, path: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    own = client is None
    http = client or httpx.Client(timeout=30)
    try:
        resp = http.get(base_url.rstrip("/") + path)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data
    finally:
        if own:
            http.close()


def linkify_citations(answer: str, valid: Iterable[int], anchor: str = "source") -> str:
    """Turn ``[1]`` / ``[1, 3]`` into Markdown links to ``#{anchor}-n`` anchors."""
    ok = set(valid)

    def repl(m: re.Match[str]) -> str:
        parts = re.split(r"\s*,\s*", m.group(1))
        out = []
        for p in parts:
            if p.isdigit() and int(p) in ok:
                out.append(f"[\\[{p}\\]](#{anchor}-{p})")
            else:
                out.append(f"[{p}]")
        return "".join(out)

    return _CITE.sub(repl, answer)


def source_caption(source: dict[str, Any]) -> str:
    bits = [str(source.get("title") or "Untitled")]
    for key in ("circular_number", "date"):
        if source.get(key):
            bits.append(str(source[key]))
    return " | ".join(bits)
