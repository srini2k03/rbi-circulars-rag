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


# -- compliance page ----------------------------------------------------------------------

VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "conflict": ("red", "CONFLICT"),
    "gap": ("orange", "GAP"),
    "compliant": ("green", "COMPLIANT"),
    "insufficient_evidence": ("blue", "INSUFFICIENT EVIDENCE"),
    "not_applicable": ("gray", "NOT APPLICABLE"),
}
VERDICT_ORDER = ("conflict", "gap", "compliant", "insufficient_evidence", "not_applicable")


def verdict_badge(verdict: str) -> str:
    """Streamlit Markdown for a coloured verdict label."""
    colour, label = VERDICT_STYLE.get(verdict, ("gray", verdict.upper()))
    return f":{colour}[**{label}**]"


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    timeout: float = 60,
) -> tuple[int, dict[str, Any]]:
    """POST JSON; returns (status, body) without raising, so the UI can show API errors."""
    own = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        resp = http.post(base_url.rstrip("/") + path, json=payload)
        return resp.status_code, _body(resp)
    finally:
        if own:
            http.close()


def post_file(
    base_url: str,
    path: str,
    *,
    filename: str,
    content: bytes,
    data: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    timeout: float = 120,
) -> tuple[int, dict[str, Any]]:
    """POST a multipart upload (``file`` plus form fields); returns (status, body)."""
    own = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        resp = http.post(
            base_url.rstrip("/") + path,
            files={"file": (filename, content)},
            data={k: v for k, v in (data or {}).items() if v not in (None, "")},
        )
        return resp.status_code, _body(resp)
    finally:
        if own:
            http.close()


def _body(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {"detail": resp.text[:500]}
    return data if isinstance(data, dict) else {"data": data}


def findings_rows(findings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One table row per checked clause."""
    rows = []
    for f in findings:
        p = f["prediction"]
        conf = p.get("confidence")
        rows.append(
            {
                "clause": f["clause"].get("number") or f["clause"]["clause_id"],
                "verdict": VERDICT_STYLE.get(p["verdict"], ("", p["verdict"]))[1],
                "confidence": None if conf is None else round(float(conf), 2),
                "sources": ", ".join(s["doc_id"] for s in f.get("sources", [])) or "-",
                "steps": len(p.get("tool_calls", [])),
                "error": p.get("error") or "",
            }
        )
    return rows


def verdict_counts(findings: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(VERDICT_ORDER, 0)
    for f in findings:
        v = f["prediction"]["verdict"]
        counts[v] = counts.get(v, 0) + 1
    return counts
