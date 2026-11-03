"""In-memory request log (optionally mirrored to JSONL) and its aggregate summary."""

from __future__ import annotations

import json
import threading
from collections import Counter, deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

STAGES = ("retrieve", "rerank", "generate", "total")


class RequestLogEntry(BaseModel):
    request_id: str
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    endpoint: str
    status: int
    mode: str | None = None
    backend: str | None = None
    model: str | None = None
    refused: bool | None = None
    refusal_reason: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str | None = None


class StageStats(BaseModel):
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


class MetricsSummary(BaseModel):
    requests: int
    errors: int
    refusals: int
    refusal_reasons: dict[str, int]
    by_backend: dict[str, int]
    by_mode: dict[str, int]
    latency: dict[str, StageStats]
    input_tokens: int
    output_tokens: int
    tokens_per_answer: float | None
    cost_usd: float
    cached: int = 0
    since: str | None


def _stats(values: Sequence[float]) -> StageStats:
    arr = np.asarray(values, dtype=np.float64)
    return StageStats(
        count=int(arr.size),
        mean_ms=round(float(arr.mean()), 3),
        p50_ms=round(float(np.percentile(arr, 50)), 3),
        p95_ms=round(float(np.percentile(arr, 95)), 3),
    )


class RequestLog:
    def __init__(self, path: Path | None = None, maxlen: int = 10_000) -> None:
        self.path = path
        self._entries: deque[RequestLogEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, entry: RequestLogEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(entry.model_dump_json() + "\n")

    def entries(self) -> list[RequestLogEntry]:
        with self._lock:
            return list(self._entries)

    def summary(self) -> MetricsSummary:
        entries = self.entries()
        queries = [e for e in entries if e.endpoint.startswith("/query")]
        ok = [e for e in queries if e.error is None]
        fresh = [e for e in ok if not e.cached]  # cache hits skew latency and add no tokens
        latency: dict[str, StageStats] = {}
        for stage in STAGES:
            values = [e.timings_ms[stage] for e in fresh if stage in e.timings_ms]
            if values:
                latency[stage] = _stats(values)
        answered = [e for e in ok if e.model]
        tokens = [e.input_tokens + e.output_tokens for e in answered]
        reasons = Counter(e.refusal_reason for e in ok if e.refused and e.refusal_reason)
        return MetricsSummary(
            requests=len(queries),
            errors=sum(1 for e in queries if e.error is not None),
            refusals=sum(1 for e in ok if e.refused),
            refusal_reasons=dict(reasons),
            by_backend=dict(Counter(e.backend or "none" for e in ok)),
            by_mode=dict(Counter(e.mode or "none" for e in ok)),
            latency=latency,
            input_tokens=sum(e.input_tokens for e in fresh),
            output_tokens=sum(e.output_tokens for e in fresh),
            tokens_per_answer=round(sum(tokens) / len(tokens), 1) if tokens else None,
            cost_usd=round(sum(e.cost_usd for e in fresh), 8),
            cached=sum(1 for e in ok if e.cached),
            since=entries[0].ts if entries else None,
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
