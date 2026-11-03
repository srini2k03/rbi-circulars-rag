"""Local Ollama backend (``/api/chat``, streaming NDJSON) over httpx."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from rbirag.generation.llm import LLMResponse, Message, StreamChunk, TokenUsage, timed_ms


class OllamaError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        *,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        timeout_seconds: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.client = client or httpx.Client(timeout=timeout_seconds)

    @property
    def backend(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def payload(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m.role, "content": m.content} for m in messages)
        body: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "stream": True,
            # Greedy decoding keeps evaluation runs reproducible.
            "options": {
                "temperature": 0,
                "seed": 0,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if json_schema is not None:
            body["format"] = json_schema
        return body

    def _chunks(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self.client.stream("POST", f"{self.base_url}/api/chat", json=body) as resp:
            if resp.status_code != 200:
                resp.read()
                raise OllamaError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise OllamaError(str(data["error"]))
                yield data

    def _final(self, text: str, last: dict[str, Any], started: float) -> LLMResponse:
        return LLMResponse(
            text=text,
            model=self._model,
            backend=self.backend,
            usage=TokenUsage(
                input_tokens=int(last.get("prompt_eval_count", 0) or 0),
                output_tokens=int(last.get("eval_count", 0) or 0),
            ),
            stop_reason=str(last.get("done_reason") or "stop"),
            latency_ms=timed_ms(started),
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]:
        started = time.perf_counter()
        parts: list[str] = []
        last: dict[str, Any] = {}
        for data in self._chunks(self.payload(messages, system=system, max_tokens=max_tokens)):
            piece = (data.get("message") or {}).get("content", "")
            if piece:
                parts.append(piece)
                yield StreamChunk(text=piece)
            if data.get("done"):
                last = data
        yield StreamChunk(final=self._final("".join(parts), last, started))

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        parts: list[str] = []
        last: dict[str, Any] = {}
        body = self.payload(messages, system=system, max_tokens=max_tokens, json_schema=json_schema)
        for data in self._chunks(body):
            parts.append((data.get("message") or {}).get("content", ""))
            if data.get("done"):
                last = data
        return self._final("".join(parts), last, started)
