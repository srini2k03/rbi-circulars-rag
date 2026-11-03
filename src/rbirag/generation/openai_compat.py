"""OpenAI-compatible chat-completions backend over httpx (streaming SSE).

Defaults target Google Gemini's OpenAI-compatible endpoint (free tier); Groq, OpenRouter
or any other compatible server works by changing the base URL, model and the *name* of
the environment variable holding the key. The key is read from the environment at
construction, sent only in the Authorization header, and never logged.

Free-tier hygiene: a client-side :class:`RequestLimiter` (RPM + persisted RPD), and
exponential backoff on 429/5xx honouring ``Retry-After``. A 429 that names a *daily* quota
raises :class:`QuotaExhaustedError` immediately instead of retrying.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx

from rbirag.generation.limits import RequestLimiter
from rbirag.generation.llm import (
    LLMError,
    LLMResponse,
    Message,
    QuotaExhaustedError,
    StreamChunk,
    TokenUsage,
    timed_ms,
)

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
RETRYABLE = frozenset({429, 500, 502, 503, 504})
_DAILY_MARKERS = ("perday", "per day", "per_day", "daily")


class MissingAPIKeyError(RuntimeError):
    pass


def read_api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"{env_var} is not set or is empty. Export it locally, or add a repository "
            f"secret named {env_var} for GitHub Actions."
        )
    return key


def _is_daily_quota(body: str) -> bool:
    low = body.lower().replace(" ", "")
    return any(m.replace(" ", "") in low for m in _DAILY_MARKERS)


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = GEMINI_BASE_URL,
        api_key_env: str = "GEMINI_API_KEY",
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        json_mode: str = "json_schema",
        stream_usage: bool = True,
        timeout_seconds: float = 120.0,
        limiter: RequestLimiter | None = None,
        max_retries: int = 6,
        backoff_base_seconds: float = 5.0,
        backoff_max_seconds: float = 120.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self.base_url = base_url.rstrip("/") + "/"
        key = api_key if api_key is not None else read_api_key(api_key_env)
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.stream_usage = stream_usage
        self.limiter = limiter
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.backoff_max = backoff_max_seconds
        self._sleep = sleep
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    @property
    def backend(self) -> str:
        return "openai_compatible"

    @property
    def model(self) -> str:
        return self._model

    # -- request building --------------------------------------------------------------
    def payload(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m.role, "content": m.content} for m in messages)
        body: dict[str, Any] = {"model": self._model, "messages": msgs, "max_tokens": max_tokens}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if json_schema is not None and self.json_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": json_schema, "strict": True},
            }
        elif json_schema is not None and self.json_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
            if self.stream_usage:
                body["stream_options"] = {"include_usage": True}
        return body

    # -- transport with retries --------------------------------------------------------
    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_max)
            except ValueError:
                pass
        return float(min(self.backoff_base * 2**attempt + random.uniform(0, 1), self.backoff_max))

    def _send(self, body: dict[str, Any]) -> httpx.Response:
        """POST with limiter + retries; returns an *open* streaming response with 200."""
        url = self.base_url + "chat/completions"
        last_error = ""
        for attempt in range(self.max_retries + 1):
            if self.limiter is not None:
                self.limiter.acquire(self._model)
            try:
                request = self.client.build_request("POST", url, json=body, headers=self._headers)
                resp = self.client.send(request, stream=True)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}"
                delay = self._delay(attempt, None)
            else:
                if resp.status_code == 200:
                    return resp
                text = resp.read().decode("utf-8", "replace")
                resp.close()
                last_error = f"HTTP {resp.status_code}: {text[:300]}"
                if resp.status_code == 429 and _is_daily_quota(text):
                    raise QuotaExhaustedError(
                        f"provider daily quota exhausted for {self._model}; rerun later"
                    )
                if resp.status_code not in RETRYABLE:
                    raise LLMError(last_error)
                delay = self._delay(attempt, resp.headers.get("retry-after"))
            if attempt < self.max_retries:
                logger.warning("LLM request failed (%s); retry in %.1fs", last_error[:80], delay)
                self._sleep(delay)
        raise LLMError(f"giving up after {self.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _usage(data: dict[str, Any] | None) -> TokenUsage:
        if not data:
            return TokenUsage()
        details = data.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0)
        prompt = int(data.get("prompt_tokens", 0) or 0)
        return TokenUsage(
            input_tokens=max(0, prompt - cached),
            output_tokens=int(data.get("completion_tokens", 0) or 0),
            cache_read_input_tokens=cached,
        )

    def _response(
        self, text: str, usage: TokenUsage, finish: str | None, started: float
    ) -> LLMResponse:
        refused = finish in {"content_filter", "safety"}
        return LLMResponse(
            text="" if refused else text,
            model=self._model,
            backend=self.backend,
            usage=usage,
            stop_reason=finish,
            refused=refused,
            latency_ms=timed_ms(started),
        )

    # -- public API ----------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        resp = self._send(
            self.payload(messages, system=system, max_tokens=max_tokens, json_schema=json_schema)
        )
        try:
            data = json.loads(resp.read())
        finally:
            resp.close()
        choices = data.get("choices") or [{}]
        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        return self._response(
            text, self._usage(data.get("usage")), choice.get("finish_reason"), started
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]:
        started = time.perf_counter()
        resp = self._send(self.payload(messages, system=system, max_tokens=max_tokens, stream=True))
        parts: list[str] = []
        usage = TokenUsage()
        finish: str | None = None
        try:
            for line in resp.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                data = json.loads(data_str)
                if data.get("usage"):
                    usage = self._usage(data["usage"])
                for choice in data.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                        yield StreamChunk(text=piece)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        finally:
            resp.close()
        yield StreamChunk(final=self._response("".join(parts), usage, finish, started))
