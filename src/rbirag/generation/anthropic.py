"""Anthropic Messages API backend (streaming), via the official ``anthropic`` SDK.

* The API key is read by the SDK from ``ANTHROPIC_API_KEY``; this module only checks
  that it is non-empty and never logs or echoes it.
* Requests always stream (``messages.stream``) and use ``get_final_message()`` for usage,
  so long answers never hit HTTP timeouts.
* No sampling parameters are sent: current models reject non-default values.
* Models with server-side refusal fallbacks (Opus 5.5 / Sonnet 5.5 family) are called
  through the beta endpoint with ``fallbacks="default"``; every call checks
  ``stop_reason == "refusal"`` before reading content.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from typing import Any

from rbirag.generation.llm import LLMResponse, Message, StreamChunk, TokenUsage, timed_ms

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side ``fallbacks="default"`` parameter on the Claude API.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1")


class MissingAPIKeyError(RuntimeError):
    pass


def require_api_key() -> None:
    """Fail fast (without revealing anything about the value) if no key is configured."""
    if not (
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    ):
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set or is empty. Export it locally, or add it as a "
            "repository secret named ANTHROPIC_API_KEY for GitHub Actions."
        )


def _usage(message: Any) -> TokenUsage:
    u = getattr(message, "usage", None)
    if u is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
    )


def _text(message: Any) -> str:
    return "".join(
        getattr(b, "text", "")
        for b in getattr(message, "content", []) or []
        if getattr(b, "type", "") == "text"
    )


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        *,
        effort: str | None = "low",
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
        client: Any = None,
    ) -> None:
        self._model = model
        self.effort = effort
        if client is None:
            require_api_key()
            import anthropic

            client = anthropic.Anthropic(timeout=timeout_seconds, max_retries=max_retries)
        self.client = client

    @property
    def backend(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    @property
    def uses_server_fallback(self) -> bool:
        return self._model.startswith(FALLBACK_MODELS)

    def request_params(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            params["system"] = system
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            params["output_config"] = output_config
        if self.uses_server_fallback:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _open_stream(self, params: dict[str, Any]) -> AbstractContextManager[Any]:
        if "betas" in params:
            return self.client.beta.messages.stream(**params)  # type: ignore[no-any-return]
        return self.client.messages.stream(**params)  # type: ignore[no-any-return]

    def _response(self, message: Any, started: float) -> LLMResponse:
        stop_reason = getattr(message, "stop_reason", None)
        refused = stop_reason == "refusal"
        if refused:
            details = getattr(message, "stop_details", None)
            logger.warning(
                "model declined the request (category=%s)", getattr(details, "category", None)
            )
        return LLMResponse(
            text="" if refused else _text(message),
            model=str(getattr(message, "model", None) or self._model),
            backend=self.backend,
            usage=_usage(message),
            stop_reason=stop_reason,
            refused=refused,
            latency_ms=timed_ms(started),
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        params = self.request_params(
            messages, system=system, max_tokens=max_tokens, json_schema=json_schema
        )
        started = time.perf_counter()
        with self._open_stream(params) as stream:
            message = stream.get_final_message()
        return self._response(message, started)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]:
        params = self.request_params(messages, system=system, max_tokens=max_tokens)
        started = time.perf_counter()
        with self._open_stream(params) as stream:
            for text in stream.text_stream:
                if text:
                    yield StreamChunk(text=text)
            message = stream.get_final_message()
        yield StreamChunk(final=self._response(message, started))
