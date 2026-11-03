"""The LLM protocol shared by every backend, plus deterministic fakes for tests."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


class LLMResponse(BaseModel):
    text: str
    model: str
    backend: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    stop_reason: str | None = None
    refused: bool = False  # the provider declined (e.g. a safety classifier)
    latency_ms: float = 0.0
    cached: bool = False  # served from the on-disk LLM cache

    @property
    def ok(self) -> bool:
        """A successful completion: not refused, not empty, not an error stop."""
        return not self.refused and bool(self.text.strip()) and self.stop_reason != "error"


@dataclass
class StreamChunk:
    """One streamed piece of text; the last chunk carries the final :class:`LLMResponse`."""

    text: str = ""
    final: LLMResponse | None = None


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str


@runtime_checkable
class LLM(Protocol):
    @property
    def backend(self) -> str: ...

    @property
    def model(self) -> str: ...

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]: ...


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for fakes and cost estimates."""
    return max(1, len(text) // 4) if text else 0


# ----------------------------------------------------------------------------- fakes


@dataclass
class ScriptedLLM:
    """Returns canned replies in order (or computed by a function). For tests."""

    replies: list[str] | Callable[[Sequence[Message], str | None], str]
    model: str = "scripted"
    backend: str = "fake"
    calls: list[tuple[str | None, list[Message]]] = field(default_factory=list)

    def _reply(self, messages: Sequence[Message], system: str | None) -> str:
        self.calls.append((system, list(messages)))
        if callable(self.replies):
            return self.replies(messages, system)
        if not self.replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        text = self._reply(messages, system)
        prompt = (system or "") + "".join(m.content for m in messages)
        return LLMResponse(
            text=text,
            model=self.model,
            backend=self.backend,
            usage=TokenUsage(input_tokens=approx_tokens(prompt), output_tokens=approx_tokens(text)),
            stop_reason="end_turn",
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]:
        response = self.generate(messages, system=system, max_tokens=max_tokens)
        for piece in re.findall(r"\S+\s*", response.text):
            yield StreamChunk(text=piece)
        yield StreamChunk(final=response)


class FakeLLM(ScriptedLLM):
    """Extractive, deterministic "LLM": answers with the first sentence of source [1].

    Refuses (with the configured refusal sentence) when the prompt holds no sources.
    """

    def __init__(self, refusal: str | None = None) -> None:
        from rbirag.generation.prompts import REFUSAL_TEXT

        self._refusal = refusal or REFUSAL_TEXT
        super().__init__(replies=self._answer, model="fake-extractive", backend="fake")

    def _answer(self, messages: Sequence[Message], system: str | None) -> str:
        prompt = messages[-1].content if messages else ""
        m = re.search(
            r"^\[1\][^\n]*\n(?P<body>.+?)(?:\n\[2\]|\n\nQuestion:|\Z)", prompt, re.S | re.M
        )
        if not m:
            return self._refusal
        body = " ".join(m.group("body").split())
        sentence = re.split(r"(?<=[.;])\s", body, maxsplit=1)[0][:300].rstrip(" .;")
        return f"{sentence} [1]."


def timed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class LLMError(RuntimeError):
    """A provider error that survived retries."""


class QuotaExhaustedError(LLMError):
    """The (daily) request budget is used up; rerun later to resume from the cache."""


class ConfigurationError(LLMError):
    """Fatal setup problem (unknown model, bad or unauthorised key). Never retried: the
    run must stop and nothing it produced may be committed."""


FATAL_HTTP_STATUS = frozenset({401, 403, 404})


def fatal_http_message(status: int, model: str, detail: str = "") -> str:
    reason = {
        401: "the API key was rejected (401 Unauthorized)",
        403: "the API key is not permitted to use this model or endpoint (403 Forbidden)",
        404: f"model {model!r} was not found or is not available to this key (404)",
    }.get(status, f"HTTP {status}")
    hint = " Check the model id (list the models available to the key) and the key secret."
    return f"Fatal configuration error: {reason}.{hint}" + (
        f" Provider said: {detail}" if detail else ""
    )
