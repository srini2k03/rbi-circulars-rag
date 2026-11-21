"""Tool-calling chat models for the compliance agent.

One implementation covers Groq, Gemini and Ollama: all three serve the OpenAI
chat-completions API with ``tools``. It subclasses the RAG pipeline's
:class:`~rbirag.generation.openai_compat.OpenAICompatibleLLM`, so the client-side rate
limiter, model fallback chain and retry policy are the ones already tuned for the free
tiers.

:class:`CachedChatModel` stores every response under a hash of the full request. In
``replay`` mode a missing entry raises :class:`CacheMissError` instead of calling the
provider, which is how CI re-runs the whole evaluation deterministically and for free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rbirag.generation.llm import TokenUsage, timed_ms
from rbirag.generation.openai_compat import OpenAICompatibleLLM, strip_reasoning

if TYPE_CHECKING:
    from rbirag.config import Settings

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant", "tool"]
CacheMode = Literal["use", "replay", "refresh", "off"]
CHAT_BACKENDS = ("groq", "gemini", "ollama", "custom", "anthropic", "fake")


class ToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: str = "{}"  # raw JSON text, as the model produced it
    # Other provider fields of the call, sent back unchanged with it (Gemini's
    # OpenAI-compatible API returns ``extra_content`` with a thought signature that must
    # accompany the call on the next turn, or the request is rejected).
    extra: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_call_id: str | None = None
    # Provider-native content blocks of an assistant turn (Anthropic: thinking, text and
    # tool_use blocks), replayed unchanged to the same provider on the next request.
    raw: list[dict[str, Any]] | None = None

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    **(c.extra or {}),
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if msg["content"] is None and self.role != "assistant":
            msg["content"] = ""
        return msg


class ChatResult(BaseModel):
    message: ChatMessage
    model: str
    backend: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    latency_ms: float = 0.0
    cached: bool = False


class CacheMissError(RuntimeError):
    """Replay mode found no stored response for this exact request."""


@runtime_checkable
class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def backend(self) -> str: ...

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> ChatResult: ...


# ----------------------------------------------------------------------------- OpenAI-compatible


class OpenAIToolChat(OpenAICompatibleLLM):
    """OpenAI-compatible chat completions with tool calling (Groq, Gemini, Ollama /v1)."""

    def __init__(self, *args: Any, backend_name: str = "openai_compatible", **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._backend_name = backend_name

    @property
    def backend(self) -> str:
        return self._backend_name

    def chat_body(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        if json_object and not tools and self.json_mode != "none":
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> ChatResult:
        started = time.perf_counter()
        body = self.chat_body(messages, tools=tools, max_tokens=max_tokens, json_object=json_object)
        try:
            resp, served_by = self._send(body)
        except Exception as exc:
            # Groq's JSON mode rejects output that is not valid JSON; retry once without it.
            if "response_format" not in body or "json_validate_failed" not in str(exc):
                raise
            logger.warning("JSON mode rejected the output; retrying once without it")
            body = {k: v for k, v in body.items() if k != "response_format"}
            resp, served_by = self._send(body)
        try:
            data = json.loads(resp.read())
        finally:
            resp.close()
        choice = (data.get("choices") or [{}])[0]
        raw = choice.get("message") or {}
        calls = [
            ToolCallRequest(
                id=str(c.get("id") or f"call_{i}"),
                name=str((c.get("function") or {}).get("name") or ""),
                arguments=_arguments_text((c.get("function") or {}).get("arguments")),
                extra={k: v for k, v in c.items() if k not in {"id", "type", "function", "index"}}
                or None,
            )
            for i, c in enumerate(raw.get("tool_calls") or [])
        ]
        content = raw.get("content")
        usage = self._usage(data.get("usage"))
        if self.limiter is not None and usage.total_tokens:
            self.limiter.settle(served_by, usage.total_tokens)
        return ChatResult(
            message=ChatMessage(
                role="assistant",
                content=strip_reasoning(content) if isinstance(content, str) else None,
                tool_calls=calls,
            ),
            model=served_by,
            backend=self.backend,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            latency_ms=timed_ms(started),
        )


def _arguments_text(value: object) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)  # Ollama sometimes returns an object


# ----------------------------------------------------------------------------- cache


def request_key(model: str, body: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, **body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedChatModel:
    """Disk cache of chat responses keyed by the full request.

    ``use``: read, else call and store; ``replay``: read only (miss -> CacheMissError);
    ``refresh``: always call and store; ``off``: pass through.
    """

    def __init__(self, inner: ChatModel, cache_dir: Path, mode: CacheMode = "use") -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.mode = mode
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def backend(self) -> str:
        return self.inner.backend

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> ChatResult:
        if self.mode == "off":
            return self.inner.chat(
                messages, tools=tools, max_tokens=max_tokens, json_object=json_object
            )
        body = {
            "messages": [m.model_dump(exclude_defaults=True) for m in messages],
            "tools": list(tools or []),
            "max_tokens": max_tokens,
            "json_object": json_object,
        }
        key = request_key(self.inner.model, body)
        path = self._path(key)
        if self.mode in {"use", "replay"} and path.exists():
            self.hits += 1
            result = ChatResult.model_validate_json(path.read_text(encoding="utf-8"))
            return result.model_copy(update={"cached": True})
        if self.mode == "replay":
            self.misses += 1
            raise CacheMissError(f"no cached response for request {key[:12]} ({self.inner.model})")
        self.misses += 1
        result = self.inner.chat(
            messages, tools=tools, max_tokens=max_tokens, json_object=json_object
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(), encoding="utf-8")
        tmp.replace(path)
        return result


# ----------------------------------------------------------------------------- fake


Reply = (
    ChatMessage | Callable[[Sequence[ChatMessage], Sequence[dict[str, Any]] | None], ChatMessage]
)


class ScriptedChatModel:
    """Replies in order (or computed by a function). For tests and offline smoke runs."""

    def __init__(self, replies: Sequence[Reply] | Reply, *, model: str = "scripted") -> None:
        self._replies: list[Reply] | Reply
        if isinstance(replies, ChatMessage) or callable(replies):
            self._replies = replies
        else:
            self._replies = list(replies)
        self._model = model
        self.calls: list[tuple[list[ChatMessage], list[dict[str, Any]] | None]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return "fake"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> ChatResult:
        self.calls.append((list(messages), list(tools) if tools else None))
        if isinstance(self._replies, list):
            if not self._replies:
                raise RuntimeError("ScriptedChatModel ran out of replies")
            reply = self._replies.pop(0)
        else:
            reply = self._replies
        message = reply(messages, tools) if callable(reply) else reply
        prompt = sum(len(m.content or "") for m in messages)
        out = len(message.content or "") + sum(len(c.arguments) for c in message.tool_calls)
        return ChatResult(
            message=message,
            model=self._model,
            backend="fake",
            usage=TokenUsage(input_tokens=max(1, prompt // 4), output_tokens=max(1, out // 4)),
            finish_reason="tool_calls" if message.tool_calls else "stop",
        )


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ChatMessage:
    """An assistant message requesting one tool call (test helper)."""
    return ChatMessage(
        role="assistant",
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=json.dumps(arguments))],
    )


def final(payload: dict[str, Any]) -> ChatMessage:
    """An assistant message with a JSON final answer (test helper)."""
    return ChatMessage(role="assistant", content=json.dumps(payload))


class ReplayOnlyModel:
    """Stands in for a provider in replay mode: it has the configured model id (the cache
    key) but no credentials, and any call means the cache is stale."""

    def __init__(self, model: str, backend: str) -> None:
        self._model = model
        self._backend = backend

    @property
    def model(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return self._backend

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> ChatResult:
        raise CacheMissError(f"replay mode: no recorded response ({self._model})")


def configured_model(settings: Settings, backend: str) -> str:
    """The model id a backend is configured with (before any fallback)."""
    from rbirag.providers import get_preset

    if settings.compliance_model:
        return settings.compliance_model
    if backend == "anthropic":
        return settings.anthropic_model
    if backend == "ollama":
        return settings.ollama_model
    if backend in {"custom", settings.llm_provider}:
        return settings.oai_model
    preset = get_preset(backend)
    return preset.generator[0] if preset else settings.oai_model


# ----------------------------------------------------------------------------- factory


def build_chat_model(
    settings: Settings,
    backend: str | None = None,
    *,
    model: str | None = None,
    cache_mode: CacheMode | None = None,
) -> ChatModel:
    """The agent's chat model for ``backend`` (default ``settings.compliance_llm_backend``)."""
    from rbirag.generation.rag import shared_health, shared_limiter
    from rbirag.providers import get_preset

    name = backend or settings.compliance_llm_backend
    if name not in CHAT_BACKENDS:
        raise ValueError(f"unknown chat backend {name!r}; expected one of {CHAT_BACKENDS}")
    if name == "fake":
        raise ValueError("the fake backend is built in tests with ScriptedChatModel")
    mode = cache_mode or settings.compliance_llm_cache_mode
    inner: ChatModel
    if mode == "replay":
        # no credentials needed: every response must come from the cache
        replay = ReplayOnlyModel(model or configured_model(settings, name), name)
        return CachedChatModel(replay, settings.compliance_llm_cache_dir, "replay")
    if name == "anthropic":
        from rbirag.compliance.anthropic_chat import AnthropicToolChat

        inner = AnthropicToolChat(
            model or settings.compliance_model or settings.anthropic_model,
            effort=settings.anthropic_effort,
            timeout_seconds=settings.anthropic_timeout_seconds,
        )
    elif name == "ollama":
        inner = OpenAIToolChat(
            model or settings.compliance_model or settings.ollama_model,
            base_url=settings.ollama_url.rstrip("/") + "/v1",
            api_key="ollama",
            json_mode="json_object",
            timeout_seconds=settings.ollama_timeout_seconds,
            max_retries=2,
            backend_name="ollama",
        )
    else:
        provider = (
            settings if name in {settings.llm_provider, "custom"} else _for_provider(settings, name)
        )
        preset = get_preset(name)
        chain_model = model or settings.compliance_model or provider.oai_model
        requested = settings.compliance_model_fallbacks
        if requested == ["auto"]:  # the provider preset's other generator models
            fallbacks = [m for m in (preset.generator if preset else ()) if m != chain_model]
        elif requested in ([], ["none"]):  # one model only (clean comparisons)
            fallbacks = []
        else:
            fallbacks = list(requested)
        inner = OpenAIToolChat(
            chain_model,
            fallbacks=fallbacks,
            health=shared_health(provider),
            switch_after_503=provider.llm_switch_after_503,
            base_url=provider.oai_base_url,
            api_key_env=provider.oai_api_key_env,
            reasoning_effort=provider.oai_reasoning_effort,
            json_mode=provider.oai_json_mode,
            timeout_seconds=provider.oai_timeout_seconds,
            limiter=shared_limiter(provider),
            max_retries=provider.llm_max_retries,
            backoff_base_seconds=provider.llm_backoff_base_seconds,
            backend_name=name,
        )
    if mode == "off":
        return inner
    return CachedChatModel(inner, settings.compliance_llm_cache_dir, mode)


def _for_provider(settings: Settings, provider: str) -> Settings:
    from rbirag.api.service import settings_for_provider

    return settings_for_provider(settings, provider)
