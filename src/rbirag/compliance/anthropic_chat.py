"""Claude (Messages API) as a tool-calling chat model for the compliance agent.

Built on :class:`~rbirag.generation.anthropic.AnthropicLLM`, so client construction, the
API-key check, effort, streaming with ``get_final_message()`` and the mapping of
401/403/404 to :class:`~rbirag.generation.llm.ConfigurationError` are shared with the
RAG backend.

Translation from the agent's OpenAI-shaped history:

* system messages -> the ``system`` parameter, marked for prompt caching (the planner
  prompt and tool definitions repeat on every turn of every clause);
* tool schemas -> ``{"name", "description", "input_schema"}``;
* consecutive tool results -> one user message of ``tool_result`` blocks (an ``ERROR:``
  result is sent with ``is_error``);
* an assistant turn is replayed from its raw content blocks (thinking blocks included,
  unchanged), unless the agent turned a text-written tool call into a structured one.

No sampling parameters are sent (current models reject non-default values); depth is
controlled with ``effort``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from rbirag.compliance.llm import ChatMessage, ChatResult, ToolCallRequest
from rbirag.generation.anthropic import AnthropicLLM, _usage
from rbirag.generation.llm import ConfigurationError, fatal_http_message, timed_ms

logger = logging.getLogger(__name__)

# Thinking (adaptive by default on current models) counts against max_tokens; never let
# a small planner/judge budget truncate a reply.
MIN_MAX_TOKENS = 4096


def convert_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", tool)
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def _tool_input(arguments: str) -> dict[str, Any]:
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}
    return value if isinstance(value, dict) else {"_raw": value}


def _assistant_blocks(msg: ChatMessage) -> list[dict[str, Any]]:
    if msg.raw is not None:
        raw_ids = {b.get("id") for b in msg.raw if b.get("type") == "tool_use"}
        if all(c.id in raw_ids for c in msg.tool_calls):
            return msg.raw
    blocks: list[dict[str, Any]] = []
    if msg.content:
        blocks.append({"type": "text", "text": msg.content})
    blocks += [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": _tool_input(c.arguments)}
        for c in msg.tool_calls
    ]
    return blocks or [{"type": "text", "text": "(no content)"}]


def convert_messages(
    messages: Sequence[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content or "")
        elif msg.role == "tool":
            text = msg.content or ""
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id or "",
                "content": text,
            }
            if text.startswith("ERROR:"):
                block["is_error"] = True
            prev = out[-1] if out else None
            if (
                prev is not None
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and all(b.get("type") == "tool_result" for b in prev["content"])
            ):
                prev["content"].append(block)  # all results of one turn in one message
            else:
                out.append({"role": "user", "content": [block]})
        elif msg.role == "assistant":
            out.append({"role": "assistant", "content": _assistant_blocks(msg)})
        else:
            out.append({"role": "user", "content": msg.content or ""})
    return ("\n\n".join(system_parts) or None), out


class AnthropicToolChat(AnthropicLLM):
    """Claude through the official SDK, speaking the agent's ChatModel protocol."""

    def chat_params(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> dict[str, Any]:
        system, converted = convert_messages(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens, MIN_MAX_TOKENS),
            "messages": converted,
        }
        if system:
            params["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if tools:
            params["tools"] = convert_tools(tools)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.uses_server_fallback:
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        return params

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> ChatResult:
        params = self.chat_params(
            messages, tools=tools, max_tokens=max_tokens, json_object=json_object
        )
        started = time.perf_counter()
        try:
            with self._open_stream(params) as stream:
                message = stream.get_final_message()
        except self._fatal_errors() as exc:
            status = int(getattr(exc, "status_code", 404) or 404)
            raise ConfigurationError(fatal_http_message(status, self.model)) from exc
        stop_reason = getattr(message, "stop_reason", None)
        blocks = list(getattr(message, "content", []) or [])
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            logger.warning("model declined (category=%s)", getattr(details, "category", None))
            text, calls, raw = "", [], None
        else:
            text = "".join(
                getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text"
            )
            calls = [
                ToolCallRequest(
                    id=str(b.id),
                    name=str(b.name),
                    arguments=json.dumps(b.input, ensure_ascii=False),
                )
                for b in blocks
                if getattr(b, "type", "") == "tool_use"
            ]
            raw = [b.model_dump(mode="json", exclude_none=True) for b in blocks]
        return ChatResult(
            message=ChatMessage(role="assistant", content=text or None, tool_calls=calls, raw=raw),
            model=str(getattr(message, "model", None) or self.model),
            backend="anthropic",
            usage=_usage(message),
            finish_reason=stop_reason,
            latency_ms=timed_ms(started),
        )
