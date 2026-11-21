from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from rbirag.compliance.anthropic_chat import (
    MIN_MAX_TOKENS,
    AnthropicToolChat,
    convert_messages,
    convert_tools,
)
from rbirag.compliance.llm import ChatMessage, ToolCallRequest, tool_call
from rbirag.compliance.tools import TOOL_SCHEMAS


class Block(SimpleNamespace):
    def model_dump(self, mode: str = "python", exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def message(*blocks: Block, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model="claude-haiku-5-5",
        usage=SimpleNamespace(
            input_tokens=900,
            output_tokens=60,
            cache_read_input_tokens=1200,
            cache_creation_input_tokens=0,
        ),
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
    )


class FakeClient:
    def __init__(self, reply: SimpleNamespace) -> None:
        self.reply = reply
        self.params: dict[str, Any] = {}
        self.messages = SimpleNamespace(stream=self._stream)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    @contextmanager
    def _stream(self, **params: Any):  # type: ignore[no-untyped-def]
        self.params = params
        yield SimpleNamespace(get_final_message=lambda: self.reply)


def test_convert_tools_and_messages() -> None:
    tools = convert_tools(TOOL_SCHEMAS)
    assert [t["name"] for t in tools] == [
        "search_regulations",
        "fetch_notification",
        "compare_clause",
    ]
    assert tools[0]["input_schema"]["required"] == ["query"] and "description" in tools[0]

    thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
    raw = [
        thinking,
        {"type": "tool_use", "id": "t1", "name": "search_regulations", "input": {"query": "kyc"}},
        {
            "type": "tool_use",
            "id": "t2",
            "name": "fetch_notification",
            "input": {"doc_id": "rbi-1"},
        },
    ]
    history = [
        ChatMessage(role="system", content="You are an analyst."),
        ChatMessage(role="user", content="Clause 3.2 ..."),
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCallRequest(id="t1", name="search_regulations", arguments='{"query": "kyc"}'),
                ToolCallRequest(
                    id="t2", name="fetch_notification", arguments='{"doc_id": "rbi-1"}'
                ),
            ],
            raw=raw,
        ),
        ChatMessage(role="tool", tool_call_id="t1", content='{"hits": []}'),
        ChatMessage(role="tool", tool_call_id="t2", content="ERROR: no notification 'rbi-1'"),
        # a text-written tool call the agent made structured: raw has no matching tool_use
        ChatMessage(
            role="assistant",
            content='{"tool": "search_regulations"}',
            tool_calls=[
                ToolCallRequest(
                    id="text_3", name="search_regulations", arguments='{"query": "x y"}'
                )
            ],
            raw=[{"type": "text", "text": '{"tool": "search_regulations"}'}],
        ),
        ChatMessage(role="tool", tool_call_id="text_3", content="{}"),
    ]
    system, msgs = convert_messages(history)
    assert system == "You are an analyst."
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"] == raw  # replayed unchanged, thinking block included
    results = msgs[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]  # one message for the turn
    assert "is_error" not in results[0] and results[1]["is_error"] is True
    rebuilt = msgs[3]["content"]
    assert rebuilt[0]["type"] == "text" and rebuilt[1] == {
        "type": "tool_use",
        "id": "text_3",
        "name": "search_regulations",
        "input": {"query": "x y"},
    }


def test_chat_request_and_tool_use_response() -> None:
    reply = message(
        Block(type="thinking", thinking="", signature="abc"),
        Block(type="text", text="Searching."),
        Block(
            type="tool_use",
            id="toolu_1",
            name="search_regulations",
            input={"query": "periodic updation"},
        ),
    )
    client = FakeClient(reply)
    chat = AnthropicToolChat("claude-haiku-5-5", effort="low", client=client)
    result = chat.chat(
        [ChatMessage(role="system", content="S"), ChatMessage(role="user", content="U")],
        tools=TOOL_SCHEMAS,
        max_tokens=512,
    )
    p = client.params
    assert p["model"] == "claude-haiku-5-5" and p["max_tokens"] == MIN_MAX_TOKENS
    assert p["system"] == [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]
    assert p["output_config"] == {"effort": "low"} and len(p["tools"]) == 3
    assert not {"temperature", "top_p", "top_k", "betas"} & set(p)
    msg = result.message
    assert msg.content == "Searching." and msg.tool_calls[0].id == "toolu_1"
    assert json.loads(msg.tool_calls[0].arguments) == {"query": "periodic updation"}
    assert msg.raw is not None and msg.raw[0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "abc",
    }
    assert result.usage.input_tokens == 900 and result.usage.cache_read_input_tokens == 1200
    assert result.backend == "anthropic" and result.finish_reason == "tool_use"


def test_refusal_and_fallback_models() -> None:
    client = FakeClient(message(Block(type="text", text="partial"), stop_reason="refusal"))
    result = AnthropicToolChat("claude-haiku-5-5", client=client).chat(
        [ChatMessage(role="user", content="U")]
    )
    assert (
        result.message.content is None
        and result.message.tool_calls == []
        and result.finish_reason == "refusal"
    )
    client = FakeClient(
        message(Block(type="text", text='{"verdict": "gap"}'), stop_reason="end_turn")
    )
    result = AnthropicToolChat("claude-opus-5-5", client=client).chat(
        [ChatMessage(role="user", content="U")], json_object=True
    )
    assert client.params["fallbacks"] == "default" and client.params["betas"] == [
        "server-side-fallback-2026-07-01"
    ]
    assert result.message.content == '{"verdict": "gap"}'


def test_missing_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from rbirag.generation.anthropic import MissingAPIKeyError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        AnthropicToolChat("claude-haiku-5-5")


def test_agent_replays_raw_blocks_through_the_loop() -> None:
    """The agent passes the assistant's raw blocks back on the next turn."""
    from rbirag.compliance.agent import ComplianceAgent
    from rbirag.compliance.llm import ScriptedChatModel, final

    first = tool_call("search_regulations", {"query": "kyc periodic updation"})
    first.raw = [
        {"type": "thinking", "thinking": "", "signature": "s1"},
        {
            "type": "tool_use",
            "id": "c1",
            "name": "search_regulations",
            "input": {"query": "kyc periodic updation"},
        },
    ]
    chat = ScriptedChatModel(
        [first, final({"verdict": "insufficient_evidence", "confidence": 0.2})]
    )

    class Store:
        documents: ClassVar[dict[str, Any]] = {}
        graph = SimpleNamespace(replaced_by=lambda d: [])

        def search(self, query: str, **kw: Any) -> Any:
            return SimpleNamespace(hits=[], confidence=None, excluded_other_entity=0)

    from rbirag.compliance.judge import ClauseInput

    ComplianceAgent(Store(), chat).check(ClauseInput(label="1", text="Some clause about KYC."))  # type: ignore[arg-type]
    second_turn = chat.calls[1][0]
    assert second_turn[2].raw == first.raw
