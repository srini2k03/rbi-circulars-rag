"""Backends against mocked transports/clients: no network, no keys."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rbirag.generation.anthropic import AnthropicLLM, MissingAPIKeyError, require_api_key
from rbirag.generation.cache import CachedLLM
from rbirag.generation.limits import RequestLimiter
from rbirag.generation.llm import LLMError, Message, QuotaExhaustedError, ScriptedLLM
from rbirag.generation.ollama import OllamaLLM
from rbirag.generation.openai_compat import OpenAICompatibleLLM, read_api_key

SECRET = "test-secret-value-123"
MSGS = [Message(role="user", content="hello")]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def oai(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> OpenAICompatibleLLM:
    clock = kw.pop("clock", Clock())
    return OpenAICompatibleLLM(
        "gemini-3.5-flash-lite",
        api_key=SECRET,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        backoff_base_seconds=1.0,
        **kw,
    )


def completion(text: str, finish: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    }


# ----------------------------------------------------------------------------- openai-compatible


def test_openai_compat_generate_payload_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion("Answer [1]."))

    llm = oai(handler)
    resp = llm.generate(MSGS, system="sys", max_tokens=50, json_schema={"type": "object"})
    req = seen[0]
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    body = json.loads(req.content)
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["max_tokens"] == 50
    assert body["response_format"]["type"] == "json_schema"
    assert "temperature" not in body
    assert resp.text == "Answer [1]."
    assert (
        resp.usage.input_tokens,
        resp.usage.cache_read_input_tokens,
        resp.usage.output_tokens,
    ) == (100, 20, 30)
    assert resp.backend == "openai_compatible"


def test_openai_compat_streaming_sse() -> None:
    events = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo [1]."}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    chunks = list(oai(handler).stream(MSGS))
    assert "".join(c.text for c in chunks) == "Hello [1]."
    final = chunks[-1].final
    assert final is not None
    assert final.text == "Hello [1]."
    assert final.usage.output_tokens == 3


def test_openai_compat_retries_429_with_retry_after() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "rate limit per minute"}},
                headers={"retry-after": "7"},
            )
        if calls["n"] == 2:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=completion("ok"))

    clock = Clock()
    assert oai(handler, clock=clock).generate(MSGS).text == "ok"
    assert calls["n"] == 3
    assert clock.sleeps[0] == 7.0
    assert 2.0 <= clock.sleeps[1] <= 3.0  # base 1s * 2**1 + jitter


def test_openai_compat_daily_quota_stops_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429,
            json={
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                }
            },
        )

    with pytest.raises(QuotaExhaustedError):
        oai(handler).generate(MSGS)
    assert calls["n"] == 1


def test_openai_compat_client_error_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMError, match="HTTP 400"):
        oai(handler).generate(MSGS)
    assert calls["n"] == 1


def test_openai_compat_content_filter_is_refusal() -> None:
    llm = oai(lambda r: httpx.Response(200, json=completion("partial", finish="content_filter")))
    resp = llm.generate(MSGS)
    assert resp.refused
    assert resp.text == ""


def test_missing_key_message_never_contains_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(Exception, match="GEMINI_API_KEY is not set"):
        read_api_key("GEMINI_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert read_api_key("GEMINI_API_KEY") == SECRET


def test_key_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    with caplog.at_level(logging.DEBUG), pytest.raises(LLMError) as exc:
        oai(handler, max_retries=1).generate(MSGS)
    assert SECRET not in caplog.text
    assert SECRET not in str(exc.value)


# ----------------------------------------------------------------------------- limiter


def test_limiter_rpm_spacing_and_persisted_rpd(tmp_path: Path) -> None:
    clock = Clock()
    lim = RequestLimiter(
        rpm=10,
        rpd=3,
        state_dir=tmp_path,
        clock=clock.clock,
        sleep=clock.sleep,
        today=lambda: "2026-11-03",
    )
    lim.acquire("m")
    lim.acquire("m")
    assert clock.sleeps == [pytest.approx(6.0)]
    lim.acquire("other")  # separate per-model spacing and budget
    assert clock.sleeps == [pytest.approx(6.0)]
    lim.acquire("m")
    with pytest.raises(QuotaExhaustedError):
        lim.acquire("m")
    # a new process sees the same daily count
    again = RequestLimiter(rpm=10, rpd=3, state_dir=tmp_path, today=lambda: "2026-11-03")
    assert again.used_today("m") == 3
    assert again.remaining_today("other") == 2
    tomorrow = RequestLimiter(rpm=10, rpd=3, state_dir=tmp_path, today=lambda: "2026-11-04")
    assert tomorrow.remaining_today("m") == 3


# ----------------------------------------------------------------------------- cache


def test_cache_reuses_generate_and_stream(tmp_path: Path) -> None:
    inner = ScriptedLLM(replies=["first answer [1].", "second answer [1]."])
    llm = CachedLLM(inner, tmp_path)
    a = llm.generate(MSGS, system="s")
    b = llm.generate(MSGS, system="s")
    assert a.text == b.text == "first answer [1]."
    assert b.cached
    assert (llm.hits, llm.misses) == (1, 1)
    # different request -> miss; stream of a cached request replays text
    llm.generate([Message(role="user", content="other")], system="s")
    streamed = list(llm.stream(MSGS, system="s"))
    assert "".join(c.text for c in streamed) in {"first answer [1].", "first answer [1]. "}
    # a fresh wrapper over the same directory still hits (persisted on disk)
    again = CachedLLM(ScriptedLLM(replies=[]), tmp_path)
    assert again.generate(MSGS, system="s").text == "first answer [1]."


def test_cache_does_not_store_refusals(tmp_path: Path) -> None:
    from rbirag.generation.llm import LLMResponse

    class Refuser(ScriptedLLM):
        def generate(self, messages, **kw):  # type: ignore[no-untyped-def,override]
            return LLMResponse(text="", model="m", backend="fake", refused=True)

    llm = CachedLLM(Refuser(replies=[]), tmp_path)
    llm.generate(MSGS)
    llm.generate(MSGS)
    assert llm.misses == 2


# ----------------------------------------------------------------------------- ollama


def test_ollama_stream_and_generate() -> None:
    lines = [
        {"message": {"content": "Banks "}, "done": False},
        {"message": {"content": "comply [1]."}, "done": False},
        {
            "message": {"content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 200,
            "eval_count": 9,
        },
    ]
    body = "\n".join(json.dumps(x) for x in lines) + "\n"
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text=body)

    llm = OllamaLLM(client=httpx.Client(transport=httpx.MockTransport(handler)))
    chunks = list(llm.stream(MSGS, system="sys", max_tokens=64))
    assert "".join(c.text for c in chunks) == "Banks comply [1]."
    final = chunks[-1].final
    assert final is not None
    assert (final.usage.input_tokens, final.usage.output_tokens) == (200, 9)
    assert seen[0]["options"]["temperature"] == 0
    assert seen[0]["options"]["num_predict"] == 64
    assert seen[0]["messages"][0]["role"] == "system"
    assert llm.generate(MSGS).text == "Banks comply [1]."


def test_ollama_error() -> None:
    llm = OllamaLLM(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(404, text="model not found"))
        )
    )
    with pytest.raises(Exception, match="404"):
        llm.generate(MSGS)


# ----------------------------------------------------------------------------- anthropic


class FakeStream:
    def __init__(self, message: Any, texts: list[str]) -> None:
        self.message = message
        self.text_stream = iter(texts)

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.message


def fake_client(message: Any, texts: list[str], calls: list[dict[str, Any]]) -> Any:
    def stream(**params: Any) -> FakeStream:
        calls.append(params)
        return FakeStream(message, texts)

    ns = SimpleNamespace(stream=stream)
    return SimpleNamespace(messages=ns, beta=SimpleNamespace(messages=ns))


def message(text: str, stop: str = "end_turn", model: str = "claude-haiku-5-5") -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop,
        stop_details=None,
        model=model,
        usage=SimpleNamespace(
            input_tokens=50,
            output_tokens=7,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=0,
        ),
    )


def test_anthropic_params_and_response() -> None:
    calls: list[dict[str, Any]] = []
    llm = AnthropicLLM(
        "claude-haiku-5-5",
        effort="low",
        client=fake_client(message("Hi [1]."), ["Hi ", "[1]."], calls),
    )
    resp = llm.generate(MSGS, system="sys", max_tokens=100, json_schema={"type": "object"})
    p = calls[0]
    assert p["model"] == "claude-haiku-5-5"
    assert p["system"] == "sys"
    assert p["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert "temperature" not in p
    assert "fallbacks" not in p  # Haiku has no server-side fallback
    assert resp.text == "Hi [1]."
    assert resp.usage.input_tokens == 50
    chunks = list(llm.stream(MSGS))
    assert [c.text for c in chunks[:-1]] == ["Hi ", "[1]."]


def test_anthropic_fallbacks_and_refusal() -> None:
    calls: list[dict[str, Any]] = []
    llm = AnthropicLLM(
        "claude-sonnet-5-5", client=fake_client(message("partial", stop="refusal"), [], calls)
    )
    resp = llm.generate(MSGS)
    assert calls[0]["fallbacks"] == "default"
    assert calls[0]["betas"] == ["server-side-fallback-2026-07-01"]
    assert resp.refused
    assert resp.text == ""


def test_anthropic_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(MissingAPIKeyError):
        require_api_key()


# ----------------------------------------------------------------------------- fallback chain


def chain_llm(
    handler: Callable[[httpx.Request], httpx.Response], health: Any = None, **kw: Any
) -> OpenAICompatibleLLM:
    from rbirag.generation.openai_compat import ModelHealth

    return OpenAICompatibleLLM(
        "gemini-3.8-flash",
        fallbacks=["gemini-3.7-flash", "gemini-3.5-flash"],
        api_key=SECRET,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
        backoff_base_seconds=0.01,
        health=health or ModelHealth(cooldown_seconds=300),
        **kw,
    )


def by_model(
    responses: dict[str, list[int]], calls: list[str]
) -> Callable[[httpx.Request], httpx.Response]:
    """Per-model scripted status codes (the last one repeats); 200 returns a completion."""

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        script = responses[model]
        status = script.pop(0) if len(script) > 1 else script[0]
        if status == 200:
            return httpx.Response(200, json=completion(f"answer from {model}"))
        return httpx.Response(status, json={"error": {"message": f"status {status} for {model}"}})

    return handler


def test_fallback_after_two_503s_and_cooldown() -> None:
    from rbirag.generation.openai_compat import ModelHealth

    now = {"t": 0.0}
    health = ModelHealth(cooldown_seconds=300, clock=lambda: now["t"])
    calls: list[str] = []
    responses = {
        "gemini-3.8-flash": [503, 503, 200],
        "gemini-3.7-flash": [200],
        "gemini-3.5-flash": [200],
    }
    llm = chain_llm(by_model(responses, calls), health)
    first = llm.generate(MSGS)
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.7-flash"]
    assert first.model == "gemini-3.7-flash"  # the model that actually answered
    assert health.in_cooldown("gemini-3.8-flash")
    calls.clear()
    llm.generate(MSGS)
    assert calls == ["gemini-3.7-flash"]  # the overloaded model is skipped while cooling down
    now["t"] = 301.0
    calls.clear()
    third = llm.generate(MSGS)
    assert calls == ["gemini-3.8-flash"]  # cooldown over: back to the primary
    assert third.model == "gemini-3.8-flash"


def test_single_503_is_retried_on_same_model() -> None:
    calls: list[str] = []
    responses = {
        "gemini-3.8-flash": [503, 200],
        "gemini-3.7-flash": [200],
        "gemini-3.5-flash": [200],
    }
    assert chain_llm(by_model(responses, calls)).generate(MSGS).model == "gemini-3.8-flash"
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash"]


def test_all_models_overloaded_gives_up_with_retry_budget() -> None:
    calls: list[str] = []
    responses = {m: [503] for m in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash")}
    with pytest.raises(LLMError, match="giving up"):
        chain_llm(by_model(responses, calls), max_retries=6).generate(MSGS)
    assert set(calls) == {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"}


def test_404_drops_model_and_continues_without_retry() -> None:
    calls: list[str] = []
    responses = {"gemini-3.8-flash": [404], "gemini-3.7-flash": [200], "gemini-3.5-flash": [200]}
    llm = chain_llm(by_model(responses, calls))
    assert llm.generate(MSGS).model == "gemini-3.7-flash"
    assert calls == ["gemini-3.8-flash", "gemini-3.7-flash"]
    calls.clear()
    llm.generate(MSGS)
    assert calls == ["gemini-3.7-flash"]  # dropped for the rest of the run


def test_every_model_fatal_aborts_with_clear_message() -> None:
    from rbirag.generation.llm import ConfigurationError

    calls: list[str] = []
    responses = {"gemini-3.8-flash": [404], "gemini-3.7-flash": [403], "gemini-3.5-flash": [404]}
    with pytest.raises(ConfigurationError, match="every model in the chain") as exc:
        chain_llm(by_model(responses, calls)).generate(MSGS)
    assert calls == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]  # no retries
    assert SECRET not in str(exc.value)


def test_single_model_fatal_401_no_retry() -> None:
    from rbirag.generation.llm import ConfigurationError

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "API key not valid"}})

    with pytest.raises(ConfigurationError, match="401"):
        oai(handler).generate(MSGS)
    assert calls["n"] == 1


def test_daily_quota_moves_to_next_model() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "gemini-3.8-flash":
            return httpx.Response(
                429, json={"error": {"message": "GenerateRequestsPerDayPerProjectPerModel"}}
            )
        return httpx.Response(200, json=completion("ok"))

    assert chain_llm(handler).generate(MSGS).model == "gemini-3.7-flash"
    assert calls == ["gemini-3.8-flash", "gemini-3.7-flash"]


def test_client_side_rpd_exhaustion_moves_to_next_model(tmp_path: Path) -> None:
    lim = RequestLimiter(rpm=1000, rpd=1, state_dir=tmp_path, today=lambda: "d")
    calls: list[str] = []
    responses = {m: [200] for m in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash")}
    llm = chain_llm(by_model(responses, calls), limiter=lim)
    assert [llm.generate(MSGS).model for _ in range(3)] == [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
    ]
    with pytest.raises(QuotaExhaustedError):
        llm.generate(MSGS)


# ----------------------------------------------------------------------------- cache hygiene


def test_cache_never_stores_errors(tmp_path: Path) -> None:
    from rbirag.generation.llm import LLMResponse

    attempts = {"n": 0}

    class Flaky(ScriptedLLM):
        def generate(self, messages, **kw):  # type: ignore[no-untyped-def,override]
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise LLMError("HTTP 503: overloaded")
            if attempts["n"] == 2:
                return LLMResponse(text="partial", model="m", backend="b", stop_reason="error")
            if attempts["n"] == 3:
                return LLMResponse(text="not json at all", model="m", backend="b")
            return LLMResponse(text='{"ok": true}', model="m", backend="b")

    llm = CachedLLM(Flaky(replies=[]), tmp_path)
    schema = {"type": "object"}
    with pytest.raises(LLMError):
        llm.generate(MSGS, json_schema=schema)
    assert not any(tmp_path.rglob("*.json"))  # the exception was not cached
    assert llm.generate(MSGS, json_schema=schema).stop_reason == "error"
    assert not any(tmp_path.rglob("*.json"))  # error stop not cached
    assert llm.generate(MSGS, json_schema=schema).text == "not json at all"
    assert not any(tmp_path.rglob("*.json"))  # invalid JSON for a JSON request not cached
    assert llm.generate(MSGS, json_schema=schema).text == '{"ok": true}'
    assert len(list(tmp_path.rglob("*.json"))) == 1
    assert llm.generate(MSGS, json_schema=schema).cached
    assert attempts["n"] == 4


def test_fallback_settings_parse_comma_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    from rbirag.config import Settings

    monkeypatch.setenv("RBIRAG_JUDGE_MODEL_FALLBACKS", "gemini-3.7-flash, gemini-3.6-flash")
    monkeypatch.setenv("RBIRAG_OAI_MODEL_FALLBACKS", "")
    s = Settings(llm_provider="gemini")
    assert s.judge_model_fallbacks == ["gemini-3.7-flash", "gemini-3.6-flash"]
    assert s.oai_model_fallbacks == []
    assert s.qa_model_fallbacks == ["gemini-3.7-flash", "gemini-3.5-flash"]


# ----------------------------------------------------------------------------- providers


def test_groq_is_the_default_provider() -> None:
    from rbirag.config import Settings

    s = Settings()
    assert s.llm_provider == "groq"
    assert s.oai_base_url == "https://api.groq.com/openai/v1"
    assert s.oai_api_key_env == "GROQ_API_KEY"
    assert (s.judge_model, s.qa_model) == ("openai/gpt-oss-120b", "openai/gpt-oss-120b")
    assert s.judge_model_fallbacks == ["qwen/qwen3.8-27b"]
    assert (s.oai_model, s.oai_model_fallbacks) == ("openai/gpt-oss-20b", ["qwen/qwen3.8-27b"])
    assert s.oai_model != s.judge_model  # generator differs from the judge
    assert s.llm_rpm == 25
    assert s.rpd_for("openai/gpt-oss-120b") <= 1000
    assert 0 < s.tpm_for("openai/gpt-oss-120b") <= 8_000
    assert s.prices["openai/gpt-oss-120b"].input == 0.0


def test_provider_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from rbirag.config import Settings

    monkeypatch.setenv("RBIRAG_JUDGE_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("RBIRAG_LLM_RPD", "50")
    s = Settings()
    assert s.judge_model == "qwen/qwen3.8-27b"
    # an explicit primary keeps the rest of the preset chain as fallbacks
    assert s.judge_model_fallbacks == ["openai/gpt-oss-120b"]
    assert s.rpd_for("openai/gpt-oss-20b") == 50
    g = Settings(llm_provider="gemini", llm_rpd=0)
    assert g.oai_base_url.startswith("https://generativelanguage.googleapis.com")
    assert g.oai_api_key_env == "GEMINI_API_KEY"
    assert g.llm_rpm == 10
    assert g.tpm_for("gemini-3.8-flash") == 0


def test_custom_provider_requires_explicit_settings() -> None:
    from pydantic import ValidationError

    from rbirag.config import Settings

    with pytest.raises(ValidationError, match="requires"):
        Settings(llm_provider="custom")
    s = Settings(
        llm_provider="custom", oai_base_url="https://openrouter.ai/api/v1", oai_api_key_env="OPENROUTER_API_KEY",
        oai_model="m1", qa_model="m2", judge_model="m3",
    )  # fmt: skip
    assert s.judge_model_fallbacks == []
    assert s.rpd_for("m1") == 200


def test_limiter_per_model_rpd_and_tpm_budget() -> None:
    clock = Clock()
    lim = RequestLimiter(
        rpm=1000,
        rpd=lambda m: 2 if m == "big" else 100,
        tpm=lambda m: 1000 if m == "big" else 0,
        clock=clock.clock,
        sleep=clock.sleep,
        today=lambda: "d",
    )
    lim.acquire("big", tokens=600)
    lim.settle("big", 700)  # actual usage replaces the estimate
    clock.now += 10
    lim.acquire("big", tokens=400)  # 700 + 400 > 1000: wait until the first leaves the window
    assert clock.sleeps[-1] == pytest.approx(50.0, abs=0.1)
    with pytest.raises(QuotaExhaustedError):
        lim.acquire("big")  # rpd 2 reached
    for _ in range(5):
        lim.acquire("small", tokens=10_000)  # no TPM budget for this model
    assert lim.remaining_today("small") == 95


def test_groq_daily_token_limit_moves_to_next_model() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "llama-3.3-70b-versatile":
            msg = ("Rate limit reached for model `llama-3.3-70b-versatile` on tokens per day (TPD): "
                   "Limit 100000, Used 99000, Requested 3000.")  # fmt: skip
            return httpx.Response(
                429,
                json={"error": {"message": msg, "type": "tokens", "code": "rate_limit_exceeded"}},
            )
        return httpx.Response(200, json=completion("ok"))

    llm = OpenAICompatibleLLM(
        "llama-3.3-70b-versatile", fallbacks=["openai/gpt-oss-120b"], api_key=SECRET,
        client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None,
    )  # fmt: skip
    assert llm.generate(MSGS).model == "openai/gpt-oss-120b"
    assert calls == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]


def test_groq_tpm_429_retries_same_model_with_retry_after() -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        if len(calls) == 1:
            msg = "Rate limit reached for model `llama-3.1-8b-instant` on tokens per minute (TPM)"
            return httpx.Response(
                429, json={"error": {"message": msg}}, headers={"retry-after": "3"}
            )
        return httpx.Response(200, json=completion("ok"))

    llm = OpenAICompatibleLLM(
        "llama-3.1-8b-instant", fallbacks=["openai/gpt-oss-20b"], api_key=SECRET,
        client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleeps.append,
    )  # fmt: skip
    assert llm.generate(MSGS).model == "llama-3.1-8b-instant"
    assert calls == ["llama-3.1-8b-instant", "llama-3.1-8b-instant"]
    assert sleeps == [3.0]


def test_groq_preset_uses_json_object_and_strips_reasoning() -> None:
    from rbirag.config import Settings
    from rbirag.generation.openai_compat import strip_reasoning

    assert Settings().oai_json_mode == "json_object"
    assert Settings(llm_provider="gemini").oai_json_mode == "json_schema"
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion('<think>\nweigh {options}\n</think>\n{"a": 1}'))

    llm = oai(handler, json_mode="json_object")
    assert llm.generate(MSGS, json_schema={"type": "object"}).text == '{"a": 1}'
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert strip_reasoning("plain answer [1].") == "plain answer [1]."
    assert strip_reasoning("<think>unfinished") == ""


def test_groq_stream_usage_in_x_groq() -> None:
    events = [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
         "x_groq": {"usage": {"prompt_tokens": 11, "completion_tokens": 2}}},
    ]  # fmt: skip
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    chunks = list(oai(lambda r: httpx.Response(200, text=body)).stream(MSGS))
    final = chunks[-1].final
    assert final is not None
    assert (final.usage.input_tokens, final.usage.output_tokens) == (11, 2)


# ----------------------------------------------------------------------------- per-item rejections


def test_json_validate_failed_retries_once_without_json_mode() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(
                400,
                json={
                    "error": {"message": "Failed to generate JSON", "code": "json_validate_failed"}
                },
            )
        return httpx.Response(200, json=completion('Sure: {"question": "q", "answer": "a"}'))

    resp = oai(handler, json_mode="json_object").generate(MSGS, json_schema={"type": "object"})
    assert resp.text.endswith('{"question": "q", "answer": "a"}')
    assert len(bodies) == 2
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]


def test_rejected_twice_raises_request_rejected_without_retries() -> None:
    from rbirag.generation.llm import RequestRejectedError

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"code": "json_validate_failed"}})

    with pytest.raises(RequestRejectedError):
        oai(handler).generate(MSGS, json_schema={"type": "object"})
    assert calls["n"] == 2  # original + one retry without JSON mode
    calls["n"] = 0
    with pytest.raises(RequestRejectedError):
        oai(handler).generate(MSGS)  # no JSON mode: no second attempt
    assert calls["n"] == 1
