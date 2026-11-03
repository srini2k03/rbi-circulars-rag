from __future__ import annotations

import pytest

from rbirag.config import DEFAULT_PRICES, ModelPrice
from rbirag.generation.cost import compute_cost, cost_usd
from rbirag.generation.llm import TokenUsage


def test_cost_hand_computed() -> None:
    price = ModelPrice(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=2000,
        cache_creation_input_tokens=400,
    )
    # 1000*2 + 500*10 + 2000*0.2 + 400*2.5 = 2000 + 5000 + 400 + 1000 = 8400 per 1M
    assert cost_usd(usage, price) == pytest.approx(0.0084)
    cost = compute_cost(usage, model="m", backend="anthropic", prices={"m": price}, usd_to_inr=88.0)
    assert cost.usd == pytest.approx(0.0084)
    assert cost.inr == pytest.approx(0.7392)
    assert cost.priced


def test_cache_prices_default_from_input() -> None:
    price = ModelPrice(input=1.0, output=1.0)
    usage = TokenUsage(cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    assert cost_usd(usage, price) == pytest.approx(0.1 + 1.25)


def test_free_tier_and_local_cost_zero_but_tokens_recorded() -> None:
    usage = TokenUsage(input_tokens=5000, output_tokens=300)
    gemini = compute_cost(
        usage,
        model="gemini-2.5-flash",
        backend="openai_compatible",
        prices=DEFAULT_PRICES,
        usd_to_inr=88,
    )
    assert gemini.usd == 0.0
    assert gemini.priced
    assert compute_cost(usage, model="qwen", backend="ollama", prices={}, usd_to_inr=88).usd == 0.0
    assert usage.total_tokens == 5300


def test_unknown_model_flagged() -> None:
    cost = compute_cost(
        TokenUsage(input_tokens=1), model="x", backend="anthropic", prices={}, usd_to_inr=1
    )
    assert not cost.priced
    assert cost.usd == 0.0


def test_usage_addition() -> None:
    total = TokenUsage(input_tokens=1, output_tokens=2) + TokenUsage(
        input_tokens=3, cache_read_input_tokens=4
    )
    assert total == TokenUsage(input_tokens=4, output_tokens=2, cache_read_input_tokens=4)
