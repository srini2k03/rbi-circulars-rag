"""Token cost accounting from the price table in settings."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from rbirag.config import ModelPrice
from rbirag.generation.llm import TokenUsage

LOCAL_BACKENDS = frozenset({"ollama", "fake"})


class Cost(BaseModel):
    usd: float
    inr: float
    priced: bool  # False when the model is missing from the price table


def cost_usd(usage: TokenUsage, price: ModelPrice) -> float:
    cache_read = price.cache_read if price.cache_read is not None else price.input * 0.1
    cache_write = price.cache_write if price.cache_write is not None else price.input * 1.25
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * cache_read
        + usage.cache_creation_input_tokens * cache_write
    ) / 1_000_000


def compute_cost(
    usage: TokenUsage,
    *,
    model: str,
    backend: str,
    prices: Mapping[str, ModelPrice],
    usd_to_inr: float,
) -> Cost:
    """Cost of one call. Local backends (Ollama) cost 0; unknown API models are flagged."""
    if backend in LOCAL_BACKENDS:
        return Cost(usd=0.0, inr=0.0, priced=True)
    price = prices.get(model)
    if price is None:
        return Cost(usd=0.0, inr=0.0, priced=False)
    usd = cost_usd(usage, price)
    return Cost(usd=round(usd, 8), inr=round(usd * usd_to_inr, 6), priced=True)
