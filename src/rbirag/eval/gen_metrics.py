"""Per-answer generation records and their aggregation (with bootstrap CIs).

Metrics:
* ``judge_faithfulness`` - supported claims / claims (LLM judge, claim decomposition).
* ``judge_relevance``    - judge's 1-5 relevance mapped to [0, 1].
* ``nli_faithfulness``   - share of sentences entailed by their cited sources (NLI).
* ``citation_accuracy``  - share of (sentence, citation) pairs that are entailed (NLI).
* ``emb_relevance``      - cosine(question, answer) with the retrieval embedder.
* ``citation_valid``     - every citation resolves and every factual sentence is cited.
* refusal precision / recall - positives are the *unanswerable* questions.

Quality metrics are averaged over answerable questions that were actually answered.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, Field

from rbirag.eval.stats import Estimate, bootstrap_ci, bootstrap_ratio_ci


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class GenerationRecord(BaseModel):
    qa_id: str
    config: str
    backend: str
    model: str | None
    answerable: bool
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    confidence: float | None = None
    citation_valid: bool
    n_citations: int
    n_invalid_citations: int
    n_uncited_sentences: int
    judge_model: str | None = None
    judge_faithfulness: float | None = None
    judge_relevance: float | None = None
    judge_error: str | None = None
    nli_faithfulness: float | None = None
    citation_accuracy: float | None = None
    emb_relevance: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False  # answer served from the LLM cache (no new tokens, no latency)
    judge_cached: bool = False
    # when this version of the record was written (used to merge runs; see eval.records)
    updated_at: str | None = Field(default_factory=_now)
    latency_ms: dict[str, float] = {}
    error: str | None = None


QUALITY_METRICS = (
    "judge_faithfulness",
    "judge_relevance",
    "nli_faithfulness",
    "citation_accuracy",
    "emb_relevance",
)


def _values(records: Sequence[GenerationRecord], name: str) -> list[float]:
    out: list[float] = []
    for r in records:
        v = getattr(r, name)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out.append(float(v))
    return out


def summarize(
    records: Sequence[GenerationRecord], *, n_resamples: int = 1000, seed: int = 13
) -> dict[str, Estimate]:
    ok = [r for r in records if r.error is None]
    answered = [r for r in ok if r.answerable and not r.refused]
    out: dict[str, Estimate] = {}
    for name in QUALITY_METRICS:
        out[name] = bootstrap_ci(_values(answered, name), n_resamples=n_resamples, seed=seed)
    out["citation_valid"] = bootstrap_ci(
        [float(r.citation_valid) for r in answered], n_resamples=n_resamples, seed=seed
    )
    unans = [float(not r.answerable) for r in ok]
    refused = [float(r.refused) for r in ok]
    tp = [u * f for u, f in zip(unans, refused, strict=True)]
    out["refusal_precision"] = bootstrap_ratio_ci(tp, refused, n_resamples=n_resamples, seed=seed)
    out["refusal_recall"] = bootstrap_ratio_ci(tp, unans, n_resamples=n_resamples, seed=seed)
    answerable = [r for r in ok if r.answerable]
    out["answer_rate"] = bootstrap_ci(
        [float(not r.refused) for r in answerable], n_resamples=n_resamples, seed=seed
    )
    lat = [
        r.latency_ms.get("total", 0.0) for r in ok if not r.cached and "generate" in r.latency_ms
    ]
    if lat:
        out["latency_ms_p50"] = Estimate(
            mean=float(np.percentile(lat, 50)), low=float("nan"), high=float("nan"), n=len(lat)
        )
        out["latency_ms_p95"] = Estimate(
            mean=float(np.percentile(lat, 95)), low=float("nan"), high=float("nan"), n=len(lat)
        )
    fresh = [r for r in ok if not r.cached]  # cached answers cost nothing new
    out["cost_usd_total"] = Estimate(
        mean=float(sum(r.cost_usd for r in fresh)),
        low=float("nan"),
        high=float("nan"),
        n=len(fresh),
    )
    out["tokens_per_answer"] = bootstrap_ci(
        [float(r.input_tokens + r.output_tokens) for r in ok if r.model],
        n_resamples=n_resamples,
        seed=seed,
    )
    return out


def refusal_counts(records: Sequence[GenerationRecord]) -> dict[str, int]:
    tp = sum(1 for r in records if r.refused and not r.answerable)
    fp = sum(1 for r in records if r.refused and r.answerable)
    fn = sum(1 for r in records if not r.refused and not r.answerable)
    tn = sum(1 for r in records if not r.refused and r.answerable)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
