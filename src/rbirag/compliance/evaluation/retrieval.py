"""Does search find the notifications a clause should be judged against?

Retrieval-only (no LLM): for every labelled clause with gold citations, search with the
clause text and measure how many gold citation groups appear in the top k hits, per
retrieval mode and with / without the entity filter. Used on the dev split to choose the
search configuration of the agent; reported on test for completeness.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from rbirag.compliance.evaluation.dataset import EvalItem
from rbirag.compliance.evaluation.metrics import retrieval_recall_at_k
from rbirag.compliance.regulations import RegulationStore
from rbirag.eval.stats import Estimate, bootstrap_ci

KS = (1, 3, 5, 10)


class RetrievalRow(BaseModel):
    config: str  # e.g. "fixed / bm25 / entity-filter"
    n: int
    recall: dict[str, Estimate]  # "R@5" -> estimate over clauses
    mrr: float  # mean reciprocal rank of the first hit from any gold group
    p50_ms: float


def evaluate_retrieval(
    store: RegulationStore,
    items: Sequence[EvalItem],
    *,
    config: str,
    entity_filter: bool,
    n_resamples: int = 1000,
) -> RetrievalRow:
    gold_items = [it for it in items if it.label.citations]
    per_k: dict[int, list[float]] = {k: [] for k in KS}
    rr: list[float] = []
    times: list[float] = []
    for it in gold_items:
        result = store.search(it.label.text, top_k=max(KS), entity_filter=entity_filter)
        times.append(result.timings_ms.get("total", 0.0))
        ids = [h.doc_id for h in result.hits]
        for k in KS:
            per_k[k].append(retrieval_recall_at_k(ids, it.label.citations, k))
        gold = it.label.gold_doc_ids
        rank = next((i for i, d in enumerate(ids, start=1) if d in gold), None)
        rr.append(1.0 / rank if rank else 0.0)
    return RetrievalRow(
        config=config,
        n=len(gold_items),
        recall={f"R@{k}": bootstrap_ci(v, n_resamples=n_resamples) for k, v in per_k.items()},
        mrr=float(np.mean(rr)) if rr else float("nan"),
        p50_ms=float(np.median(times)) if times else float("nan"),
    )


def retrieval_table(rows: Sequence[RetrievalRow]) -> str:
    head = "| config | n | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | p50 ms |"
    lines = [head, "|---|---|" + "---|" * len(KS) + "---|---|"]
    for r in rows:
        cells = " | ".join(r.recall[f"R@{k}"].fmt(2) for k in KS)
        lines.append(f"| {r.config} | {r.n} | {cells} | {r.mrr:.3f} | {r.p50_ms:.0f} |")
    return "\n".join(lines)
