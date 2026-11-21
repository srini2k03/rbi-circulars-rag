"""Metrics for the compliance checker, with bootstrap confidence intervals.

Per clause:

* **verdict**: accuracy and macro-F1 over the five verdicts (``insufficient_evidence`` is
  a class: abstaining correctly is a correct answer);
* **citations**: precision = cited notifications that are gold or acceptable / cited
  (micro, over all clauses); recall = gold citation groups covered / groups (mean over
  clauses that have gold citations);
* **tools**: argument validity (valid calls / calls), fetch grounding (fetched ids that an
  earlier search showed / fetches), and tool-selection accuracy (the clause's trajectory
  used every expected tool, called ``compare_clause`` before any substantive verdict, and
  stayed within the step budget);
* **task success**: correct verdict, every gold citation group covered, no invalid tool
  call, no error;
* **cost**: steps, LLM calls, tokens, USD and latency per clause and per policy document;
* **calibration**: expected calibration error and accuracy-vs-coverage of the confidence.

Intervals are percentile bootstraps over clauses (seeded); per-document numbers resample
documents.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field

from rbirag.compliance.entities import OTHER_ENTITIES, entity_of
from rbirag.compliance.evaluation.dataset import EvalItem
from rbirag.compliance.records import (
    SUBSTANTIVE_VERDICTS,
    VERDICTS,
    ClausePrediction,
)
from rbirag.eval.stats import Estimate, bootstrap_ci, bootstrap_ratio_ci

DEFAULT_MAX_STEPS = 8

FailureType = str
FAILURE_TYPES: tuple[str, ...] = (
    "run_error",  # the run crashed or timed out
    "unsupported_verdict",  # gold insufficient_evidence; the system gave a verdict anyway
    "over_abstention",  # the system abstained although the corpus settles the clause
    "scope_error",  # not_applicable confused with an applicable verdict
    "wrong_entity",  # wrong verdict while citing another entity type's Directions
    "outdated_rule_missed",  # accepted a clause that follows a replaced rule
    "missed_issue",  # gold conflict/gap; predicted compliant
    "false_alarm",  # gold compliant; predicted conflict/gap
    "conflict_gap_confusion",  # conflict predicted as gap or vice versa
    "citation_miss",  # verdict right, a gold citation group not cited
    "invalid_tool_call",  # verdict and citations right, but a tool call was invalid
    "step_budget",  # otherwise right, but over the step budget
)


class ItemScore(BaseModel):
    item_id: str
    policy_id: str
    split: str
    tags: list[str]
    gold: str
    pred: str
    correct: bool
    cited: int
    cited_allowed: int
    groups: int
    groups_hit: int
    calls: int
    valid_calls: int
    fetches: int
    grounded_fetches: int
    tool_selection_ok: bool
    success: bool
    confidence: float | None
    steps: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    error: bool
    wrong_entity_citations: int
    failure: str | None = None

    @property
    def citation_recall(self) -> float | None:
        return self.groups_hit / self.groups if self.groups else None


def score_item(
    item: EvalItem,
    pred: ClausePrediction,
    *,
    titles: Mapping[str, str] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ItemScore:
    label = item.label
    cited = list(dict.fromkeys(pred.citations))  # de-duplicated, order kept
    allowed = label.allowed_doc_ids
    groups_hit = sum(1 for group in label.citations if set(group) & set(cited))
    calls = pred.tool_calls
    seen: set[str] = set()
    fetches = grounded = 0
    for call in calls:
        if call.name == "fetch_notification":
            fetches += 1
            target = call.result_doc_ids[0] if call.result_doc_ids else None
            if target is not None and target in seen:
                grounded += 1
        if call.name == "search_regulations":
            seen.update(call.result_doc_ids)
    used = {c.name for c in calls if c.valid}
    compared = any(c.name == "compare_clause" and c.valid for c in calls)
    tool_selection_ok = (
        set(label.expected_tools) <= used
        and (pred.verdict not in SUBSTANTIVE_VERDICTS or compared)
        and len(calls) <= max_steps
    )
    titles = titles or {}
    wrong_entity = sum(1 for d in cited if entity_of(titles.get(d)) in OTHER_ENTITIES)
    correct = pred.verdict == label.verdict and pred.error is None
    valid_calls = sum(1 for c in calls if c.valid)
    success = (
        correct
        and groups_hit == len(label.citations)
        and valid_calls == len(calls)
        and pred.error is None
    )
    score = ItemScore(
        item_id=item.item_id,
        policy_id=label.policy_id,
        split=item.split,
        tags=list(label.tags),
        gold=label.verdict,
        pred=pred.verdict,
        correct=correct,
        cited=len(cited),
        cited_allowed=sum(1 for d in cited if d in allowed),
        groups=len(label.citations),
        groups_hit=groups_hit,
        calls=len(calls),
        valid_calls=valid_calls,
        fetches=fetches,
        grounded_fetches=grounded,
        tool_selection_ok=tool_selection_ok,
        success=success,
        confidence=pred.confidence,
        steps=len(calls),
        llm_calls=pred.llm_calls,
        input_tokens=pred.input_tokens,
        output_tokens=pred.output_tokens,
        cost_usd=pred.cost_usd,
        latency_ms=pred.latency_ms,
        error=pred.error is not None,
        wrong_entity_citations=wrong_entity,
    )
    score.failure = classify_failure(score, max_steps=max_steps)
    return score


def classify_failure(score: ItemScore, *, max_steps: int = DEFAULT_MAX_STEPS) -> str | None:
    """The first matching failure type, or ``None`` for a successful clause."""
    if score.error:
        return "run_error"
    gold, pred = score.gold, score.pred
    if gold != pred:
        if gold == "insufficient_evidence":
            return "unsupported_verdict"
        if pred == "insufficient_evidence":
            return "over_abstention"
        if "not_applicable" in (gold, pred):
            return "scope_error"
        if score.wrong_entity_citations:
            return "wrong_entity"
        if gold in {"conflict", "gap"} and pred == "compliant":
            return "outdated_rule_missed" if "outdated_rule" in score.tags else "missed_issue"
        if gold == "compliant":
            return "false_alarm"
        return "conflict_gap_confusion"
    if score.groups_hit < score.groups:
        return "citation_miss"
    if score.valid_calls < score.calls:
        return "invalid_tool_call"
    if score.steps > max_steps:
        return "step_budget"
    return None


# ----------------------------------------------------------------------------- statistics


def _bootstrap_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_statistic(
    n: int,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> Estimate:
    """Percentile CI of ``statistic(indices)`` for statistics over paired per-item data
    (macro-F1, ECE) that a mean of per-item values cannot express."""
    if n == 0:
        return Estimate(mean=float("nan"), low=float("nan"), high=float("nan"), n=0)
    point = statistic(np.arange(n))
    samples = np.array([statistic(row) for row in _bootstrap_indices(n, n_resamples, seed)])
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return Estimate(mean=point, low=float("nan"), high=float("nan"), n=n)
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Estimate(mean=float(point), low=float(low), high=float(high), n=n)


def per_class(
    golds: Sequence[str], preds: Sequence[str], labels: Sequence[str] = VERDICTS
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c in labels:
        tp = sum(1 for g, p in zip(golds, preds, strict=True) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds, strict=True) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds, strict=True) if g == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def macro_f1(golds: Sequence[str], preds: Sequence[str], labels: Sequence[str] = VERDICTS) -> float:
    """Mean F1 over the classes present in the gold labels (absent classes are skipped so
    a resample without, say, any ``gap`` item does not score 0 for it)."""
    stats = per_class(golds, preds, labels)
    present = [c for c in labels if stats[c]["support"] > 0]
    return float(np.mean([stats[c]["f1"] for c in present])) if present else float("nan")


def confusion_matrix(
    golds: Sequence[str], preds: Sequence[str], labels: Sequence[str] = VERDICTS
) -> dict[str, dict[str, int]]:
    counts = Counter(zip(golds, preds, strict=True))
    return {g: {p: counts[(g, p)] for p in labels} for g in labels}


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> float:
    """Weighted mean |accuracy - confidence| over equal-width confidence bins."""
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
    return float(total)


def risk_coverage(
    confidence: Sequence[float], correct: Sequence[bool]
) -> tuple[list[tuple[float, float]], float]:
    """Accuracy when answering only the most confident fraction of clauses, at every
    coverage level, and the area under the risk-coverage curve (lower is better)."""
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return [], float("nan")
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(corr[order])
    n = np.arange(1, conf.size + 1)
    accuracy = cum / n
    curve = [(float(k / conf.size), float(a)) for k, a in zip(n, accuracy, strict=True)]
    aurc = float(np.mean(1.0 - accuracy))
    return curve, aurc


def paired_bootstrap_difference(
    a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000, seed: int = 13
) -> Estimate:
    """CI of mean(a) - mean(b) for two systems scored on the same items."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci((arr_a - arr_b).tolist(), n_resamples=n_resamples, seed=seed)


# ----------------------------------------------------------------------------- segmentation


class SegmentationScore(BaseModel):
    gold: int
    predicted: int
    matched: int
    precision: float
    recall: float
    f1: float


def _segment_key(text: str) -> str:
    # whitespace-insensitive, and a line break after a hyphen ("low-\nrisk") is not a
    # boundary difference
    return re.sub(r"-\s+", "-", " ".join(text.split()))


def segmentation_score(gold: Sequence[str], predicted: Sequence[str]) -> SegmentationScore:
    """Exact clause matches (whitespace-normalised text), one-to-one."""
    pool = Counter(_segment_key(t) for t in gold)
    matched = 0
    for text in predicted:
        key = _segment_key(text)
        if pool[key] > 0:
            pool[key] -= 1
            matched += 1
    p = matched / len(predicted) if predicted else 0.0
    r = matched / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return SegmentationScore(
        gold=len(gold), predicted=len(predicted), matched=matched, precision=p, recall=r, f1=f1
    )


def retrieval_recall_at_k(
    hit_doc_ids: Sequence[str], groups: Sequence[Sequence[str]], k: int
) -> float:
    """Share of gold citation groups with a member among the first ``k`` hits."""
    if not groups:
        return float("nan")
    top = set(hit_doc_ids[:k])
    return sum(1 for g in groups if top & set(g)) / len(groups)


# ----------------------------------------------------------------------------- report


class Report(BaseModel):
    system: str
    split: str
    n: int
    verdict_accuracy: Estimate
    macro_f1: Estimate
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    citation_precision: Estimate
    citation_recall: Estimate
    tool_argument_validity: Estimate
    fetch_grounding: Estimate
    tool_selection_accuracy: Estimate
    task_success: Estimate
    avg_steps: Estimate
    avg_llm_calls: Estimate
    latency_ms_per_clause: Estimate
    cost_usd_per_clause: Estimate
    tokens_per_clause: Estimate
    latency_ms_per_document: Estimate
    cost_usd_per_document: Estimate
    tokens_per_document: Estimate
    documents: int
    document_success: float = Field(description="Documents whose clauses all ran without error.")
    ece: Estimate | None = None
    aurc: float | None = None
    accuracy_at_coverage: dict[str, float] = Field(default_factory=dict)
    by_tag: dict[str, dict[str, float]] = Field(default_factory=dict)
    failures: dict[str, int] = Field(default_factory=dict)
    errors: int = 0


def summarize(
    scores: Sequence[ItemScore],
    *,
    system: str = "agent",
    split: str = "test",
    n_resamples: int = 1000,
    seed: int = 13,
) -> Report:
    if not scores:
        raise ValueError("no scored items")
    n = len(scores)
    golds = [s.gold for s in scores]
    preds = [s.pred for s in scores]

    def ci(values: Sequence[float]) -> Estimate:
        return bootstrap_ci(values, n_resamples=n_resamples, seed=seed)

    def ratio(num: Sequence[float], den: Sequence[float]) -> Estimate:
        return bootstrap_ratio_ci(num, den, n_resamples=n_resamples, seed=seed)

    with_gold = [s for s in scores if s.groups]
    docs: dict[str, list[ItemScore]] = defaultdict(list)
    for s in scores:
        docs[s.policy_id].append(s)
    doc_latency = [sum(x.latency_ms for x in v) for v in docs.values()]
    doc_cost = [sum(x.cost_usd for x in v) for v in docs.values()]
    doc_tokens = [sum(x.input_tokens + x.output_tokens for x in v) for v in docs.values()]

    conf_scores = [s for s in scores if s.confidence is not None]
    ece: Estimate | None = None
    aurc: float | None = None
    at_cov: dict[str, float] = {}
    if len(conf_scores) == n:
        conf = np.array([s.confidence for s in conf_scores], dtype=np.float64)
        corr = np.array([s.correct for s in conf_scores], dtype=bool)
        ece = bootstrap_statistic(
            n,
            lambda idx: expected_calibration_error(conf[idx], corr[idx]),
            n_resamples=n_resamples,
            seed=seed,
        )
        curve, aurc = risk_coverage(conf.tolist(), corr.tolist())
        for target in (0.5, 0.8, 1.0):
            k = max(1, int(np.ceil(target * n)))
            at_cov[f"{target:.0%}"] = curve[k - 1][1]

    tags: dict[str, list[ItemScore]] = defaultdict(list)
    for s in scores:
        for t in s.tags:
            tags[t].append(s)
    by_tag = {
        t: {
            "n": float(len(v)),
            "accuracy": float(np.mean([x.correct for x in v])),
            "success": float(np.mean([x.success for x in v])),
        }
        for t, v in sorted(tags.items())
    }
    g, p = np.array(golds), np.array(preds)
    return Report(
        system=system,
        split=split,
        n=n,
        verdict_accuracy=ci([float(s.correct) for s in scores]),
        macro_f1=bootstrap_statistic(
            n,
            lambda idx: macro_f1(g[idx].tolist(), p[idx].tolist()),
            n_resamples=n_resamples,
            seed=seed,
        ),
        per_class=per_class(golds, preds),
        confusion=confusion_matrix(golds, preds),
        citation_precision=ratio([s.cited_allowed for s in scores], [s.cited for s in scores]),
        citation_recall=ci([s.groups_hit / s.groups for s in with_gold]),
        tool_argument_validity=ratio([s.valid_calls for s in scores], [s.calls for s in scores]),
        fetch_grounding=ratio([s.grounded_fetches for s in scores], [s.fetches for s in scores]),
        tool_selection_accuracy=ci([float(s.tool_selection_ok) for s in scores]),
        task_success=ci([float(s.success) for s in scores]),
        avg_steps=ci([float(s.steps) for s in scores]),
        avg_llm_calls=ci([float(s.llm_calls) for s in scores]),
        latency_ms_per_clause=ci([s.latency_ms for s in scores]),
        cost_usd_per_clause=ci([s.cost_usd for s in scores]),
        tokens_per_clause=ci([float(s.input_tokens + s.output_tokens) for s in scores]),
        latency_ms_per_document=ci(doc_latency),
        cost_usd_per_document=ci(doc_cost),
        tokens_per_document=ci([float(x) for x in doc_tokens]),
        documents=len(docs),
        document_success=float(np.mean([all(not x.error for x in v) for v in docs.values()])),
        ece=ece,
        aurc=aurc,
        accuracy_at_coverage=at_cov,
        by_tag=by_tag,
        failures=dict(Counter(s.failure for s in scores if s.failure).most_common()),
        errors=sum(1 for s in scores if s.error),
    )


def score_predictions(
    items: Sequence[EvalItem],
    predictions: Mapping[str, ClausePrediction],
    *,
    titles: Mapping[str, str] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[ItemScore]:
    """Score every item; an item with no prediction counts as a run error."""
    scores: list[ItemScore] = []
    for item in items:
        pred = predictions.get(item.item_id) or ClausePrediction(
            item_id=item.item_id, verdict="insufficient_evidence", error="no prediction"
        )
        scores.append(score_item(item, pred, titles=titles, max_steps=max_steps))
    return scores
