"""Agreement between two LLM judges on the answers both have judged.

Every record keeps each judge's verdict in ``verdicts`` (see ``eval.records``), so when
the reported judge changes, the earlier judge's verdicts remain available to check how
far the two agree:

* faithfulness (share of supported claims, numeric in [0, 1]): Pearson and Spearman
  correlation, mean absolute difference, and agreement / Cohen's kappa on the binary
  "fully supported" label (faithfulness == 1);
* relevance (a 1-5 rating, i.e. categorical): exact and within-one-point agreement,
  Cohen's kappa (unweighted and quadratic-weighted) and Spearman correlation.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from rbirag.eval.gen_metrics import GenerationRecord
from rbirag.eval.records import all_verdicts


def cohen_kappa(
    a: Sequence[Hashable], b: Sequence[Hashable], *, weights: str | None = None
) -> float | None:
    """Cohen's kappa for two raters' labels. ``weights="quadratic"`` treats the labels as
    ordered numbers. None when undefined (no items, or chance agreement is total)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    if not a:
        return None
    labels = sorted(set(a) | set(b), key=lambda x: (str(type(x)), x))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    if weights is None:
        w = 1.0 - np.eye(k)
    elif weights == "quadratic":
        vals = np.array([float(lab) for lab in labels])  # type: ignore[arg-type]
        span = (vals.max() - vals.min()) or 1.0
        w = ((vals[:, None] - vals[None, :]) / span) ** 2
    else:
        raise ValueError(f"unknown weights: {weights!r}")
    denom = float((w * exp).sum())
    if denom == 0:
        return None
    return float(1.0 - (w * obs).sum() / denom)


def _ranks(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # average ranks for ties
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3:
        return None
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3:
        return None
    return pearson(
        list(_ranks(np.asarray(a, dtype=float))), list(_ranks(np.asarray(b, dtype=float)))
    )


def _r(v: float | None, digits: int = 3) -> float | None:
    return None if v is None else round(v, digits)


def judge_agreement(records: Sequence[GenerationRecord], judge: str, other: str) -> dict[str, Any]:
    """Agreement of ``other`` with ``judge`` on the records both have judged."""
    f_a: list[float] = []
    f_b: list[float] = []
    r_a: list[int] = []
    r_b: list[int] = []
    n = 0
    for rec in records:
        v = all_verdicts(rec)
        if judge not in v or other not in v:
            continue
        if not any(x is not None for x in v[judge].values()) or not any(
            x is not None for x in v[other].values()
        ):
            continue  # a failed attempt by either judge
        n += 1
        fa, fb = v[judge].get("faithfulness"), v[other].get("faithfulness")
        if fa is not None and fb is not None:
            f_a.append(fa)
            f_b.append(fb)
        ra, rb = v[judge].get("relevance"), v[other].get("relevance")
        if ra is not None and rb is not None:
            r_a.append(round(ra * 4) + 1)  # back to the judge's 1-5 rating
            r_b.append(round(rb * 4) + 1)
    full_a = [x >= 1.0 for x in f_a]
    full_b = [x >= 1.0 for x in f_b]
    return {
        "judge": judge,
        "other": other,
        "n": n,
        "faithfulness": {
            "n": len(f_a),
            "mean_judge": _r(float(np.mean(f_a)) if f_a else None),
            "mean_other": _r(float(np.mean(f_b)) if f_b else None),
            "mean_abs_diff": _r(float(np.mean(np.abs(np.subtract(f_a, f_b)))) if f_a else None),
            "pearson": _r(pearson(f_a, f_b)),
            "spearman": _r(spearman(f_a, f_b)),
            "fully_supported_agreement": _r(
                sum(x == y for x, y in zip(full_a, full_b, strict=True)) / len(f_a) if f_a else None
            ),
            "fully_supported_kappa": _r(cohen_kappa(full_a, full_b)),
        },
        "relevance": {
            "n": len(r_a),
            "exact_agreement": _r(
                sum(x == y for x, y in zip(r_a, r_b, strict=True)) / len(r_a) if r_a else None
            ),
            "within_one_agreement": _r(
                sum(abs(x - y) <= 1 for x, y in zip(r_a, r_b, strict=True)) / len(r_a)
                if r_a
                else None
            ),
            "kappa": _r(cohen_kappa(r_a, r_b)),
            "kappa_quadratic": _r(cohen_kappa(r_a, r_b, weights="quadratic")),
            "spearman": _r(spearman([float(x) for x in r_a], [float(x) for x in r_b])),
        },
    }


def agreement_report(records: Sequence[GenerationRecord], judge: str) -> list[dict[str, Any]]:
    """Agreement of every other judge found in the records with ``judge`` (most overlap
    first); judges with no overlapping item are left out."""
    others: dict[str, int] = {}
    for rec in records:
        v = all_verdicts(rec)
        if judge in v:
            for m in v:
                if m != judge:
                    others[m] = others.get(m, 0) + 1
    return [
        judge_agreement(records, judge, m)
        for m, _ in sorted(others.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def agreement_markdown(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    head = (
        "| other judge | n | faithfulness r (Spearman) | mean abs diff | fully-supported "
        "agreement | kappa | relevance exact | within 1 | kappa | weighted kappa |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        f, rel = r["faithfulness"], r["relevance"]
        body += (
            f"| {r['other']} | {r['n']} | {_fmt(f['pearson'])} ({_fmt(f['spearman'])}) | "
            f"{_fmt(f['mean_abs_diff'])} | {_fmt(f['fully_supported_agreement'])} | "
            f"{_fmt(f['fully_supported_kappa'])} | {_fmt(rel['exact_agreement'])} | "
            f"{_fmt(rel['within_one_agreement'])} | {_fmt(rel['kappa'])} | "
            f"{_fmt(rel['kappa_quadratic'])} |\n"
        )
    return head + body
