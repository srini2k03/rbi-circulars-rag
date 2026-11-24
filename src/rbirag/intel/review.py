"""Precision of the document-intelligence extraction on a reviewed sample."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (well-behaved for small n and p near 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def precision_report(sample_path: Path, review_path: Path) -> dict[str, Any]:
    sample = {x["id"]: x for x in _read(sample_path)}
    verdicts = [v for v in _read(review_path) if v["id"] in sample]
    reviewers = sorted({v.get("reviewer", "unknown") for v in verdicts})
    out: dict[str, Any] = {
        "reviewed": len(verdicts),
        "sample_size": len(sample),
        "reviewers": reviewers,
        "human_verified": any(not r.startswith("ai-") for r in reviewers),
        "label": "AI-checked (LLM-assisted), not human-verified"
        if all(r.startswith("ai-") for r in reviewers)
        else "reviewed",
    }
    groups: dict[str, dict[str, list[bool]]] = {}
    for v in verdicts:
        item = sample[v["id"]]
        key = item["type"] if item["kind"] == "relation" else "effective_date"
        g = groups.setdefault(key, {"target": [], "type": [], "both": []})
        if item["kind"] == "relation":
            g["target"].append(bool(v["target_correct"]))
            g["type"].append(bool(v["type_correct"]))
            g["both"].append(bool(v["target_correct"]) and bool(v["type_correct"]))
        else:
            g["both"].append(bool(v["date_correct"]))
    for key, g in sorted(groups.items()):
        row: dict[str, Any] = {}
        for field, vals in g.items():
            if vals:
                lo, hi = wilson(sum(vals), len(vals))
                row[field] = {"precision": sum(vals) / len(vals), "n": len(vals),
                              "ci95": [round(lo, 3), round(hi, 3)]}  # fmt: skip
        out[key] = row
    meta_path = sample_path.with_suffix(".meta.json")
    population = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    counts = population.get("resolved_relations_by_type", {})
    weighted = [(counts[t], groups[t]["both"]) for t in counts if t in groups and groups[t]["both"]]
    if weighted:
        total = sum(n for n, _ in weighted)
        out["population_weighted_precision"] = {
            "precision": sum(n * (sum(v) / len(v)) for n, v in weighted) / total,
            "weights": {t: counts[t] for t in counts if t in groups},
            "note": "per-type precision weighted by how often each type occurs among the "
            "resolved relations (the sample is stratified by type, not proportional)",
        }
    rel = [v for v in verdicts if sample[v["id"]]["kind"] == "relation"]
    if rel:
        both = [bool(v["target_correct"]) and bool(v["type_correct"]) for v in rel]
        lo, hi = wilson(sum(both), len(both))
        out["all_relations"] = {"precision": sum(both) / len(both), "n": len(both),
                                "ci95": [round(lo, 3), round(hi, 3)]}  # fmt: skip
    return out
