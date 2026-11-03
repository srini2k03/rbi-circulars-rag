"""Tables (CSV + Markdown) and charts (matplotlib) for the evaluation results."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rbirag.eval.calibrate import Calibration
from rbirag.eval.dataset import QAPair
from rbirag.eval.gen_metrics import GenerationRecord, refusal_counts, summarize
from rbirag.eval.stats import Estimate


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_ci(row: dict[str, Any], key: str) -> str:
    ci = row.get(f"{key}_ci")
    if ci and all(isinstance(x, (int, float)) and not math.isnan(x) for x in ci):
        return f"{_fmt(row.get(key))} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return _fmt(row.get(key))


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    k: ";".join(f"{x:.6g}" for x in v) if isinstance(v, list) else v
                    for k, v in r.items()
                }
            )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- retrieval


def retrieval_markdown(rows: Sequence[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda r: -(r.get("chunk_ndcg@10") or 0.0))
    headers = ["config", "chunks", "doc R@5", "chunk R@1", "chunk R@5", "chunk R@10",
               "chunk MRR@10", "chunk nDCG@10 [95% CI]", "p50 ms"]  # fmt: skip
    body = [
        [
            r["config"],
            str(r.get("n_chunks", "-")),
            _fmt(r.get("doc_recall@5")),
            _fmt(r.get("chunk_recall@1")),
            _fmt(r.get("chunk_recall@5")),
            _fmt(r.get("chunk_recall@10")),
            _fmt(r.get("chunk_mrr@10")),
            _fmt_ci(r, "chunk_ndcg@10"),
            _fmt(r.get("latency_ms_p50"), 1),
        ]
        for r in ranked
    ]
    return markdown_table(headers, body)


# ----------------------------------------------------------------------------- generation


def generation_rows(
    records: Sequence[GenerationRecord], *, n_resamples: int = 1000
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[GenerationRecord]] = {}
    for r in records:
        groups.setdefault((r.config, r.backend), []).append(r)
    rows: list[dict[str, Any]] = []
    for (config, backend), recs in sorted(groups.items()):
        summary: dict[str, Estimate] = summarize(recs, n_resamples=n_resamples)
        row: dict[str, Any] = {
            "config": config,
            "backend": backend,
            "model": next((r.model for r in recs if r.model), None),
            "judge_model": next((r.judge_model for r in recs if r.judge_model), None),
            "n": len(recs),
            "n_answerable": sum(r.answerable for r in recs),
            **{f"refusal_{k}": v for k, v in refusal_counts(recs).items()},
            "input_tokens": sum(r.input_tokens for r in recs),
            "output_tokens": sum(r.output_tokens for r in recs),
            "judge_tokens": sum(r.judge_input_tokens + r.judge_output_tokens for r in recs),
        }
        for name, est in summary.items():
            row[name] = est.mean
            row[f"{name}_ci"] = [est.low, est.high]
        rows.append(row)
    return rows


def generation_markdown(rows: Sequence[dict[str, Any]]) -> str:
    headers = ["config", "backend (model)", "n", "faithfulness (judge)", "faithfulness (NLI)",
               "relevance (judge)", "citation acc. (NLI)", "citations valid", "refusal P",
               "refusal R", "p50 ms", "cost $"]  # fmt: skip
    body = [
        [
            r["config"],
            f"{r['backend']} ({r.get('model') or '-'})",
            str(r["n"]),
            _fmt_ci(r, "judge_faithfulness"),
            _fmt_ci(r, "nli_faithfulness"),
            _fmt_ci(r, "judge_relevance"),
            _fmt_ci(r, "citation_accuracy"),
            _fmt(r.get("citation_valid")),
            _fmt_ci(r, "refusal_precision"),
            _fmt_ci(r, "refusal_recall"),
            _fmt(r.get("latency_ms_p50"), 0),
            _fmt(r.get("cost_usd_total"), 4),
        ]
        for r in rows
    ]
    return markdown_table(headers, body)


# ----------------------------------------------------------------------------- charts


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def chart_recall_at_k(rows: Sequence[dict[str, Any]], path: Path, top: int = 6) -> None:
    plt = _plt()
    ranked = sorted(rows, key=lambda r: -(r.get("chunk_ndcg@10") or 0.0))[:top]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = [1, 3, 5, 10]
    for r in ranked:
        ax.plot(ks, [r.get(f"chunk_recall@{k}") or 0 for k in ks], marker="o", label=r["config"])
    ax.set(
        xlabel="k",
        ylabel="chunk-level Recall@k",
        title="Recall@k (top configs)",
        xticks=ks,
        ylim=(0, 1),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_mrr_by_config(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    ranked = sorted(rows, key=lambda r: -(r.get("chunk_mrr@10") or 0.0))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(ranked))))
    names = [r["config"] for r in ranked][::-1]
    vals = [r.get("chunk_mrr@10") or 0 for r in ranked][::-1]
    errs = []
    for r in ranked[::-1]:
        ci = r.get("chunk_mrr@10_ci") or [None, None]
        v = r.get("chunk_mrr@10") or 0
        errs.append([v - ci[0] if ci[0] is not None else 0, ci[1] - v if ci[1] is not None else 0])
    ax.barh(names, vals, xerr=list(zip(*errs, strict=True)) if errs else None, color="#4c78a8")
    ax.set(xlabel="chunk-level MRR@10 (95% CI)", title="MRR@10 by config", xlim=(0, 1))
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_size_sweep(
    rows: Sequence[dict[str, Any]], path: Path, mode: str = "hybrid_rerank"
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    sel = [r for r in rows if r["mode"] == mode] or list(rows)
    for chunker in sorted({r["chunker"] for r in sel}):
        pts = sorted(
            (r["size"], r.get("chunk_ndcg@10") or 0) for r in sel if r["chunker"] == chunker
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=chunker)
    ax.set(xscale="log", xlabel="chunk size (tokens)", ylabel="chunk-level nDCG@10",
           title=f"Chunk-size sweep ({mode})")  # fmt: skip
    ax.set_xticks([256, 512, 1024], labels=["256", "512", "1024"])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_latency_quality(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for mode in sorted({r["mode"] for r in rows}):
        pts = [
            (r.get("latency_ms_p50") or 0, r.get("chunk_ndcg@10") or 0)
            for r in rows
            if r["mode"] == mode
        ]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=mode, alpha=0.8)
    ax.set(xscale="log", xlabel="retrieval latency p50 (ms, CPU)", ylabel="chunk-level nDCG@10",
           title="Latency vs quality")  # fmt: skip
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_calibration(cal: Calibration, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    t = [p.threshold for p in cal.curve]
    ax.plot(t, [p.refusal_precision for p in cal.curve], label="refusal precision")
    ax.plot(t, [p.refusal_recall for p in cal.curve], label="refusal recall")
    ax.plot(t, [p.answerable_coverage for p in cal.curve], label="answerable coverage")
    ax.axvline(cal.threshold, color="k", ls="--", lw=1, label=f"tau = {cal.threshold:.2f}")
    ax.set(xlabel="gate threshold tau on P(relevant)", ylim=(0, 1.02),
           title=f"Confidence-gate calibration (dev, n={cal.n_dev})")  # fmt: skip
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- metadata


def git_sha() -> str | None:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(qa: Sequence[QAPair], **extra: Any) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "questions": len(qa),
        "answerable": sum(p.answerable for p in qa),
        "unanswerable": sum(not p.answerable for p in qa),
        "human_verified": sum(p.verified for p in qa),
        "unverified_model_generated": sum(not p.verified for p in qa),
        **extra,
    }


def build_report(
    results_dir: Path,
    *,
    retrieval_rows: Sequence[dict[str, Any]],
    generation_records: Sequence[GenerationRecord],
    calibration: Calibration | None,
    metadata: dict[str, Any],
    n_resamples: int = 1000,
) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    charts = results_dir / "charts"
    charts.mkdir(exist_ok=True)
    written: list[Path] = []
    md_parts = [
        "# Evaluation results\n",
        f"Generated {metadata.get('generated_at')} at commit `{metadata.get('commit')}`. "
        f"Questions: {metadata.get('questions')} ({metadata.get('answerable')} answerable, "
        f"{metadata.get('unanswerable')} unanswerable); human-verified: "
        f"{metadata.get('human_verified')}. Judge: {metadata.get('judge_model')}.\n",
    ]
    if retrieval_rows:
        write_csv(retrieval_rows, results_dir / "retrieval_ablation.csv")
        md = retrieval_markdown(retrieval_rows)
        (results_dir / "retrieval_ablation.md").write_text(md, encoding="utf-8")
        md_parts += ["## Retrieval ablation\n", md]
        for fn, name in (
            (chart_recall_at_k, "recall_at_k.png"),
            (chart_mrr_by_config, "mrr_by_config.png"),
            (chart_size_sweep, "chunk_size_sweep.png"),
            (chart_latency_quality, "latency_vs_quality.png"),
        ):
            fn(retrieval_rows, charts / name)
            written.append(charts / name)
        written += [results_dir / "retrieval_ablation.csv", results_dir / "retrieval_ablation.md"]
    if generation_records:
        rows = generation_rows(generation_records, n_resamples=n_resamples)
        write_csv(rows, results_dir / "generation.csv")
        md = generation_markdown(rows)
        (results_dir / "generation.md").write_text(md, encoding="utf-8")
        md_parts += ["## Generation\n", md]
        written += [results_dir / "generation.csv", results_dir / "generation.md"]
    if calibration is not None:
        (results_dir / "calibration.json").write_text(
            calibration.model_dump_json(indent=2), encoding="utf-8"
        )
        chart_calibration(calibration, charts / "calibration.png")
        chosen = calibration.chosen
        md_parts.append(
            f"## Confidence gate\n\nCalibrated tau = {calibration.threshold:.2f} on "
            f"{calibration.n_dev} dev questions: refusal precision "
            f"{chosen.refusal_precision:.3f}, recall {chosen.refusal_recall:.3f}, "
            f"answerable coverage {chosen.answerable_coverage:.3f}.\n"
        )
        written += [results_dir / "calibration.json", charts / "calibration.png"]
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (results_dir / "README.md").write_text("\n".join(md_parts), encoding="utf-8")
    written += [results_dir / "run_metadata.json", results_dir / "README.md"]
    return written
