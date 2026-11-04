"""Tables (CSV + Markdown) and charts (matplotlib) for the evaluation results."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from rbirag.eval.calibrate import Calibration
from rbirag.eval.dataset import QAPair
from rbirag.eval.gen_metrics import GenerationRecord, refusal_counts, summarize
from rbirag.eval.stats import Estimate

if TYPE_CHECKING:
    from rbirag.eval.ablate import RetrievalQueryResult

_T = TypeVar("_T")


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


MIN_SUCCESS_FRACTION = 0.9


def record_succeeded(r: GenerationRecord) -> bool:
    """Answer generated without error and, if it needed judging, judged successfully."""
    if r.error is not None:
        return False
    if r.answerable and not r.refused:
        return r.judge_faithfulness is not None or r.judge_relevance is not None
    return True


def mix(models: Sequence[str | None]) -> str:
    """``"gemini-3.8-flash 70%, gemini-3.7-flash 30%"`` (models that actually answered)."""
    present = [m for m in models if m]
    if not present:
        return "-"
    counts = Counter(present)
    return ", ".join(f"{m} {100 * c / len(present):.0f}%" for m, c in counts.most_common())


def generation_rows(
    records: Sequence[GenerationRecord],
    *,
    planned: Mapping[str, int] | None = None,
    samples: Mapping[str, str] | None = None,
    min_success: float = MIN_SUCCESS_FRACTION,
    n_resamples: int = 1000,
) -> list[dict[str, Any]]:
    """One row per (config, backend). ``status`` is ``complete`` only when at least
    ``min_success`` of the planned questions were answered (and judged where needed);
    otherwise ``incomplete (k/N)``, and the row is kept out of the headline table.

    ``planned`` and ``samples`` (e.g. ``"subset"``) are keyed ``"config/backend"``, or by
    config alone for every backend."""
    groups: dict[tuple[str, str], list[GenerationRecord]] = {}
    for r in records:
        groups.setdefault((r.config, r.backend), []).append(r)
    rows: list[dict[str, Any]] = []
    for (config, backend), recs in sorted(groups.items()):
        summary: dict[str, Estimate] = summarize(recs, n_resamples=n_resamples)
        plan = planned or {}
        n_planned = max(plan.get(f"{config}/{backend}", plan.get(config, 0)), len(recs))
        k = sum(record_succeeded(r) for r in recs)
        measured = n_planned > 0 and k / n_planned >= min_success
        sample = (samples or {}).get(f"{config}/{backend}", (samples or {}).get(config, ""))
        row: dict[str, Any] = {
            "config": config,
            "backend": backend,
            "status": "complete" if measured else f"incomplete ({k}/{n_planned})",
            "sample": f"{sample} (n={n_planned})" if sample else f"n={n_planned}",
            "successful": k,
            "planned": n_planned,
            "model": mix([r.model for r in recs]),
            "judge_model": mix([r.judge_model for r in recs]),
            "n": len(recs),
            "n_answerable": sum(r.answerable for r in recs),
            **{f"refusal_{k}": v for k, v in refusal_counts(recs).items()},
            # token totals count only calls actually made (cached responses add none)
            "n_cached": sum(r.cached for r in recs),
            "input_tokens": sum(r.input_tokens for r in recs if not r.cached),
            "output_tokens": sum(r.output_tokens for r in recs if not r.cached),
            "judge_tokens": sum(
                r.judge_input_tokens + r.judge_output_tokens for r in recs if not r.judge_cached
            ),
        }
        for name, est in summary.items():
            row[name] = est.mean
            row[f"{name}_ci"] = [est.low, est.high]
        rows.append(row)
    return rows


def incomplete_markdown(rows: Sequence[dict[str, Any]]) -> str:
    body = [
        [r["config"], f"{r['backend']} ({r.get('model') or '-'})", r["status"]]
        for r in rows
        if not is_complete(r)
    ]
    if not body:
        return ""
    return (
        "Not reported as results: too many generation or judge calls failed or are still "
        "pending (quota). Re-run the workflow to complete them.\n\n"
        + markdown_table(["config", "backend (model)", "status"], body)
    )


def is_complete(row: Mapping[str, Any]) -> bool:
    """Headline-ready row (``measured`` is the label used before ``complete``)."""
    return row.get("status", "complete") in ("complete", "measured")


def judge_label(row: Mapping[str, Any]) -> str:
    return str(row.get("judge_model") or "-").replace(" 100%", "")


def generation_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Headline table: complete rows only."""
    rows = [r for r in rows if is_complete(r)]
    headers = ["config", "backend (model)", "questions", "n", "judge", "faithfulness (judge)",
               "faithfulness (NLI)",
               "relevance (judge)", "citation acc. (NLI)", "citations valid", "refusal P",
               "refusal R", "p50 ms", "cost $"]  # fmt: skip
    body = [
        [
            r["config"],
            f"{r['backend']} ({r.get('model') or '-'})",
            str(r.get("sample") or "-"),
            str(r["n"]),
            judge_label(r),
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


def is_reviewed_set(source: str | None) -> bool:
    return bool(source) and Path(str(source)).name == "qa_v1.jsonl"


def human_verified(source: str | None, qa: Sequence[QAPair]) -> int:
    """Human verification only exists in the reviewed set; candidates count as 0."""
    return sum(p.verified for p in qa) if source is None or is_reviewed_set(source) else 0


def ai_reviewed(qa: Sequence[QAPair]) -> int:
    """Pairs reviewed (accepted or edited) by the AI assistant. Not human verification."""
    return sum(p.ai_review is not None for p in qa)


def describe_test_set(source: str | None, qa: Sequence[QAPair], ai_rejected: int = 0) -> str:
    """How the results README describes the test set that was used, e.g.
    ``model-generated test set (`qa_candidates.jsonl`): 116 model-generated pairs
    (43 AI-reviewed by Claude, 0 human-verified); 7 AI-rejected pairs excluded``."""
    kind = "reviewed test set" if is_reviewed_set(source) else "model-generated test set"
    text = (
        f"{kind} (`{source or 'qa_candidates.jsonl'}`): {len(qa)} model-generated pairs "
        f"({ai_reviewed(qa)} AI-reviewed by Claude, {human_verified(source, qa)} human-verified)"
    )
    if ai_rejected:
        text += f"; {ai_rejected} AI-rejected pairs excluded"
    return text


def run_metadata(
    qa: Sequence[QAPair], test_set: str | None = None, ai_rejected: int = 0, **extra: Any
) -> dict[str, Any]:
    verified = human_verified(test_set, qa)
    reviews = Counter(p.ai_review.decision for p in qa if p.ai_review is not None)
    return {
        "test_set_file": test_set,
        "test_set": describe_test_set(test_set, qa, ai_rejected),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "questions": len(qa),
        "answerable": sum(p.answerable for p in qa),
        "unanswerable": sum(not p.answerable for p in qa),
        "human_verified": verified,
        "ai_review": {
            "reviewer": "ai-assistant (Claude); an AI review, not human verification",
            "reviewed_in_test_set": ai_reviewed(qa),
            "accepted": reviews["accept"],
            "edited": reviews["edit"],
            "rejected_excluded": ai_rejected,
        },
        "unverified_model_generated": len(qa) - verified,
        **extra,
    }


def current_items(
    items: Sequence[_T], questions: Mapping[str, str], edited: AbstractSet[str] = frozenset()
) -> list[_T]:
    """Per-question results that still describe a pair in the test set: the pair was not
    dropped (e.g. AI-rejected) and was run with its current wording. Results without a
    recorded question are kept unless the pair has been edited since."""
    out: list[_T] = []
    for it in items:
        q = questions.get(str(getattr(it, "qa_id", "")))
        if q is None:
            continue
        ran = getattr(it, "question", None)
        if (ran is not None and ran != q) or (ran is None and getattr(it, "qa_id", "") in edited):
            continue
        out.append(it)
    return out


def resummarize_retrieval(
    rows: Sequence[dict[str, Any]],
    per_query: Sequence[RetrievalQueryResult],
    questions: Mapping[str, str],
    *,
    edited: AbstractSet[str] = frozenset(),
    n_resamples: int = 1000,
) -> list[dict[str, Any]]:
    """Retrieval rows recomputed from per-query results restricted to ``questions``.

    A row whose config has no remaining per-query results is dropped."""
    from rbirag.eval.ablate import RetrievalConfig, summarize_retrieval

    by_config: dict[str, list[RetrievalQueryResult]] = {}
    for it in current_items(per_query, questions, edited):
        by_config.setdefault(it.config, []).append(it)
    out: list[dict[str, Any]] = []
    for row in rows:
        items = by_config.get(row["config"])
        if not items:
            continue
        new = summarize_retrieval(
            RetrievalConfig.parse(row["config"]),
            items,
            n_chunks=int(row.get("n_chunks") or 0),
            n_resamples=n_resamples,
        )
        if new["n_queries"]:
            out.append(new)
    return out


def build_report(
    results_dir: Path,
    *,
    retrieval_rows: Sequence[dict[str, Any]],
    generation_records: Sequence[GenerationRecord],
    calibration: Calibration | None,
    metadata: dict[str, Any],
    planned: Mapping[str, int] | None = None,
    samples: Mapping[str, str] | None = None,
    n_resamples: int = 1000,
    ai_reviewed_retrieval_rows: Sequence[dict[str, Any]] = (),
    ai_reviewed_generation_records: Sequence[GenerationRecord] = (),
    dev_retrieval_rows: Sequence[dict[str, Any]] = (),
) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    charts = results_dir / "charts"
    charts.mkdir(exist_ok=True)
    written: list[Path] = []
    md_parts = [
        "# Evaluation results\n",
        f"**Test set: {metadata.get('test_set') or describe_test_set(None, [])}.**\n",
        f"Generated {metadata.get('generated_at')} at commit `{metadata.get('commit')}`. "
        f"Questions: {metadata.get('questions')} ({metadata.get('answerable')} answerable, "
        f"{metadata.get('unanswerable')} unanswerable); human-verified: "
        f"{metadata.get('human_verified')}; AI-reviewed by Claude (not human verification): "
        f"{(metadata.get('ai_review') or {}).get('reviewed_in_test_set', 0)}, with "
        f"{(metadata.get('ai_review') or {}).get('rejected_excluded', 0)} AI-rejected pairs "
        "excluded.\n",
        split_line(metadata),
        "Models that actually answered (share of calls):\n",
        *[f"- {role}: {desc}" for role, desc in (metadata.get("model_mix") or {}).items()],
        "",
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
    if dev_retrieval_rows:
        dev_csv = results_dir / "retrieval_ablation_dev.csv"
        write_csv(dev_retrieval_rows, dev_csv)
        md_parts += [
            "## Dev split (selection only, not a result)\n",
            "The configuration for generation and the gate threshold were chosen on these "
            "dev questions; they are shown for transparency and are not reported results.\n",
            retrieval_markdown(dev_retrieval_rows),
        ]
        written.append(dev_csv)
    if generation_records:
        rows = generation_rows(
            generation_records, planned=planned, samples=samples, n_resamples=n_resamples
        )
        write_csv(rows, results_dir / "generation.csv")
        measured = [r for r in rows if is_complete(r)]
        md = judge_line(metadata) + "\n\n"
        md += generation_markdown(rows) if measured else "No configuration is complete yet.\n"
        incomplete = incomplete_markdown(rows)
        if incomplete:
            md += "\n### Incomplete runs\n\n" + incomplete
        (results_dir / "generation.md").write_text(md, encoding="utf-8")
        md_parts += ["## Generation\n", md]
        written += [results_dir / "generation.csv", results_dir / "generation.md"]
    agreement = metadata.get("judge_agreement") or []
    if agreement:
        from rbirag.eval.agreement import agreement_markdown

        (results_dir / "judge_agreement.json").write_text(
            json.dumps(agreement, indent=2), encoding="utf-8"
        )
        md_parts += [
            "## Judge agreement\n",
            f"Agreement of earlier API judges with the reported judge "
            f"({(metadata.get('judge') or {}).get('model', '-')}) on the answers both "
            "judged (a side analysis; the reported numbers use only the reported judge):\n",
            agreement_markdown(agreement),
        ]
        written.append(results_dir / "judge_agreement.json")
    limitations = metadata.get("limitations") or []
    if limitations:
        md_parts += ["## Limitations\n", *[f"- {x}" for x in limitations], ""]
    n_ai = int((metadata.get("ai_review") or {}).get("reviewed_in_test_set") or 0)
    if n_ai:
        md_parts += [
            f"## AI-reviewed subset ({n_ai} pairs)\n",
            f"The same runs restricted to the {n_ai} pairs an AI assistant (Claude) accepted "
            "or edited after checking them against the source documents (see "
            "`docs/qa_review.md`). This is an AI review, not human verification; edited "
            "pairs use the edited wording.\n",
        ]
        if ai_reviewed_retrieval_rows:
            sub_csv = results_dir / "retrieval_ablation_ai_reviewed.csv"
            write_csv(ai_reviewed_retrieval_rows, sub_csv)
            md_parts += ["### Retrieval\n", retrieval_markdown(ai_reviewed_retrieval_rows)]
            written.append(sub_csv)
        if ai_reviewed_generation_records:
            sub = generation_rows(
                ai_reviewed_generation_records, samples=samples, n_resamples=n_resamples
            )
            write_csv(sub, results_dir / "generation_ai_reviewed.csv")
            sub_md = (
                generation_markdown(sub)
                if any(is_complete(r) for r in sub)
                else "No configuration is complete on this subset yet.\n"
            )
            md_parts += ["### Generation\n", sub_md]
            written.append(results_dir / "generation_ai_reviewed.csv")
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


def judge_line(metadata: Mapping[str, Any]) -> str:
    """Which judge produced the judge columns, and how it agrees with earlier judges."""
    j = metadata.get("judge") or {}
    if not j:
        return f"Judge: {metadata.get('judge_model') or '-'}."
    text = f"**Judge: {j.get('description') or j.get('model')}**"
    speed = j.get("seconds_per_verdict")
    if speed:
        text += f" ({speed:.0f} s per verdict on a CI runner)"
    agreement = metadata.get("judge_agreement") or []
    if agreement:
        parts = []
        for a in agreement:
            f, rel = a["faithfulness"], a["relevance"]
            kappa = rel.get("kappa_quadratic")
            corr = f.get("pearson")
            parts.append(
                f"{a['other']} n={a['n']}: faithfulness r="
                + ("-" if corr is None else f"{corr:.2f}")
                + ", relevance weighted kappa="
                + ("-" if kappa is None else f"{kappa:.2f}")
            )
        text += "; agreement with earlier judges: " + "; ".join(parts)
    return text + ". NLI faithfulness and citation metrics are judge-independent."


def split_line(metadata: dict[str, Any]) -> str:
    sp = metadata.get("split") or {}
    excluded = metadata.get("gold_mismatch_excluded") or []
    note = (
        f"**{len(excluded)} pairs excluded** because their gold spans no longer match the "
        "corpus release in use.\n"
        if excluded
        else ""
    )
    if not sp:
        return note
    return note + (
        f"**Split by document:** {sp.get('test_questions')} test questions "
        f"({sp.get('test_documents')} documents) for every reported number; "
        f"{sp.get('dev_questions')} dev questions ({sp.get('dev_documents')} documents) used "
        "only to choose configurations and calibrate the gate. No document contributes "
        "questions to both.\n"
    )


README_START = "<!-- RESULTS:START (generated by `rbirag eval report --update-readme`) -->"
README_END = "<!-- RESULTS:END -->"


def readme_results_section(
    metadata: dict[str, Any],
    retrieval_rows: Sequence[dict[str, Any]],
    generation: Sequence[dict[str, Any]],
    *,
    top: int = 5,
) -> str:
    """Headline results for the project README (full tables live in results/)."""
    commit = str(metadata.get("commit") or "")[:7]
    parts = [
        f"**Test set: {metadata.get('test_set')}.** Generated {metadata.get('generated_at')} "
        f"at commit `{commit}`.",
        "",
        judge_line(metadata),
        "",
        split_line(metadata),
    ]
    if retrieval_rows:
        best = sorted(retrieval_rows, key=lambda r: -(r.get("chunk_ndcg@10") or 0.0))[:top]
        parts += [
            f"Retrieval, top {top} of {len(retrieval_rows)} configurations by chunk-level "
            "nDCG@10 (95% bootstrap CIs):",
            "",
            retrieval_markdown(best),
        ]
    measured = [r for r in generation if is_complete(r)]
    if measured:
        parts += ["Generation (rows marked complete: at least 90% of the planned questions "
                  "answered and judged; `questions` is the sample each row was run on):", "",
                  generation_markdown(measured)]  # fmt: skip
    elif generation:
        parts += [
            "Generation: no configuration is complete yet (free-tier quota; resumes daily).",
            "",
        ]
    limitations = metadata.get("limitations") or []
    if limitations:
        parts += ["", "Limitations:", *[f"- {x}" for x in limitations], ""]
    parts.append("Full tables, charts and run metadata: [results/](results/README.md).")
    return "\n".join(parts).strip() + "\n"


def update_readme(readme: Path, section: str) -> bool:
    """Replace the text between the RESULTS markers. Returns False if markers are absent."""
    text = readme.read_text(encoding="utf-8")
    start = text.find(README_START)
    end = text.find(README_END)
    if start < 0 or end < start:
        return False
    new = text[: start + len(README_START)] + "\n" + section + text[end:]
    readme.write_text(new, encoding="utf-8")
    return True
