"""Markdown tables for compliance evaluation reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rbirag.compliance.evaluation.metrics import ItemScore, Report, paired_bootstrap_difference
from rbirag.compliance.evaluation.runs import RunRecord
from rbirag.compliance.records import VERDICTS
from rbirag.eval.stats import Estimate


def _fmt(est: Estimate | None, digits: int = 3) -> str:
    if est is None or est.n == 0 or est.mean != est.mean:  # NaN check
        return "-"
    if est.low != est.low:
        return f"{est.mean:.{digits}f}"
    return est.fmt(digits)


HEADLINE: tuple[tuple[str, str, int], ...] = (
    ("verdict accuracy", "verdict_accuracy", 3),
    ("macro-F1", "macro_f1", 3),
    ("citation precision", "citation_precision", 3),
    ("citation recall", "citation_recall", 3),
    ("tool-argument validity", "tool_argument_validity", 3),
    ("fetch grounding", "fetch_grounding", 3),
    ("tool-selection accuracy", "tool_selection_accuracy", 3),
    ("task success", "task_success", 3),
    ("steps / clause", "avg_steps", 2),
    ("LLM calls / clause", "avg_llm_calls", 2),
    ("tokens / document", "tokens_per_document", 0),
    ("cost $ / document", "cost_usd_per_document", 4),
    ("latency s / document", "latency_ms_per_document", 0),
    ("ECE", "ece", 3),
)


def headline_table(reports: Sequence[Report]) -> str:
    """One column per system, one row per metric, 95% bootstrap CIs."""
    head = "| metric | " + " | ".join(f"{r.system} ({r.split}, n={r.n})" for r in reports) + " |"
    lines = [head, "|---|" + "---|" * len(reports)]
    for name, attr, digits in HEADLINE:
        cells = []
        for r in reports:
            est = getattr(r, attr)
            if attr == "latency_ms_per_document" and est is not None and est.n:
                est = Estimate(
                    mean=est.mean / 1000, low=est.low / 1000, high=est.high / 1000, n=est.n
                )
            cells.append(_fmt(est, digits))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confusion_table(report: Report) -> str:
    short = {
        v: v.replace("insufficient_evidence", "insufficient").replace("not_applicable", "n/a")
        for v in VERDICTS
    }
    lines = [
        "| gold \\ predicted | " + " | ".join(short[v] for v in VERDICTS) + " |",
        "|---|" + "---|" * len(VERDICTS),
    ]
    for g in VERDICTS:
        lines.append(
            f"| {short[g]} | " + " | ".join(str(report.confusion[g][p]) for p in VERDICTS) + " |"
        )
    return "\n".join(lines)


def failure_table(report: Report) -> str:
    if not report.failures:
        return "_No failures._"
    lines = ["| failure type | clauses | share |", "|---|---|---|"]
    for name, count in report.failures.items():
        lines.append(f"| {name} | {count} | {count / report.n:.0%} |")
    return "\n".join(lines)


def tag_table(report: Report) -> str:
    lines = ["| tag | n | accuracy | task success |", "|---|---|---|---|"]
    for tag, row in report.by_tag.items():
        lines.append(f"| {tag} | {int(row['n'])} | {row['accuracy']:.2f} | {row['success']:.2f} |")
    return "\n".join(lines)


def render_report(reports: Sequence[Report]) -> str:
    parts = ["## Headline metrics (95% bootstrap CIs)", "", headline_table(reports)]
    for r in reports:
        parts += [
            "",
            f"## {r.system}: confusion matrix",
            "",
            confusion_table(r),
            "",
            f"## {r.system}: failures by type",
            "",
            failure_table(r),
            "",
            f"## {r.system}: by tag",
            "",
            tag_table(r),
        ]
    return "\n".join(parts) + "\n"


def paired_table(
    scored: Mapping[str, Sequence[ItemScore]], *, agent_key: str, baseline_key: str
) -> str:
    """Agent minus baseline on the same clauses, with paired bootstrap CIs."""
    a = {s.item_id: s for s in scored[agent_key]}
    b = {s.item_id: s for s in scored[baseline_key]}
    common = sorted(set(a) & set(b))
    rows: list[tuple[str, list[float], list[float]]] = [
        (
            "verdict accuracy",
            [float(a[i].correct) for i in common],
            [float(b[i].correct) for i in common],
        ),
        (
            "task success",
            [float(a[i].success) for i in common],
            [float(b[i].success) for i in common],
        ),
    ]
    with_gold = [i for i in common if a[i].groups]
    rows.append(
        (
            "citation recall",
            [a[i].groups_hit / a[i].groups for i in with_gold],
            [b[i].groups_hit / b[i].groups for i in with_gold],
        )
    )
    lines = [
        f"| metric (n={len(common)}) | agent | baseline | agent - baseline [95% CI] |",
        "|---|---|---|---|",
    ]
    for name, va, vb in rows:
        diff = paired_bootstrap_difference(va, vb)
        lines.append(
            f"| {name} | {sum(va) / len(va):.3f} | {sum(vb) / len(vb):.3f} | "
            f"{diff.mean:+.3f} [{diff.low:+.3f}, {diff.high:+.3f}] |"
        )
    return "\n".join(lines)


def per_class_table(reports: Sequence[Report]) -> str:
    head = "| verdict | support | " + " | ".join(f"{r.system} F1" for r in reports) + " |"
    lines = [head, "|---|---|" + "---|" * len(reports)]
    for v in VERDICTS:
        support = int(reports[0].per_class[v]["support"])
        cells = " | ".join(f"{r.per_class[v]['f1']:.2f}" for r in reports)
        lines.append(f"| {v} | {support} | {cells} |")
    return "\n".join(lines)


def comparison_report(
    reports: Sequence[Report],
    scored: Mapping[str, Sequence[ItemScore]],
    runs: Sequence[RunRecord],
) -> str:
    split = reports[0].split
    parts = [f"## Results on the {split} split", ""]
    parts.append(
        "Runs: "
        + "; ".join(
            f"{r.system} = `{r.model}` via {r.backend} ({r.completed} clauses, "
            f"{r.errors} errors, {r.generated_at[:10]}, commit `{(r.commit or '?')[:7]}`)"
            for r in runs
        )
    )
    parts += ["", "### Headline metrics (95% bootstrap CIs)", "", headline_table(reports)]
    agent = next((k for k in scored if k.startswith("agent")), None)
    baseline = next((k for k in scored if k.startswith("baseline")), None)
    if agent and baseline:
        parts += [
            "",
            "### Agent vs fixed pipeline (paired, same clauses)",
            "",
            paired_table(scored, agent_key=agent, baseline_key=baseline),
        ]
    parts += ["", "### F1 per verdict", "", per_class_table(reports)]
    for r in reports:
        parts += [
            "",
            f"### {r.system}: confusion matrix",
            "",
            confusion_table(r),
            "",
            f"### {r.system}: failures by type",
            "",
            failure_table(r),
            "",
            f"### {r.system}: by tag",
            "",
            tag_table(r),
        ]
    return "\n".join(parts) + "\n"
