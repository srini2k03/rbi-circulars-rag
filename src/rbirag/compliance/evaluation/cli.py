"""``rbirag comply eval ...``: inspect, check and score the labelled clause set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rbirag.compliance.evaluation.dataset import DEFAULT_ROOT, Dataset, DatasetError, load_dataset

eval_app = typer.Typer(help="Labelled clause set: check, render, score.", no_args_is_help=True)

RootOpt = Annotated[Path, typer.Option(help="Directory with policies.json and labels.jsonl.")]


def _load(root: Path) -> Dataset:
    try:
        return load_dataset(root)
    except DatasetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@eval_app.command("stats")
def stats(root: RootOpt = DEFAULT_ROOT) -> None:
    """Counts by split, verdict and tag."""
    from rbirag.compliance.evaluation.dataset import dataset_stats

    typer.echo(json.dumps(dataset_stats(_load(root)), indent=2))


@eval_app.command("render")
def render(
    root: RootOpt = DEFAULT_ROOT,
    out_dir: Annotated[Path | None, typer.Option(help="Default: <root>/policies")] = None,
    pdf: Annotated[bool, typer.Option("--pdf", help="Also write PDFs.")] = False,
) -> None:
    """Write the policy documents rendered from the labels."""
    from rbirag.compliance.evaluation.dataset import render_all, render_pdf

    dataset = _load(root)
    paths = render_all(dataset, out_dir or root / "policies")
    for path in paths:
        typer.echo(str(path))
        if pdf:
            typer.echo(str(render_pdf(path.read_text("utf-8"), path.with_suffix(".pdf"))))


@eval_app.command("check")
def check(
    root: RootOpt = DEFAULT_ROOT,
    documents: Annotated[
        Path | None, typer.Option(help="Corpus documents JSONL (default: settings).")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Write the gold check JSON here.")] = None,
) -> None:
    """Validate the set and check every citation and quote against the corpus."""
    from rbirag.compliance.evaluation.dataset import check_gold
    from rbirag.config import get_settings
    from rbirag.document import iter_documents_jsonl

    dataset = _load(root)
    path = documents or get_settings().documents_path
    docs = {d.doc_id: d for d in iter_documents_jsonl(path)}
    result = check_gold(dataset, docs)
    payload = {"corpus_documents": len(docs), **result.model_dump()}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2))
    if not result.ok:
        raise typer.Exit(code=1)


@eval_app.command("score")
def score(
    predictions: Annotated[Path, typer.Argument(help="ClausePrediction JSONL.")],
    root: RootOpt = DEFAULT_ROOT,
    split: Annotated[str, typer.Option(help="dev | test | all")] = "test",
    documents: Annotated[
        Path | None, typer.Option(help="Corpus JSONL, for wrong-entity detection.")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Write the report JSON here.")] = None,
) -> None:
    """Score predictions against the labels and print the report."""
    from rbirag.compliance.evaluation.metrics import score_predictions, summarize
    from rbirag.compliance.evaluation.report import render_report
    from rbirag.compliance.records import ClausePrediction
    from rbirag.document import iter_documents_jsonl

    if split not in {"dev", "test", "all"}:
        raise typer.BadParameter("split must be dev, test or all")
    dataset = _load(root)
    items = dataset.items(None if split == "all" else ("dev" if split == "dev" else "test"))
    preds: dict[str, ClausePrediction] = {}
    systems: set[str] = set()
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = ClausePrediction.model_validate_json(line)
            preds[p.item_id] = p
            systems.add(p.system)
    titles = {d.doc_id: d.title for d in iter_documents_jsonl(documents)} if documents else None
    report = summarize(
        score_predictions(items, preds, titles=titles),
        system="+".join(sorted(systems)) or "unknown",
        split=split,
    )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(render_report([report]))


@eval_app.command("segmentation")
def segmentation(
    root: RootOpt = DEFAULT_ROOT,
    pdf: Annotated[bool, typer.Option("--pdf/--no-pdf", help="Also score rendered PDFs.")] = True,
    out: Annotated[Path | None, typer.Option(help="Write the rows as JSON here.")] = None,
) -> None:
    """Clause-splitting precision / recall / F1 against the gold clause boundaries."""
    from rbirag.compliance.evaluation.segmentation import segmentation_report, segmentation_table

    rows = segmentation_report(_load(root), pdf=pdf)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([r.model_dump() for r in rows], indent=2) + "\n", "utf-8")
    typer.echo(segmentation_table(rows))


@eval_app.command("review")
def review(
    root: RootOpt = DEFAULT_ROOT,
    reviewer: Annotated[str, typer.Option(help="Your name; stored as human:<name>.")] = "",
    limit: Annotated[int, typer.Option(min=1, help="Stop after this many items.")] = 20,
    split: Annotated[str, typer.Option(help="dev | test | all")] = "all",
) -> None:
    """Review unreviewed labels: (a)ccept, (e)dit the verdict, (r)eject, (s)kip, (q)uit.

    Each decision is written to labels.jsonl immediately, so you can stop at any time."""
    from rbirag.compliance.evaluation.dataset import Review, save_labels
    from rbirag.compliance.records import VERDICTS

    name = reviewer.strip() or typer.prompt("Reviewer name")
    who = f"human:{name}"
    dataset = _load(root)
    by_policy = {p.policy_id: p for p in dataset.policies}
    done = 0
    for label in dataset.labels:
        if done >= limit:
            break
        if any(r.reviewer == who for r in label.reviews):
            continue
        if split != "all" and by_policy[label.policy_id].split != split:
            continue
        typer.echo("\n" + "=" * 78)
        typer.echo(f"{label.item_id}  [{by_policy[label.policy_id].title}]")
        typer.echo(f"\nCLAUSE: {label.number} {label.text}")
        typer.echo(f"\nLABEL:  {label.verdict}   tags: {', '.join(label.tags) or '-'}")
        typer.echo(f"CITES:  {label.citations or '-'}   acceptable: {label.acceptable or '-'}")
        typer.echo(f"WHY:    {label.rationale}")
        for ev in label.evidence:
            typer.echo(f'  {ev.doc_id} para {ev.paragraph}: "{ev.quote}"')
        choice = (
            typer.prompt("\n(a)ccept (e)dit (r)eject (s)kip (q)uit", default="s").strip().lower()
        )
        if choice == "q":
            break
        if choice not in {"a", "e", "r"}:
            continue
        note = typer.prompt("Note (optional)", default="", show_default=False)
        if choice == "e":
            verdict = typer.prompt(f"New verdict {VERDICTS}", default=label.verdict)
            if verdict not in VERDICTS:
                typer.echo(f"unknown verdict {verdict!r}; skipped", err=True)
                continue
            note = f"verdict {label.verdict} -> {verdict}. {note}".strip()
            label.verdict = verdict
        decision = {"a": "accepted", "e": "edited", "r": "rejected"}[choice]
        label.reviews.append(Review(reviewer=who, decision=decision, note=note))  # type: ignore[arg-type]
        save_labels(dataset.labels, root / "labels.jsonl")
        done += 1
    typer.echo(f"\n{done} decision(s) saved to {root / 'labels.jsonl'}")
