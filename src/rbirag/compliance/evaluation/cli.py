"""``rbirag comply eval ...``: inspect, check and score the labelled clause set."""

from __future__ import annotations

import json
from collections import Counter
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


@eval_app.command("run")
def run(
    system: Annotated[str, typer.Option(help="agent | baseline")] = "agent",
    split: Annotated[str, typer.Option(help="dev | test | all")] = "dev",
    root: RootOpt = DEFAULT_ROOT,
    backend: Annotated[
        str | None, typer.Option(help="anthropic | groq | gemini | ollama | custom")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Override the model id.")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N items.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Predictions JSONL (appended, resumable).")
    ] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where run records go.")] = Path(
        "results/compliance/runs"
    ),
    fresh: Annotated[bool, typer.Option("--fresh", help="Ignore earlier predictions.")] = False,
) -> None:
    """Run the agent or the baseline over labelled clauses, record the run, score it.

    Stops cleanly (exit 3) when the provider's quota runs out: predictions made so far
    are kept and the next run resumes from them."""
    from rbirag.compliance.agent import FATAL_ERRORS
    from rbirag.compliance.cli import build_checker
    from rbirag.compliance.evaluation.metrics import score_predictions, summarize
    from rbirag.compliance.evaluation.report import render_report
    from rbirag.compliance.evaluation.run import run_items
    from rbirag.compliance.evaluation.runs import (
        RunRecord,
        current_commit,
        prompt_settings,
        run_name,
        save_run,
    )
    from rbirag.compliance.llm import CachedChatModel, configured_model
    from rbirag.config import get_settings

    if split not in {"dev", "test", "all"}:
        raise typer.BadParameter("split must be dev, test or all")
    s = get_settings()
    backend_name = backend or s.compliance_llm_backend
    dataset = _load(root)
    items = dataset.items(None if split == "all" else ("dev" if split == "dev" else "test"))
    if limit:
        items = items[:limit]
    path = (
        out
        or Path("results/compliance/predictions") / f"{run_name(system, split, backend_name)}.jsonl"
    )
    checker, store = build_checker(system, backend=backend_name, model=model, settings=s)

    def progress(item, pred):  # type: ignore[no-untyped-def]
        ok = "ok " if pred.verdict == item.label.verdict and not pred.error else "MISS"
        typer.echo(f"{ok} {item.item_id}: {pred.verdict} (gold {item.label.verdict})", err=True)

    stopped: BaseException | None = None
    try:
        run_items(items, checker, path, resume=not fresh, on_item=progress)
    except FATAL_ERRORS as exc:
        stopped = exc
        typer.echo(f"run stopped: {type(exc).__name__}: {exc}", err=True)
    from rbirag.compliance.evaluation.run import load_predictions

    preds = {k: v for k, v in load_predictions(path).items() if k in {i.item_id for i in items}}
    chat = getattr(checker, "chat", None) or getattr(getattr(checker, "judge", None), "chat", None)
    record = RunRecord(
        system=system,
        split=split,
        backend=backend_name,
        model=model or configured_model(s, backend_name),
        predictions=str(path),
        item_ids=[i.item_id for i in items],
        settings=prompt_settings(s),
        commit=current_commit(),
        cache_hits=chat.hits if isinstance(chat, CachedChatModel) else 0,
        cache_misses=chat.misses if isinstance(chat, CachedChatModel) else 0,
        completed=sum(1 for p in preds.values() if p.error is None),
        errors=sum(1 for p in preds.values() if p.error),
    )
    save_run(record, runs_dir)
    titles = {d.doc_id: d.title for d in store.documents.values()}
    report = summarize(score_predictions(items, preds, titles=titles), system=system, split=split)
    typer.echo(render_report([report]))
    if stopped is not None:
        raise typer.Exit(code=3)


@eval_app.command("replay")
def replay(
    runs_dir: Annotated[Path, typer.Option(help="Run records to replay.")] = Path(
        "results/compliance/runs"
    ),
    root: RootOpt = DEFAULT_ROOT,
    split: Annotated[str | None, typer.Option(help="Only runs of this split.")] = None,
) -> None:
    """Re-run recorded evaluations from the response cache (no provider calls) and check
    that every prediction is reproduced. Exit 1: a prediction changed (a regression).
    Exit 2: a request is not in the cache (prompts or code changed: re-run live)."""
    import tempfile

    from rbirag.compliance.cli import build_checker
    from rbirag.compliance.evaluation.run import load_predictions, run_items
    from rbirag.compliance.evaluation.runs import apply_run_settings, load_runs
    from rbirag.compliance.llm import CacheMissError
    from rbirag.config import get_settings

    records = load_runs(runs_dir, split=split)
    if not records:
        typer.echo(f"no run records in {runs_dir}", err=True)
        raise typer.Exit(code=1)
    dataset = _load(root)
    by_id = {i.item_id: i for i in dataset.items(include_rejected=True)}
    changed = stale = 0
    for record in records:
        s = apply_run_settings(get_settings(), record).model_copy(
            update={"compliance_llm_cache_mode": "replay"}
        )
        recorded = load_predictions(Path(record.predictions))
        items = [
            by_id[i]
            for i in record.item_ids
            if i in by_id and i in recorded and recorded[i].error is None
        ]
        checker, _ = build_checker(
            record.system, backend=record.backend, model=record.model, settings=s
        )
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fresh = run_items(items, checker, Path(tmp) / "replay.jsonl", resume=False)
            except CacheMissError as exc:
                stale += 1
                typer.echo(f"STALE {record.system}/{record.split}/{record.backend}: {exc}")
                continue
        diffs = []
        for item in items:
            a, b = recorded[item.item_id], fresh[item.item_id]
            if (a.verdict, a.citations, [c.name for c in a.tool_calls]) != (
                b.verdict,
                b.citations,
                [c.name for c in b.tool_calls],
            ):
                diffs.append(
                    f"  {item.item_id}: {a.verdict} {a.citations} -> {b.verdict} {b.citations}"
                )
        status = "OK   " if not diffs else "DIFF "
        typer.echo(
            f"{status}{record.system}/{record.split}/{record.backend}: {len(items)} items replayed"
        )
        for line in diffs:
            typer.echo(line)
        changed += bool(diffs)
    if changed:
        raise typer.Exit(code=1)
    if stale:
        raise typer.Exit(code=2)


@eval_app.command("readme")
def readme(
    documents: Annotated[
        Path, typer.Option(help="Corpus JSONL (wrong-entity detection, as in the reports).")
    ] = Path("data/processed/documents.jsonl"),
    readme_path: Annotated[Path, typer.Option("--readme")] = Path("README.md"),
    runs_dir: Annotated[Path, typer.Option()] = Path("results/compliance/runs"),
    retrieval_path: Annotated[Path, typer.Option("--retrieval")] = Path(
        "results/compliance/retrieval.json"
    ),
    backend: Annotated[
        str | None, typer.Option(help="Only runs on this backend (default: all).")
    ] = None,
    root: RootOpt = DEFAULT_ROOT,
    check: Annotated[
        bool, typer.Option("--check", help="Exit 1 if the README section is out of date.")
    ] = False,
) -> None:
    """Rebuild the README's compliance results section from the recorded runs."""
    from rbirag.compliance.evaluation.readme import readme_section, update_readme
    from rbirag.document import iter_documents_jsonl

    titles = {d.doc_id: d.title for d in iter_documents_jsonl(documents)}
    section = readme_section(
        _load(root),
        runs_dir=runs_dir,
        retrieval_path=retrieval_path,
        titles=titles,
        backend=backend,
    )
    if check:
        current = readme_path.read_text(encoding="utf-8")
        if section not in current:
            typer.echo("README compliance section is out of date: run `rbirag comply eval readme`")
            raise typer.Exit(code=1)
        typer.echo("README compliance section is current")
        return
    changed = update_readme(readme_path, section)
    typer.echo(f"{readme_path}: {'updated' if changed else 'unchanged'}")


@eval_app.command("report")
def report(
    split: Annotated[str, typer.Option(help="dev | test")] = "test",
    runs_dir: Annotated[Path, typer.Option()] = Path("results/compliance/runs"),
    root: RootOpt = DEFAULT_ROOT,
    documents: Annotated[
        Path | None, typer.Option(help="Corpus JSONL (wrong-entity detection).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Write Markdown here (and .json next to it).")
    ] = None,
) -> None:
    """Compare every recorded system on a split: headline metrics, paired differences."""
    from rbirag.compliance.evaluation.metrics import score_predictions, summarize
    from rbirag.compliance.evaluation.report import comparison_report
    from rbirag.compliance.evaluation.run import load_predictions
    from rbirag.compliance.evaluation.runs import load_runs
    from rbirag.document import iter_documents_jsonl

    records = load_runs(runs_dir, split=split)
    if not records:
        typer.echo(f"no runs for split {split}", err=True)
        raise typer.Exit(code=1)
    dataset = _load(root)
    items = dataset.items("dev" if split == "dev" else "test")
    titles = {d.doc_id: d.title for d in iter_documents_jsonl(documents)} if documents else None
    scored = {}
    served: dict[str, dict[str, int]] = {}
    for record in records:
        preds = load_predictions(Path(record.predictions))
        label = f"{record.system} ({record.model})"
        scored[label] = score_predictions(items, preds, titles=titles)
        # with a fallback chain, the configured model is not always the one that answered
        served[f"{record.system} via {record.backend}"] = dict(
            Counter(p.model or "?" for p in preds.values())
        )
    reports = [summarize(v, system=k, split=split) for k, v in scored.items()]
    md = comparison_report(reports, scored, records, served=served)
    typer.echo(md)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        payload = {
            "split": split,
            "runs": [r.model_dump() for r in records],
            "served_by": served,
            "reports": [r.model_dump() for r in reports],
        }
        out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@eval_app.command("retrieval")
def retrieval(
    split: Annotated[str, typer.Option(help="dev | test | all")] = "dev",
    modes: Annotated[str, typer.Option(help="Comma-separated retrieval modes.")] = "bm25",
    root: RootOpt = DEFAULT_ROOT,
    out: Annotated[Path | None, typer.Option(help="Append rows to this JSON file.")] = None,
) -> None:
    """Recall of gold notifications for clause queries, per mode, with and without the
    entity filter (no LLM). Uses the index of the configured compliance chunker."""
    from rbirag.compliance.evaluation.retrieval import evaluate_retrieval, retrieval_table
    from rbirag.compliance.references import ReferenceGraph
    from rbirag.compliance.regulations import RegulationStore
    from rbirag.config import get_settings
    from rbirag.document import iter_documents_jsonl
    from rbirag.index.embedder import build_embedder
    from rbirag.retrieval.rerank import build_reranker

    s = get_settings()
    dataset = _load(root)
    items = dataset.items(None if split == "all" else ("dev" if split == "dev" else "test"))
    documents = {d.doc_id: d for d in iter_documents_jsonl(s.documents_path)}
    graph = ReferenceGraph.build(documents.values())
    wanted = [m.strip() for m in modes.split(",") if m.strip()]
    embedder = reranker = None
    if any(m != "bm25" for m in wanted):
        embedder = build_embedder(
            s.embedder,
            model_name=s.embedding_model,
            query_instruction=s.embedding_query_instruction,
            batch_size=s.embedding_batch_size,
            device=s.device,
            fake_dim=s.fake_embedding_dim,
        )
    if "hybrid_rerank" in wanted:
        reranker = build_reranker(s.reranker, model_name=s.reranker_model, device=s.device)
    rows = []
    for mode in wanted:
        store = RegulationStore.from_settings(
            s.model_copy(update={"compliance_search_mode": mode}),
            embedder=embedder,
            reranker=reranker,
            graph=graph,
            documents=documents,
        )
        for flt in (False, True):
            chunker = s.compliance_chunker or s.chunker
            label = f"{chunker} / {mode} / {'entity filter' if flt else 'no filter'}"
            rows.append(evaluate_retrieval(store, items, config=label, entity_filter=flt))
        store.close()  # one local Qdrant client at a time
    table = retrieval_table(rows)
    typer.echo(f"split={split}\n{table}")
    if out:
        existing = json.loads(out.read_text("utf-8")) if out.exists() else []
        existing = [
            r
            for r in existing
            if r.get("split") != split or r["config"] not in {x.config for x in rows}
        ]
        existing += [{"split": split, **r.model_dump()} for r in rows]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


@eval_app.command("threshold")
def threshold(
    predictions: Annotated[Path, typer.Argument(help="Dev-split predictions JSONL.")],
    root: RootOpt = DEFAULT_ROOT,
    out: Annotated[Path | None, typer.Option(help="Write the sweep as JSON here.")] = None,
) -> None:
    """Sweep the abstention threshold over recorded dev predictions (no LLM calls)."""
    from rbirag.compliance.evaluation.run import load_predictions
    from rbirag.compliance.evaluation.threshold import choose, sweep, threshold_table

    items = _load(root).items("dev")
    rows = sweep(items, load_predictions(predictions))
    best = choose(rows)
    typer.echo(threshold_table(rows, best))
    typer.echo(f"\nchosen threshold: {best.threshold:.2f} (macro-F1 {best.macro_f1:.3f} on dev)")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chosen": best.model_dump(), "rows": [r.model_dump() for r in rows]}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
