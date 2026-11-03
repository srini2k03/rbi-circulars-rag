"""``rbirag eval ...`` commands: Q/A generation and review, ablations, calibration, report."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer

from rbirag.config import LLMBackend, Settings, get_settings

logger = logging.getLogger(__name__)

eval_app = typer.Typer(
    help="Evaluation: test set, metrics, ablations, report.", no_args_is_help=True
)


@contextmanager
def fatal_errors_exit(marker: Path | None = None) -> Iterator[None]:
    """Turn a fatal configuration error into a clear message and exit code 2. With
    ``marker``, also write it (workflows then refuse to commit anything from the run)."""
    from rbirag.generation.llm import ConfigurationError

    try:
        yield
    except ConfigurationError as exc:
        typer.echo(f"::error::{exc}", err=True)
        if marker is not None:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"error": str(exc)[:500]}), encoding="utf-8")
        raise typer.Exit(code=2) from exc


def model_mix(models: Sequence[str]) -> dict[str, float]:
    """Share of items served by each model, e.g. {"gemini-3.8-flash": 0.7, ...}."""
    counts = Counter(models)
    total = sum(counts.values())
    return {m: round(c / total, 3) for m, c in counts.most_common()} if total else {}


def _echo_json(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def qa_path(settings: Settings, path: Path | None = None) -> Path:
    """The human-reviewed set (``qa_v1.jsonl``) once it exists, else the candidates."""
    if path is not None:
        return path
    reviewed = settings.eval_dir / "qa_v1.jsonl"
    return reviewed if reviewed.exists() else settings.eval_dir / "qa_candidates.jsonl"


def load_test_set(settings: Settings, path: Path | None = None) -> tuple[list[Any], list[str]]:
    """Usable pairs from :func:`qa_path` (human-rejected and auto-failed dropped), with the
    AI review in ``qa_ai_reviewed.jsonl`` applied: AI-rejected pairs dropped, AI-edited
    wording used. Returns the pairs and the ids of the AI-rejected pairs."""
    from rbirag.eval.ai_review import AI_REVIEW_FILE, apply_ai_reviews, read_ai_reviews
    from rbirag.eval.dataset import read_qa_jsonl, usable

    path = qa_path(settings, path)
    pairs, rejected = apply_ai_reviews(
        usable(read_qa_jsonl(path)), read_ai_reviews(settings.eval_dir / AI_REVIEW_FILE)
    )
    logger.info(
        "loaded %d usable Q/A pairs from %s (%d AI-reviewed, %d AI-rejected excluded)",
        len(pairs),
        path,
        sum(p.ai_review is not None for p in pairs),
        len(rejected),
    )
    return pairs, rejected


def load_qa(settings: Settings, path: Path | None = None) -> list[Any]:
    """The evaluation pairs (see :func:`load_test_set`)."""
    return load_test_set(settings, path)[0]


def _models(settings: Settings, embedder: str | None, reranker: str | None) -> tuple[Any, Any]:
    from rbirag.index.embedder import build_embedder
    from rbirag.retrieval.rerank import build_reranker

    emb = build_embedder(
        embedder or settings.embedder,
        model_name=settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        batch_size=settings.embedding_batch_size,
        device=settings.device,
        fake_dim=settings.fake_embedding_dim,
    )
    rr = build_reranker(
        reranker or settings.reranker, model_name=settings.reranker_model, device=settings.device
    )
    return emb, rr


# ----------------------------------------------------------------------------- test set


@eval_app.command("qa-generate")
def qa_generate(
    chunks: Annotated[Path, typer.Option(help="chunks.jsonl to sample from (section chunker).")],
    out: Annotated[Path | None, typer.Option(help="Candidates JSONL.")] = None,
    n_answerable: Annotated[int, typer.Option(min=1)] = 220,
    n_unanswerable: Annotated[int, typer.Option(min=0)] = 30,
    backend: Annotated[LLMBackend | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    stats_out: Annotated[Path | None, typer.Option(help="Write run stats JSON here.")] = None,
) -> None:
    """Generate candidate Q/A pairs (resumable: existing candidates are kept).

    Exit codes: 0 ok (possibly incomplete), 2 fatal configuration error (nothing written),
    3 the run attempted work but produced no new pairs.
    """
    from rbirag.eval.dataset import read_qa_jsonl, write_qa_jsonl
    from rbirag.eval.qa_generate import generate_candidates
    from rbirag.generation.rag import build_llm
    from rbirag.index.store import ChunkStore

    s = get_settings()
    out_path = out or s.eval_dir / "qa_candidates.jsonl"
    existing = read_qa_jsonl(out_path) if out_path.exists() else []
    llm = build_llm(
        s,
        backend or s.qa_backend,
        model or s.qa_model,
        fallbacks=s.qa_model_fallbacks if model is None else [],
    )
    store = ChunkStore.load(chunks)
    with fatal_errors_exit():
        run = generate_candidates(
            llm,
            list(store),
            n_answerable=n_answerable,
            n_unanswerable=n_unanswerable,
            seed=s.seed if seed is None else seed,
            existing=existing,
        )
    new_items = len(run.pairs) - len(existing)
    n = write_qa_jsonl(sorted(run.pairs, key=lambda p: p.qa_id), out_path) if new_items else 0
    stats = {
        "new_items": new_items,
        "total": len(run.pairs),
        "written": n,
        "answerable": sum(p.answerable for p in run.pairs),
        "unanswerable": sum(not p.answerable for p in run.pairs),
        "auto_check_failed": sum(p.auto_check is False for p in run.pairs),
        "llm_calls_this_run": run.attempted,
        "complete": run.complete,
        "stopped_because": run.stopped_because,
        "models_used": model_mix([p.generator_model for p in run.pairs if p.generator_model]),
    }
    _echo_json(stats)
    if stats_out:
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not run.complete:
        typer.echo(f"INCOMPLETE ({run.stopped_because}); rerun later to continue.", err=True)
    if run.attempted and new_items == 0:
        typer.echo("::error::No Q/A pairs were produced in this run.", err=True)
        raise typer.Exit(code=3)


@eval_app.command("verify")
def verify(
    candidates: Annotated[Path | None, typer.Option(help="Candidates JSONL.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Reviewed set (written after every item).")
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Review at most N pairs now.")] = None,
    seed: Annotated[int, typer.Option(help="Order of presentation.")] = 13,
    skip_auto_failed: Annotated[
        bool, typer.Option(help="Hide pairs that failed auto-checks.")
    ] = False,
) -> None:
    """Review candidates one at a time: [a]ccept, [e]dit, [r]eject, [s]kip, [q]uit."""
    from rbirag.eval.verify_cli import run_review

    s = get_settings()
    run_review(
        candidates or s.eval_dir / "qa_candidates.jsonl",
        out or s.eval_dir / "qa_v1.jsonl",
        limit=limit,
        seed=seed,
        include_auto_failed=not skip_auto_failed,
    )


# ----------------------------------------------------------------------------- retrieval


@eval_app.command("retrieval")
def retrieval(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    qa: Annotated[
        Path | None, typer.Option(help="Q/A JSONL (default: qa_v1 or candidates).")
    ] = None,
    chunkers: Annotated[str, typer.Option(help="Comma list.")] = "fixed,semantic,section",
    sizes: Annotated[
        str, typer.Option(help="Comma list of chunk sizes (tokens).")
    ] = "256,512,1024",
    modes: Annotated[str, typer.Option(help="Comma list.")] = "bm25,dense,hybrid,hybrid_rerank",
    out_dir: Annotated[Path | None, typer.Option()] = None,
    tag: Annotated[str, typer.Option(help="Suffix for output files (matrix job name).")] = "all",
    embedder: Annotated[str | None, typer.Option()] = None,
    reranker: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Retrieval ablation over chunker x size x mode (no LLM calls)."""
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.ablate import run_retrieval_ablation

    s = get_settings()
    out = out_dir or s.results_dir / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    per_query = out / f"per_query-{tag}.jsonl"
    per_query.unlink(missing_ok=True)
    emb, rr = _models(s, embedder, reranker)
    rows = run_retrieval_ablation(
        read_documents_jsonl(docs or s.documents_path),
        load_qa(s, qa),
        settings=s,
        embedder=emb,
        reranker=rr,
        chunkers=_csv(chunkers),
        sizes=[int(x) for x in _csv(sizes)],
        modes=_csv(modes),
        per_query_path=per_query,
        on_row=lambda r: typer.echo(
            f"{r['config']}: chunk nDCG@10={r['chunk_ndcg@10']:.3f} doc R@5={r['doc_recall@5']:.3f}"
        ),
    )
    (out / f"rows-{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    typer.echo(f"wrote {len(rows)} rows to {out / f'rows-{tag}.json'}")


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in sorted((results_dir / "retrieval").glob("rows-*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8")))
    return rows


def _load_per_query(results_dir: Path) -> list[Any]:
    from rbirag.eval.ablate import RetrievalQueryResult

    out: list[Any] = []
    for f in sorted((results_dir / "retrieval").glob("per_query-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(RetrievalQueryResult.model_validate_json(line))
    return out


def dev_test(s: Settings, pairs: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """The document-level dev/test split used everywhere (selection/calibration on dev,
    headline results on test)."""
    from rbirag.eval.dataset import split

    return split(pairs, s.dev_fraction, s.seed)


def retrieval_rows_for(
    s: Settings, pairs: Sequence[Any], *, rows: Sequence[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Retrieval rows recomputed on exactly ``pairs`` (e.g. the dev split), from the
    per-query results. Without per-query results nothing can be restricted, so this
    refuses rather than silently mixing splits."""
    from rbirag.eval.report import resummarize_retrieval

    rows = list(rows) if rows is not None else _load_rows(s.results_dir)
    per_query = _load_per_query(s.results_dir)
    if rows and not per_query:
        raise typer.BadParameter(
            "per-query retrieval results are missing; rerun `rbirag eval retrieval` "
            "(they are needed to select configurations on the dev split only)"
        )
    questions = {p.qa_id: p.question for p in pairs}
    edited = {p.qa_id for p in pairs if p.ai_review is not None and p.ai_review.decision == "edit"}
    return resummarize_retrieval(
        rows, per_query, questions, edited=edited, n_resamples=s.bootstrap_samples
    )


@eval_app.command("calibrate")
def calibrate(
    config: Annotated[
        str | None, typer.Option(help="hybrid_rerank config (default: best).")
    ] = None,
    qa: Annotated[Path | None, typer.Option()] = None,
    beta: Annotated[float, typer.Option(help="F-beta of refusal; <1 favours precision.")] = 0.5,
) -> None:
    """Choose the confidence-gate threshold on the dev split. The default config is the
    best hybrid_rerank config *on the dev split* (test pairs are never looked at)."""
    from rbirag.eval.calibrate import calibrate_threshold

    s = get_settings()
    pairs = load_qa(s, qa)
    dev, _ = dev_test(s, pairs)
    if config is None:
        rows = [r for r in retrieval_rows_for(s, dev) if r["mode"] == "hybrid_rerank"]
        if not rows:
            raise typer.BadParameter("no hybrid_rerank rows; run `rbirag eval retrieval` first")
        config = max(rows, key=lambda r: r.get("chunk_ndcg@10") or 0)["config"]
    dev_ids = {p.qa_id: p.answerable for p in dev}
    dev_questions = {p.qa_id: p.question for p in dev}
    probs: list[float] = []
    answerable: list[bool] = []
    for f in sorted((s.results_dir / "retrieval").glob("per_query-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if (
                item["config"] == config
                and item["qa_id"] in dev_ids
                and item.get("question", dev_questions[item["qa_id"]])
                == dev_questions[item["qa_id"]]
                and item.get("confidence") is not None
            ):
                probs.append(float(item["confidence"]))
                answerable.append(dev_ids[item["qa_id"]])
    cal = calibrate_threshold(probs, answerable, beta=beta)
    out = s.results_dir / "calibration.json"
    out.write_text(json.dumps({"config": config, **cal.model_dump()}, indent=2), encoding="utf-8")
    c = cal.chosen
    typer.echo(
        f"{config}: tau={cal.threshold:.2f} on {cal.n_dev} dev questions "
        f"({cal.n_unanswerable} unanswerable): refusal P={c.refusal_precision:.3f} "
        f"R={c.refusal_recall:.3f}, coverage={c.answerable_coverage:.3f} -> {out}"
    )


# ----------------------------------------------------------------------------- generation


@eval_app.command("plan")
def plan(
    qa: Annotated[Path | None, typer.Option()] = None,
    backends: Annotated[
        str, typer.Option(help="Comma list of generator backends.")
    ] = "openai_compatible,ollama",
    full_configs: Annotated[int, typer.Option(min=1)] = 1,
    subset_configs: Annotated[int, typer.Option(min=0)] = 2,
    subset_size: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    """Pick generation configs from the retrieval results and print the planned LLM calls."""
    from rbirag.eval.ablate import plan_generation, planned_calls

    s = get_settings()
    pairs = load_qa(s, qa)
    dev, test = dev_test(s, pairs)
    # configurations are chosen on dev; generation runs (and is reported) on test only
    p = plan_generation(
        retrieval_rows_for(s, dev),
        test,
        full_configs=full_configs,
        subset_configs=subset_configs,
        subset_size=subset_size,
        seed=s.seed,
    )
    calls = planned_calls(p, test, _csv(backends))
    out = s.results_dir / "generation" / "plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({**p.model_dump(), "planned_calls": calls}, indent=2), encoding="utf-8"
    )
    typer.echo(f"split by document: {len(dev)} dev (selection) / {len(test)} test (reported)")
    typer.echo(f"full test set ({len(p.all_ids)} q): {p.full}")
    typer.echo(f"subset ({len(p.subset_ids)} q): {p.subset}")
    typer.echo("planned LLM calls (upper bound; the confidence gate skips some):")
    for k, v in calls.items():
        typer.echo(f"  {k}: {v}")
    gen_calls = calls.get("generate[openai_compatible]", 0)
    judge_calls = calls.get("judge (all backends)", 0)
    for role, model, n in (
        ("generator", s.oai_model, gen_calls),
        ("judge", s.judge_model, judge_calls),
    ):
        rpd = s.rpd_for(model)
        typer.echo(
            f"  {role} {model} ({s.llm_provider}): {n} calls at {rpd}/day "
            f"<= {n / max(1, rpd):.1f} day(s); fallbacks share the load and runs resume"
        )


@eval_app.command("generate")
def generate(
    backend: Annotated[LLMBackend, typer.Option(help="Generator backend.")],
    model: Annotated[str | None, typer.Option()] = None,
    configs: Annotated[
        str | None, typer.Option(help="Only these plan configs (comma list).")
    ] = None,
    docs: Annotated[Path | None, typer.Option()] = None,
    qa: Annotated[Path | None, typer.Option()] = None,
    judge: Annotated[bool, typer.Option("--judge/--no-judge")] = True,
    judge_only: Annotated[
        bool, typer.Option("--judge-only", help="Never generate; only judge existing answers.")
    ] = False,
    nli: Annotated[str, typer.Option(help="model | fake | none")] = "model",
    embedder: Annotated[str | None, typer.Option()] = None,
    reranker: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Answer + judge the planned questions for one backend (resumable, quota-aware)."""
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.ablate import RetrievalConfig, build_config_index, run_generation_eval
    from rbirag.eval.nli import CrossEncoderNLI, FakeNLI, NLIScorer
    from rbirag.generation.rag import RAGPipeline, build_llm
    from rbirag.retrieval.pipeline import make_retriever

    s = get_settings()
    plan_path = s.results_dir / "generation" / "plan.json"
    if not plan_path.exists():
        raise typer.BadParameter("no plan; run `rbirag eval plan` first")
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    wanted = set(_csv(configs))
    items = [(c, plan_data["all_ids"]) for c in plan_data["full"]] + [
        (c, plan_data["subset_ids"]) for c in plan_data["subset"]
    ]
    items = [(c, ids) for c, ids in items if not wanted or c in wanted]
    threshold = s.confidence_threshold
    cal_path = s.results_dir / "calibration.json"
    if cal_path.exists():
        threshold = float(json.loads(cal_path.read_text(encoding="utf-8"))["threshold"])
    by_id = {p.qa_id: p for p in load_qa(s, qa)}
    documents = read_documents_jsonl(docs or s.documents_path)
    emb, rr = _models(s, embedder, reranker)
    marker = s.results_dir / "generation" / f"fatal-{backend}.json"
    if judge_only:
        from rbirag.eval.ablate import JudgeOnlyLLM

        llm: Any = JudgeOnlyLLM(backend, model or "judge-only")
    else:
        with fatal_errors_exit(marker):
            llm = build_llm(s, backend, model)
    judge_llm = None
    if judge:
        with fatal_errors_exit(marker):
            judge_llm = build_llm(
                s, s.judge_backend, s.judge_model, fallbacks=s.judge_model_fallbacks
            )
    scorer: NLIScorer | None = (
        CrossEncoderNLI(s.nli_model, device=s.device)
        if nli == "model"
        else FakeNLI()
        if nli == "fake"
        else None
    )
    status: dict[str, Any] = {
        "backend": backend,
        "model": llm.model,
        "threshold": threshold,
        "configs": {},
    }
    complete = True
    for name, ids in items:
        cfg = RetrievalConfig.parse(name)
        _, bundle = build_config_index(documents, cfg.chunker, cfg.size, s, emb)
        mode: Any = cfg.mode
        retriever = make_retriever(
            s,
            bundle.store,
            bm25=bundle.bm25,
            dense=bundle.dense,
            reranker=rr if cfg.mode == "hybrid_rerank" else None,
            mode=mode,
            confidence_threshold=threshold,
        )
        pipeline = RAGPipeline.from_settings(retriever, llm, s)
        records = s.results_dir / "generation" / f"records-{backend}-{name}.jsonl"
        with fatal_errors_exit(marker):
            run = run_generation_eval(
                pipeline,
                [by_id[i] for i in ids if i in by_id],
                config=name,
                records_path=records,
                judge=judge_llm,
                nli=scorer,
                embedder=emb,
                judge_only=judge_only,
            )
        status["configs"][name] = vars(run)
        typer.echo(f"{name}: {vars(run)}")
        if bundle.dense is not None:
            bundle.dense.close()
        if not run.complete:
            complete = False
            break
    status["complete"] = complete
    status["successful_items"] = sum(
        c["generated"] + c["judged"] for c in status["configs"].values()
    )
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", ",".join(sorted(wanted))) if wanted else "all"
    out = s.results_dir / "generation" / f"status-{backend}-{tag}.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not complete:
        typer.echo("INCOMPLETE: request quota exhausted; rerun later to resume.", err=True)


# ----------------------------------------------------------------------------- report


@eval_app.command("report")
def report(
    qa: Annotated[Path | None, typer.Option()] = None,
    update_readme: Annotated[
        Path | None, typer.Option(help="Also refresh the results section of this README.")
    ] = None,
) -> None:
    """Write CSV/Markdown tables, charts and run metadata into the results directory."""
    from rbirag.eval.ablate import load_records
    from rbirag.eval.calibrate import Calibration
    from rbirag.eval.gen_metrics import GenerationRecord
    from rbirag.eval.report import (
        build_report,
        current_items,
        mix,
        resummarize_retrieval,
        run_metadata,
    )

    s = get_settings()
    records: list[GenerationRecord] = []
    for f in sorted((s.results_dir / "generation").glob("records-*.jsonl")):
        records.extend(load_records(f).values())
    cal = None
    cal_path = s.results_dir / "calibration.json"
    if cal_path.exists():
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        data.pop("config", None)
        cal = Calibration.model_validate(data)
    pairs, ai_rejected = load_test_set(s, qa)
    dev, test = dev_test(s, pairs)
    from rbirag.eval.dataset import split_summary

    split_info = split_summary(dev, test, dev_fraction=s.dev_fraction, seed=s.seed)
    # headline numbers use the test split only; dev was used to pick configs and tau
    questions = {p.qa_id: p.question for p in test}
    dev_questions = {p.qa_id: p.question for p in dev}
    edited = {p.qa_id for p in pairs if p.ai_review is not None and p.ai_review.decision == "edit"}
    ai_questions = {p.qa_id: p.question for p in test if p.ai_review is not None}
    # results for AI-rejected pairs, dev pairs, or runs before an edit do not count
    records = current_items(records, questions, edited)
    retrieval_rows = _load_rows(s.results_dir)
    per_query = _load_per_query(s.results_dir)
    ai_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    if per_query:
        ai_rows = resummarize_retrieval(
            retrieval_rows, per_query, ai_questions, edited=edited, n_resamples=s.bootstrap_samples
        )
        dev_rows = resummarize_retrieval(
            retrieval_rows, per_query, dev_questions, edited=edited, n_resamples=s.bootstrap_samples
        )
        retrieval_rows = resummarize_retrieval(
            retrieval_rows, per_query, questions, edited=edited, n_resamples=s.bootstrap_samples
        )
    elif retrieval_rows:
        typer.echo("no per-query results: retrieval rows cannot be split; omitted", err=True)
        retrieval_rows = []

    model_mix: dict[str, str] = {}
    for backend in sorted({r.backend for r in records}):
        model_mix[f"generator ({backend})"] = mix(
            [r.model for r in records if r.backend == backend]
        )
    if any(r.judge_model for r in records):
        model_mix["judge"] = mix([r.judge_model for r in records])
    if any(p.generator_model for p in pairs):
        model_mix["Q/A writer"] = mix([p.generator_model for p in pairs])
    meta = run_metadata(
        pairs,
        test_set=qa_path(s, qa).as_posix(),
        split=split_info,
        ai_rejected=len(ai_rejected),
        model_mix=model_mix,
        judge_model=model_mix.get("judge", s.judge_model),
        generator_models=sorted({f"{r.backend}:{r.model}" for r in records if r.model}),
        embedding_model=s.embedding_model,
        reranker_model=s.reranker_model,
        nli_model=s.nli_model,
        llm_cost_usd=round(sum(r.cost_usd for r in records if not r.cached), 6),
        llm_tokens=sum(r.input_tokens + r.output_tokens for r in records if not r.cached),
        judge_tokens=sum(
            r.judge_input_tokens + r.judge_output_tokens for r in records if not r.judge_cached
        ),
    )
    planned: dict[str, int] = {}
    plan_path = s.results_dir / "generation" / "plan.json"
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        planned.update({c: len(plan_data["all_ids"]) for c in plan_data["full"]})
        planned.update({c: len(plan_data["subset_ids"]) for c in plan_data["subset"]})
    written = build_report(
        s.results_dir,
        retrieval_rows=retrieval_rows,
        generation_records=records,
        calibration=cal,
        metadata=meta,
        planned=planned,
        n_resamples=s.bootstrap_samples,
        ai_reviewed_retrieval_rows=ai_rows,
        dev_retrieval_rows=dev_rows,
        ai_reviewed_generation_records=[r for r in records if r.qa_id in ai_questions],
    )
    for p in written:
        typer.echo(str(p))
    if update_readme is not None:
        from rbirag.eval.report import generation_rows, readme_results_section
        from rbirag.eval.report import update_readme as _update

        gen_rows = generation_rows(records, planned=planned, n_resamples=s.bootstrap_samples)
        section = readme_results_section(meta, retrieval_rows, gen_rows)
        if _update(update_readme, section):
            typer.echo(f"updated results section in {update_readme}")
        else:
            typer.echo(f"no RESULTS markers in {update_readme}; left unchanged", err=True)


# ----------------------------------------------------------------------------- scheduling


def candidate_status(
    path: Path, n_answerable: int, n_unanswerable: int, tolerance: float
) -> dict[str, Any]:
    from rbirag.eval.ai_review import AI_REVIEW_FILE, read_ai_reviews
    from rbirag.eval.dataset import read_qa_jsonl

    # Targets count generated candidates (not human-rejected); pairs failing the automatic
    # checks still count, because regenerating them would not change the sample.
    pairs = (
        [p for p in read_qa_jsonl(path) if p.review_status != "rejected"] if path.exists() else []
    )
    answerable = sum(p.answerable for p in pairs)
    unanswerable = sum(not p.answerable for p in pairs)
    ai = read_ai_reviews(path.parent / AI_REVIEW_FILE)
    reviewed = [p for p in pairs if p.qa_id in ai]
    need_ans = int(n_answerable * (1 - tolerance))
    need_unans = int(n_unanswerable * (1 - tolerance))
    return {
        "answerable": answerable,
        "unanswerable": unanswerable,
        "passing_auto_checks": sum(p.auto_check is not False for p in pairs),
        "human_verified": sum(p.verified for p in pairs),
        "ai_reviewed": len(reviewed),
        "ai_rejected": sum(ai[p.qa_id].review.decision == "reject" for p in reviewed),
        "target_answerable": n_answerable,
        "target_unanswerable": n_unanswerable,
        "tolerance": tolerance,
        "targets_met": answerable >= need_ans and unanswerable >= need_unans,
    }


@eval_app.command("qa-status")
def qa_status(
    candidates: Annotated[Path | None, typer.Option()] = None,
    n_answerable: Annotated[int, typer.Option(min=0)] = 220,
    n_unanswerable: Annotated[int, typer.Option(min=0)] = 30,
    tolerance: Annotated[
        float, typer.Option(min=0, max=1, help="Allowed shortfall (items the model keeps failing).")
    ] = 0.05,
) -> None:
    """Whether the candidate set has reached its targets (used by scheduled workflows)."""
    s = get_settings()
    _echo_json(
        candidate_status(
            candidates or s.eval_dir / "qa_candidates.jsonl",
            n_answerable,
            n_unanswerable,
            tolerance,
        )
    )


def generation_progress(
    results_dir: Path, backends: Sequence[str], questions: Mapping[str, str]
) -> dict[str, Any]:
    """Pending work per (config, backend) of the plan, for the pairs in ``questions``
    (qa_id -> current wording). An item is pending if it has no record, its record was for
    an earlier wording, its answer errored (except permanent per-item rejections), or it
    still needs a judge verdict that was never attempted or hit the quota."""
    from rbirag.eval.ablate import load_records

    plan_path = results_dir / "generation" / "plan.json"
    if not plan_path.exists():
        return {"plan_exists": False, "done": False, "pending": None, "configs": {}}
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    items = [(c, plan_data["all_ids"]) for c in plan_data["full"]] + [
        (c, plan_data["subset_ids"]) for c in plan_data["subset"]
    ]
    configs: dict[str, dict[str, int]] = {}
    total_pending = 0
    for backend in backends:
        records: dict[tuple[str, str, str], Any] = {}
        for f in (results_dir / "generation").glob(f"records-{backend}-*.jsonl"):
            records.update(load_records(f))
        for config, ids in items:
            pending = 0
            for qa_id in ids:
                if qa_id not in questions:
                    continue
                rec = records.get((config, backend, qa_id))
                if rec is None or rec.question != questions[qa_id]:
                    pending += 1
                elif rec.error is not None:
                    pending += 0 if rec.error.startswith("rejected") else 1
                elif (
                    rec.answerable
                    and not rec.refused
                    and rec.judge_faithfulness is None
                    and rec.judge_relevance is None
                    and (rec.judge_error is None or rec.judge_error.startswith("quota"))
                ):
                    pending += 1
            configs[f"{config}/{backend}"] = {"planned": len(ids), "pending": pending}
            total_pending += pending
    return {
        "plan_exists": True,
        "done": total_pending == 0,
        "pending": total_pending,
        "configs": configs,
    }


@eval_app.command("progress")
def progress(
    backends: Annotated[str, typer.Option()] = "openai_compatible,ollama",
    qa: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Pending generation/judge work for the current plan (used by scheduled workflows)."""
    s = get_settings()
    questions = {p.qa_id: p.question for p in load_qa(s, qa)}
    _echo_json(generation_progress(s.results_dir, _csv(backends), questions))
