"""``rbirag eval ...`` commands: Q/A generation and review, ablations, calibration, report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from rbirag.config import LLMBackend, Settings, get_settings

logger = logging.getLogger(__name__)

eval_app = typer.Typer(
    help="Evaluation: test set, metrics, ablations, report.", no_args_is_help=True
)


def _echo_json(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def load_qa(settings: Settings, path: Path | None = None) -> list[Any]:
    """The reviewed set if present, else the candidates; rejected/auto-failed dropped."""
    from rbirag.eval.dataset import read_qa_jsonl, usable

    if path is None:
        reviewed = settings.eval_dir / "qa_v1.jsonl"
        path = reviewed if reviewed.exists() else settings.eval_dir / "qa_candidates.jsonl"
    pairs = usable(read_qa_jsonl(path))
    logger.info("loaded %d usable Q/A pairs from %s", len(pairs), path)
    return pairs


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
) -> None:
    """Generate candidate Q/A pairs (resumable: existing candidates are kept)."""
    from rbirag.eval.dataset import read_qa_jsonl, write_qa_jsonl
    from rbirag.eval.qa_generate import generate_candidates
    from rbirag.generation.rag import build_llm
    from rbirag.index.store import ChunkStore

    s = get_settings()
    out_path = out or s.eval_dir / "qa_candidates.jsonl"
    existing = read_qa_jsonl(out_path) if out_path.exists() else []
    llm = build_llm(s, backend or s.qa_backend, model or s.qa_model)
    store = ChunkStore.load(chunks)
    run = generate_candidates(
        llm,
        list(store),
        n_answerable=n_answerable,
        n_unanswerable=n_unanswerable,
        seed=s.seed if seed is None else seed,
        existing=existing,
    )
    n = write_qa_jsonl(sorted(run.pairs, key=lambda p: p.qa_id), out_path)
    stats = {
        "written": n,
        "answerable": sum(p.answerable for p in run.pairs),
        "unanswerable": sum(not p.answerable for p in run.pairs),
        "auto_check_failed": sum(p.auto_check is False for p in run.pairs),
        "llm_calls_this_run": run.attempted,
        "complete": run.complete,
        "model": llm.model,
    }
    _echo_json(stats)
    if not run.complete:
        typer.echo("INCOMPLETE: request quota exhausted; rerun later to continue.", err=True)


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


@eval_app.command("calibrate")
def calibrate(
    config: Annotated[
        str | None, typer.Option(help="hybrid_rerank config (default: best).")
    ] = None,
    qa: Annotated[Path | None, typer.Option()] = None,
    beta: Annotated[float, typer.Option(help="F-beta of refusal; <1 favours precision.")] = 0.5,
) -> None:
    """Choose the confidence-gate threshold on the dev split."""
    from rbirag.eval.calibrate import calibrate_threshold
    from rbirag.eval.dataset import split

    s = get_settings()
    rows = [r for r in _load_rows(s.results_dir) if r["mode"] == "hybrid_rerank"]
    if config is None:
        if not rows:
            raise typer.BadParameter("no hybrid_rerank rows; run `rbirag eval retrieval` first")
        config = max(rows, key=lambda r: r.get("chunk_ndcg@10") or 0)["config"]
    pairs = load_qa(s, qa)
    dev, _ = split(pairs, s.dev_fraction, s.seed)
    dev_ids = {p.qa_id: p.answerable for p in dev}
    probs: list[float] = []
    answerable: list[bool] = []
    for f in sorted((s.results_dir / "retrieval").glob("per_query-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if (
                item["config"] == config
                and item["qa_id"] in dev_ids
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
    p = plan_generation(
        _load_rows(s.results_dir),
        pairs,
        full_configs=full_configs,
        subset_configs=subset_configs,
        subset_size=subset_size,
        seed=s.seed,
    )
    calls = planned_calls(p, pairs, _csv(backends))
    out = s.results_dir / "generation" / "plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({**p.model_dump(), "planned_calls": calls}, indent=2), encoding="utf-8"
    )
    typer.echo(f"full set ({len(p.all_ids)} q): {p.full}")
    typer.echo(f"subset ({len(p.subset_ids)} q): {p.subset}")
    typer.echo("planned LLM calls (upper bound; the confidence gate skips some):")
    for k, v in calls.items():
        typer.echo(f"  {k}: {v}")
    days = calls["api_calls_upper_bound"] / max(1, s.llm_rpd)
    typer.echo(
        f"  at {s.llm_rpd} requests/day per model this is <= {days:.1f} day(s) of quota; "
        "runs resume from the cache"
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
    nli: Annotated[str, typer.Option(help="model | fake | none")] = "model",
    embedder: Annotated[str | None, typer.Option()] = None,
    reranker: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Answer + judge the planned questions for one backend (resumable, quota-aware)."""
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.ablate import RetrievalConfig, build_config_index, run_generation_eval
    from rbirag.eval.nli import CrossEncoderNLI, FakeNLI, NLIScorer
    from rbirag.generation.rag import RAGPipeline, build_llm
    from rbirag.retrieval.pipeline import Retriever

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
    llm = build_llm(s, backend, model)
    judge_llm = build_llm(s, s.judge_backend, s.judge_model) if judge else None
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
        retriever = Retriever(
            bundle.store,
            bm25=bundle.bm25,
            dense=bundle.dense,
            reranker=rr if cfg.mode == "hybrid_rerank" else None,
            mode=mode,
            candidate_k=s.candidate_k,
            rerank_k=s.rerank_k,
            rrf_k=s.rrf_k,
            confidence_threshold=threshold,
            confidence_temperature=s.confidence_temperature,
        )
        pipeline = RAGPipeline.from_settings(retriever, llm, s)
        records = s.results_dir / "generation" / f"records-{backend}-{name}.jsonl"
        run = run_generation_eval(
            pipeline,
            [by_id[i] for i in ids if i in by_id],
            config=name,
            records_path=records,
            judge=judge_llm,
            nli=scorer,
            embedder=emb,
        )
        status["configs"][name] = vars(run)
        typer.echo(f"{name}: {vars(run)}")
        if bundle.dense is not None:
            bundle.dense.close()
        if not run.complete:
            complete = False
            break
    status["complete"] = complete
    out = s.results_dir / "generation" / f"status-{backend}.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not complete:
        typer.echo("INCOMPLETE: request quota exhausted; rerun later to resume.", err=True)


# ----------------------------------------------------------------------------- report


@eval_app.command("report")
def report(
    qa: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Write CSV/Markdown tables, charts and run metadata into the results directory."""
    from rbirag.eval.ablate import load_records
    from rbirag.eval.calibrate import Calibration
    from rbirag.eval.gen_metrics import GenerationRecord
    from rbirag.eval.report import build_report, run_metadata

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
    pairs = load_qa(s, qa)
    judge_models = sorted({r.judge_model for r in records if r.judge_model})
    meta = run_metadata(
        pairs,
        judge_model=", ".join(judge_models) or s.judge_model,
        generator_models=sorted({f"{r.backend}:{r.model}" for r in records if r.model}),
        embedding_model=s.embedding_model,
        reranker_model=s.reranker_model,
        nli_model=s.nli_model,
        llm_cost_usd=round(sum(r.cost_usd for r in records), 6),
        llm_tokens=sum(r.input_tokens + r.output_tokens for r in records),
        judge_tokens=sum(r.judge_input_tokens + r.judge_output_tokens for r in records),
    )
    written = build_report(
        s.results_dir,
        retrieval_rows=_load_rows(s.results_dir),
        generation_records=records,
        calibration=cal,
        metadata=meta,
        n_resamples=s.bootstrap_samples,
    )
    for p in written:
        typer.echo(str(p))
