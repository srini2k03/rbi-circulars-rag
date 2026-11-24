"""``rbirag eval bench``: the retrieval strategy benchmark (see :mod:`rbirag.eval.benchmark`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from rbirag.config import get_settings
from rbirag.eval.cli import _csv, _models, checked_pairs, dev_test, eval_app, load_qa


def _parse_config(value: str) -> tuple[str, int]:
    chunker, size = value.rsplit("-", 1)
    return chunker, int(size)


@eval_app.command("bench")
def bench(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    qa: Annotated[
        Path | None, typer.Option(help="Q/A JSONL (default: qa_v1 or candidates).")
    ] = None,
    primary: Annotated[str, typer.Option(help="Chunk config of the main index.")] = "fixed-1024",
    child: Annotated[
        str, typer.Option(help="Chunk config for parent-child ('' = skip).")
    ] = "fixed-256",
    out_dir: Annotated[Path | None, typer.Option(help="Default: results/benchmark")] = None,
    models: Annotated[
        bool,
        typer.Option(
            help="Use the embedder/reranker for cache misses (CI). --no-models replays the "
            "committed score cache only."
        ),
    ] = True,
    embedder: Annotated[str | None, typer.Option(help="Override RBIRAG_EMBEDDER.")] = None,
    reranker: Annotated[str | None, typer.Option(help="Override RBIRAG_RERANKER.")] = None,
    families: Annotated[str, typer.Option(help="Comma list to run only these families.")] = "",
    timing: Annotated[bool, typer.Option(help="Live latency pass (needs the models).")] = True,
    log: Annotated[bool, typer.Option(help="Append to results/experiments.jsonl.")] = True,
) -> None:
    """Compare retrieval strategies on the same held-out questions: tune on dev, report
    on test (CIs, paired tests vs BM25, failure categories), save the score cache."""
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.benchmark import (
        BenchContext,
        build_bench_index,
        doc_index,
        file_sha256,
        live_timing,
        run_benchmark,
        strategy_grid,
        title_relatedness,
        write_outputs,
    )
    from rbirag.eval.history import log_experiment
    from rbirag.eval.score_cache import ScoreCache
    from rbirag.index.build import build_index
    from rbirag.retrieval.expansion import AcronymExpander

    s = get_settings()
    out = out_dir or s.results_dir / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    docs_path = docs or s.documents_path
    documents = read_documents_jsonl(docs_path)
    pairs = checked_pairs(load_qa(s, qa), documents, path=out / "gold_check.json")
    dev, test = dev_test(s, pairs)

    emb = rr = None
    if models:
        emb, rr = _models(s, embedder, reranker)

    def dense_factory(name: str) -> Any:
        if emb is None:
            return None

        def build(chunks: list[Any]) -> Any:
            bundle = build_index(chunks, embedder=emb, collection=f"bench_{name}")
            if hasattr(emb, "save"):
                emb.save()  # persist the embedding cache for the next CI job
            return bundle.dense

        return build

    indexes: dict[str, Any] = {}
    for role, cfg in (("primary", primary), ("child", child)):
        if not cfg:
            continue
        chunker, size = _parse_config(cfg)
        cache_path = out / f"score_cache-{chunker}-{size}.jsonl.gz"
        indexes[role] = build_bench_index(
            documents, chunker, size, s,
            cache=ScoreCache.load(cache_path),
            live_dense=dense_factory(cfg),
            live_reranker=rr,
        )  # fmt: skip
        indexes[role + "_cache"] = cache_path
    ctx = BenchContext(
        settings=s,
        docs=documents,
        dev=dev,
        test=test,
        primary=indexes["primary"],
        child=indexes.get("child"),
        doc_bm25=doc_index(documents),
        acronyms=AcronymExpander.default(),
        related=title_relatedness(documents),
        doc_meta={d.doc_id: d.metadata() for d in documents},
    )
    grid = strategy_grid(
        have_dense=ctx.primary.dense is not None, have_reranker=ctx.primary.reranker is not None
    )
    wanted = set(_csv(families))
    if wanted:
        grid = [st for st in grid if st.family in wanted or st.name == "bm25"]
    per_query = out / "per_query.jsonl.gz"
    per_query.unlink(missing_ok=True)
    report = run_benchmark(ctx, strategies=grid, per_query_path=per_query, on_progress=typer.echo)
    for role in ("primary", "child"):
        idx = indexes.get(role)
        if idx is not None and (
            idx.cache.misses or (models and not indexes[role + "_cache"].exists())
        ):
            idx.cache.save(indexes[role + "_cache"])
    if timing and models and emb is not None:
        live_dense: dict[str, Any] = {}
        for role in ("primary", "child"):
            idx = indexes.get(role)
            lazy = getattr(getattr(idx, "dense", None), "live", None)
            live_dense[role] = lazy
        report.live_timing = live_timing(
            ctx,
            [ctx.selected[f] for f in ctx.selected],
            dense=live_dense,
            reranker=rr,
            cached=ctx.test_results,
        )
    if not models:
        report.notes.append("replayed from the committed score cache (no models loaded)")
    write_outputs(report, out)
    if log:
        log_experiment(
            s.results_dir,
            {
                "experiment": "retrieval-benchmark",
                "command": "rbirag eval bench",
                "commit": report.commit,
                "corpus_sha256": file_sha256(docs_path),
                "qa_sha256": file_sha256(qa or s.eval_dir / "qa_candidates.jsonl"),
                "n_dev": report.n_dev,
                "n_test": report.n_test,
                "primary": report.primary,
                "child": report.child,
                "replayed_from_cache": not models,
                "recommended": report.recommended,
                "rows": [
                    {
                        "strategy": r["strategy"],
                        "params_hash": r["params_hash"],
                        "test_chunk_ndcg@10": round(r["chunk_ndcg@10"], 4),
                        "test_doc_recall@5": round(r["doc_recall@5"], 4),
                    }
                    for r in report.rows
                ],
            },
        )
    typer.echo(json.dumps({"recommended": report.recommended, "out": str(out)}))
