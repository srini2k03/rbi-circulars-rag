"""``rbirag eval nli-study``: the LLM-judge vs NLI faithfulness investigation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Annotated

import typer

from rbirag.config import get_settings
from rbirag.eval.cli import _csv, eval_app


@eval_app.command("nli-study")
def nli_study(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    configs: Annotated[
        str, typer.Option(help="BM25 generation configs to study (sources re-retrieved).")
    ] = "fixed-1024-bm25",
    out_dir: Annotated[Path | None, typer.Option(help="Default: results/analysis")] = None,
    nli: Annotated[str, typer.Option(help="cross-encoder | fake")] = "cross-encoder",
    limit: Annotated[int, typer.Option(help="Study at most this many answers (0 = all).")] = 0,
    log: Annotated[bool, typer.Option(help="Append to results/experiments.jsonl.")] = True,
) -> None:
    """Re-score committed answers with several NLI protocols and compare with the judge."""
    from rbirag.chunking import build_chunker
    from rbirag.document import read_documents_jsonl
    from rbirag.eval.ablate import RetrievalConfig, settings_for
    from rbirag.eval.history import log_experiment
    from rbirag.eval.nli import CrossEncoderNLI, FakeNLI
    from rbirag.eval.nli_study import VARIANTS, study_answer, summarize_study
    from rbirag.eval.records import load_records
    from rbirag.index.build import build_index
    from rbirag.retrieval.params import RetrievalParams
    from rbirag.retrieval.pipeline import Retriever

    s = get_settings()
    out = out_dir or s.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    documents = read_documents_jsonl(docs or s.documents_path)
    scorer: CrossEncoderNLI | FakeNLI = (
        FakeNLI() if nli == "fake" else CrossEncoderNLI(s.nli_model, device=s.device)
    )
    rows = []
    for name in _csv(configs):
        cfg = RetrievalConfig.parse(name)
        if cfg.mode != "bm25":
            raise typer.BadParameter(f"{name}: only BM25 configs can be re-retrieved offline")
        cs = settings_for(s, cfg.size)
        chunker = build_chunker(cfg.chunker, cs)
        bundle = build_index([c for d in documents for c in chunker.chunk(d)], embedder=None)
        retriever = Retriever(bundle.store, bm25=bundle.bm25, params=RetrievalParams(mode="bm25"))
        for path in sorted((s.results_dir / "generation").glob(f"records-*-{name}.jsonl")):
            for rec in load_records(path).values():
                if rec.refused or not rec.answerable or rec.error or rec.judge_faithfulness is None:
                    continue
                result = retriever.retrieve(rec.question, top_k=s.answer_top_k)
                sources = [r.chunk.text for r in result.results]
                rows.append(study_answer(rec, sources, scorer, scorer.count_pieces))
                if limit and len(rows) >= limit:
                    break
                if len(rows) % 25 == 0:
                    typer.echo(f"studied {len(rows)} answers")
    summary = summarize_study(rows, seed=s.seed)
    summary["nli_model"] = scorer.name
    summary["configs"] = _csv(configs)
    (out / "nli_study.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with gzip.open(out / "nli_study_answers.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")
    if log:
        means = {v: summary[v]["mean"]["mean"] for v in VARIANTS}
        log_experiment(
            s.results_dir,
            {
                "experiment": "nli-vs-judge-study",
                "command": "rbirag eval nli-study",
                "nli_model": scorer.name,
                "answers": summary["answers"],
                "means": means,
            },
        )
    typer.echo(json.dumps({k: summary[k] for k in ("answers", "whole_chunk_reproduces_committed")}))
