"""Command-line interface: ``rbirag fetch | ingest | index | query``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from rbirag.config import (
    ChunkerName,
    EmbedderBackend,
    LLMBackend,
    RerankerBackend,
    RetrievalMode,
    Settings,
    get_settings,
)

if TYPE_CHECKING:
    from rbirag.retrieval.pipeline import Retriever

app = typer.Typer(
    help="Hybrid-retrieval RAG over RBI notifications.", no_args_is_help=True, add_completion=False
)


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _echo_json(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# ----------------------------------------------------------------------------- fetch


@app.command()
def fetch(
    start_id: Annotated[int | None, typer.Option(help="Highest notification id to try.")] = None,
    min_id: Annotated[int | None, typer.Option(help="Lowest id to try.")] = None,
    max_docs: Annotated[int | None, typer.Option(help="Stop after this many documents.")] = None,
    probe: Annotated[
        bool, typer.Option("--probe", help="Fetch ~5 ids and print diagnostics.")
    ] = False,
    probe_ids: Annotated[str | None, typer.Option(help="Comma-separated ids for --probe.")] = None,
    probe_out: Annotated[
        Path | None, typer.Option(help="Also write the probe report here.")
    ] = None,
    raw_dir: Annotated[
        Path | None, typer.Option(help="Where raw pages and the manifest go.")
    ] = None,
    rate_limit: Annotated[
        float | None, typer.Option(min=1.0, help="Seconds between requests.")
    ] = None,
    pdfs: Annotated[bool, typer.Option("--pdfs/--no-pdfs", help="Download linked PDFs.")] = True,
    refetch: Annotated[bool, typer.Option(help="Ignore the manifest and fetch again.")] = False,
    ignore_robots: Annotated[bool, typer.Option(help="Do not consult robots.txt.")] = False,
) -> None:
    """Crawl RBI notification pages (polite, resumable)."""
    from rbirag.ingest.fetch_rbi import (
        CrawlBlockedError,
        FetcherConfig,
        RBIFetcher,
        default_probe_ids,
        descending_ids,
    )

    s = get_settings()
    start = start_id or s.fetch_start_id
    cfg = FetcherConfig(
        raw_dir=raw_dir or s.raw_dir,
        url_template=s.fetch_base_url,
        user_agent=s.user_agent,
        rate_limit_seconds=rate_limit or s.fetch_rate_limit_seconds,
        max_retries=s.fetch_max_retries,
        backoff_base_seconds=s.fetch_backoff_base_seconds,
        timeout_seconds=s.fetch_timeout_seconds,
        fetch_pdfs=pdfs and s.fetch_pdfs,
        max_consecutive_misses=s.fetch_max_consecutive_misses,
        respect_robots=not ignore_robots,
    )
    with RBIFetcher(cfg) as fetcher:
        if probe:
            ids = (
                [int(x) for x in probe_ids.split(",") if x.strip()]
                if probe_ids
                else default_probe_ids(start)
            )
            report = fetcher.probe(ids)
            _echo_json(report)
            if probe_out:
                probe_out.parent.mkdir(parents=True, exist_ok=True)
                probe_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            ok = sum(
                1 for r in report if r.get("http_status") == 200 and not r.get("looks_not_found")
            )
            typer.echo(f"probe: {ok}/{len(report)} ids returned parseable notifications")
            return

        def progress(entry: object) -> None:
            from rbirag.ingest.fetch_rbi import ManifestEntry

            assert isinstance(entry, ManifestEntry)
            typer.echo(
                f"{entry.id}\t{entry.status}\t{entry.parse_strategy or ''}\t{entry.title or ''}"[
                    :160
                ]
            )

        try:
            stats = fetcher.crawl(
                descending_ids(start, min_id or s.fetch_min_id),
                max_docs=max_docs or s.fetch_max_docs,
                refetch=refetch,
                on_entry=progress,
            )
        except CrawlBlockedError as exc:
            typer.echo(f"crawl blocked: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        _echo_json(stats.as_dict())


# ----------------------------------------------------------------------------- ingest


@app.command()
def ingest(
    raw_dir: Annotated[Path | None, typer.Option(help="Crawler output directory.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output documents JSONL.")] = None,
    stats_out: Annotated[Path | None, typer.Option(help="Write corpus stats JSON here.")] = None,
) -> None:
    """Parse, clean and annotate raw pages into documents.jsonl; print corpus stats."""
    from rbirag.document import write_documents_jsonl
    from rbirag.ingest.corpus import build_documents, corpus_stats, iter_raw_records

    s = get_settings()
    raw = raw_dir or s.raw_dir
    out_path = out or s.documents_path
    docs = list(build_documents(iter_raw_records(raw, s.fetch_base_url)))
    n = write_documents_jsonl(docs, out_path)
    stats = corpus_stats(docs)
    _echo_json(stats)
    if stats_out:
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    typer.echo(f"wrote {n} documents to {out_path}")


# ----------------------------------------------------------------------------- index


@app.command()
def index(
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl to index.")] = None,
    chunker: Annotated[ChunkerName | None, typer.Option(help="fixed | semantic | section")] = None,
    embedder: Annotated[
        EmbedderBackend | None, typer.Option(help="sentence-transformers | fake")
    ] = None,
    index_dir: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
    no_dense: Annotated[bool, typer.Option("--no-dense", help="BM25 only.")] = False,
) -> None:
    """Chunk documents and build the BM25 + Qdrant indexes."""
    from rbirag.chunking import build_chunker
    from rbirag.document import read_documents_jsonl
    from rbirag.index.build import build_index, chunk_documents, save_bundle
    from rbirag.index.embedder import build_embedder

    s = get_settings()
    chunker_name = chunker or s.chunker
    out_dir = (index_dir or s.index_dir) / chunker_name
    emb = build_embedder(
        embedder or s.embedder,
        model_name=s.embedding_model,
        query_instruction=s.embedding_query_instruction,
        batch_size=s.embedding_batch_size,
        device=s.device,
        fake_dim=s.fake_embedding_dim,
    )
    documents = read_documents_jsonl(docs or s.documents_path)
    chunks = chunk_documents(documents, build_chunker(chunker_name, s, embedder=emb))
    qdrant_path = None if s.qdrant_url else (s.qdrant_path or out_dir / "qdrant")
    bundle = build_index(
        chunks,
        embedder=None if no_dense else emb,
        collection=f"{s.qdrant_collection}_{chunker_name}",
        qdrant_url=s.qdrant_url,
        qdrant_path=qdrant_path,
        batch_size=s.qdrant_upsert_batch_size,
    )
    save_bundle(
        bundle,
        out_dir,
        info={
            "chunker": chunker_name,
            "documents": len(documents),
            "chunks": len(chunks),
            "embedder": None if no_dense else emb.name,
            "tokenizer": s.tokenizer,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
        },
    )
    if bundle.dense is not None:
        bundle.dense.close()
    typer.echo(f"indexed {len(documents)} documents -> {len(chunks)} chunks in {out_dir}")


# ----------------------------------------------------------------------------- query


@app.command()
def query(
    text: Annotated[str, typer.Argument(help="The question.")],
    mode: Annotated[
        RetrievalMode | None, typer.Option(help="bm25 | dense | hybrid | hybrid_rerank")
    ] = None,
    top_k: Annotated[int | None, typer.Option(min=1)] = None,
    chunker: Annotated[ChunkerName | None, typer.Option()] = None,
    embedder: Annotated[EmbedderBackend | None, typer.Option()] = None,
    reranker: Annotated[RerankerBackend | None, typer.Option()] = None,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    retrieve_only: Annotated[
        bool, typer.Option("--retrieve-only", help="Only retrieve; do not call an LLM.")
    ] = False,
    llm: Annotated[
        LLMBackend | None, typer.Option(help="openai_compatible | anthropic | ollama | fake")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Override the backend's model id.")] = None,
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="Stream tokens.")] = True,
    gate: Annotated[
        bool, typer.Option("--gate/--no-gate", help="Refuse when retrieval confidence is low.")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON.")] = False,
) -> None:
    """Answer a question with cited sources (or just retrieve with --retrieve-only)."""
    from rbirag.index.build import load_bundle
    from rbirag.index.embedder import build_embedder
    from rbirag.retrieval.pipeline import make_retriever
    from rbirag.retrieval.rerank import build_reranker

    s = get_settings()
    chunker_name = chunker or s.chunker
    in_dir = (index_dir or s.index_dir) / chunker_name
    retrieval_mode: RetrievalMode = mode or s.retrieval_mode
    needs_dense = retrieval_mode != "bm25"
    emb = (
        build_embedder(
            embedder or s.embedder,
            model_name=s.embedding_model,
            query_instruction=s.embedding_query_instruction,
            device=s.device,
            fake_dim=s.fake_embedding_dim,
        )
        if needs_dense
        else None
    )
    bundle = load_bundle(
        in_dir,
        embedder=emb,
        collection=f"{s.qdrant_collection}_{chunker_name}",
        qdrant_url=s.qdrant_url,
        qdrant_path=None if s.qdrant_url else (s.qdrant_path or in_dir / "qdrant"),
    )
    retriever = make_retriever(
        s,
        bundle.store,
        bm25=bundle.bm25,
        dense=bundle.dense,
        reranker=build_reranker(
            reranker or s.reranker, model_name=s.reranker_model, device=s.device
        )
        if retrieval_mode == "hybrid_rerank"
        else None,
        mode=retrieval_mode,
    )
    try:
        if retrieve_only:
            _print_retrieval(retriever, text, top_k or s.top_k, as_json)
        else:
            _print_answer(retriever, text, s, llm, model, top_k, stream, gate, as_json)
    finally:
        if bundle.dense is not None:
            bundle.dense.close()


def _print_retrieval(retriever: Retriever, text: str, top_k: int, as_json: bool) -> None:
    result = retriever.retrieve(text, top_k=top_k)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"mode={result.mode} timings_ms={result.timings_ms}")
    if result.confidence:
        c = result.confidence
        typer.echo(f"confidence p={c.probability:.3f} confident={c.is_confident}")
    for r in result.results:
        md = r.chunk.metadata
        typer.echo(
            f"[{r.rank}] {r.score:.4f} {r.chunk.chunk_id} | {md.get('title')} "
            f"({md.get('date')}, {md.get('circular_number')})"
        )
        typer.echo("    " + r.chunk.text[:200].replace("\n", " "))


def _print_answer(
    retriever: Retriever,
    text: str,
    s: Settings,
    backend: str | None,
    model: str | None,
    top_k: int | None,
    stream: bool,
    gate: bool,
    as_json: bool,
) -> None:
    from rbirag.generation.rag import RAGAnswer, RAGPipeline, build_llm

    pipeline = RAGPipeline.from_settings(retriever, build_llm(s, backend, model), s)
    pipeline.gate = gate and s.confidence_gate
    if top_k:
        pipeline.top_k = top_k
    answer: RAGAnswer | None = None
    if stream and not as_json:
        for event in pipeline.stream(text):
            if event.type == "delta":
                typer.echo(event.text, nl=False)
            elif event.type == "done":
                answer = event.answer
        typer.echo("")
    else:
        answer = pipeline.answer(text)
    assert answer is not None
    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    if not stream:
        typer.echo(answer.answer)
    typer.echo("")
    for src in answer.sources:
        if src.n in answer.citations.cited_ids:
            typer.echo(f"[{src.n}] {src.title} ({src.date}, {src.circular_number}) {src.url or ''}")
    flags = []
    if answer.refused:
        flags.append(f"refused={answer.refusal_reason}")
    if answer.citations.invalid_ids:
        flags.append(f"invalid citations={answer.citations.invalid_ids}")
    if answer.citations.uncited_sentences:
        flags.append(f"uncited sentences={len(answer.citations.uncited_sentences)}")
    conf = f"{answer.confidence.probability:.3f}" if answer.confidence else "n/a"
    typer.echo(
        f"-- {answer.backend or '-'}:{answer.model or '-'} confidence={conf} "
        f"tokens={answer.usage.input_tokens}+{answer.usage.output_tokens} "
        f"cost=${answer.cost.usd:.6f} (INR {answer.cost.inr:.4f}) "
        f"timings_ms={answer.timings_ms} {' '.join(flags)}"
    )


@app.command("export-web")
def export_web(
    out: Annotated[Path, typer.Option(help="Output directory (served by the static site).")] = Path(
        "web/data/index"
    ),
    docs: Annotated[Path | None, typer.Option(help="documents.jsonl")] = None,
    chunker: Annotated[
        ChunkerName | None, typer.Option(help="Default: best retrieval config.")
    ] = None,
    size: Annotated[int | None, typer.Option(help="Chunk size (tokens).")] = None,
    embedder: Annotated[EmbedderBackend | None, typer.Option()] = None,
    cap_mb: Annotated[float, typer.Option(help="Size budget for the browser index.")] = 25.0,
) -> None:
    """Export the static browser index (chunks, int8 embeddings, BM25 stats)."""
    from rbirag.document import read_documents_jsonl
    from rbirag.index.embedder import build_embedder
    from rbirag.web_export import best_config, export_web_index

    s = get_settings()
    best_chunker, best_size, source = best_config(s.results_dir)
    emb = build_embedder(
        embedder or s.embedder,
        model_name=s.embedding_model,
        query_instruction=s.embedding_query_instruction,
        batch_size=s.embedding_batch_size,
        device=s.device,
        fake_dim=s.fake_embedding_dim,
    )
    manifest = export_web_index(
        read_documents_jsonl(docs or s.documents_path),
        out,
        settings=s,
        embedder=emb,
        chunker=chunker or best_chunker,
        size=size or best_size,
        cap_mb=cap_mb,
        config_source=source if not (chunker or size) else "command line",
    )
    _echo_json(
        {
            k: manifest[k]
            for k in (
                "config",
                "config_source",
                "n_chunks",
                "n_documents",
                "years",
                "note",
                "total_bytes",
            )
        }
    )


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    reload: Annotated[bool, typer.Option(help="Auto-reload (development).")] = False,
) -> None:
    """Run the HTTP API (FastAPI) with JSON logs."""
    import uvicorn

    from rbirag.api.logging import configure_json_logging

    s = get_settings()
    configure_json_logging()
    uvicorn.run(
        "rbirag.api.main:create_app",
        factory=True,
        host=host or s.api_host,
        port=port or s.api_port,
        reload=reload,
        log_config=None,
    )


from rbirag.eval.cli import eval_app  # noqa: E402

app.add_typer(eval_app, name="eval")


if __name__ == "__main__":  # pragma: no cover
    app()
