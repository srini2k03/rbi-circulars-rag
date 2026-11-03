# rbi-circulars-rag

Hybrid-retrieval RAG over 2,000+ Reserve Bank of India notifications. It cites its
sources and says "I don't know" when retrieval confidence is low.

> **Status: work in progress.** Ingestion, chunking and retrieval are implemented.
> Generation, evaluation, the API, the UI and the full write-up are coming next.
> No results are reported yet: every number in this README will come from a real,
> reproducible run.

## What exists today

- **Ingestion:**
  - A polite, resumable crawler for RBI notification pages and their PDFs. It keeps 1 s
    or more between requests, retries with backoff, records a SHA-256 manifest, and has
    a `--probe` mode.
  - HTML and PDF parsing, using defensive selector strategies with a fallback to
    main-content extraction.
  - Text cleaning, and metadata extraction: date, circular and reference numbers,
    department, addressee and sections.
- **Chunking:** fixed-size token windows, semantic chunking (split on embedding-similarity
  drops) and section-aware chunking (headings and numbered paragraphs).
- **Retrieval:**
  - BM25 (rank_bm25) and dense search (`BAAI/bge-small-en-v1.5` on Qdrant).
  - Reciprocal Rank Fusion (k=60) and a cross-encoder reranker
    (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - A confidence score computed from the reranker scores.

## Quick start

```bash
pip install -e ".[ingest,retrieval,dev]"     # light: no torch
make check                                    # ruff, mypy, pytest (offline)
make smoke                                    # offline end-to-end run on fixtures

pip install -e ".[ml]"                        # real embedder + reranker
rbirag fetch --probe                          # check the parser against live RBI pages
rbirag fetch --max-docs 2200 && rbirag ingest
rbirag index --chunker section
rbirag query "What is the KYC updation period for high risk customers?" --mode hybrid_rerank
```

The corpus is built on GitHub Actions (`corpus.yml`) and published as the `corpus-v1`
release asset. It is not committed to the repository.

## License

MIT, see [LICENSE](LICENSE).
