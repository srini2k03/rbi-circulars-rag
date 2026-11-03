PY ?= python
START_ID ?= 13050
MAX_DOCS ?= 2200

.PHONY: help install install-all lint format typecheck test check smoke probe fetch ingest index verify eval-retrieval eval-report clean

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

install:  ## Light install (what CI uses): no torch
	$(PY) -m pip install -e ".[ingest,retrieval,llm,eval,dev]"

install-all:  ## Everything incl. sentence-transformers (torch), Anthropic SDK, tiktoken
	$(PY) -m pip install -e ".[ingest,retrieval,llm,eval,ml,anthropic,tiktoken,dev]"

lint:  ## ruff check + format check
	ruff check .
	ruff format --check .

format:  ## Auto-format
	ruff format .
	ruff check --fix .

typecheck:  ## mypy (strict)
	mypy

test:  ## pytest (no network)
	pytest

check: lint typecheck test  ## Everything CI runs

smoke:  ## Offline end-to-end run on the HTML fixtures
	./scripts/smoke_fixtures.sh

probe:  ## Fetch ~5 RBI ids and print parser diagnostics
	rbirag fetch --probe --start-id $(START_ID)

fetch:  ## Crawl RBI notifications (polite; resumable)
	rbirag fetch --start-id $(START_ID) --max-docs $(MAX_DOCS)

ingest:  ## Raw pages -> data/processed/documents.jsonl (+ stats)
	rbirag ingest --stats-out data/processed/stats.json

index:  ## Build BM25 + Qdrant indexes for every chunker
	for c in fixed semantic section; do rbirag index --chunker $$c; done

verify:  ## Review Q/A candidates (accept/edit/reject); resumable
	rbirag eval verify

eval-retrieval:  ## Retrieval ablation (all chunkers x sizes x modes)
	rbirag eval retrieval

eval-report:  ## Tables + charts into results/
	rbirag eval report

clean:  ## Remove caches and local smoke output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .smoke htmlcov .coverage
