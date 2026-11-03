PY ?= python
START_ID ?= 13050
MAX_DOCS ?= 2200

.PHONY: help install install-all lint format typecheck test check smoke probe fetch ingest index verify eval-retrieval eval-report serve ui export-web web docker-up docker-smoke clean

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

install:  ## Light install (what CI uses): no torch
	$(PY) -m pip install -e ".[ingest,retrieval,llm,eval,api,dev]"

install-all:  ## Everything incl. sentence-transformers (torch), Anthropic SDK, tiktoken
	$(PY) -m pip install -e ".[ingest,retrieval,llm,eval,api,ui,ml,anthropic,tiktoken,web-test,dev]"

lint:  ## ruff check + format check
	ruff check .
	ruff format --check .

format:  ## Auto-format
	ruff format .
	ruff check --fix .

typecheck:  ## mypy (strict)
	mypy

test:  ## pytest (no network; browser tests need Playwright)
	pytest -m "not browser"

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

serve:  ## Run the API on :8000 (Swagger at /docs)
	rbirag serve

ui:  ## Run the Streamlit chat UI on :8501 (expects the API on :8000)
	streamlit run ui/streamlit_app.py

export-web:  ## Export the browser index into web/data/index
	rbirag export-web --out web/data/index

web:  ## Serve the static site on :8080
	$(PY) -m http.server -d web 8080

docker-up:  ## api + ui + qdrant with Docker Compose
	docker compose up -d --build

docker-smoke:  ## The CI smoke stack: fake LLM + fixture index
	docker compose -f docker-compose.yml -f docker/compose.smoke.yml up -d --build qdrant api ui

clean:  ## Remove caches and local smoke output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .smoke htmlcov .coverage
