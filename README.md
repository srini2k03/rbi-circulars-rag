# rbi-circulars-rag

Hybrid-retrieval RAG over 2,000+ Reserve Bank of India notifications. It cites its
sources and says "I don't know" when retrieval confidence is low.

> **Status: work in progress.** Ingestion, retrieval, generation and the evaluation
> harness are implemented. The API, the UI and the full write-up come next. No results
> are reported yet: every number in this README will come from a real, reproducible run
> whose date, commit, judge model and human-verified count are recorded in
> `results/run_metadata.json`.

## What exists today

- **Ingestion:**
  - A polite, resumable crawler for RBI notification pages (at least 1 s between
    requests, retries with backoff, a SHA-256 manifest, and a `--probe` mode).
  - HTML/PDF parsing with defensive selector strategies, text cleaning, and metadata:
    date, circular and reference numbers, department, addressee and sections.
- **Chunking:** fixed-size token windows, semantic (split on embedding-similarity drops)
  and section-aware (headings and numbered paragraphs).
- **Retrieval:** BM25 + dense (`BAAI/bge-small-en-v1.5` on Qdrant), Reciprocal Rank
  Fusion (k=60), a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), and a
  confidence score from the reranker.
- **Generation:**
  - A prompt with numbered sources that asks for `[n]` citations after every factual
    sentence.
  - Citation parsing and validation: every cited id must exist, and uncited sentences
    are flagged.
  - A confidence gate that answers "I don't know" without calling the LLM when retrieval
    is weak.
  - Streaming, plus token and cost accounting.
- **LLM backends:**
  - **Groq, free tier** (the default API backend, through Groq's OpenAI-compatible
    endpoint). Google Gemini's free tier is a built-in alternative
    (`RBIRAG_LLM_PROVIDER=gemini`); any other OpenAI-compatible provider, such as
    OpenRouter, works with `RBIRAG_LLM_PROVIDER=custom` plus a base URL, models and the
    name of the key's environment variable.
  - **Ollama** for a local model (`qwen2.5:3b-instruct` by default).
  - An optional **Anthropic** backend, used only when selected and `ANTHROPIC_API_KEY`
    is set.

  Free-tier hygiene:
  - client-side limits per model: requests per minute, requests per day and a
    tokens-per-minute budget (Groq defaults: 25 requests/min; 950/day for the 70B-class
    models and 14,000/day for Llama 3.1 8B; a token budget just under the published
    per-minute limits);
  - exponential backoff on 429 and 5xx;
  - a fallback chain of models: after 2 consecutive 503 "overloaded" responses the
    request moves to the next model, and the overloaded model is skipped for a few
    minutes; a model that returns 404/403 is dropped, and the run aborts only if every
    model in the chain fails that way;
  - a clean stop when the daily quota is used up;
  - an on-disk cache of successful LLM calls only (errors are never cached), so reruns
    cost nothing.
- **Evaluation:**
  - A test set: about 220 candidate questions sampled by year and department (factual,
    numeric, procedural, multi-chunk), plus about 30 deliberately unanswerable ones,
    which you then review by hand.
  - Retrieval metrics: Recall@k, MRR@10 and nDCG@10, scored at both document and chunk
    level against character-span gold evidence, so different chunkers are compared
    fairly.
  - Generation metrics:
    - faithfulness from the LLM judge (it splits each answer into claims and checks
      each one);
    - answer relevance, from the judge and from embedding similarity;
    - citation accuracy and an NLI faithfulness cross-check (DeBERTa NLI, no API
      needed);
    - precision and recall of refusals on the unanswerable set.
  - Bootstrap 95% CIs, an ablation over 4 retrieval modes × 3 chunkers × 3 chunk sizes,
    and calibration of the confidence gate on a dev split.
  - CSV/Markdown tables and matplotlib charts in `results/`.

### Which models produce the numbers

Groq free tier (Llama 3.3 70B judge, Llama 3.1 8B generator), plus local Ollama
Qwen2.5-3B.

- **API models (Groq), configurable per role:**

  | Role | Primary model | Fallbacks |
  |---|---|---|
  | API generator | `llama-3.1-8b-instant` | `openai/gpt-oss-20b` |
  | Judge | `llama-3.3-70b-versatile` (a different, larger model than the generator) | `openai/gpt-oss-120b`, `qwen/qwen3-32b` |
  | Q/A writer | `llama-3.3-70b-versatile` | `openai/gpt-oss-120b`, `qwen/qwen3-32b` |

  Every record stores the model that actually answered, and `results/README.md` reports
  the mix (for example "judge: llama-3.3-70b-versatile 70%, openai/gpt-oss-120b 30%").
  The first four Q/A candidates were written by Gemini (`gemini-3.5-flash`) before the
  switch to Groq; each pair records its writer model, and the results README shows the
  writer mix.
- **Local model:** Ollama `qwen2.5:3b-instruct` is evaluated alongside, and judged by
  the same Groq judge.
- **Judge-independent check:** NLI-based faithfulness and citation accuracy are reported
  next to the judge's scores, because a small free judge can be lenient.
- **Cost:** the free tier is reported as **$0**; token counts are still recorded per
  answer.

## Quick start

```bash
pip install -e ".[ingest,retrieval,llm,eval,dev]"   # light: no torch
make check                                         # ruff, mypy, pytest (offline)
make smoke                                         # offline end-to-end run on fixtures

pip install -e ".[ml]"                             # real embedder, reranker, NLI model
export GROQ_API_KEY=...                            # free key from console.groq.com
rbirag fetch --max-docs 2200 && rbirag ingest
rbirag index --chunker section
rbirag query "What is the KYC updation period for high risk customers?"
rbirag query "..." --llm ollama                    # local model
rbirag query "..." --retrieve-only --json          # retrieval only
```

## Reviewing the test set (about 30–40 minutes, on your own machine)

The `qa.yml` workflow writes model-generated candidates to `data/eval/qa_candidates.jsonl`.
Reviewing them by hand is what makes the test set "human-verified"; the README will state
exactly how many pairs were reviewed.

```bash
git clone https://github.com/srini2k03/rbi-circulars-rag && cd rbi-circulars-rag
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                                      # the review tool needs no extras
rbirag eval verify --limit 50                         # review 50 random pairs
```

For each pair you see the source passage (with the evidence highlighted), the question
and the reference answer. Press:

- **a** to accept;
- **e** to edit the question and/or answer (press Enter to keep a field as it is);
- **r** to reject, with an optional reason;
- **s** to skip;
- **q** to quit.

Every decision is saved immediately to `data/eval/qa_v1.jsonl`. Run the command again to
continue where you stopped. When you are done:

```bash
git add data/eval/qa_v1.jsonl
git commit -m "Review Q/A pairs"
git push
```

`verified: true` marks pairs a human accepted or edited. Rejected pairs are excluded from
evaluation, as are pairs that failed the automatic checks; unreviewed pairs are still
used and are reported as model-generated.

## Workflows

| Workflow | What it does |
|---|---|
| `ci.yml` | ruff, mypy, pytest (Python 3.11/3.12, no network, no torch) and the fixture smoke run |
| `corpus.yml` | Crawls RBI and publishes the processed corpus as release `corpus-v1` |
| `qa.yml` | Generates Q/A candidates with the API provider (Groq by default) and commits `data/eval/qa_candidates.jsonl` |
| `evaluate.yml` | Runs the retrieval ablation, gate calibration, generation (API provider and Ollama), judging and the report, then commits `results/` |

Both take a `provider` input (`groq` or `gemini`). They need the matching repository
secret (`GROQ_API_KEY` or `GEMINI_API_KEY`), fail early (without printing it) if it is
missing, and list the models the key can use. Each provider has its own concurrency
group, so runs on the same provider never compete for its quota. Both stop cleanly when the daily free-tier quota
runs out: rerun them later and they continue from the committed records and the cached
LLM responses. `evaluate.yml` prints the planned number of LLM calls before generating.

Guards against misleading output:
- an unknown model or rejected key (404/401/403 on every model in the chain) aborts the
  run with a clear message, and nothing from that run is committed;
- a run that produced no successful items commits nothing;
- a configuration whose generation or judge calls mostly failed (fewer than 90% of the
  planned questions succeeded) is shown as "incomplete (k/N)" and kept out of the
  headline tables.

## License

MIT, see [LICENSE](LICENSE).
