# API

`rbirag serve` (or `docker compose up`) starts FastAPI on port 8000; the OpenAPI schema with
examples is at `/docs` and `/openapi.json`. `/v1` is the current API. The unversioned
`/query`, `/query/stream`, `/ingest`, `/health` and `/metrics` are unchanged for existing
clients.

| method | path | what it does |
|---|---|---|
| POST | `/v1/ask` | answer with verified citations (see below) |
| POST | `/v1/retrieve` | the passages a strategy retrieves, with every stage score (no LLM; cached) |
| POST | `/v1/retrieval/compare` | up to four strategies on one question, side by side, with top-k overlap |
| POST | `/v1/citations/verify` | verify an answer's `[n]` citations against given passages (texts or chunk ids) |
| POST | `/v1/eval/run` | retrieval metrics of up to four strategies on the dev/test split (guarded) |
| GET | `/v1/documents/{doc_id}` | metadata, versions, original, latest document *in the indexed corpus*, changes, related |
| GET | `/v1/documents/{doc_id}/relations` | every resolved relation in and out, and the unresolved references |
| GET | `/v1/health` | liveness and what is loaded |
| GET | `/v1/ready` | readiness (503 until an index can serve queries) |
| GET | `/metrics` | request metrics (latency per stage, abstention, failed retrieval, verdicts, errors) |

## `POST /v1/ask`

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' -d '{
  "question": "How often must banks update KYC for high-risk customers?"
}'
```

Request fields (all but `question` optional):

| field | meaning |
|---|---|
| `question` | 3-1,000 characters |
| `history` | up to 5 earlier questions; a follow-up ("and for low-risk ones?") is rewritten against the last one for retrieval, and the rewrite is returned |
| `filters` | `departments`, `categories`, `date_from`, `date_to`, `doc_ids`, applied before ranking |
| `apply_suggested_filters` | apply the filters detected in the question (they are returned either way, never applied silently) |
| `strategy` | overrides of the served retrieval configuration: `mode`, `fusion`, `bm25_weight`, `dense_weight`, `score_norm`, `rrf_k`, `rerank_depth`, `rerank_weight`, `expansion`, `parent`, `rerank_min_score`, `dense_min_similarity` |
| `top_k` | passages given to the LLM (1-10) |
| `backend` | `groq`, `gemini`, `ollama`, `anthropic` or `fake` |
| `gate` | refuse when the retrieval confidence is below the threshold (default true) |
| `refuse_out_of_scope` | refuse questions about other regulators or unrelated topics (default true) |
| `abstain_if_unsupported` | withhold an answer none of whose claims the passages support (default: list the unsupported claims) |

Response (abridged):

```json
{
  "request_id": "5c2f...", "answer": "... [1].", "refused": false, "refusal_reason": null,
  "citations": [{"n": 1, "doc_id": "rbi-12655", "title": "...", "circular_number": "...",
                 "url": "https://www.rbi.org.in/...", "section": "...", "text": "...",
                 "stage_scores": {"bm25": 21.4, "rerank": 6.1}, "stage_ranks": {"bm25": 1, "rerank": 1}}],
  "passages": ["... every passage given to the LLM ..."],
  "verification": {"verdict": "supported", "claims": [{"claim": "...", "status": "supported",
                   "supporting_source": 1, "support": 0.92}], "invalid_citations": [],
                   "unknown_references": [], "statement": null},
  "query": {"qtype": "numeric", "ambiguous": false, "follow_up": false, "out_of_scope": false,
            "filters": null},
  "context": {"kept": 5, "dropped_duplicates": [], "distinct_documents": 3, "conflicts": []},
  "confidence": {"probability": 0.997, "threshold": 0.49,
                 "definition": "sigmoid(top cross-encoder score / temperature) ... not a probability that the answer is correct."},
  "timings_ms": {"retrieve": 5.1, "rerank": 1210.4, "generate": 812.0, "verify": 3.2, "total": 2030.7}
}
```

Refusal reasons: `out_of_scope`, `needs_clarification` (too vague to retrieve for), `low_confidence`
(the gate), `no_sources`, `model_refusal`, `model_declined`, `unsupported` (with
`abstain_if_unsupported`).

## Protections

| setting (`RBIRAG_*`) | default | effect |
|---|---|---|
| `API_RATE_LIMIT_PER_MINUTE` | 30 | per client IP token bucket on every `/v1` POST; 429 with `Retry-After` (0 disables) |
| `API_MAX_BODY_BYTES` | 65,536 | larger request bodies get 413 |
| `API_TIMEOUT_SECONDS` | 180 | wall-clock budget of one `/v1/ask`; 504 after it |
| `API_EVAL_TOKEN` | unset | `/v1/eval/run` is disabled (403) unless set; requests must send it in `X-Eval-Token` (compared in constant time) |
| `RETRIEVAL_CACHE_SIZE` | 256 | in-memory LRU of retrieval results (cleared when the index changes) |
| `INTEL_GRAPH_PATH` | `data/processed/graph.json.gz` | relation graph for the document endpoints (`rbirag intel build`) |

Validation rejects unknown fields (422). Logs are JSON with a request id; they carry the stage
timings, mode, verdict and refusal reason, never the question or answer text.

## Observability

`GET /metrics` aggregates the in-memory request log (optionally mirrored to JSONL with
`RBIRAG_REQUEST_LOG_PATH`): requests and errors per endpoint, errors by component
(`index`, `retrieval`, `generation`), p50/p95 latency per stage (`retrieve`, `rerank`, `generate`,
`verify`, `total`), abstention rate and refusal reasons, failed-retrieval rate (no passage
reached the context), verification verdicts, mean accepted citations and passages per answer,
tokens and cost. The evaluation dashboard on the site is built from the recorded evaluation
runs (`results/`), not from the API.
