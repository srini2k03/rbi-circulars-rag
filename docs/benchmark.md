# Retrieval benchmark

Every retrieval strategy in `rbirag.retrieval`, compared on the same held-out questions.
Source of every number below: [`results/benchmark/strategies.json`](../results/benchmark/strategies.json),
produced by the `Retrieval benchmark` workflow, run
[45364004807](https://github.com/srini2k03/rbi-circulars-rag/actions/runs/45364004807)
at commit `1c16491` on 2026-11-24 (logged in [`results/experiments.jsonl`](../results/experiments.jsonl)).

## Protocol

- **Questions.** The model-generated test set (`data/eval/qa_candidates.jsonl`, AI-review applied,
  gold spans verified against the corpus): 216 questions, split **by document** into dev
  (61 questions, 56 answerable) and test (155 questions, 130 answerable). No document has
  questions in both splits. The questions are model-written and AI-reviewed (LLM-assisted);
  none is human-verified.
- **Index.** `fixed-1024` chunks (5,618 passages from 2,153 notifications), the configuration the
  original ablation selected on dev. Parent-child strategies retrieve over `fixed-256`
  (19,299 passages) and return ~1,024-token windows.
- **Selection.** Each family (BM25 + RM3, weighted fusion, rerank depth and interpolation, ...) is
  tuned on **dev** by chunk-level nDCG@10; 108 configurations were tried. Only the dev-selected
  configuration of each family is run on **test**. Ties keep the simpler configuration.
- **Statistics.** 95% percentile-bootstrap CIs (1,000 resamples); for each strategy the paired
  difference to plain BM25 with a bootstrap CI and a sign-flip randomisation test (10,000
  permutations), with Holm-Bonferroni adjustment across the 16 comparisons.
- **Same code as served.** Strategies run through the production `Retriever`. Dense and
  cross-encoder scores are cached (`score_cache-*.jsonl.gz`, committed) so the comparison
  replays without the models (`rbirag eval bench --no-models`); the workflow verifies that the
  replay reproduces the run, and a live pass with the real models reproduced the cached top-10
  for 155/155 test questions in every strategy except one (154/155, a floating-point tie).
- **Check against history.** The original configurations reproduce the committed 36-config
  ablation exactly: fixed-1024 BM25 0.697, hybrid RRF 0.582, hybrid + rerank 0.525, dense 0.481.

## Results (test split, chunk-level nDCG@10)

| strategy (dev-selected) | dev | test [95% CI] | Δ vs BM25 [95% CI] | Holm p | chunk R@5 | doc R@5 |
|---|---|---|---|---|---|---|
| BM25, filter = gold document's year (oracle) | 0.859 | 0.836 [0.790, 0.886] | +0.139 [+0.103, +0.176] | 0.002 | 0.931 | 0.962 |
| BM25, filter = gold document's department (oracle) | 0.726 | 0.751 [0.694, 0.812] | +0.054 [+0.029, +0.083] | 0.002 | 0.823 | 0.885 |
| parent-child (256-token children, 1,024-token windows) | 0.629 | 0.715 [0.654, 0.776] | +0.017 [-0.032, +0.067] | 1.000 | 0.788 | 0.831 |
| weighted fusion (z-score, BM25 0.7 / dense 0.3) + RM3 | 0.711 | 0.704 [0.646, 0.762] | +0.007 [-0.026, +0.039] | 1.000 | 0.785 | 0.838 |
| BM25 + RM3 (3 feedback passages, 10 terms, 0.85 original) | 0.705 | 0.704 [0.645, 0.766] | +0.006 [-0.012, +0.024] | 1.000 | 0.815 | 0.846 |
| **weighted fusion (z-score, 0.7 / 0.3)**, best deployable on dev | **0.713** | 0.704 [0.644, 0.766] | +0.006 [-0.023, +0.035] | 1.000 | 0.777 | 0.838 |
| BM25 + acronym expansion (weight 0.25) | 0.644 | 0.700 [0.642, 0.766] | +0.003 [-0.007, +0.014] | 1.000 | 0.785 | 0.854 |
| **BM25** (baseline) | 0.668 | 0.697 [0.638, 0.763] | - | - | 0.785 | 0.846 |
| weighted fusion + cross-encoder (depth 20, weight 0.25) | 0.710 | 0.688 [0.629, 0.746] | -0.010 [-0.047, +0.026] | 1.000 | 0.792 | 0.846 |
| document-first (passages from the top-20 documents) | 0.676 | 0.687 [0.628, 0.754] | -0.010 [-0.023, +0.000] | 1.000 | 0.762 | 0.823 |
| BM25 + cross-encoder (depth 20, weight 0.5) | 0.705 | 0.663 [0.603, 0.723] | -0.035 [-0.072, +0.002] | 0.527 | 0.769 | 0.831 |
| BM25, filters detected in the question | 0.659 | 0.660 [0.594, 0.729] | -0.037 [-0.074, -0.006] | 0.285 | 0.746 | 0.808 |
| hybrid RRF, tuned (k = 10, BM25 weight 1.5) | 0.696 | 0.639 [0.579, 0.696] | -0.058 [-0.098, -0.020] | 0.034 | 0.765 | 0.838 |
| parent-child + cross-encoder | 0.667 | 0.634 [0.572, 0.692] | -0.064 [-0.121, -0.007] | 0.288 | 0.750 | 0.831 |
| hybrid RRF, original (k = 60, equal weights) | 0.640 | 0.582 [0.520, 0.645] | -0.116 [-0.173, -0.060] | 0.002 | 0.681 | 0.800 |
| hybrid RRF + cross-encoder, original | 0.620 | 0.525 [0.462, 0.594] | -0.172 [-0.240, -0.105] | 0.002 | 0.581 | 0.738 |
| dense (bge-small) | 0.512 | 0.481 [0.414, 0.550] | -0.216 [-0.293, -0.142] | 0.002 | 0.569 | 0.738 |

Precision@5 is between 0.12 and 0.20 for every strategy because most questions have a single
gold passage (P@5 cannot exceed 0.2 then); context precision, coverage and context-token counts
are in `strategies.json`.

## What the numbers say

1. **Nothing deployable beats plain BM25 with significance.** The best configuration on dev,
   z-score weighted fusion (0.713 on dev), lands at 0.704 on test, +0.006 over BM25 with a CI
   that straddles zero. RM3 and acronym expansion behave the same way. The dev gains of the
   tuned families (up to +0.045) shrink on test: selection on 56 dev questions overfits.
2. **The original hybrid was hurt by how it fused, not by having a dense arm.** Equal-weight RRF
   gives the much weaker dense retriever's ranks the same vote as BM25's, which cost 0.116
   nDCG@10. Normalising the scores and weighting BM25 0.7 recovers BM25-level quality.
3. **The MS MARCO cross-encoder does not help ranking here.** As a reranker it lowers nDCG@10
   (BM25 + rerank -0.035; the original hybrid + rerank -0.172), and the interpolation weight
   chosen on dev is low (0.25-0.5) - the reranker is mostly overruled.
4. **But it is the best abstention signal.** On the 25 unanswerable test questions, the top
   cross-encoder probability over the BM25 top 20 separates them from answerable ones with
   AUROC 0.952 (dense top cosine 0.902, BM25 top score 0.804). At the threshold chosen on dev
   (0.49; only 5 unanswerable dev questions, so this threshold is fragile) it refuses with
   precision 0.77 and recall 0.68 while answering 96% of the answerable questions.
   This is why the served default ranks with BM25 and uses the cross-encoder only as the gate
   (`retrieval_mode=bm25_rerank`, `rerank_weight=0`).
5. **Metadata filters are the largest lever, if the user supplies them.** Restricting to the
   gold document's year (an oracle, not a deployable system) adds +0.139; its department +0.054.
   Filters inferred automatically from the question text *hurt* (-0.037): years in questions
   often name a direction's title ("Directions, 2025") rather than the publication date, and a
   department filter drops the 443 documents with no department. The API and the site therefore
   only *suggest* detected filters and apply explicit ones.
6. **Parent-child and document-level retrieval do not change the picture.** Parent windows built
   from 256-token passages score +0.017 (n.s.) at the same context budget (~4,400 vs ~4,200
   tokens); ranking whole documents first is -0.010. Document-level BM25 (title + full text) and
   passage BM25 aggregated to documents find the gold document in the top 5 equally often
   (0.854 both), passage aggregation is better at rank 1 (0.585 vs 0.569).

## Where the gold passage ends up (test, 130 answerable questions)

| strategy | ranked 1st | ranks 2-5 | retrieved, below 5 | right document, other passage | related document | not retrieved |
|---|---|---|---|---|---|---|
| BM25 | 69 | 33 | 26 | 1 | 0 | 1 |
| weighted fusion (z-score 0.7/0.3) | 70 | 31 | 27 | 1 | 0 | 1 |
| BM25 + cross-encoder | 60 | 40 | 28 | 0 | 1 | 1 |
| hybrid RRF, original | 49 | 40 | 39 | 1 | 1 | 0 |
| hybrid RRF + cross-encoder, original | 45 | 31 | 52 | 0 | 1 | 1 |
| dense | 43 | 32 | 53 | 1 | 0 | 1 |

The dominant failure is **ranking, not recall**: for BM25, 26 of the 28 misses are passages that a
first-stage retriever does find (in the final top 10 or in the BM25 / dense top 50) but ranks
below the 5-passage context. Only 1-2 questions per strategy have no trace of the gold document.
Categories are defined in `strategies.json` (`failure_categories`); "related document" uses
title families.

## Latency (live pass on a GitHub-hosted 4-vCPU runner, 155 test questions)

| strategy | p50 ms | p95 ms |
|---|---|---|
| BM25 (postings scorer) | 4.0 | 4.5 |
| BM25 + RM3 | 8.8 | 9.6 |
| parent-child BM25 | 13.2 | 14.9 |
| dense (query embedding + search) | 73.2 | 120.4 |
| weighted fusion | 78.0 | 91.6 |
| BM25 + cross-encoder (20 passages) | 1,221 | 1,276 |
| hybrid RRF + cross-encoder, original | 1,352 | 1,380 |

Only the live pass measures model time; the `latency_ms_*` columns of `strategies.csv` time the
cached replay and must not be read as serving latency. The BM25 scorer was rewritten to use
postings: bit-identical scores to `rank_bm25` (unit-tested) and, measured locally on the
sandbox's 4 CPUs over the 216 questions, p50 26.8 ms -> 4.3 ms on the 1,024-token index.

## Reproduce

```bash
gh workflow run benchmark.yml            # or: Actions -> Retrieval benchmark -> Run workflow
# replay locally from the committed score cache (no models, no network):
rbirag eval bench --no-models --docs data/processed/documents.jsonl --out-dir /tmp/bench
```

The replay needs the `corpus-v1` release (`gh release download corpus-v1`).
