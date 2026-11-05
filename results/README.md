# Evaluation results

**Test set: model-generated test set (`data/eval/qa_candidates.jsonl`): 216 model-generated pairs (43 AI-reviewed by Claude, 0 human-verified); 7 AI-rejected pairs excluded.**

Generated 2026-11-05T18:46:17+00:00 at commit `050f1ee86d90556a20a689c90790de388920c0f2`. Questions: 216 (186 answerable, 30 unanswerable); human-verified: 0; AI-reviewed by Claude (not human verification): 43, with 7 AI-rejected pairs excluded.

**Split by document:** 155 test questions (130 documents) for every reported number; 61 dev questions (56 documents) used only to choose configurations and calibrate the gate. No document contributes questions to both.

Models that actually answered (share of calls):

- generator (ollama): qwen2.5:3b-instruct 100%
- generator (openai_compatible): openai/gpt-oss-20b 91%, qwen/qwen3.8-27b 5%, openai/gpt-oss-120b 3%
- judge: qwen2.5:7b-instruct 100%
- Q/A writer: openai/gpt-oss-120b 83%, qwen/qwen3.8-27b 15%, gemini-3.5-flash 2%

## Retrieval ablation

| config | chunks | doc R@5 | chunk R@1 | chunk R@5 | chunk R@10 | chunk MRR@10 | chunk nDCG@10 [95% CI] | p50 ms |
|---|---|---|---|---|---|---|---|---|
| fixed-512-bm25 | 10131 | 0.846 | 0.531 | 0.800 | 0.881 | 0.650 | 0.703 [0.643, 0.764] | 40.1 |
| fixed-1024-bm25 | 5618 | 0.846 | 0.527 | 0.785 | 0.885 | 0.639 | 0.697 [0.638, 0.763] | 22.4 |
| semantic-1024-bm25 | 12497 | 0.869 | 0.496 | 0.785 | 0.846 | 0.637 | 0.679 [0.622, 0.743] | 71.2 |
| semantic-512-bm25 | 15152 | 0.815 | 0.481 | 0.742 | 0.850 | 0.620 | 0.666 [0.609, 0.728] | 86.3 |
| section-1024-bm25 | 19369 | 0.831 | 0.458 | 0.746 | 0.827 | 0.610 | 0.654 [0.598, 0.717] | 118.8 |
| fixed-256-bm25 | 19299 | 0.831 | 0.477 | 0.723 | 0.835 | 0.606 | 0.654 [0.591, 0.716] | 67.3 |
| semantic-256-bm25 | 22796 | 0.831 | 0.488 | 0.700 | 0.812 | 0.614 | 0.648 [0.586, 0.712] | 131.8 |
| section-512-bm25 | 29655 | 0.815 | 0.450 | 0.704 | 0.792 | 0.613 | 0.641 [0.579, 0.700] | 203.1 |
| semantic-256-hybrid | 22796 | 0.846 | 0.454 | 0.692 | 0.831 | 0.590 | 0.637 [0.574, 0.700] | 190.9 |
| semantic-512-hybrid | 15152 | 0.862 | 0.423 | 0.762 | 0.835 | 0.582 | 0.636 [0.580, 0.694] | 144.2 |
| section-512-hybrid | 29655 | 0.854 | 0.427 | 0.704 | 0.812 | 0.590 | 0.632 [0.571, 0.693] | 247.3 |
| semantic-1024-hybrid | 12497 | 0.854 | 0.446 | 0.715 | 0.812 | 0.576 | 0.625 [0.562, 0.691] | 132.1 |
| section-1024-hybrid_rerank | 19369 | 0.823 | 0.454 | 0.681 | 0.804 | 0.577 | 0.625 [0.558, 0.693] | 2374.0 |
| section-256-hybrid | 43864 | 0.831 | 0.446 | 0.669 | 0.758 | 0.605 | 0.623 [0.557, 0.690] | 313.4 |
| fixed-512-hybrid | 10131 | 0.823 | 0.408 | 0.731 | 0.831 | 0.561 | 0.623 [0.562, 0.686] | 123.5 |
| section-512-hybrid_rerank | 29655 | 0.862 | 0.450 | 0.692 | 0.765 | 0.589 | 0.617 [0.549, 0.681] | 2423.9 |
| section-1024-hybrid | 19369 | 0.831 | 0.415 | 0.704 | 0.781 | 0.570 | 0.612 [0.550, 0.675] | 173.1 |
| semantic-512-hybrid_rerank | 15152 | 0.831 | 0.423 | 0.677 | 0.808 | 0.558 | 0.611 [0.550, 0.673] | 2200.6 |
| section-256-bm25 | 43864 | 0.831 | 0.415 | 0.681 | 0.750 | 0.581 | 0.601 [0.538, 0.665] | 284.7 |
| semantic-256-hybrid_rerank | 22796 | 0.823 | 0.427 | 0.662 | 0.765 | 0.551 | 0.595 [0.531, 0.661] | 1567.9 |
| fixed-512-hybrid_rerank | 10131 | 0.815 | 0.388 | 0.692 | 0.796 | 0.519 | 0.584 [0.516, 0.648] | 1343.8 |
| fixed-1024-hybrid | 5618 | 0.800 | 0.373 | 0.681 | 0.800 | 0.512 | 0.582 [0.520, 0.645] | 91.3 |
| fixed-256-hybrid | 19299 | 0.846 | 0.358 | 0.704 | 0.781 | 0.522 | 0.579 [0.511, 0.644] | 159.7 |
| section-256-hybrid_rerank | 43864 | 0.854 | 0.431 | 0.631 | 0.712 | 0.559 | 0.576 [0.511, 0.644] | 1638.3 |
| semantic-1024-hybrid_rerank | 12497 | 0.777 | 0.373 | 0.638 | 0.785 | 0.512 | 0.572 [0.511, 0.637] | 2219.2 |
| section-512-dense | 29655 | 0.831 | 0.377 | 0.658 | 0.735 | 0.531 | 0.570 [0.507, 0.636] | 87.8 |
| section-256-dense | 43864 | 0.831 | 0.388 | 0.665 | 0.708 | 0.543 | 0.567 [0.502, 0.637] | 98.6 |
| fixed-256-hybrid_rerank | 19299 | 0.800 | 0.350 | 0.665 | 0.754 | 0.501 | 0.560 [0.494, 0.627] | 1047.4 |
| section-1024-dense | 19369 | 0.769 | 0.346 | 0.623 | 0.719 | 0.491 | 0.539 [0.476, 0.607] | 75.9 |
| semantic-256-dense | 22796 | 0.785 | 0.369 | 0.615 | 0.685 | 0.499 | 0.535 [0.468, 0.606] | 83.0 |
| semantic-512-dense | 15152 | 0.777 | 0.373 | 0.619 | 0.685 | 0.498 | 0.535 [0.465, 0.607] | 76.2 |
| fixed-1024-hybrid_rerank | 5618 | 0.738 | 0.346 | 0.581 | 0.742 | 0.458 | 0.525 [0.462, 0.594] | 1326.7 |
| semantic-1024-dense | 12497 | 0.769 | 0.369 | 0.577 | 0.662 | 0.478 | 0.517 [0.449, 0.591] | 72.2 |
| fixed-512-dense | 10131 | 0.700 | 0.315 | 0.562 | 0.658 | 0.441 | 0.487 [0.420, 0.556] | 53.6 |
| fixed-1024-dense | 5618 | 0.738 | 0.327 | 0.569 | 0.635 | 0.433 | 0.481 [0.414, 0.550] | 48.8 |
| fixed-256-dense | 19299 | 0.769 | 0.242 | 0.569 | 0.650 | 0.406 | 0.457 [0.397, 0.530] | 69.6 |

## Dev split (selection only, not a result)

The configuration for generation and the gate threshold were chosen on these dev questions; they are shown for transparency and are not reported results.

| config | chunks | doc R@5 | chunk R@1 | chunk R@5 | chunk R@10 | chunk MRR@10 | chunk nDCG@10 [95% CI] | p50 ms |
|---|---|---|---|---|---|---|---|---|
| fixed-1024-bm25 | 5618 | 0.786 | 0.500 | 0.732 | 0.857 | 0.609 | 0.668 [0.556, 0.761] | 21.8 |
| fixed-1024-hybrid | 5618 | 0.839 | 0.446 | 0.750 | 0.821 | 0.581 | 0.640 [0.537, 0.735] | 88.6 |
| fixed-512-bm25 | 10131 | 0.821 | 0.446 | 0.696 | 0.848 | 0.576 | 0.639 [0.541, 0.728] | 39.1 |
| fixed-1024-hybrid_rerank | 5618 | 0.821 | 0.429 | 0.732 | 0.804 | 0.561 | 0.620 [0.504, 0.713] | 1330.2 |
| fixed-512-hybrid_rerank | 10131 | 0.875 | 0.411 | 0.714 | 0.830 | 0.533 | 0.602 [0.503, 0.690] | 1332.8 |
| fixed-512-hybrid | 10131 | 0.893 | 0.375 | 0.750 | 0.795 | 0.536 | 0.598 [0.502, 0.694] | 123.1 |
| semantic-1024-bm25 | 12497 | 0.839 | 0.384 | 0.679 | 0.786 | 0.524 | 0.585 [0.483, 0.682] | 69.1 |
| fixed-256-hybrid_rerank | 19299 | 0.821 | 0.393 | 0.643 | 0.750 | 0.536 | 0.574 [0.468, 0.678] | 1037.9 |
| fixed-256-bm25 | 19299 | 0.821 | 0.375 | 0.634 | 0.741 | 0.507 | 0.559 [0.450, 0.663] | 65.7 |
| semantic-512-bm25 | 15152 | 0.804 | 0.375 | 0.634 | 0.750 | 0.503 | 0.557 [0.447, 0.654] | 89.4 |
| semantic-1024-hybrid_rerank | 12497 | 0.750 | 0.357 | 0.607 | 0.768 | 0.500 | 0.554 [0.447, 0.651] | 2215.2 |
| semantic-512-hybrid_rerank | 15152 | 0.750 | 0.366 | 0.616 | 0.741 | 0.504 | 0.547 [0.446, 0.641] | 2204.4 |
| semantic-256-bm25 | 22796 | 0.821 | 0.393 | 0.598 | 0.679 | 0.518 | 0.543 [0.434, 0.647] | 135.3 |
| section-1024-bm25 | 19369 | 0.875 | 0.375 | 0.670 | 0.714 | 0.500 | 0.539 [0.432, 0.640] | 113.4 |
| semantic-256-hybrid | 22796 | 0.821 | 0.375 | 0.562 | 0.643 | 0.499 | 0.517 [0.407, 0.622] | 186.8 |
| fixed-256-hybrid | 19299 | 0.821 | 0.312 | 0.598 | 0.741 | 0.457 | 0.514 [0.415, 0.613] | 159.1 |
| fixed-1024-dense | 5618 | 0.768 | 0.286 | 0.661 | 0.714 | 0.446 | 0.512 [0.406, 0.618] | 48.6 |
| semantic-1024-hybrid | 12497 | 0.786 | 0.286 | 0.616 | 0.723 | 0.458 | 0.511 [0.414, 0.605] | 131.7 |
| semantic-512-hybrid | 15152 | 0.768 | 0.277 | 0.562 | 0.741 | 0.447 | 0.510 [0.405, 0.608] | 144.1 |
| fixed-512-dense | 10131 | 0.804 | 0.259 | 0.652 | 0.759 | 0.437 | 0.508 [0.423, 0.599] | 53.1 |
| section-1024-hybrid_rerank | 19369 | 0.804 | 0.357 | 0.598 | 0.652 | 0.483 | 0.503 [0.392, 0.605] | 2371.7 |
| section-512-hybrid_rerank | 29655 | 0.732 | 0.321 | 0.527 | 0.670 | 0.462 | 0.491 [0.384, 0.587] | 2469.8 |
| section-256-hybrid_rerank | 43864 | 0.786 | 0.348 | 0.518 | 0.634 | 0.465 | 0.485 [0.379, 0.588] | 1614.7 |
| semantic-256-hybrid_rerank | 22796 | 0.804 | 0.321 | 0.562 | 0.643 | 0.454 | 0.482 [0.371, 0.588] | 1554.5 |
| section-1024-hybrid | 19369 | 0.804 | 0.277 | 0.554 | 0.661 | 0.422 | 0.472 [0.369, 0.568] | 166.9 |
| section-512-bm25 | 29655 | 0.786 | 0.286 | 0.545 | 0.679 | 0.417 | 0.468 [0.362, 0.575] | 195.8 |
| section-256-hybrid | 43864 | 0.786 | 0.250 | 0.509 | 0.670 | 0.407 | 0.454 [0.354, 0.540] | 301.4 |
| section-512-hybrid | 29655 | 0.750 | 0.250 | 0.473 | 0.661 | 0.402 | 0.451 [0.348, 0.549] | 235.9 |
| section-256-bm25 | 43864 | 0.768 | 0.259 | 0.527 | 0.607 | 0.402 | 0.441 [0.337, 0.544] | 273.9 |
| fixed-256-dense | 19299 | 0.625 | 0.241 | 0.464 | 0.643 | 0.370 | 0.423 [0.324, 0.517] | 69.3 |
| section-1024-dense | 19369 | 0.768 | 0.223 | 0.500 | 0.598 | 0.353 | 0.408 [0.312, 0.504] | 75.5 |
| semantic-256-dense | 22796 | 0.786 | 0.205 | 0.491 | 0.598 | 0.352 | 0.402 [0.303, 0.497] | 84.2 |
| section-256-dense | 43864 | 0.732 | 0.268 | 0.455 | 0.527 | 0.380 | 0.400 [0.291, 0.503] | 98.2 |
| semantic-512-dense | 15152 | 0.732 | 0.196 | 0.446 | 0.598 | 0.321 | 0.383 [0.285, 0.480] | 76.1 |
| section-512-dense | 29655 | 0.643 | 0.232 | 0.384 | 0.554 | 0.346 | 0.382 [0.274, 0.483] | 87.1 |
| semantic-1024-dense | 12497 | 0.696 | 0.214 | 0.429 | 0.571 | 0.325 | 0.380 [0.276, 0.478] | 71.2 |

## Generation

**Judge: qwen2.5:7b-instruct, run locally with Ollama on CPU** (292 s per verdict on a CI runner); agreement with earlier judges: qwen/qwen3.8-27b n=23: faithfulness r=0.71, relevance weighted kappa=0.78; openai/gpt-oss-120b n=7: faithfulness r=-, relevance weighted kappa=0.00. NLI faithfulness and citation metrics are judge-independent.

| config | backend (model) | questions | n | judge | faithfulness (judge) | faithfulness (NLI) | relevance (judge) | citation acc. (NLI) | citations valid | refusal P | refusal R | p50 ms | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-1024-bm25 | ollama (qwen2.5:3b-instruct 100%) | full test set (n=155) | 155 | qwen2.5:7b-instruct | 0.953 [0.918, 0.984] | 0.212 [0.147, 0.280] | 0.941 [0.902, 0.975] | 0.170 [0.095, 0.264] | 0.438 | 0.913 [0.782, 1.000] | 0.840 [0.678, 0.966] | 245980 | 0.0000 |
| fixed-1024-bm25 | openai_compatible (openai/gpt-oss-20b 91%, qwen/qwen3.8-27b 5%, openai/gpt-oss-120b 3%) | fixed subset (n=60) | 60 | qwen2.5:7b-instruct | 0.986 [0.965, 1.000] | 0.131 [0.036, 0.238] | 0.977 [0.936, 1.000] | 0.185 [0.037, 0.333] | 0.488 | 0.615 [0.333, 0.875] | 0.800 [0.500, 1.000] | 57435 | 0.0000 |
| fixed-1024-hybrid | ollama (qwen2.5:3b-instruct 100%) | fixed subset (n=60) | 60 | qwen2.5:7b-instruct | 0.891 [0.783, 0.978] | 0.289 [0.167, 0.416] | 0.851 [0.745, 0.937] | 0.304 [0.130, 0.478] | 0.404 | 0.750 [0.499, 1.000] | 0.900 [0.667, 1.000] | 237035 | 0.0000 |
| fixed-1024-hybrid_rerank | ollama (qwen2.5:3b-instruct 100%) | fixed subset (n=60) | 60 | qwen2.5:7b-instruct | 0.918 [0.848, 0.978] | 0.290 [0.170, 0.402] | 0.897 [0.810, 0.968] | 0.185 [0.062, 0.333] | 0.478 | 0.714 [0.462, 0.929] | 1.000 [1.000, 1.000] | 237289 | 0.0000 |

## Judge agreement

Agreement of earlier API judges with the reported judge (qwen2.5:7b-instruct) on the answers both judged (a side analysis; the reported numbers use only the reported judge):

| other judge | n | faithfulness r (Spearman) | mean abs diff | fully-supported agreement | kappa | relevance exact | within 1 | kappa | weighted kappa |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen3.8-27b | 23 | 0.71 (0.81) | 0.06 | 0.96 | 0.83 | 0.83 | 0.91 | 0.54 | 0.78 |
| openai/gpt-oss-120b | 7 | - (-) | 0.25 | 0.67 | 0.00 | 0.71 | 0.71 | 0.00 | 0.00 |

## Limitations

- The test set is model-generated and AI-reviewed (by Claude); no pair is human-verified yet.
- The judge is a small local model (qwen2.5:7b-instruct), less reliable than a frontier judge. It was chosen because the free API tier's daily token limit allows only about 30 judge calls a day for these ~6K-token prompts; its agreement with the larger API judges on the answers both judged is reported.
- The openai_compatible backend was evaluated with the best configuration only, on a fixed 60-question subset (seeded, stratified by answerability), because the free tier's daily token quota allows only about 30 answers a day; its confidence intervals are wider than the full-test-set rows.

## AI-reviewed subset (43 pairs)

The same runs restricted to the 43 pairs an AI assistant (Claude) accepted or edited after checking them against the source documents (see `docs/qa_review.md`). This is an AI review, not human verification; edited pairs use the edited wording.

### Retrieval

| config | chunks | doc R@5 | chunk R@1 | chunk R@5 | chunk R@10 | chunk MRR@10 | chunk nDCG@10 [95% CI] | p50 ms |
|---|---|---|---|---|---|---|---|---|
| semantic-1024-bm25 | 12497 | 0.964 | 0.696 | 0.911 | 0.982 | 0.817 | 0.848 [0.763, 0.926] | 77.1 |
| fixed-256-bm25 | 19299 | 0.964 | 0.696 | 0.893 | 0.964 | 0.806 | 0.836 [0.734, 0.919] | 76.5 |
| fixed-512-bm25 | 10131 | 0.964 | 0.625 | 0.946 | 1.000 | 0.768 | 0.823 [0.729, 0.909] | 43.6 |
| semantic-256-bm25 | 22796 | 1.000 | 0.661 | 0.911 | 0.964 | 0.786 | 0.819 [0.720, 0.906] | 151.1 |
| semantic-512-bm25 | 15152 | 0.964 | 0.625 | 0.911 | 0.982 | 0.773 | 0.815 [0.721, 0.900] | 97.3 |
| fixed-1024-bm25 | 5618 | 0.929 | 0.625 | 0.893 | 0.964 | 0.749 | 0.799 [0.687, 0.901] | 24.7 |
| section-1024-bm25 | 19369 | 1.000 | 0.625 | 0.911 | 0.911 | 0.781 | 0.796 [0.689, 0.889] | 131.4 |
| section-512-bm25 | 29655 | 0.964 | 0.607 | 0.821 | 0.839 | 0.747 | 0.752 [0.605, 0.871] | 229.2 |
| semantic-1024-hybrid | 12497 | 0.929 | 0.554 | 0.875 | 0.911 | 0.706 | 0.748 [0.626, 0.855] | 137.3 |
| fixed-512-hybrid | 10131 | 0.893 | 0.554 | 0.875 | 0.875 | 0.704 | 0.741 [0.611, 0.865] | 128.4 |
| section-1024-hybrid | 19369 | 0.929 | 0.500 | 0.839 | 0.893 | 0.688 | 0.719 [0.611, 0.830] | 198.7 |
| fixed-1024-hybrid | 5618 | 0.893 | 0.500 | 0.839 | 0.893 | 0.651 | 0.715 [0.583, 0.835] | 95.8 |
| semantic-512-hybrid | 15152 | 0.964 | 0.411 | 0.911 | 0.946 | 0.641 | 0.708 [0.610, 0.807] | 150.7 |
| semantic-256-hybrid | 22796 | 0.929 | 0.482 | 0.768 | 0.911 | 0.640 | 0.695 [0.562, 0.817] | 209.6 |
| fixed-256-hybrid | 19299 | 0.929 | 0.482 | 0.839 | 0.857 | 0.653 | 0.694 [0.573, 0.818] | 169.2 |
| section-1024-hybrid_rerank | 19369 | 0.857 | 0.571 | 0.750 | 0.821 | 0.639 | 0.679 [0.525, 0.815] | 2356.6 |
| section-256-bm25 | 43864 | 0.893 | 0.500 | 0.750 | 0.804 | 0.659 | 0.677 [0.530, 0.813] | 329.4 |
| section-512-hybrid | 29655 | 0.893 | 0.429 | 0.714 | 0.893 | 0.621 | 0.672 [0.560, 0.794] | 282.5 |
| section-256-hybrid | 43864 | 0.786 | 0.500 | 0.661 | 0.786 | 0.641 | 0.661 [0.515, 0.807] | 338.7 |
| fixed-1024-hybrid_rerank | 5618 | 0.786 | 0.464 | 0.679 | 0.839 | 0.579 | 0.640 [0.495, 0.772] | 1323.3 |
| fixed-512-hybrid_rerank | 10131 | 0.857 | 0.464 | 0.714 | 0.786 | 0.578 | 0.629 [0.469, 0.772] | 1344.8 |
| semantic-512-hybrid_rerank | 15152 | 0.857 | 0.446 | 0.625 | 0.839 | 0.571 | 0.624 [0.479, 0.753] | 2209.8 |
| semantic-512-dense | 15152 | 0.821 | 0.393 | 0.732 | 0.804 | 0.546 | 0.605 [0.463, 0.740] | 74.8 |
| fixed-256-hybrid_rerank | 19299 | 0.857 | 0.429 | 0.661 | 0.750 | 0.556 | 0.603 [0.451, 0.748] | 1018.3 |
| semantic-1024-dense | 12497 | 0.821 | 0.393 | 0.696 | 0.768 | 0.547 | 0.598 [0.450, 0.746] | 74.0 |
| fixed-1024-dense | 5618 | 0.821 | 0.393 | 0.714 | 0.768 | 0.537 | 0.594 [0.452, 0.740] | 50.2 |
| semantic-1024-hybrid_rerank | 12497 | 0.786 | 0.429 | 0.607 | 0.768 | 0.529 | 0.585 [0.433, 0.727] | 2219.8 |
| section-1024-dense | 19369 | 0.750 | 0.375 | 0.679 | 0.786 | 0.527 | 0.579 [0.438, 0.715] | 78.2 |
| semantic-256-hybrid_rerank | 22796 | 0.857 | 0.429 | 0.589 | 0.714 | 0.529 | 0.568 [0.409, 0.709] | 1563.9 |
| section-256-dense | 43864 | 0.893 | 0.357 | 0.714 | 0.714 | 0.532 | 0.561 [0.424, 0.707] | 102.5 |
| section-512-hybrid_rerank | 29655 | 0.857 | 0.429 | 0.643 | 0.696 | 0.542 | 0.557 [0.412, 0.706] | 2241.3 |
| fixed-512-dense | 10131 | 0.750 | 0.339 | 0.643 | 0.732 | 0.505 | 0.554 [0.415, 0.698] | 54.0 |
| semantic-256-dense | 22796 | 0.821 | 0.339 | 0.643 | 0.714 | 0.494 | 0.540 [0.396, 0.688] | 83.9 |
| section-512-dense | 29655 | 0.786 | 0.250 | 0.696 | 0.750 | 0.476 | 0.521 [0.404, 0.648] | 90.7 |
| section-256-hybrid_rerank | 43864 | 0.821 | 0.429 | 0.500 | 0.625 | 0.517 | 0.521 [0.370, 0.684] | 1622.3 |
| fixed-256-dense | 19299 | 0.786 | 0.250 | 0.589 | 0.643 | 0.442 | 0.477 [0.345, 0.623] | 71.3 |

### Generation

| config | backend (model) | questions | n | judge | faithfulness (judge) | faithfulness (NLI) | relevance (judge) | citation acc. (NLI) | citations valid | refusal P | refusal R | p50 ms | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-1024-bm25 | ollama (qwen2.5:3b-instruct 100%) | full test set (n=28) | 28 | qwen2.5:7b-instruct | 1.000 [1.000, 1.000] | 0.285 [0.128, 0.453] | 0.964 [0.893, 1.000] | 0.278 [0.111, 0.500] | 0.536 | - | - | 208400 | 0.0000 |
| fixed-1024-bm25 | openai_compatible (openai/gpt-oss-20b 100%) | fixed subset (n=9) | 9 | qwen2.5:7b-instruct | 1.000 [1.000, 1.000] | 0.250 [0.000, 0.625] | 1.000 [1.000, 1.000] | 0.400 [0.000, 0.800] | 0.375 | 0.000 [0.000, 0.000] | - | 57402 | 0.0000 |
| fixed-1024-hybrid | ollama (qwen2.5:3b-instruct 100%) | fixed subset (n=9) | 9 | qwen2.5:7b-instruct | 0.889 [0.667, 1.000] | 0.444 [0.111, 0.778] | 0.889 [0.667, 1.000] | 0.429 [0.143, 0.857] | 0.778 | - | - | 193129 | 0.0000 |
| fixed-1024-hybrid_rerank | ollama (qwen2.5:3b-instruct 100%) | fixed subset (n=9) | 9 | qwen2.5:7b-instruct | 0.889 [0.722, 1.000] | 0.611 [0.333, 0.889] | 0.833 [0.583, 1.000] | 0.400 [0.000, 0.800] | 0.444 | - | - | 160743 | 0.0000 |

## Confidence gate

Calibrated tau = 0.49 on 61 dev questions: refusal precision 1.000, recall 0.600, answerable coverage 1.000.
