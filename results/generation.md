**Judge: qwen2.5:7b-instruct, run locally with Ollama on CPU** (290 s per verdict on a CI runner); agreement with earlier judges: qwen/qwen3.8-27b n=23: faithfulness r=0.71, relevance weighted kappa=0.78; openai/gpt-oss-120b n=7: faithfulness r=-, relevance weighted kappa=0.00. NLI faithfulness and citation metrics are judge-independent.

| config | backend (model) | questions | n | judge | faithfulness (judge) | faithfulness (NLI) | relevance (judge) | citation acc. (NLI) | citations valid | refusal P | refusal R | p50 ms | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-1024-bm25 | ollama (qwen2.5:3b-instruct 100%) | full test set (n=155) | 155 | qwen2.5:7b-instruct | 0.953 [0.918, 0.984] | 0.212 [0.147, 0.280] | 0.941 [0.902, 0.975] | 0.170 [0.095, 0.264] | 0.438 | 0.913 [0.782, 1.000] | 0.840 [0.678, 0.966] | 245980 | 0.0000 |
| fixed-1024-hybrid | ollama (qwen2.5:3b-instruct 100%) | fixed subset (n=60) | 60 | qwen2.5:7b-instruct | 0.891 [0.783, 0.978] | 0.289 [0.167, 0.416] | 0.851 [0.745, 0.937] | 0.304 [0.130, 0.478] | 0.404 | 0.750 [0.499, 1.000] | 0.900 [0.667, 1.000] | 237035 | 0.0000 |
| fixed-1024-hybrid_rerank | ollama (qwen2.5:3b-instruct 100%) | fixed subset (n=60) | 60 | qwen2.5:7b-instruct | 0.918 [0.848, 0.978] | 0.290 [0.170, 0.402] | 0.897 [0.810, 0.968] | 0.185 [0.062, 0.333] | 0.478 | 0.714 [0.462, 0.929] | 1.000 [1.000, 1.000] | 237289 | 0.0000 |

### Incomplete runs

Not reported as results: too many generation or judge calls failed or are still pending (quota). Re-run the workflow to complete them.

| config | backend (model) | status |
|---|---|---|
| fixed-1024-bm25 | openai_compatible (openai/gpt-oss-20b 85%, qwen/qwen3.8-27b 11%, openai/gpt-oss-120b 4%) | incomplete (27/60) |
