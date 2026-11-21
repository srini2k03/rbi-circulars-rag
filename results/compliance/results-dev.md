## Results on the dev split

Runs: agent = `gemini-3.5-flash-lite` via gemini (26 clauses, 0 errors, 2026-11-21, commit `ecec82d`); agent = `openai/gpt-oss-20b` via groq (2 clauses, 0 errors, 2026-11-21, commit `9e1d074`); baseline = `gemini-3.5-flash-lite` via gemini (26 clauses, 0 errors, 2026-11-21, commit `9edeb74`); baseline = `openai/gpt-oss-20b` via groq (26 clauses, 0 errors, 2026-11-21, commit `8002198`)

Models that actually answered (per clause; the provider's fallback chain is used when the configured model is rate-limited): agent via groq: `qwen/qwen3.8-27b` x2; baseline via groq: `qwen/qwen3.8-27b` x25, `openai/gpt-oss-20b` x1

### Headline metrics (95% bootstrap CIs)

| metric | agent (gemini-3.5-flash-lite) (dev, n=26) | agent (openai/gpt-oss-20b) (dev, n=26) | baseline (gemini-3.5-flash-lite) (dev, n=26) | baseline (openai/gpt-oss-20b) (dev, n=26) |
|---|---|---|---|---|
| verdict accuracy | 0.885 [0.769, 1.000] | 0.077 [0.000, 0.192] | 0.885 [0.731, 1.000] | 0.846 [0.692, 0.962] |
| macro-F1 | 0.916 [0.747, 1.000] | 0.112 [0.015, 0.228] | 0.905 [0.752, 1.000] | 0.800 [0.567, 0.968] |
| citation precision | 0.955 [0.857, 1.000] | 1.000 [1.000, 1.000] | 0.815 [0.687, 0.952] | 0.719 [0.561, 0.897] |
| citation recall | 0.909 [0.773, 1.000] | 0.091 [0.000, 0.227] | 0.864 [0.682, 1.000] | 0.909 [0.773, 1.000] |
| tool-argument validity | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| fetch grounding | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | - | - |
| tool-selection accuracy | 1.000 [1.000, 1.000] | 0.154 [0.038, 0.308] | 0.269 [0.115, 0.462] | 0.269 [0.115, 0.462] |
| task success | 0.808 [0.654, 0.962] | 0.077 [0.000, 0.192] | 0.769 [0.615, 0.923] | 0.808 [0.654, 0.923] |
| steps / clause | 3.58 [3.08, 4.04] | 0.35 [0.00, 0.88] | 2.00 [2.00, 2.00] | 2.00 [2.00, 2.00] |
| LLM calls / clause | 5.62 [4.96, 6.19] | 0.46 [0.00, 1.19] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| tokens / document | 220588 [192334, 248843] | 21141 [0, 42282] | 46003 [42458, 49548] | 46984 [43299, 50669] |
| cost $ / document | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| latency s / document | 539 [472, 606] | 332 [0, 663] | 196 [174, 219] | 1500 [1439, 1560] |
| ECE | 0.092 [0.013, 0.212] | - | 0.135 [0.040, 0.285] | 0.122 [0.029, 0.267] |

### Agent vs fixed pipeline via gemini (paired, same clauses)

| metric (n=26) | agent | baseline | agent - baseline [95% CI] |
|---|---|---|---|
| verdict accuracy | 0.885 | 0.885 | +0.000 [-0.115, +0.115] |
| task success | 0.808 | 0.769 | +0.038 [-0.154, +0.231] |
| citation recall | 0.909 | 0.864 | +0.045 [-0.182, +0.227] |

### Agent vs fixed pipeline via groq (paired, same clauses)

| metric (n=26) | agent | baseline | agent - baseline [95% CI] |
|---|---|---|---|
| verdict accuracy | 0.077 | 0.846 | -0.769 [-0.923, -0.538] |
| task success | 0.077 | 0.808 | -0.731 [-0.923, -0.500] |
| citation recall | 0.091 | 0.909 | -0.818 [-0.955, -0.636] |

### F1 per verdict

| verdict | support | agent (gemini-3.5-flash-lite) F1 | agent (openai/gpt-oss-20b) F1 | baseline (gemini-3.5-flash-lite) F1 | baseline (openai/gpt-oss-20b) F1 |
|---|---|---|---|---|---|
| compliant | 8 | 0.82 | 0.22 | 0.88 | 0.86 |
| conflict | 10 | 0.90 | 0.18 | 0.90 | 0.95 |
| gap | 4 | 0.86 | 0.00 | 0.75 | 0.73 |
| not_applicable | 2 | 1.00 | 0.00 | 1.00 | 0.80 |
| insufficient_evidence | 2 | 1.00 | 0.15 | 1.00 | 0.67 |

### agent (gemini-3.5-flash-lite): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 7 | 1 | 0 | 0 | 0 |
| conflict | 1 | 9 | 0 | 0 | 0 |
| gap | 1 | 0 | 3 | 0 | 0 |
| n/a | 0 | 0 | 0 | 2 | 0 |
| insufficient | 0 | 0 | 0 | 0 | 2 |

### agent (gemini-3.5-flash-lite): failures by type

| failure type | clauses | share |
|---|---|---|
| missed_issue | 2 | 8% |
| citation_miss | 2 | 8% |
| false_alarm | 1 | 4% |

### agent (gemini-3.5-flash-lite): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 2 | 0.50 | 0.00 |
| entity_confusion | 1 | 0.00 | 0.00 |
| future_effective | 1 | 1.00 | 1.00 |
| not_in_corpus | 2 | 1.00 | 1.00 |
| numeric_threshold | 12 | 0.92 | 0.92 |
| out_of_scope | 2 | 1.00 | 1.00 |
| outdated_rule | 2 | 1.00 | 1.00 |
| paraphrase | 1 | 1.00 | 1.00 |
| partial_compliance | 4 | 0.75 | 0.50 |
| prohibition | 1 | 1.00 | 1.00 |
| stricter_than_required | 2 | 0.50 | 0.50 |

### agent (openai/gpt-oss-20b): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 1 | 0 | 0 | 0 | 7 |
| conflict | 0 | 1 | 0 | 0 | 9 |
| gap | 0 | 0 | 0 | 0 | 4 |
| n/a | 0 | 0 | 0 | 0 | 2 |
| insufficient | 0 | 0 | 0 | 0 | 2 |

### agent (openai/gpt-oss-20b): failures by type

| failure type | clauses | share |
|---|---|---|
| run_error | 24 | 92% |

### agent (openai/gpt-oss-20b): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 2 | 0.00 | 0.00 |
| entity_confusion | 1 | 0.00 | 0.00 |
| future_effective | 1 | 0.00 | 0.00 |
| not_in_corpus | 2 | 0.00 | 0.00 |
| numeric_threshold | 12 | 0.08 | 0.08 |
| out_of_scope | 2 | 0.00 | 0.00 |
| outdated_rule | 2 | 0.00 | 0.00 |
| paraphrase | 1 | 1.00 | 1.00 |
| partial_compliance | 4 | 0.00 | 0.00 |
| prohibition | 1 | 0.00 | 0.00 |
| stricter_than_required | 2 | 0.00 | 0.00 |

### baseline (gemini-3.5-flash-lite): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 7 | 0 | 1 | 0 | 0 |
| conflict | 1 | 9 | 0 | 0 | 0 |
| gap | 0 | 1 | 3 | 0 | 0 |
| n/a | 0 | 0 | 0 | 2 | 0 |
| insufficient | 0 | 0 | 0 | 0 | 2 |

### baseline (gemini-3.5-flash-lite): failures by type

| failure type | clauses | share |
|---|---|---|
| citation_miss | 3 | 12% |
| conflict_gap_confusion | 1 | 4% |
| missed_issue | 1 | 4% |
| false_alarm | 1 | 4% |

### baseline (gemini-3.5-flash-lite): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 2 | 0.50 | 0.50 |
| entity_confusion | 1 | 1.00 | 1.00 |
| future_effective | 1 | 1.00 | 0.00 |
| not_in_corpus | 2 | 1.00 | 1.00 |
| numeric_threshold | 12 | 0.92 | 0.75 |
| out_of_scope | 2 | 1.00 | 1.00 |
| outdated_rule | 2 | 1.00 | 0.50 |
| paraphrase | 1 | 1.00 | 1.00 |
| partial_compliance | 4 | 0.75 | 0.75 |
| prohibition | 1 | 1.00 | 1.00 |
| stricter_than_required | 2 | 0.50 | 0.50 |

### baseline (openai/gpt-oss-20b): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 6 | 0 | 2 | 0 | 0 |
| conflict | 0 | 9 | 1 | 0 | 0 |
| gap | 0 | 0 | 4 | 0 | 0 |
| n/a | 0 | 0 | 0 | 2 | 0 |
| insufficient | 0 | 0 | 0 | 1 | 1 |

### baseline (openai/gpt-oss-20b): failures by type

| failure type | clauses | share |
|---|---|---|
| false_alarm | 2 | 8% |
| citation_miss | 1 | 4% |
| conflict_gap_confusion | 1 | 4% |
| unsupported_verdict | 1 | 4% |

### baseline (openai/gpt-oss-20b): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 2 | 1.00 | 1.00 |
| entity_confusion | 1 | 1.00 | 1.00 |
| future_effective | 1 | 0.00 | 0.00 |
| not_in_corpus | 2 | 0.50 | 0.50 |
| numeric_threshold | 12 | 0.83 | 0.83 |
| out_of_scope | 2 | 1.00 | 1.00 |
| outdated_rule | 2 | 1.00 | 1.00 |
| paraphrase | 1 | 0.00 | 0.00 |
| partial_compliance | 4 | 1.00 | 1.00 |
| prohibition | 1 | 1.00 | 1.00 |
| stricter_than_required | 2 | 1.00 | 1.00 |
