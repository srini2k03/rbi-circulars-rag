## Results on the dev split

Runs: agent = `openai/gpt-oss-20b` via groq (22 clauses, 4 errors, 2026-11-21, commit `8002198`); baseline = `openai/gpt-oss-20b` via groq (26 clauses, 0 errors, 2026-11-21, commit `8002198`)

### Headline metrics (95% bootstrap CIs)

| metric | agent (openai/gpt-oss-20b) (dev, n=26) | baseline (openai/gpt-oss-20b) (dev, n=26) |
|---|---|---|
| verdict accuracy | 0.192 [0.038, 0.346] | 0.846 [0.692, 0.962] |
| macro-F1 | 0.395 [0.139, 0.536] | 0.800 [0.567, 0.968] |
| citation precision | 1.000 [1.000, 1.000] | 0.719 [0.561, 0.897] |
| citation recall | 0.136 [0.000, 0.318] | 0.909 [0.773, 1.000] |
| tool-argument validity | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| fetch grounding | 1.000 [1.000, 1.000] | - |
| tool-selection accuracy | 0.308 [0.154, 0.500] | 0.269 [0.115, 0.462] |
| task success | 0.192 [0.038, 0.346] | 0.808 [0.654, 0.923] |
| steps / clause | 3.00 [2.54, 3.54] | 2.00 [2.00, 2.00] |
| LLM calls / clause | 4.04 [3.54, 4.50] | 1.00 [1.00, 1.00] |
| tokens / document | 165816 [155639, 175992] | 46984 [43299, 50669] |
| cost $ / document | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| latency s / document | 2298 [1928, 2669] | 1500 [1439, 1560] |
| ECE | - | 0.122 [0.029, 0.267] |

### Agent vs fixed pipeline (paired, same clauses)

| metric (n=26) | agent | baseline | agent - baseline [95% CI] |
|---|---|---|---|
| verdict accuracy | 0.192 | 0.846 | -0.654 [-0.808, -0.462] |
| task success | 0.192 | 0.808 | -0.615 [-0.808, -0.423] |
| citation recall | 0.136 | 0.909 | -0.773 [-0.909, -0.591] |

### F1 per verdict

| verdict | support | agent (openai/gpt-oss-20b) F1 | baseline (openai/gpt-oss-20b) F1 |
|---|---|---|---|
| compliant | 8 | 0.40 | 0.86 |
| conflict | 10 | 0.00 | 0.95 |
| gap | 4 | 0.40 | 0.73 |
| not_applicable | 2 | 1.00 | 0.80 |
| insufficient_evidence | 2 | 0.17 | 0.67 |

### agent (openai/gpt-oss-20b): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 2 | 0 | 0 | 0 | 6 |
| conflict | 0 | 0 | 0 | 0 | 10 |
| gap | 0 | 0 | 1 | 0 | 3 |
| n/a | 0 | 0 | 0 | 2 | 0 |
| insufficient | 0 | 0 | 0 | 0 | 2 |

### agent (openai/gpt-oss-20b): failures by type

| failure type | clauses | share |
|---|---|---|
| over_abstention | 17 | 65% |
| run_error | 4 | 15% |

### agent (openai/gpt-oss-20b): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 2 | 0.50 | 0.50 |
| entity_confusion | 1 | 0.00 | 0.00 |
| future_effective | 1 | 0.00 | 0.00 |
| not_in_corpus | 2 | 0.00 | 0.00 |
| numeric_threshold | 12 | 0.08 | 0.08 |
| out_of_scope | 2 | 1.00 | 1.00 |
| outdated_rule | 2 | 0.00 | 0.00 |
| paraphrase | 1 | 0.00 | 0.00 |
| partial_compliance | 4 | 0.25 | 0.25 |
| prohibition | 1 | 0.00 | 0.00 |
| stricter_than_required | 2 | 0.00 | 0.00 |

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
