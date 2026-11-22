## Results on the test split

Runs: agent = `gemini-3.5-flash-lite` via gemini (39 clauses, 0 errors, 2026-11-22, commit `e40ecc8`); baseline = `gemini-3.5-flash-lite` via gemini (1 clauses, 0 errors, 2026-11-22, commit `e40ecc8`)

### Headline metrics (95% bootstrap CIs)

| metric | agent (gemini-3.5-flash-lite) (test, n=63) | baseline (gemini-3.5-flash-lite) (test, n=63) |
|---|---|---|
| verdict accuracy | 0.540 [0.429, 0.667] | 0.016 [0.000, 0.048] |
| macro-F1 | 0.531 [0.356, 0.665] | 0.049 [0.012, 0.096] |
| citation precision | 0.818 [0.667, 0.938] | 1.000 [1.000, 1.000] |
| citation recall | 0.491 [0.358, 0.642] | 0.019 [0.000, 0.057] |
| tool-argument validity | 0.994 [0.978, 1.000] | 1.000 [1.000, 1.000] |
| fetch grounding | 1.000 [1.000, 1.000] | - |
| tool-selection accuracy | 0.651 [0.524, 0.762] | 0.095 [0.032, 0.175] |
| task success | 0.429 [0.302, 0.540] | 0.016 [0.000, 0.048] |
| steps / clause | 2.46 [1.90, 3.03] | 0.03 [0.00, 0.10] |
| LLM calls / clause | 3.67 [2.87, 4.43] | 0.02 [0.00, 0.05] |
| tokens / document | 138710 [48747, 227810] | 768 [0, 2305] |
| cost $ / document | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| latency s / document | 361 [116, 599] | 0 [0, 1] |
| ECE | - | - |

### Agent vs fixed pipeline via gemini (paired, same clauses)

| metric (n=63) | agent | baseline | agent - baseline [95% CI] |
|---|---|---|---|
| verdict accuracy | 0.540 | 0.016 | +0.524 [+0.397, +0.651] |
| task success | 0.429 | 0.016 | +0.413 [+0.302, +0.540] |
| citation recall | 0.491 | 0.019 | +0.472 [+0.340, +0.604] |

### F1 per verdict

| verdict | support | agent (gemini-3.5-flash-lite) F1 | baseline (gemini-3.5-flash-lite) F1 |
|---|---|---|---|
| compliant | 20 | 0.61 | 0.10 |
| conflict | 25 | 0.81 | 0.00 |
| gap | 8 | 0.36 | 0.00 |
| not_applicable | 5 | 0.57 | 0.00 |
| insufficient_evidence | 5 | 0.30 | 0.15 |

### agent (gemini-3.5-flash-lite): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 10 | 0 | 1 | 0 | 9 |
| conflict | 0 | 17 | 0 | 0 | 8 |
| gap | 3 | 0 | 2 | 0 | 3 |
| n/a | 0 | 0 | 0 | 2 | 3 |
| insufficient | 0 | 0 | 0 | 0 | 5 |

### agent (gemini-3.5-flash-lite): failures by type

| failure type | clauses | share |
|---|---|---|
| run_error | 24 | 38% |
| citation_miss | 6 | 10% |
| missed_issue | 3 | 5% |
| false_alarm | 1 | 2% |
| over_abstention | 1 | 2% |
| invalid_tool_call | 1 | 2% |

### agent (gemini-3.5-flash-lite): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 4 | 0.25 | 0.25 |
| entity_confusion | 9 | 0.67 | 0.44 |
| future_effective | 1 | 0.00 | 0.00 |
| multi_source | 2 | 0.50 | 0.00 |
| not_in_corpus | 5 | 0.60 | 0.60 |
| numeric_threshold | 24 | 0.62 | 0.46 |
| out_of_scope | 5 | 0.40 | 0.40 |
| outdated_rule | 4 | 0.50 | 0.00 |
| paraphrase | 4 | 0.50 | 0.50 |
| partial_compliance | 8 | 0.25 | 0.25 |
| prohibition | 10 | 0.50 | 0.40 |
| stricter_than_required | 4 | 0.25 | 0.25 |

### baseline (gemini-3.5-flash-lite): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 1 | 0 | 0 | 0 | 19 |
| conflict | 0 | 0 | 0 | 0 | 25 |
| gap | 0 | 0 | 0 | 0 | 8 |
| n/a | 0 | 0 | 0 | 0 | 5 |
| insufficient | 0 | 0 | 0 | 0 | 5 |

### baseline (gemini-3.5-flash-lite): failures by type

| failure type | clauses | share |
|---|---|---|
| run_error | 62 | 98% |

### baseline (gemini-3.5-flash-lite): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 4 | 0.00 | 0.00 |
| entity_confusion | 9 | 0.00 | 0.00 |
| future_effective | 1 | 0.00 | 0.00 |
| multi_source | 2 | 0.00 | 0.00 |
| not_in_corpus | 5 | 0.00 | 0.00 |
| numeric_threshold | 24 | 0.04 | 0.04 |
| out_of_scope | 5 | 0.00 | 0.00 |
| outdated_rule | 4 | 0.00 | 0.00 |
| paraphrase | 4 | 0.00 | 0.00 |
| partial_compliance | 8 | 0.00 | 0.00 |
| prohibition | 10 | 0.00 | 0.00 |
| stricter_than_required | 4 | 0.00 | 0.00 |
