## Results on the test split

Runs: agent = `gemini-3.5-flash-lite` via gemini (8 clauses, 0 errors, 2026-11-21, commit `e66924e`); baseline = `gemini-3.5-flash-lite` via gemini (1 clauses, 0 errors, 2026-11-21, commit `e66924e`)

### Headline metrics (95% bootstrap CIs)

| metric | agent (gemini-3.5-flash-lite) (test, n=63) | baseline (gemini-3.5-flash-lite) (test, n=63) |
|---|---|---|
| verdict accuracy | 0.111 [0.032, 0.190] | 0.016 [0.000, 0.048] |
| macro-F1 | 0.136 [0.068, 0.209] | 0.049 [0.012, 0.096] |
| citation precision | 0.750 [0.363, 1.000] | 1.000 [1.000, 1.000] |
| citation recall | 0.113 [0.038, 0.208] | 0.019 [0.000, 0.057] |
| tool-argument validity | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| fetch grounding | 1.000 [1.000, 1.000] | - |
| tool-selection accuracy | 0.206 [0.111, 0.317] | 0.095 [0.032, 0.175] |
| task success | 0.095 [0.032, 0.175] | 0.016 [0.000, 0.048] |
| steps / clause | 0.44 [0.16, 0.75] | 0.03 [0.00, 0.10] |
| LLM calls / clause | 0.70 [0.25, 1.17] | 0.02 [0.00, 0.05] |
| tokens / document | 26745 [0, 80235] | 768 [0, 2305] |
| cost $ / document | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| latency s / document | 71 [0, 214] | 0 [0, 1] |
| ECE | - | - |

### Agent vs fixed pipeline via gemini (paired, same clauses)

| metric (n=63) | agent | baseline | agent - baseline [95% CI] |
|---|---|---|---|
| verdict accuracy | 0.111 | 0.016 | +0.095 [+0.032, +0.175] |
| task success | 0.095 | 0.016 | +0.079 [+0.016, +0.143] |
| citation recall | 0.113 | 0.019 | +0.094 [+0.019, +0.170] |

### F1 per verdict

| verdict | support | agent (gemini-3.5-flash-lite) F1 | baseline (gemini-3.5-flash-lite) F1 |
|---|---|---|---|
| compliant | 20 | 0.18 | 0.10 |
| conflict | 25 | 0.33 | 0.00 |
| gap | 8 | 0.00 | 0.00 |
| not_applicable | 5 | 0.00 | 0.00 |
| insufficient_evidence | 5 | 0.17 | 0.15 |

### agent (gemini-3.5-flash-lite): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 2 | 0 | 1 | 0 | 17 |
| conflict | 0 | 5 | 0 | 0 | 20 |
| gap | 0 | 0 | 0 | 0 | 8 |
| n/a | 0 | 0 | 0 | 0 | 5 |
| insufficient | 0 | 0 | 0 | 0 | 5 |

### agent (gemini-3.5-flash-lite): failures by type

| failure type | clauses | share |
|---|---|---|
| run_error | 55 | 87% |
| false_alarm | 1 | 2% |
| citation_miss | 1 | 2% |

### agent (gemini-3.5-flash-lite): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 4 | 0.00 | 0.00 |
| entity_confusion | 9 | 0.11 | 0.11 |
| future_effective | 1 | 0.00 | 0.00 |
| multi_source | 2 | 0.00 | 0.00 |
| not_in_corpus | 5 | 0.00 | 0.00 |
| numeric_threshold | 24 | 0.21 | 0.17 |
| out_of_scope | 5 | 0.00 | 0.00 |
| outdated_rule | 4 | 0.00 | 0.00 |
| paraphrase | 4 | 0.00 | 0.00 |
| partial_compliance | 8 | 0.00 | 0.00 |
| prohibition | 10 | 0.00 | 0.00 |
| stricter_than_required | 4 | 0.00 | 0.00 |

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
