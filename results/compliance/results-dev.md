## Results on the dev split

Runs: agent = `gemini-3.5-flash-lite` via gemini (26 clauses, 0 errors, 2026-11-21, commit `6d6c54e`); baseline = `gemini-3.5-flash-lite` via gemini (26 clauses, 0 errors, 2026-11-21, commit `6d6c54e`)

### Headline metrics (95% bootstrap CIs)

| metric | agent (gemini-3.5-flash-lite) (dev, n=26) | baseline (gemini-3.5-flash-lite) (dev, n=26) |
|---|---|---|
| verdict accuracy | 0.923 [0.808, 1.000] | 0.885 [0.731, 1.000] |
| macro-F1 | 0.944 [0.840, 1.000] | 0.918 [0.760, 1.000] |
| citation precision | 0.909 [0.773, 1.000] | 0.875 [0.760, 1.000] |
| citation recall | 0.864 [0.682, 1.000] | 0.909 [0.773, 1.000] |
| tool-argument validity | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| fetch grounding | 1.000 [1.000, 1.000] | - |
| tool-selection accuracy | 1.000 [1.000, 1.000] | 0.269 [0.115, 0.462] |
| task success | 0.808 [0.654, 0.962] | 0.808 [0.654, 0.962] |
| steps / clause | 3.69 [3.19, 4.19] | 2.00 [2.00, 2.00] |
| LLM calls / clause | 5.58 [5.00, 6.12] | 1.00 [1.00, 1.00] |
| tokens / document | 218881 [197335, 240427] | 46278 [42826, 49731] |
| cost $ / document | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| latency s / document | 510 [465, 556] | 149 [143, 156] |
| ECE | 0.067 [0.006, 0.183] | 0.150 [0.042, 0.304] |

### Agent vs fixed pipeline via gemini (paired, same clauses)

| metric (n=26) | agent | baseline | agent - baseline [95% CI] |
|---|---|---|---|
| verdict accuracy | 0.923 | 0.885 | +0.038 [-0.077, +0.192] |
| task success | 0.808 | 0.808 | +0.000 [-0.192, +0.231] |
| citation recall | 0.864 | 0.909 | -0.045 [-0.273, +0.136] |

### F1 per verdict

| verdict | support | agent (gemini-3.5-flash-lite) F1 | baseline (gemini-3.5-flash-lite) F1 |
|---|---|---|---|
| compliant | 8 | 0.93 | 0.88 |
| conflict | 10 | 0.90 | 0.86 |
| gap | 4 | 0.89 | 0.86 |
| not_applicable | 2 | 1.00 | 1.00 |
| insufficient_evidence | 2 | 1.00 | 1.00 |

### agent (gemini-3.5-flash-lite): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 7 | 1 | 0 | 0 | 0 |
| conflict | 0 | 9 | 1 | 0 | 0 |
| gap | 0 | 0 | 4 | 0 | 0 |
| n/a | 0 | 0 | 0 | 2 | 0 |
| insufficient | 0 | 0 | 0 | 0 | 2 |

### agent (gemini-3.5-flash-lite): failures by type

| failure type | clauses | share |
|---|---|---|
| citation_miss | 3 | 12% |
| conflict_gap_confusion | 1 | 4% |
| false_alarm | 1 | 4% |

### agent (gemini-3.5-flash-lite): by tag

| tag | n | accuracy | task success |
|---|---|---|---|
| amendment | 2 | 0.50 | 0.00 |
| entity_confusion | 1 | 1.00 | 1.00 |
| future_effective | 1 | 1.00 | 1.00 |
| not_in_corpus | 2 | 1.00 | 1.00 |
| numeric_threshold | 12 | 0.92 | 0.75 |
| out_of_scope | 2 | 1.00 | 1.00 |
| outdated_rule | 2 | 1.00 | 0.50 |
| paraphrase | 1 | 1.00 | 1.00 |
| partial_compliance | 4 | 1.00 | 0.75 |
| prohibition | 1 | 1.00 | 1.00 |
| stricter_than_required | 2 | 0.50 | 0.50 |

### baseline (gemini-3.5-flash-lite): confusion matrix

| gold \ predicted | compliant | conflict | gap | n/a | insufficient |
|---|---|---|---|---|---|
| compliant | 7 | 1 | 0 | 0 | 0 |
| conflict | 1 | 9 | 0 | 0 | 0 |
| gap | 0 | 1 | 3 | 0 | 0 |
| n/a | 0 | 0 | 0 | 2 | 0 |
| insufficient | 0 | 0 | 0 | 0 | 2 |

### baseline (gemini-3.5-flash-lite): failures by type

| failure type | clauses | share |
|---|---|---|
| citation_miss | 2 | 8% |
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
| numeric_threshold | 12 | 0.92 | 0.83 |
| out_of_scope | 2 | 1.00 | 1.00 |
| outdated_rule | 2 | 1.00 | 1.00 |
| paraphrase | 1 | 1.00 | 1.00 |
| partial_compliance | 4 | 0.75 | 0.75 |
| prohibition | 1 | 1.00 | 1.00 |
| stricter_than_required | 2 | 0.50 | 0.50 |
