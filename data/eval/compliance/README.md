# Compliance evaluation set

89 clauses from 7 synthetic policy documents of a fictional scheduled commercial bank
("Sample Bank Ltd."), each labelled with the verdict a compliance reviewer should reach
against RBI notifications in `corpus-v2` (snapshot of notifications up to 2026-11-11),
the notifications that establish it, and verbatim evidence.

**Status: AI-authored (Claude), 0 human-verified.** Every citation and evidence quote is
machine-checked against the corpus (`results/compliance/gold_check.json`: 0 missing
documents, 0 missing quotes). The verdicts themselves are judgements that a human has
not yet reviewed; see [Reviewing](#reviewing).

## Contents

| Split | Policies | Clauses | compliant | conflict | gap | not applicable | insufficient evidence |
|---|---|---|---|---|---|---|---|
| dev | KYC / AML, retail products (cards and deposits) | 26 | 8 | 10 | 4 | 2 | 2 |
| test | fraud risk management, outsourcing, IT / cyber security, customer service, digital lending | 63 | 20 | 25 | 8 | 5 | 5 |
| **all** | 7 | **89** | 28 | 35 | 12 | 7 | 7 |

Splits are by policy document. Dev is for prompt and threshold development; every
reported number is on test.

Deliberately tricky items (tags; an item can carry several):

| Tag | Items | Example |
|---|---|---|
| `outdated_rule` | 6 | Beneficial-owner threshold "more than 25 per cent" (2016 Master Direction) vs "more than 10 percent" now |
| `entity_confusion` | 10 | OTP e-KYC term-loan cap Rs. 50,000 (the UCB value) vs Rs. 60,000 for commercial banks |
| `partial_compliance` | 12 | Cyber incidents reported to RBI within six hours, but no CERT-In notification |
| `stricter_than_required` | 6 | Transaction records kept ten years (minimum five): still compliant |
| `amendment` / `future_effective` | 6 / 2 | Recovery-call retention: six months only from 1 January 2027, so not yet required |
| `out_of_scope` | 7 | Reserved parking for the Fraud Monitoring Cell |
| `not_in_corpus` | 7 | Quantum-safe cryptography migration deadline; absence verified by corpus search |
| `numeric_threshold` | 36 | Days, years, amounts, rates |
| `prohibition` | 11 | Outsourcing internal audit; debit cards on cash credit accounts |

## Files

| File | What it is |
|---|---|
| `policies.json` | The 7 policies: title, topic, split, format (Markdown or plain), sections |
| `labels.jsonl` | One labelled clause per line (schema: `rbirag.compliance.evaluation.dataset.Label`) |
| `policies/*.md` | Policy documents rendered from the two files above (`rbirag comply eval render`) |
| `LABELING.md` | Verdict definitions, citation rules, tags |

`labels.jsonl` is the source of truth; the rendered policies are checked to be current
by the test suite.

## Commands

```bash
rbirag comply eval stats                                   # counts
rbirag comply eval check --documents data/processed/documents.jsonl   # quotes vs corpus
rbirag comply eval segmentation                            # clause splitting P/R/F1
rbirag comply eval render --pdf                            # policy documents (+ PDFs)
rbirag comply eval score predictions.jsonl --split test    # metrics for a system's output
```

## Reviewing

```bash
rbirag comply eval review --reviewer <your-name> --limit 20
```

For each clause you see the label, rationale and evidence quotes. Press `a` to accept,
`e` to change the verdict, `r` to reject or `s` to skip. Each decision is appended to the
label's `reviews` and saved at once. Rejected items are excluded from evaluation.
Accepted or edited items count as human-verified, and reports state how many there are.

## Limitations

- **AI-authored labels.** The verdicts, not just the wording, come from one model, so its
  blind spots are not caught by checking quotes. Until reviewed, the set measures
  agreement with Claude's reading of the notifications.
- **Imbalanced verdicts.** There are more conflicts (35) than gaps (12). Macro-F1 is
  reported next to accuracy so the minority classes count, but small classes have wide
  intervals.
- **Small test set.** 63 test clauses give a 95% interval of roughly ±12 points on
  accuracy.
- **Synthetic, tidy documents.** The policies are clean and consistently numbered.
  Clause splitting scores 1.00 on them (text and PDF), which says little about messy
  real policies. The renderer and the splitter were also written by the same author.
- **One entity type and one snapshot.** Labels hold for a scheduled commercial bank as of
  2026-11-11. Some rules change after the snapshot, and labels follow the rule in force
  then.
