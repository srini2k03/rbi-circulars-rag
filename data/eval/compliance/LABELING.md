# Labelling guide: policy clauses vs RBI regulations

This guide defines the labels in `labels.jsonl`. Every label is judged **as of the
corpus snapshot** (`corpus-v2`, notifications up to 2026-11-11), for a **scheduled
commercial bank** (not an NBFC, co-operative bank, RRB, small finance bank or payments
bank).

## Verdicts

| Verdict | Use when | Example |
|---|---|---|
| `compliant` | The clause meets every requirement the governing notification(s) set on the point it addresses. Stricter-than-required clauses are compliant. | RBI: records kept "for at least five years"; policy: ten years. |
| `conflict` | The clause contradicts a requirement: a weaker number (longer timeline, lower limit, less frequent review), permitting what is prohibited, or forbidding what is mandated. A clause that follows a **replaced** rule is a conflict with the current rule. | RBI: high-risk KYC updation at least every two years; policy: every three years. |
| `gap` | The clause addresses the requirement's topic and says nothing wrong, but leaves out a mandatory element (a second obligation, a required party, a customer notification, a coverage condition). | RBI: report within X days **and** inform the customer; policy: only "report within X days". |
| `not_applicable` | The clause concerns an internal matter that no RBI regulation governs (staff leave, office facilities, internal meeting logistics). | "Staff are entitled to twelve days of casual leave." |
| `insufficient_evidence` | The clause concerns a matter that could plausibly be regulated, but **no notification in the corpus** governs it, so no verdict can be supported. Absence is checked by searching the corpus (recorded in the rationale). | A clause on a matter governed only by another regulator or statute that the corpus does not cover. |

Decision order when unsure:

1. Is the subject an internal matter outside banking regulation? Then `not_applicable`.
2. Does a corpus notification applicable to commercial banks govern it? If not,
   `insufficient_evidence`.
3. Does the clause state anything inconsistent with that notification? Then `conflict`.
4. Does it omit a mandatory element of the same requirement? Then `gap`.
5. Otherwise `compliant`.

`conflict` beats `gap` when a clause both contradicts one element and omits another.

## Citations

- `citations` is a list of **groups**. Each group lists interchangeable notifications that
  each establish one requirement the verdict relies on. Typical group: the consolidated
  Directions *and* the amendment that inserted the paragraph. A clause is fully cited
  when at least one notification from every group is cited.
- `acceptable` lists notifications that are relevant and correct to cite but not needed
  (for example a related Direction). Citing them is not penalised.
- Older or replaced notifications are **never** gold for a `conflict` caused by the
  change, and other entity types' Directions are never gold.
- `not_applicable` and `insufficient_evidence` have no citation groups.

## Evidence

Every substantive verdict (`compliant`, `conflict`, `gap`) has at least one evidence
quote: verbatim text (whitespace may differ) from a cited notification, with its
paragraph number. `rbirag comply eval check` verifies every quote against the corpus;
`results/compliance/gold_check.json` records the result.

## Tags

| Tag | Meaning |
|---|---|
| `numeric_threshold` | The verdict turns on a number (days, years, amounts, rates). |
| `partial_compliance` | The clause covers the topic but omits a required element (most `gap` items). |
| `outdated_rule` | The clause matches an older notification (still in the corpus) that a newer one replaced. |
| `amendment` | The rule was changed by an amendment Direction. |
| `future_effective` | A relevant change is issued but not yet in force on the snapshot date; the label follows the rule in force. |
| `entity_confusion` | Directions for another entity type state the rule differently (or are likely to be retrieved instead). |
| `stricter_than_required` | The policy goes beyond the minimum; still `compliant`. |
| `paraphrase` | Compliant in substance, worded differently from the notification. |
| `prohibition` | The rule forbids something. |
| `multi_source` | The verdict needs more than one notification. |
| `out_of_scope` | Internal matter (all `not_applicable` items). |
| `not_in_corpus` | No governing notification in the corpus (all `insufficient_evidence` items). |

## Expected tools

`expected_tools` is the **minimum** set a reasonable trajectory uses: substantive verdicts
need `search_regulations` and `compare_clause`, plus `fetch_notification` when the deciding
text is deep in a long Direction. `insufficient_evidence` needs `search_regulations` (the
absence has to be looked for). `not_applicable` needs none.

## Provenance and review

All labels were written by an AI assistant (Claude, `author: "ai:claude"`) from the full
text of the cited notifications, with every quote machine-checked. **None is
human-verified yet.** Reviews are appended to a label's `reviews` list
(`{"reviewer": "human:<name>", "decision": "accepted" | "edited" | "rejected", "note": ...}`);
rejected items are excluded from evaluation, and reports state how many items are
human-verified.

## Splits

Splits are by **policy document**: every clause of a policy is in the same split, so
prompts tuned on dev never see a test policy. Dev is used for prompt and threshold
development only; every reported number is on test.
