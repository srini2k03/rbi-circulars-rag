# AI review of the generated Q/A pairs

**This was a review by an AI assistant (Claude), not a human verification.** No pair
reviewed here has `verified: true`, and `human_verified` stays 0 everywhere. A person can
still review the same pairs with `rbirag eval verify`. A human decision always takes
precedence over the AI one.

- Reviewer: `ai-assistant (Claude)`. Review date: 2026-11-03.
- Records: [`data/eval/qa_ai_reviewed.jsonl`](../data/eval/qa_ai_reviewed.jsonl), one
  line per reviewed pair.
- Candidates reviewed: `data/eval/qa_candidates.jsonl` as of commit `9e0c1db`. At that
  commit there were 123 pairs, all answerable: 119 written by `openai/gpt-oss-120b` and
  4 by `gemini-3.5-flash`. 88 of them passed the automatic checks.

## Method

**Source text.** The corpus was the `documents.jsonl.gz` asset of release `corpus-v1`
(2,153 documents). `gh release download` was refused: it uses GraphQL, which is blocked
from this environment. The same asset was fetched instead with the REST API
(`gh api -H "Accept: application/octet-stream" repos/srini2k03/rbi-circulars-rag/releases/assets/705659337`).
Every pair was checked against the full document text at its gold character spans, not
only against the stored passage and quote.

**Sampling.** Seed `20261103`, using `rbirag.eval.ai_review.select_for_review(pairs,
n_answerable=50, n_unanswerable=10, seed=20261103)`:

1. The pool is answerable pairs that pass the automatic checks and have no human
   decision. This leaves out the 35 auto-failed pairs, which evaluation already
   excludes, so the reviewed pairs are a subset of the evaluated pairs. The pool is
   sorted by `qa_id` and shuffled with `random.Random(seed)`.
2. Each question type gets a quota proportional to its share of the pool (largest
   remainder): factual 16, procedural 15, multi-chunk 10, numeric 9.
3. Slots are filled round-robin over the types. Each slot takes the first shuffled pair
   whose year and department are least represented so far.
4. Unanswerable pairs would be shuffled with the same seed and the first 10 taken.
   **The candidate file has no unanswerable pairs yet**, so none were reviewed and none
   could be checked against the corpus. They will be added by later generator runs.

The 50 pairs cover 14 years (1999 to 2025) and 20 named departments, plus pairs with no
department.

**Checks per pair.** Each pair was checked on four points:

- the question is clear and self-contained;
- the gold answer is fully supported by the evidence and the document, including its
  modality ("may" vs "must") and completeness;
- the gold span holds the evidence;
- the question is neither answerable without the document nor ambiguous across many
  circulars. Where a near-identical text might exist in another document, the corpus
  was searched with grep.

**Decisions.**

- **accept**: the pair is used as it is.
- **edit**: the question and/or answer wording is fixed. The new wording stays within the
  quoted evidence and the passage the pair was written from. Gold spans are not changed.
- **reject**: the pair is excluded from evaluation.

Each decision carries a one-line note.

## Results

| type | accepted | edited | rejected | total |
|---|---:|---:|---:|---:|
| factual | 9 | 5 | 2 | 16 |
| numeric | 4 | 4 | 1 | 9 |
| procedural | 5 | 7 | 3 | 15 |
| multi-chunk | 2 | 7 | 1 | 10 |
| unanswerable | 0 | 0 | 0 | 0 (none exist yet) |
| **total** | **20** | **23** | **7** | **50** |

Only 40% of the sampled pairs were usable as written. Multi-chunk questions fared worst:
2 of 10 were accepted as written.

## Main failure patterns

1. **Questions that do not say which circular they mean (about 15 pairs, all edited).**
   RBI repeats near-identical notices: "Liquidity Adjustment Facility - Change in
   rates", "Standing Liquidity Facility for Primary Dealers", GoI auction notices,
   Sovereign Gold Bond tranches, and master circulars for banks vs. UCBs. Questions such
   as "What are the revised SDF and MSF rates?" or "What auction method must be used for
   the securities?" match dozens of documents, while the gold names only one. The model
   dropped the date, scheme or entity type that pins the document down. The edits add
   that anchor: a date, the scheme year, "for primary (urban) co-operative banks", or
   the direction's name.
2. **Answers that overstate or contradict the text (4 pairs).**
   - One answer inverted a rule. AIF investments are "treated in the same manner" as
     other group exposures, which are *deducted* when computing NOF; the answer said they
     are "counted towards" it.
   - "may be excluded" became "must exclude".
   - "it shall be the endeavour to collect by" became "must be collected by".
   - "PAN details" became "PAN card".
3. **Partial answers presented as complete (3 pairs).** These answers listed two of the
   four C-SOC responsibilities, two of the four geo-tagging benefits, or one BSBD
   facility without its free-of-charge condition.
4. **Multi-chunk questions that are not really multi-chunk, or are incoherent (2 pairs).**
   One used only part 1. Another asked how *dematerialised* bonds transfer "after
   re-materialisation".
5. **Evidence that is a list row or title, with the answer made up from it (2 rejected).**
   In master circulars and withdrawal notices, the appendix lists of circulars led the
   model to treat a circular's title as a rule: "MPs must participate in DLRC meetings".
   It also took a circular's issue date for a deadline: "appoint auditors by June 13,
   1972".
6. **Trivial or metadata questions (3 rejected, 1 accepted).** These include the date of
   a letter, a regional-office code ("HY" means Hyderabad), and boilerplate ("ensure
   instructions are adhered to in letter and spirit"). One date question was also a
   retrieval trap: the cited letter is itself in the corpus (`rbi-12544`), but the gold
   names only the later letter that refers to it.
7. **Repealed text presented as current (1 rejected).** The evidence came from a Master
   Direction's endnotes, which quote provisions as they read before the June 2026
   repeal.

Evidence spans were accurate. Every sampled gold span was found in the document and held
the quoted text. This was expected, because the sample only includes pairs that passed
the automatic location check.

## Worked examples

**1. Accepted: `q5a40d37a599e` (multi-chunk, 2004, Banking Supervision).**

- Q: "According to the RBI's 2004 circular on concurrent audit, what condition must banks
  satisfy regarding the Exchange House's vostro account before they remit funds to
  beneficiaries under the Speed Remittance arrangement?"
- Gold answer: sufficient funds in the vostro account, credited before payment, with no
  overdraft.
- Verdict: the question is dated and specific. Part 1 gives the sufficient-funds and
  no-overdraft rule, and part 2 gives the credit-before-payment audit check. The answer
  uses both, and nothing is added.

**2. Edited for ambiguity: `qad4ae4be3ce5` (factual, 2023, FMOD).**

- Original: "What are the updated rates for the SDF and the MSF as announced in the RBI
  Liquidity Adjustment Facility notification?"
- Problem: the corpus has a "LAF - Change in rates" notice for every policy move, so the
  question has many correct answers.
- Edited: "After the repo rate increase announced in the Monetary Policy Statement of
  February 8, 2023, what were the SDF and MSF rates?" The answer is unchanged: 6.25% and
  6.75%.

**3. Edited for a wrong answer: `q813407e63fdd` (multi-chunk, 2024, Regulation).**

- The original answer said an HFC's AIF investment "is treated as part of the Net Owned
  Fund" when the HFC supplied 50% or more of its funds.
- The circular says such investments are "treated in the same manner" as direct
  exposures to group entities. Those are reduced from owned fund above 10% when arriving
  at NOF. The original answer had the meaning backwards.
- The question and answer were rewritten to state the deduction and both conditions:
  the AIF company case and the AIF trust case.

**4. Rejected: `q3a537e55107e` (procedural, 2020, Financial Inclusion).**

- Q: "According to the Lead Bank Scheme, what is required of Members of Parliament and
  public representatives regarding DLRC meetings?"
- The "evidence" is a title in the Master Circular's list of consolidated circulars:
  "Lead Bank Scheme - Participation of Members of Parliament ... in DLRC meetings".
- The answer ("they must participate") is not stated anywhere in the document.

**5. Rejected: `qa98f0dc39676` (procedural, 2024).**

- Q: "By what date must the Central Statutory Auditors be appointed for the year 1972?"
- The evidence is a row in a 2024 list of withdrawn circulars: "Appointment of Central
  Statutory Auditors for the year 1972 June 13, 1972". June 13, 1972 is the circular's
  issue date, not a deadline, so the gold answer is unsupported.

## How the review is used

- `rbirag eval` commands load the test set through `load_test_set`, which:
  - drops human-rejected and auto-failed pairs;
  - applies `qa_ai_reviewed.jsonl`: AI-rejected pairs are dropped and AI-edited wording
    is used.

  A review is ignored if a human has already decided on the pair, or if the candidate's
  question or answer no longer matches the one reviewed.
- The file is append-only and keyed by `qa_id`: the last line for an id wins. Pairs the
  daily generator adds later have no review and are used as model-generated pairs.
  Rejected pairs still count towards the generator's targets, because regenerating them
  would not change the sample.
- Retrieval per-query results and generation records now store the question wording. A
  result for a pair that has since been edited or rejected is left out of the report.
  `eval generate` answers edited questions again, and `eval progress` counts them as
  pending.
- The results README and `run_metadata.json`:
  - label the test set, for example "model-generated test set
    (`data/eval/qa_candidates.jsonl`): 81 model-generated pairs (43 AI-reviewed by
    Claude, 0 human-verified); 7 AI-rejected pairs excluded";
  - give the headline tables on the full set;
  - give an "AI-reviewed subset" section with the same metrics on the accepted and
    edited pairs.
