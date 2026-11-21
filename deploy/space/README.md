---
title: RBI Policy Compliance Checker
emoji: 🏦
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Check a bank policy clause by clause against RBI rules
---

# RBI policy compliance checker

Upload a bank's internal policy (PDF, text or Markdown). Each clause is checked against
2,903 Reserve Bank of India notifications (snapshot as of 2026-11-11) and gets a verdict:
**conflict**, **gap**, **compliant**, **insufficient evidence** or **not applicable**, with
the notifications relied on, verified evidence quotes and a confidence score. The
**Ask** page answers questions about the notifications with cited sources.

Runs entirely on free tiers (Hugging Face CPU Space, Gemini / Groq free API tiers). The
demo checks at most 15 clauses per run. Without an API key it falls back to an offline
demo model that always answers "insufficient evidence".

Not legal advice. Source code, evaluation (labelled clause set, metrics with bootstrap
confidence intervals, failure analysis) and limitations:
https://github.com/srini2k03/rbi-circulars-rag
