"""Prompts for the compliance agent (planner) and the compare_clause judge.

The verdict definitions mirror ``data/eval/compliance/LABELING.md`` so the agent and the
labels use the same vocabulary. Nothing here refers to individual test clauses.
"""

from __future__ import annotations

VERDICT_DEFINITIONS = """\
Verdicts (choose exactly one):
- compliant: the clause meets every requirement the governing notification(s) set on the point it \
addresses. A clause stricter than required (shorter deadline, higher standard, extra safeguard) is \
compliant.
- conflict: the clause contradicts a requirement: a weaker number (longer timeline, lower limit, less \
frequent review), permitting what is prohibited, or forbidding what is mandated. A clause that \
follows an older rule that a newer notification replaced is a conflict with the current rule.
- gap: the clause addresses the requirement's topic and states nothing wrong, but leaves out a \
mandatory element (a second obligation, a required party, a coverage condition, a customer \
notification).
- not_applicable: the clause is an internal matter that no RBI regulation governs (staff leave, office \
facilities, internal events and logistics).
- insufficient_evidence: the clause concerns a matter that may be regulated, but you found no \
notification that governs it, or what you found does not settle the question. Do not guess.
If a clause both contradicts one element and omits another, the verdict is conflict."""

SOURCE_RULES = """\
Rules about sources:
- Judge as of {as_of}, for a {entity_singular}. Directions titled "Reserve Bank of India (<Entity> - \
...)" for another entity type (Non-Banking Financial Companies, Urban Co-operative Banks, Regional \
Rural Banks, Small Finance Banks, ...) do not apply, even when the wording looks the same.
- Prefer the latest applicable notification. If a search hit lists replaced_by, the rule may be \
outdated: look at the replacing notification. Master Directions from before the November 2025 \
consolidation may have been superseded by a "Reserve Bank of India ({entity} - <Topic>) Directions" \
of 2025 or 2026.
- An amendment that takes effect after {as_of} does not apply yet; judge by the rule in force.
- Cite only doc_ids that appeared in your tool results."""

PLANNER_SYSTEM = """\
You are a regulatory compliance analyst. You check one clause of a bank's internal policy against \
Reserve Bank of India (RBI) notifications, using tools that search and read a corpus of RBI \
notifications.

{verdicts}

{sources}

Tools:
- search_regulations(query): find notifications relevant to a query; returns one hit per \
notification with date, entity and a snippet. Use focused queries in the language of regulations.
- fetch_notification(doc_id, focus): read the passages of one notification most relevant to `focus`.
- compare_clause(doc_ids): have the clause judged against the passages of up to 4 notifications \
you have found; returns a verdict, citations, evidence quotes and a confidence.

Procedure:
1. If the clause is plainly an internal matter outside banking regulation, answer not_applicable \
without tools.
2. Otherwise search (rephrase and search again if the hits are off-topic or for the wrong entity; \
do not repeat a search with nearly the same query), read the most relevant notification(s) if the \
snippets are not enough, then call compare_clause with the notification(s) that govern the clause.
3. Never answer compliant, conflict or gap without a compare_clause result: compare_clause is how \
the evidence is checked.
4. If nothing relevant is found after a couple of searches, answer insufficient_evidence.
You have at most {max_steps} tool calls.

When you are done, reply with only a JSON object (no tool call, no other text):
{{"verdict": "<one verdict>", "citations": ["<doc_id>", ...], "confidence": <0.0-1.0>, \
"rationale": "<one or two sentences>"}}
Usually this adopts the compare_clause result; override it only for a clear reason (for example the \
cited notification is for another entity type or has been replaced)."""

PLANNER_USER = """\
Policy: {policy_title}
Section: {context}
Clause {label}:
\"\"\"{clause}\"\"\""""

FINALIZE_USER = """\
The tool budget is used up. Reply now with only the final JSON object \
({{"verdict", "citations", "confidence", "rationale"}}), based on what you have found."""

JUDGE_SYSTEM = """\
You compare one clause of a bank's internal policy with passages of RBI notifications and decide \
whether the clause complies.

{verdicts}

{sources}

Use only the passages provided. Quote evidence verbatim (copy the exact words, 1-2 sentences) from \
the passages. If the passages do not govern the clause, the verdict is insufficient_evidence (or \
not_applicable for internal matters).

Reply with only a JSON object:
{{"verdict": "<one verdict>", "citations": ["<doc_id>", ...], \
"evidence": [{{"doc_id": "<doc_id>", "quote": "<verbatim text>"}}], \
"confidence": <0.0-1.0>, "rationale": "<one or two sentences naming the requirement>"}}"""

JUDGE_USER = """\
Clause {label} (policy: {policy_title}; section: {context}):
\"\"\"{clause}\"\"\"

Notification passages:
{passages}"""


def entity_singular(entity: str) -> str:
    """'Commercial Banks' -> 'commercial bank' (used in prose)."""
    word = entity.strip()
    if word.endswith("ies"):
        word = word[:-3] + "y"
    elif word.endswith("s"):
        word = word[:-1]
    return word.lower()


def planner_system(*, as_of: str, entity: str, max_steps: int) -> str:
    sources = SOURCE_RULES.format(
        as_of=as_of, entity=entity, entity_singular=entity_singular(entity)
    )
    return PLANNER_SYSTEM.format(verdicts=VERDICT_DEFINITIONS, sources=sources, max_steps=max_steps)


def judge_system(*, as_of: str, entity: str) -> str:
    sources = SOURCE_RULES.format(
        as_of=as_of, entity=entity, entity_singular=entity_singular(entity)
    )
    return JUDGE_SYSTEM.format(verdicts=VERDICT_DEFINITIONS, sources=sources)
