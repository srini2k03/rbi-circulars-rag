"""Turn the agent's proposed answer into a defensible decision.

Rules, applied in order (each action is recorded in ``notes``):

1. Citations must be notifications the agent actually saw in a search or fetch result;
   anything else is dropped (a hallucinated or guessed id).
2. Citations of Directions addressed to another entity type are dropped.
3. A substantive verdict (compliant / conflict / gap) needs a successful
   ``compare_clause`` call, at least one remaining citation and, unless disabled, a
   verified evidence quote from a cited notification. Otherwise the verdict becomes
   ``insufficient_evidence``: the system says it cannot support an answer rather than
   guessing. Non-substantive verdicts carry no citations (what was considered is noted).
4. Confidence is the proposed confidence, discounted when citations were dropped, when
   a cited notification is marked as replaced, or when no quote was verified.
5. A substantive verdict below the abstention threshold becomes
   ``insufficient_evidence`` (threshold chosen on the dev split).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from rbirag.compliance.entities import TARGET_ENTITY, is_other_entity
from rbirag.compliance.judge import Judgement
from rbirag.compliance.records import SUBSTANTIVE_VERDICTS, EvidenceQuote, Verdict

DROPPED_CITATION_FACTOR = 0.85
REPLACED_CITATION_FACTOR = 0.8
UNVERIFIED_QUOTE_FACTOR = 0.7


class Proposal(BaseModel):
    """What the planner (or a baseline) proposes before the guardrails."""

    verdict: Verdict
    citations: list[str] = Field(default_factory=list)
    confidence: float | None = None
    rationale: str = ""


class Decision(BaseModel):
    verdict: Verdict
    citations: list[str]
    confidence: float
    rationale: str
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def decide(
    proposal: Proposal | None,
    judgement: Judgement | None,
    *,
    seen_doc_ids: set[str],
    titles: Mapping[str, str],
    replaced_by: Callable[[str], Sequence[str]],
    entity: str = TARGET_ENTITY,
    require_verified_quote: bool = True,
    abstain_threshold: float = 0.0,
) -> Decision:
    notes: list[str] = []
    if proposal is None and judgement is None:
        return Decision(
            verdict="insufficient_evidence",
            citations=[],
            confidence=0.0,
            rationale="no answer was produced",
            notes=["no proposal and no comparison: abstained"],
        )
    if proposal is None:
        assert judgement is not None
        proposal = Proposal(
            verdict=judgement.verdict,
            citations=judgement.citations,
            confidence=judgement.confidence,
            rationale=judgement.rationale,
        )
        notes.append("no final answer from the planner: adopted the comparison")
    verdict: Verdict = proposal.verdict
    confidence = (
        proposal.confidence
        if proposal.confidence is not None
        else (judgement.confidence if judgement else 0.5)
    )
    rationale = proposal.rationale or (judgement.rationale if judgement else "")

    citations = list(dict.fromkeys(proposal.citations))
    if not citations and judgement is not None and verdict in SUBSTANTIVE_VERDICTS:
        citations = list(judgement.citations)
    unseen = [c for c in citations if c not in seen_doc_ids]
    if unseen:
        notes.append(f"dropped citation(s) never shown by a tool: {', '.join(unseen)}")
    citations = [c for c in citations if c in seen_doc_ids]
    wrong_entity = [c for c in citations if is_other_entity(titles.get(c), entity)]
    if wrong_entity:
        notes.append(
            f"dropped citation(s) addressed to another entity type: {', '.join(wrong_entity)}"
        )
    citations = [c for c in citations if c not in wrong_entity]
    if unseen or wrong_entity:
        confidence *= DROPPED_CITATION_FACTOR

    evidence = [e for e in (judgement.evidence if judgement else []) if e.doc_id in citations]
    if verdict in SUBSTANTIVE_VERDICTS:
        reason = None
        if judgement is None:
            reason = "no compare_clause result supports it"
        elif not citations:
            reason = "no valid citation remains"
        elif require_verified_quote and not any(e.verified for e in evidence):
            reason = "no evidence quote could be verified in the notifications read"
        if reason:
            notes.append(f"{verdict} -> insufficient_evidence: {reason}")
            if citations:
                notes.append(f"considered: {', '.join(citations)}")
            return Decision(
                verdict="insufficient_evidence",
                citations=[],
                confidence=min(confidence, 0.5),
                rationale=rationale,
                evidence=evidence,
                notes=notes,
            )
        if not any(e.verified for e in evidence):
            confidence *= UNVERIFIED_QUOTE_FACTOR
        replaced = [c for c in citations if any(r not in citations for r in replaced_by(c))]
        if replaced:
            notes.append(f"cited notification(s) marked as replaced: {', '.join(replaced)}")
            confidence *= REPLACED_CITATION_FACTOR
        if confidence < abstain_threshold:
            notes.append(
                f"{verdict} -> insufficient_evidence: confidence {confidence:.2f} below "
                f"threshold {abstain_threshold:.2f}"
            )
            notes.append(f"considered: {', '.join(citations)}")
            return Decision(
                verdict="insufficient_evidence",
                citations=[],
                confidence=confidence,
                rationale=rationale,
                evidence=evidence,
                notes=notes,
            )
    else:
        if citations:
            notes.append(f"considered: {', '.join(citations)}")
        citations = []  # not_applicable / insufficient_evidence assert nothing to cite
    return Decision(
        verdict=verdict,
        citations=citations,
        confidence=round(min(1.0, max(0.0, confidence)), 4),
        rationale=rationale,
        evidence=evidence,
        notes=notes,
    )
