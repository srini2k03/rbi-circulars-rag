"""Check a whole policy document: split it into clauses, check each, build a report."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from rbirag.compliance.clauses import Clause, split_clauses
from rbirag.compliance.judge import ClauseInput
from rbirag.compliance.policy import PolicyDocument
from rbirag.compliance.records import VERDICTS, ClausePrediction
from rbirag.compliance.regulations import RegulationStore


class ClauseChecker(Protocol):
    def check(self, clause: ClauseInput, *, item_id: str | None = None) -> ClausePrediction: ...


class SourceRef(BaseModel):
    doc_id: str
    title: str
    date: str | None
    url: str
    quotes: list[str] = Field(default_factory=list)


class ClauseFinding(BaseModel):
    clause: Clause
    prediction: ClausePrediction
    sources: list[SourceRef] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    policy_name: str
    source_type: str
    pages: int | None = None
    checked_at: str
    as_of: str
    entity: str
    system: str
    model: str | None
    findings: list[ClauseFinding]
    summary: dict[str, int]
    llm_calls: int
    tokens: int
    cost_usd: float
    latency_ms: float
    errors: int


def clause_input(clause: Clause, policy_title: str) -> ClauseInput:
    text = clause.text
    if clause.number and text.startswith(clause.number):
        text = text[len(clause.number) :].lstrip(" .:)-")
    return ClauseInput(
        label=clause.label, text=text, policy_title=policy_title, context=clause.context
    )


def check_policy(
    doc: PolicyDocument,
    checker: ClauseChecker,
    store: RegulationStore,
    *,
    as_of: str,
    entity: str,
    system: str = "agent",
    max_clauses: int | None = None,
    on_progress: Callable[[int, int, ClauseFinding], None] | None = None,
) -> ComplianceReport:
    clauses = split_clauses(doc.text)
    if max_clauses is not None:
        clauses = clauses[:max_clauses]
    title = doc.name.rsplit(".", 1)[0]
    findings: list[ClauseFinding] = []
    for i, clause in enumerate(clauses, start=1):
        pred = checker.check(clause_input(clause, title), item_id=clause.label)
        quotes: dict[str, list[str]] = {}
        for ev in pred.evidence:
            if ev.verified:
                quotes.setdefault(ev.doc_id, []).append(ev.quote)
        sources = []
        for doc_id in pred.citations:
            d = store.documents.get(doc_id)
            if d is not None:
                sources.append(
                    SourceRef(
                        doc_id=doc_id,
                        title=d.title,
                        date=d.date.isoformat() if d.date else None,
                        url=d.url,
                        quotes=quotes.get(doc_id, []),
                    )
                )
        finding = ClauseFinding(clause=clause, prediction=pred, sources=sources)
        findings.append(finding)
        if on_progress:
            on_progress(i, len(clauses), finding)
    preds = [f.prediction for f in findings]
    counts: Counter[str] = Counter(str(p.verdict) for p in preds)
    models = [p.model for p in preds if p.model]
    return ComplianceReport(
        policy_name=doc.name,
        source_type=doc.source_type,
        pages=doc.pages,
        checked_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        as_of=as_of,
        entity=entity,
        system=system,
        model=max(set(models), key=models.count) if models else None,
        findings=findings,
        summary={v: counts[v] for v in VERDICTS},
        llm_calls=sum(p.llm_calls for p in preds),
        tokens=sum(p.input_tokens + p.output_tokens for p in preds),
        cost_usd=round(sum(p.cost_usd for p in preds), 6),
        latency_ms=round(sum(p.latency_ms for p in preds), 1),
        errors=sum(1 for p in preds if p.error),
    )


_BADGE = {
    "compliant": "compliant",
    "conflict": "CONFLICT",
    "gap": "GAP",
    "not_applicable": "not applicable",
    "insufficient_evidence": "insufficient evidence",
}


def render_markdown(report: ComplianceReport) -> str:
    s = report.summary
    lines = [
        f"# Compliance check: {report.policy_name}",
        "",
        f"Checked against RBI notifications as of {report.as_of}, for {report.entity.lower()}. "
        f"System: {report.system} ({report.model or 'n/a'}).",
        "",
        f"**{len(report.findings)} clauses**: {s['conflict']} conflict, {s['gap']} gap, "
        f"{s['compliant']} compliant, {s['insufficient_evidence']} insufficient evidence, "
        f"{s['not_applicable']} not applicable."
        + (f" {report.errors} clause(s) failed to run." if report.errors else ""),
        "",
        f"LLM calls: {report.llm_calls}; tokens: {report.tokens:,}; "
        f"cost: ${report.cost_usd:.4f}; time: {report.latency_ms / 1000:.1f} s.",
        "",
        "| clause | verdict | confidence | sources |",
        "|---|---|---|---|",
    ]
    for f in report.findings:
        p = f.prediction
        conf = "-" if p.confidence is None else f"{p.confidence:.2f}"
        srcs = ", ".join(s.doc_id for s in f.sources) or "-"
        lines.append(f"| {f.clause.label} | {_BADGE[p.verdict]} | {conf} | {srcs} |")
    for f in report.findings:
        p = f.prediction
        lines += [
            "",
            f"## {f.clause.label}: {_BADGE[p.verdict]}",
            "",
            f"> {' '.join(f.clause.text.split())}",
            "",
        ]
        if p.error:
            lines.append(f"**Run failed:** {p.error}")
            continue
        if p.rationale:
            lines.append(p.rationale)
        for src in f.sources:
            lines += ["", f"- [{src.doc_id}]({src.url}) {src.title} ({src.date or 'undated'})"]
            lines += [f'  - "{q}"' for q in src.quotes]
        if p.notes:
            lines += ["", "_Checks: " + "; ".join(p.notes) + "_"]
    return "\n".join(lines) + "\n"
