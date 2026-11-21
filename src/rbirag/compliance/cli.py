"""``rbirag comply ...``: inspect clause splitting and the agent's regulation tools."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from rbirag.config import RetrievalMode, Settings, get_settings

if TYPE_CHECKING:
    from rbirag.compliance.checker import ClauseChecker, ClauseFinding
    from rbirag.compliance.regulations import RegulationStore

comply_app = typer.Typer(
    help="Check a bank policy against RBI regulations (tools and clause splitting).",
    no_args_is_help=True,
)


def _register_eval() -> None:
    from rbirag.compliance.evaluation.cli import eval_app

    comply_app.add_typer(eval_app, name="eval")


_register_eval()


def _echo_json(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _store(mode: RetrievalMode | None) -> RegulationStore:
    from rbirag.compliance.regulations import RegulationStore

    s = get_settings()
    if mode is not None:
        s = s.model_copy(update={"compliance_search_mode": mode})
    try:
        return RegulationStore.from_settings(s)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@comply_app.command("clauses")
def clauses(
    policy: Annotated[Path, typer.Argument(help="Policy document (.pdf, .txt, .md).")],
    as_json: Annotated[bool, typer.Option("--json", help="Print clauses as JSON.")] = False,
) -> None:
    """Split a policy document into clauses."""
    from rbirag.compliance.clauses import split_clauses
    from rbirag.compliance.policy import PolicyLoadError, load_policy

    try:
        doc = load_policy(policy)
    except PolicyLoadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    found = split_clauses(doc.text)
    if as_json:
        _echo_json([c.model_dump() for c in found])
        return
    typer.echo(f"{doc.name}: {len(found)} clauses ({doc.source_type})")
    for c in found:
        head = f"[{c.label}]" + (f" {c.context}" if c.context else "")
        typer.echo(f"\n{head}\n  {' '.join(c.text.split())[:300]}")


@comply_app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Clause text or a search query.")],
    top_k: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    mode: Annotated[RetrievalMode | None, typer.Option(help="Retrieval mode override.")] = None,
    date_from: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    date_to: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """search_regulations: notifications relevant to a clause, one hit per notification."""
    store = _store(mode)
    result = store.search(
        query,
        top_k=top_k,
        date_from=dt.date.fromisoformat(date_from) if date_from else None,
        date_to=dt.date.fromisoformat(date_to) if date_to else None,
    )
    if as_json:
        _echo_json(result.model_dump())
        return
    conf = "n/a" if result.confidence is None else f"{result.confidence:.2f}"
    typer.echo(f"mode={result.mode} confidence={conf}")
    for h in result.hits:
        status = f"  [replaced by {', '.join(h.replaced_by)}]" if h.replaced_by else ""
        typer.echo(f"\n{h.rank}. {h.doc_id} ({h.date}) {h.title}{status}\n   {h.snippet[:240]}")


@comply_app.command("fetch")
def fetch(
    notification: Annotated[str, typer.Argument(help="rbi-12655, 12655, or a circular no.")],
    query: Annotated[str | None, typer.Option(help="Select passages relevant to this.")] = None,
    max_tokens: Annotated[int | None, typer.Option(min=100)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """fetch_notification: metadata and the relevant passages of one notification."""
    from rbirag.compliance.regulations import UnknownNotificationError

    store = _store("bm25")  # fetching needs no retrieval models
    try:
        view = store.fetch(notification, query=query, max_tokens=max_tokens)
    except UnknownNotificationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        _echo_json(view.model_dump())
        return
    typer.echo(f"{view.doc_id} ({view.date}) {view.title}\n{view.url}")
    if view.replaced_by:
        typer.echo(f"replaced by: {', '.join(view.replaced_by)}")
    for p in view.passages:
        typer.echo(f"\n--- {p.number or ''} {p.heading or ''}".rstrip())
        typer.echo(p.text)
    if view.truncated:
        typer.echo(f"\n[{view.total_chars} chars in total; showing the most relevant passages]")


def build_checker(
    system: str,
    *,
    backend: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> tuple[ClauseChecker, RegulationStore]:
    """The agent or the baseline, wired from settings (shared by check, eval run and
    eval replay)."""
    from rbirag.compliance.agent import ComplianceAgent
    from rbirag.compliance.baseline import FixedPipeline
    from rbirag.compliance.llm import build_chat_model
    from rbirag.compliance.regulations import RegulationStore

    s = settings or get_settings()
    try:
        store = RegulationStore.from_settings(s)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    chat = build_chat_model(s, backend, model=model)
    if system == "agent":
        return ComplianceAgent.from_settings(s, store, chat), store
    if system == "baseline":
        return FixedPipeline.from_settings(s, store, chat), store
    raise typer.BadParameter("system must be agent or baseline")


@comply_app.command("check")
def check(
    policy: Annotated[Path, typer.Argument(help="Policy document (.pdf, .txt, .md).")],
    system: Annotated[str, typer.Option(help="agent | baseline")] = "agent",
    backend: Annotated[str | None, typer.Option(help="groq | gemini | ollama | custom")] = None,
    model: Annotated[str | None, typer.Option(help="Override the model id.")] = None,
    max_clauses: Annotated[int | None, typer.Option(min=1)] = None,
    out: Annotated[Path | None, typer.Option(help="Write the Markdown report here.")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Full report JSON.")] = None,
) -> None:
    """Check every clause of a policy document and print a report."""
    from rbirag.compliance.checker import check_policy, render_markdown
    from rbirag.compliance.policy import PolicyLoadError, load_policy

    try:
        doc = load_policy(policy)
    except PolicyLoadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    checker, store = build_checker(system, backend=backend, model=model)
    s = get_settings()

    def progress(i: int, n: int, finding: ClauseFinding) -> None:
        typer.echo(f"[{i}/{n}] {finding.clause.label}: {finding.prediction.verdict}", err=True)

    report = check_policy(
        doc,
        checker,
        store,
        as_of=s.compliance_as_of,
        entity=s.compliance_entity,
        system=system,
        max_clauses=max_clauses,
        on_progress=progress,
    )
    md = render_markdown(report)
    if out:
        out.write_text(md, encoding="utf-8")
    if json_out:
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(md)
