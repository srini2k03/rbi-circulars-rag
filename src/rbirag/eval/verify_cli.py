"""Terminal review of candidate Q/A pairs: accept / edit / reject, resumable.

Decisions are written to the output file (``qa_v1.jsonl``) after every item, merged with
all candidates, so quitting at any point loses nothing and a later run resumes with the
next unreviewed pair. ``verified`` is set to True only for pairs a human accepted or
edited.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from rbirag.eval.dataset import QAPair, read_qa_jsonl, write_qa_jsonl

AskChoice = Callable[[str, list[str], str], str]
AskText = Callable[[str, str], str]


def _rich_choice(prompt: str, choices: list[str], default: str) -> str:
    return Prompt.ask(prompt, choices=choices, default=default)


def _rich_text(prompt: str, default: str) -> str:
    return Prompt.ask(prompt, default=default)


def merge(candidates: list[QAPair], reviewed: list[QAPair]) -> list[QAPair]:
    """Candidates in order, with any reviewed version replacing its candidate."""
    by_id = {p.qa_id: p for p in reviewed}
    merged = [by_id.pop(c.qa_id, c) for c in candidates]
    merged.extend(by_id.values())  # reviewed pairs no longer in the candidate file
    return merged


def review_summary(pairs: list[QAPair]) -> dict[str, int]:
    c = Counter(p.review_status for p in pairs)
    return {
        "total": len(pairs),
        "pending": c["pending"],
        "accepted": c["accepted"],
        "edited": c["edited"],
        "rejected": c["rejected"],
        "human_verified": sum(1 for p in pairs if p.verified),
    }


def _render(console: Console, pair: QAPair, position: int, total: int) -> None:
    meta = Table.grid(padding=(0, 2))
    meta.add_row("type", pair.qtype)
    meta.add_row("answerable", "yes" if pair.answerable else "NO (should not be answerable)")
    if pair.doc_title:
        meta.add_row("document", f"{pair.doc_title} ({pair.year or '?'}, {pair.department or '?'})")
    if pair.auto_check is False:
        meta.add_row("auto-check", "[yellow]failed (evidence not found or question too vague)[/]")
    console.rule(f"[bold]Pair {position}/{total}[/]  {pair.qa_id}")
    console.print(meta)
    if pair.source_text:
        body = Text(pair.source_text)
        if pair.evidence:
            body.highlight_words(
                [pair.evidence], style="bold black on yellow", case_sensitive=False
            )
        console.print(Panel(body, title="source passage", border_style="blue"))
    console.print(Panel(pair.question, title="question", border_style="green"))
    if pair.answerable:
        console.print(
            Panel(pair.answer or "(empty)", title="reference answer", border_style="green")
        )


def run_review(
    candidates_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    seed: int = 13,
    include_auto_failed: bool = True,
    console: Console | None = None,
    ask_choice: AskChoice = _rich_choice,
    ask_text: AskText = _rich_text,
) -> dict[str, int]:
    console = console or Console()
    candidates = read_qa_jsonl(candidates_path)
    reviewed = read_qa_jsonl(out_path) if out_path.exists() else []
    pairs = merge(candidates, reviewed)
    index = {p.qa_id: i for i, p in enumerate(pairs)}

    pending = [
        p for p in pairs if p.review_status == "pending" and (include_auto_failed or p.auto_check)
    ]
    rng = random.Random(seed)
    rng.shuffle(pending)
    if limit is not None:
        pending = pending[:limit]

    summary = review_summary(pairs)
    console.print(
        f"[bold]{summary['human_verified']}[/] verified so far, {summary['pending']} pending. "
        "Keys: [b]a[/]ccept, [b]e[/]dit, [b]r[/]eject, [b]s[/]kip, [b]q[/]uit."
    )
    write_qa_jsonl(pairs, out_path)  # materialise the merged file immediately

    for n, pair in enumerate(pending, start=1):
        _render(console, pair, n, len(pending))
        choice = ask_choice("decision", ["a", "e", "r", "s", "q"], "a")
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "a":
            updated = pair.model_copy(update={"review_status": "accepted", "verified": True})
        elif choice == "e":
            question = ask_text("question", pair.question).strip() or pair.question
            answer = pair.answer
            if pair.answerable:
                answer = ask_text("answer", pair.answer).strip() or pair.answer
            updated = pair.model_copy(
                update={
                    "question": question,
                    "answer": answer,
                    "review_status": "edited",
                    "verified": True,
                }
            )
        else:
            note = ask_text("reason (optional)", "").strip()
            updated = pair.model_copy(
                update={"review_status": "rejected", "verified": False, "reviewer_note": note}
            )
        pairs[index[pair.qa_id]] = updated
        write_qa_jsonl(pairs, out_path)  # persist after every decision

    summary = review_summary(pairs)
    console.print(
        f"Saved to {out_path}: {summary['human_verified']} human-verified "
        f"({summary['accepted']} accepted, {summary['edited']} edited), "
        f"{summary['rejected']} rejected, {summary['pending']} pending."
    )
    return summary
