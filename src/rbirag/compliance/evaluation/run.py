"""Run a checker (agent or baseline) over labelled items, resumably."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from rbirag.compliance.checker import ClauseChecker
from rbirag.compliance.evaluation.dataset import EvalItem
from rbirag.compliance.judge import ClauseInput
from rbirag.compliance.records import ClausePrediction


def item_input(item: EvalItem) -> ClauseInput:
    return ClauseInput(
        label=item.label.number,
        text=item.label.text,
        policy_title=item.policy_title,
        context=item.section_title,
    )


def load_predictions(path: Path) -> dict[str, ClausePrediction]:
    out: dict[str, ClausePrediction] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = ClausePrediction.model_validate_json(line)
                out[p.item_id] = p  # the last record for an item wins
    return out


def run_items(
    items: Sequence[EvalItem],
    checker: ClauseChecker,
    out_path: Path,
    *,
    resume: bool = True,
    on_item: Callable[[EvalItem, ClausePrediction], None] | None = None,
) -> dict[str, ClausePrediction]:
    """Check every item, appending one JSON line per prediction as it is made.

    With ``resume``, items that already have an error-free prediction in ``out_path`` are
    skipped. Run-stopping errors (quota exhausted, bad configuration, replay cache miss)
    propagate; everything predicted until then is already on disk.
    """
    done = load_predictions(out_path) if resume else {}
    if not resume and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for item in items:
            prev = done.get(item.item_id)
            if prev is not None and prev.error is None:
                continue
            pred = checker.check(item_input(item), item_id=item.item_id)
            fh.write(pred.model_dump_json() + "\n")
            fh.flush()
            done[item.item_id] = pred
            if on_item:
                on_item(item, pred)
    return {i.item_id: done[i.item_id] for i in items if i.item_id in done}
