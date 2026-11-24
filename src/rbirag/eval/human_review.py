"""Human review of generated answers, and agreement of the automatic checks with it.

``rbirag eval review-answers`` draws a seeded sample of committed answers, stratified by
the LLM judge's verdict and the citation verifier's verdict (so the disagreements are
represented), shows each answer with the passages it was generated from, and records the
reviewer's labels in ``data/eval/answer_reviews.jsonl`` (one line per answer, resumable).
``rbirag eval review-agreement`` reports Cohen's kappa of the judge, the lexical verifier
and NLI against those labels. LLM verdicts are never treated as ground truth; until a
person has reviewed answers, the agreement report says so and contains no numbers.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

REVIEW_FILE = "answer_reviews.jsonl"
LABELS = {"f": "faithful", "p": "partially faithful", "u": "unfaithful", "s": "skip"}


class AnswerReview(BaseModel):
    qa_id: str
    config: str
    backend: str
    reviewer: str  # a person's name or handle; "ai-..." marks an AI review
    faithful: str  # faithful | partially faithful | unfaithful
    correct: bool | None = None  # matches the reference answer
    note: str = ""
    reviewed_at: str


def sample_for_review(
    items: Sequence[dict[str, Any]], n: int, *, seed: int = 13
) -> list[dict[str, Any]]:
    """Seeded sample stratified by (judge fully faithful?, verifier verdict)."""
    strata: dict[tuple[bool, str], list[dict[str, Any]]] = defaultdict(list)
    for it in sorted(items, key=lambda x: (x["qa_id"], x["backend"])):
        strata[((it.get("judge") or 0) >= 0.999, str(it.get("verdict")))].append(it)
    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(out) < n:
                out.append(strata[key].pop())
    return out


def read_reviews(path: Path) -> dict[tuple[str, str, str], AnswerReview]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], AnswerReview] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = AnswerReview.model_validate_json(line)
            out[(r.qa_id, r.config, r.backend)] = r
    return out


def review_loop(
    items: Sequence[dict[str, Any]],
    show: Callable[[dict[str, Any]], str],
    path: Path,
    *,
    reviewer: str,
    ask: Callable[[str], str] = input,
    echo: Callable[[str], None] = print,
) -> int:
    """Interactive labelling; returns the number of new reviews. ``q`` quits."""
    done = read_reviews(path)
    added = 0
    for it in items:
        key = (it["qa_id"], it["config"], it["backend"])
        if key in done:
            continue
        echo(show(it))
        label = ask(
            "faithful to the sources? [f]aithful / [p]artially / [u]nfaithful / [s]kip / [q]uit: "
        )
        label = label.strip().lower()[:1]
        if label == "q":
            break
        if label not in LABELS or label == "s":
            continue
        correct = ask("matches the reference answer? [y/n/enter=unsure]: ").strip().lower()[:1]
        note = ask("note (optional): ").strip()
        review = AnswerReview(
            qa_id=it["qa_id"], config=it["config"], backend=it["backend"], reviewer=reviewer,
            faithful=LABELS[label], correct={"y": True, "n": False}.get(correct), note=note,
            reviewed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(review.model_dump_json() + "\n")
        added += 1
    return added


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def agreement(
    reviews: dict[tuple[str, str, str], AnswerReview],
    checks: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Agreement of each automatic check with *human* labels (faithful vs not)."""
    human = {k: r for k, r in reviews.items() if not r.reviewer.lower().startswith("ai-")}
    out: dict[str, Any] = {"human_reviews": len(human)}
    if not human:
        out["note"] = (
            "No human reviews yet: run `rbirag eval review-answers`. LLM-judge verdicts are "
            "not ground truth, so no agreement is reported."
        )
        return out
    signals: dict[str, Callable[[dict[str, Any]], bool | None]] = {
        "llm_judge (faithfulness = 1)": lambda c: (
            None if c.get("judge") is None else c["judge"] >= 0.999
        ),
        "lexical verifier (verdict = supported)": lambda c: (
            None if c.get("verdict") is None else c["verdict"] == "supported"
        ),
        "nli verifier (verdict = supported)": lambda c: (
            None if c.get("nli_verdict") is None else c["nli_verdict"] == "supported"
        ),
    }
    for name, fn in signals.items():
        pairs = [
            (r.faithful == "faithful", fn(checks[k]))
            for k, r in human.items()
            if k in checks and fn(checks[k]) is not None
        ]
        a = [h for h, _ in pairs]
        b = [bool(s) for _, s in pairs]
        out[name] = {
            "n": len(pairs),
            "agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(pairs)
            if pairs
            else None,
            "kappa": cohen_kappa(a, b),
        }
    return out


def load_check_rows(results_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Judge + verifier verdicts per answer, from the committed verification outputs."""
    import gzip

    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name, field in (("lexical", "verdict"), ("nli", "nli_verdict")):
        path = results_dir / "analysis" / f"citation_verification-{name}_answers.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                key = (p["qa_id"], p["config"], p["backend"])
                rows.setdefault(key, {"qa_id": p["qa_id"], "config": p["config"],
                                      "backend": p["backend"], "judge": p["judge"]})  # fmt: skip
                rows[key][field] = p["verdict"]
    return rows
