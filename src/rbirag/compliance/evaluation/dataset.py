"""The labelled clause set: schema, loading, validation, policy rendering, gold checks.

Single source of truth: ``policies.json`` (one entry per synthetic policy document, with
its sections and dev/test split) and ``labels.jsonl`` (one labelled clause per line,
including the clause's exact text). Policy documents are *rendered* from these, so the
gold clause boundaries used to score clause splitting are independent of the splitter.

Citations are groups of interchangeable notifications: a clause is fully cited when at
least one notification of every group is cited (e.g. the Directions or the amendment
that changed the paragraph).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

from rbirag.compliance.records import SUBSTANTIVE_VERDICTS, TOOL_NAMES, ToolName, Verdict
from rbirag.document import Document

Split = Literal["dev", "test"]
PolicyFormat = Literal["markdown", "plain"]
Tag = Literal[
    "numeric_threshold",  # the verdict turns on a number (days, years, amounts, rates)
    "partial_compliance",  # the clause covers the topic but leaves out a required element
    "outdated_rule",  # the clause matches an older notification that has been replaced
    "amendment",  # the rule was changed by an amendment Direction
    "future_effective",  # a relevant change is issued but not yet in force
    "entity_confusion",  # other entity types' Directions say something different
    "stricter_than_required",  # the policy goes beyond the minimum (still compliant)
    "paraphrase",  # compliant in substance, worded differently
    "prohibition",  # the rule forbids something
    "multi_source",  # needs more than one notification
    "out_of_scope",  # an internal matter no RBI regulation governs
    "not_in_corpus",  # a regulated matter the corpus has no notification on
]
TAGS: tuple[str, ...] = get_args(Tag)
DEFAULT_ROOT = Path("data/eval/compliance")
AUTHORED_BY_AI = "ai:claude"


class DatasetError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"{len(problems)} dataset problem(s):\n- " + "\n- ".join(problems))


class Section(BaseModel):
    number: str
    title: str


class Policy(BaseModel):
    policy_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    topic: str
    split: Split
    format: PolicyFormat = "markdown"
    sections: list[Section]


class Evidence(BaseModel):
    doc_id: str
    paragraph: str | None = None
    quote: str = Field(min_length=20)


class Review(BaseModel):
    reviewer: str  # "human:<name>" or "ai:<model>"
    decision: Literal["accepted", "edited", "rejected"]
    note: str = ""


class Label(BaseModel):
    item_id: str  # "<policy_id>/<number>"
    policy_id: str
    section: str  # the Section.number the clause sits under
    number: str
    text: str = Field(min_length=10)
    verdict: Verdict
    citations: list[list[str]] = Field(default_factory=list)
    acceptable: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str = Field(min_length=10)
    tags: list[Tag] = Field(default_factory=list)
    expected_tools: list[ToolName] = Field(default_factory=list)
    author: str = AUTHORED_BY_AI
    reviews: list[Review] = Field(default_factory=list)

    @field_validator("citations")
    @classmethod
    def _non_empty_groups(cls, value: list[list[str]]) -> list[list[str]]:
        if any(not group for group in value):
            raise ValueError("citation groups must not be empty")
        return value

    @property
    def gold_doc_ids(self) -> set[str]:
        return {d for group in self.citations for d in group}

    @property
    def allowed_doc_ids(self) -> set[str]:
        """Citing any of these is not a false positive."""
        return self.gold_doc_ids | set(self.acceptable)

    @property
    def human_verified(self) -> bool:
        return any(
            r.reviewer.startswith("human:") and r.decision != "rejected" for r in self.reviews
        )

    @property
    def rejected(self) -> bool:
        return any(r.decision == "rejected" for r in self.reviews)


class EvalItem(BaseModel):
    """A label plus what the evaluation needs about its policy."""

    label: Label
    split: Split
    policy_title: str
    section_title: str

    @property
    def item_id(self) -> str:
        return self.label.item_id

    @property
    def clause_text(self) -> str:
        return f"{self.label.number} {self.label.text}"


class Dataset(BaseModel):
    policies: list[Policy]
    labels: list[Label]

    def items(
        self, split: Split | None = None, *, include_rejected: bool = False
    ) -> list[EvalItem]:
        by_policy = {p.policy_id: p for p in self.policies}
        out: list[EvalItem] = []
        for label in self.labels:
            if label.rejected and not include_rejected:
                continue
            policy = by_policy[label.policy_id]
            if split is not None and policy.split != split:
                continue
            section = next(s for s in policy.sections if s.number == label.section)
            out.append(
                EvalItem(
                    label=label,
                    split=policy.split,
                    policy_title=policy.title,
                    section_title=section.title,
                )
            )
        return out

    def labels_for(self, policy_id: str) -> list[Label]:
        return [lb for lb in self.labels if lb.policy_id == policy_id]


# ----------------------------------------------------------------------------- io


def load_dataset(root: Path = DEFAULT_ROOT, *, validate: bool = True) -> Dataset:
    policies = [
        Policy.model_validate(p)
        for p in json.loads((root / "policies.json").read_text(encoding="utf-8"))
    ]
    labels: list[Label] = []
    with (root / "labels.jsonl").open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    labels.append(Label.model_validate_json(line))
                except ValueError as exc:
                    raise DatasetError([f"labels.jsonl:{lineno}: {exc}"]) from exc
    dataset = Dataset(policies=policies, labels=labels)
    if validate:
        problems = validate_dataset(dataset)
        if problems:
            raise DatasetError(problems)
    return dataset


def save_labels(labels: Iterable[Label], path: Path) -> None:
    """Rewrite ``labels.jsonl`` (same order, one JSON object per line), atomically."""
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for lb in labels:
            fh.write(lb.model_dump_json(exclude_none=False) + "\n")
    tmp.replace(path)


def validate_dataset(dataset: Dataset) -> list[str]:
    """Structural problems (no corpus needed). An empty list means the set is consistent."""
    problems: list[str] = []
    policy_ids = Counter(p.policy_id for p in dataset.policies)
    problems += [f"duplicate policy {pid!r}" for pid, n in policy_ids.items() if n > 1]
    by_policy = {p.policy_id: p for p in dataset.policies}
    item_ids = Counter(lb.item_id for lb in dataset.labels)
    problems += [f"duplicate item {iid!r}" for iid, n in item_ids.items() if n > 1]

    for lb in dataset.labels:
        where = lb.item_id
        policy = by_policy.get(lb.policy_id)
        if policy is None:
            problems.append(f"{where}: unknown policy {lb.policy_id!r}")
            continue
        if lb.item_id != f"{lb.policy_id}/{lb.number}":
            problems.append(f"{where}: item_id must be '{lb.policy_id}/{lb.number}'")
        if lb.section not in {s.number for s in policy.sections}:
            problems.append(f"{where}: unknown section {lb.section!r}")
        elif not (lb.number == lb.section or lb.number.startswith(lb.section + ".")):
            problems.append(f"{where}: number {lb.number} is not inside section {lb.section}")
        if lb.text != lb.text.strip() or lb.text.startswith(lb.number):
            problems.append(f"{where}: text must be stripped and must not repeat the number")
        if lb.verdict in SUBSTANTIVE_VERDICTS:
            if not lb.citations:
                problems.append(f"{where}: {lb.verdict} needs at least one citation group")
            if not lb.evidence:
                problems.append(f"{where}: {lb.verdict} needs evidence quotes")
        elif lb.citations:
            problems.append(f"{where}: {lb.verdict} must not have citation groups")
        if lb.verdict == "not_applicable" and "out_of_scope" not in lb.tags:
            problems.append(f"{where}: not_applicable items are tagged out_of_scope")
        if lb.verdict == "insufficient_evidence" and "not_in_corpus" not in lb.tags:
            problems.append(f"{where}: insufficient_evidence items are tagged not_in_corpus")
        for ev in lb.evidence:
            if ev.doc_id not in lb.allowed_doc_ids:
                problems.append(f"{where}: evidence from {ev.doc_id} is not a citation")
        for tool in lb.expected_tools:
            if tool not in TOOL_NAMES:
                problems.append(f"{where}: unknown tool {tool!r}")
        if len(set(lb.tags)) != len(lb.tags):
            problems.append(f"{where}: duplicate tags")
    for policy in dataset.policies:
        numbers = [lb.number for lb in dataset.labels_for(policy.policy_id)]
        if not numbers:
            problems.append(f"{policy.policy_id}: no labelled clauses")
        for s in policy.sections:
            if not any(n == s.number or n.startswith(s.number + ".") for n in numbers):
                problems.append(f"{policy.policy_id}: section {s.number} has no clauses")
    return problems


# ----------------------------------------------------------------------------- render


def render_policy(policy: Policy, labels: Iterable[Label]) -> str:
    """The policy document as a bank would write it (Markdown or plain numbered text)."""
    by_section: dict[str, list[Label]] = {s.number: [] for s in policy.sections}
    for lb in labels:
        by_section[lb.section].append(lb)
    lines: list[str] = [
        f"# {policy.title}" if policy.format == "markdown" else policy.title.upper()
    ]
    for section in policy.sections:
        heading = f"{section.number}. {section.title}"
        lines += ["", f"## {heading}" if policy.format == "markdown" else heading.upper()]
        for lb in by_section[section.number]:
            lines += ["", f"{lb.number} {lb.text}"]
    return "\n".join(lines) + "\n"


def render_all(dataset: Dataset, out_dir: Path) -> list[Path]:
    """Write ``<policy_id>.md`` for every policy; returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for policy in dataset.policies:
        path = out_dir / f"{policy.policy_id}.md"
        path.write_text(render_policy(policy, dataset.labels_for(policy.policy_id)), "utf-8")
        paths.append(path)
    return paths


def render_pdf(text: str, path: Path, *, width_chars: int = 95) -> Path:
    """A plain one-column PDF of ``text`` (wrapped), to exercise the PDF path."""
    import importlib
    import textwrap
    from typing import Any

    pymupdf: Any = importlib.import_module("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    y, line_height, bottom = 60.0, 13.0, 790.0
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width_chars) or [""]
        for line in wrapped:
            if y > bottom:
                page = doc.new_page()
                y = 60.0
            page.insert_text((50, y), line.lstrip("# "), fontsize=9.5)
            y += line_height
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    doc.close()
    return path


# ----------------------------------------------------------------------------- gold check


class GoldCheck(BaseModel):
    items: int
    documents_checked: int
    missing_documents: list[str] = Field(default_factory=list)  # "<item>: <doc_id>"
    missing_quotes: list[str] = Field(default_factory=list)  # "<item>: <doc_id>: <quote...>"

    @property
    def ok(self) -> bool:
        return not self.missing_documents and not self.missing_quotes


def _norm(text: str) -> str:
    return " ".join(text.split())


def check_gold(dataset: Dataset, documents: Mapping[str, Document]) -> GoldCheck:
    """Every cited notification exists and every evidence quote occurs in it verbatim
    (up to whitespace)."""
    missing_docs: list[str] = []
    missing_quotes: list[str] = []
    normalised: dict[str, str] = {}
    for lb in dataset.labels:
        for doc_id in sorted(lb.allowed_doc_ids):
            if doc_id not in documents:
                missing_docs.append(f"{lb.item_id}: {doc_id}")
        for ev in lb.evidence:
            doc = documents.get(ev.doc_id)
            if doc is None:
                continue
            if ev.doc_id not in normalised:
                normalised[ev.doc_id] = _norm(doc.text)
            if _norm(ev.quote) not in normalised[ev.doc_id]:
                missing_quotes.append(f"{lb.item_id}: {ev.doc_id}: {ev.quote[:80]}")
    return GoldCheck(
        items=len(dataset.labels),
        documents_checked=len(normalised),
        missing_documents=missing_docs,
        missing_quotes=missing_quotes,
    )


# ----------------------------------------------------------------------------- stats


def dataset_stats(dataset: Dataset) -> dict[str, object]:
    items = dataset.items(include_rejected=False)
    by_split: dict[str, Counter[str]] = {"dev": Counter(), "test": Counter()}
    for it in items:
        by_split[it.split][it.label.verdict] += 1
    tags: Counter[str] = Counter(t for it in items for t in it.label.tags)
    return {
        "policies": len(dataset.policies),
        "items": len(items),
        "rejected": sum(1 for lb in dataset.labels if lb.rejected),
        "human_verified": sum(1 for it in items if it.label.human_verified),
        "by_split": {k: dict(sorted(v.items())) for k, v in by_split.items()},
        "by_verdict": dict(sorted(Counter(it.label.verdict for it in items).items())),
        "tags": dict(sorted(tags.items())),
        "policies_by_split": {
            s: [p.policy_id for p in dataset.policies if p.split == s] for s in ("dev", "test")
        },
        "multi_group_items": sum(1 for it in items if len(it.label.citations) > 1),
    }
