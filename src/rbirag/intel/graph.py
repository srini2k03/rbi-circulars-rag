"""The relationship graph over the indexed corpus, and the questions it answers.

Edges are stored in the direction of *effect*: ``A -amends-> B`` means A changed B, whether
the reference was written in A ("... is amended as under") or in B (a consolidated
direction's footnote "Amended vide circular A"). Unresolved references stay as unknown
targets and are reported, not guessed.

Every answer about "the latest version" is scoped to the indexed corpus: the graph knows
only the documents that were crawled, so it can say which document in the corpus is the
newest in a chain, never what the regulatory position is today.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from rbirag.chunking.base import split_sentences
from rbirag.document import Document
from rbirag.intel.extract import DocIntel, Relation

EdgeType = Literal["supersedes", "partially_modifies", "amends", "consolidates", "refers_to"]
CHANGE_TYPES = frozenset({"supersedes", "partially_modifies", "amends"})
VERSIONED = frozenset({"master_direction", "master_circular", "directions", "regulations"})


def disclaimer(corpus_size: int, newest: dt.date | None) -> str:
    snapshot = newest.isoformat() if newest else "unknown"
    return (
        f"Based only on the {corpus_size} RBI notifications in the indexed corpus (newest "
        f"dated {snapshot}). This is the latest document in the indexed corpus, not a verified "
        "statement of the current regulatory position: RBI may have issued later changes that "
        "are not indexed. No legal effect is implied."
    )


class Edge(BaseModel):
    actor: str  # the document that acts (amends, supersedes, refers to)
    affected: str
    type: EdgeType
    confidence: str
    evidence: str  # the sentence that states it
    found_in: str  # the document the sentence is in


class LatestInCorpus(BaseModel):
    doc_id: str
    latest: str  # newest document of the chain in the indexed corpus
    path: list[str]  # doc_id -> ... -> latest
    how: list[str]  # "newer version" / "superseded by" for each step
    amended_by: list[str]  # amendments to the latest document found in the corpus
    is_latest: bool
    disclaimer: str


class DocDiff(BaseModel):
    a: str
    b: str
    similarity: float  # difflib ratio over sentences
    removed: list[str] = Field(default_factory=list)  # only in a
    added: list[str] = Field(default_factory=list)  # only in b
    changed: list[tuple[str, str]] = Field(default_factory=list)  # (in a, in b), similar
    numbers_removed: list[str] = Field(default_factory=list)
    numbers_added: list[str] = Field(default_factory=list)


class RelationGraph:
    def __init__(self, intel: dict[str, DocIntel], relations: Sequence[Relation]) -> None:
        self.intel = intel
        self.relations = list(relations)
        self.out: dict[str, list[Edge]] = defaultdict(list)
        self.inc: dict[str, list[Edge]] = defaultdict(list)
        seen: set[tuple[str, str, str]] = set()
        for r in relations:
            if r.cited is None or r.cited == r.citing:
                continue
            if r.type == "amended_by":
                actor, affected, etype = r.cited, r.citing, "amends"
            else:
                actor, affected, etype = r.citing, r.cited, r.type
            key = (actor, affected, etype)
            if key in seen:
                continue
            seen.add(key)
            edge = Edge(
                actor=actor, affected=affected, type=etype, confidence=r.confidence,  # type: ignore[arg-type]
                evidence=r.sentence, found_in=r.citing,
            )  # fmt: skip
            self.out[actor].append(edge)
            self.inc[affected].append(edge)
        self.families: dict[str, list[str]] = defaultdict(list)
        for info in intel.values():
            if info.category in VERSIONED and len(info.family.split()) >= 3:
                self.families[info.family].append(info.doc_id)
        dates = [i.publication_date for i in intel.values() if i.publication_date]
        self.newest = max(dates) if dates else None

    # -- basic views -----------------------------------------------------------------
    @property
    def edges(self) -> list[Edge]:
        return [e for edges in self.out.values() for e in edges]

    def _date(self, doc_id: str) -> dt.date:
        return self.intel[doc_id].publication_date or dt.date.min

    def changed_by(self, doc_id: str) -> list[Edge]:
        """Documents in the corpus that amend, partially modify or supersede ``doc_id``."""
        edges = [e for e in self.inc.get(doc_id, []) if e.type in CHANGE_TYPES]
        return sorted(edges, key=lambda e: (self._date(e.actor), e.actor))

    def changes(self, doc_id: str) -> list[Edge]:
        """Documents that ``doc_id`` amends, partially modifies or supersedes."""
        return [e for e in self.out.get(doc_id, []) if e.type in CHANGE_TYPES]

    def consolidated_into(self, doc_id: str) -> list[str]:
        """Master circulars / directions in the corpus that consolidate ``doc_id``."""
        return [e.actor for e in self.inc.get(doc_id, []) if e.type == "consolidates"]

    def references(self, doc_id: str) -> list[Edge]:
        return [e for e in self.out.get(doc_id, []) if e.type == "refers_to"]

    def cited_by(self, doc_id: str) -> list[Edge]:
        return [e for e in self.inc.get(doc_id, []) if e.type == "refers_to"]

    def unresolved(self, doc_id: str) -> list[Relation]:
        """References in ``doc_id`` that point outside the corpus or are ambiguous."""
        return [r for r in self.relations if r.citing == doc_id and r.cited is None]

    def versions(self, doc_id: str) -> list[str]:
        """Versions of the same direction/circular in the corpus (same title family),
        oldest first; just ``[doc_id]`` when the document is not versioned."""
        info = self.intel[doc_id]
        group = self.families.get(info.family) if info.category in VERSIONED else None
        if not group or doc_id not in group:
            return [doc_id]
        return sorted(group, key=lambda d: (self._date(d), d))

    def original(self, doc_id: str) -> str:
        """The earliest version of ``doc_id`` in the corpus."""
        return self.versions(doc_id)[0]

    # -- latest in the indexed corpus ------------------------------------------------
    def latest_in_corpus(self, doc_id: str) -> LatestInCorpus:
        """Follow newer versions and ``supersedes`` edges to the newest document. Cycles
        (two documents citing each other) stop the walk."""
        path, how = [doc_id], []
        current = doc_id
        while True:
            nxt: tuple[str, str] | None = None
            newer = [v for v in self.versions(current) if self._date(v) > self._date(current)]
            if newer:
                nxt = (newer[-1], "newer version in the corpus")
            sup = [
                e.actor for e in self.inc.get(current, [])
                if e.type == "supersedes" and self._date(e.actor) >= self._date(current)
            ]  # fmt: skip
            if sup:
                best = max(sup, key=lambda d: (self._date(d), d))
                if nxt is None or self._date(best) > self._date(nxt[0]):
                    nxt = (best, "superseded by")
            if nxt is None or nxt[0] in path:
                break
            path.append(nxt[0])
            how.append(nxt[1])
            current = nxt[0]
        amendments = [
            e.actor for e in self.changed_by(current)
            if e.type in {"amends", "partially_modifies"}
        ]  # fmt: skip
        return LatestInCorpus(
            doc_id=doc_id, latest=current, path=path, how=how,
            amended_by=list(dict.fromkeys(amendments)), is_latest=current == doc_id,
            disclaimer=disclaimer(len(self.intel), self.newest),
        )  # fmt: skip

    def related(self, doc_id: str, limit: int = 10) -> list[tuple[str, str]]:
        """``(doc_id, why)`` neighbours: versions, documents changing or changed by it,
        then plain references in either direction."""
        out: dict[str, str] = {}
        for v in self.versions(doc_id):
            if v != doc_id:
                out.setdefault(v, "another version")
        for e in self.changed_by(doc_id):
            out.setdefault(e.actor, f"{e.type.replace('_', ' ')} this document")
        for e in self.changes(doc_id):
            out.setdefault(e.affected, f"{e.type.replace('_', ' ')} by this document")
        for actor in self.consolidated_into(doc_id):
            out.setdefault(actor, "consolidates this document")
        for e in self.references(doc_id):
            out.setdefault(e.affected, "referred to by this document")
        for e in self.cited_by(doc_id):
            out.setdefault(e.actor, "refers to this document")
        return list(out.items())[:limit]

    def stats(self) -> dict[str, Any]:
        types = Counter(e.type for e in self.edges)
        resolved = [r for r in self.relations if r.cited]
        return {
            "documents": len(self.intel),
            "references": len(self.relations),
            "references_resolved": len(resolved),
            "references_unknown": len(self.relations) - len(resolved),
            "resolution": dict(Counter(r.resolution for r in self.relations)),
            "edges": len(self.edges),
            "edges_by_type": dict(types),
            "documents_with_changes": len(
                {e.affected for e in self.edges if e.type in CHANGE_TYPES}
            ),
            "version_families": sum(1 for g in self.families.values() if len(g) > 1),
            "categories": dict(Counter(i.category for i in self.intel.values())),
            "effective_date_kinds": dict(Counter(i.effective.kind for i in self.intel.values())),
        }

    def export(self) -> dict[str, Any]:
        """Compact JSON for the static site and the API (no full texts)."""
        return {
            "newest": self.newest.isoformat() if self.newest else None,
            "disclaimer": disclaimer(len(self.intel), self.newest),
            "docs": {
                d: {
                    "t": i.subject, "c": i.category, "d": i.publication_date.isoformat()
                    if i.publication_date else None, "n": i.circular_number, "r": i.ref_number,
                    "e": i.effective.date.isoformat() if i.effective.date else None,
                    "ek": i.effective.kind, "dep": i.department, "v": i.version_note,
                }
                for d, i in self.intel.items()
            },
            "edges": [
                [e.actor, e.affected, e.type, e.confidence, e.evidence[:240]] for e in self.edges
            ],
            "families": {f: ids for f, ids in self.families.items() if len(ids) > 1},
        }  # fmt: skip


# ----------------------------------------------------------------------------- diff

_NUM_RE = re.compile(
    r"(?:Rs\.?|₹|INR)?\s?\d[\d,]*(?:\.\d+)?\s?(?:%|per cent|lakh|crore|days|months|years)?"
)


def _sentences(text: str) -> list[str]:
    return [" ".join(text[a:b].split()) for a, b in split_sentences(text)]


def compare_documents(a: Document, b: Document, *, limit: int = 40) -> DocDiff:
    """What changed from ``a`` to ``b``: sentences only in one of them, similar sentences
    that were reworded, and the figures that appear or disappear. Text comparison only;
    it does not interpret the legal effect of a change."""
    sa, sb = _sentences(a.text), _sentences(b.text)
    matcher = difflib.SequenceMatcher(a=sa, b=sb, autojunk=False)
    diff = DocDiff(a=a.doc_id, b=b.doc_id, similarity=round(matcher.ratio(), 4))
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old, new = sa[i1:i2], sb[j1:j2]
        if tag == "replace":
            for x, y in zip(old, new, strict=False):
                if difflib.SequenceMatcher(a=x, b=y).ratio() >= 0.6:
                    diff.changed.append((x, y))
                else:
                    diff.removed.append(x)
                    diff.added.append(y)
            diff.removed += old[len(new) :]
            diff.added += new[len(old) :]
        elif tag == "delete":
            diff.removed += old
        else:
            diff.added += new
    nums_a = {m.group(0).strip() for s in diff.removed + [x for x, _ in diff.changed]
              for m in _NUM_RE.finditer(s)}  # fmt: skip
    nums_b = {m.group(0).strip() for s in diff.added + [y for _, y in diff.changed]
              for m in _NUM_RE.finditer(s)}  # fmt: skip
    diff.numbers_removed = sorted(n for n in nums_a - nums_b if len(n) > 1)[:limit]
    diff.numbers_added = sorted(n for n in nums_b - nums_a if len(n) > 1)[:limit]
    diff.removed, diff.added, diff.changed = (
        diff.removed[:limit],
        diff.added[:limit],
        diff.changed[:limit],
    )
    return diff
