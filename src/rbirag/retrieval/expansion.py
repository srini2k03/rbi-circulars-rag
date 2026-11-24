"""Query expansion for the BM25 arm: RBI acronyms (mined from the corpus) and RM3.

* **Acronyms.** RBI text defines most abbreviations inline ("Know Your Customer (KYC)").
  :func:`mine_acronyms` collects these definitions from the corpus; a query that uses one
  form gets the other form added at a reduced weight ("KYC norms" also searches for
  "know your customer"). Only unambiguous, repeatedly defined acronyms are kept.
* **RM3** (Lavrenko & Croft's relevance model, as in Anserini): the top ``fb_docs`` BM25
  hits are treated as relevant, the ``fb_terms`` most probable terms of their language
  model are added, and the result is interpolated with the original query.

Expansion never changes what the user asked: the expanded terms are returned in
:class:`ExpandedQuery` and shown next to the results.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from rbirag.index.bm25 import BM25Index, bm25_tokenize

ACRONYMS_FILE = "acronyms.json"

# "Long Form Words (ACR)" -- up to 9 words before a parenthesised capitalised token.
_DEFINITION_RE = re.compile(
    r"\b((?:[A-Za-z][A-Za-z\-&']*\s+){1,8}?[A-Za-z][A-Za-z\-&']*)\s*\(\s*([A-Z][A-Za-z]{1,9})\s*\)"
)
_SMALL_WORDS = frozenset({"of", "and", "the", "for", "in", "on", "to", "a", "an", "&", "by"})


@dataclass(frozen=True)
class AcronymEntry:
    acronym: str
    expansion: str
    count: int  # how often this definition occurs in the corpus
    share: float  # its share of all definitions of the acronym
    # whether a lower-case spelling in a query may be read as the acronym: false when the
    # lower-case word itself is common in the corpus (e.g. "pan", "arc")
    lower_ok: bool = True


def _initials(words: Sequence[str], *, skip_small: bool) -> str:
    return "".join(w[0].upper() for w in words if not (skip_small and w.lower() in _SMALL_WORDS))


def _match_definition(long_form: str, acronym: str) -> str | None:
    """The shortest suffix of ``long_form`` whose initials spell ``acronym``."""
    words = [w for w in re.split(r"[\s\-]+", long_form.strip()) if w]
    for i in range(len(words) - 1, -1, -1):
        cand = words[i:]
        if cand[0].lower() in _SMALL_WORDS:
            continue
        if acronym in (_initials(cand, skip_small=True), _initials(cand, skip_small=False)):
            return " ".join(cand).lower()
    return None


def mine_acronyms(
    texts: Sequence[str], *, min_count: int = 3, min_share: float = 0.6, min_length: int = 3
) -> list[AcronymEntry]:
    """Acronym definitions found in ``texts``, keeping an acronym only if it has at least
    ``min_length`` letters, its most common expansion occurs ``min_count`` times and
    accounts for ``min_share`` of its definitions (so "IT", "SC" or "DR" are dropped)."""
    counts: Counter[tuple[str, str]] = Counter()
    for text in texts:
        for m in _DEFINITION_RE.finditer(text):
            acronym = m.group(2)
            if acronym.endswith("s") and acronym[:-1].isupper():
                acronym = acronym[:-1]  # plural: "NBFCs"
            if not acronym.isupper() or len(acronym) < min_length:
                continue
            expansion = _match_definition(m.group(1), acronym)
            if expansion:
                counts[(acronym, expansion)] += 1
    by_acronym: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (acronym, expansion), n in counts.items():
        by_acronym[acronym].append((n, expansion))
    out: list[AcronymEntry] = []
    for acronym, options in by_acronym.items():
        options.sort(key=lambda x: (-x[0], x[1]))
        total = sum(n for n, _ in options)
        n, expansion = options[0]
        if n >= min_count and n / total >= min_share:
            out.append(AcronymEntry(acronym, expansion, n, round(n / total, 3)))
    lower_words: Counter[str] = Counter()
    wanted = {e.acronym.lower() for e in out}
    for text in texts:
        lower_words.update(w for w in re.findall(r"\b[a-z]{3,10}\b", text) if w in wanted)
    out = [
        AcronymEntry(e.acronym, e.expansion, e.count, e.share, lower_words[e.acronym.lower()] < 5)
        for e in out
    ]
    return sorted(out, key=lambda e: (-e.count, e.acronym))


def save_acronyms(entries: Sequence[AcronymEntry], path: Path) -> None:
    payload = [e.__dict__ for e in entries]
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_acronyms(path: Path | None = None) -> list[AcronymEntry]:
    """The committed dictionary (``rbirag/retrieval/acronyms.json``) unless ``path``."""
    if path is not None:
        raw = path.read_text(encoding="utf-8")
    else:
        raw = resources.files("rbirag.retrieval").joinpath(ACRONYMS_FILE).read_text("utf-8")
    return [AcronymEntry(**item) for item in json.loads(raw)]


@dataclass
class ExpandedQuery:
    """Weighted BM25 terms plus a record of what was added and why."""

    terms: list[tuple[str, float]]
    original: list[str]
    added: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.added:
            return "no expansion"
        top = sorted(self.added.items(), key=lambda kv: -kv[1])[:8]
        return "; ".join(self.notes) + " | added: " + ", ".join(f"{t}({w:.2f})" for t, w in top)


class AcronymExpander:
    """Adds the other form of every RBI acronym the query uses (acronym <-> expansion)."""

    def __init__(self, entries: Sequence[AcronymEntry]) -> None:
        self.entries = list(entries)
        self._by_acronym = {e.acronym.lower(): e for e in self.entries}
        self._phrases = [(bm25_tokenize(e.expansion), e) for e in self.entries]

    @classmethod
    def default(cls) -> AcronymExpander:
        return cls(load_acronyms())

    def expand(self, query: str, *, weight: float) -> ExpandedQuery:
        original = bm25_tokenize(query)
        terms = [(t, 1.0) for t in original]
        added: dict[str, float] = {}
        notes: list[str] = []
        present = set(original)
        # acronym in the query (any case) -> add the expansion's tokens
        for raw in re.findall(r"[A-Za-z]{3,11}", query):
            key = raw.lower()
            entry = self._by_acronym.get(key)
            if entry is None and key.endswith("s"):
                entry = self._by_acronym.get(key[:-1])  # plural: "NBFCs"
            if entry is None or f"{entry.acronym} = {entry.expansion}" in notes:
                continue
            upper = raw.isupper() or (raw.endswith("s") and raw[:-1].isupper())
            if not upper and not entry.lower_ok:
                continue  # "pan" in lower case is a word, not PAN
            notes.append(f"{entry.acronym} = {entry.expansion}")
            for tok in bm25_tokenize(entry.expansion):
                if tok not in present and tok not in added:
                    added[tok] = weight
        # expansion phrase in the query -> add the acronym token
        for phrase, entry in self._phrases:
            if phrase and _contains(original, phrase):
                tok = entry.acronym.lower()
                if tok not in present and tok not in added:
                    added[tok] = weight
                    notes.append(f"{entry.expansion} = {entry.acronym}")
        terms.extend(added.items())
        return ExpandedQuery(terms=terms, original=original, added=added, notes=notes)


def _contains(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    n = len(phrase)
    return any(list(tokens[i : i + n]) == list(phrase) for i in range(len(tokens) - n + 1))


def rm3(
    index: BM25Index,
    query_terms: Sequence[tuple[str, float]],
    *,
    fb_docs: int,
    fb_terms: int,
    orig_weight: float,
    feedback: Sequence[tuple[str, float]] | None = None,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """RM3 expansion of a weighted query.

    ``P(w|R) ∝ sum_d P(w|d) * P(d|q)`` over the top ``fb_docs`` hits (``P(d|q)`` = BM25
    score normalised over the feedback set, ``P(w|d)`` = term frequency / length), keep
    the ``fb_terms`` best terms, renormalise, and interpolate with the original query
    model: ``P'(w) = orig_weight * P(w|q) + (1 - orig_weight) * P(w|R)``. Returns the new
    weighted terms and the weights of the terms that RM3 added.
    """
    hits = list(feedback) if feedback is not None else index.search_terms(query_terms, fb_docs)
    hits = hits[:fb_docs]
    q_total = sum(w for _, w in query_terms)
    if not hits or q_total <= 0:
        return list(query_terms), {}
    q_model: dict[str, float] = defaultdict(float)
    for t, w in query_terms:
        q_model[t] += w / q_total
    score_total = sum(s for _, s in hits)
    rel: dict[str, float] = defaultdict(float)
    for cid, score in hits:
        tokens = index.tokens_of(cid)
        if not tokens:
            continue
        doc_weight = score / score_total if score_total > 0 else 1.0 / len(hits)
        for tok, tf in Counter(tokens).items():
            if len(tok) < 2 or tok.isdigit():
                continue
            rel[tok] += doc_weight * tf / len(tokens)
    best = sorted(rel.items(), key=lambda kv: (-kv[1], kv[0]))[:fb_terms]
    rel_total = sum(v for _, v in best)
    if rel_total <= 0:
        return list(query_terms), {}
    merged: dict[str, float] = {t: orig_weight * w for t, w in q_model.items()}
    added: dict[str, float] = {}
    for t, v in best:
        contribution = (1 - orig_weight) * v / rel_total
        if t not in merged:
            added[t] = contribution
        merged[t] = merged.get(t, 0.0) + contribution
    return sorted(merged.items(), key=lambda kv: (-kv[1], kv[0])), added


def expand_query(
    query: str,
    index: BM25Index,
    *,
    expansion: str,
    acronyms: AcronymExpander | None,
    acronym_weight: float,
    fb_docs: int,
    fb_terms: int,
    orig_weight: float,
) -> ExpandedQuery:
    """Apply the configured expansion (``none`` | ``acronyms`` | ``rm3`` | ``acronyms+rm3``)."""
    if expansion in {"acronyms", "acronyms+rm3"}:
        if acronyms is None:
            raise ValueError("acronym expansion needs an AcronymExpander")
        eq = acronyms.expand(query, weight=acronym_weight)
    else:
        original = bm25_tokenize(query)
        eq = ExpandedQuery(terms=[(t, 1.0) for t in original], original=original)
    if expansion in {"rm3", "acronyms+rm3"}:
        terms, added = rm3(
            index, eq.terms, fb_docs=fb_docs, fb_terms=fb_terms, orig_weight=orig_weight
        )
        eq = ExpandedQuery(
            terms=terms,
            original=eq.original,
            added={**eq.added, **added},
            notes=[*eq.notes, f"RM3 ({fb_docs} feedback passages, {fb_terms} terms)"],
        )
    return eq


def term_weights(terms: Iterable[tuple[str, float]]) -> Mapping[str, float]:
    out: dict[str, float] = defaultdict(float)
    for t, w in terms:
        out[t] += w
    return dict(out)
