"""Candidate Q/A generation from stratified chunk samples (+ an unanswerable set).

Sampling is stratified by (year, department) with proportional allocation (at least one
per stratum, largest-remainder rounding); question types (factual, numeric, procedural,
multi-chunk) are assigned round-robin so each type is spread across strata. Every pair
carries a verbatim evidence quote that is located in the source document to give a
chunker-independent gold span. Pairs failing the automatic checks are kept but flagged
(``auto_check=False``) for the human reviewer.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rbirag.chunking.base import Chunk
from rbirag.eval.dataset import GoldSpan, QAPair, QuestionType
from rbirag.eval.judge import JudgeParseError, extract_json_object
from rbirag.generation.llm import (
    LLM,
    ConfigurationError,
    LLMError,
    Message,
    QuotaExhaustedError,
)

logger = logging.getLogger(__name__)

ANSWERABLE_TYPES: tuple[QuestionType, ...] = ("factual", "numeric", "procedural", "multi_chunk")

QA_SYSTEM = """You write evaluation questions for a question-answering system over Reserve \
Bank of India (RBI) notifications. Questions must be answerable from the given passage \
alone, and must make sense to someone who has not seen the passage."""

TYPE_GUIDANCE: dict[str, str] = {
    "factual": "Ask about a specific fact stated in the passage (who, what, which entities, "
    "which rule applies).",
    "numeric": "Ask about a specific number, amount, percentage, limit, period or date stated "
    "in the passage; the answer must contain that number.",
    "procedural": "Ask how something must be done, by when, or what steps/conditions apply, "
    "as stated in the passage.",
    "multi_chunk": "The passage has two consecutive parts. Ask a question whose answer needs "
    "information from BOTH parts.",
}

QA_INSTRUCTIONS = """Write ONE {qtype} question.
{guidance}

Rules:
- Make the question self-contained: name the subject (e.g. the scheme, the entity type, \
the direction) instead of saying "this circular" or "the passage".
- Do not copy long phrases from the passage into the question.
- The answer must be short (one to three sentences) and fully supported by the passage.
- "evidence" must be an exact, contiguous quote from the passage (at most 300 \
characters) that supports the answer. For a two-part passage give the quote from part 1.

Return only JSON: {{"question": str, "answer": str, "evidence": str}}

Notification: {title} ({date}{circ})

PASSAGE:
{passage}"""

QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["question", "answer", "evidence"],
    "additionalProperties": False,
}

UNANSWERABLE_INSTRUCTIONS = """Write {n} questions that sound like realistic questions a \
banker or compliance officer might ask a system that answers from RBI notifications, but \
that RBI notifications cannot answer. Mix these kinds:
- rules of other regulators (SEBI, IRDAI, PFRDA, foreign central banks);
- tax, company-law or accounting questions (income tax rates, GST, Ind AS details);
- questions about specific private companies' internal policies or market prices;
- questions about RBI circulars with invented, very specific numbers or dates outside \
{date_range}.
Keep each question specific and self-contained. Do not mention that it is unanswerable.

Return only JSON: {{"questions": [str, ...]}}"""

UNANSWERABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}

_DEICTIC = re.compile(r"\b(this|the) (passage|circular|notification|document|text)\b", re.I)


@dataclass(frozen=True)
class Sample:
    chunks: tuple[Chunk, ...]
    qtype: QuestionType

    @property
    def key(self) -> str:
        return "+".join(c.chunk_id for c in self.chunks) + f"|{self.qtype}"


def _stratum(chunk: Chunk) -> tuple[str, str]:
    md = chunk.metadata
    return (str(md.get("year") or "unknown"), str(md.get("department") or "unknown"))


def eligible(chunk: Chunk, min_tokens: int = 60) -> bool:
    if chunk.n_tokens < min_tokens:
        return False
    sentences = re.findall(r"[.;:](?:\s|$)", chunk.text)
    return len(sentences) >= 2


def allocate(sizes: dict[Any, int], n: int) -> dict[Any, int]:
    """Proportional allocation (largest remainder), >= 1 per non-empty stratum when ``n``
    allows, never more than a stratum holds."""
    strata = [k for k, v in sizes.items() if v > 0]
    if not strata or n <= 0:
        return dict.fromkeys(sizes, 0)
    n = min(n, sum(sizes[k] for k in strata))
    total = sum(sizes[k] for k in strata)
    quotas = {k: n * sizes[k] / total for k in strata}
    floor_min = 1 if n >= len(strata) else 0
    alloc = {k: min(sizes[k], max(floor_min, int(quotas[k]))) for k in strata}

    def remainder(k: Any) -> float:
        return quotas[k] - alloc[k]

    while sum(alloc.values()) > n:  # too many after the minimums: trim the most over-served
        k = min((k for k in strata if alloc[k] > floor_min), key=lambda k: (remainder(k), str(k)))
        alloc[k] -= 1
    while sum(alloc.values()) < n:  # hand out the rest by largest remainder, then size
        k = max(
            (k for k in strata if alloc[k] < sizes[k]),
            key=lambda k: (remainder(k), sizes[k], str(k)),
        )
        alloc[k] += 1
    return {k: alloc.get(k, 0) for k in sizes}


def stratified_samples(chunks: Sequence[Chunk], n: int, seed: int = 13) -> list[Sample]:
    rng = random.Random(seed)
    pool = [c for c in chunks if eligible(c)]
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: c.index)
    strata: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for c in pool:
        strata[_stratum(c)].append(c)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)

    picked: list[Chunk] = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda c: c.chunk_id)
        rng.shuffle(members)
        # at most one chunk per document within a stratum, for diversity
        seen_docs: set[str] = set()
        chosen: list[Chunk] = []
        for c in members:
            if len(chosen) >= alloc[key]:
                break
            if c.doc_id in seen_docs:
                continue
            seen_docs.add(c.doc_id)
            chosen.append(c)
        for c in members:  # top up if documents ran out
            if len(chosen) >= alloc[key]:
                break
            if c not in chosen:
                chosen.append(c)
        picked.extend(chosen)
    rng.shuffle(picked)

    samples: list[Sample] = []
    for i, c in enumerate(picked):
        qtype = ANSWERABLE_TYPES[i % len(ANSWERABLE_TYPES)]
        if qtype == "numeric" and not re.search(r"\d", c.text):
            qtype = "factual"
        if qtype == "multi_chunk":
            siblings = by_doc[c.doc_id]
            pos = next(j for j, s in enumerate(siblings) if s.chunk_id == c.chunk_id)
            if pos + 1 < len(siblings) and siblings[pos + 1].n_tokens >= 30:
                samples.append(Sample((c, siblings[pos + 1]), "multi_chunk"))
                continue
            qtype = "factual"
        samples.append(Sample((c,), qtype))
    return samples


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def locate(evidence: str, chunk: Chunk) -> GoldSpan | None:
    """Character span of ``evidence`` in the document, via the chunk's offsets."""
    ev = evidence.strip().strip('"').strip()
    if len(ev) < 10:
        return None
    idx = chunk.text.find(ev)
    if idx >= 0:
        return GoldSpan(
            doc_id=chunk.doc_id, start=chunk.start + idx, end=chunk.start + idx + len(ev)
        )
    # whitespace/case-insensitive match: map normalised positions back to the original
    words = ev.split()
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, chunk.text, re.IGNORECASE)
    if m:
        return GoldSpan(
            doc_id=chunk.doc_id, start=chunk.start + m.start(), end=chunk.start + m.end()
        )
    return None


def qa_id_for(key: str) -> str:
    return "q" + hashlib.sha256(key.encode()).hexdigest()[:12]


def build_prompt(sample: Sample) -> str:
    first = sample.chunks[0]
    md = first.metadata
    if len(sample.chunks) == 1:
        passage = first.text
    else:
        passage = "\n\n".join(f"PART {i}:\n{c.text}" for i, c in enumerate(sample.chunks, 1))
    circ = f", {md['circular_number']}" if md.get("circular_number") else ""
    return QA_INSTRUCTIONS.format(
        qtype=sample.qtype.replace("_", "-"),
        guidance=TYPE_GUIDANCE[sample.qtype],
        title=md.get("title") or "Untitled",
        date=md.get("date") or "undated",
        circ=circ,
        passage=passage,
    )


def generate_pair(llm: LLM, sample: Sample) -> QAPair | None:
    response = llm.generate(
        [Message(role="user", content=build_prompt(sample))],
        system=QA_SYSTEM,
        max_tokens=1024,
        json_schema=QA_SCHEMA,
    )
    if response.refused or not response.text.strip():
        return None
    try:
        obj = extract_json_object(response.text)
    except JudgeParseError:
        logger.warning("unparseable QA output for %s", sample.key)
        return None
    question = str(obj.get("question", "")).strip()
    answer = str(obj.get("answer", "")).strip()
    evidence = str(obj.get("evidence", "")).strip()
    first = sample.chunks[0]
    gold: list[GoldSpan] = []
    located = None
    for c in sample.chunks:
        located = located or locate(evidence, c)
    if located is not None:
        gold.append(located)
    # multi-chunk questions also need the second part
    if sample.qtype == "multi_chunk":
        for c in sample.chunks:
            if (
                located is None
                or c.doc_id != located.doc_id
                or not (c.start <= located.start < c.end)
            ):
                gold.append(GoldSpan(doc_id=c.doc_id, start=c.start, end=c.end))
    auto_ok = (
        located is not None
        and len(question.split()) >= 5
        and bool(answer)
        and not _DEICTIC.search(question)
    )
    if not gold:
        gold = [GoldSpan(doc_id=c.doc_id, start=c.start, end=c.end) for c in sample.chunks]
    md = first.metadata
    year = md.get("year")
    return QAPair(
        qa_id=qa_id_for(sample.key),
        question=question,
        answer=answer,
        qtype=sample.qtype,
        answerable=True,
        gold=gold,
        evidence=evidence,
        source_chunk_ids=[c.chunk_id for c in sample.chunks],
        source_text="\n\n".join(c.text for c in sample.chunks),
        doc_title=str(md.get("title")) if md.get("title") else None,
        year=int(year) if isinstance(year, int) else None,
        department=str(md.get("department")) if md.get("department") else None,
        generator_model=response.model,
        auto_check=auto_ok,
    )


def generate_unanswerable(llm: LLM, n: int, *, date_range: str, batch: int = 10) -> list[QAPair]:
    out: list[QAPair] = []
    seen: set[str] = set()
    round_ = 0
    while len(out) < n and round_ < max(3, n // batch * 2):
        k = min(batch, n - len(out))
        prompt = UNANSWERABLE_INSTRUCTIONS.format(n=k, date_range=date_range)
        if round_:
            prompt += f"\n\n(Batch {round_ + 1}: avoid repeating earlier topics.)"
        round_ += 1
        response = llm.generate(
            [Message(role="user", content=prompt)],
            system=QA_SYSTEM,
            max_tokens=2048,
            json_schema=UNANSWERABLE_SCHEMA,
        )
        try:
            questions = extract_json_object(response.text).get("questions", [])
        except JudgeParseError:
            continue
        for q in questions:
            q = str(q).strip()
            if len(q.split()) < 5 or _norm(q) in seen:
                continue
            seen.add(_norm(q))
            out.append(
                QAPair(
                    qa_id=qa_id_for("unanswerable|" + _norm(q)),
                    question=q,
                    answer="",
                    qtype="unanswerable",
                    answerable=False,
                    generator_model=response.model,
                    auto_check=True,
                )
            )
            if len(out) >= n:
                break
    return out


@dataclass
class GenerationRun:
    pairs: list[QAPair]
    complete: bool
    attempted: int
    stopped_because: str | None = None


def generate_candidates(
    llm: LLM,
    chunks: Sequence[Chunk],
    *,
    n_answerable: int,
    n_unanswerable: int,
    seed: int = 13,
    existing: Sequence[QAPair] = (),
) -> GenerationRun:
    """Generate candidates; stops cleanly (``complete=False``) when the quota runs out."""
    done = {p.qa_id: p for p in existing}
    samples = stratified_samples(chunks, n_answerable, seed)
    attempted = 0
    complete = True
    stopped: str | None = None
    try:
        for sample in samples:
            qid = qa_id_for(sample.key)
            if qid in done:
                continue
            attempted += 1
            pair = generate_pair(llm, sample)
            if pair is not None:
                done[pair.qa_id] = pair
        n_existing_unans = sum(1 for p in done.values() if not p.answerable)
        if n_existing_unans < n_unanswerable:
            years = sorted(int(y) for c in chunks if isinstance((y := c.metadata.get("year")), int))
            date_range = f"{years[0]}-{years[-1]}" if years else "the corpus years"
            for p in generate_unanswerable(
                llm, n_unanswerable - n_existing_unans, date_range=date_range
            ):
                done.setdefault(p.qa_id, p)
    except ConfigurationError:
        raise
    except (QuotaExhaustedError, LLMError) as exc:
        logger.warning("stopping early: %s", exc)
        complete = False
        stopped = type(exc).__name__
    return GenerationRun(
        pairs=list(done.values()), complete=complete, attempted=attempted, stopped_because=stopped
    )
