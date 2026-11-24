"""Evidence units for entailment checks: short premise windows and atomic claims.

NLI cross-encoders are trained on single-sentence premises (MNLI/SNLI) and truncate the
input at 512 word pieces. Checking a one-line claim against a whole 1,024-token passage
therefore fails twice: the evidence may be cut off, and even when it fits, a long premise
full of unrelated text dilutes the entailment probability. Checking each claim against
short overlapping windows of a passage (and taking the best window) avoids both, and also
says *where* the support is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rbirag.chunking.base import split_sentences

WINDOW_SENTENCES = 3  # sentences per premise window
WINDOW_STRIDE = 2  # consecutive windows overlap by one sentence
WINDOW_MAX_WORDS = 120  # longer windows are cut (word boundary) to stay far below 512 pieces
_WORD = re.compile(r"\S+")
_CLAIM_SPLIT = re.compile(
    r";\s+|,\s+(?:and|but|while|whereas)\s+|\s+(?:and|but)\s+(?=(?:the|it|this|they|such|banks?|RBI|an?)\b)"
)


@dataclass(frozen=True)
class Window:
    start: int  # character offsets into the source text
    end: int
    text: str


def premise_windows(
    text: str,
    *,
    sentences: int = WINDOW_SENTENCES,
    stride: int = WINDOW_STRIDE,
    max_words: int = WINDOW_MAX_WORDS,
) -> list[Window]:
    """Windows of ``sentences`` consecutive sentences, starting every ``stride`` sentences
    (so consecutive windows share ``sentences - stride`` sentences). A window longer than
    ``max_words`` words (RBI sentences can run for a paragraph) is split into overlapping
    word windows, so no text is ever dropped. A short text is a single window."""
    spans = split_sentences(text)
    if not spans:
        return []
    last = max(1, len(spans) - sentences + 1)
    starts = list(range(0, last, stride))
    if starts[-1] != last - 1:
        starts.append(last - 1)  # always cover the tail
    out: list[Window] = []
    seen: set[tuple[int, int]] = set()
    for i in starts:
        group = spans[i : i + sentences]
        for a, b in _word_windows(text, group[0][0], group[-1][1], max_words):
            if (a, b) not in seen:
                seen.add((a, b))
                out.append(Window(a, b, text[a:b]))
    return out


def _word_windows(text: str, start: int, end: int, max_words: int) -> list[tuple[int, int]]:
    words = [m.span() for m in _WORD.finditer(text, start, end)]
    if len(words) <= max_words:
        return [(start, end)]
    step = max(1, (max_words * 2) // 3)  # one third overlap
    out: list[tuple[int, int]] = []
    for w in range(0, len(words), step):
        chunk = words[w : w + max_words]
        out.append((chunk[0][0], chunk[-1][1]))
        if w + max_words >= len(words):
            break
    return out


def split_claims(sentence: str, *, min_words: int = 4) -> list[str]:
    """Heuristic clause split of one answer sentence into atomic claims: semicolons and
    coordinated independent clauses ("..., and the limit is ..."). Parts shorter than
    ``min_words`` words stay attached, so a list like "KYC and AML rules" is not split."""
    parts = [p.strip(" ,.") for p in _CLAIM_SPLIT.split(sentence) if p and p.strip(" ,.")]
    merged: list[str] = []
    for part in parts:
        if merged and len(part.split()) < min_words:
            merged[-1] = f"{merged[-1]} {part}"
        elif not merged or len(merged[-1].split()) >= min_words:
            merged.append(part)
        else:
            merged[-1] = f"{merged[-1]} {part}"
    return merged or [sentence.strip()]


NON_ANSWER_RE = re.compile(
    r"\b(?:do(?:es)? not (?:specify|mention|provide|state|contain|include)|"
    r"(?:is|are) not (?:specified|mentioned|provided|stated)|no (?:specific )?information|"
    r"not (?:explicitly )?(?:covered|addressed|available))\b",
    re.IGNORECASE,
)


def is_non_answer(answer: str) -> bool:
    """An answer that says the sources do not contain the requested fact (it is not a
    factual claim about RBI rules, so it cannot be entailed by the sources)."""
    return bool(NON_ANSWER_RE.search(answer))
