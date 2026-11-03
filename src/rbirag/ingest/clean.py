"""Text normalisation and boilerplate removal for RBI pages and PDFs."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

# Characters that NFKC leaves alone but that hurt lexical matching.
# Built from code points so the source stays ASCII-only.
_QUOTES_SINGLE = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
_QUOTES_DOUBLE = (0x201C, 0x201D, 0x201E, 0x2033)
_DASHES = (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)
_BULLETS = (0x2022, 0x25CF, 0x25AA, 0xF0B7)  # 0xF0B7: Symbol-font bullet in Word PDFs
_SPACES = (0x00A0, 0x2002, 0x2003, 0x2009, 0x202F, 0x3000, 0x0009)
_TRANSLATE: dict[int, str] = {
    **dict.fromkeys(_QUOTES_SINGLE, "'"),
    **dict.fromkeys(_QUOTES_DOUBLE, '"'),
    **dict.fromkeys(_DASHES, "-"),
    **dict.fromkeys(_BULLETS, "-"),
    **dict.fromkeys(_SPACES, " "),
}
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ ]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Whole lines that are site chrome, never content. Matched case-insensitively against a
# stripped line.
_BOILERPLATE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^top$",
        r"^print$",
        r"^back$",
        r"^share$",
        r"^home$",
        r"^skip to (main )?content$",
        r"^home\s*[>\u00bb].*$",
        r"^notifications?$",
        r"^page last updated on\s*:?.*$",
        r"^last updated\s*:?.*$",
        r"^\(?\s*\d+(\.\d+)?\s*[km]b\s*\)?$",  # "(310 kb)" next to a PDF icon
        r"^pdf( icon)?$",
        r"^download( the)? pdf$",
        r"^click here.*$",
        r"^\u00a9.*reserve bank of india.*$",
        r"^copyright.*reserve bank of india.*$",
        r"^all rights reserved\.?$",
        r"^disclaimer$",
        r"^best viewed in .*$",
        r"^(site map|sitemap|faqs?|contact us|feedback|rti|accessibility)$",
        r"^(?:page\s*)?\d+\s*(?:of|/)\s*\d+$",  # "Page 2 of 7", "2/7"
        r"^-\s*\d+\s*-$",  # "- 3 -"
        r"^date\s*:?$",
    )
)


def normalize_unicode(text: str) -> str:
    """NFKC + quote/dash/space folding + zero-width/control stripping. Keeps newlines."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATE)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces, trim lines, and keep at most one blank line in a row."""
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _BOILERPLATE_LINE_PATTERNS)


def remove_boilerplate(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not is_boilerplate_line(line))


def strip_repeated_page_lines(
    pages: Sequence[str], *, edge_lines: int = 3, min_fraction: float = 0.6
) -> list[str]:
    """Drop running headers/footers: lines near a page edge that repeat on most pages.

    Digits are masked before counting so "Page 3" and "Page 4" are recognised as the same
    footer.
    """
    if len(pages) < 3:
        return list(pages)

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    def n_edge(n_lines: int) -> int:
        # Short pages: never treat more than a third of the page as header/footer.
        return max(1, min(edge_lines, n_lines // 3))

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.split("\n") if ln.strip()]
        e = n_edge(len(lines))
        edge = {key(ln) for ln in lines[:e] + lines[-e:]}
        counts.update(edge)
    threshold = max(2, int(len(pages) * min_fraction + 0.5))
    repeated = {k for k, c in counts.items() if c >= threshold and k}

    cleaned: list[str] = []
    for page in pages:
        lines = page.split("\n")
        non_empty_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        e = n_edge(len(non_empty_idx))
        edge_idx = set(non_empty_idx[:e] + non_empty_idx[-e:])
        kept = [ln for i, ln in enumerate(lines) if not (i in edge_idx and key(ln) in repeated)]
        cleaned.append("\n".join(kept))
    return cleaned


_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?-?\s*\d{1,3}\s*-?\s*$", re.IGNORECASE)


def strip_page_numbers(page: str, *, edge_lines: int = 2) -> str:
    """Remove bare page numbers, but only from the first/last lines of a page."""
    lines = page.split("\n")
    non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
    edges = set(non_empty[:edge_lines] + non_empty[-edge_lines:])
    return "\n".join(
        ln for i, ln in enumerate(lines) if not (i in edges and _PAGE_NUMBER.match(ln))
    )


def clean_text(text: str) -> str:
    """Full cleaning pipeline used for both HTML- and PDF-derived text."""
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = remove_boilerplate(text)
    return normalize_whitespace(text)
