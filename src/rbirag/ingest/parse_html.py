"""Defensive HTML parser for RBI notification pages.

RBI's ``NotificationUser.aspx`` pages are legacy ASP.NET table layouts. Because the markup
cannot be inspected from every environment (and has changed over the years), extraction
tries an ordered list of strategies and records which one matched:

1. ``rbi_tableheader``  - classic layout: title in ``td.tableheader``, body in
   ``td.tablecontent2`` / ``td.tablecontent1`` (or the table that holds the header).
2. ``rbi_container``    - known content containers (``#doublescroll``, ASP.NET
   ``ContentPlaceHolder`` ids, ``main``/``article``, ``#content`` ...).
3. ``main_text_density`` - chrome stripped, then descend from ``<body>`` into whichever
   child holds most of the non-link text (a small readability-style heuristic).
4. ``body_text``        - all visible body text with chrome removed. Last resort.

Each strategy must yield at least ``min_chars`` of text to be accepted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin

from selectolax.lexbor import LexborHTMLParser, LexborNode

from rbirag.ingest.clean import clean_text

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.rbi.org.in/Scripts/"
DEFAULT_MIN_CHARS = 200

_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "head", "iframe", "svg", "select", "button",
     "input", "option", "textarea", "object", "embed", "map"}
)  # fmt: skip
_BLOCK_TAGS = frozenset(
    {"p", "div", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tbody",
     "thead", "tfoot", "section", "article", "blockquote", "pre", "hr", "dl", "dt", "dd",
     "center", "form", "header", "footer", "main", "nav", "aside", "caption", "address",
     "figure", "figcaption"}
)  # fmt: skip
_CELL_TAGS = frozenset({"td", "th"})

# Site chrome removed before the generic fallbacks run.
_CHROME_SELECTORS = (
    "nav",
    "header",
    "footer",
    "aside",
    "form#aspnetForm > input",
    "[role=navigation]",
    "[role=banner]",
    "[role=contentinfo]",
)
_CHROME_ATTR_RE = re.compile(
    r"(^|[\s_-])(nav|navbar|menu|footer|header|breadcrumb|sidebar|banner|topbar|"
    r"skip|social|share|search|accessibility|copyright)([\s_-]|$)",
    re.IGNORECASE,
)

_CONTAINER_SELECTORS = (
    "#doublescroll",
    "[id*=ContentPlaceHolder]",
    "[id*=contentplaceholder]",
    "div.notification",
    "div.notification-content",
    "main",
    "article",
    "#content",
    "#maincontent",
    "#main-content",
    "div.content",
)

_NOT_FOUND_RE = re.compile(
    r"page (you are looking for|not found|cannot be found)|error (has )?occurred|"
    r"no (record|data|notification)s? (found|available)|invalid (request|id)|"
    r"runtime error|server error in",
    re.IGNORECASE,
)
_HEADER_DATE_RE = re.compile(r"^\s*date\s*:?\s*(?P<date>.+?)\s*$", re.IGNORECASE)
_GENERIC_TITLES = frozenset(
    {"notifications", "notification", "reserve bank of india", "circulars", "press releases"}
)
_TITLE_SUFFIX_RE = re.compile(
    r"\s*[-|:]\s*reserve bank of india\s*$|^reserve bank of india\s*[-|:]\s*", re.I
)


@dataclass
class ParsedHTML:
    """Result of parsing one notification page."""

    title: str | None
    text: str
    strategy: str
    pdf_links: list[str] = field(default_factory=list)
    date_hint: str | None = None
    page_title: str | None = None
    tried: list[str] = field(default_factory=list)
    layout_detected: bool = False  # classic RBI table classes present on the page

    @property
    def is_empty(self) -> bool:
        """No strategy matched. (A PDF stub page has a title + PDF link but little text.)"""
        return self.strategy == "none"


# ----------------------------------------------------------------------------- text walking


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def node_to_text(node: LexborNode) -> str:
    """Visible text of ``node`` with block structure preserved as newlines."""
    parts: list[str] = []

    def walk(n: LexborNode, in_pre: bool) -> None:
        for child in n.iter(include_text=True):
            if child.is_text_node:
                raw = child.text_content or ""
                parts.append(raw if in_pre else _collapse(raw))
                continue
            tag = child.tag or ""
            if tag.startswith("-") or tag in _SKIP_TAGS:
                continue
            if tag == "br":
                parts.append("\n")
            elif tag in _BLOCK_TAGS:
                parts.append("\n")
                walk(child, in_pre or tag == "pre")
                parts.append("\n")
            elif tag in _CELL_TAGS:
                parts.append(" ")
                walk(child, in_pre)
                parts.append(" ")
            else:
                walk(child, in_pre)

    walk(node, node.tag == "pre")
    return "".join(parts)


def _link_text_len(node: LexborNode) -> int:
    return sum(len(_collapse(a.text() or "").strip()) for a in node.css("a"))


# ----------------------------------------------------------------------------- helpers


def _decode(html: str | bytes) -> str:
    if isinstance(html, str):
        return html
    for enc in ("utf-8", "cp1252"):
        try:
            return html.decode(enc)
        except UnicodeDecodeError:
            continue
    return html.decode("latin-1")


def _strip_chrome(tree: LexborHTMLParser) -> None:
    for sel in _CHROME_SELECTORS:
        for n in tree.css(sel):
            n.decompose()
    for n in tree.css("[id], [class]"):
        attrs = n.attributes
        label = f"{attrs.get('id') or ''} {attrs.get('class') or ''}"
        if n.tag not in {"body", "html", "main", "article"} and _CHROME_ATTR_RE.search(label):
            n.decompose()


def _page_title(tree: LexborHTMLParser) -> str | None:
    node = tree.css_first("title")
    if node is None:
        return None
    title = _TITLE_SUFFIX_RE.sub("", _collapse(node.text() or "")).strip()
    return title or None


def _is_generic(title: str) -> bool:
    return title.strip().lower().rstrip(":") in _GENERIC_TITLES


def extract_pdf_links(tree: LexborHTMLParser, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """Absolute PDF URLs (rbidocs.rbi.org.in or any ``.pdf`` href), de-duplicated in order."""
    seen: dict[str, None] = {}
    for a in tree.css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "#")):
            continue
        absolute = urljoin(base_url, href)
        lower = absolute.lower()
        if "rbidocs.rbi.org.in" in lower or lower.split("?")[0].endswith(".pdf"):
            seen.setdefault(absolute, None)
    return list(seen)


# ----------------------------------------------------------------------------- strategies

StrategyResult = tuple[str | None, str, str | None]  # (title, raw_text, date_hint)
Strategy = Callable[[LexborHTMLParser], StrategyResult | None]


def _strategy_tableheader(tree: LexborHTMLParser) -> StrategyResult | None:
    headers = tree.css(".tableheader")
    if not headers:
        return None
    title: str | None = None
    date_hint: str | None = None
    for h in headers:
        text = _collapse(h.text() or "").strip()
        if not text:
            continue
        m = _HEADER_DATE_RE.match(text)
        if m:
            date_hint = m.group("date")
            continue
        if _is_generic(text):
            continue
        if title is None or len(text) > len(title):
            title = text

    content_nodes = tree.css(".tablecontent2") or tree.css(".tablecontent1")
    if content_nodes:
        raw = "\n".join(node_to_text(n) for n in content_nodes)
    else:
        # Body sits in sibling rows of the header inside the same table.
        table = headers[0].parent
        while table is not None and table.tag != "table":
            table = table.parent
        if table is None:
            return None
        for h in table.css(".tableheader"):
            h.decompose()
        raw = node_to_text(table)
    return title, raw, date_hint


def _strategy_container(tree: LexborHTMLParser) -> StrategyResult | None:
    for sel in _CONTAINER_SELECTORS:
        node = tree.css_first(sel)
        if node is None:
            continue
        title_node = node.css_first("h1, h2, h3")
        title = _collapse(title_node.text() or "").strip() if title_node else None
        return title or None, node_to_text(node), None
    return None


def _strategy_density(tree: LexborHTMLParser) -> StrategyResult | None:
    _strip_chrome(tree)
    body = tree.body
    if body is None:
        return None

    def content_len(n: LexborNode) -> int:
        return max(0, len(_collapse(n.text() or "").strip()) - _link_text_len(n))

    current = body
    current_len = content_len(current)
    while True:
        best_child: LexborNode | None = None
        best_len = 0
        for child in current.iter():
            if child.tag in _SKIP_TAGS:
                continue
            cl = content_len(child)
            if cl > best_len:
                best_child, best_len = child, cl
        if best_child is None or best_len < 0.7 * current_len:
            break
        current, current_len = best_child, best_len
    if current is body:
        return None
    title_node = current.css_first("h1, h2, h3, b, strong")
    title = _collapse(title_node.text() or "").strip() if title_node else None
    return title or None, node_to_text(current), None


def _strategy_body(tree: LexborHTMLParser) -> StrategyResult | None:
    _strip_chrome(tree)
    body = tree.body
    if body is None:
        return None
    return None, node_to_text(body), None


STRATEGIES: tuple[tuple[str, Strategy], ...] = (
    ("rbi_tableheader", _strategy_tableheader),
    ("rbi_container", _strategy_container),
    ("main_text_density", _strategy_density),
    ("body_text", _strategy_body),
)


# ----------------------------------------------------------------------------- public API


def parse_notification_html(
    html: str | bytes,
    *,
    base_url: str = DEFAULT_BASE_URL,
    min_chars: int = DEFAULT_MIN_CHARS,
    source: str | None = None,
) -> ParsedHTML:
    """Parse a notification page, trying each strategy in order (see module docstring)."""
    decoded = _decode(html)
    probe = LexborHTMLParser(decoded)
    pdf_links = extract_pdf_links(probe, base_url)
    page_title = _page_title(probe)
    layout_detected = _has_rbi_layout(probe)
    tried: list[str] = []

    for name, strategy in STRATEGIES:
        tried.append(name)
        tree = LexborHTMLParser(decoded)  # fresh tree: strategies mutate the DOM
        for n in tree.css("script, style, noscript"):
            n.decompose()
        result = strategy(tree)
        if result is None:
            continue
        title, raw, date_hint = result
        text = clean_text(raw)
        is_pdf_stub = (
            name in {"rbi_tableheader", "rbi_container"}
            and bool(pdf_links)
            and bool(title and not _is_generic(title))
        )
        if len(text) < min_chars and not is_pdf_stub:
            logger.debug("strategy %s rejected for %s: %d chars", name, source, len(text))
            continue
        if title is None or _is_generic(title):
            title = page_title if page_title and not _is_generic(page_title) else None
        logger.info("parse_html source=%s strategy=%s chars=%d", source, name, len(text))
        return ParsedHTML(
            title=clean_text(title) if title else None,
            text=text,
            strategy=name,
            pdf_links=pdf_links,
            date_hint=date_hint,
            page_title=page_title,
            tried=tried,
            layout_detected=layout_detected,
        )

    logger.warning("parse_html source=%s strategy=none (tried %s)", source, ",".join(tried))
    return ParsedHTML(
        title=page_title,
        text="",
        strategy="none",
        pdf_links=pdf_links,
        page_title=page_title,
        tried=tried,
        layout_detected=layout_detected,
    )


def _has_rbi_layout(tree: LexborHTMLParser) -> bool:
    return tree.css_first(".tableheader, .tablecontent1, .tablecontent2, #doublescroll") is not None


def diagnose_html(html: str | bytes, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, object]:
    """Run *every* strategy independently and report what each extracts (for ``--probe``)."""
    decoded = _decode(html)
    probe = LexborHTMLParser(decoded)
    report: dict[str, object] = {
        "bytes": len(decoded.encode("utf-8")),
        "page_title": _page_title(probe),
        "rbi_layout_detected": _has_rbi_layout(probe),
        "selector_hits": {
            sel: len(probe.css(sel))
            for sel in (
                ".tableheader",
                ".tablecontent1",
                ".tablecontent2",
                "#doublescroll",
                "[id*=ContentPlaceHolder]",
                "main",
                "article",
                "table",
                "p",
            )
        },
        "pdf_links": extract_pdf_links(probe, base_url),
    }
    strategies: dict[str, dict[str, object]] = {}
    for name, strategy in STRATEGIES:
        tree = LexborHTMLParser(decoded)
        for n in tree.css("script, style, noscript"):
            n.decompose()
        result = strategy(tree)
        if result is None:
            strategies[name] = {"matched": False}
            continue
        title, raw, date_hint = result
        text = clean_text(raw)
        strategies[name] = {
            "matched": True,
            "chars": len(text),
            "title": title,
            "date_hint": date_hint,
            "preview": text[:160].replace("\n", " / "),
        }
    report["strategies"] = strategies
    return report


def looks_not_found(html: str | bytes, *, min_chars: int = DEFAULT_MIN_CHARS) -> bool:
    """Heuristic for RBI's "empty" pages returned for non-existent notification IDs."""
    parsed = parse_notification_html(html, min_chars=min_chars)
    return parsed_looks_not_found(parsed, min_chars=min_chars)


def parsed_looks_not_found(parsed: ParsedHTML, *, min_chars: int = DEFAULT_MIN_CHARS) -> bool:
    """Same as :func:`looks_not_found` but on an already-parsed page."""
    if parsed.is_empty:
        return True
    if (
        parsed.layout_detected
        and parsed.strategy in {"main_text_density", "body_text"}
        and len(parsed.text) < 4 * min_chars
    ):
        # The RBI layout is there but its content cells are empty: an unused ID.
        return True
    head = parsed.text[:600]
    return bool(_NOT_FOUND_RE.search(head)) and len(parsed.text) < 4 * min_chars
