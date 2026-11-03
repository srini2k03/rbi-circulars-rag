"""Polite, resumable crawler for RBI notification pages and their PDF attachments.

* One request at a time, at least ``rate_limit_seconds`` apart (shared across hosts).
* Retries with exponential backoff + jitter on transport errors, 429 and 5xx
  (``Retry-After`` honoured).
* Append-only JSONL manifest (``id -> status, sha256, fetched_at, ...``); the last record
  for an id wins, so an interrupted crawl resumes where it stopped.
* Unused IDs (404s, error redirects, empty RBI templates) are recorded as ``not_found`` /
  ``empty`` and skipped on resume.
* ``robots.txt`` is checked once before crawling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal
from urllib import robotparser
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from rbirag.config import DEFAULT_USER_AGENT
from rbirag.ingest.parse_html import (
    diagnose_html,
    parse_notification_html,
    parsed_looks_not_found,
)

logger = logging.getLogger(__name__)

FetchStatus = Literal["ok", "empty", "not_found", "error"]
FINAL_STATUSES: frozenset[str] = frozenset({"ok", "empty", "not_found"})
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_URL_TEMPLATE = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={id}&Mode=0"


class CrawlBlockedError(RuntimeError):
    """Raised when the server keeps refusing us (e.g. a run of 403s) or robots.txt forbids."""


class FetchError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utcnow() -> datetime:
    return datetime.now(UTC)


# ----------------------------------------------------------------------------- manifest


class ManifestEntry(BaseModel):
    id: int
    url: str
    status: FetchStatus
    http_status: int | None = None
    final_url: str | None = None
    sha256: str | None = None
    bytes: int = 0
    fetched_at: datetime
    html_path: str | None = None
    title: str | None = None
    parse_strategy: str | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    pdf_sha256: str | None = None
    pdf_status: str | None = None
    error: str | None = None


class Manifest:
    """Append-only JSONL manifest; the most recent record per id is authoritative."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[int, ManifestEntry] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = ManifestEntry.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    # A crash mid-write can leave a truncated last line; skip it.
                    logger.warning("manifest %s:%d unreadable (%s)", self.path, lineno, exc)
                    continue
                self.entries[entry.id] = entry

    def record(self, entry: ManifestEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
            fh.flush()
        self.entries[entry.id] = entry

    def is_done(self, notification_id: int) -> bool:
        entry = self.entries.get(notification_id)
        return entry is not None and entry.status in FINAL_STATUSES

    def ok_entries(self) -> list[ManifestEntry]:
        return sorted((e for e in self.entries.values() if e.status == "ok"), key=lambda e: -e.id)

    def status_counts(self) -> Counter[str]:
        return Counter(e.status for e in self.entries.values())


# ----------------------------------------------------------------------------- rate limit


class RateLimiter:
    """Guarantees at least ``min_interval`` seconds between consecutive ``wait()`` returns."""

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


# ----------------------------------------------------------------------------- crawler


@dataclass
class CrawlStats:
    attempted: int = 0
    skipped_existing: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    pdfs_ok: int = 0
    stopped_reason: str = ""
    ok_total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "skipped_existing": self.skipped_existing,
            "statuses": dict(self.statuses),
            "pdfs_ok": self.pdfs_ok,
            "ok_total": self.ok_total,
            "stopped_reason": self.stopped_reason,
        }


@dataclass
class FetcherConfig:
    raw_dir: Path = Path("data/raw")
    url_template: str = DEFAULT_URL_TEMPLATE
    user_agent: str = DEFAULT_USER_AGENT
    rate_limit_seconds: float = 1.5
    max_retries: int = 4
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    timeout_seconds: float = 30.0
    fetch_pdfs: bool = True
    max_consecutive_misses: int = 300
    max_consecutive_errors: int = 20
    respect_robots: bool = True
    min_chars: int = 200

    def __post_init__(self) -> None:
        if self.rate_limit_seconds < 1.0:
            raise ValueError("rate_limit_seconds must be >= 1.0 (be polite)")


class RBIFetcher:
    def __init__(
        self,
        config: FetcherConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.Client(
            headers={
                "User-Agent": config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IN,en;q=0.9",
            },
            timeout=config.timeout_seconds,
            follow_redirects=True,
        )
        self._sleep = sleep
        self.limiter = RateLimiter(config.rate_limit_seconds, clock=clock, sleep=sleep)
        self.manifest = Manifest(config.raw_dir / "manifest.jsonl")
        self._robots: robotparser.RobotFileParser | None = None

    # -- context management ------------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> RBIFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- HTTP --------------------------------------------------------------------------
    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.config.backoff_max_seconds)
            except ValueError:
                try:
                    delta = (parsedate_to_datetime(retry_after) - utcnow()).total_seconds()
                    return max(0.0, min(delta, self.config.backoff_max_seconds))
                except (TypeError, ValueError):
                    pass
        base = self.config.backoff_base_seconds * (2**attempt)
        return float(min(base + random.uniform(0, 1), self.config.backoff_max_seconds))

    def get(self, url: str) -> httpx.Response:
        """Rate-limited GET with retries. Returns the final response (any status)."""
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.client.get(url)
            except httpx.TransportError as exc:
                last_exc = exc
                delay = self._backoff(attempt, None)
                logger.warning("GET %s failed (%s); retry in %.1fs", url, exc, delay)
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    return response
                delay = self._backoff(attempt, response.headers.get("Retry-After"))
                last_exc = FetchError(f"HTTP {response.status_code}", response.status_code)
                logger.warning("GET %s -> %d; retry in %.1fs", url, response.status_code, delay)
            if attempt < self.config.max_retries:
                self._sleep(delay)
        status = last_exc.status_code if isinstance(last_exc, FetchError) else None
        raise FetchError(f"giving up on {url}: {last_exc}", status) from last_exc

    # -- robots ------------------------------------------------------------------------
    def allowed_by_robots(self, url: str) -> bool:
        if not self.config.respect_robots:
            return True
        if self._robots is None:
            parts = urlsplit(url)
            robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
            parser = robotparser.RobotFileParser()
            try:
                resp = self.get(robots_url)
            except FetchError as exc:
                logger.info("robots.txt unavailable (%s); proceeding", exc)
                parser.parse([])
            else:
                parser.parse(resp.text.splitlines() if resp.status_code == 200 else [])
            self._robots = parser
        return self._robots.can_fetch(self.config.user_agent, url)

    # -- single notification -----------------------------------------------------------
    def url_for(self, notification_id: int) -> str:
        return self.config.url_template.format(id=notification_id)

    def fetch_notification(self, notification_id: int) -> ManifestEntry:
        url = self.url_for(notification_id)
        now = utcnow()
        try:
            response = self.get(url)
        except FetchError as exc:
            return ManifestEntry(
                id=notification_id, url=url, status="error", http_status=exc.status_code,
                fetched_at=now, error=str(exc),
            )  # fmt: skip

        final_url = str(response.url)
        base = ManifestEntry(
            id=notification_id, url=url, status="error", http_status=response.status_code,
            final_url=final_url, fetched_at=now,
        )  # fmt: skip
        if response.status_code == 404 or "error" in urlsplit(final_url).path.lower():
            return base.model_copy(update={"status": "not_found"})
        if response.status_code != 200:
            return base.model_copy(update={"error": f"HTTP {response.status_code}"})

        content = response.content
        parsed = parse_notification_html(content, base_url=final_url, source=str(notification_id))
        if parsed_looks_not_found(parsed, min_chars=self.config.min_chars):
            return base.model_copy(
                update={"status": "empty", "bytes": len(content), "sha256": sha256_bytes(content)}
            )

        html_path = self.config.raw_dir / "html" / f"{notification_id}.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_bytes(content)
        entry = base.model_copy(
            update={
                "status": "ok",
                "bytes": len(content),
                "sha256": sha256_bytes(content),
                "html_path": html_path.relative_to(self.config.raw_dir).as_posix(),
                "title": parsed.title,
                "parse_strategy": parsed.strategy,
                "pdf_url": parsed.pdf_links[0] if parsed.pdf_links else None,
            }
        )
        if self.config.fetch_pdfs and entry.pdf_url:
            entry = self._fetch_pdf(entry)
        return entry

    def _fetch_pdf(self, entry: ManifestEntry) -> ManifestEntry:
        assert entry.pdf_url is not None
        try:
            resp = self.get(entry.pdf_url)
        except FetchError as exc:
            return entry.model_copy(update={"pdf_status": f"error: {exc}"})
        ctype = resp.headers.get("content-type", "").lower()
        if resp.status_code != 200 or not (resp.content[:5] == b"%PDF-" or "pdf" in ctype):
            return entry.model_copy(update={"pdf_status": f"http {resp.status_code} {ctype}"})
        pdf_path = self.config.raw_dir / "pdf" / f"{entry.id}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(resp.content)
        return entry.model_copy(
            update={
                "pdf_status": "ok",
                "pdf_path": pdf_path.relative_to(self.config.raw_dir).as_posix(),
                "pdf_sha256": sha256_bytes(resp.content),
            }
        )

    # -- crawl -------------------------------------------------------------------------
    def crawl(
        self,
        ids: Iterable[int],
        *,
        max_docs: int,
        refetch: bool = False,
        on_entry: Callable[[ManifestEntry], None] | None = None,
    ) -> CrawlStats:
        """Fetch ``ids`` in order until ``max_docs`` notifications are stored (``ok``).

        Already-final IDs in the manifest are skipped unless ``refetch``. Stops early after
        ``max_consecutive_misses`` misses *once at least one document was found* (scanning
        down from above the newest ID starts with a run of misses) and raises
        :class:`CrawlBlockedError` after ``max_consecutive_errors`` hard errors.
        """
        stats = CrawlStats()
        ok_total = sum(1 for e in self.manifest.entries.values() if e.status == "ok")
        misses = errors = 0
        seen_ok = ok_total > 0
        checked_robots = False
        for nid in ids:
            if ok_total >= max_docs:
                stats.stopped_reason = "max_docs reached"
                break
            if not refetch and self.manifest.is_done(nid):
                stats.skipped_existing += 1
                continue
            if not checked_robots:
                checked_robots = True
                if not self.allowed_by_robots(self.url_for(nid)):
                    raise CrawlBlockedError("robots.txt disallows notification pages")
            entry = self.fetch_notification(nid)
            self.manifest.record(entry)
            stats.attempted += 1
            stats.statuses[entry.status] += 1
            if entry.pdf_status == "ok":
                stats.pdfs_ok += 1
            if on_entry:
                on_entry(entry)
            if entry.status == "ok":
                ok_total += 1
                seen_ok = True
                misses = errors = 0
            elif entry.status == "error":
                errors += 1
                if errors >= self.config.max_consecutive_errors:
                    stats.ok_total = ok_total
                    raise CrawlBlockedError(
                        f"{errors} consecutive errors; last: {entry.error} "
                        f"(HTTP {entry.http_status})"
                    )
            else:
                misses += 1
                errors = 0
                if seen_ok and misses >= self.config.max_consecutive_misses:
                    stats.stopped_reason = f"{misses} consecutive misses"
                    break
        else:
            stats.stopped_reason = stats.stopped_reason or "id range exhausted"
        stats.ok_total = ok_total
        return stats

    # -- probe -------------------------------------------------------------------------
    def probe(self, ids: Iterable[int]) -> list[dict[str, Any]]:
        """Fetch a handful of IDs and report everything useful for debugging the parser.

        Nothing is written to the manifest.
        """
        from rbirag.ingest.metadata import extract_metadata

        reports: list[dict[str, Any]] = []
        for nid in ids:
            url = self.url_for(nid)
            report: dict[str, Any] = {"id": nid, "url": url}
            started = time.monotonic()
            try:
                resp = self.get(url)
            except FetchError as exc:
                report.update(error=str(exc), http_status=exc.status_code)
                reports.append(report)
                continue
            report.update(
                http_status=resp.status_code,
                final_url=str(resp.url),
                elapsed_s=round(time.monotonic() - started, 2),
                content_type=resp.headers.get("content-type"),
                server=resp.headers.get("server"),
                redirects=[str(r.url) for r in resp.history],
            )
            if resp.status_code == 200:
                parsed = parse_notification_html(resp.content, base_url=str(resp.url))
                report["diagnostics"] = diagnose_html(resp.content, base_url=str(resp.url))
                report["chosen_strategy"] = parsed.strategy
                report["looks_not_found"] = parsed_looks_not_found(parsed)
                report["title"] = parsed.title
                if parsed.text:
                    md = extract_metadata(
                        parsed.text, html_title=parsed.title, date_hint=parsed.date_hint
                    )
                    report["metadata"] = {
                        "date": md.date.isoformat() if md.date else None,
                        "circular_number": md.circular_number,
                        "ref_number": md.ref_number,
                        "department": md.department,
                        "addressee": md.addressee,
                        "sections": len(md.sections),
                    }
                if parsed.pdf_links and self.config.fetch_pdfs:
                    report["pdf"] = self._probe_pdf(parsed.pdf_links[0])
            reports.append(report)
        return reports

    def _probe_pdf(self, pdf_url: str) -> dict[str, Any]:
        info: dict[str, Any] = {"url": pdf_url}
        try:
            resp = self.get(pdf_url)
        except FetchError as exc:
            info["error"] = str(exc)
            return info
        info.update(
            http_status=resp.status_code,
            content_type=resp.headers.get("content-type"),
            bytes=len(resp.content),
            magic_ok=resp.content[:5] == b"%PDF-",
        )
        if info["magic_ok"]:
            from rbirag.ingest.parse_pdf import parse_pdf

            parsed = parse_pdf(resp.content, source=pdf_url)
            info.update(pages=parsed.page_count, chars=len(parsed.text))
        return info


def descending_ids(start_id: int, min_id: int) -> Iterator[int]:
    """``start_id, start_id - 1, ..., min_id``."""
    return iter(range(start_id, min_id - 1, -1))


def default_probe_ids(start_id: int) -> list[int]:
    """Five IDs spread below ``start_id``: very recent through a few years back."""
    return [start_id, start_id - 60, start_id - 250, start_id - 800, start_id - 2000]
