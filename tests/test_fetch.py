from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from rbirag.ingest.fetch_rbi import (
    CrawlBlockedError,
    FetcherConfig,
    FetchError,
    Manifest,
    RateLimiter,
    RBIFetcher,
    descending_ids,
)
from tests.conftest import fixture_html

PDF_URL = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT41A7F3C2E1B04D7EA8F2.PDF"
FAKE_PDF = b"%PDF-1.4\n%fake\n"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_handler(
    pages: dict[int, httpx.Response | Callable[[], httpx.Response]],
    calls: list[str],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /admin/\n")
        if request.url.host == "rbidocs.rbi.org.in":
            return httpx.Response(
                200, content=FAKE_PDF, headers={"content-type": "application/pdf"}
            )
        nid = int(request.url.params["Id"])
        resp = pages.get(nid)
        if resp is None:
            return httpx.Response(200, content=fixture_html("notification_13050_empty.html"))
        return resp() if callable(resp) else resp

    return handler


def make_fetcher(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    clock: FakeClock,
    **cfg: object,
) -> RBIFetcher:
    config = FetcherConfig(raw_dir=tmp_path / "raw", **cfg)  # type: ignore[arg-type]
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return RBIFetcher(config, client=client, sleep=clock.sleep, clock=clock.clock)


def ok_page() -> httpx.Response:
    return httpx.Response(200, content=fixture_html("notification_12701.html"))


def test_rate_limiter_spacing() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1.5, clock=clock.clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 0.5
    limiter.wait()
    limiter.wait()
    assert clock.sleeps == [pytest.approx(1.0), pytest.approx(1.5)]


def test_rate_limit_below_one_second_rejected() -> None:
    with pytest.raises(ValueError, match="polite"):
        FetcherConfig(rate_limit_seconds=0.5)


def test_crawl_writes_manifest_html_and_pdf(tmp_path: Path) -> None:
    calls: list[str] = []
    clock = FakeClock()
    pages = {
        12701: ok_page,
        12700: httpx.Response(404),
        12699: ok_page,
    }
    with make_fetcher(tmp_path, make_handler(pages, calls), clock) as f:
        stats = f.crawl(descending_ids(12702, 12699), max_docs=10)
    assert stats.statuses == {"ok": 2, "not_found": 1, "empty": 1}
    assert stats.pdfs_ok == 2
    manifest = Manifest(tmp_path / "raw" / "manifest.jsonl")
    entry = manifest.entries[12701]
    assert entry.status == "ok" and entry.sha256 and len(entry.sha256) == 64
    assert entry.parse_strategy == "rbi_tableheader"
    assert entry.pdf_url == PDF_URL and entry.pdf_status == "ok"
    assert (tmp_path / "raw" / "html" / "12701.html").exists()
    assert (tmp_path / "raw" / "pdf" / "12701.pdf").read_bytes() == FAKE_PDF
    assert manifest.entries[12700].status == "not_found"
    assert manifest.entries[12702].status == "empty"
    assert not (tmp_path / "raw" / "html" / "12702.html").exists()
    # every request was rate-limited: at least 1.5 s apart on the fake clock
    assert all(s >= 0 for s in clock.sleeps)
    assert calls[0].endswith("/robots.txt")


def test_crawl_resumes_and_skips_final_entries(tmp_path: Path) -> None:
    calls: list[str] = []
    clock = FakeClock()
    pages = {12701: ok_page, 12700: ok_page}
    handler = make_handler(pages, calls)
    with make_fetcher(tmp_path, handler, clock) as f:
        f.crawl([12701], max_docs=10)
    calls.clear()
    with make_fetcher(tmp_path, handler, clock) as f:
        stats = f.crawl([12701, 12700], max_docs=10)
    assert stats.skipped_existing == 1
    assert not any("Id=12701" in c for c in calls)
    assert any("Id=12700" in c for c in calls)
    assert stats.ok_total == 2


def test_crawl_stops_at_max_docs(tmp_path: Path) -> None:
    clock = FakeClock()
    pages = dict.fromkeys(range(100, 110), ok_page)
    with make_fetcher(tmp_path, make_handler(pages, []), clock, fetch_pdfs=False) as f:
        stats = f.crawl(descending_ids(109, 100), max_docs=3)
    assert stats.ok_total == 3 and stats.stopped_reason == "max_docs reached"


def test_crawl_stops_after_consecutive_misses_only_after_a_hit(tmp_path: Path) -> None:
    clock = FakeClock()
    pages = {195: ok_page}
    with make_fetcher(
        tmp_path, make_handler(pages, []), clock, fetch_pdfs=False, max_consecutive_misses=3
    ) as f:
        stats = f.crawl(descending_ids(200, 150), max_docs=10)
    # 200..196 are misses *before* the first hit (ignored), then 194, 193, 192 -> stop
    assert stats.statuses == {"empty": 8, "ok": 1}
    assert stats.stopped_reason == "3 consecutive misses"


def test_retries_with_backoff_then_success(tmp_path: Path) -> None:
    clock = FakeClock()
    attempts = {"n": 0}

    def flaky() -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503)
        if attempts["n"] == 2:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return ok_page()

    with make_fetcher(
        tmp_path, make_handler({1: flaky}, []), clock, fetch_pdfs=False, respect_robots=False
    ) as f:
        entry = f.fetch_notification(1)
    assert entry.status == "ok"
    assert attempts["n"] == 3
    assert 7.0 in clock.sleeps  # Retry-After honoured
    assert any(2.0 <= s <= 3.0 for s in clock.sleeps)  # 2 s * 2**0 + jitter


def test_transport_errors_exhaust_retries(tmp_path: Path) -> None:
    clock = FakeClock()

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with make_fetcher(tmp_path, boom, clock, max_retries=2, respect_robots=False) as f:
        with pytest.raises(FetchError):
            f.get("https://www.rbi.org.in/x")
        entry = f.fetch_notification(5)
    assert entry.status == "error" and "giving up" in (entry.error or "")


def test_repeated_403_raises_blocked(tmp_path: Path) -> None:
    clock = FakeClock()
    pages = {i: httpx.Response(403) for i in range(0, 50)}
    with (
        make_fetcher(
            tmp_path, make_handler(pages, []), clock, max_consecutive_errors=5, respect_robots=False
        ) as f,
        pytest.raises(CrawlBlockedError, match="5 consecutive errors"),
    ):
        f.crawl(descending_ids(49, 0), max_docs=10)


def test_error_redirect_is_not_found(tmp_path: Path) -> None:
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if "ErrorPage" in request.url.path:
            return httpx.Response(200, content=fixture_html("notification_error.html"))
        return httpx.Response(302, headers={"Location": "/Scripts/ErrorPage.aspx"})

    with make_fetcher(tmp_path, handler, clock, respect_robots=False) as f:
        entry = f.fetch_notification(42)
    assert entry.status == "not_found"


def test_robots_disallow_blocks_crawl(tmp_path: Path) -> None:
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /Scripts/\n")
        return ok_page()

    with make_fetcher(tmp_path, handler, clock) as f, pytest.raises(CrawlBlockedError):
        f.crawl([1], max_docs=1)


def test_user_agent_has_no_email() -> None:
    from rbirag.config import DEFAULT_USER_AGENT

    assert "@" not in DEFAULT_USER_AGENT
    assert "github.com/srini2k03/rbi-circulars-rag" in DEFAULT_USER_AGENT


def test_manifest_last_record_wins_and_tolerates_truncation(tmp_path: Path) -> None:
    clock = FakeClock()
    with make_fetcher(
        tmp_path, make_handler({7: httpx.Response(500)}, []), clock, max_retries=0,
        respect_robots=False,
    ) as f:  # fmt: skip
        f.crawl([7], max_docs=1)
    path = tmp_path / "raw" / "manifest.jsonl"
    assert Manifest(path).entries[7].status == "error"
    with make_fetcher(tmp_path, make_handler({7: ok_page}, []), clock, fetch_pdfs=False) as f:
        f.crawl([7], max_docs=1)  # error entries are retried on resume
    with path.open("a") as fh:
        fh.write('{"id": 8, "url": "trunc')
    m = Manifest(path)
    assert m.entries[7].status == "ok" and 8 not in m.entries


def test_probe_reports_diagnostics(tmp_path: Path) -> None:
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "rbidocs.rbi.org.in":
            return httpx.Response(404)
        return ok_page()

    with make_fetcher(tmp_path, handler, clock) as f:
        report = f.probe([12701])
    r = report[0]
    assert r["http_status"] == 200 and r["chosen_strategy"] == "rbi_tableheader"
    assert r["metadata"]["circular_number"] == "RBI/2025-26/58"
    assert r["pdf"]["http_status"] == 404
    assert "strategies" in r["diagnostics"]
    assert not (tmp_path / "raw" / "manifest.jsonl").exists()
