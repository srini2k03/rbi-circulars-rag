"""End-to-end test of the static site in Chromium (Playwright), with a tiny fixture index
and a mocked LLM. Runs in the CI ``web`` job; skipped locally when no browser is installed
(set RBIRAG_REQUIRE_BROWSER=1 to make a missing browser an error)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rbirag.document import Document
from tests.webutil import build_site, serve

pytestmark = pytest.mark.browser
sync_api = pytest.importorskip("playwright.sync_api")

QUESTION = "periodic updation of KYC for high risk customers"
SECRET = "gsk_test_key_123"


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:
            if os.environ.get("RBIRAG_REQUIRE_BROWSER"):
                raise
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


@pytest.fixture
def site(tmp_path: Path, fixture_docs: list[Document]) -> Iterator[str]:
    root = tmp_path / "site"
    build_site(root, fixture_docs)
    with serve(root) as base:
        yield base


def test_search_and_mocked_answer(browser: Any, site: str) -> None:
    page = browser.new_page()
    seen: list[dict[str, Any]] = []
    leaked: list[str] = []

    def groq(route: Any) -> None:
        req = route.request
        seen.append({"auth": req.headers.get("authorization"), "body": json.loads(req.post_data)})
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": "At least once every two years [1]."}}],
        }))  # fmt: skip

    page.route("https://api.groq.com/**", groq)
    page.on(
        "request",
        lambda r: (
            leaked.append(r.url)
            if SECRET in (r.url + str(r.headers)) and "groq.com" not in r.url
            else None
        ),
    )
    page.goto(f"{site}/index.html?models=off")
    page.wait_for_selector("#search-btn:not([disabled])")
    assert "BM25 only" in page.inner_text("#index-info")
    page.fill("#question", QUESTION)
    page.click("#search-btn")
    page.wait_for_selector("#results li.result")
    first = page.locator("#results li.result").first
    assert first.get_attribute("data-doc-id") == "rbi-12655"
    assert "NotificationUser.aspx?Id=12655" in first.locator("a").first.get_attribute("href")

    # without a key: no request, a hint instead
    page.click("#answer-btn")
    assert "Add your own" in page.inner_text("#answer")
    assert not seen

    page.click("summary:has-text('Answer generation')")
    page.select_option("#provider", "groq")
    page.fill("#api-key", SECRET)
    page.click("#answer-btn")
    page.wait_for_selector("#answer .answer-text")
    assert "At least once every two years" in page.inner_text("#answer")
    assert page.locator("#answer a[href='#src-1']").count() == 1
    assert seen[0]["auth"] == f"Bearer {SECRET}"
    body = seen[0]["body"]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"].startswith("Sources:\n\n[1] ")
    assert body["messages"][1]["content"].endswith(f"Question: {QUESTION}")
    assert not leaked
    # not remembered unless asked
    assert page.evaluate("localStorage.getItem('rbirag.llmKey')") is None
    page.close()


def test_dashboard_renders_results(browser: Any, site: str) -> None:
    page = browser.new_page()
    page.goto(f"{site}/index.html?models=off#evaluation")
    page.wait_for_selector("#tab-evaluation table")
    text = page.inner_text("#tab-evaluation")
    assert "model-generated test set" in text
    assert "section-512-hybrid_rerank" in text
    assert "incomplete (3/10)" in text
    page.close()


def test_dashboard_in_progress(browser: Any, tmp_path: Path, fixture_docs: list[Document]) -> None:
    root = tmp_path / "bare"
    build_site(root, fixture_docs, results=False)
    with serve(root) as base:
        page = browser.new_page()
        page.goto(f"{base}/index.html?models=off#evaluation")
        page.wait_for_selector("#tab-evaluation h2")
        assert "Evaluation in progress" in page.inner_text("#tab-evaluation")
        page.close()
