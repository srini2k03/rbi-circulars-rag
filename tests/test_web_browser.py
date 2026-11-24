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
    # query terms are highlighted; BM25 ran and the model stages are marked off
    assert first.locator("mark").count() > 0
    assert page.get_attribute(".stage[data-stage='bm25']", "data-state") == "done"
    assert page.get_attribute(".stage[data-stage='rerank']", "data-state") == "off"

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
    # the ablation heatmap has one cell per config; the hero reads its numbers from results/
    assert page.locator("#tab-evaluation table.heatmap td.heat:not(.heat--na)").count() == 2
    assert page.inner_text("[data-stat='ndcg']") == "0.70"
    assert page.inner_text("[data-stat='faith']") == "0.90"
    page.close()


def test_dashboard_in_progress(browser: Any, tmp_path: Path, fixture_docs: list[Document]) -> None:
    root = tmp_path / "bare"
    build_site(root, fixture_docs, results=False)
    with serve(root) as base:
        page = browser.new_page()
        page.goto(f"{base}/index.html?models=off#evaluation")
        page.wait_for_selector("#tab-evaluation h2")
        assert "Evaluation in progress" in page.inner_text("#tab-evaluation")
        assert page.inner_text("[data-stat='ndcg-ctx']") == "evaluation in progress"
        page.close()


def test_advanced_mode_verification_feedback_and_documents(browser: Any, site: str) -> None:
    page = browser.new_page()

    def groq(route: Any) -> None:
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content":
                "Periodic updation is carried out at least once every two years [1]. "
                "The penalty is Rs. 987654 crore [7]."}}],
        }))  # fmt: skip

    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("https://api.groq.com/**", groq)
    page.goto(f"{site}/index.html?models=off")
    page.wait_for_selector("#search-btn:not([disabled])")
    page.check("#advanced-toggle")
    page.fill("#question", QUESTION)
    page.click("#search-btn")
    page.wait_for_selector("#results li.result")
    first = page.locator("#results li.result").first
    assert first.locator("table.trace").is_visible()
    assert "in context" in first.locator("table.trace").inner_text()
    assert page.locator("#history-list li").count() == 1
    page.uncheck("#advanced-toggle")
    assert not first.locator("table.trace").is_visible()
    # document panel: metadata, the corpus-scoped "latest" with its disclaimer
    first.locator(".doc-btn").click()
    page.wait_for_selector("#doc-dialog .docmeta")
    panel = page.inner_text("#doc-dialog")
    assert "Category" in panel and "not a verified statement" in panel
    page.click("#doc-close")
    assert not page.locator("#doc-dialog").is_visible()
    # a mocked answer with an unsupported figure and a citation that does not exist
    page.evaluate("document.querySelector('#llm-settings').open = true")
    page.select_option("#provider", "groq")
    page.fill("#api-key", SECRET)
    page.click("#answer-btn")
    page.wait_for_selector("#answer .verify")
    answer = page.inner_text("#answer .answer-text")
    assert "[7]" not in answer and "[1]" in answer
    assert page.locator("#answer .claim--unsupported").count() == 1
    assert page.locator("#answer .claim--supported").count() == 1
    assert "removed" in page.inner_text("#answer .verify")
    # feedback stays local and can be exported
    assert page.locator("#feedback-export").is_disabled()
    page.click("#answer [data-rate='down']")
    assert page.locator("#feedback-export").is_enabled()
    with page.expect_download() as dl:
        page.click("#feedback-export")
    record = json.loads(Path(dl.value.path()).read_text().splitlines()[0])
    assert record["rating"] == "down" and record["verdict"] == "partially supported"
    assert SECRET not in json.dumps(record)
    page.click("#tab-btn-evaluation")  # tab switching uses window.history
    page.wait_for_selector("#tab-evaluation h2")
    assert not errors, errors
    page.close()
