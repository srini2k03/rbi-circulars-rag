"""Render the built Pages site in Chromium against the real index (no LLM) and fail if a
known question returns no passages. Used by .github/workflows/pages.yml before publishing.

    python scripts/check_site.py --site _site --question "..." [--models on|off]
"""

from __future__ import annotations

import argparse
import functools
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--question", action="append", required=True)
    ap.add_argument("--models", choices=["on", "off"], default="on")
    ap.add_argument("--timeout-s", type=float, default=600)
    args = ap.parse_args()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(args.site))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    failures = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on(
            "console",
            lambda m: (
                print(f"[console.{m.type}] {m.text}") if m.type in {"error", "warning"} else None
            ),
        )
        suffix = "?models=off" if args.models == "off" else ""
        page.goto(f"{base}/index.html{suffix}")
        page.wait_for_selector("#search-btn:not([disabled])", timeout=args.timeout_s * 1000)
        print("index:", page.inner_text("#index-info"))
        for q in args.question:
            page.fill("#question", q)
            page.click("#search-btn")
            try:
                page.wait_for_function(
                    "document.querySelectorAll('#results li.result').length > 0"
                    " || document.querySelector('#results .empty')"
                    " || document.querySelector('#status.error')",
                    timeout=args.timeout_s * 1000,
                )
            except Exception as exc:
                print(f"FAIL {q!r}: timed out ({exc})")
                failures += 1
                continue
            hits = page.locator("#results li.result")
            n = hits.count()
            meta = page.inner_text("#search-meta")
            status = page.inner_text("#status")
            if n == 0:
                print(f"FAIL {q!r}: no passages (status: {status})")
                failures += 1
                continue
            if args.models == "on" and "rerank" not in meta:
                print(f"FAIL {q!r}: dense/rerank stage did not run ({meta})")
                failures += 1
                continue
            print(f"OK {q!r}: {n} passages | {meta}")
            for i in range(min(3, n)):
                print("   ", hits.nth(i).locator(".result-head").inner_text().replace("\n", " "))
        page.click("#tab-btn-evaluation")
        page.wait_for_selector("#tab-evaluation h2, #tab-evaluation .test-set", timeout=60_000)
        print("dashboard:", page.inner_text("#tab-evaluation")[:200].replace("\n", " "))
        browser.close()
    server.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
