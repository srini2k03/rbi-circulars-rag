"""Client-side limits for free-tier APIs, per model: requests/minute, requests/day and an
optional tokens/minute budget.

The per-day counter is persisted (``<state_dir>/usage-YYYY-MM-DD.json``, UTC) so separate
processes and resumed workflow runs share one budget.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rbirag.generation.llm import QuotaExhaustedError


class RequestLimiter:
    def __init__(
        self,
        *,
        rpm: float,
        rpd: int | Callable[[str], int],
        tpm: Callable[[str], int] | None = None,
        state_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], str] = lambda: datetime.now(UTC).date().isoformat(),
    ) -> None:
        self.min_interval = 60.0 / rpm
        self._rpd = rpd
        self._tpm = tpm
        self._tokens: dict[str, deque[list[float]]] = {}  # model -> [[time, tokens], ...]
        self.state_dir = state_dir
        self._clock = clock
        self._sleep = sleep
        self._today = today
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._memory_counts: dict[str, dict[str, int]] = {}

    # -- daily counter -----------------------------------------------------------------
    def _path(self, day: str) -> Path | None:
        return self.state_dir / f"usage-{day}.json" if self.state_dir else None

    def _load(self, day: str) -> dict[str, int]:
        path = self._path(day)
        if path is None:
            return self._memory_counts.setdefault(day, {})
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): int(v) for k, v in data.items()}

    def _save(self, day: str, counts: dict[str, int]) -> None:
        path = self._path(day)
        if path is None:
            self._memory_counts[day] = counts
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(counts, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def rpd_for(self, key: str) -> int:
        return self._rpd(key) if callable(self._rpd) else self._rpd

    def tpm_for(self, key: str) -> int:
        return self._tpm(key) if self._tpm is not None else 0

    def used_today(self, key: str) -> int:
        return self._load(self._today()).get(key, 0)

    def remaining_today(self, key: str) -> int:
        return max(0, self.rpd_for(key) - self.used_today(key))

    # -- acquire -----------------------------------------------------------------------
    def acquire(self, key: str, tokens: int = 0) -> None:
        """Block until a request for ``key`` (a model id) estimated at ``tokens`` may be
        sent, and count it."""
        with self._lock:
            day = self._today()
            counts = self._load(day)
            rpd = self.rpd_for(key)
            if counts.get(key, 0) >= rpd:
                raise QuotaExhaustedError(
                    f"daily request budget for {key} reached ({rpd}/day); "
                    "rerun later - cached responses will be reused"
                )
            self._wait_for_tokens(key, tokens)
            now = self._clock()
            last = self._last.get(key)
            if last is not None:
                wait = self.min_interval - (now - last)
                if wait > 0:
                    self._sleep(wait)
                    now = self._clock()
            self._last[key] = now
            if self.tpm_for(key):
                self._tokens.setdefault(key, deque()).append([now, float(tokens)])
            counts[key] = counts.get(key, 0) + 1
            self._save(day, counts)

    # -- tokens per minute -------------------------------------------------------------
    def _window(self, key: str, now: float) -> deque[list[float]]:
        window = self._tokens.setdefault(key, deque())
        while window and now - window[0][0] >= 60.0:
            window.popleft()
        return window

    def _wait_for_tokens(self, key: str, tokens: int) -> None:
        budget = self.tpm_for(key)
        if not budget or tokens <= 0:
            return
        while True:
            now = self._clock()
            window = self._window(key, now)
            used = sum(t for _, t in window)
            if not window or used + tokens <= budget:
                return  # (a single oversized request is allowed once the window is empty)
            # wait until the oldest entry leaves the 60 s window
            self._sleep(max(0.05, 60.0 - (now - window[0][0])))

    def settle(self, key: str, actual_tokens: int) -> None:
        """Replace the estimate of the most recent request for ``key`` with actual usage."""
        with self._lock:
            window = self._tokens.get(key)
            if window:
                window[-1][1] = float(actual_tokens)
