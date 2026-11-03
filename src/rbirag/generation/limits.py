"""Client-side request limits for free-tier APIs: requests/minute and requests/day.

The per-day counter is persisted (``<state_dir>/usage-YYYY-MM-DD.json``, UTC) so separate
processes and resumed workflow runs share one budget.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rbirag.generation.llm import QuotaExhaustedError


class RequestLimiter:
    def __init__(
        self,
        *,
        rpm: float,
        rpd: int,
        state_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], str] = lambda: datetime.now(UTC).date().isoformat(),
    ) -> None:
        self.min_interval = 60.0 / rpm
        self.rpd = rpd
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

    def used_today(self, key: str) -> int:
        return self._load(self._today()).get(key, 0)

    def remaining_today(self, key: str) -> int:
        return max(0, self.rpd - self.used_today(key))

    # -- acquire -----------------------------------------------------------------------
    def acquire(self, key: str) -> None:
        """Block until a request for ``key`` (e.g. a model id) may be sent, and count it."""
        with self._lock:
            day = self._today()
            counts = self._load(day)
            if counts.get(key, 0) >= self.rpd:
                raise QuotaExhaustedError(
                    f"daily request budget for {key} reached ({self.rpd}/day); "
                    "rerun later - cached responses will be reused"
                )
            now = self._clock()
            last = self._last.get(key)
            if last is not None:
                wait = self.min_interval - (now - last)
                if wait > 0:
                    self._sleep(wait)
                    now = self._clock()
            self._last[key] = now
            counts[key] = counts.get(key, 0) + 1
            self._save(day, counts)
