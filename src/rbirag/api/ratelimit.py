"""A per-client token bucket (in memory; one process). Enough for a single-instance demo
API; a multi-instance deployment would move this to the gateway or a shared store."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict


class RateLimiter:
    def __init__(self, per_minute: int, *, burst: int | None = None, max_clients: int = 10_000):
        self.rate = per_minute / 60.0
        self.capacity = float(burst or max(1, per_minute))
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def check(self, client: str, *, now: float | None = None) -> float:
        """0 if the request may proceed, else the seconds to wait before retrying."""
        if not self.enabled:
            return 0.0
        t = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.pop(client, (self.capacity, t))
            tokens = min(self.capacity, tokens + (t - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, t)
                wait = 0.0
            else:
                self._buckets[client] = (tokens, t)
                wait = (1.0 - tokens) / self.rate
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        return wait
