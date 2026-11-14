"""A small in-memory rate limiter (sliding window per key), safe to share between threads.

One process, no database: enough for a single free-tier container. Behind several workers or
servers each one counts on its own, so the real limit is `limit` times the number of workers.
"""

import math
import threading
import time
from collections import deque


class RateLimiter:
    def __init__(
        self, limit: int, window: float = 60.0, max_keys: int = 10_000, clock=time.monotonic
    ):
        if limit < 1 or window <= 0 or max_keys < 1:
            raise ValueError("limit, window and max_keys must be positive")
        self.limit, self.window, self.max_keys, self._clock = limit, window, max_keys, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """(True, 0) and record the hit, or (False, seconds to wait) if `key` is over the limit."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                self._make_room(now)
                hits = self._hits[key] = deque()
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            return True, 0

    def _make_room(self, now: float) -> None:
        """Keep memory bounded: drop idle keys first, then the oldest-seen key."""
        if len(self._hits) < self.max_keys:
            return
        for key in [k for k, h in self._hits.items() if not h or h[-1] <= now - self.window]:
            del self._hits[key]
        while len(self._hits) >= self.max_keys:
            del self._hits[next(iter(self._hits))]


def client_ip(peer: str | None, forwarded_for: str | None, trust_proxy: bool) -> str:
    """The address to count requests against.

    Behind one trusted reverse proxy (Hugging Face, nginx) the proxy appends the caller's address
    to X-Forwarded-For, so the LAST entry is the one the proxy itself saw and cannot be faked by
    the caller. Without a trusted proxy the header is ignored: anyone could send any value.
    """
    if trust_proxy and forwarded_for:
        last = forwarded_for.split(",")[-1].strip()
        if last:
            return last
    return peer or "unknown"
