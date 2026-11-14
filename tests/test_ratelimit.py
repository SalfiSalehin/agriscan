import pytest

from app.ratelimit import RateLimiter, client_ip
from app.settings import Settings


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_blocks_with_a_wait_time():
    clock = Clock()
    limiter = RateLimiter(3, window=60, clock=clock)
    assert [limiter.allow("a")[0] for _ in range(3)] == [True, True, True]
    allowed, wait = limiter.allow("a")
    assert allowed is False and wait == 60
    clock.now += 45
    assert limiter.allow("a") == (False, 15)


def test_window_slides_and_keys_are_independent():
    clock = Clock()
    limiter = RateLimiter(2, window=10, clock=clock)
    assert limiter.allow("a")[0] and limiter.allow("a")[0] and not limiter.allow("a")[0]
    assert limiter.allow("b")[0]  # another visitor is not affected
    clock.now += 10.1
    assert limiter.allow("a")[0]  # the old hits have left the window


def test_blocked_attempts_do_not_extend_the_block():
    clock = Clock()
    limiter = RateLimiter(1, window=10, clock=clock)
    assert limiter.allow("a")[0]
    for _ in range(50):
        assert not limiter.allow("a")[0]
    clock.now += 10.1
    assert limiter.allow("a")[0]


def test_memory_stays_bounded():
    limiter = RateLimiter(5, window=60, max_keys=100)
    for i in range(1000):
        limiter.allow(f"ip-{i}")
    assert len(limiter._hits) <= 100


def test_invalid_arguments():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(5, window=0)


def test_client_ip_ignores_the_header_unless_a_proxy_is_trusted():
    assert client_ip("1.2.3.4", "9.9.9.9", trust_proxy=False) == "1.2.3.4"
    assert client_ip(None, None, trust_proxy=False) == "unknown"
    # behind one proxy the last entry is the one the proxy saw; earlier ones can be forged
    assert client_ip("10.0.0.1", "6.6.6.6, 5.5.5.5", trust_proxy=True) == "5.5.5.5"
    assert client_ip("10.0.0.1", None, trust_proxy=True) == "10.0.0.1"
    assert client_ip("10.0.0.1", " ", trust_proxy=True) == "10.0.0.1"


def test_new_settings_from_env(monkeypatch):
    monkeypatch.delenv("AGRISCAN_RATE_PER_MIN", raising=False)
    assert Settings.from_env().rate_per_min == 30
    monkeypatch.setenv("AGRISCAN_RATE_PER_MIN", "0")
    monkeypatch.setenv("AGRISCAN_TRUST_PROXY", "yes")
    monkeypatch.setenv("AGRISCAN_DOCS", "off")
    settings = Settings.from_env()
    assert (settings.rate_per_min, settings.trust_proxy, settings.docs) == (0, True, False)
    monkeypatch.setenv("AGRISCAN_TRUST_PROXY", "maybe")
    with pytest.raises(ValueError, match="AGRISCAN_TRUST_PROXY"):
        Settings.from_env()
