"""Rate-limit classification and full-jitter backoff budget."""

from __future__ import annotations

import random

import pytest

from omnigent.inner.rate_limit_retry import (
    ENV_BACKOFF_BASE_S,
    ENV_BACKOFF_MAX_S,
    ENV_BUDGET_S,
    ENV_MAX_RETRIES,
    RateLimitBackoff,
    RateLimitRetryConfig,
    RateLimitSignal,
    classify_rate_limit,
)

_CONCURRENCY_BODY = (
    '{"code":"CONCURRENT_REQUEST_LIMIT_EXCEEDED",'
    '"message":"Organization has reached its maximum number of concurrent requests."}'
)


class _BoundRng(random.Random):
    """Returns the low or high end of every ``uniform`` window."""

    def __init__(self, *, high: bool) -> None:
        super().__init__(0)
        self._high = high

    def uniform(self, a: float, b: float) -> float:
        return b if self._high else a


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        pytest.param(
            f"databricks-openai API error (429): {_CONCURRENCY_BODY}",
            "HTTP 429 CONCURRENT_REQUEST_LIMIT_EXCEEDED",
            id="gateway-concurrency",
        ),
        pytest.param(
            f"upstream rejected: {_CONCURRENCY_BODY}",
            "HTTP 429 CONCURRENT_REQUEST_LIMIT_EXCEEDED",
            id="concurrency-code-without-status",
        ),
        pytest.param(
            'openai API error (429): {"type":"rate_limit_error"}',
            "HTTP 429 rate_limit_error",
            id="generic-429-with-code",
        ),
        pytest.param("429: Too Many Requests", "HTTP 429", id="bare-429"),
        pytest.param("HTTP 429 Too Many Requests", "HTTP 429", id="http-429"),
    ],
)
def test_classify_rate_limit_matches_429(message: str, reason: str) -> None:
    assert classify_rate_limit(message) == RateLimitSignal(reason=reason)


@pytest.mark.parametrize(
    "message",
    [
        pytest.param('provider API error (402): {"message":"Out of credits"}', id="402"),
        pytest.param('provider API error (400): {"message":"bad request"}', id="400"),
        pytest.param('provider API error (401): {"message":"unauthorized"}', id="401"),
        pytest.param('provider API error (403): {"message":"forbidden"}', id="403"),
        pytest.param('provider API error (404): {"message":"no such model"}', id="404"),
        pytest.param("503 Service Unavailable", id="503-left-to-pi"),
        pytest.param('API error (429): {"code":"insufficient_quota"}', id="quota-429"),
        pytest.param("API error (429): monthly billing limit reached", id="billing-429"),
        pytest.param("Rate limited", id="no-status"),
        pytest.param(f"API error (402): {_CONCURRENCY_BODY}", id="402-wins-over-code"),
    ],
)
def test_classify_rate_limit_rejects_other_failures(message: str) -> None:
    assert classify_rate_limit(message) is None


@pytest.mark.parametrize(
    ("message", "retry_after_s"),
    [
        pytest.param('API error (429): {"retry_after": 30}', 30.0, id="json-seconds"),
        pytest.param("API error (429): retry-after: 12.5", 12.5, id="header-text"),
        pytest.param('API error (429): {"retry_after_ms": 1500}', 1.5, id="json-ms"),
        pytest.param("429: Please try again in 20 seconds.", 20.0, id="prose"),
        pytest.param('API error (429): {"retry_after": 0}', None, id="zero-ignored"),
    ],
)
def test_classify_rate_limit_reads_retry_after_hint(
    message: str, retry_after_s: float | None
) -> None:
    rate_limit = classify_rate_limit(message)
    assert rate_limit is not None
    assert rate_limit.retry_after_s == retry_after_s


def test_backoff_full_jitter_windows_double_up_to_the_cap() -> None:
    """Each delay is drawn from ``[0, min(cap, base * 2**retry)]``."""
    config = RateLimitRetryConfig(
        max_retries=6, budget_s=10_000, backoff_base_s=5, backoff_max_s=120
    )
    rate_limit = RateLimitSignal(reason="HTTP 429")
    windows = [5.0, 10.0, 20.0, 40.0, 80.0, 120.0]

    highs = RateLimitBackoff(config, rng=_BoundRng(high=True), clock=_FakeClock())
    lows = RateLimitBackoff(config, rng=_BoundRng(high=False), clock=_FakeClock())
    assert [highs.next_delay(rate_limit) for _ in windows] == windows
    assert [lows.next_delay(rate_limit) for _ in windows] == [0.0] * len(windows)

    seeded = RateLimitBackoff(config, rng=random.Random(1234), clock=_FakeClock())
    delays = [seeded.next_delay(rate_limit) for _ in windows]
    for delay, window in zip(delays, windows, strict=True):
        assert delay is not None
        assert 0.0 <= delay <= window
    assert len(set(delays)) == len(delays), "full jitter should not repeat a fixed schedule"


def test_backoff_retry_after_is_the_minimum_with_jitter_on_top() -> None:
    config = RateLimitRetryConfig(backoff_base_s=5, backoff_max_s=120)
    rate_limit = RateLimitSignal(reason="HTTP 429", retry_after_s=60.0)

    low = RateLimitBackoff(config, rng=_BoundRng(high=False), clock=_FakeClock())
    high = RateLimitBackoff(config, rng=_BoundRng(high=True), clock=_FakeClock())

    assert low.next_delay(rate_limit) == 60.0
    assert high.next_delay(rate_limit) == 65.0


def test_backoff_stops_after_max_retries() -> None:
    config = RateLimitRetryConfig(max_retries=2)
    backoff = RateLimitBackoff(config, rng=random.Random(0), clock=_FakeClock())
    rate_limit = RateLimitSignal(reason="HTTP 429")

    assert backoff.next_delay(rate_limit) is not None
    assert backoff.next_delay(rate_limit) is not None
    assert backoff.next_delay(rate_limit) is None
    assert backoff.retries == 2


def test_backoff_stops_when_the_time_budget_is_spent() -> None:
    """The budget counts the attempts too, and a delay never overshoots it."""
    clock = _FakeClock()
    config = RateLimitRetryConfig(budget_s=100, backoff_base_s=80, backoff_max_s=80)
    backoff = RateLimitBackoff(config, rng=_BoundRng(high=True), clock=clock)
    rate_limit = RateLimitSignal(reason="HTTP 429")

    assert backoff.next_delay(rate_limit) == 80.0
    clock.now += 90.0  # the wait plus a short attempt
    assert backoff.next_delay(rate_limit) == 10.0
    clock.now += 10.0
    assert backoff.next_delay(rate_limit) is None


def test_backoff_gives_up_when_retry_after_exceeds_the_budget() -> None:
    config = RateLimitRetryConfig(budget_s=30)
    backoff = RateLimitBackoff(config, rng=random.Random(0), clock=_FakeClock())

    assert backoff.next_delay(RateLimitSignal(reason="HTTP 429", retry_after_s=60.0)) is None


def test_config_from_env_reads_overrides_and_keeps_defaults_for_bad_values() -> None:
    assert RateLimitRetryConfig.from_env({}) == RateLimitRetryConfig()
    assert RateLimitRetryConfig.from_env(
        {
            ENV_MAX_RETRIES: "3",
            ENV_BUDGET_S: "90",
            ENV_BACKOFF_BASE_S: "1.5",
            ENV_BACKOFF_MAX_S: "30",
        }
    ) == RateLimitRetryConfig(max_retries=3, budget_s=90, backoff_base_s=1.5, backoff_max_s=30)
    assert RateLimitRetryConfig.from_env({ENV_MAX_RETRIES: "0"}).max_retries == 0
    assert RateLimitRetryConfig.from_env({ENV_BUDGET_S: "soon"}) == RateLimitRetryConfig()
    assert RateLimitRetryConfig.from_env({ENV_BUDGET_S: "inf"}) == RateLimitRetryConfig()
    assert RateLimitRetryConfig.from_env({ENV_MAX_RETRIES: "-1"}) == RateLimitRetryConfig()
