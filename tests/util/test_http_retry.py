"""``Retry-After`` parsing for bounded HTTP retries."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from omnigent.util import http_retry

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_FALLBACK_S = 1.5
_MAX_DELAY_S = 10.0


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        pytest.param(None, _FALLBACK_S, id="missing"),
        pytest.param("3", 3.0, id="seconds"),
        pytest.param("2.5", 2.5, id="fractional-seconds"),
        pytest.param("30", _MAX_DELAY_S, id="seconds-capped"),
        pytest.param("0", _FALLBACK_S, id="zero"),
        pytest.param("-1", _FALLBACK_S, id="negative"),
        pytest.param("nan", _FALLBACK_S, id="nan"),
        pytest.param("inf", _FALLBACK_S, id="infinity"),
        pytest.param("soon", _FALLBACK_S, id="malformed"),
        pytest.param("Thu, 01 Jan 2026 12:00:04 GMT", 4.0, id="http-date"),
        pytest.param("Thu, 01 Jan 2026 12:00:04 -0000", 4.0, id="http-date-naive"),
        pytest.param("Thu, 01 Jan 2026 12:05:00 GMT", _MAX_DELAY_S, id="http-date-capped"),
        pytest.param("Thu, 01 Jan 2026 12:00:00 GMT", _FALLBACK_S, id="http-date-now"),
        pytest.param("Thu, 01 Jan 2026 11:59:00 GMT", _FALLBACK_S, id="http-date-past"),
    ],
)
def test_bounded_retry_after_seconds(
    monkeypatch: pytest.MonkeyPatch,
    header: str | None,
    expected: float,
) -> None:
    """Usable hints are capped; absent, unusable, or non-positive hints fall back."""
    monkeypatch.setattr(http_retry, "datetime", SimpleNamespace(now=lambda tz: _NOW))
    headers = {} if header is None else {"Retry-After": header}
    response = httpx.Response(429, headers=headers)

    delay = http_retry.bounded_retry_after_seconds(
        response, fallback=_FALLBACK_S, max_delay=_MAX_DELAY_S
    )

    assert delay == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, 1.0),
        ("bad", 1.0),
        ("0", 1.0),
        ("nan", 1.0),
        ("0.1", 1.0),
        ("3", 3.5),
        ("Thu, 01 Jan 2026 12:00:04 GMT", 4.5),
        ("10", 10.0),
        ("11", None),
        ("Thu, 01 Jan 2026 12:05:00 GMT", None),
    ],
)
def test_jitter_respects_server_hint_and_wait_limit(
    monkeypatch: pytest.MonkeyPatch,
    header: str | None,
    expected: float | None,
) -> None:
    """The longest jitter never retries early or exceeds the wait budget."""
    monkeypatch.setattr(http_retry, "datetime", SimpleNamespace(now=lambda tz: _NOW))
    monkeypatch.setattr(http_retry.random, "uniform", lambda low, high: high)
    response = httpx.Response(429, headers={} if header is None else {"Retry-After": header})

    assert http_retry.jittered_retry_delay(response, backoff=0.5, max_delay=10.0) == expected


def test_transport_failure_retry_uses_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests without a response spread their exponential backoff delays."""
    monkeypatch.setattr(http_retry.random, "uniform", lambda low, high: high / 2)

    assert http_retry.jittered_retry_delay(None, backoff=2.0, max_delay=10.0) == 3.0


@pytest.mark.parametrize(
    ("retry_index", "window"),
    [(0, 5.0), (1, 10.0), (3, 40.0), (5, 120.0), (10_000, 120.0), (-1, 5.0)],
)
def test_full_jitter_delay_spans_the_capped_exponential_window(
    retry_index: int, window: float
) -> None:
    """Draws cover ``[0, min(cap, base * 2**retry_index)]``, even for huge indexes."""
    bounds: list[tuple[float, float]] = []

    class _RecordingRng(random.Random):
        def uniform(self, a: float, b: float) -> float:
            bounds.append((a, b))
            return b

    delay = http_retry.full_jitter_delay(retry_index, base=5.0, cap=120.0, rng=_RecordingRng())

    assert delay == window
    assert bounds == [(0.0, window)]
    seeded = random.Random(42)
    draws = [
        http_retry.full_jitter_delay(retry_index, base=5.0, cap=120.0, rng=seeded)
        for _ in range(50)
    ]
    assert all(0.0 <= draw <= window for draw in draws)
