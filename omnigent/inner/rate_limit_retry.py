"""Jittered retry of turns that a harness fails with a provider rate limit.

Gateways with an organization-wide concurrency cap (e.g. ``429
CONCURRENT_REQUEST_LIMIT_EXCEEDED``) reject several sessions at once; retrying
them on a fixed schedule makes them collide again, so the wait uses full jitter
(:func:`omnigent.util.http_retry.full_jitter_delay`) within a bounded budget.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from omnigent.util.http_retry import full_jitter_delay, parse_retry_after

_logger = logging.getLogger(__name__)

ENV_MAX_RETRIES = "OMNIGENT_PI_RATE_LIMIT_MAX_RETRIES"
ENV_BUDGET_S = "OMNIGENT_PI_RATE_LIMIT_BUDGET_S"
ENV_BACKOFF_BASE_S = "OMNIGENT_PI_RATE_LIMIT_BACKOFF_BASE_S"
ENV_BACKOFF_MAX_S = "OMNIGENT_PI_RATE_LIMIT_BACKOFF_MAX_S"

CONCURRENT_REQUEST_LIMIT_CODE = "CONCURRENT_REQUEST_LIMIT_EXCEEDED"

# HTTP status in provider error text: "<provider> API error (429): ...", "HTTP 429",
# a leading "429: ...", or a JSON ``"status": 429`` field.
_STATUS_PATTERNS = (
    re.compile(r"\((\d{3})\)"),
    re.compile(r"\bHTTP(?:/[\d.]+)?\s+(\d{3})\b"),
    re.compile(r"^\s*(\d{3})\b"),
    re.compile(r"\"status(?:_?code)?\"\s*:\s*(\d{3})\b", re.IGNORECASE),
)
_ERROR_CODE = re.compile(r"\"(?:code|type)\"\s*:\s*\"([A-Za-z][A-Za-z0-9_.-]{0,63})\"")
# Quota, credit, and billing exhaustion: a 429 that waiting will not fix.
_EXHAUSTED = re.compile(
    r"insufficient_quota|quota exceeded|exceeded your current quota|out of credits"
    r"|insufficient credits|out of budget|billing",
    re.IGNORECASE,
)
_RETRY_AFTER_HINTS = (
    (
        re.compile(r"retry[-_ ]?after[-_ ]?ms\"?\s*[:=]\s*\"?(\d+(?:\.\d+)?)", re.IGNORECASE),
        1000.0,
    ),
    (re.compile(r"retry[-_ ]?after\"?\s*[:=]\s*\"?(\d+(?:\.\d+)?)", re.IGNORECASE), 1.0),
    (re.compile(r"try again in (\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", re.IGNORECASE), 1.0),
)


@dataclass(frozen=True)
class RateLimitSignal:
    """A turn failure that a later attempt may get past.

    :param reason: Short, loggable cause built only from the status and error
        code, e.g. ``"HTTP 429 CONCURRENT_REQUEST_LIMIT_EXCEEDED"``.
    :param retry_after_s: Server-requested minimum wait found in the error
        text, or ``None``.
    """

    reason: str
    retry_after_s: float | None = None


@dataclass(frozen=True)
class RateLimitRetryConfig:
    """Budget and backoff shape for rate-limit retries.

    :param max_retries: Retries after the first attempt; ``0`` disables them.
    :param budget_s: Wall-clock limit from the first rate-limited failure,
        including the attempts themselves.
    :param backoff_base_s: Full-jitter window for the first retry.
    :param backoff_max_s: Largest full-jitter window.
    """

    max_retries: int = 10
    budget_s: float = 600.0
    backoff_base_s: float = 5.0
    backoff_max_s: float = 120.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        for name in ("budget_s", "backoff_base_s", "backoff_max_s"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be > 0, got {value}")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> RateLimitRetryConfig:
        """Read overrides from ``OMNIGENT_PI_RATE_LIMIT_*``; invalid values keep the default.

        :param env: Environment to read, e.g. ``os.environ``.
        """
        default = cls()
        try:
            return cls(
                max_retries=int(_env_float(env, ENV_MAX_RETRIES, default.max_retries)),
                budget_s=_env_float(env, ENV_BUDGET_S, default.budget_s),
                backoff_base_s=_env_float(env, ENV_BACKOFF_BASE_S, default.backoff_base_s),
                backoff_max_s=_env_float(env, ENV_BACKOFF_MAX_S, default.backoff_max_s),
            )
        except ValueError as exc:
            _logger.warning("Ignoring invalid rate-limit retry settings: %s", exc)
            return default


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return default
    if not math.isfinite(value):
        _logger.warning("Ignoring non-finite %s=%r", name, raw)
        return default
    return value


def _status_code(message: str) -> int | None:
    for pattern in _STATUS_PATTERNS:
        match = pattern.search(message)
        if match is not None:
            return int(match.group(1))
    return None


def _retry_after_hint(message: str) -> float | None:
    for pattern, divisor in _RETRY_AFTER_HINTS:
        match = pattern.search(message)
        if match is not None:
            value = parse_retry_after(match.group(1))
            if value is not None:
                return value / divisor
    return None


def classify_rate_limit(message: str) -> RateLimitSignal | None:
    """Return a :class:`RateLimitSignal` when *message* reports a retryable rate limit.

    Matches HTTP 429 and ``CONCURRENT_REQUEST_LIMIT_EXCEEDED``. Any other
    status (e.g. 400, 401, 402, 403, 404) and quota or billing exhaustion
    return ``None`` so they fail fast.

    :param message: Provider error text, e.g. ``'databricks-openai API error
        (429): {"code":"CONCURRENT_REQUEST_LIMIT_EXCEEDED", ...}'``.
    """
    status = _status_code(message)
    concurrency = CONCURRENT_REQUEST_LIMIT_CODE in message.upper()
    if not concurrency and status != 429:
        return None
    if status is not None and status != 429:
        return None
    if _EXHAUSTED.search(message):
        return None
    code: str | None = CONCURRENT_REQUEST_LIMIT_CODE if concurrency else None
    if code is None and (code_match := _ERROR_CODE.search(message)) is not None:
        code = code_match.group(1)
    reason = f"HTTP 429 {code}" if code else "HTTP 429"
    return RateLimitSignal(reason=reason, retry_after_s=_retry_after_hint(message))


class RateLimitBackoff:
    """Attempt count, elapsed budget, and next delay for one turn's retries.

    :param config: Budget and backoff shape.
    :param rng: Random source for the jitter; tests pass a seeded one.
    :param clock: Monotonic clock in seconds; tests pass a fake.
    """

    def __init__(
        self,
        config: RateLimitRetryConfig,
        *,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._rng = rng
        self._clock = clock
        self._started = clock()
        self.retries = 0

    def next_delay(self, signal: RateLimitSignal) -> float | None:
        """Return the wait before the next retry, or ``None`` once the budget is spent.

        A server ``Retry-After`` is the minimum wait, with full jitter added on
        top. A hint past the remaining budget ends the retries rather than
        retrying early.
        """
        if self.retries >= self._config.max_retries:
            return None
        remaining = self._config.budget_s - (self._clock() - self._started)
        retry_after = signal.retry_after_s or 0.0
        if remaining <= 0 or retry_after > remaining:
            return None
        jitter = full_jitter_delay(
            self.retries,
            base=self._config.backoff_base_s,
            cap=self._config.backoff_max_s,
            rng=self._rng,
        )
        self.retries += 1
        return min(retry_after + jitter, remaining)
