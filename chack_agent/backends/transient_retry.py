"""Bounded retries of model requests that a gateway failed transiently.

A gateway answers 502, 503 or 504 (an upstream timeout, an overloaded or
restarting upstream) or drops the connection now and then; the same request a
few seconds later usually succeeds. One such answer must not end an agent run,
so a request that failed this way is sent again with jittered exponential
backoff, a bounded number of times and for a bounded total wait, so a run's
own runtime limit still holds. 4xx answers are never retried here (429 rate
limits are left to the callers that already handle them).

Where Chack sends the requests itself (the Gemini CLI's gateway relay) it uses
``TransientRetry``. The OpenAI-compatible API backends (gateway Responses,
OpenAI, LangGraph) give their OpenAI clients ``CLIENT_MAX_RETRIES`` retries,
which the OpenAI SDK spends on the same failures with its own jittered
backoff, and ``openai_http_client``/``openai_async_http_client`` log each
retried answer. The Codex and Claude Code CLIs retry these failures
themselves, bounded and with jitter; their backends set the allowance
explicitly.

Knobs: ``CHACK_TRANSIENT_RETRY_ATTEMPTS`` (attempts per request, including
the first; default ``DEFAULT_ATTEMPTS``) and
``CHACK_TRANSIENT_RETRY_MAX_WAIT_SECONDS`` (total backoff per request;
default ``DEFAULT_MAX_WAIT_SECONDS``).
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable

TRANSIENT_STATUSES = frozenset({502, 503, 504})

ATTEMPTS_ENV = "CHACK_TRANSIENT_RETRY_ATTEMPTS"
MAX_WAIT_ENV = "CHACK_TRANSIENT_RETRY_MAX_WAIT_SECONDS"
DEFAULT_ATTEMPTS = 4
DEFAULT_MAX_WAIT_SECONDS = 90.0
# Backoff before retry n (1-based) is drawn from [d/2, d], d = BASE * 2**(n-1), capped at MAX_DELAY.
BASE_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 30.0
# Retries the OpenAI SDK clients make after the first attempt.
CLIENT_MAX_RETRIES = DEFAULT_ATTEMPTS - 1


def _env_number(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.environ.get(name) or default)
    except ValueError:
        return default
    return max(minimum, value)


def configured_attempts() -> int:
    return int(_env_number(ATTEMPTS_ENV, DEFAULT_ATTEMPTS, 1))


def configured_max_wait_seconds() -> float:
    return _env_number(MAX_WAIT_ENV, DEFAULT_MAX_WAIT_SECONDS, 0.0)


def backoff_delay(retry: int, *, rng: Callable[[], float] = random.random) -> float:
    """Jittered exponential delay before retry ``retry`` (1-based)."""
    ceiling = min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** max(0, retry - 1)))
    return ceiling / 2 + rng() * ceiling / 2


def retry_after_seconds(value: Any) -> float | None:
    """A ``Retry-After`` header given in seconds, if it is one."""
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


class TransientRetry:
    """The retry allowance of one request.

    Call ``wait`` after each transient failure: it sleeps the backoff and
    returns True when another attempt may be made, or returns False at once
    when the attempts, the total wait or the ``deadline`` (a ``time.monotonic``
    value) are spent.
    """

    def __init__(
        self,
        label: str,
        *,
        logger: logging.Logger,
        attempts: int | None = None,
        max_wait_seconds: float | None = None,
        deadline: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._label = label
        self._logger = logger
        self.attempts = attempts if attempts is not None else configured_attempts()
        self._max_wait = max_wait_seconds if max_wait_seconds is not None else configured_max_wait_seconds()
        self._deadline = deadline
        self._sleep = sleep
        self._rng = rng
        self.attempt = 1
        self.waited = 0.0
        # When the current attempt was sent (``time.monotonic``), for its latency.
        self.attempt_started = time.monotonic()

    def wait(self, failure: str, *, retry_after: float | None = None) -> bool:
        if self.attempt >= self.attempts:
            self._logger.warning(
                "%s failed transiently (%s) on attempt %d/%d; no attempts left",
                self._label, failure, self.attempt, self.attempts,
            )
            return False
        delay = backoff_delay(self.attempt, rng=self._rng)
        if retry_after is not None:
            delay = max(delay, min(retry_after, MAX_DELAY_SECONDS))
        over_deadline = self._deadline is not None and time.monotonic() + delay >= self._deadline
        if self.waited + delay > self._max_wait or over_deadline:
            self._logger.warning(
                "%s failed transiently (%s) on attempt %d/%d; no time left to retry",
                self._label, failure, self.attempt, self.attempts,
            )
            return False
        self._logger.warning(
            "%s failed transiently (%s) on attempt %d/%d; retrying in %.1fs",
            self._label, failure, self.attempt, self.attempts, delay,
        )
        self._sleep(delay)
        self.waited += delay
        self.attempt += 1
        return True


def _log_retried_answer(logger: logging.Logger, response: Any) -> None:
    status = getattr(response, "status_code", 0)
    if status not in TRANSIENT_STATUSES and status not in (408, 429):
        return
    request = getattr(response, "request", None)
    headers = getattr(request, "headers", {}) or {}
    try:
        retries_so_far = int(headers.get("x-stainless-retry-count") or 0)
    except ValueError:
        retries_so_far = 0
    logger.warning(
        "Model API %s %s answered %s on attempt %d/%d%s",
        getattr(request, "method", ""),
        getattr(getattr(request, "url", None), "path", ""),
        status,
        retries_so_far + 1,
        CLIENT_MAX_RETRIES + 1,
        "; the client retries it" if retries_so_far < CLIENT_MAX_RETRIES else "",
    )


def openai_http_client(logger: logging.Logger) -> Any:
    """A sync httpx client for an OpenAI SDK client that logs each transient
    answer (status and attempt); the SDK logs the delay it waits."""
    from openai import DefaultHttpxClient

    def on_response(response: Any) -> None:
        _log_retried_answer(logger, response)

    return DefaultHttpxClient(event_hooks={"response": [on_response]})


def openai_async_http_client(logger: logging.Logger) -> Any:
    """The async counterpart of ``openai_http_client``."""
    from openai import DefaultAsyncHttpxClient

    async def on_response(response: Any) -> None:
        _log_retried_answer(logger, response)

    return DefaultAsyncHttpxClient(event_hooks={"response": [on_response]})
