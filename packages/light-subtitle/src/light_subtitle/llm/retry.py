"""Shared retry drivers for LLM calls.

Two cross-cutting loop shapes used by multiple pipeline steps:

- :func:`chat_with_retry` — call-until-success with per-attempt logging
  and exponential backoff; re-raises the last error on exhaustion.
- :func:`generate_with_feedback` — generate → parse → validate loop that
  feeds validation problems back into the next attempt's payload.

Both drivers take callables so each call site keeps its own prompt
building, log wording, and validation rules.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from ..usage.tracker import merge_token_usage


def chat_with_retry[T](
    call: Callable[[], T],
    *,
    max_retries: int = 3,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], float | None] | None = None,
) -> T:
    """Run *call* until it succeeds, retrying up to *max_retries* times.

    Only *retry_exceptions* are retried; anything else propagates
    immediately.  *on_retry(attempt, exc)* is invoked before each retry
    (0-based *attempt*) and returns the backoff delay in seconds — a
    falsy value means no sleep.  Without *on_retry* the default backoff
    is ``2 ** attempt`` seconds.  Re-raises the last error after the
    final attempt fails.
    """
    last_error: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return call()
        except retry_exceptions as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = on_retry(attempt, e) if on_retry is not None else 2**attempt
                if delay:
                    time.sleep(delay)
    raise last_error  # type: ignore[misc]


@dataclass
class FeedbackAttempt[T]:
    """Outcome of one :func:`generate_with_feedback` round.

    Attributes:
        usage: token usage of this round (merged by the driver).
        value: success value — the loop stops and returns it.
        feedback: failure description fed into the next round's payload.
        final: partial-success value returned when rounds are exhausted.
    """

    usage: dict | None = None
    value: T | None = None
    feedback: str = ""
    final: T | None = None


def generate_with_feedback[T](
    attempt_fn: Callable[[str, int], FeedbackAttempt[T]],
    *,
    max_attempts: int,
) -> tuple[T | None, dict | None]:
    """Drive a generate → validate loop with feedback retries.

    Each round calls *attempt_fn(feedback, attempt)* where *feedback* is
    the previous round's failure description (empty on the first round)
    and *attempt* is the 0-based round index.  The callable performs one
    LLM round plus parsing/validation and reports via
    :class:`FeedbackAttempt`; logging stays inside the callable so each
    site keeps its own wording.

    Returns ``(value, usage)`` on success, ``(final, usage)`` when the
    loop is exhausted with a partial success, otherwise ``(None, usage)``.
    """
    total_usage: dict = {}
    feedback = ""
    final: T | None = None
    for attempt in range(max_attempts):
        outcome = attempt_fn(feedback, attempt)
        if outcome.usage:
            merge_token_usage(total_usage, outcome.usage)
        if outcome.value is not None:
            return outcome.value, total_usage or None
        feedback = outcome.feedback
        if outcome.final is not None:
            final = outcome.final
    return final, total_usage or None
