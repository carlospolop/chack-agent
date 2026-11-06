"""Observe the token usage and estimated cost of agent runs while they run.

A caller wraps ``Chack.run`` in ``observe(callback)``; every model call of that
run (and of agents it starts) is then reported to ``callback`` as a
``UsageEvent`` as soon as the backend knows its usage. Backends that only learn
usage when their CLI exits report once, at the end. Events carry the delta since
the previous event, priced with the run's pricing table (``cost_usd`` is None
for an unpriced model). A callback that raises is logged and ignored.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class UsageEvent:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    cache_write_tokens: int
    cost_usd: Optional[float]


UsageObserver = Callable[[UsageEvent], None]

_OBSERVER: contextvars.ContextVar[Optional[UsageObserver]] = contextvars.ContextVar(
    "chack_usage_observer", default=None
)


def current_observer() -> Optional[UsageObserver]:
    return _OBSERVER.get()


def set_usage_observer(callback: Optional[UsageObserver]):
    return _OBSERVER.set(callback)


def reset_usage_observer(token) -> None:
    _OBSERVER.reset(token)


@contextlib.contextmanager
def observe(callback: Optional[UsageObserver]) -> Iterator[None]:
    token = set_usage_observer(callback)
    try:
        yield
    finally:
        reset_usage_observer(token)
