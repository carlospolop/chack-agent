from __future__ import annotations

import contextvars
from typing import Callable, Optional


LiveCostCallback = Callable[[str, int, int, int, int], None]


_ACTIVE_LIVE_COST_CALLBACK: contextvars.ContextVar[Optional[LiveCostCallback]] = (
    contextvars.ContextVar("chack_live_cost_callback", default=None)
)
# Same signature; set for every model call of a run whose caller observes usage.
_ACTIVE_USAGE_SINK: contextvars.ContextVar[Optional[LiveCostCallback]] = (
    contextvars.ContextVar("chack_usage_sink", default=None)
)


class LiveCostLimitExceeded(TimeoutError):
    """Raised when the live spend watchdog determines the budget is exhausted."""


def set_active_live_cost_callback(callback: Optional[LiveCostCallback]):
    return _ACTIVE_LIVE_COST_CALLBACK.set(callback)


def reset_active_live_cost_callback(token) -> None:
    _ACTIVE_LIVE_COST_CALLBACK.reset(token)


def set_active_usage_sink(sink: Optional[LiveCostCallback]):
    return _ACTIVE_USAGE_SINK.set(sink)


def reset_active_usage_sink(token) -> None:
    _ACTIVE_USAGE_SINK.reset(token)


def report_live_usage(
    model_name: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> None:
    values = (
        str(model_name or ""),
        max(0, int(prompt_tokens or 0)),
        max(0, int(completion_tokens or 0)),
        max(0, int(cached_prompt_tokens or 0)),
        max(0, int(cache_write_tokens or 0)),
    )
    sink = _ACTIVE_USAGE_SINK.get()
    if sink is not None:
        # Observed before the budget check, which may raise to stop the run.
        sink(*values)
    callback = _ACTIVE_LIVE_COST_CALLBACK.get()
    if callback is None:
        return
    callback(*values)
