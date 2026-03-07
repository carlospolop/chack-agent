from __future__ import annotations

import contextvars
from typing import Any, Callable, Optional


LimitEventCallback = Callable[[str, dict[str, Any]], None]


_ACTIVE_LIMIT_EVENT_CALLBACK: contextvars.ContextVar[Optional[LimitEventCallback]] = (
    contextvars.ContextVar("chack_limit_event_callback", default=None)
)


def set_active_limit_event_callback(callback: Optional[LimitEventCallback]):
    return _ACTIVE_LIMIT_EVENT_CALLBACK.set(callback)


def reset_active_limit_event_callback(token) -> None:
    _ACTIVE_LIMIT_EVENT_CALLBACK.reset(token)


def emit_limit_reached(limit_type: str, payload: Optional[dict[str, Any]] = None) -> None:
    callback = _ACTIVE_LIMIT_EVENT_CALLBACK.get()
    if callback is None:
        return
    callback(str(limit_type or "").strip().lower(), dict(payload or {}))
