import contextvars
from typing import Any, Dict

_LOG_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "chack_log_context",
    default={},
)


def set_log_context(**kwargs):
    current = dict(_LOG_CONTEXT.get() or {})
    for key, value in kwargs.items():
        if value is None:
            continue
        current[str(key)] = value
    return _LOG_CONTEXT.set(current)


def update_log_context(**kwargs) -> None:
    current = dict(_LOG_CONTEXT.get() or {})
    for key, value in kwargs.items():
        if value is None:
            continue
        current[str(key)] = value
    _LOG_CONTEXT.set(current)


def reset_log_context(token) -> None:
    _LOG_CONTEXT.reset(token)


def clear_log_context() -> None:
    _LOG_CONTEXT.set({})


def current_log_context() -> Dict[str, Any]:
    return dict(_LOG_CONTEXT.get() or {})
