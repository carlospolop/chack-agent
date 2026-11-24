"""Listeners told when an LLM gateway shows pressure that a backend absorbed.

A backend that retries a gateway failure inside a run (the Gemini gateway
relay) hides it from whoever launches agents: the run may still succeed. An
embedding application that limits how many agents run at once can register a
listener here to slow launches while the gateway struggles.

``notify(kind, detail)`` calls every listener with one of ``TIMEOUT`` (a call
got no answer within its limit), ``RATE_LIMITED`` (HTTP 429) or ``UNAVAILABLE``
(a 5xx answer or a dropped connection). A listener that raises is logged and
skipped; it never affects the call.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

TIMEOUT = "timeout"
RATE_LIMITED = "rate_limited"
UNAVAILABLE = "unavailable"

Listener = Callable[[str, str], None]

_LOGGER = logging.getLogger(__name__)
_lock = threading.Lock()
_listeners: list[Listener] = []


def add_listener(listener: Listener) -> None:
    """Register ``listener(kind, detail)``; registering the same one twice keeps one."""
    with _lock:
        if listener not in _listeners:
            _listeners.append(listener)


def remove_listener(listener: Listener) -> None:
    with _lock:
        if listener in _listeners:
            _listeners.remove(listener)


def notify(kind: str, detail: str = "") -> None:
    with _lock:
        listeners = list(_listeners)
    for listener in listeners:
        try:
            listener(kind, str(detail)[:200])
        except Exception:
            _LOGGER.warning("Gateway pressure listener %r failed", listener, exc_info=True)
