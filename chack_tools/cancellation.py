from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional


_CURRENT_CANCEL_EVENT: contextvars.ContextVar[Optional[threading.Event]] = contextvars.ContextVar(
    "chack_current_cancel_event",
    default=None,
)
_REGISTRY_LOCK = threading.Lock()
_PROCESS_REGISTRY: dict[int, dict[int, tuple[Any, Callable[[Any], None]]]] = {}


@dataclass(frozen=True)
class ProcessRegistration:
    event: threading.Event
    process_id: int


def set_cancellation_event(event: threading.Event):
    return _CURRENT_CANCEL_EVENT.set(event)


def reset_cancellation_event(token: contextvars.Token) -> None:
    _CURRENT_CANCEL_EVENT.reset(token)


def current_cancellation_event() -> Optional[threading.Event]:
    return _CURRENT_CANCEL_EVENT.get()


def cancellation_requested() -> bool:
    event = current_cancellation_event()
    return bool(event is not None and event.is_set())


def register_process(process: Any, terminate: Callable[[Any], None]) -> Optional[ProcessRegistration]:
    event = current_cancellation_event()
    if event is None:
        return None
    key = id(event)
    process_id = id(process)
    with _REGISTRY_LOCK:
        _PROCESS_REGISTRY.setdefault(key, {})[process_id] = (process, terminate)
        already_cancelled = event.is_set()
    if already_cancelled:
        terminate(process)
    return ProcessRegistration(event=event, process_id=process_id)


def unregister_process(registration: Optional[Any]) -> None:
    if registration is None:
        return
    if isinstance(registration, ProcessRegistration):
        key = id(registration.event)
        process_id = registration.process_id
        with _REGISTRY_LOCK:
            processes = _PROCESS_REGISTRY.get(key)
            if not processes:
                return
            processes.pop(process_id, None)
            if not processes:
                _PROCESS_REGISTRY.pop(key, None)
        return
    event = registration if isinstance(registration, threading.Event) else None
    if event is None:
        return
    with _REGISTRY_LOCK:
        _PROCESS_REGISTRY.pop(id(event), None)


def request_cancel(event: Optional[threading.Event]) -> bool:
    if event is None:
        return False
    event.set()
    with _REGISTRY_LOCK:
        registered = list((_PROCESS_REGISTRY.get(id(event)) or {}).values())
    if not registered:
        return False
    cancelled = False
    for process, terminate in registered:
        try:
            terminate(process)
            cancelled = True
        except Exception:
            pass
    return cancelled
