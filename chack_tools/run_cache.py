"""Small bounded caches scoped to one agent run.

A cache lives in the closure of a tool built for a run, and every key is also
prefixed with the active session, so an entry is never served to another run
or session even when one tool list serves several of them.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional

from .task_steps_manager_state import current_session_id
from .telemetry import current_log_context


def current_run_scope() -> str:
    return str(current_session_id() or current_log_context().get("session_id") or "")


class RunScopedCache:
    """LRU map of at most ``max_entries`` items, keyed per run scope."""

    def __init__(self, max_entries: int = 64):
        self.max_entries = max(1, int(max_entries))
        self._items: "OrderedDict[tuple[str, Hashable], Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        scoped = (current_run_scope(), key)
        with self._lock:
            if scoped not in self._items:
                return None
            self._items.move_to_end(scoped)
            return self._items[scoped]

    def put(self, key: Hashable, value: Any) -> None:
        scoped = (current_run_scope(), key)
        with self._lock:
            self._items[scoped] = value
            self._items.move_to_end(scoped)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
