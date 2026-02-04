from __future__ import annotations

import contextvars
import threading
from collections import Counter
from typing import Dict, Tuple


_ACTIVE_USAGE_SESSION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chack_tool_usage_session_id",
    default=None,
)


class ToolUsageStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Counter[str]] = {}
        self._token_sessions: Dict[str, Dict[str, Tuple[int, int, int]]] = {}

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = Counter()
            self._token_sessions[session_id] = {}

    def add(self, tool_name: str, count: int = 1, session_id: str | None = None) -> None:
        sid = session_id or _ACTIVE_USAGE_SESSION_ID.get()
        if not sid or not tool_name:
            return
        with self._lock:
            counter = self._sessions.setdefault(sid, Counter())
            counter[tool_name] += max(1, int(count or 1))

    def snapshot(self, session_id: str) -> Counter[str]:
        with self._lock:
            return Counter(self._sessions.get(session_id, Counter()))

    def add_tokens(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_prompt_tokens: int = 0,
        session_id: str | None = None,
    ) -> None:
        sid = session_id or _ACTIVE_USAGE_SESSION_ID.get()
        if not sid:
            return
        model = str(model_name or "").strip()
        if not model:
            return
        prompt = max(0, int(prompt_tokens or 0))
        completion = max(0, int(completion_tokens or 0))
        cached = max(0, int(cached_prompt_tokens or 0))
        with self._lock:
            session = self._token_sessions.setdefault(sid, {})
            prev_prompt, prev_completion, prev_cached = session.get(model, (0, 0, 0))
            session[model] = (
                prev_prompt + prompt,
                prev_completion + completion,
                prev_cached + cached,
            )

    def tokens_snapshot(self, session_id: str) -> Dict[str, Tuple[int, int, int]]:
        with self._lock:
            return dict(self._token_sessions.get(session_id, {}))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._token_sessions.pop(session_id, None)


STORE = ToolUsageStore()


def set_active_usage_session(session_id: str | None):
    return _ACTIVE_USAGE_SESSION_ID.set(session_id)


def reset_active_usage_session(token) -> None:
    _ACTIVE_USAGE_SESSION_ID.reset(token)
