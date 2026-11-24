from __future__ import annotations

import contextvars
import threading
from collections import Counter
from typing import Dict, Tuple


_ACTIVE_USAGE_SESSION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chack_tool_usage_session_id",
    default=None,
)
_ACTIVE_MAX_TOOLS_USED: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "chack_tool_usage_max_tools",
    default=None,
)

_NON_COUNTED_TOOL_NAMES = frozenset(
    {
        "task_steps_manager",
        "update_plan",
        "todowrite",
        "taskcreate",
        "taskupdate",
        "tasklist",
        "taskget",
        "enterplanmode",
        "exitplanmode",
        "check_budget_status",
    }
)


def canonical_tool_name(tool_name: str) -> str:
    return str(tool_name or "").strip().split("__")[-1].lower()


def is_non_counted_tool_name(tool_name: str) -> bool:
    name = canonical_tool_name(tool_name)
    return not name or name in _NON_COUNTED_TOOL_NAMES


# Tools that record an agent's results (a verdict, a finding) must work after the tool budget is spent: rejecting them
# discards work the agent already finished. A tool opts in with ``budget_exempt(tool)``.
BUDGET_EXEMPT_ATTRIBUTE = "chack_budget_exempt"
_BUDGET_EXEMPT_TOOL_NAMES: set[str] = set()
_BUDGET_EXEMPT_LOCK = threading.Lock()


def budget_exempt(tool):
    """Mark a tool as exempt from the tool budget; returns the tool so it can wrap a definition."""
    setattr(tool, BUDGET_EXEMPT_ATTRIBUTE, True)
    return tool


def register_budget_exempt_tools(tools) -> None:
    """Remember the names of the marked tools among those an agent is given."""
    names = {canonical_tool_name(getattr(tool, "name", "") or getattr(tool, "__name__", ""))
             for tool in tools or [] if getattr(tool, BUDGET_EXEMPT_ATTRIBUTE, False)}
    names.discard("")
    if names:
        with _BUDGET_EXEMPT_LOCK:
            _BUDGET_EXEMPT_TOOL_NAMES.update(names)


def is_budget_exempt_tool_name(tool_name: str) -> bool:
    """Whether calls to the tool neither count against nor are blocked by the tool budget."""
    return is_non_counted_tool_name(tool_name) or canonical_tool_name(tool_name) in _BUDGET_EXEMPT_TOOL_NAMES


class ToolUsageStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Counter[str]] = {}
        self._token_sessions: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
        self._agent_run_sessions: Dict[str, int] = {}
        # Sessions owned by a run in another process: their usage goes to the
        # run-state file that run reads (see ``share_session_with_parent``).
        self._parent_process_sessions: set[str] = set()

    def share_session_with_parent(self, session_id: str | None) -> None:
        """Send usage added for ``session_id`` to the run that owns it in another process.

        A tool process (MCP tool server, isolated researcher) calls this for
        the parent run's sessions so the sub-agents it runs count in that run.
        """
        sid = str(session_id or "").strip()
        if sid:
            with self._lock:
                self._parent_process_sessions.add(sid)

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = Counter()
            self._token_sessions[session_id] = {}
            self._agent_run_sessions[session_id] = 0

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
        cache_write_tokens: int = 0,
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
        cache_write = max(0, int(cache_write_tokens or 0))
        with self._lock:
            shared = sid in self._parent_process_sessions
        if shared:
            from .run_lifecycle import record_tool_process_usage

            record_tool_process_usage(sid, model, (prompt, completion, cached, cache_write))
            return
        with self._lock:
            session = self._token_sessions.setdefault(sid, {})
            prev_prompt, prev_completion, prev_cached, prev_cache_write = session.get(
                model, (0, 0, 0, 0)
            )
            session[model] = (
                prev_prompt + prompt,
                prev_completion + completion,
                prev_cached + cached,
                prev_cache_write + cache_write,
            )

    def tokens_snapshot(self, session_id: str) -> Dict[str, Tuple[int, int, int, int]]:
        """Usage of the agents a run started: in this process plus its tool processes."""
        with self._lock:
            usage = dict(self._token_sessions.get(session_id, {}))
        for model, values in self.tool_process_tokens_snapshot(session_id).items():
            previous = usage.get(model, (0, 0, 0, 0))
            usage[model] = tuple(previous[i] + values[i] for i in range(4))  # type: ignore[assignment]
        return usage

    @staticmethod
    def tool_process_tokens_snapshot(session_id: str) -> Dict[str, Tuple[int, int, int, int]]:
        """The part of ``tokens_snapshot`` spent in tool processes (never seen by this process's observers)."""
        if not session_id:
            return {}
        from .run_lifecycle import read_tool_process_usage

        return read_tool_process_usage(session_id)[0]

    def add_agent_runs(self, count: int = 1, session_id: str | None = None) -> None:
        """Count agent runs started for the run that owns the session (sub-agents, nested runs)."""
        sid = session_id or _ACTIVE_USAGE_SESSION_ID.get()
        runs = max(0, int(count or 0))
        if not sid or not runs:
            return
        with self._lock:
            shared = sid in self._parent_process_sessions
            if not shared:
                self._agent_run_sessions[sid] = self._agent_run_sessions.get(sid, 0) + runs
                return
        from .run_lifecycle import record_tool_process_usage

        record_tool_process_usage(sid, agent_runs=runs)

    def agent_runs_snapshot(self, session_id: str) -> int:
        """Agent runs started for a run: in this process plus its tool processes."""
        if not session_id:
            return 0
        from .run_lifecycle import read_tool_process_usage

        with self._lock:
            runs = self._agent_run_sessions.get(session_id, 0)
        return runs + read_tool_process_usage(session_id)[1]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._token_sessions.pop(session_id, None)
            self._agent_run_sessions.pop(session_id, None)


STORE = ToolUsageStore()


def set_active_usage_session(session_id: str | None):
    return _ACTIVE_USAGE_SESSION_ID.set(session_id)


def reset_active_usage_session(token) -> None:
    _ACTIVE_USAGE_SESSION_ID.reset(token)


def set_active_max_tools_used(value: int | None):
    if value is None:
        return _ACTIVE_MAX_TOOLS_USED.set(None)
    return _ACTIVE_MAX_TOOLS_USED.set(max(0, int(value)))


def reset_active_max_tools_used(token) -> None:
    _ACTIVE_MAX_TOOLS_USED.reset(token)


def current_usage_session_id() -> str | None:
    return _ACTIVE_USAGE_SESSION_ID.get()


def current_max_tools_used() -> int:
    return int(_ACTIVE_MAX_TOOLS_USED.get() or 0)


def active_max_tools_used() -> int | None:
    """Return the per-run override/config value when a run context is active."""
    value = _ACTIVE_MAX_TOOLS_USED.get()
    if value is None:
        return None
    return max(0, int(value))


def effective_max_tools_used(configured: int) -> int:
    """Resolve the run-scoped tool maximum, preserving an explicit zero override."""
    active = active_max_tools_used()
    if active is None:
        return max(0, int(configured or 0))
    return active


def non_task_tool_count(counter: Counter[str]) -> int:
    total = 0
    for tool_name, count in counter.items():
        if is_budget_exempt_tool_name(tool_name):
            continue
        total += max(0, int(count or 0))
    return total
