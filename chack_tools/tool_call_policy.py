"""Token-saving policies applied to every tool call, whatever the backend.

Two policies:

* **Result reuse.** Models often repeat the exact same read-only call a few
  turns later (the same search, the same document fetch). Inside a short
  window the earlier result is returned instead of running the tool again.
  Shell/script tools, tools that act on a live target, tools that change or
  report on changing state, and long results are never reused, and any state-changing call drops what was kept.
* **Output spill.** A result too large for the context is saved to a file;
  the agent receives its start and end plus the path, and reads only the
  parts it needs. When the agent cannot read files, the result is cut to the
  same preview.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from .telemetry.otel import tool_result_error
from .tool_usage_state import BUDGET_EXEMPT_ATTRIBUTE

RESULT_REUSE_ATTRIBUTE = "chack_result_reuse"

# Tools that run code or commands: their result depends on more than their
# arguments, and repeating them is usually deliberate.
_SCRIPT_NAME_PARTS = frozenset(
    {"exec", "shell", "bash", "sh", "cli", "command", "script", "python", "run", "eval", "terminal"}
)
# Tools that act on a live target: repeating the same request is often the
# point (timing, retesting, waiting for a callback).
_LIVE_TARGET_NAME_PARTS = frozenset(
    {
        "attack", "auth", "browser", "click", "curl", "exploit", "fuzz", "http",
        "https", "login", "navigate", "oast", "ping", "probe", "request",
        "requests", "scan", "screenshot", "test", "type", "verify",
    }
)
# Tools that change state, or whose answer is expected to change between calls.
_STATEFUL_NAME_PARTS = frozenset(
    {
        "add", "append", "cancel", "check", "clear", "create", "delete", "edit",
        "init", "insert", "job", "jobs", "list", "manage", "manager", "mark", "move", "poll",
        "post", "put", "record", "register", "reindex", "remove", "rename",
        "report", "reset", "result", "results", "retry", "save", "send", "set",
        "start", "status", "stop", "store", "submit", "task", "tasks", "update",
        "upload", "upsert", "wait", "write",
    }
)


# Of the above, the ones that only read changing state.
_READ_STATE_NAME_PARTS = frozenset(
    {"check", "job", "jobs", "list", "poll", "result", "results", "status", "task", "tasks", "wait"}
)


def reuse_tool_results(tool: Any, enabled: bool = True) -> Any:
    """Mark ``tool`` as safe (or unsafe) to answer from an earlier identical call."""
    setattr(tool, RESULT_REUSE_ATTRIBUTE, bool(enabled))
    return tool


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()


def _name_parts(name: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", str(name or "").lower()) if part}


def is_reusable_tool(tool: Any) -> bool:
    """Whether an identical call to ``tool`` may be answered from an earlier result."""
    explicit = getattr(tool, RESULT_REUSE_ATTRIBUTE, None)
    if explicit is not None:
        return bool(explicit)
    if getattr(tool, BUDGET_EXEMPT_ATTRIBUTE, False):
        return False
    parts = _name_parts(_tool_name(tool))
    if not parts:
        return False
    return not (parts & (_SCRIPT_NAME_PARTS | _LIVE_TARGET_NAME_PARTS | _STATEFUL_NAME_PARTS))


def changes_state(tool: Any) -> bool:
    """Whether a call to ``tool`` may invalidate results kept for reuse."""
    explicit = getattr(tool, RESULT_REUSE_ATTRIBUTE, None)
    if explicit is True:
        return False
    if getattr(tool, BUDGET_EXEMPT_ATTRIBUTE, False):
        return True
    parts = _name_parts(_tool_name(tool))
    return bool(parts & (_STATEFUL_NAME_PARTS - _READ_STATE_NAME_PARTS))


def _call_key(name: str, raw_args: Any) -> str:
    args = raw_args
    if isinstance(raw_args, (str, bytes)):
        try:
            args = json.loads(raw_args or "{}")
        except (TypeError, ValueError):
            args = raw_args
    try:
        canonical = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = str(args)
    return f"{name}\x00{canonical}"


@dataclass
class _Kept:
    result: str
    at: float


class ToolResultReuse:
    """Short-lived memory of read-only tool results for one run."""

    def __init__(
        self,
        window_seconds: float = 60,
        max_chars: int = 20000,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int = 256,
    ) -> None:
        self.window_seconds = max(0.0, float(window_seconds or 0))
        self.max_chars = max(0, int(max_chars or 0))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._kept: dict[str, _Kept] = {}
        self._lock = threading.Lock()
        self.reused_calls = 0

    @property
    def enabled(self) -> bool:
        return self.window_seconds > 0 and self.max_chars > 0

    def lookup(self, tool: Any, raw_args: Any) -> Optional[str]:
        """Return the reused result for an identical recent call, or ``None``."""
        if not self.enabled:
            return None
        if not is_reusable_tool(tool):
            if changes_state(tool):
                with self._lock:
                    self._kept.clear()
            return None
        key = _call_key(_tool_name(tool), raw_args)
        now = self._clock()
        with self._lock:
            kept = self._kept.get(key)
            if kept is None:
                return None
            age = now - kept.at
            if age > self.window_seconds:
                self._kept.pop(key, None)
                return None
            self.reused_calls += 1
        return (
            f"[Same result as the identical {_tool_name(tool)} call {int(age)}s ago; "
            "the tool was not run again.]\n" + kept.result
        )

    def store(self, tool: Any, raw_args: Any, result: Any) -> None:
        if not self.enabled or not isinstance(result, str) or not is_reusable_tool(tool):
            return
        if len(result) > self.max_chars or _looks_like_error(result):
            return
        key = _call_key(_tool_name(tool), raw_args)
        with self._lock:
            self._kept[key] = _Kept(result=result, at=self._clock())
            while len(self._kept) > self.max_entries:
                self._kept.pop(next(iter(self._kept)))


def _looks_like_error(result: str) -> bool:
    return bool(tool_result_error(result))


def spill_large_output(
    text: str,
    *,
    max_chars: int,
    preview_chars: int,
    directory: str = "",
    readable: bool = True,
) -> str:
    """Return ``text`` unchanged when it fits; otherwise a preview (plus file path)."""
    value = str(text if text is not None else "")
    limit = int(max_chars or 0)
    if limit <= 0 or len(value) <= limit:
        return value
    preview = max(0, min(int(preview_chars or 0), limit))
    head_len = preview * 2 // 3
    tail_len = preview - head_len
    head = value[:head_len]
    tail = value[len(value) - tail_len :] if tail_len else ""
    omitted = len(value) - len(head) - len(tail)
    path = _write_output_file(value, directory) if readable else ""
    if path:
        marker = (
            f"\n... [{omitted} characters omitted. The full result ({len(value)} characters) "
            f"is saved at {path}; search it or read only the line ranges you need.] ...\n"
        )
    else:
        marker = f"\n... [{omitted} characters omitted from this result.] ...\n"
    return f"{head}{marker}{tail}"


def _write_output_file(value: str, directory: str) -> str:
    target = str(directory or "").strip() or os.path.join(
        tempfile.gettempdir(), "chack-tool-outputs"
    )
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
    path = os.path.join(target, f"tool-output-{digest}.txt")
    try:
        os.makedirs(target, exist_ok=True)
        if not os.path.exists(path):
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8", errors="replace") as handle:
                handle.write(value)
            os.replace(tmp, path)
    except OSError:
        return ""
    return path


class ToolCallPolicy:
    """Result reuse and output spill configured from ``ToolsConfig``."""

    def __init__(self, tools_config: Any = None, *, outputs_readable: bool = True) -> None:
        def _int(name: str, default: int) -> int:
            try:
                return int(getattr(tools_config, name, default))
            except (TypeError, ValueError):
                return default

        self.reuse = ToolResultReuse(
            window_seconds=_int("tool_result_reuse_seconds", 60),
            max_chars=_int("tool_result_reuse_max_chars", 20000),
        )
        self.max_chars = _int("tool_output_max_chars", 40000)
        self.preview_chars = _int("tool_output_preview_chars", 12000)
        self.directory = str(getattr(tools_config, "tool_output_dir", "") or "")
        self.outputs_readable = bool(outputs_readable)

    def fit(self, text: str) -> str:
        return spill_large_output(
            text,
            max_chars=self.max_chars,
            preview_chars=self.preview_chars,
            directory=self.directory,
            readable=self.outputs_readable,
        )

    def wrap(self, tools: Iterable[Any]) -> list[Any]:
        """Apply reuse and spill to each function tool's ``on_invoke_tool``."""
        wrapped: list[Any] = []
        for tool in tools:
            # Wrap the tool's own handler, not an earlier run's wrapper, so a
            # reused tool object never answers from another run's results.
            original = getattr(tool, "_chack_call_policy_original", None) or getattr(
                tool, "on_invoke_tool", None
            )
            if original is not None:

                async def _invoke(ctx, raw_args, _orig=original, _tool=tool):
                    reused = self.reuse.lookup(_tool, raw_args)
                    if reused is not None:
                        return self.fit(reused)
                    result = await _orig(ctx, raw_args)
                    if not isinstance(result, str):
                        return result
                    self.reuse.store(_tool, raw_args, result)
                    return self.fit(result)

                tool.on_invoke_tool = _invoke  # type: ignore[attr-defined]
                tool._chack_call_policy_original = original  # type: ignore[attr-defined]
            wrapped.append(tool)
        return wrapped


_SHELL_TOOL_NAMES = frozenset({"exec", "run_shell_command"})


def has_exec_tool(tools: Iterable[Any]) -> bool:
    """Whether the agent can read a spilled output file through a shell tool."""
    return any(_tool_name(tool) in _SHELL_TOOL_NAMES for tool in tools)
