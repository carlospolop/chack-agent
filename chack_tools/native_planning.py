from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from .task_steps_manager_state import (
    STORE,
    current_run_label,
    current_session_id,
)


_NATIVE_BACKENDS = {
    "codex": "codex",
    "code": "codex",
    "claude": "claude",
    "claude_code": "claude",
    "claude-code": "claude",
}


def native_planning_backend(value: Any) -> str:
    """Return the canonical backend name when native live planning is available."""
    raw = str(value or "").strip().lower()
    return _NATIVE_BACKENDS.get(raw, "")


def uses_native_planning(value: Any) -> bool:
    return bool(native_planning_backend(value))


def native_planning_prompt(backend: Any, *, required_first: bool) -> str:
    canonical = native_planning_backend(backend)
    if canonical == "codex":
        if required_first:
            return (
                "- Before any non-planning tool call, use Codex's built-in `update_plan` "
                "tool to create a concise plan. Keep that native plan updated whenever "
                "steps start, finish, or change."
            )
        return (
            "- For multi-step work, use Codex's built-in `update_plan` tool and keep "
            "the native plan current as work progresses."
        )
    if canonical == "claude":
        tools = "Claude Code's built-in planning tools (`TodoWrite` or `TaskCreate`/`TaskUpdate`)"
        if required_first:
            return (
                f"- Before any non-planning tool call, use {tools} to create a concise "
                "plan. Keep that native plan updated whenever steps start, finish, or change."
            )
        return (
            f"- For multi-step work, use {tools} and keep the native plan current as "
            "work progresses."
        )
    return ""


def _status(raw: Any, *, completed: Any = None) -> str:
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"done", "completed", "complete", "success", "succeeded"}:
        return "done"
    if value in {"doing", "in_progress", "inprogress", "active", "started", "working"}:
        return "doing"
    if completed is True:
        return "done"
    return "todo"


def _task_text(raw: dict[str, Any]) -> str:
    return str(
        raw.get("text")
        or raw.get("content")
        or raw.get("step")
        or raw.get("subject")
        or raw.get("description")
        or ""
    ).strip()


def sync_native_plan_snapshot(
    items: Iterable[Any],
    *,
    source: str,
    infer_current: bool = False,
) -> bool:
    """Mirror a complete native plan into the common callback/rendering store."""
    session_id = str(current_session_id() or "").strip()
    if not session_id:
        return False
    run_label = str(current_run_label() or "Run 1")
    tasks: list[dict[str, str]] = []
    for index, value in enumerate(items or [], start=1):
        if not isinstance(value, dict):
            continue
        text = _task_text(value)
        if not text:
            continue
        tasks.append(
            {
                "text": text,
                "status": _status(value.get("status"), completed=value.get("completed")),
                "notes": str(value.get("notes") or "").strip(),
                "source_id": str(
                    value.get("id")
                    or value.get("taskId")
                    or value.get("task_id")
                    or index
                ).strip(),
            }
        )

    if infer_current and tasks and not any(task["status"] == "doing" for task in tasks):
        current = next((task for task in tasks if task["status"] != "done"), None)
        if current is not None:
            current["status"] = "doing"

    STORE.replace_snapshot(session_id, run_label, tasks, source=source)
    return True


def _result_task_id(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("taskId", "task_id", "id"):
            value = result.get(key)
            if value not in (None, ""):
                return str(value).strip()
        for value in result.values():
            found = _result_task_id(value)
            if found:
                return found
        return ""
    if isinstance(result, list):
        for value in result:
            found = _result_task_id(value)
            if found:
                return found
        return ""
    text = str(result or "").strip()
    if text.startswith(("{", "[")):
        try:
            found = _result_task_id(json.loads(text))
            if found:
                return found
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    match = re.search(r"\btask\s*#?\s*([A-Za-z0-9._:-]+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def sync_claude_native_task(
    tool_name: Any,
    tool_input: Any,
    *,
    status: str,
    result: Any = None,
) -> bool:
    """Mirror Claude Code TodoWrite and TaskCreate/TaskUpdate calls after success."""
    if str(status or "").strip().lower() not in {"success", "completed", "complete"}:
        return False
    name = str(tool_name or "").strip().split("__")[-1].lower()
    arguments = tool_input if isinstance(tool_input, dict) else {}
    session_id = str(current_session_id() or "").strip()
    if not session_id:
        return False
    run_label = str(current_run_label() or "Run 1")

    if name == "todowrite":
        todos = arguments.get("todos")
        if not isinstance(todos, list):
            return False
        return sync_native_plan_snapshot(
            todos,
            source="claude:TodoWrite",
            infer_current=False,
        )

    if name == "taskcreate":
        text = _task_text(arguments)
        if not text:
            return False
        native_id = str(
            arguments.get("taskId")
            or arguments.get("task_id")
            or arguments.get("id")
            or _result_task_id(result)
            or ""
        ).strip()
        STORE.upsert_native_task(
            session_id,
            run_label,
            source_id=native_id,
            text=text,
            status=arguments.get("status") or "pending",
            notes=str(arguments.get("description") or "").strip(),
            source="claude:TaskCreate",
        )
        return True

    if name == "taskupdate":
        native_id = str(
            arguments.get("taskId")
            or arguments.get("task_id")
            or arguments.get("id")
            or ""
        ).strip()
        if not native_id:
            return False
        raw_status = str(arguments.get("status") or "").strip().lower()
        STORE.upsert_native_task(
            session_id,
            run_label,
            source_id=native_id,
            text=_task_text(arguments),
            status=raw_status,
            notes=str(arguments.get("description") or "").strip(),
            delete=raw_status in {"deleted", "removed", "cancelled", "canceled"},
            source="claude:TaskUpdate",
        )
        return True

    if name == "tasklist":
        candidate = result
        if isinstance(candidate, str) and candidate.strip().startswith(("{", "[")):
            try:
                candidate = json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
        if isinstance(candidate, dict):
            candidate = candidate.get("tasks") or candidate.get("items")
        if not isinstance(candidate, list):
            return False
        return sync_native_plan_snapshot(
            candidate,
            source="claude:TaskList",
            infer_current=False,
        )

    return False
