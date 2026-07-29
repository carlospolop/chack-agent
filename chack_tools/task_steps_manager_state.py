from __future__ import annotations

import contextvars
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


_ACTIVE_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "chack_task_steps_manager_session_id",
    default=None,
)
_ACTIVE_RUN_LABEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chack_task_steps_manager_run_label",
    default="Run 1",
)
_LOGGER = logging.getLogger("chack.task_steps_manager")


@dataclass
class TaskItem:
    id: int
    text: str
    status: str = "todo"  # todo | doing | done
    notes: str = ""
    # Backend-native task identifier (for example Claude Code TaskCreate IDs).
    # It is intentionally omitted from public snapshots/rendering.
    source_id: str = ""


@dataclass
class TaskRun:
    label: str
    initialized: bool = False
    next_id: int = 1
    tasks: List[TaskItem] = field(default_factory=list)
    completed_emitted: bool = False


@dataclass
class TaskSession:
    session_id: str
    title: str = "Task Steps Manager"
    runs: Dict[str, TaskRun] = field(default_factory=dict)


class TaskStepsManagerStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, TaskSession] = {}
        self._listeners: Dict[str, List[Callable[[str], None]]] = {}

    def create_session(self, session_id: str, title: str = "Task Steps Manager") -> TaskSession:
        with self._lock:
            session = TaskSession(session_id=session_id, title=title)
            self._sessions[session_id] = session
            self._listeners.setdefault(session_id, [])
            return session

    def get_session(self, session_id: str) -> Optional[TaskSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def ensure_run(self, session_id: str, run_label: str) -> TaskRun:
        with self._lock:
            session = self._sessions.setdefault(session_id, TaskSession(session_id=session_id))
            run = session.runs.get(run_label)
            if run is None:
                run = TaskRun(label=run_label)
                session.runs[run_label] = run
            return run

    def register_listener(self, session_id: str, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._listeners.setdefault(session_id, []).append(callback)

    def unregister_listener(self, session_id: str, callback: Callable[[str], None]) -> None:
        with self._lock:
            callbacks = self._listeners.get(session_id, [])
            if callback in callbacks:
                callbacks.remove(callback)

    def _notify(self, session_id: str, *, reason: str = "") -> None:
        text = self.render(session_id)
        if reason:
            _LOGGER.info("Task list updated (%s):\n%s", reason, text)
        else:
            _LOGGER.info("Task list updated:\n%s", text)
        callbacks = list(self._listeners.get(session_id, []))
        for cb in callbacks:
            try:
                cb(text)
            except Exception:
                pass

    @staticmethod
    def _is_completed(run: TaskRun) -> bool:
        if not run.initialized:
            return False
        if not run.tasks:
            return True
        return all(task.status == "done" for task in run.tasks)

    @staticmethod
    def _log_event(
        event_type: str,
        *,
        payload: Dict[str, Any],
        session_id: str,
        run_label: str,
    ) -> None:
        try:
            from chack_tools.telemetry import log_event
        except Exception:
            return
        log_event(
            event_type,
            payload=payload,
            task_session_id=session_id,
            run_label=run_label,
        )

    def _maybe_emit_completion(self, session_id: str, run: TaskRun) -> None:
        completed = self._is_completed(run)
        if completed and not run.completed_emitted:
            self._log_event(
                "tasklist_completed",
                payload={
                    "tasks_total": len(run.tasks),
                    "tasks_done": len([t for t in run.tasks if t.status == "done"]),
                    "tasks": self._snapshot(run),
                },
                session_id=session_id,
                run_label=run.label,
            )
            run.completed_emitted = True
            return
        if not completed:
            run.completed_emitted = False

    @staticmethod
    def _snapshot(run: TaskRun) -> List[Dict[str, Any]]:
        return [
            {
                "id": task.id,
                "text": task.text,
                "status": task.status,
                "notes": task.notes,
            }
            for task in run.tasks
        ]

    @staticmethod
    def _run_counts(run: TaskRun) -> Dict[str, int]:
        tasks_total = len(run.tasks)
        tasks_done = len([task for task in run.tasks if task.status == "done"])
        tasks_doing = len([task for task in run.tasks if task.status == "doing"])
        tasks_todo = max(0, tasks_total - tasks_done - tasks_doing)
        return {
            "tasks_total": tasks_total,
            "tasks_done": tasks_done,
            "tasks_doing": tasks_doing,
            "tasks_todo": tasks_todo,
        }

    @classmethod
    def _progress_percent(cls, run: TaskRun) -> float:
        counts = cls._run_counts(run)
        tasks_total = counts["tasks_total"]
        if tasks_total <= 0:
            return 0.0
        return round((counts["tasks_done"] / tasks_total) * 100.0, 2)

    @classmethod
    def _current_task(cls, run: TaskRun) -> str:
        for task in run.tasks:
            if task.status == "doing":
                return task.text
        for task in run.tasks:
            if task.status != "done":
                return task.text
        return ""

    @classmethod
    def _run_snapshot(cls, run: TaskRun) -> Dict[str, Any]:
        counts = cls._run_counts(run)
        return {
            "label": run.label,
            "initialized": run.initialized,
            "completed": cls._is_completed(run),
            "progress_percent": cls._progress_percent(run),
            "current_task": cls._current_task(run),
            **counts,
            "tasks": cls._snapshot(run),
        }

    def snapshot(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {
                    "session_id": session_id,
                    "title": "Task Steps Manager",
                    "runs": [],
                    "tasks_total": 0,
                    "tasks_done": 0,
                    "tasks_doing": 0,
                    "tasks_todo": 0,
                    "progress_percent": 0.0,
                }

            runs = [self._run_snapshot(run) for run in session.runs.values()]
            tasks_total = sum(int(run["tasks_total"]) for run in runs)
            tasks_done = sum(int(run["tasks_done"]) for run in runs)
            tasks_doing = sum(int(run["tasks_doing"]) for run in runs)
            tasks_todo = sum(int(run["tasks_todo"]) for run in runs)
            progress_percent = round((tasks_done / tasks_total) * 100.0, 2) if tasks_total > 0 else 0.0
            active_run = next((run for run in reversed(runs) if not bool(run["completed"])), runs[-1] if runs else None)
            return {
                "session_id": session.session_id,
                "title": session.title,
                "runs": runs,
                "tasks_total": tasks_total,
                "tasks_done": tasks_done,
                "tasks_doing": tasks_doing,
                "tasks_todo": tasks_todo,
                "progress_percent": progress_percent,
                "active_run_label": active_run["label"] if active_run else "",
                "current_task": active_run["current_task"] if active_run else "",
                "completed": bool(runs) and all(bool(run["completed"]) for run in runs),
            }

    @staticmethod
    def _normalize_status(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if raw in {"done", "completed", "complete", "success", "succeeded"}:
            return "done"
        if raw in {"doing", "in_progress", "inprogress", "active", "started", "working"}:
            return "doing"
        return "todo"

    def replace_snapshot(
        self,
        session_id: str,
        run_label: str,
        tasks: List[Dict[str, Any]],
        *,
        source: str = "native_plan",
    ) -> str:
        """Atomically mirror a backend-native plan and emit at most one update.

        Native planners generally publish the complete current plan repeatedly.  A
        snapshot operation avoids the burst of Telegram/Discord edits that would
        result from replaying one ``replace`` plus N ``update`` actions.
        """
        run = self.ensure_run(session_id, run_label)
        normalized: List[TaskItem] = []
        for index, raw in enumerate(tasks or [], start=1):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            normalized.append(
                TaskItem(
                    id=len(normalized) + 1,
                    text=text,
                    status=self._normalize_status(raw.get("status")),
                    notes=str(raw.get("notes") or "").strip(),
                    source_id=str(raw.get("source_id") or index).strip(),
                )
            )

        before = [
            (task.text, task.status, task.notes, task.source_id)
            for task in run.tasks
        ]
        after = [
            (task.text, task.status, task.notes, task.source_id)
            for task in normalized
        ]
        if run.initialized and before == after:
            return "SUCCESS: native task snapshot unchanged"

        was_initialized = run.initialized
        run.tasks = normalized
        run.next_id = len(normalized) + 1
        run.initialized = True
        run.completed_emitted = False
        self._notify(session_id, reason=f"{run_label}:{source}:snapshot")
        event_type = "tasklist_updated" if was_initialized else "tasklist_defined"
        self._log_event(
            event_type,
            payload={
                "action": "native_snapshot",
                "source": source,
                **self._run_counts(run),
                "progress_percent": self._progress_percent(run),
                "current_task": self._current_task(run),
                "tasks": self._snapshot(run),
            },
            session_id=session_id,
            run_label=run.label,
        )
        self._maybe_emit_completion(session_id, run)
        return f"SUCCESS: mirrored {len(run.tasks)} native plan tasks"

    def upsert_native_task(
        self,
        session_id: str,
        run_label: str,
        *,
        source_id: str = "",
        text: str = "",
        status: str = "",
        notes: str = "",
        delete: bool = False,
        source: str = "native_plan",
    ) -> str:
        """Apply a single native TaskCreate/TaskUpdate-style delta."""
        run = self.ensure_run(session_id, run_label)
        native_id = str(source_id or "").strip()
        clean_text = str(text or "").strip()
        task = next(
            (item for item in run.tasks if native_id and item.source_id == native_id),
            None,
        )
        if task is None and native_id.isdigit():
            task = next((item for item in run.tasks if item.id == int(native_id)), None)
        if task is None and clean_text:
            task = next((item for item in run.tasks if item.text == clean_text), None)

        if delete:
            if task is None:
                return "SUCCESS: native task already absent"
            run.tasks = [item for item in run.tasks if item is not task]
        elif task is None:
            if not clean_text:
                return "ERROR: native task text is required for a new task"
            task = TaskItem(
                id=run.next_id,
                text=clean_text,
                status=self._normalize_status(status),
                notes=str(notes or "").strip(),
                source_id=native_id,
            )
            run.tasks.append(task)
            run.next_id += 1
        else:
            before = (task.text, task.status, task.notes, task.source_id)
            if clean_text:
                task.text = clean_text
            if str(status or "").strip():
                task.status = self._normalize_status(status)
            if str(notes or "").strip():
                task.notes = str(notes).strip()
            if native_id:
                task.source_id = native_id
            if before == (task.text, task.status, task.notes, task.source_id):
                return "SUCCESS: native task unchanged"

        run.initialized = True
        run.completed_emitted = False
        self._notify(session_id, reason=f"{run_label}:{source}:delta")
        self._log_event(
            "tasklist_updated",
            payload={
                "action": "native_delete" if delete else "native_upsert",
                "source": source,
                "source_id": native_id,
                **self._run_counts(run),
                "progress_percent": self._progress_percent(run),
                "current_task": self._current_task(run),
                "tasks": self._snapshot(run),
            },
            session_id=session_id,
            run_label=run.label,
        )
        self._maybe_emit_completion(session_id, run)
        return "SUCCESS: native task plan updated"

    def apply(
        self,
        session_id: str,
        run_label: str,
        action: str,
        task_id: Optional[int] = None,
        text: str = "",
        status: str = "",
        tasks_text: str = "",
        notes: str = "",
    ) -> str:
        action = (action or "").strip().lower()
        if not action:
            return "ERROR: action is required"
        run = self.ensure_run(session_id, run_label)

        def _parse_tasks(raw: str) -> List[str]:
            parts = [line.strip() for line in (raw or "").splitlines()]
            return [p for p in parts if p]

        if action == "init":
            if run.initialized:
                return (
                    "SUCCESS: task list already initialized for this run; ignored duplicate init. "
                    "Use action=replace or action=update instead."
                )
            items = _parse_tasks(tasks_text)
            if len(items) < 2:
                return (
                    "ERROR: action=init requires at least 2 tasks. "
                    "Provide `tasks` as newline-separated items with at least two lines."
                )
            run.tasks = []
            run.next_id = 1
            run.completed_emitted = False
            for item in items:
                run.tasks.append(TaskItem(id=run.next_id, text=item, status="todo"))
                run.next_id += 1
            run.initialized = True
            self._notify(session_id, reason=f"{run_label}:init")
            self._log_event(
                "tasklist_defined",
                payload={
                    "tasks_total": len(run.tasks),
                    "tasks_done": 0,
                    "tasks_doing": 0,
                    "tasks_todo": len(run.tasks),
                    "progress_percent": 0.0,
                    "tasks": self._snapshot(run),
                },
                session_id=session_id,
                run_label=run.label,
            )
            self._maybe_emit_completion(session_id, run)
            return f"SUCCESS: initialized {len(run.tasks)} tasks for {run_label}"

        if action == "list":
            return self.render(session_id)

        if not run.initialized:
            return "ERROR: Task list not initialized for this run. First call must be action=init."

        if action == "add":
            if not text.strip():
                return "ERROR: text is required for action=add"
            run.tasks.append(TaskItem(id=run.next_id, text=text.strip(), status=(status or "todo")))
            run.next_id += 1
            self._notify(session_id, reason=f"{run_label}:add")
            self._log_event(
                "tasklist_updated",
                payload={
                    "action": "add",
                    "task_id": run.tasks[-1].id,
                    "text": run.tasks[-1].text,
                    "status": run.tasks[-1].status,
                    "notes": run.tasks[-1].notes,
                    **self._run_counts(run),
                    "progress_percent": self._progress_percent(run),
                    "current_task": self._current_task(run),
                    "tasks": self._snapshot(run),
                },
                session_id=session_id,
                run_label=run.label,
            )
            self._maybe_emit_completion(session_id, run)
            return f"SUCCESS: added task {run.tasks[-1].id}"

        if action in {"update", "complete", "delete"}:
            if task_id is None:
                return f"ERROR: task_id is required for action={action}"
            task = next((t for t in run.tasks if t.id == int(task_id)), None)
            if task is None:
                return f"ERROR: task_id {task_id} not found"
            if action == "delete":
                run.tasks = [t for t in run.tasks if t.id != int(task_id)]
                self._notify(session_id, reason=f"{run_label}:delete:{task_id}")
                self._log_event(
                    "tasklist_updated",
                    payload={
                        "action": "delete",
                        "task_id": int(task_id),
                        **self._run_counts(run),
                        "progress_percent": self._progress_percent(run),
                        "current_task": self._current_task(run),
                        "tasks": self._snapshot(run),
                    },
                    session_id=session_id,
                    run_label=run.label,
                )
                self._maybe_emit_completion(session_id, run)
                return f"SUCCESS: deleted task {task_id}"
            if action == "complete":
                task.status = "done"
                if notes.strip():
                    task.notes = notes.strip()
                self._notify(session_id, reason=f"{run_label}:complete:{task_id}")
                self._log_event(
                    "tasklist_item_completed",
                    payload={
                        "task_id": task.id,
                        "text": task.text,
                        "notes": task.notes,
                        **self._run_counts(run),
                        "progress_percent": self._progress_percent(run),
                        "current_task": self._current_task(run),
                        "tasks": self._snapshot(run),
                    },
                    session_id=session_id,
                    run_label=run.label,
                )
                self._log_event(
                    "tasklist_updated",
                    payload={
                        "action": "complete",
                        "task_id": task.id,
                        "text": task.text,
                        "status": task.status,
                        "notes": task.notes,
                        **self._run_counts(run),
                        "progress_percent": self._progress_percent(run),
                        "current_task": self._current_task(run),
                        "tasks": self._snapshot(run),
                    },
                    session_id=session_id,
                    run_label=run.label,
                )
                self._maybe_emit_completion(session_id, run)
                return f"SUCCESS: completed task {task_id}"
            if text.strip():
                task.text = text.strip()
            if status.strip():
                task.status = status.strip().lower()
            if notes.strip():
                task.notes = notes.strip()
            self._notify(session_id, reason=f"{run_label}:update:{task_id}")
            self._log_event(
                "tasklist_updated",
                payload={
                    "action": "update",
                    "task_id": task.id,
                    "text": task.text,
                    "status": task.status,
                    "notes": task.notes,
                    **self._run_counts(run),
                    "progress_percent": self._progress_percent(run),
                    "current_task": self._current_task(run),
                    "tasks": self._snapshot(run),
                },
                session_id=session_id,
                run_label=run.label,
            )
            self._maybe_emit_completion(session_id, run)
            return f"SUCCESS: updated task {task_id}"

        if action == "replace":
            items = _parse_tasks(tasks_text)
            run.tasks = []
            run.next_id = 1
            run.completed_emitted = False
            for item in items:
                run.tasks.append(TaskItem(id=run.next_id, text=item, status="todo"))
                run.next_id += 1
            run.initialized = True
            self._notify(session_id, reason=f"{run_label}:replace")
            self._log_event(
                "tasklist_defined",
                payload={
                    "tasks_total": len(run.tasks),
                    "tasks_done": 0,
                    "tasks_doing": 0,
                    "tasks_todo": len(run.tasks),
                    "progress_percent": 0.0,
                    "tasks": self._snapshot(run),
                },
                session_id=session_id,
                run_label=run.label,
            )
            self._maybe_emit_completion(session_id, run)
            return f"SUCCESS: replaced tasks for {run_label} with {len(run.tasks)} items"

        return (
            "ERROR: unsupported action. Use one of: init, list, add, update, "
            "complete, delete, replace"
        )

    def render(self, session_id: str) -> str:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return "Task list session not found."
            lines = [f"🗂 {session.title}"]
            if not session.runs:
                lines.append("- (no runs yet)")
                return "\n".join(lines)
            for run_label, run in session.runs.items():
                lines.append("")
                lines.append(f"{run_label}:")
                if not run.tasks:
                    state = "not initialized" if not run.initialized else "no tasks"
                    lines.append(f"- ({state})")
                    continue
                for task in run.tasks:
                    mark = "x" if task.status == "done" else ("~" if task.status == "doing" else " ")
                    lines.append(f"- [{mark}] {task.id}. {task.text}")
                    if task.notes:
                        lines.append(f"  note: {task.notes}")
            return "\n".join(lines)


STORE = TaskStepsManagerStore()


def set_active_context(session_id: Optional[str], run_label: str):
    token_session = _ACTIVE_SESSION_ID.set(session_id)
    token_run = _ACTIVE_RUN_LABEL.set(run_label)
    return token_session, token_run


def reset_active_context(tokens) -> None:
    token_session, token_run = tokens
    _ACTIVE_SESSION_ID.reset(token_session)
    _ACTIVE_RUN_LABEL.reset(token_run)


def current_session_id() -> Optional[str]:
    return _ACTIVE_SESSION_ID.get()


def current_run_label() -> str:
    return _ACTIVE_RUN_LABEL.get()
