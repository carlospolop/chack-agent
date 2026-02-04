from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


_ACTIVE_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "chack_tasklist_session_id",
    default=None,
)
_ACTIVE_RUN_LABEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chack_tasklist_run_label",
    default="Run 1",
)


@dataclass
class TaskItem:
    id: int
    text: str
    status: str = "todo"  # todo | doing | done
    notes: str = ""


@dataclass
class TaskRun:
    label: str
    initialized: bool = False
    next_id: int = 1
    tasks: List[TaskItem] = field(default_factory=list)


@dataclass
class TaskSession:
    session_id: str
    title: str = "Task List"
    runs: Dict[str, TaskRun] = field(default_factory=dict)


class TaskListStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, TaskSession] = {}
        self._listeners: Dict[str, List[Callable[[str], None]]] = {}

    def create_session(self, session_id: str, title: str = "Task List") -> TaskSession:
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

    def _notify(self, session_id: str) -> None:
        text = self.render(session_id)
        callbacks = list(self._listeners.get(session_id, []))
        for cb in callbacks:
            try:
                cb(text)
            except Exception:
                pass

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
            items = _parse_tasks(tasks_text)
            run.tasks = []
            run.next_id = 1
            for item in items:
                run.tasks.append(TaskItem(id=run.next_id, text=item, status="todo"))
                run.next_id += 1
            run.initialized = True
            self._notify(session_id)
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
            self._notify(session_id)
            return f"SUCCESS: added task {run.tasks[-1].id}"

        if action in {"update", "complete", "delete"}:
            if task_id is None:
                return f"ERROR: task_id is required for action={action}"
            task = next((t for t in run.tasks if t.id == int(task_id)), None)
            if task is None:
                return f"ERROR: task_id {task_id} not found"
            if action == "delete":
                run.tasks = [t for t in run.tasks if t.id != int(task_id)]
                self._notify(session_id)
                return f"SUCCESS: deleted task {task_id}"
            if action == "complete":
                task.status = "done"
                if notes.strip():
                    task.notes = notes.strip()
                self._notify(session_id)
                return f"SUCCESS: completed task {task_id}"
            if text.strip():
                task.text = text.strip()
            if status.strip():
                task.status = status.strip().lower()
            if notes.strip():
                task.notes = notes.strip()
            self._notify(session_id)
            return f"SUCCESS: updated task {task_id}"

        if action == "clear":
            run.tasks = []
            run.next_id = 1
            run.initialized = True
            self._notify(session_id)
            return f"SUCCESS: cleared tasks for {run_label}"

        if action == "replace":
            items = _parse_tasks(tasks_text)
            run.tasks = []
            run.next_id = 1
            for item in items:
                run.tasks.append(TaskItem(id=run.next_id, text=item, status="todo"))
                run.next_id += 1
            run.initialized = True
            self._notify(session_id)
            return f"SUCCESS: replaced tasks for {run_label} with {len(run.tasks)} items"

        return (
            "ERROR: unsupported action. Use one of: init, list, add, update, "
            "complete, delete, clear, replace"
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


STORE = TaskListStore()


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
