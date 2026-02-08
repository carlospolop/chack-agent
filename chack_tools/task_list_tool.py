import time
from datetime import datetime, timezone
from typing import Optional

try:
    from agents import function_tool
except ImportError:
    function_tool = None

from .config import ToolsConfig
from .telemetry import log_tool_started, log_tool_executed

from .task_list_state import STORE, current_run_label, current_session_id


class TaskListTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def manage(
        self,
        action: str,
        task_id: Optional[int] = None,
        text: str = "",
        status: str = "",
        tasks: str = "",
        notes: str = "",
    ) -> str:
        action_name = (action or "").strip().lower()
        session_id = current_session_id()
        if not session_id:
            return "ERROR: Task list context is not available for this request."
        run_label = current_run_label()
        result = STORE.apply(
            session_id=session_id,
            run_label=run_label,
            action=action_name,
            task_id=task_id,
            text=text,
            status=status,
            tasks_text=tasks,
            notes=notes,
        )
        board = STORE.render(session_id)
        reminders = []
        if action_name == "init" and result.startswith("SUCCESS:"):
            reminders.append(
                "Reminder: update this task list every time you complete a task calling this tool with `action=complete` and providing any relevant `notes` about the completion. This will help you keep track of your progress and next steps."
            )
        if action_name == "complete" and result.startswith("SUCCESS:"):
            reminders.append(
                "Reminder: only if needed, update/modify/add tasks based on new knowledge to be able to get all the needed context and information to solve the user's problem perfectly."
            )
        if reminders:
            return f"{result}\n\n" + "\n".join(reminders) + f"\n\n{board}"
        return f"{result}\n\n{board}"


def get_task_list_tool(helper: TaskListTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="task_list")
    def task_list(
        action: str,
        task_id: Optional[int] = None,
        text: str = "",
        status: str = "",
        tasks: str = "",
        notes: str = "",
    ) -> str:
        """Create and maintain the live per-request task plan shown to the user.

        Guardrails and behavior:
        - In each run, the first use of this tool MUST be `action="init"`.
        - Task-list calls DO NOT count toward the minimum non-task tool usage target.
        - The rendered board is persisted per request and split by run label
          (e.g. "Run 1" and "Run 2 (self-critique)").
        - Every mutating action updates the live board message in chat/thread.

        Actions and arguments:
        - `init`: initialize the current run list. Provide `tasks` as newline-separated items.
        - `list`: return the current full board.
        - `add`: add one task with `text` (optional `status`, `notes`).
        - `update`: update task `task_id` fields (`text`, `status`, `notes`).
        - `complete`: mark `task_id` as done (optional `notes`).
        - `delete`: remove `task_id`.
        - `clear`: clear all tasks in the current run.
        - `replace`: replace the current run list with newline-separated `tasks`.

        Status values:
        - `todo`, `doing`, `done`.
        """
        tool_input = {
            "action": action,
            "task_id": task_id,
            "text": text,
            "status": status,
            "tasks": tasks,
            "notes": notes,
        }
        start_ts = log_tool_started("task_list", tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.manage(
                action=action,
                task_id=task_id,
                text=text,
                status=status,
                tasks=tasks,
                notes=notes,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            duration_ms = int((time.time() - start_time) * 1000)
            log_tool_executed(
                "task_list",
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    return task_list
