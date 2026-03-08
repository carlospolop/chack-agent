import unittest
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_tools" / "task_steps_manager_state.py"
MODULE_SPEC = importlib.util.spec_from_file_location("task_steps_manager_state", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)
TaskStepsManagerStore = MODULE.TaskStepsManagerStore


class TaskStepsManagerStateTests(unittest.TestCase):
    def test_snapshot_tracks_progress_and_current_task(self) -> None:
        store = TaskStepsManagerStore()
        session_id = "session-1"

        store.create_session(session_id)
        result = store.apply(
            session_id=session_id,
            run_label="Run 1",
            action="init",
            tasks_text="Inspect repo\nRun checks\nWrite summary",
        )
        self.assertIn("SUCCESS", result)

        store.apply(
            session_id=session_id,
            run_label="Run 1",
            action="update",
            task_id=2,
            status="doing",
        )
        store.apply(
            session_id=session_id,
            run_label="Run 1",
            action="complete",
            task_id=1,
        )

        snapshot = store.snapshot(session_id)
        self.assertEqual(snapshot["tasks_total"], 3)
        self.assertEqual(snapshot["tasks_done"], 1)
        self.assertEqual(snapshot["tasks_doing"], 1)
        self.assertEqual(snapshot["tasks_todo"], 1)
        self.assertEqual(snapshot["progress_percent"], 33.33)
        self.assertEqual(snapshot["active_run_label"], "Run 1")
        self.assertEqual(snapshot["current_task"], "Run checks")

    def test_snapshot_aggregates_multiple_runs(self) -> None:
        store = TaskStepsManagerStore()
        session_id = "session-2"

        store.create_session(session_id)
        store.apply(
            session_id=session_id,
            run_label="Run 1",
            action="init",
            tasks_text="Task A\nTask B",
        )
        store.apply(
            session_id=session_id,
            run_label="Run 1",
            action="complete",
            task_id=1,
        )
        store.apply(
            session_id=session_id,
            run_label="Run 2",
            action="init",
            tasks_text="Task C\nTask D",
        )

        snapshot = store.snapshot(session_id)
        self.assertEqual(snapshot["tasks_total"], 4)
        self.assertEqual(snapshot["tasks_done"], 1)
        self.assertEqual(snapshot["tasks_doing"], 0)
        self.assertEqual(snapshot["tasks_todo"], 3)
        self.assertEqual(snapshot["progress_percent"], 25.0)
        self.assertEqual(len(snapshot["runs"]), 2)
        self.assertEqual(snapshot["active_run_label"], "Run 2")


if __name__ == "__main__":
    unittest.main()
