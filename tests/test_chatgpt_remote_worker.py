from pathlib import Path
from typing import Any, cast

from chack_tools.chatgpt_remote_worker import ChatGPTRemoteWorker


class FakeWorkerClient:
    def __init__(self):
        self.completions = []
        self.heartbeats = []

    def heartbeat(self, job_id, **kwargs):
        self.heartbeats.append((job_id, kwargs))
        return {"status": "RUNNING", "cancel_requested": False}

    def complete(self, job_id, **kwargs):
        self.completions.append((job_id, kwargs))
        return {"status": kwargs["status"]}


def _worker(tmp_path: Path) -> tuple[ChatGPTRemoteWorker, FakeWorkerClient]:
    worker = ChatGPTRemoteWorker.__new__(ChatGPTRemoteWorker)
    client = FakeWorkerClient()
    worker.client = cast(Any, client)
    worker.cdp_url = "http://127.0.0.1:9226"
    worker.worker_id = "test-worker"
    worker.poll_seconds = 2
    worker.heartbeat_seconds = 10
    worker.state_root = tmp_path
    return worker, client


def test_worker_forces_local_backend_and_posts_success(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)

    def browser(_self, prompt, *, run_state_path, partial_path):
        assert prompt == "P" * 200
        _self._write_json(run_state_path, {"terminal_state": "extracted", "answer_chars": 2000})
        return "A" * 2000, "https://chatgpt.com/c/worker-test", {"terminal_state": "extracted"}

    monkeypatch.setattr("chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research", browser)
    worker.process(
        {
            "job_id": "job_00000000-0000-0000-0000-000000000001",
            "lease_id": "lease-1",
            "mode": "pro",
            "prompt": "P" * 200,
            "output_timeout_seconds": 1800,
        }
    )

    assert worker._config("pro", 1800).chatgpt_execution_backend == "local"
    assert client.heartbeats[0][1]["stage"] == "launching_browser"
    _, completion = client.completions[0]
    assert completion["status"] == "SUCCEEDED"
    assert completion["result"] == "A" * 2000
    assert completion["metadata"]["execution_backend"] == "local_worker"
    assert "conversation_url" not in completion["metadata"]


def test_worker_posts_timeout_with_partial_output(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)

    def browser(_self, _prompt, *, run_state_path, partial_path):
        _self._write_json(run_state_path, {"terminal_state": "timeout", "conversation_url": "https://chatgpt.com/c/partial"})
        _self._write_partial(partial_path, "partial answer")
        raise RuntimeError("browser timed out at /home/tester/private/path")

    monkeypatch.setattr("chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research", browser)
    worker.process(
        {
            "job_id": "job_00000000-0000-0000-0000-000000000002",
            "lease_id": "lease-2",
            "mode": "deep",
            "prompt": "P" * 200,
            "output_timeout_seconds": 4500,
        }
    )

    _, completion = client.completions[0]
    assert completion["status"] == "TIMED_OUT"
    assert completion["partial_result"] == "partial answer"
    assert completion["error_code"] == "BROWSER_OUTPUT_TIMEOUT"
    assert "/home/tester" not in completion["error_message"]
    assert "conversation_url" not in completion["metadata"]


def test_worker_refuses_ambiguous_retry_after_prior_browser_submission(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)
    job_id = "job_00000000-0000-0000-0000-000000000003"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "chatgpt-run.json").write_text(
        '{"conversation_url":"https://chatgpt.com/c/private","terminal_state":"running"}'
    )
    (job_dir / "chatgpt-pro-partial.md").write_text("preserved partial")

    def should_not_launch(*_args, **_kwargs):
        raise AssertionError("ambiguous retry must not submit to ChatGPT again")

    monkeypatch.setattr(
        "chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research",
        should_not_launch,
    )
    worker.process(
        {
            "job_id": job_id,
            "lease_id": "lease-3",
            "mode": "pro",
            "prompt": "P" * 200,
            "output_timeout_seconds": 1800,
            "attempt": 2,
        }
    )

    _, completion = client.completions[0]
    assert completion["status"] == "FAILED"
    assert completion["partial_result"] == "preserved partial"
    assert completion["error_code"] == "AMBIGUOUS_PRIOR_BROWSER_SUBMISSION"
    assert completion["metadata"]["prior_browser_submission_detected"] is True
    assert "conversation_url" not in completion["metadata"]
