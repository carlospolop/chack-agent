import fcntl
from pathlib import Path
import threading
import time
from typing import Any, cast

from chack_tools.chatgpt_remote_worker import ChatGPTRemoteWorker
from chack_tools.chatgpt_research_agents import _XHIGH_COMPAT_PROMPT_PREFIX


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
    worker.concurrency = 1
    worker.state_root = tmp_path
    worker._completion_lock = threading.Lock()
    return worker, client


def test_worker_concurrency_environment_is_bounded_to_five(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_CHATGPT_ASYNC_API_URL", "https://broker.example")
    monkeypatch.setenv("CHACK_CHATGPT_ASYNC_WORKER_SECRET", "worker-test-secret")
    monkeypatch.setenv("CHACK_CHATGPT_WORKER_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("CHACK_CHATGPT_WORKER_CONCURRENCY", "99")
    assert ChatGPTRemoteWorker().concurrency == 5

    monkeypatch.setenv("CHACK_CHATGPT_WORKER_CONCURRENCY", "0")
    assert ChatGPTRemoteWorker().concurrency == 1


def test_worker_forces_local_backend_and_posts_success(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)

    def browser(_self, prompt, *, run_state_path, partial_path):
        assert prompt == "P" * 200
        assert _self._timeout_seconds() == 5400
        _self._write_json(run_state_path, {"terminal_state": "extracted", "answer_chars": 2000})
        return "A" * 2000, "https://chatgpt.com/c/worker-test", {"terminal_state": "extracted"}

    monkeypatch.setattr("chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research", browser)
    worker.process(
        {
            "job_id": "job_00000000-0000-0000-0000-000000000001",
            "lease_id": "lease-1",
            "mode": "pro",
            "prompt": "P" * 200,
        }
    )

    assert worker._config("pro", 5400).chatgpt_execution_backend == "local"
    assert worker._config("pro", 5400).chatgpt_pro_timeout_seconds == 5400
    assert client.heartbeats[0][1]["stage"] == "launching_browser"
    _, completion = client.completions[0]
    assert completion["status"] == "SUCCEEDED"
    assert completion["result"] == "A" * 2000
    assert completion["metadata"]["execution_backend"] == "local_worker"
    assert "conversation_url" not in completion["metadata"]


def test_worker_accepts_xhigh_and_uses_its_mode_specific_timeout(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)

    def browser(_self, prompt, *, run_state_path, partial_path):
        assert prompt == "X" * 200
        assert _self.mode == "xhigh"
        assert _self.tool_name == "chatgptxhigh"
        assert _self._timeout_seconds() == 1234
        return "XHIGH_OK " + ("A" * 300), "", {
            "mode": "xhigh",
            "terminal_state": "extracted",
        }

    monkeypatch.setattr(
        "chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research",
        browser,
    )
    worker.process(
        {
            "job_id": "job_00000000-0000-0000-0000-000000000099",
            "lease_id": "lease-xhigh",
            "mode": "xhigh",
            "prompt": "X" * 200,
            "output_timeout_seconds": 1234,
        }
    )

    cfg = worker._config("xhigh", 1234)
    assert cfg.chatgpt_execution_backend == "local"
    assert cfg.chatgpt_xhigh_timeout_seconds == 1234
    _, completion = client.completions[0]
    assert completion["status"] == "SUCCEEDED"
    assert completion["result"].startswith("XHIGH_OK")
    assert completion["metadata"]["mode"] == "xhigh"


def test_worker_restores_xhigh_from_stale_broker_compatibility_envelope(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)
    original_prompt = "Compatibility prompt " + ("X" * 200)

    def browser(_self, prompt, *, run_state_path, partial_path):
        assert _self.mode == "xhigh"
        assert prompt == original_prompt
        assert _XHIGH_COMPAT_PROMPT_PREFIX not in prompt
        return "XHIGH_COMPAT_WORKER_OK " + ("A" * 300), "", {
            "mode": "xhigh",
            "terminal_state": "extracted",
        }

    monkeypatch.setattr(
        "chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research",
        browser,
    )
    worker.process(
        {
            "job_id": "job_00000000-0000-0000-0000-000000000098",
            "lease_id": "lease-xhigh-compat",
            "mode": "pro",
            "prompt": _XHIGH_COMPAT_PROMPT_PREFIX + original_prompt,
            "output_timeout_seconds": 5400,
        }
    )
    _, completion = client.completions[0]
    assert completion["status"] == "SUCCEEDED"
    assert completion["metadata"]["mode"] == "xhigh"


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


def test_worker_runs_five_browser_jobs_concurrently_with_isolated_artifacts(monkeypatch, tmp_path):
    worker, _ = _worker(tmp_path)
    worker.concurrency = 5
    worker.poll_seconds = 1
    jobs = [
        {
            "job_id": f"job_00000000-0000-0000-0000-0000000001{i:02d}",
            "lease_id": f"lease-{i}",
            "mode": "pro" if i % 2 else "deep",
            "prompt": f"Prompt {i} " * 30,
            "output_timeout_seconds": 1800 if i % 2 else 4500,
        }
        for i in range(5)
    ]

    class ConcurrentClient(FakeWorkerClient):
        def __init__(self):
            super().__init__()
            self.jobs = list(jobs)
            self.lock = threading.Lock()

        def lease(self, *, worker_id):
            assert worker_id == "test-worker"
            with self.lock:
                return self.jobs.pop(0) if self.jobs else None

        def heartbeat(self, job_id, **kwargs):
            with self.lock:
                return super().heartbeat(job_id, **kwargs)

        def complete(self, job_id, **kwargs):
            with self.lock:
                return super().complete(job_id, **kwargs)

    client = ConcurrentClient()
    worker.client = cast(Any, client)
    barrier = threading.Barrier(5)
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def browser(_self, prompt, *, run_state_path, partial_path):
        nonlocal active, maximum_active
        assert run_state_path.parent.name.startswith("job_")
        assert partial_path.parent == run_state_path.parent
        assert prompt.startswith("Prompt ")
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=5)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "parallel answer " + prompt[:20], "https://chatgpt.com/c/private", {"terminal_state": "extracted"}

    monkeypatch.setattr("chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research", browser)
    worker.run_until_idle()

    assert maximum_active == 5
    assert len(client.completions) == 5
    assert {job_id for job_id, _ in client.completions} == {job["job_id"] for job in jobs}
    assert all(payload["status"] == "SUCCEEDED" for _, payload in client.completions)
    assert all("conversation_url" not in payload["metadata"] for _, payload in client.completions)
    assert len(list((tmp_path / "jobs").glob("*/request.txt"))) == 5


def test_worker_refuses_retry_after_submission_marker_without_conversation_url(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)
    job_id = "job_00000000-0000-0000-0000-000000000004"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "browser-submission-attempted.json").write_text('{"attempted_at":1}')

    monkeypatch.setattr(
        "chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not resubmit")),
    )
    worker.process(
        {
            "job_id": job_id,
            "lease_id": "lease-4",
            "mode": "pro",
            "prompt": "P" * 200,
            "output_timeout_seconds": 1800,
            "attempt": 2,
        }
    )

    _, completion = client.completions[0]
    assert completion["status"] == "FAILED"
    assert completion["error_code"] == "AMBIGUOUS_PRIOR_BROWSER_SUBMISSION"


def test_worker_rejects_invalid_job_id_before_filesystem_use(tmp_path):
    worker, client = _worker(tmp_path)
    worker.process(
        {
            "job_id": "../../outside",
            "lease_id": "lease-bad",
            "mode": "pro",
            "prompt": "P" * 200,
        }
    )
    assert not client.completions
    assert not (tmp_path.parent / "outside").exists()


def test_worker_refuses_a_second_execution_lock_for_the_same_job(monkeypatch, tmp_path):
    worker, client = _worker(tmp_path)
    job_id = "job_00000000-0000-0000-0000-000000000005"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    lock_path = job_dir / ".execution.lock"

    monkeypatch.setattr(
        "chack_tools.chatgpt_remote_worker.ChatGPTWebResearchAgentTool._browser_research",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not launch twice")),
    )
    with lock_path.open("a+") as held_lock:
        fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker.process(
            {
                "job_id": job_id,
                "lease_id": "lease-duplicate",
                "mode": "deep",
                "prompt": "P" * 200,
            }
        )

    _, completion = client.completions[0]
    assert completion["status"] == "FAILED"
    assert completion["error_code"] == "DUPLICATE_ACTIVE_EXECUTION"
