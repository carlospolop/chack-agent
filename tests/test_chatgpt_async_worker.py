import json

import pytest

from chack_tools.chatgpt_async_worker import BrokerWorkerError, ChatGPTAsyncWorker


def _worker():
    return ChatGPTAsyncWorker(
        api_url="https://broker.example.test",
        worker_secret="worker-secret",
        worker_id="test-worker",
        heartbeat_seconds=10,
    )


def test_worker_requires_remote_https_broker_and_distinct_worker_secret():
    with pytest.raises(BrokerWorkerError, match="HTTPS"):
        ChatGPTAsyncWorker(api_url="http://127.0.0.1:8080", worker_secret="secret")
    with pytest.raises(BrokerWorkerError, match="secret"):
        ChatGPTAsyncWorker(api_url="https://broker.example.test", worker_secret="")


def test_worker_leases_from_remote_service_and_does_not_log_prompt(monkeypatch, capsys):
    worker = _worker()
    calls = []
    prompt = "Private research request that must never be logged."
    lease = {
        "job_id": "job_123",
        "lease_id": "lease-123",
        "mode": "pro",
        "prompt": prompt,
        "output_timeout_seconds": 1800,
    }

    def lease_request(*, worker_id):
        calls.append(worker_id)
        return lease

    monkeypatch.setattr(worker.client, "lease", lease_request)
    monkeypatch.setattr(worker, "execute_lease", lambda value: value["job_id"])

    assert worker.run_once() is True
    assert calls == ["test-worker"]
    output = capsys.readouterr().out
    assert "job_123" in output
    assert prompt not in output
    assert "worker-secret" not in output


def test_worker_returns_browser_result_to_broker(monkeypatch):
    worker = _worker()
    completions = []

    class FakeHelper:
        def __init__(self, config, mode):
            assert mode == "pro"

        def _browser_research(self, prompt, *, run_state_path, partial_path):
            assert prompt == "Research this sufficiently detailed private request."
            run_state_path.write_text(json.dumps({"terminal_state": "extracted", "answer_chars": 300}))
            return "A" * 300, "https://chatgpt.com/c/private", {"mode": "pro", "terminal_state": "extracted"}

    monkeypatch.setattr("chack_tools.chatgpt_async_worker.ChatGPTWebResearchAgentTool", FakeHelper)
    monkeypatch.setattr(worker, "_complete", lambda job_id, payload: completions.append((job_id, payload)))

    job_id = worker.execute_lease(
        {
            "job_id": "job_123",
            "lease_id": "lease-123",
            "mode": "pro",
            "prompt": "Research this sufficiently detailed private request.",
            "output_timeout_seconds": 1800,
        }
    )

    assert job_id == "job_123"
    assert completions[0][0] == "job_123"
    assert completions[0][1]["status"] == "SUCCEEDED"
    assert completions[0][1]["result"] == "A" * 300
    assert completions[0][1]["metadata"]["execution_backend"] == "local_worker"
    assert "conversation_url" not in completions[0][1]["metadata"]
