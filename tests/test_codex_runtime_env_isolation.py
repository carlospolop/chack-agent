import json
import os
import sys
import time

from chack_agent.backends.codex_backend import (
    CodexExecutor,
    _resolve_codex_exec_cwd,
    _resolve_codex_exec_timeout,
)


def test_executor_runtime_env_wins_over_process_env(monkeypatch):
    monkeypatch.setenv("CHACK_MCP_TOOL_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("CHACK_CODEX_EXEC_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("CHACK_EXEC_CWD", "/wrong-agent")

    executor = CodexExecutor.__new__(CodexExecutor)
    executor._runtime_env_json = json.dumps(
        {
            "CHACK_MCP_TOOL_TIMEOUT_SECONDS": "10800",
            "CHACK_CODEX_EXEC_TIMEOUT_SECONDS": "10800",
            "CHACK_EXEC_CWD": "/dependency-master",
        }
    )

    runtime_env = executor._runtime_env()
    assert executor._runtime_env_value("CHACK_MCP_TOOL_TIMEOUT_SECONDS") == "10800"
    assert _resolve_codex_exec_timeout("dependency_vuln", runtime_env) == 10800
    assert _resolve_codex_exec_cwd(runtime_env) == "/dependency-master"


def test_executor_runtime_env_falls_back_to_process_env(monkeypatch):
    monkeypatch.setenv("CHACK_CODEX_EXEC_TIMEOUT_SECONDS", "321")
    monkeypatch.setenv("CHACK_EXEC_CWD", "/process-default")

    executor = CodexExecutor.__new__(CodexExecutor)
    executor._runtime_env_json = "{}"

    assert _resolve_codex_exec_timeout("check_auditor", executor._runtime_env()) == 321
    assert _resolve_codex_exec_cwd(executor._runtime_env()) == "/process-default"


def test_silent_codex_process_cannot_bypass_execution_timeout(tmp_path):
    executor = CodexExecutor.__new__(CodexExecutor)
    executor._runtime_env_json = json.dumps(
        {
            "CHACK_CODEX_EXEC_TIMEOUT_SECONDS": "1",
            "CHACK_EXEC_CWD": str(tmp_path),
        }
    )
    executor._sub_action = "silent_timeout_test"
    executor._model_name = "test-model"
    executor._model_provider = "codex"
    executor._thread_id = None
    executor._build_command = lambda: [
        sys.executable,
        "-c",
        "import sys, time; sys.stdin.read(); time.sleep(30)",
    ]
    executor._build_env = lambda: dict(os.environ)
    executor._log_codex_failure = lambda *args, **kwargs: None

    started_at = time.monotonic()
    output, steps, raw_result = executor._run_codex_once(
        "test prompt",
        allow_api_key_fallback=False,
    )
    elapsed = time.monotonic() - started_at

    assert output == "ERROR: Codex execution timed out after 1s."
    assert steps == []
    assert raw_result.raw_responses == []
    assert elapsed < 5
