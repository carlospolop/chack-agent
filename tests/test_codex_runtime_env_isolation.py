import json

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
