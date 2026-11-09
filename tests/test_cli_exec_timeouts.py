"""Claude Code and opencode runs get the agent's runtime budget, like Codex, Gemini and Copilot runs."""
from __future__ import annotations

import pytest

from chack_agent.backends import claude_code_backend, opencode_cli_backend
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

TIMEOUT_ENVS = (
    "CHACK_BUDGET_MAX_RUNTIME_SECONDS",
    "CHACK_CLAUDE_EXEC_TIMEOUT_SECONDS",
    "CHACK_OPENCODE_EXEC_TIMEOUT_SECONDS",
)


def _config(provider: str, model: str) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary=model, provider=provider),
        agent=AgentConfig(main_action="test", sub_action="test"),
        session=SessionConfig(max_turns=2),
        tools=ToolsConfig(exec_enabled=False, task_steps_manager_enabled=False),
        credentials=CredentialsConfig(anthropic_api_key="sk-ant-test", openai_api_key="sk-test"),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHACK_CLAUDE_HOME_BASE", str(tmp_path / "claude-homes"))
    for name in TIMEOUT_ENVS:
        monkeypatch.delenv(name, raising=False)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _run_missing_cli(backend, monkeypatch, tmp_path):
    """Runs the backend with a CLI path that does not exist; returns its start log message."""
    missing = str(tmp_path / "no-such-cli")
    if backend == "claude":
        monkeypatch.setenv("CLAUDE_CLI_PATH", missing)
        executor = claude_code_backend.build_executor(
            _config("claude", "claude-haiku-5-5"), system_prompt="system", max_turns=2,
            memory_max_messages=10, memory_reset_to_messages=5,
        )
        return executor._run_claude("hi"), "Starting Claude Code process"
    monkeypatch.setenv("OPENCODE_CLI_PATH", missing)
    executor = opencode_cli_backend.build_executor(
        _config("opencode", "gpt-6-sol"), system_prompt="system", max_turns=2,
        memory_max_messages=10, memory_reset_to_messages=5,
    )
    return executor._run_opencode("hi"), "Starting opencode process"


def _start_messages(caplog, start_message):
    return [record.getMessage() for record in caplog.records if start_message in record.getMessage()]


@pytest.mark.parametrize("backend", ["claude", "opencode"])
def test_a_run_without_its_own_timeout_gets_the_agents_runtime_budget(cli_env, monkeypatch, caplog, backend, tmp_path):
    # Dynamic's poc-agent has max_runtime_minutes: 45; a flat 900 s cut it off after 15.
    monkeypatch.setenv("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "2700")

    with caplog.at_level("INFO"):
        (output, _steps, _raw), start_message = _run_missing_cli(backend, monkeypatch, tmp_path)

    assert "not found" in output
    starts = _start_messages(caplog, start_message)
    assert starts and "timeout_seconds=3000" in starts[-1]


@pytest.mark.parametrize(
    ("backend", "timeout_env"),
    [("claude", "CHACK_CLAUDE_EXEC_TIMEOUT_SECONDS"), ("opencode", "CHACK_OPENCODE_EXEC_TIMEOUT_SECONDS")],
)
def test_an_explicit_timeout_still_wins_over_the_runtime_budget(cli_env, monkeypatch, caplog, backend, timeout_env, tmp_path):
    monkeypatch.setenv("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "2700")
    monkeypatch.setenv(timeout_env, "600")

    with caplog.at_level("INFO"):
        _result, start_message = _run_missing_cli(backend, monkeypatch, tmp_path)

    starts = _start_messages(caplog, start_message)
    assert starts and "timeout_seconds=600" in starts[-1]


@pytest.mark.parametrize("backend", ["claude", "opencode"])
def test_an_agent_without_a_runtime_budget_keeps_the_default(cli_env, monkeypatch, caplog, backend, tmp_path):
    with caplog.at_level("INFO"):
        _result, start_message = _run_missing_cli(backend, monkeypatch, tmp_path)

    starts = _start_messages(caplog, start_message)
    assert starts and "timeout_seconds=900" in starts[-1]
