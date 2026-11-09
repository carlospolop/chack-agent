from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chack_agent import Chack, RunModel
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)


class _Executor:
    def invoke(self, payload, context=None):
        del payload, context
        return {
            "output": "done",
            "intermediate_steps": [],
            "raw_result": SimpleNamespace(raw_responses=[]),
        }


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RUN_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("CHACK_TASK_SESSION_ID", raising=False)
    with patch("chack_agent.model_aliases._refresh_remote_alias_caches"), patch(
        "chack_agent.model_aliases._REMOTE_MODEL_CACHE",
        {"OPENAI_CHEAP_BUT_QUALITY": "gpt-6-sol"},
    ), patch("chack_agent.agent.build_executor", side_effect=lambda *a, **k: _Executor()):
        yield


def _agent(primary: str = "test-model") -> Chack:
    return Chack(
        ChackConfig(
            model=ModelConfig(primary=primary, provider="codex"),
            agent=AgentConfig(self_critique_enabled=False, thinking_effort="medium"),
            session=SessionConfig(long_term_memory_enabled=False),
            tools=replace(ToolsConfig(), min_tools_used=0),
            credentials=CredentialsConfig(),
            logging=LoggingConfig(level="ERROR"),
            system_prompt="test system",
            env={},
        )
    )


def _run(agent: Chack, session_id: str = "s"):
    return agent.run(
        session_id,
        "do work",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
        reuse_session_executor=True,
    )


def test_run_reports_the_concrete_model_an_alias_set_after_construction_ran():
    # Provider profiles set a tier alias on a live agent before its run.
    agent = _agent()
    agent.config.model.primary = "OPENAI_CHEAP_BUT_QUALITY"

    result = _run(agent)

    expected = RunModel(
        model="gpt-6-sol",
        alias="OPENAI_CHEAP_BUT_QUALITY",
        backend="codex",
        thinking_effort="medium",
    )
    assert result.run_model == expected
    assert agent.run_model == expected


def test_a_model_named_directly_drops_the_alias_of_an_earlier_run():
    agent = _agent("OPENAI_CHEAP_BUT_QUALITY")
    assert _run(agent, "first").run_model.alias == "OPENAI_CHEAP_BUT_QUALITY"

    agent.config.model.primary = "gpt-6-luna"
    result = _run(agent, "second")

    assert result.run_model.model == "gpt-6-luna"
    assert result.run_model.alias == ""


def test_a_reused_session_executor_reports_the_effort_it_was_built_with():
    agent = _agent("OPENAI_CHEAP_BUT_QUALITY")
    _run(agent)

    # The config changes, but the session keeps the executor that runs medium.
    agent.config.agent.thinking_effort = "high"
    result = _run(agent)

    assert result.run_model.thinking_effort == "medium"
    assert result.run_model.model == "gpt-6-sol"
