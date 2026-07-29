from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from chack_agent import Chack
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)


def _agent(*, idle_minutes: int = 0, max_age_minutes: int = 0) -> Chack:
    return Chack(
        ChackConfig(
            model=ModelConfig(primary="test-model", provider="openai"),
            agent=AgentConfig(self_critique_enabled=False),
            session=SessionConfig(
                long_term_memory_enabled=False,
                idle_reset_minutes=idle_minutes,
                max_age_minutes=max_age_minutes,
            ),
            tools=replace(ToolsConfig(), min_tools_used=0),
            credentials=CredentialsConfig(),
            logging=LoggingConfig(level="ERROR"),
            system_prompt="test system",
            env={},
        )
    )


def _seed_live_session(agent: Chack, session_id: str, *, started: float, last: float) -> None:
    agent._executors[f"{session_id}:test"] = object()
    agent._session_started_at[session_id] = started
    agent._last_activity_at[session_id] = last


def test_session_does_not_rotate_before_boundaries() -> None:
    agent = _agent(idle_minutes=120, max_age_minutes=120)
    _seed_live_session(agent, "chat", started=1_000, last=1_050)

    with patch("chack_agent.agent.time.time", return_value=1_100):
        reason = agent._prepare_session_for_run("chat")

    assert reason is None
    assert agent.has_session("chat")
    assert agent._session_started_at["chat"] == 1_000


def test_session_rotates_at_absolute_max_age() -> None:
    agent = _agent(idle_minutes=120, max_age_minutes=120)
    _seed_live_session(agent, "chat", started=1_000, last=8_100)

    with patch("chack_agent.agent.time.time", return_value=8_200):
        reason = agent._prepare_session_for_run("chat")

    assert reason == "max_age"
    assert not agent.has_session("chat")
    assert agent._session_started_at["chat"] == 8_200


def test_session_rotates_after_idle_boundary() -> None:
    agent = _agent(idle_minutes=60, max_age_minutes=0)
    _seed_live_session(agent, "chat", started=1_000, last=2_000)

    with patch("chack_agent.agent.time.time", return_value=5_600):
        reason = agent._prepare_session_for_run("chat")

    assert reason == "idle"
    assert not agent.has_session("chat")
    assert agent._session_started_at["chat"] == 5_600


def test_reset_clears_all_session_lifecycle_state() -> None:
    agent = _agent(idle_minutes=120, max_age_minutes=120)
    _seed_live_session(agent, "chat", started=1_000, last=1_050)

    agent.reset_session("chat", finalize_long_term_memory=False)

    assert not agent.has_session("chat")
    assert "chat" not in agent._session_started_at
    assert "chat" not in agent._last_activity_at


def test_reset_cleans_retained_runtime_artifacts() -> None:
    class RetainedExecutor:
        def __init__(self):
            self.cleaned = 0

        def _runtime_env_value(self, name, default=""):
            assert name == "CHACK_CLEANUP_CODEX_HOME_AFTER_RUN"
            return "true"

        def cleanup_runtime_artifacts(self):
            self.cleaned += 1

    agent = _agent()
    executor = RetainedExecutor()
    agent._executors["chat:retained"] = executor

    agent.reset_session("chat", finalize_long_term_memory=False)

    assert executor.cleaned == 1
    assert not agent.has_session("chat")


def test_reuse_session_caches_executor_with_empty_tool_override(monkeypatch) -> None:
    agent = _agent()
    built = []
    executor = object()

    def fake_build_executor(*args, **kwargs):
        built.append((args, kwargs))
        return executor

    monkeypatch.setattr("chack_agent.agent.build_executor", fake_build_executor)

    first = agent._get_executor(
        "chat",
        tools_override=[],
        reuse_session=True,
    )
    second = agent._get_executor(
        "chat",
        tools_override=[],
        reuse_session=True,
    )

    assert first is executor
    assert second is executor
    assert len(built) == 1
    assert built[0][1]["tools_override"] == []


def test_reused_live_session_suppresses_repeated_system_prompt(monkeypatch) -> None:
    class RetainedExecutor:
        def __init__(self):
            self.suppressed = 0
            self.inputs = []

        def suppress_system_prompt_for_next_invocation(self):
            self.suppressed += 1

        def invoke(self, payload, context=None):
            del context
            self.inputs.append(payload["input"])
            return {
                "output": "ok",
                "intermediate_steps": [],
                "raw_result": None,
            }

        def _runtime_env_value(self, _name, default=""):
            return default

    agent = _agent()
    executor = RetainedExecutor()
    monkeypatch.setattr(
        "chack_agent.agent.build_executor",
        lambda *_args, **_kwargs: executor,
    )

    agent.run(
        "chat",
        "first full request",
        tools_override=[],
        reuse_session=True,
    )
    agent.run(
        "chat",
        "short continuation",
        tools_override=[],
        reuse_session=True,
    )

    assert executor.inputs == ["first full request", "short continuation"]
    assert executor.suppressed == 1
