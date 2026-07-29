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
