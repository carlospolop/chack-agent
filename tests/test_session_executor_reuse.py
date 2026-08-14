from __future__ import annotations

from dataclasses import replace
import inspect
from unittest.mock import Mock, patch

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


def _agent() -> Chack:
    return Chack(
        ChackConfig(
            model=ModelConfig(primary="test-model", provider="openai"),
            agent=AgentConfig(self_critique_enabled=False),
            session=SessionConfig(long_term_memory_enabled=False),
            tools=replace(ToolsConfig(), min_tools_used=0),
            credentials=CredentialsConfig(),
            logging=LoggingConfig(level="ERROR"),
            system_prompt="test system",
            env={},
        )
    )


def test_session_executor_reuse_is_public_on_sync_and_async_runs() -> None:
    assert "reuse_session_executor" in inspect.signature(Chack.run).parameters
    assert "reuse_session_executor" in inspect.signature(Chack.arun).parameters


def test_tool_overrides_remain_per_call_by_default() -> None:
    agent = _agent()
    tool = Mock(name="tool")

    with patch(
        "chack_agent.agent.build_executor",
        side_effect=[object(), object()],
    ) as build:
        first = agent._get_executor("review", tools_override=[tool])
        second = agent._get_executor("review", tools_override=[tool])

    assert first is not second
    assert build.call_count == 2
    assert not agent.has_session("review")


def test_tool_overrides_can_reuse_one_session_executor() -> None:
    agent = _agent()
    executor = object()
    tool = Mock(name="tool")

    with patch(
        "chack_agent.agent.build_executor",
        return_value=executor,
    ) as build:
        first = agent._get_executor(
            "review",
            tools_override=[tool],
            reuse_session_executor=True,
        )
        second = agent._get_executor(
            "review",
            tools_override=[tool],
            reuse_session_executor=True,
        )

    assert first is executor
    assert second is executor
    assert build.call_count == 1
    assert build.call_args.kwargs["tools_override"] == [tool]
    assert agent.has_session("review")


def test_reset_discards_a_persistent_tool_override_executor() -> None:
    agent = _agent()
    tool = Mock(name="tool")

    with patch(
        "chack_agent.agent.build_executor",
        side_effect=[object(), object()],
    ) as build:
        first = agent._get_executor(
            "review",
            tools_override=[tool],
            reuse_session_executor=True,
        )
        agent.reset_session("review", finalize_long_term_memory=False)
        second = agent._get_executor(
            "review",
            tools_override=[tool],
            reuse_session_executor=True,
        )

    assert first is not second
    assert build.call_count == 2
