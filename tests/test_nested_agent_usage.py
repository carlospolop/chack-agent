"""Agents a run starts spend on that run: their usage reaches its totals once.

Each test runs a parent ``Chack.run`` whose model call uses a tool that starts
a nested agent run (as the research tools do), with fake model usage:

* in the parent's process (API backends),
* in the real MCP tool server, a separate process (CLI backends),
* in an isolated researcher process (researcher administrator).

The parent's ``nested_usage_by_model``, ``total_cost`` and usage observer must
count the nested run's usage exactly once.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest
from agents import function_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from chack_agent import Chack
from chack_agent.backends.tool_payloads import serialize_tools_payload
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)
from chack_agent.live_cost_state import report_live_usage
from chack_agent.pricing import ModelPricing, PricingTable
from chack_agent.usage_observer import observe
from chack_tools.run_lifecycle import run_state_dir
from chack_tools.task_steps_manager_state import current_session_id

# Tools defined here travel to tool processes by value, like a caller's own tools.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_USAGE = (1200, 80, 300, 0)
NESTED_USAGE = (500, 50, 100, 0)
PRICES = {"parent-model": (2.0, 0.2, 8.0), "sub-model": (1.0, 0.1, 4.0)}


@pytest.fixture
def isolated_run_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RUN_STATE_DIR", str(tmp_path / "run-state"))
    monkeypatch.delenv("CHACK_TASK_SESSION_ID", raising=False)
    yield tmp_path / "run-state"


def _config(model: str) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary=model, provider="openai"),
        agent=AgentConfig(self_critique_enabled=False, max_cost_usd=10, main_action="test", sub_action="nested-usage"),
        session=SessionConfig(long_term_memory_enabled=False),
        tools=replace(ToolsConfig(), min_tools_used=0),
        credentials=CredentialsConfig(),
        logging=LoggingConfig(level="ERROR"),
        system_prompt="test system",
        env={},
    )


def _pricing() -> PricingTable:
    return PricingTable(models={name: ModelPricing(*rates) for name, rates in PRICES.items()})


def _cost(model: str, usage) -> float:
    input_rate, cached_rate, output_rate = PRICES[model]
    prompt, completion, cached, _ = usage
    return ((prompt - cached) * input_rate + cached * cached_rate + completion * output_rate) / 1_000_000


def _model_result(output: str, usage) -> dict:
    prompt, completion, cached, _ = usage
    raw_usage = SimpleNamespace(
        input_tokens=prompt,
        output_tokens=completion,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    return {
        "output": output,
        "intermediate_steps": [],
        "raw_result": SimpleNamespace(raw_responses=[SimpleNamespace(usage=raw_usage)]),
    }


class _NestedModel:
    """A nested agent's backend: reports its model call live, then returns it."""

    def invoke(self, payload, context=None):
        del payload, context
        report_live_usage(
            "sub-model",
            prompt_tokens=NESTED_USAGE[0],
            completion_tokens=NESTED_USAGE[1],
            cached_prompt_tokens=NESTED_USAGE[2],
        )
        return _model_result("nested findings", NESTED_USAGE)


class _NestedChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _NestedModel()


def _run_nested_agent(prompt: str) -> str:
    # What the research tools do: start a run billed to the calling run.
    nested = _NestedChack(_config("sub-model"))
    nested._pricing = _pricing()
    result = nested.run(
        "nested-research",
        prompt,
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
        usage_session_id=current_session_id(),
    )
    return result.output


@function_tool(name_override="research_with_agent")
def research_with_agent(prompt: str) -> str:
    """Research a topic with a nested agent."""
    return _run_nested_agent(prompt)


class _DelegatingNestedModel(_NestedModel):
    """A nested agent whose own tool starts one more agent."""

    def invoke(self, payload, context=None):
        assert _run_nested_agent("find the owner's registrar") == "nested findings"
        return super().invoke(payload, context)


class _DelegatingNestedChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _DelegatingNestedModel()


@function_tool(name_override="research_with_agent")
def research_with_delegating_agent(prompt: str) -> str:
    """Research a topic with a nested agent that delegates part of it."""
    nested = _DelegatingNestedChack(_config("sub-model"))
    nested._pricing = _pricing()
    return nested.run(
        "nested-research",
        prompt,
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
        usage_session_id=current_session_id(),
    ).output


async def _call_through_mcp_tool_server(tool=research_with_agent) -> str:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
            "CHACK_MODEL_PROVIDER": "codex",
            "CHACK_DEFAULT_MODEL": "sub-model",
            "CHACK_TOOLS_CONFIG_JSON": "{}",
            "CHACK_ALLOWED_TOOLS_JSON": '["research_with_agent"]',
            "CHACK_TOOLS_APPEND_B64": serialize_tools_payload([tool]),
            "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST": "0",
            # What every CLI backend hands its tool server.
            "CHACK_TASK_SESSION_ID": current_session_id() or "",
            "CHACK_RUN_STATE_DIR": run_state_dir(),
        }
    )
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "chack_agent.backends.chack_tools_mcp_server"],
        env=env,
    )
    async with stdio_client(server) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool("research_with_agent", {"prompt": "find the owner"})
            assert not result.isError, result.content
            return result.content[0].text


class _CliModelCallingTool:
    """A CLI backend whose model calls a tool served by the MCP tool server."""

    tool = research_with_agent

    def invoke(self, payload, context=None):
        del payload, context
        assert asyncio.run(_call_through_mcp_tool_server(self.tool)) == "nested findings"
        return _model_result("parent answer", MAIN_USAGE)


class _ApiModelCallingTool:
    """An API backend whose model calls the tool in the run's own process."""

    def invoke(self, payload, context=None):
        del payload, context
        assert _run_nested_agent("find the owner") == "nested findings"
        return _model_result("parent answer", MAIN_USAGE)


class _ModelStartingIsolatedResearcher:
    """A researcher administrator call: the researcher runs in its own process."""

    evidence_dir = ""

    def invoke(self, payload, context=None):
        del payload, context
        import chack_tools.researcher_administrator_agent as admin_mod

        outcome = admin_mod._run_researcher_in_process(
            research_with_agent,
            {"prompt": "find the owner"},
            evidence_dir=self.evidence_dir,
            cancel_event=threading.Event(),
        )
        assert outcome.get("output") == "nested findings", outcome
        return _model_result("parent answer", MAIN_USAGE)


class _ParentChack(Chack):
    executor = None

    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return self.executor


def _run_parent(executor):
    parent = _ParentChack(_config("parent-model"))
    parent._pricing = _pricing()
    parent.executor = executor
    events = []
    with observe(events.append):
        result = parent.run(
            "parent-run",
            "who owns the host?",
            enable_self_critique=False,
            require_task_steps_manager_init_first=False,
        )
    return result, events


def _observed(events, model: str):
    rows = [event for event in events if event.model == model]
    return (
        sum(event.prompt_tokens for event in rows),
        sum(event.completion_tokens for event in rows),
        sum(event.cached_prompt_tokens for event in rows),
        sum(event.cache_write_tokens for event in rows),
    )


def _assert_nested_counted_once(result, events, run_state: Path, agents: int = 1) -> None:
    nested = tuple(value * agents for value in NESTED_USAGE)
    assert (result.prompt_tokens, result.completion_tokens, result.cached_prompt_tokens) == MAIN_USAGE[:3]
    assert result.nested_usage_by_model == {"sub-model": nested}
    assert result.nested_agent_runs == agents
    assert result.total_cost == pytest.approx(_cost("parent-model", MAIN_USAGE) + _cost("sub-model", nested))
    # The caller's usage observer saw each nested model call once.
    assert _observed(events, "sub-model") == nested
    assert sum(event.cost_usd for event in events if event.model == "sub-model") == pytest.approx(
        _cost("sub-model", nested)
    )
    # Nothing is left for a later run with the same session.
    assert not list(run_state.glob("*.token-usage.json"))


def test_agent_started_by_an_mcp_served_tool_counts_once_in_the_cli_run(isolated_run_state):
    result, events = _run_parent(_CliModelCallingTool())

    _assert_nested_counted_once(result, events, isolated_run_state)


def test_agents_nested_two_deep_in_the_mcp_tool_server_count_once_in_the_cli_run(isolated_run_state):
    executor = _CliModelCallingTool()
    executor.tool = research_with_delegating_agent

    result, events = _run_parent(executor)

    _assert_nested_counted_once(result, events, isolated_run_state, agents=2)


def test_agent_started_by_an_in_process_tool_counts_once_in_the_run(isolated_run_state):
    result, events = _run_parent(_ApiModelCallingTool())

    _assert_nested_counted_once(result, events, isolated_run_state)


def test_agent_in_an_isolated_researcher_process_counts_once_in_the_run(isolated_run_state, tmp_path):
    executor = _ModelStartingIsolatedResearcher()
    executor.evidence_dir = str(tmp_path / "evidence")
    os.makedirs(executor.evidence_dir, exist_ok=True)

    result, events = _run_parent(executor)

    _assert_nested_counted_once(result, events, isolated_run_state)
