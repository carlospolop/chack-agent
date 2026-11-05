from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter

from agents import function_tool
from mcp.server.fastmcp import FastMCP

from chack_agent.backends.chack_tools_mcp_server import _ServerPolicyState, _load_toolset, _register_tools
from chack_agent.backends.tool_payloads import CHACK_TOOLS_OVERRIDE_B64_ENV, serialize_tools_payload
from chack_tools import run_lifecycle
from chack_tools.tool_usage_state import budget_exempt, non_task_tool_count, register_budget_exempt_tools


def tools(suffix):
    """A probe tool and a result-saving tool, named per test so the process-wide registry cannot leak between tests."""
    @function_tool(name_override=f"probe_{suffix}")
    def probe() -> str:
        """Probe the target."""
        return "probed"

    @function_tool(name_override=f"save_verdict_{suffix}")
    def save_verdict(verdict: str) -> str:
        """Record the verdict."""
        return "recorded " + verdict

    return probe, budget_exempt(save_verdict)


def call(server, name, arguments=None):
    try:
        result = asyncio.run(server.call_tool(name, arguments or {}))
    except Exception as error:  # FastMCP raises the tool's error to the client
        return str(error)
    return "\n".join(str(block) for block in result)


def test_codex_tool_server_still_records_results_after_the_budget_is_spent(monkeypatch, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    probe, save_verdict = tools(suffix)
    monkeypatch.setattr(run_lifecycle, "_STATE_ROOT_ENV", "CHACK_TEST_STATE_ROOT")
    monkeypatch.setenv("CHACK_TEST_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("CHACK_MODEL_PROVIDER", "codex")
    # The Codex backend hands the tools to this server process pickled; the mark must survive that.
    monkeypatch.setenv(CHACK_TOOLS_OVERRIDE_B64_ENV, serialize_tools_payload([probe, save_verdict]))
    server = FastMCP("budget-test")
    _register_tools(server, _load_toolset(), _ServerPolicyState(
        require_task_steps_manager_init_first=False, max_non_task_tools=2, session_id=f"session-{suffix}"))

    # The save between two probes does not use up a slot of the budget of two.
    assert "probed" in call(server, f"probe_{suffix}")
    assert "recorded early" in call(server, f"save_verdict_{suffix}", {"verdict": "early"})
    assert "probed" in call(server, f"probe_{suffix}")
    assert "Tool budget reached" in call(server, f"probe_{suffix}")
    assert "recorded safe" in call(server, f"save_verdict_{suffix}", {"verdict": "safe"})
    assert "recorded vulnerable" in call(server, f"save_verdict_{suffix}", {"verdict": "vulnerable"})
    assert "Tool budget reached" in call(server, f"probe_{suffix}")


def test_in_process_budget_count_skips_registered_result_tools_only():
    suffix = uuid.uuid4().hex[:8]
    probe, save_verdict = tools(suffix)
    usage = Counter({f"probe_{suffix}": 3, f"save_verdict_{suffix}": 5, "task_steps_manager": 4})
    assert non_task_tool_count(usage) == 8

    register_budget_exempt_tools([probe, save_verdict])

    assert non_task_tool_count(usage) == 3
    assert non_task_tool_count(Counter({f"mcp__chack_tools__save_verdict_{suffix}": 2})) == 0
