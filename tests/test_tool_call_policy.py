from __future__ import annotations

import asyncio
import json
import os
import re

import pytest
from agents import function_tool
from agents.tool_context import ToolContext
from agents.usage import Usage
from mcp.server.fastmcp import FastMCP

from chack_agent.backends.chack_tools_mcp_server import (
    _ServerPolicyState,
    _register_tools,
    _tool_call_policy,
)
from chack_tools.config import ToolsConfig
from chack_tools.tool_call_policy import (
    ToolCallPolicy,
    ToolResultReuse,
    has_exec_tool,
    reuse_tool_results,
)
from chack_tools.tool_usage_state import budget_exempt


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _counting_tool(name: str, output=lambda args: f"result for {args}"):
    calls: list[dict] = []

    @function_tool(name_override=name)
    def _tool(query: str) -> str:
        """Test tool."""
        calls.append({"query": query})
        return output(query)

    return _tool, calls


def _invoke(tool, raw_args: str) -> str:
    ctx = ToolContext(
        context=None,
        usage=Usage(),
        tool_name=tool.name,
        tool_call_id="policy-test",
        tool_arguments=raw_args,
    )
    return asyncio.run(tool.on_invoke_tool(ctx, raw_args))


def _server(
    tools,
    *,
    max_tools: int = 0,
    policy: ToolCallPolicy | None = None,
    session_id: str = "",
) -> FastMCP:
    server = FastMCP("policy-test")
    _register_tools(
        server,
        tools,
        _ServerPolicyState(
            require_task_steps_manager_init_first=False,
            max_non_task_tools=max_tools,
            session_id=session_id,
            call_policy=policy or ToolCallPolicy(ToolsConfig()),
        ),
    )
    return server


def _call(server: FastMCP, name: str, args: dict) -> str:
    result = asyncio.run(server.call_tool(name, args))
    blocks = result[0] if isinstance(result, tuple) else result
    return "\n".join(getattr(block, "text", str(block)) for block in blocks)


def test_identical_search_is_answered_without_running_the_tool_again() -> None:
    search, calls = _counting_tool("search_web")
    server = _server([search])

    first = _call(server, "search_web", {"query": "cve-2026-1"})
    second = _call(server, "search_web", {"query": "cve-2026-1"})

    assert len(calls) == 1
    assert "result for cve-2026-1" in first
    assert "result for cve-2026-1" in second
    assert "not run again" in second


def test_different_arguments_still_run_the_tool() -> None:
    search, calls = _counting_tool("search_web")
    server = _server([search])

    _call(server, "search_web", {"query": "a"})
    _call(server, "search_web", {"query": "b"})

    assert [c["query"] for c in calls] == ["a", "b"]


def test_reused_answer_does_not_spend_the_tool_budget(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHACK_RUN_STATE_DIR", str(tmp_path))
    search, calls = _counting_tool("search_web")
    server = _server([search], max_tools=1, session_id=f"budget-{tmp_path.name}")

    _call(server, "search_web", {"query": "a"})
    reused = _call(server, "search_web", {"query": "a"})
    with pytest.raises(Exception, match="Tool budget reached"):
        _call(server, "search_web", {"query": "b"})

    assert "result for a" in reused
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["exec", "run_shell_command", "bash", "python_script"])
def test_shell_and_script_tools_always_run(name: str) -> None:
    tool, calls = _counting_tool(name)
    server = _server([tool])

    _call(server, name, {"query": "ls"})
    _call(server, name, {"query": "ls"})

    assert len(calls) == 2


@pytest.mark.parametrize("name", ["send_http_request", "probe", "browser_navigate", "verify_finding"])
def test_tools_acting_on_a_live_target_always_run(name: str) -> None:
    tool, calls = _counting_tool(name)
    server = _server([tool])

    _call(server, name, {"query": "GET /"})
    _call(server, name, {"query": "GET /"})

    assert len(calls) == 2


@pytest.mark.parametrize(
    "name", ["poll_researchers_async", "get_researcher_result", "list_research_artifacts", "job_status"]
)
def test_tools_reporting_changing_state_always_run(name: str) -> None:
    tool, calls = _counting_tool(name)
    server = _server([tool])

    _call(server, name, {"query": "x"})
    _call(server, name, {"query": "x"})

    assert len(calls) == 2


def test_a_state_changing_call_drops_reused_results() -> None:
    search, search_calls = _counting_tool("grep_research_artifacts")
    register, _ = _counting_tool("register_research_artifact")
    server = _server([search, register])

    _call(server, "grep_research_artifacts", {"query": "token"})
    _call(server, "register_research_artifact", {"query": "new file"})
    _call(server, "grep_research_artifacts", {"query": "token"})

    assert len(search_calls) == 2


def test_result_recording_tools_are_never_reused_and_reset_reuse() -> None:
    search, search_calls = _counting_tool("search_web")
    record, record_calls = _counting_tool("finding_tool")
    budget_exempt(record)
    server = _server([search, record])

    _call(server, "search_web", {"query": "q"})
    _call(server, "finding_tool", {"query": "f"})
    _call(server, "finding_tool", {"query": "f"})
    _call(server, "search_web", {"query": "q"})

    assert len(record_calls) == 2
    assert len(search_calls) == 2


def test_a_tool_can_opt_in_or_out_of_reuse() -> None:
    listing, listing_calls = _counting_tool("list_models")
    reuse_tool_results(listing, True)
    fetch, fetch_calls = _counting_tool("fetch_page")
    reuse_tool_results(fetch, False)
    server = _server([listing, fetch])

    for _ in range(2):
        _call(server, "list_models", {"query": "x"})
        _call(server, "fetch_page", {"query": "x"})

    assert len(listing_calls) == 1
    assert len(fetch_calls) == 2


def test_reuse_expires_after_the_window() -> None:
    clock = _Clock()
    reuse = ToolResultReuse(window_seconds=60, max_chars=1000, clock=clock)
    tool, _ = _counting_tool("search_web")

    reuse.store(tool, {"query": "q"}, "answer")
    clock.now += 59
    assert "answer" in (reuse.lookup(tool, {"query": "q"}) or "")
    clock.now += 2
    assert reuse.lookup(tool, {"query": "q"}) is None


def test_argument_order_and_encoding_do_not_matter() -> None:
    reuse = ToolResultReuse(window_seconds=60, max_chars=1000)
    tool, _ = _counting_tool("search_web")

    reuse.store(tool, '{"a": 1, "b": "x"}', "answer")

    assert reuse.lookup(tool, {"b": "x", "a": 1}) is not None


def test_long_and_failed_results_are_not_kept() -> None:
    reuse = ToolResultReuse(window_seconds=60, max_chars=10)
    tool, _ = _counting_tool("search_web")

    reuse.store(tool, {"query": "long"}, "x" * 11)
    reuse.store(tool, {"query": "err"}, "ERROR: upstream timed out")

    assert reuse.lookup(tool, {"query": "long"}) is None
    assert reuse.lookup(tool, {"query": "err"}) is None


def test_zero_window_disables_reuse() -> None:
    search, calls = _counting_tool("search_web")
    server = _server([search], policy=ToolCallPolicy(ToolsConfig(tool_result_reuse_seconds=0)))

    _call(server, "search_web", {"query": "q"})
    _call(server, "search_web", {"query": "q"})

    assert len(calls) == 2


def _saved_path(text: str) -> str:
    match = re.search(r"saved at (\S+?);", text)
    assert match, text
    return match.group(1)


def test_large_output_is_saved_to_a_file_and_previewed(tmp_path) -> None:
    big = "".join(f"line {i}\n" for i in range(20000))
    tool, _ = _counting_tool("fetch_page", output=lambda _q: big)
    policy = ToolCallPolicy(
        ToolsConfig(tool_output_max_chars=5000, tool_output_preview_chars=1200, tool_output_dir=str(tmp_path))
    )
    server = _server([tool], policy=policy)

    shown = _call(server, "fetch_page", {"query": "x"})

    assert len(shown) < 2000
    assert shown.startswith("line 0\n")
    assert "line 19999" in shown
    path = _saved_path(shown)
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == big


def test_output_that_fits_is_returned_unchanged(tmp_path) -> None:
    tool, _ = _counting_tool("fetch_page", output=lambda _q: "short page")
    policy = ToolCallPolicy(ToolsConfig(tool_output_dir=str(tmp_path)))
    server = _server([tool], policy=policy)

    assert _call(server, "fetch_page", {"query": "x"}) == "short page"
    assert os.listdir(tmp_path) == []


def test_mcp_token_limit_env_still_lowers_the_output_limit(monkeypatch) -> None:
    monkeypatch.setenv("CHACK_MCP_TOOL_MAX_TOKENS", "100")

    policy = _tool_call_policy({})

    assert policy.max_chars == 400
    assert len(policy.fit("y" * 1000)) < 1000


def test_api_agent_without_shell_gets_a_preview_and_no_file_path(tmp_path) -> None:
    big = "z" * 10000
    tool, _ = _counting_tool("fetch_page", output=lambda _q: big)
    [wrapped] = ToolCallPolicy(
        ToolsConfig(tool_output_max_chars=1000, tool_output_preview_chars=300, tool_output_dir=str(tmp_path)),
        outputs_readable=has_exec_tool([tool]),
    ).wrap([tool])

    shown = _invoke(wrapped, json.dumps({"query": "x"}))

    assert len(shown) < 500
    assert "saved at" not in shown
    assert os.listdir(tmp_path) == []


def test_api_wrapper_reuses_results_and_rewrapping_starts_a_fresh_run() -> None:
    search, calls = _counting_tool("search_web")
    args = json.dumps({"query": "q"})

    [first_run] = ToolCallPolicy(ToolsConfig()).wrap([search])
    _invoke(first_run, args)
    _invoke(first_run, args)
    assert len(calls) == 1

    [second_run] = ToolCallPolicy(ToolsConfig()).wrap([search])
    _invoke(second_run, args)
    assert len(calls) == 2
