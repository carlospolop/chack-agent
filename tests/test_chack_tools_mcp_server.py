from __future__ import annotations

import asyncio
import json
import os
import sys

from agents import function_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

from chack_agent.backends.chack_tools_mcp_server import (
    _ServerPolicyState,
    _load_toolset,
    _process_is_alive,
    _py_type_from_schema,
    _register_tools,
)
from chack_tools.subagent_config import enforce_prompt_str_or_list_schema

def test_name_based_administrator_tool_reconstruction(monkeypatch) -> None:
    monkeypatch.setenv("CHACK_MODEL_PROVIDER", "codex")
    monkeypatch.setenv("CHACK_DEFAULT_MODEL", "gpt-test")
    monkeypatch.setenv("CHACK_TOOLS_CONFIG_JSON", json.dumps({
        "task_steps_manager_enabled": False,
        "researcher_administrator_enabled": True,
        "researcher_administrator_researchers": ["websearcher"],
        "websearcher_enabled": True,
    }))
    monkeypatch.setenv("CHACK_RESEARCH_MASTER_DIR", "/tmp/chack-mcp-name-reconstruction")
    monkeypatch.setenv(
        "CHACK_TOOLS_OVERRIDE_NAMES_JSON",
        json.dumps([
            "run_researchers_batch",
            "start_researchers_async",
            "list_researcher_jobs",
            "get_researcher_task",
            "poll_researchers_async",
            "get_researcher_result",
            "cancel_researcher_task",
            "retry_researcher_task",
            "cancel_researchers_async",
            "list_research_artifacts",
            "read_research_artifact",
            "grep_research_artifacts",
            "delete_research_artifact",
            "register_research_artifact",
        ]),
    )
    monkeypatch.setenv("CHACK_ALLOWED_TOOLS_JSON", json.dumps([
        "run_researchers_batch",
        "start_researchers_async",
        "list_researcher_jobs",
        "get_researcher_task",
        "poll_researchers_async",
        "get_researcher_result",
        "cancel_researcher_task",
        "retry_researcher_task",
        "cancel_researchers_async",
        "list_research_artifacts",
        "read_research_artifact",
        "grep_research_artifacts",
        "delete_research_artifact",
        "register_research_artifact",
    ]))

    tools = _load_toolset()
    names = [str(getattr(tool, "name", "") or "") for tool in tools]
    assert names == [
        "run_researchers_batch",
        "start_researchers_async",
        "list_researcher_jobs",
        "get_researcher_task",
        "poll_researchers_async",
        "get_researcher_result",
        "cancel_researcher_task",
        "retry_researcher_task",
        "cancel_researchers_async",
        "list_research_artifacts",
        "read_research_artifact",
        "grep_research_artifacts",
        "delete_research_artifact",
        "register_research_artifact",
    ]


def test_mcp_watchdog_checks_exported_owner_not_direct_parent(monkeypatch) -> None:
    owner_pid = 424242
    calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, signal: int) -> None:
        calls.append((pid, signal))

    monkeypatch.setattr("chack_agent.backends.chack_tools_mcp_server.os.kill", fake_kill)

    assert _process_is_alive(owner_pid) is True
    assert calls == [(owner_pid, 0)]


def test_mcp_watchdog_treats_missing_owner_as_dead(monkeypatch) -> None:
    def fake_kill(pid: int, signal: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr("chack_agent.backends.chack_tools_mcp_server.os.kill", fake_kill)

    assert _process_is_alive(424242) is False


def test_stdio_server_completes_mcp_handshake() -> None:
    async def _list_tool_names() -> set[str]:
        env = os.environ.copy()
        env.update(
            {
                "CHACK_MODEL_PROVIDER": "codex",
                "CHACK_DEFAULT_MODEL": "gpt-5.6-sol",
                "CHACK_TOOLS_CONFIG_JSON": '{"exec_enabled": true}',
                "CHACK_ALLOWED_TOOLS_JSON": '["exec"]',
                "CHACK_DISABLE_STDOUT_EVENTS": "1",
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
                tools = await session.list_tools()
                return {tool.name for tool in tools.tools}

    assert "exec" in asyncio.run(_list_tool_names())


def test_schema_type_preserves_string_or_array_union() -> None:
    annotation = _py_type_from_schema(
        {"type": ["string", "array"], "items": {"type": "string"}}
    )

    @function_tool
    def accepts_batch(prompt: str | list[str]) -> str:
        return repr(prompt)

    accepts_batch.params_json_schema["properties"]["prompt"]["type"] = ["string", "array"]
    accepts_batch.params_json_schema["properties"]["prompt"]["items"] = {"type": "string"}

    server = FastMCP("union-test")
    tool = enforce_prompt_str_or_list_schema(accepts_batch)
    _register_tools(
        server,
        [tool],
        _ServerPolicyState(require_task_steps_manager_init_first=False, max_non_task_tools=0),
    )

    listed = asyncio.run(server.list_tools())
    prompt_schema = listed[0].inputSchema["properties"]["prompt"]
    assert "anyOf" in prompt_schema
    assert {entry.get("type") for entry in prompt_schema["anyOf"]} == {"string", "array"}

    result = asyncio.run(server.call_tool("accepts_batch", {"prompt": ["first", "second"]}))
    rendered = "\n".join(str(block) for block in result)
    assert "first" in rendered
    assert "second" in rendered
    assert "ToolError" not in rendered
    assert annotation is not str


def test_schema_type_preserves_nullable_union() -> None:
    annotation = _py_type_from_schema({"type": ["integer", "null"]})
    assert annotation is not int
