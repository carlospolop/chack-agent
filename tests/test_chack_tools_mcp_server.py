from __future__ import annotations

import asyncio
import os
import sys

from agents import function_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

from chack_agent.backends.chack_tools_mcp_server import (
    _ServerPolicyState,
    _py_type_from_schema,
    _register_tools,
)
from chack_tools.subagent_config import enforce_prompt_str_or_list_schema


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
