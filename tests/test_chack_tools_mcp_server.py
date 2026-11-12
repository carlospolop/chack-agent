from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest
from agents import function_tool
from typing_extensions import TypedDict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

from chack_agent.backends import chack_tools_mcp_server as mcp_server_module
from chack_agent.backends.chack_tools_mcp_server import (
    _ServerPolicyState,
    _load_toolset,
    _process_is_alive,
    _py_type_from_schema,
    _register_tools,
)
from chack_agent.backends.tool_payloads import (
    deserialize_tools_payload,
    preserve_mcp_input_schema,
    serialize_tools_payload,
)
from chack_tools.subagent_config import enforce_prompt_str_or_list_schema


class _SchemaLookup(TypedDict):
    symbol_name: str
    limit: int


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


def test_mcp_watchdog_never_uses_os_kill_for_windows_liveness(monkeypatch) -> None:
    queried: list[int] = []

    monkeypatch.setattr(mcp_server_module, "_uses_windows_process_api", lambda: True)
    monkeypatch.setattr(
        mcp_server_module,
        "_windows_process_is_alive",
        lambda pid: queried.append(pid) or True,
    )
    monkeypatch.setattr(
        mcp_server_module.os,
        "kill",
        lambda *_args: pytest.fail("os.kill(pid, 0) terminates the process on Windows"),
    )

    assert _process_is_alive(424242) is True
    assert queried == [424242]


def test_stdio_server_completes_mcp_handshake() -> None:
    async def _list_tool_names() -> set[str]:
        env = os.environ.copy()
        env.update(
            {
                "CHACK_MODEL_PROVIDER": "codex",
                "CHACK_DEFAULT_MODEL": "gpt-5.6-sol",
                "CHACK_TOOLS_CONFIG_JSON": '{"exec_enabled": true}',
                "CHACK_ALLOWED_TOOLS_JSON": '["exec"]',
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


def test_stdio_server_offers_only_tools() -> None:
    # Clients add resource/prompt browsing tools to every model request when a
    # server advertises those capabilities, so the server must not.
    async def _capabilities():
        env = os.environ.copy()
        env.update(
            {
                "CHACK_MODEL_PROVIDER": "codex",
                "CHACK_DEFAULT_MODEL": "gpt-5.6-sol",
                "CHACK_TOOLS_CONFIG_JSON": '{"exec_enabled": true}',
                "CHACK_ALLOWED_TOOLS_JSON": '["exec"]',
            }
        )
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "chack_agent.backends.chack_tools_mcp_server"],
            env=env,
        )
        async with stdio_client(server) as streams:
            async with ClientSession(*streams) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                return init.capabilities, {tool.name for tool in tools.tools}

    capabilities, tool_names = asyncio.run(_capabilities())

    assert capabilities.tools is not None
    assert capabilities.resources is None
    assert capabilities.prompts is None
    assert "exec" in tool_names


@pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:"
)
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
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(listed[0].inputSchema)
    assert validator.is_valid({"prompt":"first"})
    assert validator.is_valid({"prompt":["first","second"]})
    assert not validator.is_valid({"prompt":42})

    result = asyncio.run(server.call_tool("accepts_batch", {"prompt": ["first", "second"]}))
    rendered = "\n".join(str(block) for block in result)
    assert "first" in rendered
    assert "second" in rendered
    assert "ToolError" not in rendered
    assert annotation is not str


def test_schema_type_preserves_nullable_union() -> None:
    annotation = _py_type_from_schema({"type": ["integer", "null"]})
    assert annotation is not int


def test_clients_receive_each_tool_result_once_as_text() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    @function_tool(name_override="read_notes")
    def read_notes(topic: str) -> str:
        """Read notes about a topic."""
        return f"notes about {topic}: " + "x" * 2000

    server = FastMCP("result-shape-test")
    _register_tools(
        server,
        [read_notes],
        _ServerPolicyState(require_task_steps_manager_init_first=False, max_non_task_tools=0),
    )

    async def _exchange():
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            result = await client.call_tool("read_notes", {"topic": "auth"})
            return listed, result

    listed, result = asyncio.run(_exchange())

    assert [tool.outputSchema for tool in listed.tools] == [None]
    assert result.structuredContent is None
    assert [block.text for block in result.content] == ["notes about auth: " + "x" * 2000]


def test_json_answers_reach_clients_with_an_object_root() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    answers = {
        "list": json.dumps([{"path": "a.md", "score": 2}, {"path": "b.md", "score": 1}]),
        "scalar": "42",
        "object": json.dumps({"hits": ["a.md"]}),
        "text": "plain [not json",
    }

    @function_tool(name_override="lookup")
    def lookup(kind: str) -> str:
        """Look something up."""
        return answers[kind]

    server = FastMCP("object-root-test")
    _register_tools(
        server,
        [lookup],
        _ServerPolicyState(require_task_steps_manager_init_first=False, max_non_task_tools=0),
    )

    async def _exchange():
        async with create_connected_server_and_client_session(server) as client:
            return {kind: (await client.call_tool("lookup", {"kind": kind})).content[0].text for kind in answers}

    texts = asyncio.run(_exchange())

    assert json.loads(texts["list"]) == {"result": [{"path": "a.md", "score": 2}, {"path": "b.md", "score": 1}]}
    assert json.loads(texts["scalar"]) == {"result": 42}
    assert texts["object"] == answers["object"]
    assert texts["text"] == answers["text"]


def test_a_command_that_reads_stdin_leaves_later_tool_calls_answered() -> None:
    async def _exchange() -> list[str]:
        env = os.environ.copy()
        env.update(
            {
                "CHACK_MODEL_PROVIDER": "codex",
                "CHACK_DEFAULT_MODEL": "gpt-5.6-sol",
                "CHACK_TOOLS_CONFIG_JSON": '{"exec_enabled": true}',
                "CHACK_ALLOWED_TOOLS_JSON": '["exec"]',
                "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST": "0",
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
                answers = []
                for command in ("cat; echo read-done", "echo after"):
                    result = await asyncio.wait_for(
                        session.call_tool("exec", {"command": command, "timeout": 30}), 15
                    )
                    answers.append(result.content[0].text.strip())
                return answers

    assert asyncio.run(_exchange()) == ["read-done", "after"]


def test_published_batch_schema_rejects_unknown_fields_and_wrong_nested_types():
    from jsonschema import Draft202012Validator

    @function_tool
    def lookup_symbols(requests: list[_SchemaLookup]) -> str:
        return ','.join(f"{r['symbol_name']}:{r['limit']}" for r in requests)

    server = FastMCP('nested-schema-test')
    tools = deserialize_tools_payload(serialize_tools_payload([preserve_mcp_input_schema(lookup_symbols)]))
    _register_tools(server, tools, _ServerPolicyState(False, 0))
    schema = asyncio.run(server.list_tools())[0].inputSchema
    validator = Draft202012Validator(schema)
    valid = {'requests':[{'symbol_name':'alpha','limit':3}]}
    assert validator.is_valid(valid)
    assert not validator.is_valid({'requests':[{'name':'alpha','limit':3}]})
    assert not validator.is_valid({'requests':[{'symbol_name':'alpha','limit':'many'}]})
    result = asyncio.run(server.call_tool('lookup_symbols', valid))
    assert [block.text for block in result] == ['alpha:3']


def test_original_json_argument_names_and_previous_python_aliases_both_invoke_tool():
    from agents.tool import FunctionTool
    from jsonschema import Draft202012Validator
    from chack_agent.backends.chack_tools_mcp_server import _safe_identifier

    async def invoke(_context, arguments):
        return json.loads(arguments)['class']['kind']

    tool = FunctionTool(name='read_class', description='Read a selected class',
        params_json_schema={
            'type':'object', 'properties':{'class':{'$ref':'#/$defs/Options'}},
            'required':['class'], 'additionalProperties':False,
            '$defs':{'Options':{'type':'object','properties':{'kind':{'enum':['public','private']}},
                'required':['kind'],'additionalProperties':False}},
        }, on_invoke_tool=invoke)
    server = FastMCP('json-name-test')
    _register_tools(server, [preserve_mcp_input_schema(tool)], _ServerPolicyState(False, 0))
    schema = asyncio.run(server.list_tools())[0].inputSchema
    validator = Draft202012Validator(schema)
    valid = {'class':{'kind':'public'}}
    assert validator.is_valid(valid)
    assert not validator.is_valid({'class':{'kind':'unknown'}})
    assert [b.text for b in asyncio.run(server.call_tool('read_class', valid))] == ['public']
    old_name = _safe_identifier('class', set())
    assert [b.text for b in asyncio.run(server.call_tool('read_class', {old_name:{'kind':'private'}}))] == ['private']


def test_duplicate_tool_name_keeps_first_schema_and_handler():
    from jsonschema import Draft202012Validator

    @function_tool(name_override='read_value')
    def first(value: str) -> str:
        return 'first '+value

    @function_tool(name_override='read_value')
    def second(number: int) -> str:
        return str(number)

    server = FastMCP('duplicate-name-test')
    _register_tools(
        server,
        [preserve_mcp_input_schema(first), preserve_mcp_input_schema(second)],
        _ServerPolicyState(False, 0),
    )
    validator = Draft202012Validator(asyncio.run(server.list_tools())[0].inputSchema)
    assert validator.is_valid({'value':'alpha'})
    assert not validator.is_valid({'number':3})
    assert [b.text for b in asyncio.run(server.call_tool('read_value', {'value':'alpha'}))] == ['first alpha']
