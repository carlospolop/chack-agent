from __future__ import annotations

import asyncio
import inspect
import json
import keyword
import os
import uuid
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

from agents.tool_context import ToolContext
from agents.usage import Usage

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig


def _safe_identifier(name: str, used: set[str]) -> str:
    base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not base or base[0].isdigit() or keyword.iskeyword(base):
        base = f"arg_{base}"
    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _py_type_from_schema(prop_schema: dict[str, Any]) -> Any:
    raw_type = prop_schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        raw_type = non_null[0] if non_null else "string"
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(str(raw_type or "string"), Any)


async def _invoke_function_tool(tool: Any, args: dict[str, Any]) -> str:
    raw_args = json.dumps(args, ensure_ascii=False)
    ctx = ToolContext(
        context=None,
        usage=Usage(),
        tool_name=str(getattr(tool, "name", "tool") or "tool"),
        tool_call_id=f"mcp-{uuid.uuid4()}",
        tool_arguments=raw_args,
        tool_input=args,
    )
    result = await tool.on_invoke_tool(ctx, raw_args)

    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, (dict, list, tuple)):
        return json.dumps(result, ensure_ascii=False)
    try:
        return json.dumps(result.model_dump(), ensure_ascii=False)  # pydantic-like
    except Exception:
        pass
    return str(result)


@dataclass
class _ServerPolicyState:
    require_task_list_init_first: bool
    max_non_task_tools: int
    has_task_list_init: bool = False
    non_task_tool_calls: int = 0


def _as_bool(raw: str, default: bool = False) -> bool:
    value = str(raw or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _as_int(raw: str, default: int = 0) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception:
        return default


def _load_toolset() -> list[Any]:
    tools_cfg_raw = os.environ.get("CHACK_TOOLS_CONFIG_JSON", "{}").strip() or "{}"
    try:
        tools_cfg_data = json.loads(tools_cfg_raw)
    except json.JSONDecodeError:
        tools_cfg_data = {}
    if not isinstance(tools_cfg_data, dict):
        tools_cfg_data = {}

    tool_profile = os.environ.get("CHACK_TOOL_PROFILE", "all") or "all"
    default_model = os.environ.get("CHACK_DEFAULT_MODEL", "")
    social_network_model = os.environ.get("CHACK_SOCIAL_NETWORK_MODEL", "")
    scientific_model = os.environ.get("CHACK_SCIENTIFIC_MODEL", "")
    websearcher_model = os.environ.get("CHACK_WEBSEARCHER_MODEL", "")
    tester_model = os.environ.get("CHACK_TESTER_MODEL", "")

    def _to_int(name: str, default: int) -> int:
        raw = os.environ.get(name, str(default)).strip()
        try:
            return int(raw)
        except Exception:
            return default

    toolset = AgentsToolset(
        ToolsConfig(**tools_cfg_data),
        tool_profile=tool_profile,
        default_model=default_model,
        social_network_model=social_network_model,
        scientific_model=scientific_model,
        websearcher_model=websearcher_model,
        tester_model=tester_model,
        social_network_max_turns=_to_int("CHACK_SOCIAL_NETWORK_MAX_TURNS", 30),
        scientific_max_turns=_to_int("CHACK_SCIENTIFIC_MAX_TURNS", 30),
        websearcher_max_turns=_to_int("CHACK_WEBSEARCHER_MAX_TURNS", 30),
        tester_max_turns=_to_int("CHACK_TESTER_MAX_TURNS", 30),
    )
    return list(getattr(toolset, "tools", []) or [])


def _register_tools(mcp: FastMCP, tools: list[Any], state: _ServerPolicyState) -> None:
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").strip()
        if not name:
            continue
        schema = getattr(tool, "params_json_schema", None) or {}
        description = str(getattr(tool, "description", "") or "")

        properties = schema.get("properties") if isinstance(schema, dict) else {}
        required = set(schema.get("required", []) if isinstance(schema, dict) else [])
        if not isinstance(properties, dict):
            properties = {}

        used_identifiers: set[str] = set()
        mapping_py_to_json: dict[str, str] = {}
        annotations: dict[str, Any] = {}
        parameters: list[inspect.Parameter] = []

        for json_name, prop_schema in properties.items():
            json_key = str(json_name)
            py_name = _safe_identifier(json_key, used_identifiers)
            mapping_py_to_json[py_name] = json_key
            annotations[py_name] = _py_type_from_schema(
                prop_schema if isinstance(prop_schema, dict) else {}
            )
            default = inspect.Parameter.empty if json_key in required else None
            parameters.append(
                inspect.Parameter(
                    py_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotations[py_name],
                )
            )

        async def _proxy(_tool=tool, _mapping=mapping_py_to_json, _name=name, **kwargs: Any) -> str:
            payload: dict[str, Any] = {}
            for py_name, value in kwargs.items():
                if value is None:
                    continue
                payload[_mapping.get(py_name, py_name)] = value

            if state.require_task_list_init_first and not state.has_task_list_init:
                if _name != "task_list" or str(payload.get("action", "")).strip().lower() != "init":
                    raise RuntimeError(
                        "You must call task_list with action=init before using any other tool."
                    )

            if _name == "task_list":
                if str(payload.get("action", "")).strip().lower() == "init":
                    state.has_task_list_init = True
            else:
                if state.max_non_task_tools > 0 and state.non_task_tool_calls >= state.max_non_task_tools:
                    raise RuntimeError(
                        f"Tool budget reached ({state.max_non_task_tools}). Finish using gathered context."
                    )
                state.non_task_tool_calls += 1

            return await _invoke_function_tool(_tool, payload)

        _proxy.__name__ = f"tool_{name}".replace("-", "_")
        _proxy.__doc__ = description
        _proxy.__annotations__ = dict(annotations)
        _proxy.__annotations__["return"] = str
        _proxy.__signature__ = inspect.Signature(parameters=parameters, return_annotation=str)

        mcp.add_tool(_proxy, name=name, description=description)


def main() -> None:
    mcp = FastMCP("Chack Tools MCP")
    tools = _load_toolset()
    state = _ServerPolicyState(
        require_task_list_init_first=_as_bool(
            os.environ.get("CHACK_REQUIRE_TASK_LIST_INIT_FIRST", "1"),
            default=True,
        ),
        max_non_task_tools=max(0, _as_int(os.environ.get("CHACK_MAX_TOOLS_USED", "0"), 0)),
    )
    _register_tools(mcp, tools, state)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
