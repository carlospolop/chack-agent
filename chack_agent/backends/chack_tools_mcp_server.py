from __future__ import annotations

import inspect
import json
import keyword
import os
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

from agents.tool_context import ToolContext
from agents.usage import Usage

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.task_steps_manager_state import STORE, set_active_context
from chack_tools.tool_usage_state import is_non_counted_tool_name
from chack_tools.run_lifecycle import (
    ToolBudgetClaim,
    claim_non_task_tool_slot,
    mark_task_manager_initialized,
    record_mcp_tool_usage,
    task_manager_initialized,
    tool_budget_warning,
)
from chack_agent.limit_event_state import emit_limit_reached
from chack_agent.budget_warning_state import budget_status_from_env, inject_budget_warning_from_env
from chack_agent.thinking_effort import normalize_thinking_effort
from .tool_payloads import (
    CHACK_ALLOWED_TOOLS_JSON_PATH_ENV,
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_APPEND_B64_PATH_ENV,
    CHACK_TOOLS_CONFIG_JSON_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    CHACK_TOOLS_OVERRIDE_B64_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV,
    CHACK_TOOLS_APPEND_NAMES_JSON_ENV,
    deserialize_tools_payload,
    read_payload_from_env_or_file,
)


_MCP_STARTUP_STATUS_PATH_ENV = "CHACK_MCP_STARTUP_STATUS_PATH"
_MCP_PARENT_PID_ENV = "CHACK_MCP_PARENT_PID"


def _write_startup_status(*, tool_names: list[str] | None = None, error: str = "") -> None:
    path = str(os.environ.get(_MCP_STARTUP_STATUS_PATH_ENV, "") or "").strip()
    if not path:
        return
    payload = {
        "tool_names": sorted(str(name) for name in (tool_names or []) if str(name).strip()),
        "error": str(error or "")[:2000],
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary_path = f"{path}.{os.getpid()}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
        os.replace(temporary_path, path)
    except Exception:
        pass


# Tools blocked from MCP exposure even if they are present in the Python
# toolset. Historically `exec` was included here as a conservative default,
# but Chack deployments now intentionally use the controlled ExecTool as the
# local command path (with Chack timeouts, output limits, and telemetry). If a
# deployment does not want local command execution it must set
# tools.exec_enabled=false instead of relying on this transport-level denylist.
_MCP_DENYLIST_TOOL_NAMES = {
    "shell_command",
    "run_terminal_cmd",
    "command_execution",
}

_MCP_ROLE_AGENT_FIELDS = {
    "social_network": "social_network_agent",
    "scientific": "scientific_agent",
    "websearcher": "websearcher_agent",
    "business": "business_agent",
    "product": "product_agent",
    "travel": "travel_agent",
    "legal": "legal_agent",
    "data_statistics": "data_statistics_agent",
    "news_media": "news_media_agent",
    "knowledge_graph": "knowledge_graph_agent",
    "religious": "religious_agent",
    "cli": "cli_agent",
    "subchack": "subchack_agent",
    "researcher_administrator": "researcher_administrator_agent",
    "researcher_queue": "researcher_queue_agent",
}


def _mcp_tools_config(tools_cfg_data: dict[str, Any]) -> ToolsConfig:
    """Load MCP tool config with explicit, normalized researcher effort.

    Normal backend launches carry per-role settings inside
    ``CHACK_TOOLS_CONFIG_JSON``. Standalone/shared MCP servers can instead set
    ``CHACK_THINKING_EFFORT`` for every role or
    ``CHACK_<ROLE>_THINKING_EFFORT`` for one role. With no configuration, all
    MCP-created agents are explicitly set to ``high``.
    """
    allowed_keys = set(getattr(ToolsConfig, "__dataclass_fields__", {}).keys())
    compatible_data = {
        key: value for key, value in tools_cfg_data.items() if key in allowed_keys
    }
    config = ToolsConfig(**compatible_data)
    global_raw = str(os.environ.get("CHACK_THINKING_EFFORT", "") or "").strip()

    for role, field_name in _MCP_ROLE_AGENT_FIELDS.items():
        role_env = f"CHACK_{role.upper()}_THINKING_EFFORT"
        role_raw = str(os.environ.get(role_env, "") or "").strip()
        settings = dict(getattr(config, field_name, {}) or {})
        configured_raw = settings.get("thinking_effort")
        # Environment values are explicit MCP-process overrides. Otherwise,
        # preserve serialized per-role configuration and finally default high.
        selected = role_raw or global_raw or configured_raw or "high"
        settings["thinking_effort"] = normalize_thinking_effort(selected)
        setattr(config, field_name, settings)

    return config


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
        raw_types = [str(value) for value in raw_type]
    elif raw_type:
        raw_types = [str(raw_type)]
    else:
        any_of = prop_schema.get("anyOf")
        raw_types = [
            str(entry.get("type"))
            for entry in any_of or []
            if isinstance(entry, dict) and entry.get("type")
        ]

    scalar_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "null": type(None),
    }
    resolved: list[Any] = []
    for type_name in raw_types:
        if type_name == "array":
            item_schema = prop_schema.get("items")
            item_type = _py_type_from_schema(item_schema if isinstance(item_schema, dict) else {})
            resolved.append(list[item_type])
        else:
            resolved.append(scalar_mapping.get(type_name, Any))

    if not resolved or Any in resolved:
        return Any
    annotation = resolved[0]
    for candidate in resolved[1:]:
        annotation = annotation | candidate
    return annotation


def _schema_is_nullable(prop_schema: dict[str, Any]) -> bool:
    raw_type = prop_schema.get("type")
    if isinstance(raw_type, list):
        return "null" in raw_type
    any_of = prop_schema.get("anyOf")
    if isinstance(any_of, list):
        for entry in any_of:
            if isinstance(entry, dict) and entry.get("type") == "null":
                return True
    return False


async def _invoke_function_tool(tool: Any, args: dict[str, Any]) -> str:
    raw_args = json.dumps(args, ensure_ascii=False)
    base_kwargs = {
        "context": None,
        "usage": Usage(),
        "tool_name": str(getattr(tool, "name", "tool") or "tool"),
        "tool_call_id": f"mcp-{uuid.uuid4()}",
        "tool_arguments": raw_args,
    }
    ctx = ToolContext(**base_kwargs)
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
    require_task_steps_manager_init_first: bool
    max_non_task_tools: int
    session_id: str = ""
    has_task_steps_manager_init: bool = False
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


def _truncate_tool_output(value: str) -> str:
    token_budget = max(1, _as_int(os.environ.get("CHACK_MCP_TOOL_MAX_TOKENS", "10000"), 10000))
    max_bytes = token_budget * 4
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value

    marker = "\n... truncated by MCP tool response limit ...\n"
    half = max(0, (max_bytes - len(marker.encode("utf-8"))) // 2)
    prefix = encoded[:half].decode("utf-8", errors="ignore")
    suffix = encoded[-half:].decode("utf-8", errors="ignore")
    return f"{prefix}{marker}{suffix}"


def _process_is_alive(pid: int) -> bool:
    """Return whether the configured owner process still exists.

    The MCP server is usually a grandchild of the Python backend: the backend
    starts Claude Code, and Claude Code starts this stdio server. Therefore
    ``os.getppid()`` is Claude Code's PID, not the owner PID exported by the
    backend. Comparing those two PIDs makes a healthy server exit after the
    first watchdog tick. ``kill(pid, 0)`` checks the exported owner directly
    and also works after the server is reparented when the owner dies.
    """
    if int(pid or 0) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # An inaccessible process still exists; do not kill the MCP server
        # merely because this check cannot inspect it.
        return True
    except OSError:
        return False
    return True


def _start_parent_watchdog() -> threading.Event:
    """Stop an orphaned per-provider MCP server and reconcile its jobs."""
    configured_parent = str(os.environ.get(_MCP_PARENT_PID_ENV, "") or "").strip()
    direct_parent_pid = os.getppid()
    try:
        parent_pid = int(configured_parent) if configured_parent else direct_parent_pid
    except (TypeError, ValueError):
        parent_pid = direct_parent_pid
    stop = threading.Event()

    def _watch() -> None:
        while not stop.wait(1.0):
            # The exported backend PID protects against an orphaned server if
            # the backend dies, while the direct parent protects against a
            # normal Claude invocation ending while the backend stays alive.
            if _process_is_alive(parent_pid) and _process_is_alive(direct_parent_pid):
                continue
            try:
                from chack_tools.researcher_administrator_agent import _shutdown_all_research_jobs

                _shutdown_all_research_jobs(timeout_seconds=15.0)
            finally:
                # The server is scoped to the provider process. Once that
                # process is gone, keeping stdio/MCP alive would strand daemon
                # workers and make the durable ledger lie about liveness.
                os._exit(0)

    thread = threading.Thread(target=_watch, name="chack-mcp-parent-watchdog", daemon=True)
    thread.start()
    return stop


def _load_toolset() -> list[Any]:
    tools_cfg_raw = (
        read_payload_from_env_or_file(
            os.environ.get("CHACK_TOOLS_CONFIG_JSON", ""),
            os.environ.get(CHACK_TOOLS_CONFIG_JSON_PATH_ENV, ""),
        ).strip()
        or "{}"
    )
    try:
        tools_cfg_data = json.loads(tools_cfg_raw)
    except json.JSONDecodeError:
        tools_cfg_data = {}
    if not isinstance(tools_cfg_data, dict):
        tools_cfg_data = {}
    allowed_tool_cfg_keys = set(getattr(ToolsConfig, "__dataclass_fields__", {}).keys())
    tools_cfg_data = {k: v for k, v in tools_cfg_data.items() if k in allowed_tool_cfg_keys}

    model_provider = str(os.environ.get("CHACK_MODEL_PROVIDER", "") or "").strip()
    if not model_provider:
        raise RuntimeError("CHACK_MODEL_PROVIDER must be defined")
    default_model = os.environ.get("CHACK_DEFAULT_MODEL", "")
    social_network_model = os.environ.get("CHACK_SOCIAL_NETWORK_MODEL", "")
    scientific_model = os.environ.get("CHACK_SCIENTIFIC_MODEL", "")
    websearcher_model = os.environ.get("CHACK_WEBSEARCHER_MODEL", "")
    business_model = os.environ.get("CHACK_BUSINESS_MODEL", "")
    product_model = os.environ.get("CHACK_PRODUCT_MODEL", "")
    travel_model = os.environ.get("CHACK_TRAVEL_MODEL", "")
    legal_model = os.environ.get("CHACK_LEGAL_MODEL", "")
    data_statistics_model = os.environ.get("CHACK_DATA_STATISTICS_MODEL", "")
    news_media_model = os.environ.get("CHACK_NEWS_MEDIA_MODEL", "")
    knowledge_graph_model = os.environ.get("CHACK_KNOWLEDGE_GRAPH_MODEL", "")
    religious_model = os.environ.get("CHACK_RELIGIOUS_MODEL", "")
    cli_model = os.environ.get("CHACK_CLI_MODEL", "")
    subchack_model = os.environ.get("CHACK_SUBCHACK_MODEL", "")
    researcher_administrator_model = os.environ.get("CHACK_RESEARCHER_ADMINISTRATOR_MODEL", "")

    def _to_int(name: str, default: int) -> int:
        raw = os.environ.get(name, str(default)).strip()
        try:
            return int(raw)
        except Exception:
            return default

    serialized_tools_override_b64 = read_payload_from_env_or_file(
        os.environ.get(CHACK_TOOLS_OVERRIDE_B64_ENV, ""),
        os.environ.get(CHACK_TOOLS_OVERRIDE_B64_PATH_ENV, ""),
    )
    serialized_tools_append_b64 = read_payload_from_env_or_file(
        os.environ.get(CHACK_TOOLS_APPEND_B64_ENV, ""),
        os.environ.get(CHACK_TOOLS_APPEND_B64_PATH_ENV, ""),
    )
    override_names_raw = str(
        os.environ.get(CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV, "") or ""
    ).strip()
    append_names_raw = str(
        os.environ.get(CHACK_TOOLS_APPEND_NAMES_JSON_ENV, "") or ""
    ).strip()
    try:
        override_names = json.loads(override_names_raw) if override_names_raw else []
        append_names = json.loads(append_names_raw) if append_names_raw else []
    except json.JSONDecodeError as exc:
        raise RuntimeError("Name-based tool transport payload is not valid JSON") from exc
    if not isinstance(override_names, list) or not all(
        isinstance(name, str) for name in override_names
    ):
        raise RuntimeError(
            f"{CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV} must be a JSON string array"
        )
    if not isinstance(append_names, list) or not all(
        isinstance(name, str) for name in append_names
    ):
        raise RuntimeError(
            f"{CHACK_TOOLS_APPEND_NAMES_JSON_ENV} must be a JSON string array"
        )
    override_tools = deserialize_tools_payload(serialized_tools_override_b64)
    append_tools = deserialize_tools_payload(serialized_tools_append_b64)

    toolset = AgentsToolset(
        _mcp_tools_config(tools_cfg_data),
        model_provider=model_provider,
        default_model=default_model,
        social_network_model=social_network_model,
        scientific_model=scientific_model,
        websearcher_model=websearcher_model,
        business_model=business_model,
        product_model=product_model,
        travel_model=travel_model,
        legal_model=legal_model,
        data_statistics_model=data_statistics_model,
        news_media_model=news_media_model,
        knowledge_graph_model=knowledge_graph_model,
        religious_model=religious_model,
        cli_model=cli_model,
        subchack_model=subchack_model,
        researcher_administrator_model=researcher_administrator_model,
        social_network_max_turns=_to_int("CHACK_SOCIAL_NETWORK_MAX_TURNS", 50),
        scientific_max_turns=_to_int("CHACK_SCIENTIFIC_MAX_TURNS", 50),
        websearcher_max_turns=_to_int("CHACK_WEBSEARCHER_MAX_TURNS", 50),
        self_critique_enabled=os.environ.get("CHACK_SELF_CRITIQUE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"},
        self_critique_rounds=_to_int("CHACK_SELF_CRITIQUE_ROUNDS", 0),
        business_max_turns=_to_int("CHACK_BUSINESS_MAX_TURNS", 50),
        product_max_turns=_to_int("CHACK_PRODUCT_MAX_TURNS", 50),
        travel_max_turns=_to_int("CHACK_TRAVEL_MAX_TURNS", 50),
        legal_max_turns=_to_int("CHACK_LEGAL_MAX_TURNS", 50),
        data_statistics_max_turns=_to_int("CHACK_DATA_STATISTICS_MAX_TURNS", 50),
        news_media_max_turns=_to_int("CHACK_NEWS_MEDIA_MAX_TURNS", 50),
        knowledge_graph_max_turns=_to_int("CHACK_KNOWLEDGE_GRAPH_MAX_TURNS", 50),
        religious_max_turns=_to_int("CHACK_RELIGIOUS_MAX_TURNS", 50),
        cli_max_turns=_to_int("CHACK_CLI_MAX_TURNS", 50),
        subchack_max_turns=_to_int("CHACK_SUBCHACK_MAX_TURNS", 100),
        researcher_administrator_max_turns=_to_int("CHACK_RESEARCHER_ADMINISTRATOR_MAX_TURNS", 100),
    )
    allowed_tools_raw = read_payload_from_env_or_file(
        os.environ.get("CHACK_ALLOWED_TOOLS_JSON", ""),
        os.environ.get(CHACK_ALLOWED_TOOLS_JSON_PATH_ENV, ""),
    ).strip()
    allowed_tools: set[str] | None = None
    if allowed_tools_raw:
        try:
            parsed_allowed = json.loads(allowed_tools_raw)
        except json.JSONDecodeError:
            parsed_allowed = None
        if isinstance(parsed_allowed, list):
            allowed_tools = {
                str(item).strip().lower()
                for item in parsed_allowed
                if str(item).strip()
            }
    if str(serialized_tools_override_b64 or "").strip():
        tools = list(override_tools)
    elif override_names:
        helper = getattr(toolset, "_researcher_administrator_helper", None)
        management_names = {
            "run_researchers_batch",
            "start_researchers_async",
            "list_researcher_jobs",
            "get_researcher_task",
            "poll_researchers_async",
            "get_researcher_result",
            "cancel_researcher_task",
            "retry_researcher_task",
            "cancel_researchers_async",
        }
        if helper is None or not (set(override_names) & management_names):
            raise RuntimeError(
                "Name-based tool override requires a researcher administrator helper"
            )
        artifact_root = (
            str(os.environ.get("CHACK_RESEARCH_MASTER_DIR", "") or "").strip()
            or str(os.environ.get("CHACK_RESEARCH_DATA_DIR", "") or "").strip()
        )
        reconstructed = helper._build_subagent_tools(
            helper._enabled_researchers(),
            artifact_root=artifact_root,
        )
        reconstructed_by_name = {
            str(getattr(tool, "name", "") or "").strip(): tool
            for tool in reconstructed
        }
        missing = [
            name for name in override_names if name not in reconstructed_by_name
        ]
        if missing:
            raise RuntimeError(
                "Name-based researcher administrator tool reconstruction is missing: "
                + ", ".join(sorted(set(missing)))
            )
        tools = [reconstructed_by_name[name] for name in override_names]
    else:
        tools = list(getattr(toolset, "tools", []) or [])
        if append_tools:
            tools.extend(list(append_tools))
    if append_names:
        base_by_name = {
            str(getattr(tool, "name", "") or "").strip(): tool
            for tool in (getattr(toolset, "tools", []) or [])
        }
        for name in append_names:
            if name not in base_by_name:
                raise RuntimeError(
                    f"Name-based appended tool reconstruction is missing: {name}"
                )
            if not any(
                str(getattr(tool, "name", "") or "").strip() == name
                for tool in tools
            ):
                tools.append(base_by_name[name])
    filtered_tools: list[Any] = []
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").strip().lower()
        if name in _MCP_DENYLIST_TOOL_NAMES:
            continue
        if allowed_tools is not None and name not in allowed_tools:
            continue
        filtered_tools.append(tool)
    return filtered_tools


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
            schema_obj = prop_schema if isinstance(prop_schema, dict) else {}
            has_schema_default = "default" in schema_obj
            schema_default = schema_obj.get("default")
            is_nullable = _schema_is_nullable(schema_obj)
            is_required = bool(json_key in required and not has_schema_default and not is_nullable)
            if is_required:
                default = inspect.Parameter.empty
            elif has_schema_default:
                default = schema_default
            else:
                default = None
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
            try:
                has_persistent_init = bool(
                    state.session_id and task_manager_initialized(state.session_id)
                )
                if has_persistent_init:
                    state.has_task_steps_manager_init = True
                if state.require_task_steps_manager_init_first and not state.has_task_steps_manager_init:
                    if _name != "task_steps_manager" or str(payload.get("action", "")).strip().lower() != "init":
                        raise RuntimeError(
                            "You must call task_steps_manager with action=init before using any other tool."
                        )

                is_task_steps_manager_init = (
                    _name == "task_steps_manager"
                    and str(payload.get("action", "")).strip().lower() == "init"
                )
                tool_claim = ToolBudgetClaim(
                    allowed=True,
                    used=state.non_task_tool_calls,
                    max_tools=state.max_non_task_tools,
                )
                if not is_non_counted_tool_name(_name):
                    if state.max_non_task_tools > 0:
                        try:
                            warning_ratio = float(
                                os.environ.get("CHACK_BUDGET_WARNING_RATIO", "0.6") or "0.6"
                            )
                        except (TypeError, ValueError):
                            warning_ratio = 0.6
                        try:
                            critical_ratio = float(
                                os.environ.get("CHACK_BUDGET_CRITICAL_RATIO", "0.9") or "0.9"
                            )
                        except (TypeError, ValueError):
                            critical_ratio = 0.9
                        tool_claim = claim_non_task_tool_slot(
                            state.session_id,
                            state.max_non_task_tools,
                            warning_ratio=warning_ratio,
                            critical_ratio=critical_ratio,
                        )
                        state.non_task_tool_calls = max(
                            state.non_task_tool_calls,
                            tool_claim.used,
                        )
                    else:
                        state.non_task_tool_calls += 1
                        tool_claim = ToolBudgetClaim(
                            allowed=True,
                            used=state.non_task_tool_calls,
                            max_tools=state.max_non_task_tools,
                        )
                    if not tool_claim.allowed:
                        emit_limit_reached(
                            "tools",
                            {
                                "max_tools_used": state.max_non_task_tools,
                                "used": tool_claim.used,
                                "tool": _name,
                            },
                        )
                        raise RuntimeError(
                            f"Tool budget reached ({state.max_non_task_tools}). Finish using gathered context."
                        )

                record_mcp_tool_usage(_name, state.session_id)
                result = await _invoke_function_tool(_tool, payload)
                if is_task_steps_manager_init and str(result).startswith("SUCCESS:"):
                    state.has_task_steps_manager_init = True
                    if state.session_id:
                        mark_task_manager_initialized(state.session_id)
                result = str(result) + tool_budget_warning(tool_claim)
                result = inject_budget_warning_from_env(result)
                return _truncate_tool_output(result)
            except Exception:
                raise

        _proxy.__name__ = f"tool_{name}".replace("-", "_")
        _proxy.__doc__ = description
        _proxy.__annotations__ = dict(annotations)
        _proxy.__annotations__["return"] = str
        _proxy.__signature__ = inspect.Signature(parameters=parameters, return_annotation=str)

        mcp.add_tool(_proxy, name=name, description=description)


def main() -> None:
    parent_watchdog_stop = threading.Event()
    transport_hint = (
        str(os.environ.get("CHACK_MCP_TRANSPORT", "stdio") or "stdio")
        .strip()
        .lower()
        .replace("_", "-")
    )
    if transport_hint == "stdio":
        parent_watchdog_stop = _start_parent_watchdog()
    try:
        try:
            from chack_tools.telemetry import sqs_logger as _sqs_logger  # type: ignore

            def _noop_stdout_event(_event):
                return None

            _sqs_logger._emit_stdout_event = _noop_stdout_event  # type: ignore[attr-defined]
        except Exception:
            pass

        session_id = str(os.environ.get("CHACK_TASK_SESSION_ID", "") or "").strip()
        run_label = str(os.environ.get("CHACK_RUN_LABEL", "") or "").strip() or "Run 1"
        if session_id:
            STORE.ensure_run(session_id, run_label)
            set_active_context(session_id, run_label)

        # Transport: stdio per-agent by default; run as a single long-lived
        # streamable-http/sse service (CHACK_MCP_TRANSPORT=http) so many external
        # agents connect to one process and share the in-memory researcher queue.
        transport = (
            str(os.environ.get("CHACK_MCP_TRANSPORT", "stdio") or "stdio")
            .strip()
            .lower()
            .replace("_", "-")
        )
        mcp = FastMCP(
            "chack_tools",
            host=str(os.environ.get("CHACK_MCP_HOST", "127.0.0.1") or "127.0.0.1"),
            port=_as_int(os.environ.get("CHACK_MCP_PORT", "8000"), 8000),
            stateless_http=_as_bool(os.environ.get("CHACK_MCP_STATELESS_HTTP", "1"), default=True),
        )
        tools = _load_toolset()
        state = _ServerPolicyState(
            require_task_steps_manager_init_first=_as_bool(
                os.environ.get("CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST", "1"),
                default=True,
            ),
            max_non_task_tools=max(0, _as_int(os.environ.get("CHACK_MAX_TOOLS_USED", "0"), 0)),
            session_id=session_id,
        )
        _register_tools(mcp, tools, state)
        _write_startup_status(
            tool_names=[str(getattr(tool, "name", "") or "") for tool in tools]
            + ["check_budget_status"]
        )

        # ── Built-in budget status tool (not part of the dynamic toolset) ──
        @mcp.tool(
            name="check_budget_status",
            description=(
                "Returns the current runtime and cost budget status. "
                "Use it only in a long run after at least five other tool calls, "
                "then at most once every 5-10 additional calls, or after an explicit "
                "budget warning. Do not call it before finishing a short task; hard "
                "runtime and cost limits are enforced automatically."
            ),
        )
        async def _check_budget_status() -> str:
            record_mcp_tool_usage("check_budget_status", state.session_id)
            return budget_status_from_env()

        if transport in {"http", "streamable-http"}:
            mcp.run(transport="streamable-http")
        elif transport == "sse":
            mcp.run(transport="sse")
        else:
            mcp.run(transport="stdio")
    except Exception:
        trace = traceback.format_exc()
        _write_startup_status(error=trace)
        try:
            print(trace, file=sys.stderr, flush=True)
        except Exception:
            pass
        try:
            with open("/tmp/chack_mcp_crash.log", "a", encoding="utf-8") as handle:
                handle.write(trace)
                handle.write("\n")
        except Exception:
            pass
        raise
    finally:
        # Both synchronous and async researcher jobs are owned by this MCP
        # process, not by the provider process that launched it. Reconcile their
        # physical child termination and persist terminal task state before
        # stdio shutdown destroys daemon supervisor threads.
        try:
            from chack_tools.researcher_administrator_agent import _shutdown_all_research_jobs

            _shutdown_all_research_jobs(timeout_seconds=15.0)
        except Exception:
            pass
        parent_watchdog_stop.set()


if __name__ == "__main__":
    main()
