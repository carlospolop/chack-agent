from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
from typing import Any


CHACK_TOOLS_OVERRIDE_B64_ENV = "CHACK_TOOLS_OVERRIDE_B64"
CHACK_TOOLS_APPEND_B64_ENV = "CHACK_TOOLS_APPEND_B64"
CHACK_TOOLS_OVERRIDE_B64_PATH_ENV = "CHACK_TOOLS_OVERRIDE_B64_PATH"
CHACK_TOOLS_APPEND_B64_PATH_ENV = "CHACK_TOOLS_APPEND_B64_PATH"
CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV = "CHACK_TOOLS_OVERRIDE_NAMES_JSON"
CHACK_TOOLS_APPEND_NAMES_JSON_ENV = "CHACK_TOOLS_APPEND_NAMES_JSON"
CHACK_TOOLS_CONFIG_JSON_PATH_ENV = "CHACK_TOOLS_CONFIG_JSON_PATH"
CHACK_ALLOWED_TOOLS_JSON_PATH_ENV = "CHACK_ALLOWED_TOOLS_JSON_PATH"
CHACK_INLINE_ENV_VALUE_MAX_CHARS = 24000
MCP_INPUT_SCHEMA_ATTRIBUTE = "chack_preserve_mcp_input_schema"
# The caller's own PYTHONPATH (JSON: a string or null), so the exec tool can
# give shell commands the caller's environment instead of the import roots
# added for the MCP server.
CHACK_CALLER_PYTHONPATH_JSON_ENV = "CHACK_CALLER_PYTHONPATH_JSON"


def augment_subprocess_pythonpath(env: dict[str, str]) -> None:
    """Preserve application import roots for MCP subprocess deserialization."""
    env.setdefault(CHACK_CALLER_PYTHONPATH_JSON_ENV, json.dumps(env.get("PYTHONPATH")))
    candidates: list[str] = []
    candidates.extend(str(env.get("PYTHONPATH", "") or "").split(os.pathsep))
    candidates.extend(str(entry or os.getcwd()) for entry in sys.path)
    if sys.argv:
        candidates.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    candidates.append(os.getcwd())

    resolved: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        path = os.path.abspath(os.path.expanduser(raw))
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        resolved.append(path)
    if resolved:
        env["PYTHONPATH"] = os.pathsep.join(resolved)


def _cloudpickle():
    try:
        import cloudpickle  # type: ignore

        return cloudpickle
    except Exception as exc:  # pragma: no cover - depends on runtime packaging
        raise RuntimeError(
            "cloudpickle is required to pass custom tools to CLI backends"
        ) from exc


def preserve_mcp_input_schema(tool: Any) -> Any:
    """Publish the declared JSON input schema when serving this tool over MCP."""
    setattr(tool, MCP_INPUT_SCHEMA_ATTRIBUTE, True)
    return tool


def serialize_tools_payload(tools: list[Any] | None) -> str:
    if tools is None:
        return ""
    payload = _cloudpickle().dumps(list(tools))
    return base64.b64encode(payload).decode("ascii")


def write_payload_to_file(payload: str, *, prefix: str, directory: str = "") -> str:
    raw = str(payload or "")
    if not raw:
        return ""
    target_dir = directory or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt", dir=target_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(raw)
    return path


def read_payload_from_env_or_file(payload: str = "", payload_path: str = "") -> str:
    raw = str(payload or "").strip()
    if raw:
        return raw
    path = str(payload_path or "").strip()
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""


def deserialize_tools_payload(payload_b64: str) -> list[Any]:
    raw = str(payload_b64 or "").strip()
    if not raw:
        return []
    try:
        encoded = base64.b64decode(raw.encode("ascii"))
    except Exception as exc:
        raise RuntimeError("Invalid serialized tool payload encoding") from exc
    try:
        tools = _cloudpickle().loads(encoded)
    except Exception as exc:
        raise RuntimeError("Failed to deserialize serialized tool payload") from exc
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise RuntimeError("Serialized tool payload must deserialize to a list")
    return list(tools)


def configured_mcp_env_names(tools_config_json: str) -> tuple[str, ...]:
    """Extra environment names explicitly selected by the trusted caller."""
    names = json.loads(tools_config_json or "{}").get("mcp_env_vars", [])
    if not isinstance(names, list) or len(names) > 64:
        raise ValueError("tools.mcp_env_vars must be a list of at most 64 environment names")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names):
        raise ValueError("tools.mcp_env_vars contains an invalid environment name")
    return tuple(dict.fromkeys(names))
