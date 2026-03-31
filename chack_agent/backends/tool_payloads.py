from __future__ import annotations

import base64
import os
import tempfile
from typing import Any


CHACK_TOOLS_OVERRIDE_B64_ENV = "CHACK_TOOLS_OVERRIDE_B64"
CHACK_TOOLS_APPEND_B64_ENV = "CHACK_TOOLS_APPEND_B64"
CHACK_TOOLS_OVERRIDE_B64_PATH_ENV = "CHACK_TOOLS_OVERRIDE_B64_PATH"
CHACK_TOOLS_APPEND_B64_PATH_ENV = "CHACK_TOOLS_APPEND_B64_PATH"
CHACK_TOOLS_CONFIG_JSON_PATH_ENV = "CHACK_TOOLS_CONFIG_JSON_PATH"
CHACK_ALLOWED_TOOLS_JSON_PATH_ENV = "CHACK_ALLOWED_TOOLS_JSON_PATH"
CHACK_INLINE_ENV_VALUE_MAX_CHARS = 24000


def _cloudpickle():
    try:
        import cloudpickle  # type: ignore

        return cloudpickle
    except Exception as exc:  # pragma: no cover - depends on runtime packaging
        raise RuntimeError(
            "cloudpickle is required to pass custom tools to CLI backends"
        ) from exc


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
