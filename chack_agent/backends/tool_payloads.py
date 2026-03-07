from __future__ import annotations

import base64
from typing import Any


CHACK_TOOLS_OVERRIDE_B64_ENV = "CHACK_TOOLS_OVERRIDE_B64"
CHACK_TOOLS_APPEND_B64_ENV = "CHACK_TOOLS_APPEND_B64"


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