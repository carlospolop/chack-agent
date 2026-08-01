import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .context import current_log_context

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


_QUEUE_URL_ENV = "CHACK_LOG_QUEUE_URL"
_QUEUE_REGION_ENV = "CHACK_LOG_QUEUE_REGION"
_HTTP_URL_ENV = "CHACK_LOGS_HTTP_URL"
_HTTP_TIMEOUT_ENV = "CHACK_LOGS_HTTP_TIMEOUT"
_TOOL_RESULT_MAX_BYTES_ENV = "CHACK_TELEMETRY_TOOL_RESULT_MAX_BYTES"
_SOURCE = "chack-agent"
_DEFAULT_TOOL_RESULT_MAX_BYTES = 8192

_LOCK = threading.Lock()
_CLIENT = None
_QUEUE_URL: Optional[str] = None
_REGION: Optional[str] = None


def _emit_stdout_event(event: Dict[str, Any]) -> None:
    """Mirror telemetry events to stdout so they are visible in CI logs."""
    if (os.environ.get("CHACK_DISABLE_STDOUT_EVENTS", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    try:
        print(
            f"CHACK_EVENT {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )
    except Exception:
        # Logging should never break agent execution.
        return


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_queue_url() -> str:
    return (os.environ.get(_QUEUE_URL_ENV, "") or "").strip()


def _infer_region(queue_url: str) -> Optional[str]:
    if not queue_url:
        return None
    match = re.search(r"sqs[.-]([a-z0-9-]+)\.amazonaws\.com", queue_url)
    if match:
        return match.group(1)
    return None


def _get_client():
    global _CLIENT, _QUEUE_URL, _REGION
    if boto3 is None:
        return None
    with _LOCK:
        queue_url = _resolve_queue_url()
        if not queue_url:
            _QUEUE_URL = None
            _CLIENT = None
            _REGION = None
            return None
        if queue_url != _QUEUE_URL or _CLIENT is None:
            region = (os.environ.get(_QUEUE_REGION_ENV, "") or "").strip()
            if not region:
                region = _infer_region(queue_url) or None
            _QUEUE_URL = queue_url
            _REGION = region
            if region:
                _CLIENT = boto3.client("sqs", region_name=region)
            else:
                _CLIENT = boto3.client("sqs")
        return _CLIENT


def _queue_url() -> Optional[str]:
    return _QUEUE_URL or _resolve_queue_url() or None


def _truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max(0, max_len - 3)] + "..."


def _sanitize(value: Any, *, max_str_len: int = 4000) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return _truncate(value.replace("\x00", ""), max_str_len)
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v, max_str_len=max_str_len) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, max_str_len=max_str_len) for v in value]
    return _truncate(str(value), max_str_len)


def _serialized_size_and_preview(value: Any, *, preview_len: int = 512) -> tuple[int, str]:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        serialized = str(value)
    return len(serialized.encode("utf-8", errors="replace")), _truncate(
        serialized.replace("\x00", ""),
        preview_len,
    )


def _tool_result_max_bytes() -> int:
    raw = (os.environ.get(_TOOL_RESULT_MAX_BYTES_ENV, "") or "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_TOOL_RESULT_MAX_BYTES
    except (TypeError, ValueError):
        value = _DEFAULT_TOOL_RESULT_MAX_BYTES
    return max(512, min(value, 65536))


def _compact_tool_event_payload(
    event_type: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Bound completed tool results in telemetry without mutating runtime data.

    Tool arguments and small results stay intact so status/checkpoint recovery can
    reconstruct successful writes. Large read/search results are replaced only in
    the telemetry copy with their size and a short preview; the agent still receives
    the original result.
    """
    if event_type != "tool_called" or not isinstance(payload, dict):
        return payload
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict) or "result" not in tool_input:
        return payload
    result_size, preview = _serialized_size_and_preview(tool_input.get("result"))
    if result_size <= _tool_result_max_bytes():
        return payload
    compact_input = dict(tool_input)
    compact_input["result"] = {
        "_telemetry_truncated": True,
        "original_bytes": result_size,
        "preview": preview,
    }
    compact_payload = dict(payload)
    compact_payload["tool_input"] = compact_input
    return compact_payload


def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None, **context) -> bool:
    http_url = (os.environ.get(_HTTP_URL_ENV, "") or "").strip()
    event_type = (event_type or "").strip().lower()
    if not event_type:
        return False

    base_context = current_log_context()
    base_context.pop("_chack_tool_progress_callback", None)
    for key, value in context.items():
        if value is None:
            continue
        base_context[str(key)] = value

    event_payload = _compact_tool_event_payload(event_type, payload or {})
    event = {
        "schema_version": 1,
        "source": _SOURCE,
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "ts": _timestamp(),
        "context": _sanitize(base_context),
        "payload": _sanitize(event_payload),
    }

    _emit_stdout_event(event)

    if http_url and requests is not None:
        timeout_raw = os.environ.get(_HTTP_TIMEOUT_ENV, "")
        try:
            timeout = float(timeout_raw) if timeout_raw else 5.0
        except Exception:
            timeout = 5.0
        try:
            resp = requests.post(http_url, json=event, timeout=timeout)
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    client = _get_client()
    queue_url = _queue_url()
    if not client or not queue_url:
        return False
    try:
        client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(event, ensure_ascii=False),
        )
    except Exception:
        return False
    return True
