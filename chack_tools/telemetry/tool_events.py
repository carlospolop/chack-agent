from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar

from .context import current_log_context
from .sqs_logger import log_event

T = TypeVar("T")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _notify_tool_progress(event_type: str, payload: Dict[str, Any]) -> None:
    callback = current_log_context().get("_chack_tool_progress_callback")
    if not callable(callback):
        return
    try:
        callback(event_type, dict(payload or {}))
    except Exception:
        return


def log_tool_started(tool: str, tool_input: Dict[str, Any]) -> str:
    start_ts = _timestamp()
    payload = {
        "tool": tool,
        "tool_input": tool_input,
        "tool_start_ts": start_ts,
    }
    _notify_tool_progress("tool_started", payload)
    log_event(
        "tool_started",
        payload=payload,
    )
    return start_ts


def log_tool_executed(
    tool: str,
    tool_input: Dict[str, Any],
    *,
    start_ts: str,
    end_ts: Optional[str],
    duration_ms: Optional[int],
    error: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "tool": tool,
        "tool_input": tool_input,
        "tool_start_ts": start_ts,
        "tool_end_ts": end_ts,
        "duration_ms": duration_ms,
    }
    if error:
        payload["error"] = error
    _notify_tool_progress("tool_executed", payload)
    log_event("tool_executed", payload=payload)


def log_tool_error(
    tool: str,
    tool_input: Dict[str, Any],
    *,
    error: str,
    trace: str,
) -> None:
    payload = {
        "tool": tool,
        "tool_input": tool_input,
        "error": error,
        "traceback": trace,
    }
    _notify_tool_progress("tool_error", payload)
    log_event(
        "tool_error",
        payload=payload,
    )


def run_with_tool_logging(
    tool: str,
    tool_input: Dict[str, Any],
    func: Callable[[], T],
) -> T:
    start_ts = log_tool_started(tool, tool_input)
    start_time = time.time()
    error = None
    try:
        return func()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            log_tool_error(
                tool,
                tool_input,
                error=error,
                trace=traceback.format_exc(),
            )
        except Exception:
            pass
        raise
    finally:
        end_ts = _timestamp()
        duration_ms = int((time.time() - start_time) * 1000)
        log_tool_executed(
            tool,
            tool_input,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_ms=duration_ms,
            error=error,
        )
