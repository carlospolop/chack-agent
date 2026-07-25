from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_STATE_ROOT_ENV = "CHACK_RUN_STATE_DIR"
_TASK_SESSION_ENV = "CHACK_TASK_SESSION_ID"


@dataclass(frozen=True)
class ToolBudgetClaim:
    allowed: bool
    used: int
    max_tools: int
    milestone: str = ""  # warning | critical | limit


def active_task_session_id(session_id: str = "") -> str:
    value = str(session_id or "").strip()
    if value:
        return value
    value = str(os.environ.get(_TASK_SESSION_ENV, "") or "").strip()
    if value:
        return value
    try:
        from chack_tools.task_steps_manager_state import current_session_id

        return str(current_session_id() or "").strip()
    except Exception:
        return ""


def _state_root() -> Path:
    configured = str(os.environ.get(_STATE_ROOT_ENV, "") or "").strip()
    root = Path(configured).expanduser() if configured else Path(tempfile.gettempdir()) / "chack-agent-run-state"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root


def _state_path(kind: str, session_id: str) -> Path | None:
    sid = active_task_session_id(session_id)
    if not sid:
        return None
    digest = hashlib.sha256(sid.encode("utf-8", errors="replace")).hexdigest()
    return _state_root() / f"{digest}.{kind}.json"


def _read_locked_json(handle, default: Any, *, strict: bool = False) -> Any:
    handle.seek(0)
    raw = handle.read()
    if not raw.strip():
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        if strict:
            raise RuntimeError(
                "Corrupt Chack run-state file; refusing to reset a finite budget"
            ) from exc
        return default


def _write_locked_json(handle, value: Any) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    handle.flush()
    os.fsync(handle.fileno())


def claim_non_task_tool_slot(
    session_id: str,
    max_tools: int,
    *,
    warning_ratio: float = 0.6,
    critical_ratio: float = 0.9,
) -> ToolBudgetClaim:
    """Atomically claim one run-wide non-task tool slot.

    The counter is file-backed so a restarted MCP subprocess continues from the
    same count rather than silently resetting the limit.
    """
    maximum = max(0, int(max_tools or 0))
    path = _state_path("tools", session_id)
    if maximum <= 0:
        return ToolBudgetClaim(True, 0, maximum)
    if path is None:
        return ToolBudgetClaim(False, 0, maximum, "limit")

    warning_at = max(1, int(math.ceil(maximum * max(0.0, float(warning_ratio)))))
    critical_at = max(warning_at, int(math.ceil(maximum * max(0.0, float(critical_ratio)))))
    critical_at = min(maximum, critical_at)

    with path.open("a+", encoding="utf-8") as handle:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = _read_locked_json(
            handle,
            {"used": 0, "milestone": 0},
            strict=True,
        )
        used = max(0, int((state or {}).get("used", 0) or 0))
        emitted = max(0, int((state or {}).get("milestone", 0) or 0))
        if used >= maximum:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return ToolBudgetClaim(False, used, maximum, "limit")

        used += 1
        milestone = ""
        if used >= critical_at and emitted < 2:
            emitted = 2
            milestone = "critical"
        elif used >= warning_at and emitted < 1:
            emitted = 1
            milestone = "warning"
        _write_locked_json(handle, {"used": used, "milestone": emitted})
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return ToolBudgetClaim(True, used, maximum, milestone)


def tool_budget_warning(claim: ToolBudgetClaim) -> str:
    if claim.milestone not in {"warning", "critical"} or claim.max_tools <= 0:
        return ""
    remaining = max(0, claim.max_tools - claim.used)
    if claim.milestone == "critical":
        notice = "Tool budget is nearly exhausted."
        guidance = "Stop gathering context and finish the requested work and final response now."
    else:
        notice = "Tool budget is running low."
        guidance = "Avoid optional checks and prioritize completing the task and final response."
    return (
        "\n\n--- [TOOL BUDGET WARNING] ---\n"
        f"{notice}\nUsed {claim.used}/{claim.max_tools} non-task calls ({remaining} remaining).\n"
        f"{guidance}\n"
        "-----------------------------"
    )


def record_mcp_tool_usage(tool_name: str, session_id: str = "") -> None:
    """Persist one top-level MCP tool attempt for complete run telemetry.

    Provider CLIs can compact their transcript, time out, or exit before
    returning earlier tool-call events to Chack. This file-backed counter is
    updated at the actual MCP execution boundary, so the parent process can
    still report every call. It is scan-local, tiny, and removed with the other
    run-state files when the agent finishes.
    """
    normalized_name = str(tool_name or "").strip()
    path = _state_path("tool-usage", session_id)
    if not normalized_name or path is None:
        return
    try:
        with path.open("a+", encoding="utf-8") as handle:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state = _read_locked_json(handle, {"counts": {}})
            raw_counts = (state or {}).get("counts") or {}
            counts = {
                str(name): max(0, int(count or 0))
                for name, count in raw_counts.items()
                if str(name).strip()
            }
            counts[normalized_name] = counts.get(normalized_name, 0) + 1
            _write_locked_json(handle, {"counts": counts})
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, TypeError, ValueError):
        # Observability must never prevent the requested tool from executing.
        return


def read_mcp_tool_usage(session_id: str = "") -> Counter[str]:
    """Return the cumulative MCP-boundary tool counts for one agent run."""
    path = _state_path("tool-usage", session_id)
    if path is None or not path.exists():
        return Counter()
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            state = _read_locked_json(handle, {"counts": {}})
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return Counter(
            {
                str(name): max(0, int(count or 0))
                for name, count in ((state or {}).get("counts") or {}).items()
                if str(name).strip() and int(count or 0) > 0
            }
        )
    except (OSError, TypeError, ValueError):
        return Counter()


def mark_task_manager_initialized(session_id: str) -> None:
    """Persist init-first state so an MCP restart does not require a second init."""
    path = _state_path("task-manager", session_id)
    if path is None:
        return
    with path.open("a+", encoding="utf-8") as handle:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _write_locked_json(handle, {"initialized": True})
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def task_manager_initialized(session_id: str) -> bool:
    path = _state_path("task-manager", session_id)
    if path is None or not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            state = _read_locked_json(handle, {})
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return bool((state or {}).get("initialized"))
    except (OSError, TypeError, ValueError):
        return False


def write_live_cost(session_id: str, spent_usd: float) -> None:
    path = _state_path("budget", session_id)
    if path is None:
        return
    with path.open("a+", encoding="utf-8") as handle:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        previous = _read_locked_json(handle, {"spent_usd": 0.0})
        previous_spent = max(0.0, float((previous or {}).get("spent_usd", 0.0) or 0.0))
        current = max(previous_spent, max(0.0, float(spent_usd or 0.0)))
        _write_locked_json(handle, {"spent_usd": current})
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_live_cost(session_id: str = "") -> float | None:
    path = _state_path("budget", session_id)
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            state = _read_locked_json(handle, {})
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return max(0.0, float((state or {}).get("spent_usd", 0.0) or 0.0))
    except (OSError, TypeError, ValueError):
        return None


def register_process_group(session_id: str, pgid: int) -> None:
    path = _state_path("process-groups", session_id)
    group = int(pgid or 0)
    if path is None or group <= 1 or group == os.getpgrp():
        return
    with path.open("a+", encoding="utf-8") as handle:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = _read_locked_json(handle, {"groups": []})
        groups = {int(value) for value in ((state or {}).get("groups") or []) if int(value) > 1}
        groups.add(group)
        _write_locked_json(handle, {"groups": sorted(groups)})
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(int(pgid), sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def terminate_process_group(pgid: int, *, grace_seconds: float = 0.5) -> None:
    group = int(pgid or 0)
    if group <= 1 or group == os.getpgrp():
        return
    if not _signal_group(group, signal.SIGTERM):
        return
    deadline = time.time() + max(0.0, float(grace_seconds))
    while time.time() < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.05)
    _signal_group(group, signal.SIGKILL)


def cleanup_process_groups(session_id: str, *, grace_seconds: float = 0.5) -> list[int]:
    path = _state_path("process-groups", session_id)
    if path is None or not path.exists():
        return []
    groups: list[int] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            state = _read_locked_json(handle, {"groups": []})
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        groups = sorted({int(value) for value in ((state or {}).get("groups") or []) if int(value) > 1})
    except (OSError, TypeError, ValueError):
        groups = []
    for group in groups:
        terminate_process_group(group, grace_seconds=grace_seconds)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    return groups


def cleanup_run_state(session_id: str) -> None:
    cleanup_process_groups(session_id)
    for kind in ("tools", "tool-usage", "budget", "process-groups", "task-manager"):
        path = _state_path(kind, session_id)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
