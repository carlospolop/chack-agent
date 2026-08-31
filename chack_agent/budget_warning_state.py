"""Budget warning state for tool-output injection.

Two modes of operation:
 1. **In-process** (LangGraph, OpenRouter, OpenAI Compaction):
    Uses contextvars set by agent.py before executor.invoke().
    Call ``inject_budget_warning(tool_output)`` after each tool execution.

 2. **MCP subprocess** (Claude Code, Codex, Copilot, Gemini CLI):
    Reads ``CHACK_BUDGET_*`` environment variables set by the parent process.
    Call ``inject_budget_warning_from_env(tool_output)`` in the MCP proxy.
"""
from __future__ import annotations

import contextvars
import os
import time
from typing import Optional

from chack_tools.run_lifecycle import active_task_session_id, read_live_cost


# ── Env-var names (used by subprocess MCP and backend _build_env) ─────

BUDGET_ENV_KEYS = (
    "CHACK_BUDGET_START_EPOCH",
    "CHACK_BUDGET_MAX_RUNTIME_SECONDS",
    "CHACK_BUDGET_MAX_COST_USD",
    "CHACK_BUDGET_SPENT_USD",
    "CHACK_BUDGET_WARNING_RATIO",
    "CHACK_BUDGET_CRITICAL_RATIO",
    "CHACK_BUDGET_INJECTION_ENABLED",
)

# ── Context vars (in-process backends) ────────────────────────────────

_CTX_START_EPOCH: contextvars.ContextVar[float] = contextvars.ContextVar(
    "chack_budget_start_epoch", default=0.0
)
_CTX_MAX_RUNTIME: contextvars.ContextVar[float] = contextvars.ContextVar(
    "chack_budget_max_runtime_seconds", default=0.0
)
_CTX_MAX_COST: contextvars.ContextVar[float] = contextvars.ContextVar(
    "chack_budget_max_cost_usd", default=0.0
)
_CTX_SPENT_USD: contextvars.ContextVar[float] = contextvars.ContextVar(
    "chack_budget_spent_usd", default=0.0
)
_CTX_WARNING_RATIO: contextvars.ContextVar[float] = contextvars.ContextVar(
    "chack_budget_warning_ratio", default=0.6
)
_CTX_CRITICAL_RATIO: contextvars.ContextVar[float] = contextvars.ContextVar(
    "chack_budget_critical_ratio", default=0.9
)
_CTX_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "chack_budget_injection_enabled", default=True
)
# Track which milestone level was already emitted so we don't repeat
# 0 = none, 1 = warning shown, 2 = critical shown
_CTX_RUNTIME_MILESTONE: contextvars.ContextVar[int] = contextvars.ContextVar(
    "chack_budget_runtime_milestone", default=0
)
_CTX_COST_MILESTONE: contextvars.ContextVar[int] = contextvars.ContextVar(
    "chack_budget_cost_milestone", default=0
)


def set_budget_context(
    *,
    start_epoch: float,
    max_runtime_seconds: float,
    max_cost_usd: float,
    warning_ratio: float = 0.6,
    critical_ratio: float = 0.9,
    injection_enabled: bool = True,
) -> list:
    """Activate budget tracking for in-process backends.  Returns tokens."""
    return [
        _CTX_START_EPOCH.set(start_epoch),
        _CTX_MAX_RUNTIME.set(max_runtime_seconds),
        _CTX_MAX_COST.set(max_cost_usd),
        _CTX_SPENT_USD.set(0.0),
        _CTX_WARNING_RATIO.set(warning_ratio),
        _CTX_CRITICAL_RATIO.set(critical_ratio),
        _CTX_ENABLED.set(injection_enabled),
        _CTX_RUNTIME_MILESTONE.set(0),
        _CTX_COST_MILESTONE.set(0),
    ]


_CTX_VARS = [
    _CTX_START_EPOCH,
    _CTX_MAX_RUNTIME,
    _CTX_MAX_COST,
    _CTX_SPENT_USD,
    _CTX_WARNING_RATIO,
    _CTX_CRITICAL_RATIO,
    _CTX_ENABLED,
    _CTX_RUNTIME_MILESTONE,
    _CTX_COST_MILESTONE,
]


def reset_budget_context(tokens: list) -> None:
    for var, tok in zip(_CTX_VARS, tokens):
        var.reset(tok)


def update_spent_usd(spent: float) -> None:
    """Update the in-process cost tracker (called from agent.py)."""
    _CTX_SPENT_USD.set(max(0.0, float(spent or 0.0)))


# ── Warning text builders ────────────────────────────────────────────

def _runtime_warning_text(
    elapsed_seconds: float,
    max_runtime_seconds: float,
    is_critical: bool,
) -> str:
    elapsed_min = elapsed_seconds / 60.0
    max_min = max_runtime_seconds / 60.0
    remaining_min = max(0.0, (max_runtime_seconds - elapsed_seconds) / 60.0)
    if is_critical:
        notice = "Runtime budget is nearly exhausted."
        guidance = (
            "Finish immediately. Focus only on the minimum work needed "
            "to complete before the limit is reached."
        )
    else:
        notice = "Runtime budget is running low."
        guidance = (
            "Prioritize completion and organize output to finish "
            "before the limit is reached."
        )
    return (
        f"\n\n"
        f"--- [BUDGET TIME WARNING] ---\n"
        f"{notice}\n"
        f"Used {elapsed_min:.1f}/{max_min:.1f} min ({remaining_min:.1f} min remaining).\n"
        f"{guidance}\n"
        f"------------------------"
    )


def _cost_warning_text(
    spent_usd: float,
    max_cost_usd: float,
    is_critical: bool,
) -> str:
    remaining_usd = max(0.0, max_cost_usd - spent_usd)
    if is_critical:
        notice = "Cost budget is nearly exhausted."
        guidance = (
            "Finish immediately. Avoid extra tool usage and focus only on "
            "the minimum work needed to complete before the limit."
        )
    else:
        notice = "Cost budget is running low."
        guidance = (
            "Prioritize completion and reduce unnecessary tool calls."
        )
    return (
        f"\n\n"
        f"--- [BUDGET WARNING] ---\n"
        f"{notice}\n"
        f"Spent ${spent_usd:.4f}/${max_cost_usd:.4f} (${remaining_usd:.4f} remaining).\n"
        f"{guidance}\n"
        f"------------------------"
    )


# ── In-process injection (contextvars) ────────────────────────────────

def inject_budget_warning(tool_output: str) -> str:
    """Append budget warning to *tool_output* when a threshold is crossed.

    Uses contextvars — works for LangGraph, OpenRouter, OpenAI Compaction.
    """
    if not _CTX_ENABLED.get():
        return tool_output

    parts: list[str] = []

    # --- runtime ---
    max_runtime = _CTX_MAX_RUNTIME.get()
    if max_runtime > 0:
        start = _CTX_START_EPOCH.get()
        if start > 0:
            elapsed = time.time() - start
            ratio = elapsed / max_runtime
            warning_ratio = _CTX_WARNING_RATIO.get()
            critical_ratio = _CTX_CRITICAL_RATIO.get()
            milestone = _CTX_RUNTIME_MILESTONE.get()
            if ratio >= critical_ratio and milestone < 2:
                parts.append(_runtime_warning_text(elapsed, max_runtime, is_critical=True))
                _CTX_RUNTIME_MILESTONE.set(2)
            elif ratio >= warning_ratio and milestone < 1:
                parts.append(_runtime_warning_text(elapsed, max_runtime, is_critical=False))
                _CTX_RUNTIME_MILESTONE.set(1)

    # --- cost ---
    max_cost = _CTX_MAX_COST.get()
    if max_cost > 0:
        spent = _CTX_SPENT_USD.get()
        ratio = spent / max_cost
        warning_ratio = _CTX_WARNING_RATIO.get()
        critical_ratio = _CTX_CRITICAL_RATIO.get()
        milestone = _CTX_COST_MILESTONE.get()
        if ratio >= critical_ratio and milestone < 2:
            parts.append(_cost_warning_text(spent, max_cost, is_critical=True))
            _CTX_COST_MILESTONE.set(2)
        elif ratio >= warning_ratio and milestone < 1:
            parts.append(_cost_warning_text(spent, max_cost, is_critical=False))
            _CTX_COST_MILESTONE.set(1)

    if not parts:
        return tool_output
    return str(tool_output or "") + "".join(parts)


# ── MCP subprocess injection (env vars) ───────────────────────────────

def _current_spent_usd_from_env() -> float:
    env_spent = float(os.environ.get("CHACK_BUDGET_SPENT_USD", "0") or "0")
    session_id = active_task_session_id()
    shared_spent = read_live_cost(session_id) if session_id else None
    if shared_spent is not None:
        return max(0.0, float(shared_spent or 0.0))
    return max(0.0, env_spent)


def inject_budget_warning_from_env(tool_output: str) -> str:
    """Append live runtime/cost warnings for an MCP subprocess.

    Static limits come from environment variables. Live cost is read from the
    run-scoped shared state written by the parent agent, so it remains visible
    even though an already-running subprocess cannot observe later env updates.
    """
    if os.environ.get("CHACK_BUDGET_INJECTION_ENABLED", "1").strip() not in ("1", "true", "yes"):
        return tool_output

    warning_ratio = float(os.environ.get("CHACK_BUDGET_WARNING_RATIO", "0.6") or "0.6")
    critical_ratio = float(os.environ.get("CHACK_BUDGET_CRITICAL_RATIO", "0.9") or "0.9")
    parts: list[str] = []

    max_runtime = float(os.environ.get("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "0") or "0")
    start = float(os.environ.get("CHACK_BUDGET_START_EPOCH", "0") or "0")
    if max_runtime > 0 and start > 0:
        elapsed = time.time() - start
        ratio = elapsed / max_runtime
        if ratio >= critical_ratio:
            parts.append(_runtime_warning_text(elapsed, max_runtime, is_critical=True))
        elif ratio >= warning_ratio:
            parts.append(_runtime_warning_text(elapsed, max_runtime, is_critical=False))

    max_cost = float(os.environ.get("CHACK_BUDGET_MAX_COST_USD", "0") or "0")
    spent = _current_spent_usd_from_env()
    if max_cost > 0 and spent > 0:
        ratio = spent / max_cost
        if ratio >= critical_ratio:
            parts.append(_cost_warning_text(spent, max_cost, is_critical=True))
        elif ratio >= warning_ratio:
            parts.append(_cost_warning_text(spent, max_cost, is_critical=False))

    if not parts:
        return tool_output
    return str(tool_output or "") + "".join(parts)


# ── Export env vars (called from agent.py before executor.invoke) ─────

def budget_prompt_warning(
    *,
    start_epoch: float,
    max_runtime_seconds: float,
    elapsed_runtime_seconds: float = 0.0,
    spent_usd: float = 0.0,
    max_cost_usd: float = 0.0,
    warning_ratio: float = 0.6,
    critical_ratio: float = 0.9,
) -> str:
    """Return a budget warning string suitable for injecting into a follow-up
    prompt (between invocations).  Returns empty string if no threshold is met.

    This is the fallback channel for subprocess backends whose native tools
    (bash, read, write…) cannot have warnings injected into their results.
    """
    parts: list[str] = []

    if max_runtime_seconds > 0 and start_epoch > 0:
        elapsed = elapsed_runtime_seconds or (time.time() - start_epoch)
        ratio = elapsed / max_runtime_seconds
        if ratio >= critical_ratio:
            parts.append(_runtime_warning_text(elapsed, max_runtime_seconds, is_critical=True))
        elif ratio >= warning_ratio:
            parts.append(_runtime_warning_text(elapsed, max_runtime_seconds, is_critical=False))

    if max_cost_usd > 0 and spent_usd > 0:
        ratio = spent_usd / max_cost_usd
        if ratio >= critical_ratio:
            parts.append(_cost_warning_text(spent_usd, max_cost_usd, is_critical=True))
        elif ratio >= warning_ratio:
            parts.append(_cost_warning_text(spent_usd, max_cost_usd, is_critical=False))

    return "".join(parts)


def export_budget_env(
    *,
    start_epoch: float,
    max_runtime_seconds: float,
    max_cost_usd: float = 0.0,
    warning_ratio: float = 0.6,
    critical_ratio: float = 0.9,
    injection_enabled: bool = True,
) -> None:
    """Set budget env vars so CLI backends and their MCP subprocesses
    can read them."""
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(start_epoch)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = str(max_runtime_seconds)
    os.environ["CHACK_BUDGET_MAX_COST_USD"] = str(max_cost_usd)
    os.environ["CHACK_BUDGET_SPENT_USD"] = "0"
    os.environ["CHACK_BUDGET_WARNING_RATIO"] = str(warning_ratio)
    os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = str(critical_ratio)
    os.environ["CHACK_BUDGET_INJECTION_ENABLED"] = "1" if injection_enabled else "0"


def export_spent_usd_env(spent: float) -> None:
    """Update the env var with current cost spent (for subprocess visibility)."""
    os.environ["CHACK_BUDGET_SPENT_USD"] = str(max(0.0, float(spent or 0.0)))


# ── MCP tool: budget status report from env ──────────────────────────

def budget_status_from_env() -> str:
    """Return a human-readable budget status string using env vars.

    Intended for the ``check_budget_status`` MCP tool so that subprocess
    backends can query the current budget situation mid-run.
    """
    lines: list[str] = []

    max_runtime = float(os.environ.get("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "0") or "0")
    start = float(os.environ.get("CHACK_BUDGET_START_EPOCH", "0") or "0")
    if max_runtime > 0 and start > 0:
        elapsed = time.time() - start
        remaining = max(0.0, max_runtime - elapsed)
        ratio = elapsed / max_runtime
        lines.append(
            f"Runtime: {elapsed / 60:.1f}/{max_runtime / 60:.1f} min used "
            f"({remaining / 60:.1f} min remaining, {ratio * 100:.0f}% consumed)"
        )
        warning_ratio = float(os.environ.get("CHACK_BUDGET_WARNING_RATIO", "0.6") or "0.6")
        critical_ratio = float(os.environ.get("CHACK_BUDGET_CRITICAL_RATIO", "0.9") or "0.9")
        if ratio >= critical_ratio:
            lines.append("STATUS: CRITICAL — Finish immediately!")
        elif ratio >= warning_ratio:
            lines.append("STATUS: WARNING — Prioritize completion.")
        else:
            lines.append("STATUS: OK")
    else:
        lines.append("Runtime: No limit configured.")

    max_cost = float(os.environ.get("CHACK_BUDGET_MAX_COST_USD", "0") or "0")
    spent = _current_spent_usd_from_env()
    if max_cost > 0:
        remaining_cost = max(0.0, max_cost - spent)
        ratio = spent / max_cost if max_cost > 0 else 0.0
        lines.append(
            f"Cost: ${spent:.4f}/${max_cost:.4f} spent "
            f"(${remaining_cost:.4f} remaining, {ratio * 100:.0f}% consumed)"
        )
    else:
        lines.append("Cost: No limit configured.")

    return "\n".join(lines)
