"""Tests for budget_warning_state module."""
from __future__ import annotations

import os
import time

import pytest

from chack_agent.budget_warning_state import (
    BUDGET_ENV_KEYS,
    budget_prompt_warning,
    budget_status_from_env,
    export_budget_env,
    export_spent_usd_env,
    inject_budget_warning,
    inject_budget_warning_from_env,
    reset_budget_context,
    set_budget_context,
    update_spent_usd,
)
from chack_agent.config import AgentConfig


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_agent_config_budget_defaults():
    cfg = AgentConfig()
    assert cfg.budget_warning_ratio == 0.6
    assert cfg.budget_critical_ratio == 0.9
    assert cfg.budget_tool_injection_enabled is True


def test_agent_config_custom_thresholds():
    cfg = AgentConfig(budget_warning_ratio=0.5, budget_critical_ratio=0.8)
    assert cfg.budget_warning_ratio == 0.5
    assert cfg.budget_critical_ratio == 0.8


# ---------------------------------------------------------------------------
# In-process (contextvars) injection
# ---------------------------------------------------------------------------

def test_inject_no_budget_set():
    """Without budget context, no warning is injected."""
    assert inject_budget_warning("hello") == "hello"


def test_inject_runtime_warning():
    tokens = set_budget_context(
        start_epoch=time.time() - 40,
        max_runtime_seconds=60,
        max_cost_usd=0.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    try:
        result = inject_budget_warning("tool result")
        assert "BUDGET TIME WARNING" in result
        assert "Runtime budget is running low." in result
        assert "tool result" in result
    finally:
        reset_budget_context(tokens)


def test_inject_runtime_critical():
    tokens = set_budget_context(
        start_epoch=time.time() - 56,
        max_runtime_seconds=60,
        max_cost_usd=0.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    try:
        result = inject_budget_warning("tool result")
        assert "BUDGET TIME WARNING" in result
        assert "Runtime budget is nearly exhausted." in result
    finally:
        reset_budget_context(tokens)


def test_inject_runtime_below_threshold():
    tokens = set_budget_context(
        start_epoch=time.time() - 10,
        max_runtime_seconds=60,
        max_cost_usd=0.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    try:
        result = inject_budget_warning("tool result")
        assert result == "tool result"
    finally:
        reset_budget_context(tokens)


def test_inject_cost_warning():
    tokens = set_budget_context(
        start_epoch=time.time(),
        max_runtime_seconds=0,
        max_cost_usd=10.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    update_spent_usd(7.0)
    try:
        result = inject_budget_warning("tool result")
        assert "BUDGET WARNING" in result
        assert "Cost budget is running low." in result
    finally:
        reset_budget_context(tokens)


def test_inject_cost_critical():
    tokens = set_budget_context(
        start_epoch=time.time(),
        max_runtime_seconds=0,
        max_cost_usd=10.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    update_spent_usd(9.5)
    try:
        result = inject_budget_warning("tool result")
        assert "BUDGET WARNING" in result
        assert "Cost budget is nearly exhausted." in result
    finally:
        reset_budget_context(tokens)


def test_inject_cost_below_threshold():
    tokens = set_budget_context(
        start_epoch=time.time(),
        max_runtime_seconds=0,
        max_cost_usd=10.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    update_spent_usd(2.0)
    try:
        result = inject_budget_warning("tool result")
        assert result == "tool result"
    finally:
        reset_budget_context(tokens)


def test_inject_disabled():
    tokens = set_budget_context(
        start_epoch=time.time() - 55,
        max_runtime_seconds=60,
        max_cost_usd=0.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
        injection_enabled=False,
    )
    try:
        result = inject_budget_warning("tool result")
        assert result == "tool result"
    finally:
        reset_budget_context(tokens)


def test_milestone_escalation():
    """Warning shown once, then critical replaces it on next check."""
    tokens = set_budget_context(
        start_epoch=time.time() - 40,
        max_runtime_seconds=60,
        max_cost_usd=0.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    try:
        r1 = inject_budget_warning("first")
        assert "BUDGET TIME WARNING" in r1

        # Second call at same threshold should NOT re-inject (milestone already hit)
        r2 = inject_budget_warning("second")
        assert r2 == "second"
    finally:
        reset_budget_context(tokens)


def test_both_runtime_and_cost():
    tokens = set_budget_context(
        start_epoch=time.time() - 40,
        max_runtime_seconds=60,
        max_cost_usd=10.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    update_spent_usd(7.0)
    try:
        result = inject_budget_warning("tool result")
        assert "Runtime budget" in result
        assert "Cost budget" in result
    finally:
        reset_budget_context(tokens)


# ---------------------------------------------------------------------------
# MCP subprocess injection (env vars)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_budget_env():
    """Remove budget env vars after each test."""
    yield
    for key in BUDGET_ENV_KEYS:
        os.environ.pop(key, None)


def test_env_no_env_set():
    assert inject_budget_warning_from_env("hello") == "hello"


def test_env_runtime_warning():
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 40)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
    os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
    os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
    result = inject_budget_warning_from_env("tool result")
    assert "BUDGET TIME WARNING" in result
    assert "Runtime budget is running low." in result


def test_env_runtime_critical():
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 55)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
    os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
    os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
    result = inject_budget_warning_from_env("tool result")
    assert "BUDGET TIME WARNING" in result
    assert "Runtime budget is nearly exhausted." in result


def test_env_below_threshold():
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 10)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
    os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
    os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
    result = inject_budget_warning_from_env("tool result")
    assert result == "tool result"


def test_env_disabled():
    os.environ["CHACK_BUDGET_INJECTION_ENABLED"] = "0"
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 55)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
    result = inject_budget_warning_from_env("tool result")
    assert result == "tool result"


# ---------------------------------------------------------------------------
# export_budget_env
# ---------------------------------------------------------------------------

def test_export_budget_env():
    export_budget_env(
        start_epoch=1000.0,
        max_runtime_seconds=300.0,
        max_cost_usd=5.0,
        warning_ratio=0.5,
        critical_ratio=0.8,
        injection_enabled=True,
    )
    assert os.environ["CHACK_BUDGET_START_EPOCH"] == "1000.0"
    assert os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] == "300.0"
    assert os.environ["CHACK_BUDGET_MAX_COST_USD"] == "5.0"
    assert os.environ["CHACK_BUDGET_WARNING_RATIO"] == "0.5"
    assert os.environ["CHACK_BUDGET_CRITICAL_RATIO"] == "0.8"
    assert os.environ["CHACK_BUDGET_INJECTION_ENABLED"] == "1"


# ---------------------------------------------------------------------------
# Prompt-level warning (budget_prompt_warning)
# ---------------------------------------------------------------------------

def test_prompt_warning_runtime():
    w = budget_prompt_warning(
        start_epoch=time.time() - 45,
        max_runtime_seconds=60,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    assert "BUDGET TIME WARNING" in w
    assert "Runtime budget" in w


def test_prompt_warning_cost_critical():
    w = budget_prompt_warning(
        start_epoch=time.time(),
        max_runtime_seconds=0,
        spent_usd=9.5,
        max_cost_usd=10.0,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    assert "BUDGET WARNING" in w
    assert "Cost budget" in w


def test_prompt_warning_below_threshold():
    w = budget_prompt_warning(
        start_epoch=time.time() - 10,
        max_runtime_seconds=60,
        warning_ratio=0.6,
        critical_ratio=0.9,
    )
    assert w == ""


# ---------------------------------------------------------------------------
# export_spent_usd_env
# ---------------------------------------------------------------------------

def test_export_spent_usd_env():
    export_spent_usd_env(3.75)
    assert os.environ["CHACK_BUDGET_SPENT_USD"] == "3.75"


def test_export_budget_env_includes_spent_usd():
    export_budget_env(
        start_epoch=1000.0,
        max_runtime_seconds=300.0,
        max_cost_usd=5.0,
    )
    assert os.environ["CHACK_BUDGET_SPENT_USD"] == "0"


# ---------------------------------------------------------------------------
# budget_status_from_env (MCP tool)
# ---------------------------------------------------------------------------

def test_budget_status_no_limits():
    for key in BUDGET_ENV_KEYS:
        os.environ.pop(key, None)
    status = budget_status_from_env()
    assert "No limit configured" in status


def test_budget_status_runtime_ok():
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 10)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
    os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
    os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
    status = budget_status_from_env()
    assert "STATUS: OK" in status
    assert "Runtime:" in status


def test_budget_status_runtime_warning():
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 42)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
    os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
    os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
    status = budget_status_from_env()
    assert "STATUS: WARNING" in status


def test_budget_status_runtime_critical():
    os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 55)
    os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
    os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
    os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
    status = budget_status_from_env()
    assert "STATUS: CRITICAL" in status


def test_budget_status_cost():
    os.environ["CHACK_BUDGET_MAX_COST_USD"] = "10.0"
    os.environ["CHACK_BUDGET_SPENT_USD"] = "7.5"
    status = budget_status_from_env()
    assert "Cost:" in status
    assert "$7.5" in status
