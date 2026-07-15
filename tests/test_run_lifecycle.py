from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from chack_agent import Chack
from chack_agent.budget_warning_state import inject_budget_warning_from_env
from chack_agent.live_cost_state import report_live_usage
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)
from chack_tools.exec_tool import ExecTool
from chack_tools.run_lifecycle import (
    claim_non_task_tool_slot,
    cleanup_run_state,
    mark_task_manager_initialized,
    task_manager_initialized,
    tool_budget_warning,
    write_live_cost,
)
from chack_tools.task_steps_manager_state import STORE, current_run_label, current_session_id
from chack_tools.tool_usage_state import (
    effective_max_tools_used,
    reset_active_max_tools_used,
    set_active_max_tools_used,
)


@pytest.fixture
def isolated_run_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RUN_STATE_DIR", str(tmp_path))
    yield tmp_path


def test_tool_budget_survives_independent_claimers(isolated_run_state):
    session_id = "global-tool-budget"
    first = claim_non_task_tool_slot(session_id, 3, warning_ratio=0.5, critical_ratio=0.8)
    second = claim_non_task_tool_slot(session_id, 3, warning_ratio=0.5, critical_ratio=0.8)
    third = claim_non_task_tool_slot(session_id, 3, warning_ratio=0.5, critical_ratio=0.8)
    blocked = claim_non_task_tool_slot(session_id, 3, warning_ratio=0.5, critical_ratio=0.8)

    assert first.allowed and first.used == 1
    assert second.allowed and second.used == 2 and second.milestone == "warning"
    assert third.allowed and third.used == 3 and third.milestone == "critical"
    assert not blocked.allowed and blocked.used == 3 and blocked.milestone == "limit"
    assert "finish the requested work" in tool_budget_warning(third)
    cleanup_run_state(session_id)


def test_parallel_process_claims_are_atomic(isolated_run_state):
    session_id = "parallel-global-budget"
    code = (
        "import json; from chack_tools.run_lifecycle import claim_non_task_tool_slot; "
        f"c=claim_non_task_tool_slot({session_id!r}, 5); "
        "print(json.dumps({'allowed':c.allowed,'used':c.used}))"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        for _ in range(12)
    ]
    claims = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        claims.append(json.loads(stdout))
    assert sum(1 for claim in claims if claim["allowed"]) == 5
    assert max(claim["used"] for claim in claims) == 5
    cleanup_run_state(session_id)


def test_finite_budget_fails_closed_without_run_id(isolated_run_state, monkeypatch):
    monkeypatch.delenv("CHACK_TASK_SESSION_ID", raising=False)
    claim = claim_non_task_tool_slot("", 3)
    assert not claim.allowed
    assert claim.milestone == "limit"


def test_corrupt_finite_budget_state_fails_closed(isolated_run_state):
    session_id = "corrupt-budget"
    assert claim_non_task_tool_slot(session_id, 3).allowed
    state_path = next(isolated_run_state.glob("*.tools.json"))
    state_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Corrupt Chack run-state"):
        claim_non_task_tool_slot(session_id, 3)
    cleanup_run_state(session_id)


def test_task_manager_init_survives_mcp_restart_state(isolated_run_state):
    session_id = "persistent-task-init"
    assert not task_manager_initialized(session_id)
    mark_task_manager_initialized(session_id)
    assert task_manager_initialized(session_id)
    cleanup_run_state(session_id)
    assert not task_manager_initialized(session_id)


def test_effective_tool_override_preserves_explicit_zero():
    assert effective_max_tools_used(80) == 80
    token = set_active_max_tools_used(7)
    try:
        assert effective_max_tools_used(80) == 7
    finally:
        reset_active_max_tools_used(token)
    token = set_active_max_tools_used(0)
    try:
        assert effective_max_tools_used(80) == 0
    finally:
        reset_active_max_tools_used(token)


def test_mcp_cost_warning_reads_parent_shared_state(isolated_run_state, monkeypatch):
    session_id = "shared-live-cost"
    monkeypatch.setenv("CHACK_TASK_SESSION_ID", session_id)
    monkeypatch.setenv("CHACK_BUDGET_INJECTION_ENABLED", "1")
    monkeypatch.setenv("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "0")
    monkeypatch.setenv("CHACK_BUDGET_MAX_COST_USD", "10")
    monkeypatch.setenv("CHACK_BUDGET_SPENT_USD", "0")
    monkeypatch.setenv("CHACK_BUDGET_WARNING_RATIO", "0.6")
    monkeypatch.setenv("CHACK_BUDGET_CRITICAL_RATIO", "0.9")
    write_live_cost(session_id, 7.0)

    result = inject_budget_warning_from_env("tool output")

    assert result.startswith("tool output")
    assert "Cost budget is running low" in result
    assert "$7.0000/$10.0000" in result
    cleanup_run_state(session_id)


def test_exec_background_process_is_cleaned_at_run_end(isolated_run_state, monkeypatch):
    session_id = "process-cleanup"
    monkeypatch.setenv("CHACK_TASK_SESSION_ID", session_id)
    helper = ExecTool(replace(ToolsConfig(), exec_timeout_seconds=5))

    output = helper.run("sleep 30 >/dev/null 2>&1 & echo $!")
    pid = int(output.strip().splitlines()[-1])
    assert os.path.exists(f"/proc/{pid}")

    cleanup_run_state(session_id)
    deadline = time.time() + 2
    while time.time() < deadline and os.path.exists(f"/proc/{pid}"):
        time.sleep(0.05)
    if os.path.exists(f"/proc/{pid}"):
        state = open(f"/proc/{pid}/stat", encoding="utf-8").read().split()[2]
        assert state == "Z"


class _CompletedThenLimitedExecutor:
    def invoke(self, payload, context=None):
        del payload, context
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="PR created")
        STORE.apply(session_id, run_label, "update", task_id=2, status="done", notes="PR verified")
        raise TimeoutError("Agent run exceeded max cost budget ($10.0000).")


class _CompletedWithLiveCostExecutor:
    calls = 0

    def invoke(self, payload, context=None):
        del payload, context
        type(self).calls += 1
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="PR created")
        STORE.apply(session_id, run_label, "update", task_id=2, status="done", notes="PR verified")
        report_live_usage(
            "gpt-5.6-sol",
            prompt_tokens=10_000_000,
            completion_tokens=1_000_000,
        )
        return {
            "output": "ACTUAL_FINAL_ANSWER_FROM_EXECUTOR",
            "intermediate_steps": [],
            "raw_result": None,
        }


class _IncompleteWithLiveCostExecutor:
    def invoke(self, payload, context=None):
        del payload, context
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="PR created")
        report_live_usage(
            "gpt-5.6-sol",
            prompt_tokens=10_000_000,
            completion_tokens=1_000_000,
        )
        return {"output": "must not escape", "intermediate_steps": [], "raw_result": None}


class _TestChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _CompletedThenLimitedExecutor()


class _LiveCostChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _CompletedWithLiveCostExecutor()


class _IncompleteLiveCostChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _IncompleteWithLiveCostExecutor()


def _fallback_config() -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary="test-model", provider="openai"),
        agent=AgentConfig(
            self_critique_enabled=False,
            max_cost_usd=10,
            main_action="test",
            sub_action="completed-limit",
        ),
        session=SessionConfig(long_term_memory_enabled=False),
        tools=replace(ToolsConfig(), min_tools_used=0),
        credentials=CredentialsConfig(),
        logging=LoggingConfig(level="ERROR"),
        system_prompt="test system",
        env={},
    )


def test_completed_task_result_survives_late_cost_timeout(isolated_run_state):
    result = _TestChack(_fallback_config()).run(
        "completed-limit",
        "do work",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )

    assert "requested work completed" in result.output.lower()
    assert "Create artifact — PR created" in result.output
    assert "Verify artifact — PR verified" in result.output
    assert "Finalization limit" in result.output


def _live_cost_config() -> ChackConfig:
    config = _fallback_config()
    return replace(
        config,
        model=replace(config.model, primary="gpt-5.6-sol"),
        agent=replace(
            config.agent,
            max_cost_usd=0.000001,
            self_critique_enabled=True,
            self_critique_rounds=2,
        ),
    )


def test_completed_live_cost_crossing_preserves_actual_executor_output(isolated_run_state):
    _CompletedWithLiveCostExecutor.calls = 0
    result = _LiveCostChack(_live_cost_config()).run(
        "completed-live-cost",
        "do work",
        enable_self_critique=True,
        self_critique_rounds_override=2,
        require_task_steps_manager_init_first=False,
    )

    assert "ACTUAL_FINAL_ANSWER_FROM_EXECUTOR" in result.output
    assert "this final answer was preserved" in result.output
    assert _CompletedWithLiveCostExecutor.calls == 1


def test_incomplete_live_cost_crossing_still_raises(isolated_run_state):
    with pytest.raises(TimeoutError, match="max cost budget"):
        _IncompleteLiveCostChack(_live_cost_config()).run(
            "incomplete-live-cost",
            "do work",
            enable_self_critique=False,
            require_task_steps_manager_init_first=False,
        )
