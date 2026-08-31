from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from chack_agent import Chack
import chack_agent.agent as agent_module
from chack_agent.agent import _task_snapshot_is_complete
from chack_agent.budget_warning_state import inject_budget_warning_from_env
from chack_agent.live_cost_state import report_live_usage
from chack_agent.limit_event_state import emit_limit_reached
from chack_agent.resume_compaction import ResumeCompactionResult
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
from chack_tools.cancellation import current_cancellation_event
from chack_tools.run_lifecycle import (
    claim_non_task_tool_slot,
    cleanup_run_state,
    mark_task_manager_initialized,
    read_mcp_tool_usage,
    record_mcp_tool_usage,
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
    monkeypatch.delenv("CHACK_TASK_SESSION_ID", raising=False)
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


def test_mcp_tool_usage_survives_parallel_processes_and_cleanup(
    isolated_run_state,
):
    session_id = "parallel-mcp-tool-usage"
    code = (
        "from chack_tools.run_lifecycle import record_mcp_tool_usage; "
        f"record_mcp_tool_usage('read_file', {session_id!r})"
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
    for process in processes:
        _, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr

    record_mcp_tool_usage("search_code", session_id)
    assert read_mcp_tool_usage(session_id) == {
        "read_file": 12,
        "search_code": 1,
    }
    cleanup_run_state(session_id)
    assert read_mcp_tool_usage(session_id) == {}


def test_corrupt_mcp_telemetry_recovers_without_blocking_tools(
    isolated_run_state,
):
    session_id = "corrupt-mcp-telemetry"
    record_mcp_tool_usage("read_file", session_id)
    state_path = next(isolated_run_state.glob("*.tool-usage.json"))
    state_path.write_text("{not-json", encoding="utf-8")

    record_mcp_tool_usage("search_code", session_id)

    assert read_mcp_tool_usage(session_id) == {"search_code": 1}
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
    if not os.path.isdir("/proc"):
        pytest.skip("background process lifecycle assertion requires Linux /proc")
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
    limit_error: Exception = TimeoutError("Agent run exceeded max cost budget ($10.0000).")

    def invoke(self, payload, context=None):
        del payload, context
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="PR created")
        STORE.apply(session_id, run_label, "update", task_id=2, status="done", notes="PR verified")
        raise self.limit_error


class _CompletedThenRuntimeLimitedExecutor(_CompletedThenLimitedExecutor):
    limit_error = TimeoutError("Agent run exceeded max runtime (60 minutes).")


class _CompletedThenToolLimitedExecutor(_CompletedThenLimitedExecutor):
    limit_error = RuntimeError("Agent tool-call limit reached; finalize with completed work.")


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


class _DeadlineCrossingResult(dict):
    def __init__(self, clock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clock = clock

    def get(self, key, default=None):
        if key == "error":
            self._clock.now = 1061.0
        return super().get(key, default)


class _WarningThresholdResult(dict):
    def __init__(self, clock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clock = clock

    def get(self, key, default=None):
        if key == "error":
            self._clock.now = 1040.0
        return super().get(key, default)


class _RuntimeWarningExecutor:
    def __init__(self, clock):
        self.clock = clock

    def invoke(self, payload, context=None):
        del payload, context
        return _WarningThresholdResult(
            self.clock,
            output="CLEAN_USER_FACING_ANSWER",
            intermediate_steps=[],
            raw_result=None,
        )


class _RuntimeCrossingExecutor:
    def __init__(self, clock, *, complete: bool = True):
        self.clock = clock
        self.complete = complete
        self.calls = 0

    def invoke(self, payload, context=None):
        del payload, context
        self.calls += 1
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="created")
        if self.complete:
            STORE.apply(session_id, run_label, "update", task_id=2, status="done", notes="verified")
        return _DeadlineCrossingResult(
            self.clock,
            output="ACTUAL_RUNTIME_FINAL_ANSWER",
            intermediate_steps=[],
            raw_result=None,
        )


class _ToolCapExecutor:
    def __init__(self, *, complete: bool = True):
        self.complete = complete
        self.calls = 0

    def invoke(self, payload, context=None):
        del payload, context
        self.calls += 1
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="created")
        if self.complete:
            STORE.apply(session_id, run_label, "update", task_id=2, status="done", notes="verified")
        return {
            "output": "ACTUAL_TOOL_CAP_FINAL_ANSWER",
            "intermediate_steps": [
                {"tool": "web_search", "output": "one"},
                {"tool": "web_extract", "output": "two"},
            ],
            "raw_result": None,
        }


class _RuntimeWatchdogRaceExecutor:
    def __init__(self, clock):
        self.clock = clock
        self.calls = 0

    def invoke(self, payload, context=None):
        del payload, context
        self.calls += 1
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="created")
        STORE.apply(session_id, run_label, "update", task_id=2, status="done", notes="verified")
        cancel_event = current_cancellation_event()
        assert cancel_event is not None
        self.clock.now = 1061.0
        assert cancel_event.wait(timeout=1.0)
        return {
            "output": "QUEUED_RUNTIME_FINAL_ANSWER",
            "intermediate_steps": [],
            "raw_result": None,
        }


class _ToolLimitEventExecutor:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload, context=None):
        del payload, context
        self.calls += 1
        session_id = current_session_id()
        run_label = current_run_label()
        assert session_id
        STORE.apply(session_id, run_label, "init", tasks_text="Create artifact\nVerify artifact")
        STORE.apply(session_id, run_label, "update", task_id=1, status="done", notes="created")
        STORE.apply(session_id, run_label, "update", task_id=2, status="done", notes="verified")
        emit_limit_reached("tools", {"used": 2, "max_tools_used": 2})
        return {
            "output": "TOOL_EVENT_FINAL_ANSWER",
            "intermediate_steps": [],
            "raw_result": None,
        }


class _ObservableExecutor:
    def invoke(self, payload, context=None):
        del payload, context
        return {
            "output": "observable result",
            "intermediate_steps": [],
            "raw_result": SimpleNamespace(
                raw_responses=[],
                time_to_first_token_seconds=0.42,
            ),
        }


class _ExplicitResumeCompactionExecutor(_ObservableExecutor):
    def __init__(self):
        self.compaction_calls = []

    def compact_for_resume(self, focus_instructions):
        self.compaction_calls.append(focus_instructions)
        return ResumeCompactionResult(
            backend="test",
            method="test_compactor",
            attempted=True,
            succeeded=True,
            duration_seconds=1.25,
            raw_responses=[
                {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "input_tokens_details": {
                            "cached_tokens": 40,
                            "cache_write_tokens": 0,
                        },
                    }
                }
            ],
        )


class _TestChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _CompletedThenLimitedExecutor()


class _RuntimeLimitChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _CompletedThenRuntimeLimitedExecutor()


class _ToolLimitChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _CompletedThenToolLimitedExecutor()


class _LiveCostChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _CompletedWithLiveCostExecutor()


class _IncompleteLiveCostChack(Chack):
    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return _IncompleteWithLiveCostExecutor()


class _InjectedExecutorChack(Chack):
    executor: object | None = None

    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return self.executor


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


def test_run_result_exposes_prompt_characters_and_first_token_latency(
    isolated_run_state,
):
    agent = _InjectedExecutorChack(_fallback_config())
    agent.executor = _ObservableExecutor()

    result = agent.run(
        "observable-run",
        "do observable work",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )

    assert result.time_to_first_token_seconds == 0.42
    assert result.time_to_first_token_source == "backend_first_response_event"
    assert result.initial_prompt_chars > len("do observable work")


def test_resume_compaction_is_opt_in_and_its_usage_is_counted(
    isolated_run_state,
):
    agent = _InjectedExecutorChack(_fallback_config())
    executor = _ExplicitResumeCompactionExecutor()
    agent.executor = executor

    normal = agent.run(
        "observable-run",
        "first turn",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )
    compacted = agent.run(
        "observable-run",
        "next turn",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
        compact_before_resume=True,
        resume_compaction_instructions="Preserve the current checks.",
    )

    assert normal.resume_compaction_attempted is False
    assert executor.compaction_calls == ["Preserve the current checks."]
    assert compacted.resume_compaction_attempted is True
    assert compacted.resume_compaction_succeeded is True
    assert compacted.resume_compaction_method == "test_compactor"
    assert compacted.resume_compaction_duration_seconds == 1.25
    assert compacted.prompt_tokens == 100
    assert compacted.completion_tokens == 20
    assert compacted.cached_prompt_tokens == 40


@pytest.mark.parametrize("agent_type", [_RuntimeLimitChack, _ToolLimitChack])
def test_completed_task_result_survives_other_late_limits(
    isolated_run_state,
    agent_type,
):
    result = agent_type(_fallback_config()).run(
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

    assert result.output == "ACTUAL_FINAL_ANSWER_FROM_EXECUTOR"
    assert result.limit_reached == "cost_after_completion"
    assert result.completion_preserved_after_limit is True
    assert _CompletedWithLiveCostExecutor.calls == 1


def test_incomplete_live_cost_crossing_still_raises(isolated_run_state):
    with pytest.raises(TimeoutError, match="max cost budget"):
        _IncompleteLiveCostChack(_live_cost_config()).run(
            "incomplete-live-cost",
            "do work",
            enable_self_critique=False,
            require_task_steps_manager_init_first=False,
        )


def test_task_snapshot_completion_requires_all_tasks_done():
    assert not _task_snapshot_is_complete({"completed": True, "tasks_total": 2, "tasks_done": 1})
    assert not _task_snapshot_is_complete({"completed": True, "tasks_total": 0, "tasks_done": 0})
    assert _task_snapshot_is_complete({"completed": True, "tasks_total": 2, "tasks_done": 2})


def test_runtime_warning_is_internal_and_does_not_modify_user_output(
    isolated_run_state,
    monkeypatch,
):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(agent_module, "time", SimpleNamespace(time=lambda: clock.now))
    config = _fallback_config()
    config = replace(
        config,
        agent=replace(config.agent, max_runtime_minutes=1, max_cost_usd=0),
    )
    agent = _InjectedExecutorChack(config)
    agent.executor = _RuntimeWarningExecutor(clock)

    result = agent.run(
        "runtime-warning",
        "do work",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )

    assert result.output == "CLEAN_USER_FACING_ANSWER"


@pytest.mark.parametrize("complete", [True, False])
def test_runtime_crossing_after_result_is_strict_and_skips_critique(
    isolated_run_state,
    monkeypatch,
    complete,
):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(agent_module, "time", SimpleNamespace(time=lambda: clock.now))
    config = _fallback_config()
    config = replace(
        config,
        agent=replace(
            config.agent,
            max_runtime_minutes=1,
            max_cost_usd=0,
            self_critique_enabled=True,
            self_critique_rounds=2,
        ),
    )
    executor = _RuntimeCrossingExecutor(clock, complete=complete)
    agent = _InjectedExecutorChack(config)
    agent.executor = executor

    if not complete:
        with pytest.raises(TimeoutError, match="max runtime"):
            agent.run(
                "runtime-crossing-incomplete",
                "do work",
                self_critique_rounds_override=2,
                require_task_steps_manager_init_first=False,
            )
    else:
        result = agent.run(
            "runtime-crossing-complete",
            "do work",
            self_critique_rounds_override=2,
            require_task_steps_manager_init_first=False,
        )
        assert result.output == "ACTUAL_RUNTIME_FINAL_ANSWER"
        assert result.limit_reached == "runtime_after_completion"
        assert result.completion_preserved_after_limit is True
    assert executor.calls == 1


@pytest.mark.parametrize("complete", [True, False])
def test_exact_tool_cap_is_terminal_and_skips_critique(isolated_run_state, complete):
    config = _fallback_config()
    config = replace(
        config,
        agent=replace(
            config.agent,
            max_cost_usd=0,
            self_critique_enabled=True,
            self_critique_rounds=2,
        ),
        tools=replace(config.tools, min_tools_used=0, max_tools_used=2),
    )
    executor = _ToolCapExecutor(complete=complete)
    agent = _InjectedExecutorChack(config)
    agent.executor = executor

    result = agent.run(
        f"tool-cap-{'complete' if complete else 'incomplete'}",
        "do work",
        self_critique_rounds_override=2,
        require_task_steps_manager_init_first=False,
    )

    assert result.output == "ACTUAL_TOOL_CAP_FINAL_ANSWER"
    assert result.limit_reached == ("tools_after_completion" if complete else "tools")
    assert result.completion_preserved_after_limit is complete
    assert executor.calls == 1


def test_runtime_watchdog_preserves_success_queued_during_cancellation(
    isolated_run_state,
    monkeypatch,
):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(agent_module, "time", SimpleNamespace(time=lambda: clock.now))
    config = _fallback_config()
    config = replace(
        config,
        agent=replace(config.agent, max_runtime_minutes=1, max_cost_usd=0),
    )
    executor = _RuntimeWatchdogRaceExecutor(clock)
    agent = _InjectedExecutorChack(config)
    agent.executor = executor
    monkeypatch.setattr(agent, "_stop_thread", lambda _worker: None)

    result = agent.run(
        "runtime-watchdog-race",
        "do work",
        require_task_steps_manager_init_first=False,
    )

    assert result.output == "QUEUED_RUNTIME_FINAL_ANSWER"
    assert result.limit_reached == "runtime_after_completion"
    assert result.completion_preserved_after_limit is True
    assert executor.calls == 1


def test_tool_limit_event_is_terminal_when_denied_step_is_omitted(isolated_run_state):
    config = _fallback_config()
    config = replace(
        config,
        agent=replace(
            config.agent,
            max_runtime_minutes=1,
            max_cost_usd=0,
            self_critique_enabled=True,
            self_critique_rounds=2,
        ),
        tools=replace(config.tools, min_tools_used=0, max_tools_used=2),
    )
    executor = _ToolLimitEventExecutor()
    agent = _InjectedExecutorChack(config)
    agent.executor = executor

    result = agent.run(
        "tool-limit-event",
        "do work",
        self_critique_rounds_override=2,
        require_task_steps_manager_init_first=False,
    )

    assert result.output == "TOOL_EVENT_FINAL_ANSWER"
    assert result.limit_reached == "tools_after_completion"
    assert result.completion_preserved_after_limit is True
    assert executor.calls == 1
