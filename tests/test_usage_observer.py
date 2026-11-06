from dataclasses import replace

import pytest

from chack_agent.usage_observer import observe

from tests.test_run_lifecycle import (  # noqa: F401  (fixture)
    _IncompleteLiveCostChack,
    _LiveCostChack,
    _fallback_config,
    _live_cost_config,
    isolated_run_state,
)


def _unlimited_config():
    config = _live_cost_config()
    return replace(config, agent=replace(config.agent, max_cost_usd=0.0))


def _run(chack_cls, config, session):
    return chack_cls(config).run(
        session,
        "do work",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )


def test_observer_sees_each_model_call_priced_without_any_budget(isolated_run_state):
    events = []
    with observe(events.append):
        result = _run(_LiveCostChack, _unlimited_config(), "observed")

    assert events
    assert {(e.model, e.prompt_tokens, e.completion_tokens) for e in events} == {
        ("gpt-5.6-sol", 10_000_000, 1_000_000)
    }
    assert all(e.cost_usd is not None and e.cost_usd > 0 for e in events)
    assert result.output == "ACTUAL_FINAL_ANSWER_FROM_EXECUTOR"


def test_usage_of_a_run_that_fails_is_still_observed(isolated_run_state):
    events = []
    with observe(events.append), pytest.raises(TimeoutError, match="max cost budget"):
        _run(_IncompleteLiveCostChack, _live_cost_config(), "failing")

    assert sum(e.prompt_tokens for e in events) == 10_000_000
    assert sum(e.cost_usd for e in events) > 0


def test_a_failing_observer_never_breaks_the_run(isolated_run_state):
    def broken(_event):
        raise RuntimeError("ledger down")

    with observe(broken):
        result = _run(_LiveCostChack, _unlimited_config(), "broken-observer")

    assert result.output == "ACTUAL_FINAL_ANSWER_FROM_EXECUTOR"


def test_runs_outside_observe_report_nothing(isolated_run_state):
    events = []
    with observe(events.append):
        pass
    _run(_LiveCostChack, _unlimited_config(), "unobserved")
    assert events == []
