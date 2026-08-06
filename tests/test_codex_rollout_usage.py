"""The rollout tail is what gives a Codex turn a live cost while it runs.

`codex exec --json` reports usage only on `turn.completed`, so without this a
45-minute turn reports nothing until it ends: no cost warning, no enforceable
spend ceiling, and `check_budget_status` answering $0.00 mid-run.
"""

import ast
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "backends" / "codex_backend.py"


class _FakeLiveCostLimitExceeded(TimeoutError):
    pass


def _load_tailer(reports: list, raise_on_report: bool = False):
    """Exec just the tailer out of the backend, which is too heavy to import."""
    module_ast = ast.parse(MODULE_PATH.read_text())
    wanted = {"_RolloutUsageTailer", "_ROLLOUT_USAGE_FIELDS"}
    nodes = [
        node
        for node in module_ast.body
        if (isinstance(node, ast.ClassDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in wanted
                for target in node.targets
            )
        )
    ]
    assert len(nodes) == 2, "the tailer or its field list moved"

    def report_live_usage(model_name, **kwargs):
        if raise_on_report:
            raise _FakeLiveCostLimitExceeded("budget exhausted")
        reports.append(kwargs)

    namespace = {
        "os": os,
        "json": json,
        "glob": glob,
        "time": time,
        "Any": Any,
        "Optional": Optional,
        "report_live_usage": report_live_usage,
        "LiveCostLimitExceeded": _FakeLiveCostLimitExceeded,
    }
    isolated = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(MODULE_PATH), "exec"), namespace)
    return namespace["_RolloutUsageTailer"]


THREAD_ID = "019fd6ab-9045-75a3-b011-35df92fde5b1"


def _token_count_line(*, total_input, cached, output, cache_write=0):
    return json.dumps(
        {
            "timestamp": "2026-08-06T10:00:00.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": total_input,
                        "cached_input_tokens": cached,
                        "cache_write_input_tokens": cache_write,
                        "output_tokens": output,
                        "total_tokens": total_input + output,
                    },
                    "last_token_usage": {},
                },
            },
        }
    )


@pytest.fixture
def rollout(tmp_path):
    path = tmp_path / "sessions" / "2026" / "08" / "06" / f"rollout-2026-08-06T10-00-00-{THREAD_ID}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("")

    def append(line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    append.path = path  # type: ignore[attr-defined]
    append.home = tmp_path  # type: ignore[attr-defined]
    return append


def test_each_completed_round_is_reported_as_it_lands(rollout):
    reports: list = []
    tailer = _load_tailer(reports)(str(rollout.home), "gpt-5")
    tailer.attach(THREAD_ID)

    rollout(_token_count_line(total_input=17061, cached=0, output=136))
    tailer.poll()
    rollout(_token_count_line(total_input=36018, cached=16128, output=251))
    tailer.poll()

    assert reports == [
        {
            "prompt_tokens": 17061,
            "completion_tokens": 136,
            "cached_prompt_tokens": 0,
            "cache_write_tokens": 0,
        },
        # Cumulative totals, so only the growth is new spend.
        {
            "prompt_tokens": 18957,
            "completion_tokens": 115,
            "cached_prompt_tokens": 16128,
            "cache_write_tokens": 0,
        },
    ]


def test_usage_already_on_disk_is_a_baseline_not_new_spend(rollout):
    # A resumed thread appends to the rollout its earlier invocations wrote,
    # and those tokens are already counted in the run's cost.
    rollout(_token_count_line(total_input=50_000, cached=40_000, output=900))
    reports: list = []
    tailer = _load_tailer(reports)(str(rollout.home), "gpt-5")
    tailer.attach(THREAD_ID)

    tailer.poll()
    assert reports == []

    rollout(_token_count_line(total_input=61_000, cached=48_000, output=1_100))
    tailer.poll()
    assert reports == [
        {
            "prompt_tokens": 11_000,
            "completion_tokens": 200,
            "cached_prompt_tokens": 8_000,
            "cache_write_tokens": 0,
        }
    ]


def test_settling_the_turn_never_double_counts_what_the_tail_reported(rollout):
    reports: list = []
    tailer = _load_tailer(reports)(str(rollout.home), "gpt-5")
    tailer.attach(THREAD_ID)
    rollout(_token_count_line(total_input=30_000, cached=20_000, output=400))
    tailer.poll()

    remaining = tailer.settle(
        {
            "input_tokens": 32_000,
            "output_tokens": 450,
            "input_tokens_details": {"cached_tokens": 21_000, "cache_write_tokens": 0},
        }
    )

    assert remaining == {
        "input_tokens": 2_000,
        "cached_input_tokens": 1_000,
        "cache_write_input_tokens": 0,
        "output_tokens": 50,
    }
    # A rollout line written after the turn closed must not be reported again.
    rollout(_token_count_line(total_input=32_000, cached=21_000, output=450))
    tailer.poll()
    assert len(reports) == 1


def test_a_turn_the_tail_never_saw_settles_to_its_full_usage(rollout):
    reports: list = []
    tailer = _load_tailer(reports)(str(rollout.home), "gpt-5")
    tailer.attach(THREAD_ID)

    assert tailer.settle(
        {
            "input_tokens": 12_000,
            "output_tokens": 300,
            "input_tokens_details": {"cached_tokens": 9_000, "cache_write_tokens": 7},
        }
    ) == {
        "input_tokens": 12_000,
        "cached_input_tokens": 9_000,
        "cache_write_input_tokens": 7,
        "output_tokens": 300,
    }


def test_a_half_written_line_is_completed_rather_than_lost(rollout):
    reports: list = []
    tailer = _load_tailer(reports)(str(rollout.home), "gpt-5")
    tailer.attach(THREAD_ID)

    line = _token_count_line(total_input=9_000, cached=0, output=80)
    with rollout.path.open("a", encoding="utf-8") as handle:
        handle.write(line[:40])
    tailer.poll()
    assert reports == []

    with rollout.path.open("a", encoding="utf-8") as handle:
        handle.write(line[40:] + "\n")
    tailer.poll()
    assert reports == [
        {
            "prompt_tokens": 9_000,
            "completion_tokens": 80,
            "cached_prompt_tokens": 0,
            "cache_write_tokens": 0,
        }
    ]


def test_unrelated_and_malformed_rollout_lines_are_ignored(rollout):
    reports: list = []
    tailer = _load_tailer(reports)(str(rollout.home), "gpt-5")
    tailer.attach(THREAD_ID)

    rollout(json.dumps({"type": "response_item", "payload": {"type": "message"}}))
    rollout('{"type": "event_msg", "payload": {"type": "token_count"')
    rollout("")
    tailer.poll()

    assert reports == []


def test_a_missing_rollout_leaves_the_run_exactly_as_it_was(tmp_path):
    reports: list = []
    tailer = _load_tailer(reports)(str(tmp_path), "gpt-5")
    tailer.attach("no-such-thread")

    tailer.poll()

    assert reports == []
    assert tailer.settle(
        {"input_tokens": 5, "output_tokens": 1, "input_tokens_details": {}}
    ) == {
        "input_tokens": 5,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 1,
    }


def test_no_codex_home_means_the_tail_never_engages(rollout):
    reports: list = []
    tailer = _load_tailer(reports)("", "gpt-5")
    tailer.attach(THREAD_ID)
    rollout(_token_count_line(total_input=1_000, cached=0, output=10))

    tailer.poll()

    assert reports == []


def test_the_spend_ceiling_firing_mid_turn_reaches_the_caller(rollout):
    reports: list = []
    tailer = _load_tailer(reports, raise_on_report=True)(str(rollout.home), "gpt-5")
    tailer.attach(THREAD_ID)
    rollout(_token_count_line(total_input=99_000, cached=0, output=900))

    # Anything else is swallowed so a bad rollout cannot fail a run, but this
    # one must propagate: the caller has to stop the Codex process.
    with pytest.raises(_FakeLiveCostLimitExceeded):
        tailer.poll()
