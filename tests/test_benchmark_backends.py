"""Checks for the reproducible benign backend comparison tasks."""

import csv
import json
import signal
import time
from pathlib import Path

import pytest

from scripts.benchmark_backends import _uses_tool_family, build_tasks
from scripts import benchmark_model_backend_matrix
from scripts.benchmark_model_backend_matrix import DEFAULT_BACKENDS
from scripts.render_backend_model_failures import render as render_failures, rescore_official_source
from chack_agent.runtime_capabilities import get_supported_backends


def test_combined_report_has_checkable_file_and_web_requirements(tmp_path):
    task = next(task for task in build_tasks(tmp_path) if task.name == "combined_report")
    with (tmp_path / "orders.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    active = [row for row in rows if row["active"] == "yes"]
    assert sum(int(row["score"]) for row in active) == 55
    assert len({row["user"] for row in active}) == 3
    assert task.required_tool_families == ("local", "web")
    assert "exec" not in task.prompt.lower() and "web_search" not in task.prompt.lower()
    answer = {
        "active_total": 55, "unique_active_users": 3,
        "dict_order_guaranteed_since": "3.7",
        "source_url": "https://docs.python.org/3/library/stdtypes.html",
    }
    assert task.check(json.dumps(answer) + "\nBudget warning: 5/8 tools used")
    assert task.check(json.dumps({**answer, "source_url": "https://docs.python.org/3/reference/datamodel.html"}))
    assert task.check(json.dumps({**answer, "dict_order_guaranteed_since": "Python 3.7"}))
    assert not task.check(json.dumps({**answer, "active_total": 54}))
    assert _uses_tool_family(["Read", "WebFetch"], "local")
    assert _uses_tool_family(["Read", "WebFetch"], "web")
    assert not _uses_tool_family(["WebFetch"], "local")


def test_join_report_requires_two_files_and_correct_tie_break(tmp_path):
    task = next(task for task in build_tasks(tmp_path) if task.name == "join_report")
    with (tmp_path / "orders.csv").open(newline="", encoding="utf-8") as stream:
        active = [row for row in csv.DictReader(stream) if row["active"] == "yes"]
    with (tmp_path / "adjustments.csv").open(newline="", encoding="utf-8") as stream:
        adjustments = {row["user"]: int(row["delta"]) for row in csv.DictReader(stream)}
    totals = {}
    for row in active:
        totals[row["user"]] = totals.get(row["user"], 0) + int(row["score"])
    adjusted = {user: total + adjustments[user] for user, total in totals.items()}
    assert adjusted == {"iris": 21, "maple": 21, "cedar": 16}
    assert task.required_tool_families == ("local", "web")
    assert "exec" not in task.prompt.lower() and "web_search" not in task.prompt.lower()
    answer = {
        "winner": "iris", "adjusted_score": 21, "tied_with": ["maple"],
        "dict_order_guaranteed_since": "Python 3.7",
        "source_url": "https://docs.python.org/3/whatsnew/3.7.html",
    }
    assert task.check(json.dumps(answer))
    assert not task.check(json.dumps({**answer, "winner": "maple"}))


def test_reconcile_ledger_requires_three_files_and_excludes_void_and_unknown_payments(tmp_path):
    task = next(task for task in build_tasks(tmp_path) if task.name == "reconcile_ledger")
    assert task.required_tool_families == ("local",)
    assert task.max_tool_calls == 12
    assert "exec" not in task.prompt.lower()
    expected = {
        "customer": "bob", "outstanding_cents": 1000,
        "open_invoice_count": 3, "ignored_payment_cents": 1299,
    }
    assert task.check(json.dumps(expected))
    assert not task.check(json.dumps({**expected, "ignored_payment_cents": 999}))
    assert not task.check(json.dumps({**expected, "open_invoice_count": 2}))


def test_repair_allocator_scores_real_module_and_hidden_cases(tmp_path):
    task = next(task for task in build_tasks(tmp_path) if task.name == "repair_allocator")
    assert task.required_tool_families == ("exec",)
    assert task.max_tool_calls == 12
    assert not task.check("Fixed and tested")
    module = tmp_path / "allocator.py"
    module.write_text(
        "def allocate_cents(amount_cents, weights):\n"
        "    if amount_cents < 0 or not weights or any(w < 0 for w in weights) or not sum(weights):\n"
        "        raise ValueError('invalid')\n"
        "    total = sum(weights)\n"
        "    floors = [amount_cents * w // total for w in weights]\n"
        "    order = sorted(range(len(weights)), key=lambda i: (-(amount_cents * weights[i] % total), i))\n"
        "    for i in order[:amount_cents - sum(floors)]:\n"
        "        floors[i] += 1\n"
        "    return floors\n", encoding="utf-8",
    )
    assert task.check("Fixed and tested")


def test_matrix_includes_every_supported_backend():
    assert set(DEFAULT_BACKENDS) == get_supported_backends()


def test_failure_matrix_distinguishes_failed_unrun_and_unsupported_cases():
    rows = [
        {"target": "gateway:bakkt/example:gateway", "task": "repair_allocator", "passed": True},
        {"target": "gateway:bakkt/example:gateway", "task": "reconcile_ledger", "passed": False,
         "answer_ok": False, "tools_ok": False, "budget_ok": True},
        {"target": "openrouter:bakkt/example:gateway", "task": "repair_allocator", "passed": False,
         "status": "unsupported_route"},
    ]
    table = render_failures(
        rows, models=["bakkt/example"], backends=["gateway", "openrouter"],
        tasks=["repair_allocator", "reconcile_ledger", "join_report"],
    )
    assert "| example | L:answer+tools, J:unrun | R:route, L:unrun, J:unrun |" in table


@pytest.mark.skipif(not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"), reason="POSIX alarm required")
def test_matrix_case_has_a_wall_clock_limit(monkeypatch):
    def stalled(*_args):
        time.sleep(2)

    monkeypatch.setattr(benchmark_model_backend_matrix, "run_one", stalled)
    with pytest.raises(TimeoutError, match="Case timed out"):
        benchmark_model_backend_matrix._run_case("gateway", "bakkt/example", "repair_allocator", 12, 1)


def test_matrix_case_isolates_cli_working_directory(monkeypatch):
    before = Path.cwd()
    seen = []

    def fake_run(*_args):
        seen.append(Path.cwd())
        Path("scratch.txt").write_text("temporary", encoding="utf-8")
        return {"passed": True}

    monkeypatch.setattr(benchmark_model_backend_matrix, "run_one", fake_run)
    assert benchmark_model_backend_matrix._run_case("gateway", "bakkt/example", "repair_allocator", 12, 3)["passed"]
    assert Path.cwd() == before
    assert not seen[0].exists()
    assert not (before / "scratch.txt").exists()


def test_saved_join_result_can_be_rescored_for_official_data_model_source():
    row = {
        "task": "join_report", "passed": False, "answer_ok": False,
        "tools_ok": True, "budget_ok": True, "error": "",
        "output": json.dumps({
            "winner": "iris", "adjusted_score": 21, "tied_with": ["maple"],
            "dict_order_guaranteed_since": "3.7",
            "source_url": "https://docs.python.org/3/reference/datamodel.html",
        }),
    }
    corrected = rescore_official_source(row)
    assert corrected["passed"] and corrected["answer_ok"]
    assert corrected["rescored_source"]
    assert not row["passed"]
    already_passed = {**row, "passed": True, "answer_ok": True, "output": "truncated after 500 chars"}
    assert rescore_official_source(already_passed) == already_passed
