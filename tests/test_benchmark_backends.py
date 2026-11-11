"""Checks for the reproducible benign backend comparison tasks."""

import csv
import json

from scripts.benchmark_backends import _uses_tool_family, build_tasks


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
