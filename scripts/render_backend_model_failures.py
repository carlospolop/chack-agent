"""Render a model-by-backend table of failed benchmark tasks from JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_model_backend_matrix import DEFAULT_BACKENDS, DEFAULT_TASKS
from scripts.benchmark_backends import _join_report

TASK_CODE = {
    "repair_allocator": "R",
    "reconcile_ledger": "L",
    "join_report": "J",
}


def _reason(row: dict) -> str:
    if row.get("status") == "unsupported_route":
        return "route"
    error = str(row.get("error") or "").lower()
    details = error + " " + str(row.get("output") or "").lower()
    if "timed out" in details or "timeout" in details:
        return "timeout"
    if error or str(row.get("output") or "").lstrip().startswith("ERROR:"):
        return "error"
    failures = []
    if not row.get("answer_ok", False):
        failures.append("answer")
    if not row.get("tools_ok", False):
        failures.append("tools")
    if not row.get("budget_ok", False):
        failures.append("budget")
    return "+".join(failures) or "unknown"


def render(rows: list[dict], *, models: list[str], backends: list[str], tasks: list[str]) -> str:
    indexed = {(row["target"], row["task"]): rescore_official_source(row) for row in rows}
    header = "| Model | " + " | ".join(backends) + " |"
    lines = [header, "| --- | " + " | ".join("---" for _ in backends) + " |"]
    for model in models:
        cells = []
        for backend in backends:
            target = f"{backend}:{model}:gateway"
            failures = []
            for task in tasks:
                row = indexed.get((target, task))
                code = TASK_CODE.get(task, task)
                if row is None:
                    failures.append(f"{code}:unrun")
                elif not row.get("passed", False):
                    failures.append(f"{code}:{_reason(row)}")
            cells.append(", ".join(failures) or "—")
        lines.append("| " + model.removeprefix("bakkt/") + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rescore_official_source(row: dict) -> dict:
    """Apply the expanded official Python source allowlist to saved J answers."""
    if row.get("task") != "join_report" or row.get("answer_ok") or row.get("error") or not row.get("output"):
        return row
    answer_ok = _join_report(str(row["output"]))
    if not answer_ok:
        return row
    updated = dict(row, answer_ok=answer_ok, rescored_source=True)
    updated["passed"] = answer_ok and bool(row.get("tools_ok")) and bool(row.get("budget_ok"))
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--models", required=True)
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line]
    print(render(
        rows,
        models=[part.strip() for part in args.models.split(",") if part.strip()],
        backends=[part.strip() for part in args.backends.split(",") if part.strip()],
        tasks=[part.strip() for part in args.tasks.split(",") if part.strip()],
    ))


if __name__ == "__main__":
    main()
