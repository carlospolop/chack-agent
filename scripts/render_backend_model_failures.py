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

TASK_LABEL = {
    "repair_allocator": "Repair",
    "reconcile_ledger": "Ledger",
    "join_report": "Web",
}

BACKEND_GROUPS = (
    ("CLI backends", ("codex", "claude", "gemini", "kimi", "qwen", "opencode")),
    ("API and LangGraph", ("openai", "gateway", "copilot", "langgraph", "openrouter")),
)

BACKEND_LABEL = {
    "codex": "Codex", "claude": "Claude Code", "gemini": "Gemini CLI",
    "kimi": "Kimi Code", "qwen": "Qwen Code", "opencode": "OpenCode",
    "openai": "OpenAI*", "gateway": "Gateway", "copilot": "Copilot*",
    "langgraph": "LangGraph", "openrouter": "OpenRouter",
}

MODEL_LABEL = {
    "bakkt/kimi-k3.1": "Kimi K3.1",
    "bakkt/glm-5.2": "GLM 5.2",
    "bakkt/minimax-m2.5": "MiniMax M2.5",
    "bakkt/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "bakkt/kimi-k2.5": "Kimi K2.5",
    "bakkt/qwen3-coder-480b-a35b-instruct": "Qwen3 Coder 480B",
    "bakkt/gemini-3.5-flash": "Gemini 3.5 Flash",
    "bakkt/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "bakkt/claude-opus-4.8": "Claude Opus 4.8",
    "bakkt/claude-sonnet-5.5": "Claude Sonnet 5.5",
}


def _indexed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        target = row.get("target") or f"{row['backend']}:{row['model']}:gateway"
        indexed[(target, row["task"])] = rescore_official_source(row)
    return indexed


def render_compact(rows: list[dict], *, models: list[str], backends: list[str], tasks: list[str]) -> str:
    """Show only failed task names, with unavailable routes kept distinct."""
    indexed = _indexed(rows)
    unavailable = set()
    for model in models:
        cases = [
            row for (target, _), row in indexed.items()
            if target.split(":")[1] == model and row.get("status") != "unsupported_route"
            and row.get("reason") != "route"
        ]
        if cases and all(not row.get("passed") for row in cases) and any(
            row.get("reason") == "upstream404"
            or "404" in (str(row.get("error") or "") + str(row.get("output") or ""))
            for row in cases
        ):
            unavailable.add(model)

    sections = []
    for title, members in BACKEND_GROUPS:
        selected = [backend for backend in members if backend in backends]
        if not selected:
            continue
        lines = [f"### {title}", ""]
        lines.append("| Model | " + " | ".join(BACKEND_LABEL[name] for name in selected) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in selected) + " |")
        for model in models:
            cells = []
            for backend in selected:
                target = f"{backend}:{model}:gateway"
                cases = [indexed.get((target, task)) for task in tasks]
                if all(row is not None and (
                    row.get("status") == "unsupported_route" or row.get("reason") == "route"
                ) for row in cases):
                    cells.append("N/A")
                elif model in unavailable:
                    cells.append("404")
                else:
                    failed = [
                        TASK_LABEL.get(task, task)
                        for task, row in zip(tasks, cases)
                        if row is None or not row.get("passed")
                    ]
                    cells.append(", ".join(failed) if failed else "✓")
            lines.append("| " + MODEL_LABEL.get(model, model.removeprefix("bakkt/")) + " | " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


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
    indexed = _indexed(rows)
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
    parser.add_argument("--compact", action="store_true", help="split backends into two readable tables")
    args = parser.parse_args()
    source = args.jsonl.read_text(encoding="utf-8")
    rows = json.loads(source) if source.lstrip().startswith("[") else [
        json.loads(line) for line in source.splitlines() if line
    ]
    renderer = render_compact if args.compact else render
    print(renderer(
        rows,
        models=[part.strip() for part in args.models.split(",") if part.strip()],
        backends=[part.strip() for part in args.backends.split(",") if part.strip()],
        tasks=[part.strip() for part in args.tasks.split(",") if part.strip()],
    ))


if __name__ == "__main__":
    main()
