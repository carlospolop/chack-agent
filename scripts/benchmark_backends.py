"""Run the same small, verifiable tasks on several backends and compare them.

Each target is ``provider:model`` with an optional ``:api_key_type`` (for
example ``codex:gpt-6-sol:gateway`` or ``langgraph:gpt-6-sol:gateway``).
Every task has one checkable answer, so the table shows correctness next to
time, tokens, cache use and cost:

    python scripts/benchmark_backends.py codex:gpt-6-sol:gateway claude:claude-sonnet-5-5:gateway

Credentials come from the usual environment variables (``LLM_GATEWAY_*``,
``OPENROUTER_API_KEY``, CLI logins...).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chack_agent import Chack  # noqa: E402
from chack_agent.config import (  # noqa: E402
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
)
from chack_tools.config import ToolsConfig  # noqa: E402


@dataclass
class Task:
    name: str
    prompt: str
    check: Callable[[str], bool]
    needs_shell: bool = False
    output_schema: Optional[dict[str, Any]] = None


def _exact(expected: str) -> Callable[[str], bool]:
    return lambda output: output.strip().strip("`").strip() == expected


def _json_field(field: str, expected: str) -> Callable[[str], bool]:
    def _check(output: str) -> bool:
        try:
            return str(json.loads(output).get(field, "")).strip().lower() == expected.lower()
        except (ValueError, AttributeError):
            return False

    return _check


def build_tasks(workdir: Path) -> list[Task]:
    lines = [f"row {i}: {i * 7919 % 100003:05d}" for i in range(1, 201)]
    data_file = workdir / "rows.txt"
    data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [
        Task("answer", "What is 17 * 23? Reply with only the number.", _exact("391")),
        Task(
            "shell",
            "Run `printf 'chack-%s' $((6*7))` with your shell tool and reply with only its output.",
            _exact("chack-42"),
            needs_shell=True,
        ),
        Task(
            "file",
            f"Read line 137 of {data_file} and reply with only that line's text.",
            _exact(lines[136]),
            needs_shell=True,
        ),
        Task(
            "structured",
            "What is the capital city of Australia?",
            _json_field("capital", "Canberra"),
            output_schema={
                "type": "object",
                "properties": {"capital": {"type": "string"}},
                "required": ["capital"],
                "additionalProperties": False,
            },
        ),
    ]


def _config(provider: str, model: str, api_key_type: str, task: Task, max_turns: int) -> ChackConfig:
    tools = ToolsConfig(task_steps_manager_enabled=False, exec_enabled=task.needs_shell)
    agent = AgentConfig(
        thinking_effort="low",
        main_action="benchmark",
        sub_action=task.name,
        output_schema_json=task.output_schema,
        output_schema_name="answer" if task.output_schema else "",
    )
    return ChackConfig(
        model=ModelConfig(primary=model, provider=provider, api_key_type=api_key_type),
        agent=agent,
        session=SessionConfig(max_turns=max_turns, long_term_memory_enabled=False),
        tools=tools,
        credentials=CredentialsConfig(),
        logging=LoggingConfig(level="WARNING"),
        system_prompt="You are a precise assistant. Follow the requested output format exactly.",
        env={},
    )


def run_one(target: str, task: Task, max_turns: int) -> dict[str, Any]:
    provider, model, *rest = target.split(":")
    api_key_type = rest[0] if rest else ""
    row: dict[str, Any] = {"target": target, "task": task.name}
    started = time.monotonic()
    try:
        result = Chack(_config(provider, model, api_key_type, task, max_turns)).run(
            f"benchmark-{provider}-{task.name}",
            task.prompt,
            enable_self_critique=False,
            require_task_steps_manager_init_first=False,
        )
    except Exception as exc:  # noqa: BLE001 - report every failure in the table
        row.update(passed=False, error=f"{type(exc).__name__}: {exc}"[:200])
        row["seconds"] = round(time.monotonic() - started, 1)
        return row
    output = str(result.output or "")
    row.update(
        passed=bool(task.check(output)) and not result.error,
        seconds=round(time.monotonic() - started, 1),
        prompt_tokens=result.prompt_tokens,
        cached_prompt_tokens=result.cached_prompt_tokens,
        completion_tokens=result.completion_tokens,
        cache_hit_rate=result.cache_hit_rate,
        cost_usd=result.total_cost,
        tool_calls=sum(result.tool_counts.values()),
        error=result.error,
        output=output[:120],
    )
    return row


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if value < 1 else f"{value:.1f}"
    return str(value)


def render(rows: list[dict[str, Any]]) -> str:
    columns = [
        "target", "task", "passed", "seconds", "prompt_tokens", "cached_prompt_tokens",
        "completion_tokens", "cache_hit_rate", "cost_usd", "tool_calls", "error",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(col)) for col in columns) + " |")
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(row)
    lines.append("")
    for target, target_rows in by_target.items():
        passed = sum(1 for row in target_rows if row.get("passed"))
        cost = sum(row.get("cost_usd") or 0.0 for row in target_rows)
        seconds = sum(row.get("seconds") or 0.0 for row in target_rows)
        lines.append(f"- {target}: {passed}/{len(target_rows)} passed, ${cost:.4f}, {seconds:.0f}s")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", help="provider:model[:api_key_type]")
    parser.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--json", dest="json_path", default="", help="also write the rows to this file")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="chack-benchmark-") as tmp:
        tasks = build_tasks(Path(tmp))
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        if wanted:
            tasks = [task for task in tasks if task.name in wanted]
        rows = []
        for target in args.targets:
            for task in tasks:
                row = run_one(target, task, args.max_turns)
                print(json.dumps(row, default=str), file=sys.stderr, flush=True)
                rows.append(row)

    print(render(rows))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0 if all(row.get("passed") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
