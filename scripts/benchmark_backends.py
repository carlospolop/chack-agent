"""Run the same small, verifiable tasks on several backends and compare them.

Each target is ``provider:model`` with an optional ``:api_key_type`` (for
example ``kimi:bakkt/glm-5.2:gateway`` or ``qwen:bakkt/kimi-k3.1:gateway``).
Every task has one checkable answer, so the table shows correctness next to
time, tokens, cache use and cost:

    python scripts/benchmark_backends.py kimi:bakkt/glm-5.2:gateway qwen:bakkt/kimi-k3.1:gateway

Credentials come from the usual environment variables (``LLM_GATEWAY_*``,
``OPENROUTER_API_KEY``, CLI logins...).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    ToolsConfig,
)

_MAX_TOOL_CALLS = 8


@dataclass
class Task:
    name: str
    prompt: str
    check: Callable[[str], bool]
    needs_shell: bool = False
    needs_web: bool = False
    required_tool_families: tuple[str, ...] = ()
    output_schema: Optional[dict[str, Any]] = None


def _exact(expected: str) -> Callable[[str], bool]:
    return lambda output: output.strip().strip("`").strip() == expected


def _json_field(field: str, expected: str) -> Callable[[str], bool]:
    def _check(output: str) -> bool:
        try:
            return str(_parse_json_object(output).get(field, "")).strip().lower() == expected.lower()
        except (ValueError, AttributeError):
            return False

    return _check


def _parse_json_object(output: str) -> dict[str, Any]:
    # The run wrapper may append a budget notice after a valid final JSON
    # object. Score the object, while leaving the notice visible in the row.
    start = output.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    value, _ = json.JSONDecoder().raw_decode(output[start:])
    if not isinstance(value, dict):
        raise ValueError("JSON output was not an object")
    return value


_DICT_ORDER_SOURCE_PREFIXES = (
    "https://docs.python.org/3/whatsnew/3.7.html",
    "https://docs.python.org/3/library/stdtypes.html",
    "https://docs.python.org/3/builtins/stdtypes.html",
)


def _has_dict_order_source(text: str) -> bool:
    return any(prefix in text for prefix in _DICT_ORDER_SOURCE_PREFIXES)


def _report(expected_total: int, expected_users: int) -> Callable[[str], bool]:
    def _check(output: str) -> bool:
        try:
            data = _parse_json_object(output)
            return (
                int(data.get("active_total")) == expected_total
                and int(data.get("unique_active_users")) == expected_users
                and bool(re.fullmatch(r"(?i)(?:python\s*)?v?3\.7", str(data.get("dict_order_guaranteed_since", "")).strip()))
                and _has_dict_order_source(str(data.get("source_url", "")))
            )
        except (ValueError, TypeError, AttributeError):
            return False
    return _check


def _uses_tool_family(names: list[str], family: str) -> bool:
    for name in names:
        simple = name.lower().split("__")[-1]
        if family == "exec" and simple == "exec":
            return True
        if family == "local" and simple in {"exec", "read", "read_file", "view_file"}:
            return True
        if family == "web" and any(part in simple for part in ("web", "fetch", "search")):
            return True
    return False


_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "active_total": {"type": "integer"},
        "unique_active_users": {"type": "integer"},
        "dict_order_guaranteed_since": {"type": "string"},
        "source_url": {"type": "string"},
    },
    "required": ["active_total", "unique_active_users", "dict_order_guaranteed_since", "source_url"],
    "additionalProperties": False,
}

_JOIN_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string"},
        "adjusted_score": {"type": "integer"},
        "tied_with": {"type": "array", "items": {"type": "string"}},
        "dict_order_guaranteed_since": {"type": "string"},
        "source_url": {"type": "string"},
    },
    "required": ["winner", "adjusted_score", "tied_with", "dict_order_guaranteed_since", "source_url"],
    "additionalProperties": False,
}


def _join_report(output: str) -> bool:
    try:
        data = _parse_json_object(output)
        return (
            data.get("winner") == "iris"
            and int(data.get("adjusted_score")) == 21
            and data.get("tied_with") == ["maple"]
            and bool(re.fullmatch(r"(?i)(?:python\s*)?v?3\.7", str(data.get("dict_order_guaranteed_since", "")).strip()))
            and _has_dict_order_source(str(data.get("source_url", "")))
        )
    except (ValueError, TypeError, AttributeError):
        return False


def build_tasks(workdir: Path) -> list[Task]:
    lines = [f"row {i}: {i * 7919 % 100003:05d}" for i in range(1, 201)]
    data_file = workdir / "rows.txt"
    data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records = [
        ("iris", 8, "yes"), ("cedar", 13, "no"),
        ("iris", 11, "yes"), ("maple", 17, "yes"),
        ("elm", 5, "no"), ("cedar", 19, "yes"),
    ]
    report_file = workdir / "orders.csv"
    with report_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("user", "score", "active"))
        writer.writerows(records)
    adjustment_file = workdir / "adjustments.csv"
    with adjustment_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("user", "delta"))
        writer.writerows((
            ("iris", 2), ("cedar", -3), ("maple", 4), ("elm", 9),
        ))
    return [
        Task("answer", "What is 17 * 23? Reply with only the number.", _exact("391")),
        Task(
            "code_reasoning",
            "What does this Python program print? Reply with only the integer.\n"
            "def f(xs):\n"
            "    seen=set(); out=[]\n"
            "    for n in xs:\n"
            "        if n not in seen:\n"
            "            seen.add(n); out.append(n*n if n%2 else n+3)\n"
            "    return sum(out)\n"
            "print(f([3,2,3,5,2,4,5,-1]))",
            _exact("47"),
        ),
        Task(
            "shell",
            "Run `printf 'chack-%s' $((6*7))` with Chack's exec tool and reply with only its output.",
            _exact("chack-42"),
            needs_shell=True,
            required_tool_families=("exec",),
        ),
        Task(
            "file",
            f"Read line 137 of {data_file} and reply with only that line's text.",
            _exact(lines[136]),
            needs_shell=True,
            required_tool_families=("local",),
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
        Task(
            "public_web",
            "Check the current official Python documentation and report the Python version in which "
            "dictionary insertion order became a language guarantee. Reply with just the version "
            "and an official docs.python.org URL on the next line.",
            lambda output: "3.7" in output and _has_dict_order_source(output),
            needs_web=True,
            required_tool_families=("web",),
        ),
        Task(
            "combined_report",
            f"Prepare a concise report from the local file {report_file} and the current official "
            "Python documentation. Include the sum of scores for active rows, the number of distinct "
            "active users, the Python version that first guaranteed dictionary insertion order, and "
            "an official source URL. Return one JSON object with fields active_total, "
            "unique_active_users, dict_order_guaranteed_since, and source_url. Check both the file "
            "and documentation before answering.",
            _report(55, 3),
            needs_shell=True,
            needs_web=True,
            required_tool_families=("local", "web"),
            output_schema=_REPORT_SCHEMA,
        ),
        Task(
            "join_report",
            f"Prepare a verified report from {report_file}, {adjustment_file}, and the current official "
            "Python documentation. Sum each user's active scores from orders.csv, then apply that user's "
            "adjustment once. Name the highest scoring user; if tied, choose the user whose first active "
            "row appears earlier in orders.csv and list the other tied users. Also report the Python "
            "version that first guaranteed dictionary insertion order and a supporting official URL. "
            "Return one JSON object with fields winner, adjusted_score, tied_with, "
            "dict_order_guaranteed_since, and source_url. Verify the files and documentation.",
            _join_report,
            needs_shell=True,
            needs_web=True,
            required_tool_families=("local", "web"),
            output_schema=_JOIN_SCHEMA,
        ),
    ]


def _config(provider: str, model: str, api_key_type: str, task: Task, max_turns: int) -> ChackConfig:
    tools = ToolsConfig(
        task_steps_manager_enabled=False, exec_enabled=task.needs_shell,
        native_web_search_enabled=task.needs_web,
        keyless_web_search_enabled=task.needs_web,
        max_tools_used=_MAX_TOOL_CALLS,
    )
    agent = AgentConfig(
        thinking_effort="high",
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
    tool_names = sorted(str(name) for name in result.tool_counts)
    tools_ok = all(_uses_tool_family(tool_names, family) for family in task.required_tool_families)
    answer_ok = bool(task.check(output))
    tool_calls = sum(result.tool_counts.values())
    budget_ok = tool_calls <= _MAX_TOOL_CALLS
    row.update(
        passed=answer_ok and tools_ok and budget_ok and not result.error,
        answer_ok=answer_ok,
        tools_ok=tools_ok,
        budget_ok=budget_ok,
        seconds=round(time.monotonic() - started, 1),
        prompt_tokens=result.prompt_tokens,
        cached_prompt_tokens=result.cached_prompt_tokens,
        completion_tokens=result.completion_tokens,
        cache_hit_rate=result.cache_hit_rate,
        cost_usd=result.total_cost,
        tool_calls=tool_calls,
        tool_names=tool_names,
        error=result.error,
        output=output[:500],
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
        "target", "task", "passed", "answer_ok", "tools_ok", "budget_ok", "seconds", "prompt_tokens", "cached_prompt_tokens",
        "completion_tokens", "cache_hit_rate", "cost_usd", "tool_calls", "tool_names", "error",
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
