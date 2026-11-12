"""Run isolated benign benchmark cases across models and Chack backends.

Example (gateway credentials are read from the environment)::

    python scripts/benchmark_model_backend_matrix.py \
      --models bakkt/kimi-k3.1,bakkt/glm-5.2 \
      --tasks repair_allocator,reconcile_ledger,join_report \
      --jsonl /tmp/chack-matrix.jsonl

The JSONL file is append-only and can be resumed after interruption. Every
case gets its own temporary fixture directory and Chack session.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chack_agent.runtime_capabilities import BACKENDS_BY_API_KEY_TYPE  # noqa: E402
from scripts.benchmark_backends import build_tasks, run_one  # noqa: E402

DEFAULT_BACKENDS = (
    "openai", "openrouter", "gateway", "codex", "claude", "gemini",
    "kimi", "qwen", "copilot", "opencode", "langgraph",
)
DEFAULT_TASKS = ("repair_allocator", "reconcile_ledger", "join_report")


def _run_case(backend: str, model: str, task_name: str, max_turns: int, timeout_seconds: int) -> dict:
    target = f"{backend}:{model}:gateway"

    def _expired(_signum, _frame):
        raise TimeoutError(f"Case timed out after {timeout_seconds}s")

    alarm_available = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    previous = signal.signal(signal.SIGALRM, _expired) if alarm_available else None
    if alarm_available:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        with tempfile.TemporaryDirectory(prefix="chack-matrix-case-") as tmp:
            task = next(task for task in build_tasks(Path(tmp)) if task.name == task_name)
            original_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                return run_one(target, task, max_turns)
            finally:
                os.chdir(original_cwd)
    finally:
        if alarm_available:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def _parse_list(value: str, defaults: tuple[str, ...]) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else list(defaults)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="comma-separated gateway model IDs")
    parser.add_argument("--backends", default="", help="comma-separated backend names")
    parser.add_argument("--tasks", default="", help="comma-separated benchmark tasks")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--case-timeout", type=int, default=150, help="wall-clock seconds per case")
    parser.add_argument("--jsonl", required=True, type=Path)
    args = parser.parse_args(argv)

    models = _parse_list(args.models, ())
    backends = _parse_list(args.backends, DEFAULT_BACKENDS)
    tasks = _parse_list(args.tasks, DEFAULT_TASKS)
    with tempfile.TemporaryDirectory(prefix="chack-matrix-list-") as tmp:
        known_tasks = {task.name for task in build_tasks(Path(tmp))}
    for name in tasks:
        if name not in known_tasks:
            parser.error(f"unknown task: {name}")
    for name in backends:
        if name not in DEFAULT_BACKENDS:
            parser.error(f"unknown backend: {name}")
    if args.workers < 1 or args.case_timeout < 1:
        parser.error("--workers and --case-timeout must be positive")

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str]] = set()
    if args.jsonl.exists():
        for line in args.jsonl.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed.add((row["target"], row["task"]))
    queued = []
    unsupported = []
    for model in models:
        for backend in backends:
            target = f"{backend}:{model}:gateway"
            for task_name in tasks:
                if (target, task_name) in completed:
                    continue
                if backend not in BACKENDS_BY_API_KEY_TYPE["gateway"]:
                    unsupported.append({
                        "target": target, "task": task_name, "passed": False,
                        "error": "Backend does not support gateway credentials",
                        "status": "unsupported_route",
                    })
                else:
                    queued.append((backend, model, task_name, args.max_turns, args.case_timeout))

    with args.jsonl.open("a", encoding="utf-8") as stream:
        os.chmod(args.jsonl, 0o600)
        for row in unsupported:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_case, *case): case for case in queued}
            for future in as_completed(futures):
                backend, model, task_name, _, _ = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve one row per case
                    row = {
                        "target": f"{backend}:{model}:gateway", "task": task_name,
                        "passed": False, "error": f"{type(exc).__name__}: {exc}"[:200],
                    }
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                print(
                    f"{backend} {model} {task_name}: {'PASS' if row.get('passed') else 'FAIL'} "
                    f"{row.get('seconds', '-')}s {row.get('tool_calls', '-')} calls",
                    file=sys.stderr, flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
