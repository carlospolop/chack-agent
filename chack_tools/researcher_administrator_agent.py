from __future__ import annotations

import json
import os
import re
import time
import asyncio
import contextvars
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from .config import ToolsConfig
from .subagent_config import (
    aggregate_tool_call_counts,
    begin_researcher_response_collection,
    build_subagent_config,
    create_research_master_dir,
    create_subagent_session_id,
    end_researcher_response_collection,
    enforce_prompt_str_or_list_schema,
    inherit_subagent_limits,
    normalize_subagent_prompts,
    researcher_response_from_output,
    subagent_launch_block_reason,
)
from .task_steps_manager_state import current_session_id
from .research_artifacts import add_research_artifact_tools, cleanup_research_artifacts, reset_research_artifact_context, set_research_artifact_context
from .cancellation import request_cancel, reset_cancellation_event, set_cancellation_event
from .telemetry import current_log_context, reset_log_context, run_with_tool_logging, set_log_context

try:
    from agents import function_tool
    from agents.tool_context import ToolContext
    from agents.usage import Usage
except ImportError:
    function_tool = None
    ToolContext = None
    Usage = None


_ASYNC_RESEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=16)
_ASYNC_RESEARCH_LOCK = threading.Lock()
_ASYNC_RESEARCH_JOBS: dict[str, dict[str, Any]] = {}


class _AdministratorRunAccounting:
    """Mutable accounting scoped to one administrator invocation."""

    def __init__(self) -> None:
        self.async_job_ids: list[str] = []
        self.researcher_counts: Counter[str] = Counter()


_ADMINISTRATOR_RUN_ACCOUNTING: contextvars.ContextVar[_AdministratorRunAccounting | None] = (
    contextvars.ContextVar("chack_administrator_run_accounting", default=None)
)


def _async_job_store(job_id: str, job: dict[str, Any]) -> None:
    with _ASYNC_RESEARCH_LOCK:
        _ASYNC_RESEARCH_JOBS[job_id] = job


def _async_job_get(job_id: str) -> dict[str, Any] | None:
    with _ASYNC_RESEARCH_LOCK:
        return _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())


def _async_job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _ASYNC_RESEARCH_LOCK:
        raw_job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        if not raw_job:
            return None
        return {
            "job_id": raw_job.get("job_id"),
            "created_at": raw_job.get("created_at"),
            "tasks": {
                task_id: {
                    k: v for k, v in task.items()
                    if k not in {"future", "cancel_event"}
                }
                for task_id, task in (raw_job.get("tasks") or {}).items()
            },
        }


def _async_wait_for_completion(job_id: str, timeout_seconds: int) -> bool:
    """Wait for a whole async job, returning early when every task is terminal."""
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        event = (job or {}).get("completion_event")
    if not isinstance(event, threading.Event):
        return False
    return event.wait(timeout=max(0, int(timeout_seconds or 0)))


def _async_jobs_have_nonterminal_tasks(job_ids: list[str]) -> bool:
    """Return true while any administrator-owned async task can still write evidence."""
    terminal = {"done", "error", "cancelled"}
    with _ASYNC_RESEARCH_LOCK:
        for job_id in job_ids:
            job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
            tasks = (job or {}).get("tasks") or {}
            if tasks and any(str(task.get("status") or "") not in terminal for task in tasks.values()):
                return True
    return False


def _async_submit(fn, *args):
    return _ASYNC_RESEARCH_EXECUTOR.submit(fn, *args)


def _async_mark_task_running_or_cancelled(job_id: str, task_id: str, tool_name: str, started_at: float) -> bool:
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        if task and task.get("cancel_requested"):
            task["status"] = "cancelled"
            task["started_at"] = started_at
            task["finished_at"] = started_at
            task["last_activity_at"] = started_at
            task["latest_action"] = "cancelled before start"
            return False
        if task:
            task["status"] = "running"
            task["started_at"] = started_at
            task["last_activity_at"] = started_at
            task["latest_action"] = f"running {tool_name}"
    return True


def _async_record_task_progress(job_id: str, task_id: str, event: dict[str, Any]) -> None:
    tool = event.get("tool") or ""
    event_type = event.get("event") or ""
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        if not task:
            return
        events = task.setdefault("recent_events", [])
        events.append(event)
        if len(events) > 20:
            del events[:-20]
        task["last_activity_at"] = time.time()
        task["latest_action"] = f"{event_type} {tool}".strip()
        if event_type == "tool_started" and tool:
            live_counts = task.setdefault("live_tool_call_counts", {})
            live_counts[tool] = int(live_counts.get(tool, 0)) + 1


def _async_register_task(job_id: str, task_id: str, task: dict[str, Any]) -> None:
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        if not job:
            return
        job.setdefault("tasks", {})[task_id] = task


def _async_set_task_future(job_id: str, task_id: str, future: Any) -> None:
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        if task is not None:
            task["future"] = future


def _async_output_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:80] or "researcher"


def _persist_async_researcher_output(
    evidence_dir: str,
    task_id: str,
    tool_name: str,
    result: dict[str, Any],
) -> None:
    if not evidence_dir or not isinstance(result, dict):
        return
    parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else None
    if parsed is not None:
        response = deepcopy(parsed)
        response.setdefault("researcher_tool", tool_name)
    else:
        response = researcher_response_from_output(tool_name, result.get("output"))
    if response is None:
        return
    root = Path(str(evidence_dir or "")).expanduser()
    try:
        output_dir = root / "researcher_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"async_{_async_output_name(task_id)}_{_async_output_name(tool_name)}.json"
        (output_dir / filename).write_text(_compact_json(response), encoding="utf-8")
    except Exception:
        return


def _async_mark_task_done(job_id: str, task_id: str, future: Any) -> None:
    try:
        result = future.result()
        status = "cancelled" if result.get("cancelled") else "done"
        error = ""
    except Exception as exc:
        result = {}
        status = "cancelled" if future.cancelled() else "error"
        error = f"{type(exc).__name__}: {exc}"
    evidence_dir = ""
    tool_name = ""
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        evidence_dir = str((job or {}).get("evidence_dir") or "")
        if task:
            tool_name = str(task.get("researcher_tool") or result.get("researcher_tool") or "")

    # Persist successful output before publishing a terminal task state or waking
    # completion-aware polls. Otherwise a poll can observe "done" while the
    # durable result file is still absent.
    if status == "done" and result:
        _persist_async_researcher_output(evidence_dir, task_id, tool_name, result)

    completion_event = None
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        if task:
            task["status"] = status
            task["finished_at"] = time.time()
            task["last_activity_at"] = task["finished_at"]
            task["latest_action"] = status
            if error:
                task["error"] = error
            if result:
                task["result"] = result
        tasks = (job or {}).get("tasks") or {}
        expected = int((job or {}).get("expected_task_count") or 0)
        if expected > 0 and len(tasks) == expected and all(
            str(row.get("status") or "") in {"done", "error", "cancelled"}
            for row in tasks.values()
        ):
            completion_event = (job or {}).get("completion_event")
    if isinstance(completion_event, threading.Event):
        completion_event.set()


def _async_cancel_job(job_id: str) -> dict[str, Any]:
    cancelled: list[str] = []
    cancellation_requested: list[str] = []
    already_finished: list[str] = []
    process_kill_requested: list[str] = []
    cancel_events: list[tuple[str, threading.Event]] = []
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        if not job:
            return {"job_found": False, "job_id": job_id, "error": "Unknown async researcher job id."}
        for task_id, task in (job.get("tasks") or {}).items():
            if task.get("status") in {"done", "error", "cancelled"}:
                already_finished.append(task_id)
                continue
            future = task.get("future")
            if future is not None and future.cancel():
                task["status"] = "cancelled"
                task["latest_action"] = "cancelled before start"
                task["finished_at"] = time.time()
                task["last_activity_at"] = task["finished_at"]
                cancelled.append(task_id)
            else:
                task["cancel_requested"] = True
                task["status"] = "cancelling"
                task["latest_action"] = "cancellation requested"
                task["last_activity_at"] = time.time()
                cancel_event = task.get("cancel_event")
                cancellation_requested.append(task_id)
                if isinstance(cancel_event, threading.Event):
                    cancel_events.append((task_id, cancel_event))
    for task_id, cancel_event in cancel_events:
        if request_cancel(cancel_event):
            process_kill_requested.append(task_id)
    return {
        "job_found": True,
        "job_id": job_id,
        "cancelled": cancelled,
        "cancellation_requested": cancellation_requested,
        "process_kill_requested": process_kill_requested,
        "already_finished": already_finished,
        "note": "Queued tasks are cancelled before start. Running Codex/Claude subprocess trees are terminated when the backend has registered them for this async task.",
    }

# Canonical registry of the researchers the administrator can orchestrate.
# short-name -> (ToolsConfig enable attribute, exposed research tool name)
RESEARCHER_REGISTRY: dict[str, tuple[str, str]] = {
    "deepchatgpt": ("deepchatgpt_enabled", "deepchatgpt_researcher"),
    "prochatgpt": ("prochatgpt_enabled", "prochatgpt_researcher"),
    "chatgptxhigh": ("chatgptxhigh_enabled", "chatgptxhigh"),
    "scientific": ("scientific_enabled", "scientific_research"),
    "business": ("business_enabled", "business_research"),
    "product": ("product_enabled", "product_research"),
    "travel": ("travel_enabled", "travel_research"),
    "websearcher": ("websearcher_enabled", "websearcher_research"),
    "social_network": ("social_network_enabled", "social_network_research"),
    "legal": ("legal_enabled", "legal_research"),
    "data_statistics": ("data_statistics_enabled", "data_statistics_research"),
    "news_media": ("news_media_enabled", "news_media_research"),
    "knowledge_graph": ("knowledge_graph_enabled", "knowledge_graph_research"),
    "religious": ("religious_enabled", "religious_research"),
    "cli": ("cli_enabled", "cli_research"),
}

# Friendly aliases the yaml/config may use for a researcher short-name.
_RESEARCHER_ALIASES = {
    "deep_chatgpt": "deepchatgpt",
    "chatgpt_deep": "deepchatgpt",
    "pro_chatgpt": "prochatgpt",
    "chatgpt_pro": "prochatgpt",
    "xhigh": "chatgptxhigh",
    "extra_high": "chatgptxhigh",
    "chatgpt_xhigh": "chatgptxhigh",
    "web": "websearcher",
    "webresearcher": "websearcher",
    "websearch": "websearcher",
    "social": "social_network",
    "socialnetwork": "social_network",
    "science": "scientific",
    "scientific_research": "scientific",
    "data": "data_statistics",
    "statistics": "data_statistics",
    "news": "news_media",
    "media": "news_media",
    "kg": "knowledge_graph",
    "knowledgegraph": "knowledge_graph",
}


def normalize_researcher_name(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _RESEARCHER_ALIASES.get(key, key)
    # Tolerate passing the exposed tool name (e.g. "scientific_research").
    for short, (_attr, tool_name) in RESEARCHER_REGISTRY.items():
        if key == tool_name:
            return short
    return key


RESEARCHER_ADMINISTRATOR_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "research_worked": {
            "type": "boolean",
            "description": "True when the overall research produced a useful, evidence-backed set of conclusions; false when the run was blocked or failed.",
        },
        "failure_reason": {
            "type": "string",
            "description": "Empty when research_worked is true. If false, explain the blocker or failure clearly.",
        },
        "administrator_conclusions": {
            "type": "string",
            "description": "The administrator's own synthesized conclusions across every researcher executed: what was established, contradictions found, remaining gaps, and confidence. Write at least 2000 characters when the evidence supports it.",
        },
    },
    "required": [
        "research_worked",
        "failure_reason",
        "administrator_conclusions",
    ],
}


def researcher_administrator_output_schema(*, preserve_artifacts: bool) -> dict:
    del preserve_artifacts
    return deepcopy(RESEARCHER_ADMINISTRATOR_OUTPUT_SCHEMA)


_ADMINISTRATOR_SYSTEM_PROMPT = """### ROLE
You are a research administrator tasked with a specific research and must obtain evidence only by orchestrating the available specialized researchers, then synthesize their results. Do not answer from prior knowledge except for a trivially certain request that genuinely needs no research. And be always ciritcal checking things from all angles taking into account all kind of edge cases.

### WORKFLOW
1. Map the needed coverage: entities, aliases, timeframe, jurisdictions, claims, and relevant web/scientific/business/product/travel/legal/social/data/news/entity or other source families.
2. Give each researcher a focused prompt of at least 500 characters (better close to 2000) covering scope, sources/tools to prioritize, disconfirming angles, expected comparisons, caveats, and any leads from earlier results.
3. Researchers are blind to one another. Review every result and its `tool_call_counts`; inspect saved evidence when useful. Cross-pollinate material leads into another researcher or a focused follow-up. Repeat a researcher only for a specific unresolved source gap or contradiction, not for generic extra coverage.
4. Stop when the evidence supports a defensible answer or further work has low value. Preserve enough runtime to synthesize; state remaining gaps instead of timing out while chasing completeness.

### LONG-RUNNING RESEARCHERS
Prefer `start_researchers_async` and completion-aware `poll_researchers_async` waits for long work. Poll once immediately after launch. Ordinary jobs normally use 30-120 second waits; ChatGPT browser jobs use 300-600 seconds and may take 45-90 minutes. Queued/starting for a few minutes is not failure.
Never use `wait(..., terminate=true)`, cancellation, or process termination on a running ChatGPT browser researcher merely because it is slow or finalizing. Cancel it only on explicit user request, a proven terminal error, or the configured hard timeout. Ordinary async work may be cancelled when clearly stalled, duplicated, or no longer useful.

### EVIDENCE AND OUTPUT
Stay source-first and objective. Prefer primary or directly inspectable evidence, preserve contradictions, distinguish source claims from inference, and actively consider disconfirming evidence. Never fabricate or fill gaps from assumptions.
In `administrator_conclusions`, distinguish established, contradicted, weakly supported, and unresolved claims, with confidence and important caveats. Return only the configured compact JSON. Do not copy researcher JSON, counts, or evidence paths; runtime code appends them exactly.
"""


def _json_from_output(output: str) -> dict[str, Any] | None:
    text = str(output or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.find("{")
            if start < 0:
                return None
            payload, _end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _step_tool_name(step: Any) -> str:
    if isinstance(step, dict):
        raw = str(step.get("tool") or step.get("name") or "").strip()
        return _normalize_step_tool_name(raw)
    action = step[0] if isinstance(step, tuple) and step else step
    if isinstance(action, dict):
        raw = str(action.get("tool") or action.get("name") or "").strip()
        return _normalize_step_tool_name(raw)
    raw = str(getattr(action, "tool", "") or getattr(action, "name", "") or "").strip()
    return _normalize_step_tool_name(raw)


def _normalize_step_tool_name(raw: str) -> str:
    name = str(raw or "").strip()
    if not name:
        return ""
    if name.startswith("mcp__"):
        tail = name.rsplit("__", 1)[-1].strip()
        if tail:
            return tail
    for prefix in ("chack_tools-", "chack_tools__", "tool_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _step_tool_output(step: Any) -> Any:
    candidates: list[Any] = []
    if isinstance(step, dict):
        candidates.extend(
            [
                step.get("result"),
                step.get("output"),
                step.get("tool_output"),
                step.get("observation"),
            ]
        )
        tool_input = step.get("tool_input")
    else:
        action = step[0] if isinstance(step, tuple) and step else step
        observation = step[1] if isinstance(step, tuple) and len(step) > 1 else None
        candidates.append(observation)
        tool_input = action.get("tool_input") if isinstance(action, dict) else getattr(action, "tool_input", None)
    if isinstance(tool_input, dict):
        candidates.extend(
            [
                tool_input.get("result"),
                tool_input.get("output"),
                tool_input.get("tool_output"),
                tool_input.get("content"),
            ]
        )
    for candidate in candidates:
        if candidate not in (None, ""):
            return _normalize_step_tool_output(candidate)
    return None


def _normalize_step_tool_output(value: Any) -> Any:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            normalized = _normalize_step_tool_output(item)
            if normalized not in (None, ""):
                parts.append(str(normalized))
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "result", "output", "tool_output"):
            candidate = value.get(key)
            if candidate not in (None, ""):
                return _normalize_step_tool_output(candidate)
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)
    return value


def _researcher_responses_from_steps(steps: list[Any]) -> list[dict[str, Any]]:
    researcher_tools = {tool_name for _short, (_attr, tool_name) in RESEARCHER_REGISTRY.items()}
    responses: list[dict[str, Any]] = []
    for step in steps or []:
        tool_name = _step_tool_name(step)
        output = _step_tool_output(step)
        if tool_name == "run_researchers_batch":
            responses.extend(_researcher_responses_from_batch_output(output))
            continue
        if tool_name == "poll_researchers_async":
            responses.extend(_researcher_responses_from_poll_output(output))
            continue
        if tool_name not in researcher_tools:
            continue
        response = researcher_response_from_output(tool_name, output)
        if response is not None:
            responses.append(response)
    return responses


def _short_failure_output(value: Any, max_chars: int = 320) -> str:
    text = str(_normalize_step_tool_output(value) or "").strip()
    text = " ".join(text.split())
    if " details=" in text:
        text = text.split(" details=", 1)[0].rstrip()
    if " {\"type\":\"" in text:
        text = text.split(" {\"type\":\"", 1)[0].rstrip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _researcher_failure_record(
    tool_name: str,
    *,
    status: str = "",
    error: str = "",
    output: Any = "",
    task_id: str = "",
) -> dict[str, Any] | None:
    tool_name = str(tool_name or "").strip()
    researcher_tools = {name for _short, (_attr, name) in RESEARCHER_REGISTRY.items()}
    if tool_name not in researcher_tools:
        return None
    output_text = _short_failure_output(output)
    error_text = _short_failure_output(error)
    if not status and not error_text and not output_text:
        return None
    row = {
        "researcher_tool": tool_name,
        "status": str(status or "unparsed").strip() or "unparsed",
        "failure_reason": error_text or output_text or "Researcher call did not return a parseable researcher JSON response.",
    }
    if task_id:
        row["task_id"] = str(task_id)
    return row


def _researcher_failures_from_poll_output(output: Any) -> list[dict[str, Any]]:
    payload = _json_from_output(str(output or ""))
    if payload is None:
        return []
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    failures: list[dict[str, Any]] = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            continue
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else None
        tool_name = str(task.get("researcher_tool") or result.get("researcher_tool") or "").strip()
        if not tool_name:
            researcher = normalize_researcher_name(str(task.get("researcher") or ""))
            tool_name = RESEARCHER_REGISTRY.get(researcher, ("", ""))[1]
        if not tool_name:
            continue
        if parsed is not None or researcher_response_from_output(tool_name, result.get("output")) is not None:
            continue
        status = str(task.get("status") or result.get("status") or "unparsed")
        if status not in {"done", "error", "cancelled", "unparsed"} and not result.get("output") and not task.get("error"):
            continue
        row = _researcher_failure_record(
            tool_name,
            status=status,
            error=str(task.get("error") or result.get("error") or ""),
            output=result.get("output") or task.get("latest_action") or "",
            task_id=str(task.get("task_id") or ""),
        )
        if row is not None:
            counts = result.get("tool_call_counts") or task.get("tool_call_counts") or {}
            if isinstance(counts, dict):
                compact_counts: dict[str, int] = {}
                for name, value in counts.items():
                    tool = str(name or "").strip()
                    if not tool:
                        continue
                    try:
                        count = int(value or 0)
                    except (TypeError, ValueError):
                        continue
                    if count > 0:
                        compact_counts[tool] = compact_counts.get(tool, 0) + count
                if compact_counts:
                    row["tool_call_counts"] = dict(sorted(compact_counts.items()))
                    row["total_tool_calls"] = int(sum(compact_counts.values()))
            failures.append(row)
    return failures


def _researcher_failures_from_batch_output(output: Any) -> list[dict[str, Any]]:
    payload = _json_from_output(str(output or ""))
    if payload is None:
        return []
    failures: list[dict[str, Any]] = []
    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("researcher_tool") or "").strip()
        if not tool_name:
            researcher = normalize_researcher_name(str(row.get("researcher") or ""))
            tool_name = RESEARCHER_REGISTRY.get(researcher, ("", ""))[1]
        if not tool_name:
            continue
        if isinstance(row.get("parsed_response"), dict) or researcher_response_from_output(tool_name, row.get("output")) is not None:
            continue
        failure = _researcher_failure_record(tool_name, status="unparsed", output=row.get("output"))
        if failure is not None:
            failures.append(failure)
    for row in payload.get("errors") or []:
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("researcher_tool") or "").strip()
        if not tool_name:
            researcher = normalize_researcher_name(str(row.get("researcher") or ""))
            tool_name = RESEARCHER_REGISTRY.get(researcher, ("", ""))[1]
        failure = _researcher_failure_record(tool_name, status="error", error=str(row.get("error") or ""))
        if failure is not None:
            failures.append(failure)
    return failures


def _researcher_failures_from_steps(steps: list[Any]) -> list[dict[str, Any]]:
    researcher_tools = {tool_name for _short, (_attr, tool_name) in RESEARCHER_REGISTRY.items()}
    failures: list[dict[str, Any]] = []
    for step in steps or []:
        tool_name = _step_tool_name(step)
        output = _step_tool_output(step)
        if tool_name == "run_researchers_batch":
            failures.extend(_researcher_failures_from_batch_output(output))
            continue
        if tool_name == "poll_researchers_async":
            failures.extend(_researcher_failures_from_poll_output(output))
            continue
        if tool_name not in researcher_tools:
            continue
        if researcher_response_from_output(tool_name, output) is not None:
            continue
        if str(output or "").strip().startswith("ERROR:"):
            failure = _researcher_failure_record(tool_name, status="error", output=output)
            if failure is not None:
                failures.append(failure)
    return failures


def _researcher_failures_from_async_jobs(job_ids: list[str]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for job_id in job_ids or []:
        snapshot = _async_job_snapshot(job_id)
        if not snapshot:
            continue
        tasks = list((snapshot.get("tasks") or {}).values())
        failures.extend(_researcher_failures_from_poll_output(_compact_json({"tasks": tasks})))
    return failures


def _researcher_responses_from_poll_output(output: Any) -> list[dict[str, Any]]:
    payload = _json_from_output(str(output or ""))
    if payload is None:
        return []
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    responses: list[dict[str, Any]] = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            continue
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else None
        tool_name = str(
            task.get("researcher_tool")
            or result.get("researcher_tool")
            or ""
        ).strip()
        if not tool_name:
            researcher = normalize_researcher_name(str(task.get("researcher") or ""))
            tool_name = RESEARCHER_REGISTRY.get(researcher, ("", ""))[1]
        if not tool_name:
            continue
        if parsed is not None:
            response = deepcopy(parsed)
            response.setdefault("researcher_tool", tool_name)
        else:
            response = researcher_response_from_output(tool_name, result.get("output"))
        if response is not None:
            responses.append(response)
    return responses


def _researcher_responses_from_async_jobs(job_ids: list[str]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for job_id in job_ids or []:
        snapshot = _async_job_snapshot(job_id)
        if not snapshot:
            continue
        tasks = list((snapshot.get("tasks") or {}).values())
        responses.extend(_researcher_responses_from_poll_output(_compact_json({"tasks": tasks})))
    return responses


def _researcher_responses_from_async_output_files(evidence_dir: str) -> list[dict[str, Any]]:
    root = Path(str(evidence_dir or "")).expanduser()
    output_dir = root / "researcher_outputs"
    if not output_dir.is_dir():
        return []
    responses: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("async_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        tool_name = str(payload.get("researcher_tool") or "").strip()
        if tool_name:
            responses.append(payload)
    return responses


def _researcher_responses_from_batch_output(output: Any) -> list[dict[str, Any]]:
    payload = _json_from_output(str(output or ""))
    if payload is None:
        return []
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []
    responses: list[dict[str, Any]] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("researcher_tool") or "").strip()
        if not tool_name:
            researcher = normalize_researcher_name(str(row.get("researcher") or ""))
            tool_name = RESEARCHER_REGISTRY.get(researcher, ("", ""))[1]
        if not tool_name:
            continue
        if isinstance(row.get("parsed_response"), dict):
            response = deepcopy(row["parsed_response"])
            response.setdefault("researcher_tool", tool_name)
        else:
            response = researcher_response_from_output(tool_name, row.get("output"))
        if response is not None:
            responses.append(response)
    return responses


def _researcher_call_counts_from_async_jobs(job_ids: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    researcher_tools = {tool_name for _short, (_attr, tool_name) in RESEARCHER_REGISTRY.items()}
    with _ASYNC_RESEARCH_LOCK:
        for job_id in job_ids or []:
            job = _ASYNC_RESEARCH_JOBS.get(job_id)
            if not job:
                continue
            for task in (job.get("tasks") or {}).values():
                tool_name = str(task.get("researcher_tool") or "").strip()
                if tool_name in researcher_tools:
                    counts[tool_name] += 1
    return counts


def _dedupe_researcher_responses(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for response in responses:
        if not isinstance(response, dict):
            continue
        marker = _compact_json(response)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(response)
    return unique


def _researcher_call_counts(
    tool_counts: Counter[str],
    responses: list[dict[str, Any]],
    failures: list[dict[str, str]] | None = None,
) -> dict[str, int]:
    researcher_tools = {tool_name for _short, (_attr, tool_name) in RESEARCHER_REGISTRY.items()}
    rows: Counter[str] = Counter()
    for response in responses or []:
        if not isinstance(response, dict):
            continue
        tool_name = str(response.get("researcher_tool") or "").strip()
        if tool_name in researcher_tools:
            rows[tool_name] += 1
    for failure in failures or []:
        if not isinstance(failure, dict):
            continue
        tool_name = str(failure.get("researcher_tool") or "").strip()
        if tool_name in researcher_tools:
            rows[tool_name] += 1
    for name, count in (tool_counts or {}).items():
        normalized = _normalize_step_tool_name(str(name or ""))
        if normalized in researcher_tools and int(count or 0) > 0:
            rows[normalized] = max(int(rows.get(normalized, 0)), int(count))
    return dict(sorted(rows.items()))


def _artifact_manifest_tool_counts(folder: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    manifest = folder / "_artifact_manifest.jsonl"
    if not manifest.is_file():
        return counts
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        tool = str(payload.get("tool") or payload.get("kind") or "").strip()
        if tool:
            counts[tool] += 1
    return counts


def _enrich_failures_with_artifact_counts(
    evidence_dir: str,
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(str(evidence_dir or "")).expanduser()
    if not root.is_dir():
        return failures
    short_by_tool = {tool_name: short for short, (_attr, tool_name) in RESEARCHER_REGISTRY.items()}
    enriched: list[dict[str, Any]] = []
    for failure in failures or []:
        if not isinstance(failure, dict):
            continue
        row = deepcopy(failure)
        if isinstance(row.get("tool_call_counts"), dict) and row.get("tool_call_counts"):
            enriched.append(row)
            continue
        short = short_by_tool.get(str(row.get("researcher_tool") or "").strip())
        if not short:
            enriched.append(row)
            continue
        counts = _artifact_manifest_tool_counts(root / short)
        if counts:
            compact = dict(sorted((name, int(count)) for name, count in counts.items() if int(count) > 0))
            row["tool_call_counts"] = compact
            row["total_tool_calls"] = int(sum(compact.values()))
            reason = str(row.get("failure_reason") or "").strip()
            tool_bits = ", ".join(f"{name}:{count}" for name, count in list(compact.items())[:6])
            if tool_bits and tool_bits not in reason:
                suffix = f" Preserved artifact manifest tool/kind rows: {tool_bits}."
                row["failure_reason"] = (reason + suffix).strip()
        enriched.append(row)
    return enriched


def _partial_artifact_failures(
    evidence_dir: str,
    responses: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(str(evidence_dir or "")).expanduser()
    if not root.is_dir():
        return []
    accounted = {
        str(row.get("researcher_tool") or "").strip()
        for row in list(responses or []) + list(failures or [])
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for short, (_attr, tool_name) in RESEARCHER_REGISTRY.items():
        if tool_name in accounted:
            continue
        folder = root / short
        if not folder.is_dir():
            continue
        files = [path for path in folder.rglob("*") if path.is_file() and path.name != "_artifact_manifest.jsonl"]
        if not files:
            continue
        tool_counts = _artifact_manifest_tool_counts(folder)
        tool_bits = ", ".join(f"{name}:{count}" for name, count in sorted(tool_counts.items())[:6])
        reason = f"Researcher produced {len(files)} preserved artifact file(s) but did not return parseable final researcher JSON."
        if tool_bits:
            reason += f" Artifact manifest tool/kind rows: {tool_bits}."
        row: dict[str, Any] = {
            "researcher_tool": tool_name,
            "status": "partial_artifacts_without_result",
            "failure_reason": reason,
        }
        if tool_counts:
            compact = dict(sorted((name, int(count)) for name, count in tool_counts.items() if int(count) > 0))
            row["tool_call_counts"] = compact
            row["total_tool_calls"] = int(sum(compact.values()))
        rows.append(row)
    return rows


def finalize_researcher_administrator_output(
    output: str,
    *,
    evidence_dir: str,
    save_artifacts: bool,
    researcher_responses: list[dict[str, Any]],
    tool_counts: Counter[str],
    steps: list[Any],
    researcher_failures: list[dict[str, Any]] | None = None,
    required_researchers: list[str] | None = None,
) -> str:
    payload = _json_from_output(output)
    if payload is None:
        return output
    responses = _dedupe_researcher_responses(
        list(researcher_responses or []) + _researcher_responses_from_steps(steps)
    )
    payload["researcher_responses"] = responses
    failures = list(researcher_failures or []) + _researcher_failures_from_steps(steps)
    failures.extend(_partial_artifact_failures(evidence_dir, responses, failures))
    failures = _enrich_failures_with_artifact_counts(evidence_dir, failures)
    if failures:
        seen_failures: set[str] = set()
        unique_failures: list[dict[str, Any]] = []
        for failure in failures:
            marker = _compact_json(failure)
            if marker in seen_failures:
                continue
            seen_failures.add(marker)
            unique_failures.append(failure)
        payload["researcher_failures"] = unique_failures
    else:
        payload["researcher_failures"] = []
    payload["researcher_tool_call_counts"] = aggregate_tool_call_counts(responses + payload["researcher_failures"])
    calls = _researcher_call_counts(tool_counts, responses, payload["researcher_failures"])
    payload["researcher_call_counts"] = calls
    payload["total_researcher_calls"] = int(sum(calls.values()))
    required_shorts = ResearcherAdministratorAgentTool._normalize_researchers(required_researchers)
    if required_shorts:
        required_tools = [RESEARCHER_REGISTRY[short][1] for short in required_shorts]
        successful_tools = {
            str(response.get("researcher_tool") or "").strip()
            for response in responses
            if isinstance(response, dict) and response.get("research_worked") is True
        }
        missing_tools = [name for name in required_tools if name not in successful_tools]
        payload["required_researchers"] = required_tools
        payload["required_researchers_satisfied"] = not missing_tools
        if missing_tools:
            message = "Required researchers did not complete successfully: " + ", ".join(missing_tools) + "."
            existing_reason = str(payload.get("failure_reason") or "").strip()
            payload["research_worked"] = False
            payload["failure_reason"] = f"{existing_reason} {message}".strip()
    if save_artifacts:
        try:
            from .research_artifacts import register_untracked_research_artifacts

            root_path = Path(str(evidence_dir or "")).expanduser()
            if root_path.is_dir():
                for child in sorted(root_path.iterdir()):
                    if child.is_dir() and child.name != "researcher_outputs":
                        register_untracked_research_artifacts(child)
        except Exception:
            pass
        payload["evidence_data_path"] = evidence_dir
        output_files = _write_researcher_administrator_output_files(
            evidence_dir,
            payload,
            responses,
            payload["researcher_failures"],
        )
        if output_files:
            payload["output_files"] = output_files
    return _compact_json(payload)


def _safe_output_name(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text or fallback


def _write_compact_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _write_researcher_administrator_output_files(
    evidence_dir: str,
    admin_payload: dict[str, Any],
    researcher_responses: list[dict[str, Any]],
    researcher_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(str(evidence_dir or "")).expanduser()
    if not root:
        return {}
    try:
        root.mkdir(parents=True, exist_ok=True)
        admin_file = root / "admin_output.json"
        researcher_dir = root / "researcher_outputs"
        researcher_files: list[str] = []
        admin_copy = deepcopy(admin_payload)
        admin_copy.pop("output_files", None)
        _write_compact_json(admin_file, admin_copy)
        records = list(researcher_responses or [])
        records.extend(
            {
                "research_worked": False,
                "failure_reason": str(failure.get("failure_reason") or ""),
                "final_research_review": "",
                **failure,
            }
            for failure in (researcher_failures or [])
            if isinstance(failure, dict)
        )
        for idx, response in enumerate(records, start=1):
            tool = _safe_output_name(str(response.get("researcher_tool") or "researcher"), "researcher")
            filename = f"{idx:03d}_{tool}.json"
            _write_compact_json(researcher_dir / filename, response)
            researcher_files.append(str(Path("researcher_outputs") / filename))
        return {
            "administrator_output": "admin_output.json",
            "researcher_outputs": researcher_files,
        }
    except Exception:
        return {}


class ResearcherAdministratorAgentTool:
    def __init__(
        self,
        config: ToolsConfig,
        model_name: str = "",
        fallback_model: str = "",
        model_provider: str = "",
        max_turns: int = 100,
        researchers: Optional[list[str]] = None,
        required_researchers: Optional[list[str]] = None,
        researcher_model_overrides: Optional[dict] = None,
        researcher_max_turns_overrides: Optional[dict] = None,
        social_network_model: str = "",
        scientific_model: str = "",
        websearcher_model: str = "",
        business_model: str = "",
        product_model: str = "",
        travel_model: str = "",
        legal_model: str = "",
        data_statistics_model: str = "",
        news_media_model: str = "",
        knowledge_graph_model: str = "",
        religious_model: str = "",
        cli_model: str = "",
        social_network_max_turns: int = 30,
        scientific_max_turns: int = 30,
        websearcher_max_turns: int = 30,
        business_max_turns: int = 30,
        product_max_turns: int = 30,
        travel_max_turns: int = 40,
        legal_max_turns: int = 30,
        data_statistics_max_turns: int = 30,
        news_media_max_turns: int = 30,
        knowledge_graph_max_turns: int = 30,
        religious_max_turns: int = 30,
        cli_max_turns: int = 30,
        self_critique_enabled: bool = False,
        self_critique_rounds: int = 0,
    ):
        self.config = config
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.max_turns = max(2, int(max_turns or 100))
        self.self_critique_rounds = max(0, int(self_critique_rounds or 0))
        self.self_critique_enabled = bool(self_critique_enabled or self.self_critique_rounds > 0)
        self.researchers = self._normalize_researchers(researchers)
        self.required_researchers = self._normalize_researchers(required_researchers)
        unavailable_required = sorted(set(self.required_researchers) - set(self._enabled_researchers()))
        if unavailable_required:
            raise ValueError(
                "required_researchers must be enabled for this administrator: "
                + ", ".join(unavailable_required)
            )
        self.researcher_model_overrides = self._normalize_override_map(researcher_model_overrides)
        self.researcher_max_turns_overrides = self._normalize_override_map(researcher_max_turns_overrides)
        self.social_network_model = social_network_model
        self.scientific_model = scientific_model
        self.websearcher_model = websearcher_model
        self.business_model = business_model
        self.product_model = product_model
        self.travel_model = travel_model
        self.legal_model = legal_model
        self.data_statistics_model = data_statistics_model
        self.news_media_model = news_media_model
        self.knowledge_graph_model = knowledge_graph_model
        self.religious_model = religious_model
        self.cli_model = cli_model
        self.social_network_max_turns = max(2, int(social_network_max_turns or 30))
        self.scientific_max_turns = max(2, int(scientific_max_turns or 30))
        self.websearcher_max_turns = max(2, int(websearcher_max_turns or 30))
        self.business_max_turns = max(2, int(business_max_turns or 30))
        self.product_max_turns = max(2, int(product_max_turns or 30))
        self.travel_max_turns = max(2, int(travel_max_turns or 40))
        self.legal_max_turns = max(2, int(legal_max_turns or 30))
        self.data_statistics_max_turns = max(2, int(data_statistics_max_turns or 30))
        self.news_media_max_turns = max(2, int(news_media_max_turns or 30))
        self.knowledge_graph_max_turns = max(2, int(knowledge_graph_max_turns or 30))
        self.religious_max_turns = max(2, int(religious_max_turns or 30))
        self.cli_max_turns = max(2, int(cli_max_turns or 30))
        self._fallback_run_accounting = _AdministratorRunAccounting()

    def _current_run_accounting(self) -> _AdministratorRunAccounting:
        accounting = _ADMINISTRATOR_RUN_ACCOUNTING.get()
        return accounting if accounting is not None else self._fallback_run_accounting

    @property
    def _launched_async_job_ids(self) -> list[str]:
        return self._current_run_accounting().async_job_ids

    @_launched_async_job_ids.setter
    def _launched_async_job_ids(self, value: list[str]) -> None:
        accounting = self._current_run_accounting()
        accounting.async_job_ids[:] = list(value or [])

    @property
    def _launched_researcher_counts(self) -> Counter[str]:
        return self._current_run_accounting().researcher_counts

    @_launched_researcher_counts.setter
    def _launched_researcher_counts(self, value: Counter[str]) -> None:
        accounting = self._current_run_accounting()
        accounting.researcher_counts.clear()
        accounting.researcher_counts.update(value or {})

    def _launched_researcher_tool_counts(self) -> Counter[str]:
        """Map run-local launch attempts to canonical researcher tool names."""
        tool_names = {tool_name for _short, (_attr, tool_name) in RESEARCHER_REGISTRY.items()}
        counts: Counter[str] = Counter()
        for raw_name, raw_count in self._launched_researcher_counts.items():
            count = int(raw_count or 0)
            if count <= 0:
                continue
            name = str(raw_name or "").strip()
            if name in tool_names:
                counts[name] += count
                continue
            short = normalize_researcher_name(name)
            if short in RESEARCHER_REGISTRY:
                counts[RESEARCHER_REGISTRY[short][1]] += count
        return counts

    @staticmethod
    def _normalize_researchers(researchers: Optional[list[str]]) -> list[str]:
        if not researchers:
            return []
        seen: list[str] = []
        for item in researchers:
            short = normalize_researcher_name(item)
            if short in RESEARCHER_REGISTRY and short not in seen:
                seen.append(short)
        return seen

    @staticmethod
    def _normalize_override_map(overrides: Optional[dict]) -> dict:
        """Map researcher short-names (accepting aliases) to override values."""
        if not isinstance(overrides, dict):
            return {}
        normalized: dict = {}
        for key, value in overrides.items():
            short = normalize_researcher_name(key)
            if short in RESEARCHER_REGISTRY and value not in (None, ""):
                normalized[short] = value
        return normalized

    def _model_for(self, short: str, default: str) -> str:
        # Alias resolution happens inside the inner AgentsToolset.
        raw = str(self.researcher_model_overrides.get(short) or "").strip()
        return raw or default

    def _max_turns_for(self, short: str, default: int) -> int:
        value = self.researcher_max_turns_overrides.get(short)
        if value in (None, ""):
            return default
        try:
            return max(2, int(value))
        except (TypeError, ValueError):
            return default

    def _researcher_self_critique_enabled(self) -> bool:
        """Administrator-spawned researchers always get a try-harder pass."""
        return True

    def _researcher_self_critique_rounds(self) -> int:
        return max(1, int(self.self_critique_rounds or 0))

    def _enabled_researchers(self) -> list[str]:
        """Researcher short-names the administrator is allowed to launch.

        Uses the configured allowlist when provided; otherwise falls back to
        every researcher enabled at the top level of the parent config.
        """
        if self.researchers:
            return list(self.researchers)
        enabled: list[str] = []
        for short, (attr, _tool) in RESEARCHER_REGISTRY.items():
            if bool(getattr(self.config, attr, False)):
                enabled.append(short)
            elif short == "websearcher" and bool(getattr(self.config, "webresearcher_enabled", False)):
                enabled.append(short)
        return enabled

    def _resolved_model(self) -> Optional[str]:
        configured = (self.model_name or "").strip()
        if configured:
            return configured
        fallback = (self.fallback_model or "").strip()
        return fallback or None

    @staticmethod
    def _name_of_tool(tool) -> str:
        return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()

    @staticmethod
    def _invoke_tool_sync(tool: Any, payload: dict[str, Any]) -> str:
        if ToolContext is None or Usage is None:
            raise RuntimeError("OpenAI Agents SDK tool context is not available.")
        raw_args = json.dumps(payload, ensure_ascii=False)

        async def _invoke() -> Any:
            ctx = ToolContext(
                context=None,
                usage=Usage(),
                tool_name=str(getattr(tool, "name", "tool") or "tool"),
                tool_call_id=f"batch-{uuid.uuid4()}",
                tool_arguments=raw_args,
            )
            return await tool.on_invoke_tool(ctx, raw_args)

        result = asyncio.run(_invoke())
        if isinstance(result, str):
            return result
        if isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")
        if isinstance(result, (dict, list, tuple)):
            return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        try:
            return json.dumps(result.model_dump(), ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(result)

    def _duplicate_launch_error(self, short: str, payload: dict[str, Any]) -> str:
        if self._launched_researcher_counts.get(short, 0) <= 0:
            return ""
        prompt = str(payload.get("prompt") or "")
        duplicate_reason = str(payload.get("duplicate_reason") or "").strip()
        if not duplicate_reason:
            match = re.search(r"(?is)duplicate[_ -]?reason\s*:\s*(.{80,})", prompt)
            duplicate_reason = match.group(1).strip() if match else ""
        if len(duplicate_reason) >= 80:
            return ""
        return (
            "ERROR: duplicate researcher launch blocked. "
            f"`{short}` was already launched in this administrator run. "
            "Launch a different relevant researcher or include `Duplicate reason: ...` "
            "with at least 80 characters in the prompt explaining the material new gap, "
            "source family, or contradiction that justifies repeating this researcher."
        )

    def _guard_researcher_tool(self, tool: Any, short: str) -> Any:
        original = getattr(tool, "on_invoke_tool", None)
        if original is None:
            return tool

        async def guarded_on_invoke(ctx, raw_args):
            try:
                payload = json.loads(str(raw_args or "{}"))
            except json.JSONDecodeError:
                payload = {}
            duplicate_error = self._duplicate_launch_error(short, payload if isinstance(payload, dict) else {})
            if duplicate_error:
                return duplicate_error
            self._launched_researcher_counts[short] += 1
            return await original(ctx, raw_args)

        tool.on_invoke_tool = guarded_on_invoke
        tool.description = (
            f"{str(getattr(tool, 'description', '') or '').rstrip()}\n\n"
            "Duplicate guard: if this researcher was already launched in the current "
            "administrator run, repeat it only when the prompt includes `Duplicate reason:` "
            "with at least 80 characters explaining the material new gap or contradiction."
        ).strip()
        return tool

    def _build_capability_tools_for_researcher(self, short: str) -> list[Any]:
        """Build the actual internal tools a researcher would receive in this run."""
        if short == "deepchatgpt":
            # The exposed researcher is itself the browser-backed capability; it
            # does not spin up another Chack subagent with internal tools.
            return []
        elif short in {"prochatgpt", "chatgptxhigh"}:
            return []
        elif short == "websearcher":
            from .websearcher_agent import WebSearcherAgentTool

            helper = WebSearcherAgentTool(
                self.config,
                model_name=self._model_for("websearcher", self.websearcher_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("websearcher", self.websearcher_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "scientific":
            from .scientific_research_agent import ScientificResearchAgentTool

            helper = ScientificResearchAgentTool(
                self.config,
                model_name=self._model_for("scientific", self.scientific_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("scientific", self.scientific_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "business":
            from .business_research_agent import BusinessResearchAgentTool

            helper = BusinessResearchAgentTool(
                self.config,
                model_name=self._model_for("business", self.business_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("business", self.business_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "product":
            from .product_research_agent import ProductResearchAgentTool

            helper = ProductResearchAgentTool(
                self.config,
                model_name=self._model_for("product", self.product_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("product", self.product_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "travel":
            from .travel_research_agent import TravelResearchAgentTool

            helper = TravelResearchAgentTool(
                self.config,
                model_name=self._model_for("travel", self.travel_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("travel", self.travel_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "social_network":
            from .social_network_agent import SocialNetworkAgentTool

            helper = SocialNetworkAgentTool(
                self.config,
                model_name=self._model_for("social_network", self.social_network_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("social_network", self.social_network_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "cli":
            from .cli_research_agent import CliResearchAgentTool

            helper = CliResearchAgentTool(
                self.config,
                model_name=self._model_for("cli", self.cli_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("cli", self.cli_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "legal":
            from .open_research_agents import build_legal_agent

            helper = build_legal_agent(
                self.config,
                model_name=self._model_for("legal", self.legal_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("legal", self.legal_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "data_statistics":
            from .open_research_agents import build_data_statistics_agent

            helper = build_data_statistics_agent(
                self.config,
                model_name=self._model_for("data_statistics", self.data_statistics_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("data_statistics", self.data_statistics_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "news_media":
            from .open_research_agents import build_news_media_agent

            helper = build_news_media_agent(
                self.config,
                model_name=self._model_for("news_media", self.news_media_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("news_media", self.news_media_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "knowledge_graph":
            from .open_research_agents import build_knowledge_graph_agent

            helper = build_knowledge_graph_agent(
                self.config,
                model_name=self._model_for("knowledge_graph", self.knowledge_graph_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("knowledge_graph", self.knowledge_graph_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        elif short == "religious":
            from .open_research_agents import build_religious_agent

            helper = build_religious_agent(
                self.config,
                model_name=self._model_for("religious", self.religious_model),
                fallback_model=self.fallback_model,
                model_provider=self.model_provider,
                max_turns=self._max_turns_for("religious", self.religious_max_turns),
                self_critique_enabled=self._researcher_self_critique_enabled(),
                self_critique_rounds=self._researcher_self_critique_rounds(),
            )
        else:
            return []
        return list(helper._build_subagent_tools())

    def _researcher_capability_lines(self, enabled_researchers: list[str]) -> list[str]:
        """Return compact per-researcher internal tool names for the administrator prompt."""
        lines: list[str] = []
        for short in enabled_researchers:
            exposed_tool = RESEARCHER_REGISTRY[short][1]
            if short == "deepchatgpt":
                lines.append(
                    f"- {short} via `{exposed_tool}`: authenticated ChatGPT Deep Research browser; full response and artifacts"
                )
                continue
            if short == "prochatgpt":
                lines.append(
                    f"- {short} via `{exposed_tool}`: authenticated ChatGPT Pro browser; full response and artifacts"
                )
                continue
            if short == "chatgptxhigh":
                lines.append(
                    f"- {short} via `{exposed_tool}`: authenticated ChatGPT Extra High browser; full response and artifacts"
                )
                continue
            try:
                tools = self._build_capability_tools_for_researcher(short)
                seen: set[str] = set()
                names: list[str] = []
                for tool in tools:
                    name = self._name_of_tool(tool)
                    if name and name not in seen:
                        seen.add(name)
                        names.append(name)
                capability = ", ".join(names) if names else "no internal tools available"
            except Exception as exc:
                capability = f"capability map unavailable ({type(exc).__name__}: {exc})"
            lines.append(f"- {short} via `{exposed_tool}`: {capability}")
        return lines

    @staticmethod
    def _chatgpt_priority_instruction(enabled_researchers: list[str]) -> str:
        """Prioritize the strongest, slowest ChatGPT browser researchers when available."""
        enabled = set(enabled_researchers)
        preferred = [
            RESEARCHER_REGISTRY[short][1]
            for short in ("deepchatgpt", "prochatgpt")
            if short in enabled
        ]
        if preferred:
            tools = ", ".join(f"`{name}`" for name in preferred)
            return (
                "ChatGPT priority: "
                f"{tools} {'is' if len(preferred) == 1 else 'are'} enabled. "
                "They are the slowest and strongest researchers: start every enabled one immediately "
                "in the first wave with `start_researchers_async`, before shorter work. Use long "
                "completion-aware polls and prioritize their completed findings.\n"
            )
        if "chatgptxhigh" in enabled:
            return (
                "ChatGPT priority: neither `deepchatgpt_researcher` nor "
                "`prochatgpt_researcher` is available, but `chatgptxhigh` is enabled. It is the "
                "best available ChatGPT researcher: start it immediately in the first wave with "
                "`start_researchers_async`, before shorter work. Use long completion-aware polls "
                "and prioritize its completed findings.\n"
            )
        return ""

    def _build_batch_tool(self, tools_by_name: dict[str, Any], enabled_researchers: list[str]):
        enabled = set(enabled_researchers)
        allowed_tool_names = {RESEARCHER_REGISTRY[short][1] for short in enabled}

        @function_tool(name_override="run_researchers_batch")
        def run_researchers_batch(
            requests_json: str,
            save_artifacts: bool = False,
            max_parallel: int = 4,
        ) -> str:
            """Run several relevant specialized researchers sequentially and return all results.

            Use this for the first wave when multiple independent researcher types are genuinely
            relevant to the topic. Do not include unrelated researchers just to increase coverage.

            Args:
                requests_json: Compact JSON array of request objects. Each object must contain:
                    researcher: researcher short-name or tool name, such as "websearcher",
                        "scientific", "business", "legal", "data_statistics", "news_media",
                        "knowledge_graph", "social_network", "product", "travel", "religious", or "cli".
                    prompt: detailed researcher prompt, at least 500 characters, with scope,
                        entities, timeframe, source/tool families, disconfirming angles, and
                        expected comparisons. Keep each prompt specific to that researcher.
                save_artifacts: Pass true when source/detail artifacts should be preserved for the
                    final administrator run. When false, files may be temporary and deleted later.
                max_parallel: Accepted for compatibility but currently ignored; this in-process
                    server runs one researcher at a time so evidence paths cannot race.

            Output: Compact JSON with batch_worked, results, and errors. Each successful result
            includes researcher, researcher_tool, parsed_response when available, and raw output.
            The administrator runtime later appends these parsed researcher responses to the final
            administrator JSON.
            """
            try:
                requests = json.loads(str(requests_json or "[]"))
            except json.JSONDecodeError as exc:
                return _compact_json(
                    {
                        "batch_worked": False,
                        "errors": [{"error": f"requests_json is not valid JSON: {exc}"}],
                        "results": [],
                    }
                )
            if not isinstance(requests, list) or not requests:
                return _compact_json(
                    {
                        "batch_worked": False,
                        "errors": [{"error": "requests_json must be a non-empty JSON array."}],
                        "results": [],
                    }
                )

            normalized: list[dict[str, str]] = []
            errors: list[dict[str, str]] = []
            for index, item in enumerate(requests):
                if not isinstance(item, dict):
                    errors.append({"index": str(index), "error": "Each batch item must be an object."})
                    continue
                short = normalize_researcher_name(str(item.get("researcher") or item.get("tool") or ""))
                if short not in enabled:
                    errors.append(
                        {
                            "index": str(index),
                            "researcher": short,
                            "error": "Researcher is not enabled for this administrator.",
                        }
                    )
                    continue
                tool_name = RESEARCHER_REGISTRY[short][1]
                if tool_name not in allowed_tool_names or tool_name not in tools_by_name:
                    errors.append(
                        {
                            "index": str(index),
                            "researcher": short,
                            "error": "Researcher tool is not available in this run.",
                        }
                    )
                    continue
                prompt = str(item.get("prompt") or "").strip()
                if len(prompt) < 500:
                    errors.append(
                        {
                            "index": str(index),
                            "researcher": short,
                            "error": "Researcher prompt must be at least 500 characters.",
                        }
                    )
                    continue
                normalized.append({"researcher": short, "tool_name": tool_name, "prompt": prompt})

            configured_budget = max(0, int(getattr(self.config, "researcher_administrator_max_tools_used", 0) or 0))
            if configured_budget > 0 and len(normalized) > configured_budget:
                errors.append(
                    {
                        "error": (
                            f"Batch requested {len(normalized)} researchers but this administrator "
                            f"is configured for at most {configured_budget} researcher calls."
                        )
                    }
                )
                return _compact_json({"batch_worked": False, "errors": errors, "results": []})

            if not normalized:
                return _compact_json({"batch_worked": False, "errors": errors, "results": []})

            worker_count = 1

            def _run_one(row: dict[str, str], context: contextvars.Context) -> dict[str, Any]:
                def _inner() -> dict[str, Any]:
                    tool_name = row["tool_name"]
                    output = self._invoke_tool_sync(
                        tools_by_name[tool_name],
                        {"prompt": row["prompt"], "save_artifacts": bool(save_artifacts)},
                    )
                    parsed = researcher_response_from_output(tool_name, output)
                    result: dict[str, Any] = {
                        "researcher": row["researcher"],
                        "researcher_tool": tool_name,
                        "output": output,
                    }
                    if parsed is not None:
                        result["parsed_response"] = parsed
                    return result

                return context.run(_inner)

            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_run_one, row, contextvars.copy_context()): row
                    for row in normalized
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        errors.append(
                            {
                                "researcher": row["researcher"],
                                "researcher_tool": row["tool_name"],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
            results.sort(key=lambda item: str(item.get("researcher_tool") or ""))
            return _compact_json(
                {
                    "batch_worked": bool(results) and not errors,
                    "results": results,
                    "errors": errors,
                }
            )

        tool = run_researchers_batch
        tool.description = (
            f"{tool.description}\n\n"
            "Parameters: requests_json is a JSON array of objects with researcher and prompt; "
            "each prompt must be >=500 characters and relevant to that specific researcher. "
            "Set save_artifacts true when evidence files should be preserved. max_parallel is accepted for compatibility but execution is serialized for evidence isolation.\n"
            "Output: Compact JSON containing every researcher result plus errors; final administrator output "
            "will include parsed researcher responses and tool counts."
        )
        return tool

    def _normalize_researcher_requests(
        self,
        requests_json: str,
        *,
        enabled: set[str],
        tools_by_name: dict[str, Any],
        enforce_budget: bool = True,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        try:
            requests = json.loads(str(requests_json or "[]"))
        except json.JSONDecodeError as exc:
            return [], [{"error": f"requests_json is not valid JSON: {exc}"}]
        if not isinstance(requests, list) or not requests:
            return [], [{"error": "requests_json must be a non-empty JSON array."}]

        normalized: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        for index, item in enumerate(requests):
            if not isinstance(item, dict):
                errors.append({"index": str(index), "error": "Each batch item must be an object."})
                continue
            short = normalize_researcher_name(str(item.get("researcher") or item.get("tool") or ""))
            if short not in enabled:
                errors.append(
                    {
                        "index": str(index),
                        "researcher": short,
                        "error": "Researcher is not enabled for this administrator.",
                    }
                )
                continue
            tool_name = RESEARCHER_REGISTRY[short][1]
            if tool_name not in tools_by_name:
                errors.append(
                    {
                        "index": str(index),
                        "researcher": short,
                        "error": "Researcher tool is not available in this run.",
                    }
                )
                continue
            prompt = str(item.get("prompt") or "").strip()
            if len(prompt) < 500:
                errors.append(
                    {
                        "index": str(index),
                        "researcher": short,
                        "error": "Researcher prompt must be at least 500 characters.",
                    }
                )
                continue
            normalized.append(
                {
                    "researcher": short,
                    "tool_name": tool_name,
                    "prompt": prompt,
                }
            )

        configured_budget = max(0, int(getattr(self.config, "researcher_administrator_max_tools_used", 0) or 0))
        if enforce_budget and configured_budget > 0 and len(normalized) > configured_budget:
            errors.append(
                {
                    "error": (
                        f"Requested {len(normalized)} researchers but this administrator "
                        f"is configured for at most {configured_budget} researcher calls."
                    )
                }
            )
            return [], errors
        return normalized, errors

    def _build_async_tools(self, tools_by_name: dict[str, Any], enabled_researchers: list[str]):
        enabled = set(enabled_researchers)

        def _compact_progress_event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
            tool_input = payload.get("tool_input")
            if isinstance(tool_input, dict):
                input_keys = sorted(str(key) for key in tool_input.keys())
            else:
                input_keys = []
            event: dict[str, Any] = {
                "event": str(event_type or ""),
                "tool": str(payload.get("tool") or ""),
                "ts": str(payload.get("tool_start_ts") or payload.get("tool_end_ts") or ""),
            }
            if input_keys:
                event["input_keys"] = input_keys
            if payload.get("duration_ms") is not None:
                event["duration_ms"] = int(payload.get("duration_ms") or 0)
            if payload.get("error"):
                event["error"] = str(payload.get("error") or "")[:300]
            for key in ("stage", "answer_chars", "running", "forced_answer"):
                if payload.get(key) is not None:
                    event[key] = payload.get(key)
            return event

        def _record_progress(job_id: str, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
            event = _compact_progress_event(event_type, payload)
            _async_record_task_progress(job_id, task_id, event)

        def _run_one(
            job_id: str,
            task_id: str,
            tool_name: str,
            prompt: str,
            save_artifacts: bool,
            semaphore: threading.Semaphore,
            cancel_event: threading.Event,
        ) -> dict[str, Any]:
            with semaphore:
                started_at = time.time()
                if cancel_event.is_set():
                    return {"researcher_tool": tool_name, "cancelled": True, "finished_at": started_at}
                if not _async_mark_task_running_or_cancelled(job_id, task_id, tool_name, started_at):
                    return {"researcher_tool": tool_name, "cancelled": True, "finished_at": started_at}
                log_token = set_log_context(
                    _chack_tool_progress_callback=lambda event_type, payload: _record_progress(
                        job_id,
                        task_id,
                        event_type,
                        payload,
                    )
                )
                cancel_token = set_cancellation_event(cancel_event)
                try:
                    output = self._invoke_tool_sync(
                        tools_by_name[tool_name],
                        {"prompt": prompt, "save_artifacts": bool(save_artifacts)},
                    )
                    if cancel_event.is_set() and str(output or "").startswith("ERROR:"):
                        return {
                            "researcher_tool": tool_name,
                            "output": output,
                            "cancelled": True,
                            "finished_at": time.time(),
                        }
                finally:
                    reset_cancellation_event(cancel_token)
                    reset_log_context(log_token)
                parsed = researcher_response_from_output(tool_name, output)
                result: dict[str, Any] = {
                    "researcher_tool": tool_name,
                    "output": output,
                    "finished_at": time.time(),
                }
                if parsed is not None:
                    result["parsed_response"] = parsed
                    result["tool_call_counts"] = parsed.get("tool_call_counts") or {}
                    result["total_tool_calls"] = parsed.get("total_tool_calls") or 0
                return result

        def _task_done(job_id: str, task_id: str, future) -> None:
            _async_mark_task_done(job_id, task_id, future)

        @function_tool(name_override="start_researchers_async")
        def start_researchers_async(
            requests_json: str,
            save_artifacts: bool = False,
            max_parallel: int = 4,
        ) -> str:
            """Queue one or more specialized researchers asynchronously and return a job id.

            Use this when a researcher may take a long time and you want to queue it,
            keep orchestrating, and later poll progress/results. This does not expose
            live chain-of-thought; while running, status is limited to started/running
            metadata plus recent tool telemetry events. Once finished, poll output
            includes parsed researcher JSON and code-added tool_call_counts.

            Args:
                requests_json: Compact JSON array of objects with researcher and prompt.
                    Each prompt must be detailed and at least 500 characters.
                save_artifacts: Preserve evidence folders for these researchers.
                max_parallel: Accepted for compatibility but currently ignored; this in-process
                    server runs one researcher at a time so evidence paths cannot race.

            Output: Compact JSON with async_started, job_id, task ids, and validation errors.
            """
            normalized, errors = self._normalize_researcher_requests(
                requests_json,
                enabled=enabled,
                tools_by_name=tools_by_name,
            )
            if not normalized:
                return _compact_json({"async_started": False, "errors": errors, "job_id": "", "tasks": []})
            job_id = f"research-job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            self._launched_async_job_ids.append(job_id)
            parallel_limit = 1
            semaphore = threading.Semaphore(parallel_limit)
            job: dict[str, Any] = {
                "job_id": job_id,
                "created_at": time.time(),
                "save_artifacts": bool(save_artifacts),
                "max_parallel": parallel_limit,
                "evidence_dir": os.environ.get("CHACK_RESEARCH_MASTER_DIR", "").strip()
                or os.environ.get("CHACK_RESEARCH_DATA_DIR", "").strip(),
                "completion_event": threading.Event(),
                "expected_task_count": len(normalized),
                "tasks": {},
            }
            task_rows: list[dict[str, str]] = []
            prepared_tasks: list[tuple[str, dict[str, str], threading.Event]] = []
            _async_job_store(job_id, job)

            # Register every task before submitting any future. A fast first future
            # must never make a partially populated job appear complete and leave
            # its completion event permanently set while later tasks still run.
            for index, row in enumerate(normalized):
                task_id = f"task-{index}-{uuid.uuid4().hex[:6]}"
                cancel_event = threading.Event()
                task = {
                    "task_id": task_id,
                    "researcher": row["researcher"],
                    "researcher_tool": row["tool_name"],
                    "status": "queued",
                    "created_at": time.time(),
                    "latest_action": "queued",
                    "last_activity_at": time.time(),
                    "cancel_event": cancel_event,
                }
                _async_register_task(job_id, task_id, task)
                prepared_tasks.append((task_id, row, cancel_event))
                task_rows.append(
                    {
                        "task_id": task_id,
                        "researcher": row["researcher"],
                        "researcher_tool": row["tool_name"],
                    }
                )

            for task_id, row, cancel_event in prepared_tasks:
                future = _async_submit(
                    _run_one,
                    job_id,
                    task_id,
                    row["tool_name"],
                    row["prompt"],
                    bool(save_artifacts),
                    semaphore,
                    cancel_event,
                )
                _async_set_task_future(job_id, task_id, future)
                future.add_done_callback(lambda fut, jid=job_id, tid=task_id: _task_done(jid, tid, fut))
            return _compact_json(
                {
                    "async_started": True,
                    "job_id": job_id,
                    "tasks": task_rows,
                    "errors": errors,
                    "max_parallel": parallel_limit,
                    "next_step": (
                        "Call poll_researchers_async with this job_id immediately. Then use completion-aware "
                        "wait_seconds=300-600 for ChatGPT Pro/Deep jobs, or 30-120 for ordinary researchers. "
                        "The wait returns early on completion. Tasks run one at a time in this process "
                        "to keep evidence folders isolated; queued/running for 1-2 minutes can be normal. "
                        "Cancel only when stale, irrelevant, or near the runtime limit."
                    ),
                }
            )

        @function_tool(name_override="poll_researchers_async")
        def poll_researchers_async(job_id: str, include_outputs: bool = True, wait_seconds: int = 0) -> str:
            """Poll an asynchronous researcher job.

            Args:
                job_id: The id returned by start_researchers_async.
                include_outputs: If true, include raw output/parsed JSON for completed tasks.
                    If false, return only compact status, tool counts, errors, and timings.
                wait_seconds: Optional completion-aware seconds to wait before polling,
                    clamped to 0-900. Use 300-600 for ChatGPT Pro/Deep browser jobs and
                    30-120 for ordinary researchers. The call returns early when every
                    task reaches a terminal state.

            Output: Compact JSON with job status, per-task status/latest_action/timing, and
            elapsed_seconds, idle_seconds since the last observed event, recent tool
            events/live call counts while running, plus completed researcher
            results/tool_call_counts when available.
            """
            job_key = str(job_id or "").strip()
            job = _async_job_snapshot(job_key)
            if not job:
                return _compact_json({"job_found": False, "job_id": job_id, "error": "Unknown async researcher job id."})
            wait = max(0, min(int(wait_seconds or 0), 900))
            wait_started = time.monotonic()
            if wait:
                initial_tasks = (job.get("tasks") or {}).values()
                already_complete = bool(job.get("tasks")) and all(
                    str(task.get("status") or "") in {"done", "error", "cancelled"}
                    for task in initial_tasks
                )
                if not already_complete:
                    _async_wait_for_completion(job_key, wait)
            waited = round(time.monotonic() - wait_started, 3)
            job = _async_job_snapshot(job_key)
            if not job:
                return _compact_json({"job_found": False, "job_id": job_id, "error": "Async researcher job disappeared while polling."})
            tasks = []
            now = time.time()
            for task_id, task in sorted((job.get("tasks") or {}).items()):
                row = {
                    "task_id": task_id,
                    "researcher": task.get("researcher", ""),
                    "researcher_tool": task.get("researcher_tool", ""),
                    "status": task.get("status", "unknown"),
                    "latest_action": task.get("latest_action", ""),
                    "elapsed_seconds": round(now - float(task.get("created_at") or now), 3),
                    "idle_seconds": round(now - float(task.get("last_activity_at") or task.get("created_at") or now), 3),
                }
                if task.get("error"):
                    row["error"] = task.get("error")
                result = task.get("result") or {}
                if result:
                    if result.get("tool_call_counts") is not None:
                        row["tool_call_counts"] = result.get("tool_call_counts") or {}
                    if result.get("total_tool_calls") is not None:
                        row["total_tool_calls"] = result.get("total_tool_calls") or 0
                    if include_outputs:
                        row["result"] = result
                else:
                    live_counts = task.get("live_tool_call_counts") or {}
                    if live_counts:
                        row["tool_call_counts"] = dict(sorted(live_counts.items()))
                        row["total_tool_calls"] = int(sum(int(v) for v in live_counts.values()))
                recent_events = task.get("recent_events") or []
                if recent_events:
                    row["recent_events"] = recent_events[-10:]
                tasks.append(row)
            statuses = [str(t.get("status") or "") for t in tasks]
            has_browser_researcher = any(
                str(t.get("researcher_tool") or "")
                in {"deepchatgpt_researcher", "prochatgpt_researcher", "chatgptxhigh"}
                for t in tasks
            )
            complete = bool(tasks) and all(s in {"done", "error", "cancelled"} for s in statuses)
            if complete:
                next_step = "Review completed researcher outputs/tool counts, then synthesize or launch focused follow-ups if material gaps remain."
            elif any(s == "running" for s in statuses):
                next_step = (
                    "Some researchers are running. Continue with completion-aware wait_seconds=300-600; cancel only duplicated, clearly stalled, or no-longer-useful tasks."
                    if has_browser_researcher else
                    "Some researchers are running. Continue polling with wait_seconds=30-120; cancel only duplicated, clearly stalled, or no-longer-useful tasks."
                )
            elif any(s == "queued" for s in statuses):
                next_step = "Researchers are still queued/starting. This can take a few minutes while child sessions initialize; use completion-aware waiting unless runtime is nearly exhausted."
            else:
                next_step = "No completed outputs yet. Keep using completion-aware polling or cancel failed/stale tasks if runtime is nearly exhausted."
            return _compact_json(
                {
                    "job_found": True,
                    "job_id": job.get("job_id", job_id),
                    "complete": complete,
                    "tasks": tasks,
                    "requested_wait_seconds": wait,
                    "waited_seconds": waited,
                    "next_step": next_step,
                }
            )

        @function_tool(name_override="cancel_researchers_async")
        def cancel_researchers_async(job_id: str) -> str:
            """Request cancellation for an asynchronous researcher job.

            Queued tasks are cancelled before they start. Running Codex/Claude
            subprocess trees are terminated when the backend has registered the
            process for this async task; otherwise cancellation remains best-effort
            until the tool/backend call returns or times out.

            Args:
                job_id: The id returned by start_researchers_async.

            Output: Compact JSON with cancelled task ids and tasks that were already running/done.
            """
            return _compact_json(_async_cancel_job(job_id))

        for tool in (start_researchers_async, poll_researchers_async, cancel_researchers_async):
            tool.description = f"{tool.description}\n\nOutput: Compact JSON."
        return [start_researchers_async, poll_researchers_async, cancel_researchers_async]

    def _build_subagent_tools(self, enabled_researchers: list[str]):
        if function_tool is None:
            raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")

        # Local import to avoid circular import from agents_toolset -> this module.
        from .agents_toolset import AgentsToolset

        # Force-enable exactly the researchers this administrator manages and make
        # sure orchestrator tools (subchack, another administrator) are never built.
        overrides: dict[str, Any] = {
            "subchack_enabled": False,
            "researcher_administrator_enabled": False,
        }
        for short in enabled_researchers:
            attr, _tool = RESEARCHER_REGISTRY[short]
            overrides[attr] = True
        sub_config = replace(self.config, **overrides)

        toolset = AgentsToolset(
            sub_config,
            model_provider=self.model_provider,
            default_model=self.fallback_model,
            social_network_model=self._model_for("social_network", self.social_network_model),
            scientific_model=self._model_for("scientific", self.scientific_model),
            websearcher_model=self._model_for("websearcher", self.websearcher_model),
            business_model=self._model_for("business", self.business_model),
            product_model=self._model_for("product", self.product_model),
            travel_model=self._model_for("travel", self.travel_model),
            legal_model=self._model_for("legal", self.legal_model),
            data_statistics_model=self._model_for("data_statistics", self.data_statistics_model),
            news_media_model=self._model_for("news_media", self.news_media_model),
            knowledge_graph_model=self._model_for("knowledge_graph", self.knowledge_graph_model),
            religious_model=self._model_for("religious", self.religious_model),
            cli_model=self._model_for("cli", self.cli_model),
            social_network_max_turns=self._max_turns_for("social_network", self.social_network_max_turns),
            scientific_max_turns=self._max_turns_for("scientific", self.scientific_max_turns),
            websearcher_max_turns=self._max_turns_for("websearcher", self.websearcher_max_turns),
            business_max_turns=self._max_turns_for("business", self.business_max_turns),
            product_max_turns=self._max_turns_for("product", self.product_max_turns),
            travel_max_turns=self._max_turns_for("travel", self.travel_max_turns),
            legal_max_turns=self._max_turns_for("legal", self.legal_max_turns),
            data_statistics_max_turns=self._max_turns_for("data_statistics", self.data_statistics_max_turns),
            news_media_max_turns=self._max_turns_for("news_media", self.news_media_max_turns),
            knowledge_graph_max_turns=self._max_turns_for("knowledge_graph", self.knowledge_graph_max_turns),
            religious_max_turns=self._max_turns_for("religious", self.religious_max_turns),
            cli_max_turns=self._max_turns_for("cli", self.cli_max_turns),
            self_critique_enabled=self._researcher_self_critique_enabled(),
            self_critique_rounds=self._researcher_self_critique_rounds(),
        )

        keep = {RESEARCHER_REGISTRY[short][1] for short in enabled_researchers}
        keep.add("task_steps_manager")
        tool_to_short = {RESEARCHER_REGISTRY[short][1]: short for short in enabled_researchers}
        all_tools = []
        for tool in (getattr(toolset, "tools", []) or []):
            name = self._name_of_tool(tool)
            if name not in keep:
                continue
            if name in tool_to_short:
                tool = self._guard_researcher_tool(tool, tool_to_short[name])
            all_tools.append(tool)

        tools_by_name = {self._name_of_tool(tool): tool for tool in all_tools}
        long_browser_researchers = {
            "deepchatgpt",
            "prochatgpt",
            "chatgptxhigh",
        }.intersection(enabled_researchers)
        synchronous_researchers = [short for short in enabled_researchers if short not in long_browser_researchers]
        synchronous_tool_names = {RESEARCHER_REGISTRY[short][1] for short in synchronous_researchers}

        # ChatGPT Pro/Extra High/Deep can run for 45-90 minutes. Never expose those direct
        # blocking tools (or a synchronous batch containing them) to the Codex
        # administrator, because Codex execution cells can be manually terminated
        # before the researcher reaches terminal output. Keep the direct tools
        # private inside the async wrapper and expose only start + poll. Also omit
        # cancellation for a job containing browser researchers; the outer user or
        # configured hard timeout remains the authoritative cancellation boundary.
        tools = [
            tool for tool in all_tools
            if self._name_of_tool(tool) == "task_steps_manager"
            or self._name_of_tool(tool) in synchronous_tool_names
        ]
        if synchronous_researchers:
            synchronous_tools = {
                name: tool for name, tool in tools_by_name.items()
                if name in synchronous_tool_names
            }
            tools.append(self._build_batch_tool(synchronous_tools, synchronous_researchers))
        async_tools = self._build_async_tools(tools_by_name, enabled_researchers)
        if long_browser_researchers:
            async_tools = [tool for tool in async_tools if self._name_of_tool(tool) != "cancel_researchers_async"]
        tools.extend(async_tools)
        add_research_artifact_tools(tools, self.config)
        return tools

    def _run_single(self, prompt: str, ctx: dict[str, Any], save_artifacts: bool = False) -> str:
        accounting = _AdministratorRunAccounting()
        token = _ADMINISTRATOR_RUN_ACCOUNTING.set(accounting)
        try:
            return self._run_single_scoped(prompt, ctx, save_artifacts=save_artifacts)
        finally:
            _ADMINISTRATOR_RUN_ACCOUNTING.reset(token)

    def _run_single_scoped(self, prompt: str, ctx: dict[str, Any], save_artifacts: bool = False) -> str:
        enabled_researchers = self._enabled_researchers()
        if not enabled_researchers:
            return (
                "ERROR: researcher_administrator has no researchers enabled. "
                "Enable researchers in tools.researcher_administrator_researchers or at the top level."
            )
        tools = self._build_subagent_tools(enabled_researchers)
        if not tools:
            return "ERROR: no researcher tools available for researcher_administrator."
        model_name = self._resolved_model() or ""
        launch_block = subagent_launch_block_reason(
            parent_original_runtime_minutes=int(ctx.get("max_runtime_minutes") or 0),
            parent_remaining_runtime_minutes=float(ctx.get("remaining_runtime_minutes") or 0.0),
            parent_original_cost_usd=float(ctx.get("max_cost_usd") or 0.0),
            parent_remaining_cost_usd=float(ctx.get("remaining_cost_usd") or 0.0),
        )
        if launch_block:
            return launch_block
        effective_max_turns, effective_runtime_minutes, effective_cost_usd = inherit_subagent_limits(
            default_max_turns=self.max_turns,
            parent_max_turns=int(ctx.get("max_turns") or 0),
            parent_remaining_runtime_minutes=float(ctx.get("remaining_runtime_minutes") or 0.0),
            parent_remaining_cost_usd=float(ctx.get("remaining_cost_usd") or 0.0),
            runtime_ratio=1.0,
            runtime_cap_minutes=90,
            cost_ratio=1.0,
        )
        parent_memory_max_messages = max(1, int(ctx.get("memory_max_messages") or 8))
        parent_memory_reset_to_messages = max(1, int(ctx.get("memory_reset_to_messages") or parent_memory_max_messages))
        parent_root_session_id = str(ctx.get("session_id") or "").strip()

        requested_master_dir = str(ctx.get("research_master_dir") or "").strip()
        if requested_master_dir:
            master_dir = requested_master_dir
            os.makedirs(master_dir, exist_ok=True)
        else:
            master_dir = create_research_master_dir(parent_root_session_id)
        # Pre-create one subfolder per researcher type. The prompt can describe
        # the deterministic <master>/<researcher> layout without listing every
        # absolute path separately.
        for short in enabled_researchers:
            subfolder = os.path.join(master_dir, short)
            os.makedirs(subfolder, exist_ok=True)
        available_line = ", ".join(self._name_of_tool(tool) for tool in tools)
        capability_lines = self._researcher_capability_lines(enabled_researchers)
        chatgpt_priority_line = self._chatgpt_priority_instruction(enabled_researchers)
        admin_tool_budget = max(0, int(getattr(self.config, "researcher_administrator_max_tools_used", 0) or 0))
        admin_runtime_tool_cap = (admin_tool_budget * 4 + 8) if admin_tool_budget > 0 else 0
        budget_line = (
            f"Researcher-call budget: {admin_tool_budget} launches (hard cap; management polls/status do not count). "
            "With a budget of 3 or fewer, repeat a researcher only after complete failure.\n"
            if admin_tool_budget > 0
            else "Researcher-call budget: no configured cap; still avoid low-value repeats.\n"
        )
        master_line = (
            f"Evidence workspace (preserved; runtime appends the path): {master_dir}\n"
            if save_artifacts
            else f"Evidence workspace (temporary; do not report its path): {master_dir}\n"
        )
        required_line = (
            "Required successful researchers: "
            + ", ".join(self.required_researchers)
            + ". Launch and await each; `research_worked` must be false if any fails.\n"
            if self.required_researchers
            else ""
        )

        admin_context = (
            "\n\n### RUN CONFIGURATION\n"
            f"Available tools: {available_line}. Do not call anything absent from this list.\n"
            f"{chatgpt_priority_line}"
            f"{required_line}"
            f"{budget_line}"
            f"{master_line}"
            "Researchers share evidence in `<workspace>/<researcher-short-name>`; use artifact list/read/grep tools when inspection is useful.\n"
            "Researcher capabilities:\n"
            + "\n".join(capability_lines)
            + "\nExecution is serialized. Use `run_researchers_batch` for short work and "
            "`start_researchers_async`/`poll_researchers_async` for long work. Polling: ordinary "
            "30-120s; ChatGPT browser 300-600s, completion-aware. "
            f"Children run try-harder self-critique for {self._researcher_self_critique_rounds()} round(s). "
            "Compare `tool_call_counts` with the capabilities above; request a focused follow-up only for a material missing source/tool family. "
            "A repeated researcher prompt must include `Duplicate reason:` with at least 80 characters.\n"
            "Now plan the research and launch the needed researchers."
        )
        prompt = f"{str(prompt or '').rstrip()}{admin_context}"

        overrides = {
            "agent": {
                "self_critique_enabled": self.self_critique_enabled,
                "self_critique_rounds": self.self_critique_rounds,
                "output_schema_json": researcher_administrator_output_schema(
                    preserve_artifacts=save_artifacts
                ),
                "output_schema_name": "researcher_administrator_result",
                "output_schema_strict": True,
            },
            "session": {
                "max_turns": effective_max_turns,
                "memory_max_messages": parent_memory_max_messages,
                "memory_reset_to_messages": parent_memory_reset_to_messages,
                "long_term_memory_enabled": False,
            },
            "tools": {
                "researcher_administrator_enabled": True,
                "max_tools_used": admin_runtime_tool_cap,
            },
            "env": {
                "CHACK_RESEARCH_MASTER_DIR": master_dir,
                "CHACK_RESEARCH_DATA_DIR": master_dir,
                "CHACK_RESEARCH_SAVE_ARTIFACTS": "1" if save_artifacts else "0",
            },
        }
        overrides["agent"]["max_runtime_minutes"] = effective_runtime_minutes
        overrides["agent"]["max_cost_usd"] = effective_cost_usd
        main_action = str(ctx.get("main_action") or "").strip()
        if main_action:
            overrides["agent"]["main_action"] = main_action
        overrides["agent"]["sub_action"] = "researcher_administrator"
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=_ADMINISTRATOR_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("researcher_administrator", parent_root_session_id)

        prev_master = os.environ.get("CHACK_RESEARCH_MASTER_DIR")
        from chack_agent import Chack
        chack = Chack(config)
        artifact_context_tokens = set_research_artifact_context(master_dir, master_dir)
        collector_token, researcher_responses = begin_researcher_response_collection()
        try:
            try:
                result = chack.run(
                    session_id=subagent_session_id,
                    text=prompt,
                    min_tools_used_override=0,
                    max_tools_used_override=admin_runtime_tool_cap,
                    enable_self_critique=None,
                    require_task_steps_manager_init_first=bool(
                        getattr(self.config, "task_steps_manager_enabled", True)
                    ),
                    tools_override=tools,
                    system_prompt_override=config.system_prompt,
                    usage_session_id=parent_task_session_id,
                )
            except Exception as exc:
                combined_responses = list(researcher_responses or [])
                combined_responses.extend(_researcher_responses_from_async_jobs(self._launched_async_job_ids))
                combined_responses.extend(_researcher_responses_from_async_output_files(master_dir))
                combined_failures = _researcher_failures_from_async_jobs(self._launched_async_job_ids)
                combined_tool_counts = _researcher_call_counts_from_async_jobs(self._launched_async_job_ids)
                combined_tool_counts.update(self._launched_researcher_tool_counts())
                failure_payload = {
                    "research_worked": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "administrator_conclusions": "",
                }
                return finalize_researcher_administrator_output(
                    _compact_json(failure_payload),
                    evidence_dir=master_dir,
                    save_artifacts=save_artifacts,
                    researcher_responses=combined_responses,
                    researcher_failures=combined_failures,
                    tool_counts=combined_tool_counts,
                    steps=[],
                    required_researchers=self.required_researchers,
                )
            output = result.output.strip() if result.output else "ERROR: sub-agent returned an empty response."
            if output.startswith("ERROR:"):
                combined_responses = list(researcher_responses or [])
                combined_responses.extend(_researcher_responses_from_async_jobs(self._launched_async_job_ids))
                combined_responses.extend(_researcher_responses_from_async_output_files(master_dir))
                combined_failures = _researcher_failures_from_async_jobs(self._launched_async_job_ids)
                combined_tool_counts = _researcher_call_counts_from_async_jobs(self._launched_async_job_ids)
                combined_tool_counts.update(self._launched_researcher_tool_counts())
                failure_payload = {
                    "research_worked": False,
                    "failure_reason": output,
                    "administrator_conclusions": "",
                }
                return finalize_researcher_administrator_output(
                    _compact_json(failure_payload),
                    evidence_dir=master_dir,
                    save_artifacts=save_artifacts,
                    researcher_responses=combined_responses,
                    researcher_failures=combined_failures,
                    tool_counts=combined_tool_counts,
                    steps=result.all_steps,
                    required_researchers=self.required_researchers,
                )
            tool_counts = result.tool_counts.copy()
            tool_counts.update(_researcher_call_counts_from_async_jobs(self._launched_async_job_ids))
            tool_counts.update(self._launched_researcher_tool_counts())
            combined_responses = list(researcher_responses or [])
            combined_responses.extend(_researcher_responses_from_async_jobs(self._launched_async_job_ids))
            combined_responses.extend(_researcher_responses_from_async_output_files(master_dir))
            combined_failures = _researcher_failures_from_async_jobs(self._launched_async_job_ids)
            return finalize_researcher_administrator_output(
                output,
                evidence_dir=master_dir,
                save_artifacts=save_artifacts,
                researcher_responses=combined_responses,
                researcher_failures=combined_failures,
                tool_counts=tool_counts,
                steps=result.all_steps,
                required_researchers=self.required_researchers,
            )
        finally:
            end_researcher_response_collection(collector_token)
            # A timed-out administrator does not own the lifetime of executor
            # threads it launched. Preserve their evidence root until every task is
            # terminal instead of deleting a directory that a live browser worker
            # is still updating.
            if not _async_jobs_have_nonterminal_tasks(self._launched_async_job_ids):
                cleanup_research_artifacts(master_dir, save_artifacts=save_artifacts)
            reset_research_artifact_context(artifact_context_tokens)
            # Restore the inherited master dir so standalone researchers launched
            # later in the same process are not accidentally nested under it.
            if prev_master is None:
                os.environ.pop("CHACK_RESEARCH_MASTER_DIR", None)
            else:
                os.environ["CHACK_RESEARCH_MASTER_DIR"] = prev_master

    def run(self, prompt: str | list[str], save_artifacts: bool = False) -> str:
        # A single administrator owns one master evidence folder, so it only
        # accepts a single research request per call.
        prompts, error = normalize_subagent_prompts(prompt, min_chars=500, max_prompts=1)
        if error:
            return error
        ctx = current_log_context()
        return self._run_single(prompts[0], ctx, save_artifacts=save_artifacts)


def get_researcher_administrator_tool(
    helper: ResearcherAdministratorAgentTool,
):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="researcher_administrator")
    def researcher_administrator(prompt: str, save_artifacts: bool = False) -> str:
        """Run a research administrator that orchestrates every specialized researcher for you.

        Use this tool to delegate a whole research problem to an administrator sub-agent instead of
        calling each `*_research` researcher yourself. The administrator decomposes the request,
        launches all the relevant researchers (and relaunches them with cross-pollinated leads so no
        source is missed), reviews everything they return, and reports back:
        - its own synthesized conclusions,
        - an appended array with the exact JSON responses returned by every researcher,
        - appended aggregate tool-call counts from the researchers and counts of researcher calls,
        - when save_artifacts is true, the path of the preserved master evidence folder.

        Args:
            prompt: A single detailed research request of at least 500 characters describing the goal,
                scope, entities, timeframes, sources to prioritize, expected output, and caveats.
            save_artifacts: If true, preserve the master evidence folder after the run and return its
                path in the JSON result. If false, artifacts are deleted after the run.

        Output: Returns compact administrator JSON with worked status and conclusions. Runtime code
        appends researcher_responses, researcher_tool_call_counts, researcher_call_counts, and the
        master evidence folder path when preserved.
        """
        try:
            return run_with_tool_logging(
                "researcher_administrator",
                {"prompt": prompt, "save_artifacts": save_artifacts},
                lambda: helper.run(prompt=prompt, save_artifacts=save_artifacts),
            )
        except Exception as exc:
            return f"ERROR: researcher_administrator failed ({exc})"

    tool = researcher_administrator
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: Provide prompt as one detailed research request (>=500 chars); set save_artifacts true only when the master evidence folder must be preserved.\n"
        "Output: Returns compact administrator JSON plus code-appended researcher_responses, tool counts, researcher call counts, and the master evidence folder path when preserved."
    )
    return tool
