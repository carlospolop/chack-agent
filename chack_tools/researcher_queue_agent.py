"""Shared research queue.

Several agents can call the ``researcher_queue`` tool at the same time. Each call
is BLOCKING: it submits 1..N research requests and waits until the batch it joined
has finished. Requests are collected for a short window (or until a configured
number of participants have joined), near-duplicate requests are merged by a small
merge agent, each merged request is researched once by a ``ResearcherAdministrator``,
and every caller receives the researches relevant to the request(s) it submitted.

Sharing works in two ways with the exact same in-memory queue:
- Same process: several chacks running as threads (e.g. the factchecker verifiers)
  share the module-level ``RESEARCHER_QUEUE`` singleton directly.
- MCP service: run one long-lived ``chack_tools`` MCP server (streamable-http); every
  external client that connects hits the same process and therefore the same queue.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from .config import ToolsConfig
from .researcher_administrator_agent import ResearcherAdministratorAgentTool
from .subagent_config import (
    build_subagent_config,
    enforce_prompt_str_or_list_schema,
    normalize_subagent_prompts,
)
from .telemetry import reset_log_context, run_with_tool_logging, set_log_context

try:
    from agents import function_tool
except ImportError:  # pragma: no cover - import guard mirrors the other tools
    function_tool = None


# ── Merge agent contract (kept intentionally small: a shallow schema fails less) ──
_MERGE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "groups": {
            "type": "array",
            "description": "Groups of the input requests that should be researched together.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "One self-contained merged research prompt covering every member request in this group.",
                    },
                    "members": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The request numbers (from the input list) covered by this group.",
                    },
                    "merge_reason": {
                        "type": "string",
                        "description": "Brief reason why these requests belong together, or why this request stayed separate.",
                    },
                },
                "required": ["prompt", "members", "merge_reason"],
            },
        }
    },
    "required": ["groups"],
}

_MERGE_SYSTEM_PROMPT = """### RESEARCHER SPECIALIZATION
Your goal is to receive several research requests and merge overlapping research requests before they are dispatched, so duplicated work is avoided.
It's extremely important to:
    - Only merge research requests that are truly related
    - Don't lose the details requested in the research requests when merging them:
        - E.g. if 2 research requests ask for reasons in favour and against a topic, you can mix these in 1 research, but the research must explicitly ask for reasons in favour and against the topic, not just generic research about the topic
        - E.g. if 2 research requests ask about specific different details of the same product, mix this into 1 research still specifying to research about both details and not just to research about the product

So group together the requests that ask for the same or very similar research, and for each group write ONE self-contained merged research prompt that fully covers every member request of that group without losing the details.
Keep genuinely different requests in separate groups.
Every request number must appear in exactly one group. Do not research anything yourself and do not call any tool.
Return only the JSON object described by the schema.
"""


def _compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _clean_researcher_call_counts(value: Any) -> dict[str, int]:
    """Return a stable, positive-only researcher call-count mapping."""
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[name] = counts.get(name, 0) + count
    return dict(sorted(counts.items()))


def _researcher_usage_for(researches: Any) -> dict[str, Any]:
    """Aggregate one administrator per research plus its private researchers.

    ``complete`` is false whenever an administrator failed before returning its
    structured call ledger. Callers can therefore show known counts without
    presenting partial accounting as exact.
    """
    rows = [row for row in (researches or []) if isinstance(row, dict)]
    researcher_counts: dict[str, int] = {}
    complete = True
    for row in rows:
        raw_counts = row.get("researcher_call_counts")
        if not isinstance(raw_counts, dict) or row.get("researcher_usage_complete") is False:
            complete = False
            continue
        for name, count in _clean_researcher_call_counts(raw_counts).items():
            researcher_counts[name] = researcher_counts.get(name, 0) + count
    researcher_counts = dict(sorted(researcher_counts.items()))
    return {
        "administrator_calls": len(rows),
        "researcher_call_counts": researcher_counts,
        "total_researcher_calls": int(sum(researcher_counts.values())),
        "complete": complete,
    }


def _json_loads(output: Any) -> Optional[dict]:
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
        return None
    return payload if isinstance(payload, dict) else None


def _topic_of(prompt: str) -> str:
    text = " ".join(str(prompt or "").split())
    return text[:100] + ("…" if len(text) > 100 else "")


def _safe_path_part(value: str, fallback: str = "queue") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text or fallback


def _queue_root_for_id(queue_id: str) -> str:
    safe_id = _safe_path_part(queue_id, "queue")
    path = Path("/tmp") / "chack-research-data" / "researcher-queues" / safe_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# Env var that supplies a default queue_id when a caller does not pass one. Lets a
# host process (e.g. the factchecker) make every researcher_queue call in the
# process share one logical queue and one shared evidence folder.
QUEUE_ID_ENV = "CHACK_RESEARCHER_QUEUE_ID"


def queue_evidence_root_for_id(queue_id: str) -> str:
    """Absolute path of the shared evidence folder for a logical queue id.

    Every research submitted under this queue id writes its files under here, so a
    host can root browse-only file tools at this folder for all its agents.
    """
    return _queue_root_for_id(queue_id)


def _write_json_file(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_compact(payload), encoding="utf-8")


def _copy_or_link_research_folder(source: str, target: str) -> None:
    src = Path(str(source or "")).expanduser()
    if not src.exists():
        return
    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel, target_is_directory=src.is_dir())
    except Exception:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dst)


# ────────────────────────────── the queue itself ──────────────────────────────
class _QueueBatch:
    __slots__ = (
        "id",
        "queue_id",
        "queue_root",
        "prompts",
        "calls",
        "closed",
        "result",
        "done",
        "processor",
        "opened_at",
        "save_artifacts",
        "waiters",
        "window_seconds",
        "expected_participants",
        "max_batch_requests",
        "max_wait_seconds",
    )

    def __init__(self, batch_id: str, queue_id: str, queue_root: str) -> None:
        self.id = batch_id
        self.queue_id = queue_id
        self.queue_root = queue_root
        self.prompts: list[str] = []
        self.calls = 0
        self.closed = False
        self.result: Optional[str] = None
        self.done = threading.Event()
        self.processor: Optional[Callable[[list[str], bool], str]] = None
        self.opened_at = 0.0
        self.save_artifacts = False
        self.waiters: list[_QueueWaiter] = []
        self.window_seconds = 0
        self.expected_participants = 0
        self.max_batch_requests = 0
        self.max_wait_seconds = 0


class _QueueWaiter:
    __slots__ = ("start", "end", "done", "result", "save_artifacts", "request_id", "request_dir")

    def __init__(self, start: int, end: int, save_artifacts: bool, request_id: str = "", request_dir: str = "") -> None:
        self.start = start
        self.end = end
        self.done = threading.Event()
        self.result: Optional[str] = None
        self.save_artifacts = bool(save_artifacts)
        self.request_id = str(request_id or "").strip()
        self.request_dir = str(request_dir or "").strip()


def _waiter_status(waiter: _QueueWaiter) -> dict[str, Any]:
    return {
        "request_id": waiter.request_id,
        "prompt_start": waiter.start,
        "prompt_end": waiter.end,
        "prompt_count": max(0, waiter.end - waiter.start),
        "save_artifacts": bool(waiter.save_artifacts),
        "request_evidence_data_path": waiter.request_dir,
    }


class ResearcherQueue:
    """In-memory, thread-safe batching queue shared across a process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[_QueueBatch] = None
        self._current_by_key: dict[str, _QueueBatch] = {}
        self._queue_roots: dict[str, str] = {}
        self._counter = 0
        self._processing: dict[str, dict[str, Any]] = {}

    def create_queue(self, queue_id: str = "") -> str:
        raw = str(queue_id or "").strip() or f"queue-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        safe = _safe_path_part(raw, "queue")
        root = _queue_root_for_id(safe)
        with self._lock:
            self._queue_roots[safe] = root
        _write_json_file(
            Path(root) / "queue_metadata.json",
            {
                "queue_id": safe,
                "queue_evidence_data_path": root,
                "created_at": time.time(),
            },
        )
        return _compact({"queue_id": safe, "queue_evidence_data_path": root})

    def submit_and_wait(
        self,
        prompts: list[str],
        *,
        processor: Callable[[list[str], bool], str],
        window_seconds: int,
        expected_participants: int,
        max_batch_requests: int,
        max_wait_seconds: int,
        save_artifacts: bool = False,
        queue_id: str = "",
    ) -> str:
        flush_payload: Optional[
            tuple[
                list[str],
                Optional[Callable[[list[str], bool], str]],
                bool,
                list[_QueueWaiter],
            ]
        ] = None
        preflush_payload: Optional[
            tuple[
                _QueueBatch,
                list[str],
                Optional[Callable[[list[str], bool], str]],
                bool,
                list[_QueueWaiter],
            ]
        ] = None
        start_timer = False
        queue_key = _safe_path_part(queue_id, "__default__") if str(queue_id or "").strip() else "__default__"
        with self._lock:
            batch = self._current_by_key.get(queue_key)
            opening = batch is None or batch.closed
            if (
                not opening
                and max_batch_requests > 0
                and batch is not None
                and batch.prompts
                and len(batch.prompts) + len(prompts) > max_batch_requests
            ):
                preclosed = self._close_batch_locked(batch)
                if preclosed is not None:
                    preflush_payload = (batch, *preclosed)
                opening = True
            if opening:
                self._counter += 1
                batch_id = f"batch-{self._counter}-{uuid.uuid4().hex[:8]}"
                if queue_key == "__default__":
                    queue_root = _queue_root_for_id(f"{batch_id}-{uuid.uuid4().hex[:6]}")
                    effective_queue_id = batch_id
                else:
                    queue_root = self._queue_roots.get(queue_key) or _queue_root_for_id(queue_key)
                    self._queue_roots[queue_key] = queue_root
                    effective_queue_id = queue_key
                batch = _QueueBatch(batch_id, effective_queue_id, queue_root)
                batch.opened_at = time.time()
                # The opener's processor runs the whole batch's shared work.
                batch.processor = processor
                batch.window_seconds = max(0, int(window_seconds or 0))
                batch.expected_participants = max(0, int(expected_participants or 0))
                batch.max_batch_requests = max(0, int(max_batch_requests or 0))
                batch.max_wait_seconds = max(1, int(max_wait_seconds or 1))
                self._current_by_key[queue_key] = batch
                if queue_key == "__default__":
                    self._current = batch
            start = len(batch.prompts)
            batch.prompts.extend(prompts)
            end = len(batch.prompts)
            request_id = f"request-{batch.calls + 1}-{uuid.uuid4().hex[:8]}"
            request_dir = os.path.join(batch.queue_root, "requests", request_id)
            os.makedirs(request_dir, exist_ok=True)
            _write_json_file(
                Path(request_dir) / "request_prompts.json",
                {
                    "queue_id": batch.queue_id,
                    "batch_id": batch.id,
                    "request_id": request_id,
                    "prompt_indices": list(range(start, end)),
                    "prompts": list(prompts),
                    "save_artifacts": bool(save_artifacts),
                },
            )
            waiter = _QueueWaiter(start, end, save_artifacts=save_artifacts, request_id=request_id, request_dir=request_dir)
            batch.waiters.append(waiter)
            batch.calls += 1
            batch.save_artifacts = bool(batch.save_artifacts or save_artifacts)
            flush_now = (
                (expected_participants > 0 and batch.calls >= expected_participants)
                or (max_batch_requests > 0 and len(batch.prompts) >= max_batch_requests)
                or (opening and max(0, int(window_seconds or 0)) == 0)
            )
            if flush_now:
                flush_payload = self._close_batch_locked(batch)
            elif opening:
                start_timer = True

        if preflush_payload is not None:
            threading.Thread(
                target=self._process_closed_batch,
                args=preflush_payload,
                daemon=True,
            ).start()

        if flush_payload is not None:
            threading.Thread(
                target=self._process_closed_batch,
                args=(batch, *flush_payload),
                daemon=True,
            ).start()
        elif start_timer:
            threading.Thread(
                target=self._timer_flush, args=(batch, window_seconds), daemon=True
            ).start()

        if not waiter.done.wait(timeout=max(1, int(max_wait_seconds or 1))):
            artifacts_preserved = bool(batch.save_artifacts)
            return _compact(
                {
                    "batch_id": batch.id,
                    "queue_id": batch.queue_id,
                    "queue_evidence_data_path": batch.queue_root if artifacts_preserved else "",
                    "request_id": waiter.request_id,
                    "request_evidence_data_path": waiter.request_dir if artifacts_preserved else "",
                    "researches": [],
                    "count": 0,
                    "artifacts_preserved": artifacts_preserved,
                    "error": "researcher_queue timed out waiting for the batch to finish",
                }
            )
        return waiter.result or _compact({"researches": [], "count": 0})

    def _timer_flush(self, batch: _QueueBatch, window_seconds: int) -> None:
        remaining = (batch.opened_at + max(0, int(window_seconds or 0))) - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self._flush(batch)

    def _close_batch_locked(
        self, batch: _QueueBatch
    ) -> Optional[
        tuple[
            list[str],
            Optional[Callable[[list[str], bool], str]],
            bool,
            list[_QueueWaiter],
        ]
    ]:
        if batch.closed:
            return None
        batch.closed = True
        for key, current in list(self._current_by_key.items()):
            if current is batch:
                self._current_by_key.pop(key, None)
        if self._current is batch:
            self._current = None
        return list(batch.prompts), batch.processor, bool(batch.save_artifacts), list(batch.waiters)

    def _flush(self, batch: _QueueBatch) -> None:
        with self._lock:
            payload = self._close_batch_locked(batch)
            if payload is None:
                return
        self._process_closed_batch(batch, *payload)

    def _process_closed_batch(
        self,
        batch: _QueueBatch,
        prompts: list[str],
        processor: Optional[Callable[[list[str], bool], str]],
        save_artifacts: bool,
        waiters: list[_QueueWaiter],
    ) -> None:
        with self._lock:
            self._processing[batch.id] = {
                "id": batch.id,
                "prompt_count": len(prompts),
                "caller_count": len(waiters),
                "started_at": time.time(),
                "latest_action": "processing batch",
                "current_research_index": 0,
                "current_research_count": 0,
                "recent_events": [],
                "requests": [_waiter_status(waiter) for waiter in waiters],
                "save_artifacts": bool(save_artifacts),
                "max_wait_seconds": batch.max_wait_seconds,
                "queue_id": batch.queue_id,
                "queue_evidence_data_path": batch.queue_root,
            }

        def progress(update: dict[str, Any]) -> None:
            with self._lock:
                row = self._processing.get(batch.id)
                if not row:
                    return
                events = update.pop("recent_events", None)
                row.update(update)
                if isinstance(events, list):
                    recent = row.setdefault("recent_events", [])
                    recent.extend(events)
                    if len(recent) > 30:
                        del recent[:-30]

        try:
            if processor is None:
                result = _compact({"researches": [], "count": 0})
            else:
                result = self._call_processor(processor, prompts, save_artifacts, batch.queue_root, batch.id, progress)
        except Exception as exc:  # never leave a waiter hung
            result = _compact(
                {"researches": [], "count": 0, "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            with self._lock:
                self._processing.pop(batch.id, None)
        for waiter in waiters:
            waiter.result = self._filter_result_for_waiter(
                result,
                waiter,
                batch_id=batch.id,
                artifacts_preserved=save_artifacts,
                queue_root=batch.queue_root,
            )
            waiter.done.set()
        if not save_artifacts:
            shutil.rmtree(batch.queue_root, ignore_errors=True)

    @staticmethod
    def _call_processor(
        processor: Callable[..., str],
        prompts: list[str],
        save_artifacts: bool,
        queue_root: str,
        batch_id: str,
        progress: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        try:
            return processor(prompts, save_artifacts, queue_root, batch_id, progress)
        except TypeError as exc:
            try:
                return processor(prompts, save_artifacts, queue_root, batch_id)
            except TypeError:
                try:
                    return processor(prompts, save_artifacts)
                except TypeError:
                    raise exc

    @staticmethod
    def _filter_result_for_waiter(
        result: str,
        waiter: _QueueWaiter,
        *,
        batch_id: str,
        artifacts_preserved: bool,
        queue_root: str = "",
    ) -> str:
        payload = _json_loads(result)
        if not isinstance(payload, dict):
            return result
        if payload.get("error"):
            return _compact(payload)
        researches = payload.get("researches")
        if not isinstance(researches, list):
            return _compact(payload)
        filtered: list[dict[str, Any]] = []
        for item in researches:
            if not isinstance(item, dict):
                continue
            raw_members = item.get("members")
            if not isinstance(raw_members, list):
                filtered.append(dict(item))
                continue
            absolute_members: list[int] = []
            for raw in raw_members:
                try:
                    idx = int(raw)
                except (TypeError, ValueError):
                    continue
                absolute_members.append(idx)
            matched = [idx for idx in absolute_members if waiter.start <= idx < waiter.end]
            if not matched:
                continue
            row = {k: v for k, v in item.items() if k != "members"}
            if not waiter.save_artifacts:
                row.pop("evidence_data_path", None)
            row["matched_request_indices"] = [idx - waiter.start for idx in matched]
            filtered.append(row)
        if artifacts_preserved and waiter.save_artifacts and waiter.request_dir:
            usage = _researcher_usage_for(filtered)
            _write_json_file(
                Path(waiter.request_dir) / "matched_researches.json",
                {
                    "batch_id": batch_id,
                    "queue_evidence_data_path": queue_root,
                    "request_evidence_data_path": waiter.request_dir,
                    "researches": filtered,
                    "count": len(filtered),
                    "researcher_usage": usage,
                },
            )
            for idx, item in enumerate(filtered):
                evidence_path = str(item.get("evidence_data_path") or "").strip()
                if evidence_path:
                    _copy_or_link_research_folder(
                        evidence_path,
                        str(Path(waiter.request_dir) / "researches" / f"research-{idx:03d}"),
                    )
        return _compact(
            {
                "batch_id": batch_id,
                **(
                    {
                        "queue_evidence_data_path": queue_root,
                        "request_evidence_data_path": waiter.request_dir,
                    }
                    if artifacts_preserved and waiter.save_artifacts
                    else {}
                ),
                "researches": filtered,
                "count": len(filtered),
                "researcher_usage": _researcher_usage_for(filtered),
                "artifacts_preserved": bool(artifacts_preserved and waiter.save_artifacts),
            }
        )

    def status(self) -> str:
        now = time.time()
        with self._lock:
            open_batches = []
            for current in self._current_by_key.values():
                if current is None or current.closed:
                    continue
                open_batches.append({
                    "id": current.id,
                    "queue_id": current.queue_id,
                    "age_seconds": round(max(0.0, now - current.opened_at), 3),
                    "prompt_count": len(current.prompts),
                    "caller_count": current.calls,
                    "save_artifacts": bool(current.save_artifacts),
                    "window_seconds": current.window_seconds,
                    "expected_participants": current.expected_participants,
                    "max_batch_requests": current.max_batch_requests,
                    "max_wait_seconds": current.max_wait_seconds,
                    "queue_evidence_data_path": current.queue_root,
                    "requests": [_waiter_status(waiter) for waiter in current.waiters],
                })
            open_batch = open_batches[0] if len(open_batches) == 1 else None
            processing = []
            for row in self._processing.values():
                item = dict(row)
                started_at = float(item.pop("started_at", now) or now)
                item["age_seconds"] = round(max(0.0, now - started_at), 3)
                processing.append(item)
        return _compact(
            {
                "open_batch": open_batch,
                "open_batches": open_batches,
                "processing_batches": processing,
                "processing_count": len(processing),
            }
        )


# Process-wide singleton, shared by every caller in this process.
RESEARCHER_QUEUE = ResearcherQueue()


# ─────────────────────────────── the tool helper ───────────────────────────────
class ResearcherQueueAgentTool:
    def __init__(
        self,
        administrator: ResearcherAdministratorAgentTool,
        *,
        config: ToolsConfig,
        model_provider: str,
        merge_model: str = "",
        fallback_model: str = "",
        window_seconds: int = 300,
        expected_participants: int = 0,
        max_requests_per_call: int = 5,
        max_batch_requests: int = 20,
        max_wait_seconds: int = 5400,
        min_prompt_chars: int = 200,
        queue_id: str = "",
        queue: Optional[ResearcherQueue] = None,
    ) -> None:
        self.admin = administrator
        self.config = config
        # Per-instance default queue id (usually the host's per-job id). Keeps
        # concurrent jobs in one process isolated without a process-global env var.
        self.queue_id = str(queue_id or "").strip()
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.merge_model = str(merge_model or "").strip()
        self.fallback_model = str(fallback_model or "").strip()
        self.window_seconds = max(0, int(window_seconds or 0))
        self.expected_participants = max(0, int(expected_participants or 0))
        self.max_requests_per_call = max(1, int(max_requests_per_call or 5))
        self.max_batch_requests = max(0, int(max_batch_requests or 0))
        self.max_wait_seconds = max(60, int(max_wait_seconds or 5400))
        self.min_prompt_chars = max(1, int(min_prompt_chars or 200))
        self.queue = queue or RESEARCHER_QUEUE

    # -- public entrypoint (blocks until the batch finishes) --
    def run(self, prompt: str | list[str], save_artifacts: bool = False, queue_id: str = "") -> str:
        prompts, error = normalize_subagent_prompts(
            prompt, min_chars=self.min_prompt_chars, max_prompts=self.max_requests_per_call
        )
        if error:
            return error
        return self.queue.submit_and_wait(
            prompts,
            processor=self._process_batch,
            window_seconds=self.window_seconds,
            expected_participants=self.expected_participants,
            max_batch_requests=self.max_batch_requests,
            max_wait_seconds=self.max_wait_seconds,
            save_artifacts=save_artifacts,
            queue_id=queue_id,
        )

    # -- batch worker: merge, then one administrator per merged request --
    def _process_batch(
        self,
        prompts: list[str],
        save_artifacts: bool = False,
        queue_root: str = "",
        batch_id: str = "",
        progress: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        try:
            groups = self._merge_prompts(list(prompts))
        except Exception:
            groups = [(p, [i], "merge failed; dispatched this request separately") for i, p in enumerate(prompts)]
        if progress:
            progress(
                {
                    "latest_action": f"merged {len(prompts)} prompt(s) into {len(groups)} research request(s)",
                    "current_research_index": 0,
                    "current_research_count": len(groups),
                }
            )
        ctx = self._queue_research_context()
        researches: list[dict[str, str]] = []
        created_standalone_root = False
        if not queue_root:
            queue_root = _queue_root_for_id(f"standalone-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}")
            created_standalone_root = True
        _write_json_file(
            Path(queue_root) / "batch_prompts.json",
            {
                "batch_id": batch_id,
                "prompts": list(prompts),
                "save_artifacts": bool(save_artifacts),
            },
        )
        def _run_one_cluster(index: int, group: Any) -> dict[str, Any]:
            merged_prompt, members, merge_reason = self._normalize_group(group)
            research_id = f"research-{index:03d}-{uuid.uuid4().hex[:8]}"
            research_dir = os.path.join(queue_root, "researches", research_id)
            os.makedirs(research_dir, exist_ok=True)
            _write_json_file(
                Path(research_dir) / "merged_prompt.json",
                {
                    "batch_id": batch_id,
                    "research_id": research_id,
                    "members": list(members),
                    "merge_reason": merge_reason,
                    "prompt": merged_prompt,
                },
            )
            entry: dict[str, Any] = {
                "topic": _topic_of(merged_prompt),
                "members": list(members),
                "research_id": research_id,
            }
            if merge_reason:
                entry["merge_reason"] = merge_reason
            run_ctx = dict(ctx)
            run_ctx["research_master_dir"] = research_dir
            if batch_id:
                run_ctx["session_id"] = f"{ctx.get('session_id')}:{batch_id}:{research_id}"
            log_token = None
            if progress:
                def _record_tool_event(event_type: str, payload: dict[str, Any]) -> None:
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
                    progress(
                        {
                            "latest_action": f"{event['event']} {event['tool']}".strip(),
                            "recent_events": [event],
                        }
                    )

                log_token = set_log_context(_chack_tool_progress_callback=_record_tool_event)
            try:
                admin_result = self._run_admin(
                    merged_prompt,
                    run_ctx,
                    save_artifacts=save_artifacts,
                )
            finally:
                if log_token is not None:
                    reset_log_context(log_token)
            if isinstance(admin_result, dict):
                for key, value in admin_result.items():
                    if value not in (None, ""):
                        entry[str(key)] = value if isinstance(value, (dict, list, bool, int, float)) else str(value)
            else:
                entry["conclusions"] = str(admin_result or "").strip()
            if save_artifacts:
                entry.setdefault("evidence_data_path", research_dir)
            return entry

        # Launch ONE administrator per merged research request, all in parallel. Each
        # administrator isolates its own evidence dir via contextvars, so concurrent
        # clusters never interfere. researcher_queue_max_parallel_researches caps the
        # fan-out (0 => no cap: run every cluster at once).
        configured_cap = int(getattr(self.config, "researcher_queue_max_parallel_researches", 0) or 0)
        max_parallel = max(1, len(groups) if configured_cap <= 0 else min(configured_cap, len(groups)))
        if progress:
            progress(
                {
                    "latest_action": f"launching {len(groups)} administrator research(es) in parallel",
                    "current_research_index": 0,
                    "current_research_count": len(groups),
                }
            )
        results_by_index: dict[int, dict[str, Any]] = {}
        if len(groups) <= 1:
            for index, group in enumerate(groups):
                results_by_index[index] = _run_one_cluster(index, group)
        else:
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {}
                for index, group in enumerate(groups):
                    ctx_copy = contextvars.copy_context()
                    futures[executor.submit(ctx_copy.run, _run_one_cluster, index, group)] = index
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results_by_index[index] = future.result()
                    except Exception as exc:
                        results_by_index[index] = {
                            "topic": "",
                            "research_id": "",
                            "conclusions": f"Research failed ({type(exc).__name__}: {exc})",
                        }
        researches = [results_by_index[i] for i in sorted(results_by_index)]
        result = _compact(
            {
                "queue_evidence_data_path": queue_root if save_artifacts else "",
                "researches": researches,
                "count": len(researches),
                "researcher_usage": _researcher_usage_for(researches),
                "artifacts_preserved": bool(save_artifacts),
            }
        )
        if created_standalone_root and not save_artifacts:
            shutil.rmtree(queue_root, ignore_errors=True)
        return result

    def _queue_research_context(self) -> dict[str, Any]:
        runtime_minutes = max(0, int(getattr(self.config, "researcher_queue_max_runtime_minutes", 0) or 0))
        cost_usd = max(0.0, float(getattr(self.config, "researcher_queue_max_cost_usd", 0.0) or 0.0))
        max_turns = max(2, int(getattr(self.admin, "max_turns", 30) or 30))
        return {
            "max_turns": max_turns,
            "memory_max_messages": 8,
            "memory_reset_to_messages": 8,
            "max_runtime_minutes": runtime_minutes,
            "remaining_runtime_minutes": float(runtime_minutes),
            "max_cost_usd": cost_usd,
            "remaining_cost_usd": cost_usd,
            "session_id": f"researcher_queue:{int(time.time() * 1000)}",
            "main_action": "researcher_queue",
        }

    @staticmethod
    def _normalize_group(group: Any) -> tuple[str, list[int], str]:
        if isinstance(group, dict):
            prompt = str(group.get("prompt") or "").strip()
            raw_members = group.get("members") or []
            reason = str(group.get("merge_reason") or "").strip()
        elif isinstance(group, tuple) and len(group) >= 3:
            prompt = str(group[0] or "").strip()
            raw_members = group[1]
            reason = str(group[2] or "").strip()
        elif isinstance(group, tuple) and len(group) >= 2:
            prompt = str(group[0] or "").strip()
            raw_members = group[1]
            reason = ""
        else:
            return "", [], ""
        members: list[int] = []
        for raw in raw_members or []:
            try:
                members.append(int(raw))
            except (TypeError, ValueError):
                continue
        return prompt, members, reason

    def _run_admin(
        self,
        prompt: str,
        ctx: dict[str, Any],
        save_artifacts: bool = False,
    ) -> dict[str, Any]:
        # Call the administrator's core directly to skip its LLM-facing length gate
        # (the merge agent already produced a proper prompt) and to keep the master
        # evidence directory isolated per call. Context-local research state keeps
        # concurrently dispatched administrator runs from sharing artifact roots.
        try:
            raw = self.admin._run_single(str(prompt), ctx, save_artifacts=save_artifacts)
        except Exception as exc:
            return {
                "conclusions": f"Research failed ({type(exc).__name__}: {exc})",
                "researcher_call_counts": {},
                "total_researcher_calls": 0,
                "researcher_usage_complete": False,
            }
        payload = _json_loads(raw)
        entry: dict[str, Any] = {}
        if isinstance(payload, dict):
            raw_counts = payload.get("researcher_call_counts")
            entry["researcher_call_counts"] = _clean_researcher_call_counts(raw_counts)
            entry["total_researcher_calls"] = int(sum(entry["researcher_call_counts"].values()))
            entry["researcher_usage_complete"] = isinstance(raw_counts, dict)
            conclusions = str(payload.get("administrator_conclusions") or "").strip()
            if conclusions:
                entry["conclusions"] = conclusions
                evidence_path = str(payload.get("evidence_data_path") or "").strip()
                if save_artifacts and evidence_path:
                    entry["evidence_data_path"] = evidence_path
                output_files = payload.get("output_files")
                if save_artifacts and isinstance(output_files, dict):
                    entry["output_files"] = output_files
                return entry
            failure = str(payload.get("failure_reason") or "").strip()
            if failure:
                entry["conclusions"] = f"Research failed: {failure}"
                evidence_path = str(payload.get("evidence_data_path") or "").strip()
                if save_artifacts and evidence_path:
                    entry["evidence_data_path"] = evidence_path
                output_files = payload.get("output_files")
                if save_artifacts and isinstance(output_files, dict):
                    entry["output_files"] = output_files
                return entry
        return {
            "conclusions": str(raw or "").strip(),
            "researcher_call_counts": {},
            "total_researcher_calls": 0,
            "researcher_usage_complete": False,
        }

    # -- merge agent: cluster overlapping requests into fewer research prompts --
    def _merge_prompts(self, prompts: list[str]) -> list[tuple[str, list[int], str]]:
        if len(prompts) <= 1:
            return [(prompts[0], [0], "single request; no merge needed")] if prompts else []
        groups = self._run_merge_agent(prompts)
        if not groups:
            return [(p, [i], "merge unavailable; dispatched separately") for i, p in enumerate(prompts)]
        return groups

    def _run_merge_agent(self, prompts: list[str]) -> Optional[list[tuple[str, list[int], str]]]:
        model_name = self.merge_model or self.fallback_model or ""
        numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(prompts))
        text = (
            f"Merge these {len(prompts)} research requests. Group the ones that ask for "
            f"essentially the same research and write one merged prompt per group.\n\n{numbered}"
        )
        overrides = {
            "agent": {
                "output_schema_json": _MERGE_OUTPUT_SCHEMA,
                "output_schema_name": "research_merge_groups",
                "output_schema_strict": True,
                "self_critique_enabled": False,
                "self_critique_rounds": 0,
                "sub_action": "researcher_queue_merge",
            },
            "session": {"max_turns": 4, "long_term_memory_enabled": False},
            "tools": {"max_tools_used": 0, "task_steps_manager_enabled": False},
        }
        try:
            config = build_subagent_config(
                self.config,
                model_name=model_name,
                model_provider=self.model_provider,
                max_turns=4,
                system_prompt=_MERGE_SYSTEM_PROMPT,
                overrides=overrides,
            )
            from chack_agent import Chack

            chack = Chack(config)
            result = chack.run(
                session_id=f"researcher_queue_merge:{int(time.time() * 1000)}",
                text=text,
                min_tools_used_override=0,
                max_tools_used_override=0,
                enable_self_critique=False,
                require_task_steps_manager_init_first=False,
                tools_override=[],
                system_prompt_override=config.system_prompt,
            )
        except Exception:
            return None
        return self._parse_merge_groups(getattr(result, "output", ""), prompts)

    @staticmethod
    def _parse_merge_groups(
        output: Any, prompts: list[str]
    ) -> Optional[list[tuple[str, list[int], str]]]:
        payload = _json_loads(output)
        raw_groups = payload.get("groups") if isinstance(payload, dict) else None
        if not isinstance(raw_groups, list):
            return None
        groups: list[tuple[str, list[int], str]] = []
        covered: set[int] = set()
        for group in raw_groups:
            if not isinstance(group, dict):
                continue
            merged_prompt = str(group.get("prompt") or "").strip()
            if not merged_prompt:
                continue
            merge_reason = str(group.get("merge_reason") or "").strip()
            members: list[int] = []
            for member in group.get("members") or []:
                try:
                    idx = int(member)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(prompts) and idx not in covered:
                    members.append(idx)
                    covered.add(idx)
            if members:
                groups.append((merged_prompt, members, merge_reason))
        # Any request the merge agent forgot becomes its own research (never dropped).
        for idx, original in enumerate(prompts):
            if idx not in covered:
                groups.append((original, [idx], "merge agent omitted this request; dispatched separately"))
                covered.add(idx)
        return groups or None

    def status(self) -> str:
        return self.queue.status()


def get_researcher_queue_tool(helper: ResearcherQueueAgentTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    max_n = helper.max_requests_per_call

    @function_tool(name_override="researcher_queue")
    def researcher_queue(prompt: str | list[str], save_artifacts: bool = True, queue_id: str = "") -> str:
        """Submit research to the shared research queue and wait for the results.

        Many agents can call this at the same time. Requests are collected for a short
        window, near-duplicate requests are merged, each merged request is researched
        once by a research administrator, and each caller receives only the research
        entries covering the request(s) it submitted. This call BLOCKS until the batch finishes.

        Use it to offload research without repeating work another agent already asked
        for: identical or overlapping requests in the same window run only once.

        Args:
            prompt: One research request, or a list of requests up to the configured
                per-call maximum. Each request is a detailed instruction of what to
                research and why (scope, entities, timeframe, expected output).
            save_artifacts: Defaults to true for queue research. If true, preserve evidence files for the whole shared
                batch and return evidence_data_path for each administrator research
                that produced one. If false, evidence files are temporary and deleted.
            queue_id: Optional logical queue id returned by researcher_queue_create.
                Use it when multiple MCP callers should intentionally share the same
                long-lived queue evidence folder across requests. Leave empty for the
                process-local default batching queue.

        Output: compact JSON {"researches":[{"topic","conclusions"}],"count"} listing
        every research done in the batch; "topic" indicates what each research covered.
        """
        effective_queue_id = (
            str(queue_id or "").strip()
            or getattr(helper, "queue_id", "")
            or os.environ.get(QUEUE_ID_ENV, "").strip()
        )
        try:
            return run_with_tool_logging(
                "researcher_queue",
                {"prompt": prompt, "save_artifacts": save_artifacts, "queue_id": effective_queue_id},
                lambda: helper.run(prompt, save_artifacts=save_artifacts, queue_id=effective_queue_id),
            )
        except Exception as exc:
            return f"ERROR: researcher_queue failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(researcher_queue)
    tool.description = (
        f"{tool.description}\n\n"
        f"Parameters: prompt is one research request or a list of up to {max_n}. "
        "Set save_artifacts true when the batch should preserve evidence files and return evidence_data_path. "
        "Set queue_id to a value returned by researcher_queue_create when MCP callers should share one queue artifact folder. "
        "The call blocks until the shared queue processes the batch.\n"
        "Output: compact JSON listing relevant research for this caller (topic + conclusions, plus queue_evidence_data_path, request_evidence_data_path, and evidence_data_path when preserved)."
    )
    return tool


def get_researcher_queue_create_tool(helper: ResearcherQueueAgentTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="researcher_queue_create")
    def researcher_queue_create(queue_id: str = "") -> str:
        """Create or reuse a logical researcher queue and return its shared artifact folder.

        Use this in long-lived MCP processes when several callers should submit
        related research requests over time to the same queue/root folder. Pass the
        returned queue_id into researcher_queue.

        Args:
            queue_id: Optional caller-chosen stable id. Leave empty to generate one.

        Output: Compact JSON with queue_id and queue_evidence_data_path.
        """
        try:
            return run_with_tool_logging(
                "researcher_queue_create",
                {"queue_id": queue_id},
                lambda: helper.queue.create_queue(queue_id=queue_id),
            )
        except Exception as exc:
            return f"ERROR: researcher_queue_create failed ({exc})"

    tool = researcher_queue_create
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: queue_id is optional; provide a stable id to reuse/create a named logical queue, or omit it to generate one.\n"
        "Output: compact JSON {queue_id,queue_evidence_data_path}. Use queue_id in researcher_queue."
    )
    return tool


def get_researcher_queue_status_tool(helper: ResearcherQueueAgentTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="researcher_queue_status")
    def researcher_queue_status() -> str:
        """Return status for the shared researcher queue in this process.

        Output: Compact JSON with open/processing batches, per-request ids and
        request artifact dirs, prompt/caller counts, batch ages, current merged
        research index, recent admin/researcher tool events, artifact flag, and
        queue limits. This tool never starts research and is safe to call while waiting.
        """
        try:
            return run_with_tool_logging(
                "researcher_queue_status",
                {},
                lambda: helper.status(),
            )
        except Exception as exc:
            return f"ERROR: researcher_queue_status failed ({exc})"

    tool = researcher_queue_status
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: none.\n"
        "Output: compact JSON describing open/processing queue batches, request ids/dirs, current research progress, and recent tool events."
    )
    return tool
