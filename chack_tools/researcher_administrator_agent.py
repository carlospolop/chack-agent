from __future__ import annotations

import atexit
import json
import os
import re
import time
import asyncio
import contextvars
import hashlib
import multiprocessing
import queue
import signal
import uuid
import threading
from concurrent.futures import Future
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
    compact_researcher_digest,
    create_research_master_dir,
    create_subagent_session_id,
    end_researcher_response_collection,
    enforce_prompt_str_or_list_schema,
    inherit_subagent_limits,
    normalize_researcher_response_payload,
    normalize_subagent_prompts,
    researcher_response_from_output,
    subagent_launch_block_reason,
)
from .task_steps_manager_state import current_session_id
from .research_artifacts import (
    add_research_artifact_tools,
    cleanup_research_artifacts,
    research_artifacts_master_root,
    research_artifacts_root,
    reset_research_artifact_context,
    set_research_artifact_context,
)
from .cancellation import (
    register_process,
    request_cancel,
    reset_cancellation_event,
    set_cancellation_event,
    unregister_process,
)
from .telemetry import current_log_context, reset_log_context, run_with_tool_logging, set_log_context

try:
    from agents import function_tool
    from agents.tool_context import ToolContext
    from agents.usage import Usage
except ImportError:
    function_tool = None
    ToolContext = None
    Usage = None


class _DaemonThreadPoolExecutor:
    """Minimal daemon executor that never joins abandoned work at interpreter exit.

    The stdlib executor registers workers in a private exit-time join table, so
    merely setting ``Thread.daemon`` does not contain a blocked researcher. This
    implementation intentionally exposes only the ``submit`` and ``shutdown``
    operations needed by the researcher orchestrator.
    """

    def __init__(self, max_workers: int, thread_name_prefix: str = "research-worker") -> None:
        self._max_workers = max(1, int(max_workers))
        self._thread_name_prefix = str(thread_name_prefix or "research-worker")
        self._work_queue: queue.Queue[Any] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def _worker(self) -> None:
        while True:
            item = self._work_queue.get()
            if item is None:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

    def _ensure_workers_locked(self) -> None:
        while len(self._threads) < self._max_workers:
            index = len(self._threads)
            thread = threading.Thread(
                name=f"{self._thread_name_prefix}_{index}",
                target=self._worker,
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def submit(self, fn, /, *args, **kwargs) -> Future:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._ensure_workers_locked()
            future: Future = Future()
            self._work_queue.put((future, fn, args, kwargs))
            return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if cancel_futures:
                    while True:
                        try:
                            item = self._work_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is not None:
                            item[0].cancel()
                for _thread in self._threads:
                    self._work_queue.put(None)
            threads = list(self._threads)
        if wait:
            for thread in threads:
                thread.join()


_ASYNC_RESEARCH_EXECUTOR = _DaemonThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="async-research",
)
_ASYNC_RESEARCH_LOCK = threading.Lock()
_ASYNC_RESEARCH_JOBS: dict[str, dict[str, Any]] = {}
_SYNC_RESEARCH_LOCK = threading.Lock()
_SYNC_RESEARCH_BATCHES: dict[str, dict[str, Any]] = {}
_RESEARCHER_TERMINAL_STATUSES = {"done", "error", "cancelled", "deadline_exceeded"}
_BROWSER_RESEARCHER_TOOLS = {"deepchatgpt_researcher", "prochatgpt_researcher", "chatgptxhigh"}
_RESEARCH_WRITER_LOCK = threading.Lock()
_ACTIVE_RESEARCH_WRITERS: dict[str, int] = {}
_PENDING_RESEARCH_CLEANUPS: dict[str, bool] = {}
_DEFAULT_SYNTHESIS_RESERVE_MINUTES = 5
_DEFAULT_PROCESS_TERMINATION_GRACE_SECONDS = 5.0
_PROCESS_CONTEXT_WARM_LOCK = threading.Lock()
_PROCESS_CONTEXT_WARMED: set[str] = set()
_RESEARCHER_DEADLINE_EPOCH_ENV = "CHACK_RESEARCHER_ADMIN_RESEARCHER_DEADLINE_EPOCH"
_CURRENT_RESEARCH_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "chack_current_research_deadline",
    default=None,
)



def _async_task_is_terminal(task: dict[str, Any] | None) -> bool:
    """Return the logical terminal state published to async management tools.

    A cancelled/deadline task may remain physically active while its isolated
    supervisor unwinds. Callers that must return from the administrator use the
    separate physical-writer/process checks below; status polling intentionally
    exposes that distinction through ``execution_active``/``health``.
    """
    return str((task or {}).get("status") or "") in _RESEARCHER_TERMINAL_STATUSES


def _async_task_is_unwound_terminal(task: dict[str, Any] | None) -> bool:
    """Return true only when a logical terminal task has stopped executing."""
    row = task or {}
    return _async_task_is_terminal(row) and not bool(row.get("execution_active"))


def _researcher_process_context():
    """Return a clean, killable multiprocessing context.

    ``fork`` is unsafe from the multithreaded MCP/Agents host.  Prefer the
    forkserver on POSIX: the server is started cleanly and later children are
    forked from that single-purpose process, not from the live host.  Spawn is
    the portable fallback and remains the Windows/default-safe option.
    """
    # Keep the forkserver warm when possible. Its first startup imports the
    # application graph once; subsequent isolated researchers then start in
    # tens of milliseconds instead of repeatedly paying a multi-second spawn
    # import cost. The worker itself is still forked only by the clean server.
    for method in ("forkserver", "spawn"):
        try:
            if method == "forkserver" and hasattr(multiprocessing, "set_forkserver_preload"):
                multiprocessing.set_forkserver_preload(["chack_tools.researcher_administrator_agent"])
            return multiprocessing.get_context(method)
        except ValueError:
            continue
    raise RuntimeError(
        "Independent researcher processes require forkserver or spawn support."
    )


def _researcher_process_warmup() -> None:
    """No-op target used to pay forkserver/spawn startup before a deadline starts."""
    return


def _warm_researcher_process_context() -> Any:
    """Start the clean process server once, outside a child deadline.

    The first forkserver child has to import the application module graph. If
    that cost is charged to a researcher's deadline, short tests and real queued
    jobs can lose their entire first polling window before the provider call has
    even begun. Warmup is serialized and bounded; actual researchers remain
    separate processes with their own process groups.
    """
    context = _researcher_process_context()
    method = str(getattr(context, "get_start_method", lambda: "")())
    if method in _PROCESS_CONTEXT_WARMED:
        return context
    with _PROCESS_CONTEXT_WARM_LOCK:
        if method in _PROCESS_CONTEXT_WARMED:
            return context
        process = context.Process(target=_researcher_process_warmup, name="chack-researcher-warmup")
        process.daemon = False
        process.start()
        process.join(timeout=30.0)
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                process.terminate()
            process.join(timeout=2.0)
            raise RuntimeError("Researcher process-server warmup did not terminate.")
        if process.exitcode != 0:
            raise RuntimeError(f"Researcher process-server warmup failed (exitcode={process.exitcode}).")
        _PROCESS_CONTEXT_WARMED.add(method)
    return context


def _researcher_deadline_from_environment() -> float | None:
    """Translate the administrator deadline exported to an MCP subprocess.

    Tool payload reconstruction happens in a different process, so the
    ContextVar used by in-process backends is unavailable there. The wall-clock
    value is deliberately converted to a local monotonic deadline immediately;
    callers then retain the same deadline semantics as the in-process path.
    """
    raw = str(os.environ.get(_RESEARCHER_DEADLINE_EPOCH_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        remaining = float(raw) - time.time()
    except (TypeError, ValueError):
        return None
    return time.monotonic() + max(0.0, remaining)


def _serialize_researcher_tool(tool: Any) -> bytes:
    """Serialize one researcher tool for a clean child interpreter.

    OpenAI Agents function tools are dynamically-created closures and therefore
    cannot use stdlib pickle.  cloudpickle is already a runtime dependency for
    CLI/MCP tool transport and preserves both production tools and local test
    doubles without inheriting the parent's locks or ContextVars.
    """
    try:
        import cloudpickle  # type: ignore

        return cloudpickle.dumps(tool)
    except Exception as exc:
        raise RuntimeError(
            f"Researcher tool {getattr(tool, 'name', 'unknown')!r} could not be serialized for isolated execution: {exc}"
        ) from exc


def _deserialize_researcher_tool(payload: bytes) -> Any:
    try:
        import cloudpickle  # type: ignore

        return cloudpickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"Isolated researcher tool could not be deserialized: {exc}") from exc


def _send_researcher_process_message(connection: Any, payload: dict[str, Any]) -> None:
    try:
        connection.send(payload)
    except (BrokenPipeError, EOFError, OSError):
        # The administrator may have killed the child and closed the pipe while
        # it was unwinding. This is an expected cancellation path.
        return


def _live_process_group_members(pgid: int) -> list[int]:
    """Return non-zombie PIDs currently belonging to ``pgid`` on Linux.

    ``Process.is_alive()`` only observes the direct child.  A researcher can
    exit while a browser/HTTP descendant remains in the private session, so
    group termination must be verified independently of the multiprocessing
    object.  The fallback ``killpg(..., 0)`` is retained for non-/proc POSIX
    environments.
    """
    try:
        group = int(pgid or 0)
    except (TypeError, ValueError):
        return []
    if group <= 1:
        return []
    proc_root = Path("/proc")
    if proc_root.is_dir():
        members: list[int] = []
        try:
            entries = list(proc_root.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
                close = stat.rfind(")")
                if close < 0:
                    continue
                fields = stat[close + 2 :].split()
                # After comm: state, ppid, pgrp, ...
                if len(fields) < 3 or fields[0] == "Z" or int(fields[2]) != group:
                    continue
                members.append(int(entry.name))
            except (OSError, ValueError):
                continue
        return members
    killpg = getattr(os, "killpg", None)
    if not callable(killpg):
        return []
    try:
        killpg(group, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return []
    return [group]


def _researcher_process_entry(
    connection: Any,
    serialized_tool: bytes,
    payload: dict[str, Any],
    evidence_dir: str,
) -> None:
    """Invoke one researcher inside a separate, administrator-killable process."""
    try:
        # Every subprocess launched by the researcher inherits this private
        # session/process group and is terminated together with the child.
        setsid = getattr(os, "setsid", None)
        if callable(setsid):
            setsid()
    except OSError:
        pass
    # TERM is cooperative first: async researchers can observe the event and
    # unwind provider/browser resources. The supervisor retains the hard kill
    # boundary when the call ignores this handler.
    child_cancel_event = threading.Event()
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_child_shutdown(_signum: int, _frame: Any) -> None:
        child_cancel_event.set()

    try:
        signal.signal(signal.SIGTERM, _request_child_shutdown)
    except (ValueError, OSError):
        pass
    cancellation_token = set_cancellation_event(child_cancel_event)
    artifact_token = set_research_artifact_context(evidence_dir, evidence_dir)

    def _progress(event_type: str, event_payload: dict[str, Any]) -> None:
        compact: dict[str, Any] = {
            "event": str(event_type or ""),
            "tool": str(event_payload.get("tool") or ""),
        }
        for key in ("duration_ms", "error", "stage", "answer_chars", "running"):
            value = event_payload.get(key)
            if value is not None:
                compact[key] = value
        _send_researcher_process_message(connection, {"kind": "progress", "event": compact})

    log_token = set_log_context(_chack_tool_progress_callback=_progress)
    try:
        getpgrp = getattr(os, "getpgrp", None)
        process_group_id = int(getpgrp()) if callable(getpgrp) else 0
        _send_researcher_process_message(
            connection,
            {"kind": "started", "pid": os.getpid(), "process_group_id": process_group_id},
        )
        tool = _deserialize_researcher_tool(serialized_tool)
        output = ResearcherAdministratorAgentTool._invoke_tool_sync(tool, payload)
        _send_researcher_process_message(
            connection,
            {"kind": "result", "output": output, "finished_at": time.time()},
        )
    except BaseException as exc:
        _send_researcher_process_message(
            connection,
            {"kind": "error", "error": f"{type(exc).__name__}: {exc}", "finished_at": time.time()},
        )
    finally:
        reset_log_context(log_token)
        reset_research_artifact_context(artifact_token)
        reset_cancellation_event(cancellation_token)

        try:
            signal.signal(signal.SIGTERM, previous_sigterm)
        except (ValueError, OSError):
            pass
        try:
            connection.close()
        except Exception:
            pass


def _terminate_researcher_process(
    process: Any,
    *,
    grace_seconds: float = _DEFAULT_PROCESS_TERMINATION_GRACE_SECONDS,
) -> dict[str, Any]:
    """Terminate a child and its private descendant process group.

    The child reports the PGID created by ``setsid`` through the supervision pipe.
    Until that report arrives, never guess that the child's PID is a process-group
    leader: use ``Process.terminate`` for the race-safe fallback.  Returning the
    signal/exit metadata makes physical termination auditable in task state and
    acceptance tests.
    """
    info: dict[str, Any] = {
        "term_sent": False,
        "kill_sent": False,
        "process_alive_after": False,
        "process_exitcode": None,
        "descendant_pids_after_term": [],
        "descendant_pids_after": [],
    }
    if process is None:
        return info
    try:
        pid = int(getattr(process, "pid", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    info["process_pid"] = pid
    try:
        pgid = int(getattr(process, "_chack_process_group_id", 0) or 0)
    except (TypeError, ValueError):
        pgid = 0
    try:
        getpgrp = getattr(os, "getpgrp", None)
        supervisor_pgid = int(getpgrp()) if callable(getpgrp) else 0
    except (OSError, TypeError, ValueError):
        supervisor_pgid = 0
    # The started IPC message can race with cancellation. Discover the group
    # from the live PID when possible, but only trust a group that is distinct
    # from the supervisor's own group. Before ``setsid`` the child inherits our
    # group and therefore remains on the safe Process.terminate fallback.
    if pgid <= 1 and pid > 1:
        try:
            getpgid = getattr(os, "getpgid", None)
            discovered_pgid = int(getpgid(pid)) if callable(getpgid) else 0
        except (ProcessLookupError, PermissionError, OSError, TypeError, ValueError):
            discovered_pgid = 0
        if discovered_pgid > 1 and discovered_pgid != supervisor_pgid:
            pgid = discovered_pgid
    # The supervisor itself must never be included in a killpg call.  A reported
    # PGID is only trusted when it is a valid private group distinct from ours.
    if pgid <= 1 or (supervisor_pgid > 1 and pgid == supervisor_pgid):
        pgid = 0
    info["process_group_id"] = pgid or None
    if pid <= 1:
        return info
    try:
        alive = bool(process.is_alive())
    except Exception:
        alive = False
    group_members_before = _live_process_group_members(pgid) if pgid > 1 else []
    if not alive and not group_members_before:
        try:
            process.join(timeout=0)
        except Exception:
            pass
        info["process_exitcode"] = getattr(process, "exitcode", None)
        return info

    try:
        if pgid > 1:
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
        info["term_sent"] = True
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
            info["term_sent"] = True
        except Exception:
            pass
    try:
        process.join(timeout=max(0.0, float(grace_seconds)))
    except Exception:
        pass
    try:
        still_alive = bool(process.is_alive())
    except Exception:
        still_alive = False
    group_members = _live_process_group_members(pgid) if pgid > 1 else []
    info["descendant_pids_after_term"] = list(group_members)
    if still_alive or group_members:
        try:
            if pgid > 1:
                os.killpg(pgid, signal.SIGKILL)
            else:
                process.kill()
            info["kill_sent"] = True
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
                info["kill_sent"] = True
            except Exception:
                pass
        try:
            process.join(timeout=2.0)
        except Exception:
            pass
        deadline = time.monotonic() + 2.0
        while pgid > 1 and _live_process_group_members(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
    remaining_group_members = _live_process_group_members(pgid) if pgid > 1 else []
    info["descendant_pids_after"] = list(remaining_group_members)
    try:
        info["process_alive_after"] = bool(process.is_alive())
    except Exception:
        info["process_alive_after"] = False
    info["process_exitcode"] = getattr(process, "exitcode", None)
    return info


def _run_researcher_in_process(
    tool: Any,
    payload: dict[str, Any],
    *,
    evidence_dir: str,
    cancel_event: threading.Event,
    termination_grace_seconds: float = _DEFAULT_PROCESS_TERMINATION_GRACE_SECONDS,
    on_process_started: Any = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Supervise one isolated researcher until it returns or is terminated."""
    if cancel_event.is_set():
        return {"cancelled": True, "finished_at": time.time()}
    # Warm the process server before the caller's child deadline starts. The
    # actual researcher remains a fresh, independently killable process below.
    context = _warm_researcher_process_context()
    parent_connection, child_connection = context.Pipe(duplex=False)
    serialized_tool = _serialize_researcher_tool(tool)
    process = context.Process(
        target=_researcher_process_entry,
        args=(child_connection, serialized_tool, payload, str(evidence_dir or "")),
        name="chack-researcher-child",
    )
    process.daemon = False
    registration = None
    latest_message: dict[str, Any] | None = None
    termination_info: dict[str, Any] = {}
    termination_lock = threading.Lock()
    process_started_notified = False

    def _terminate_registered_child(child: Any) -> dict[str, Any]:
        """Run one serialized TERM → grace → KILL sequence for this child.

        Cancellation can arrive through the registry callback while the
        supervisor loop is observing the same event. Without serialization, a
        second caller can observe the already-reaped process and overwrite the
        first caller's authoritative ``kill_sent=true`` metadata with a weaker
        no-op record. That made physical termination reporting race-dependent
        even though the process-group kill had already happened.
        """
        nonlocal termination_info
        with termination_lock:
            if termination_info:
                return dict(termination_info)
            result = _terminate_researcher_process(
                child,
                grace_seconds=termination_grace_seconds,
            )
            termination_info = dict(result)
            try:
                child._chack_termination_info = dict(termination_info)
            except Exception:
                pass
            return dict(termination_info)

    def _capture_external_termination_info() -> None:
        """Copy metadata when cancellation terminated the child externally.

        ``request_cancel`` can invoke the registered process callback from a
        different thread while the supervisor is blocked in ``poll``/``join``.
        In that race the supervisor observes an already-dead process and would
        otherwise return an empty ``termination`` object even though the
        callback performed the physical TERM/KILL escalation.
        """
        nonlocal termination_info
        if termination_info:
            return
        try:
            external = getattr(process, "_chack_termination_info", None)
        except Exception:
            external = None
        if isinstance(external, dict) and external:
            termination_info = dict(external)

    try:
        process.start()
        child_connection.close()
        # Register the actual child, not the supervisor future. Cancellation can
        # now physically terminate a provider call blocked inside the child.
        registration = register_process(process, _terminate_registered_child)
        connection_open = True
        while True:
            if connection_open:
                while parent_connection.poll(0.05):
                    try:
                        message = parent_connection.recv()
                    except (EOFError, OSError):
                        connection_open = False
                        break
                    if not isinstance(message, dict):
                        continue
                    kind = str(message.get("kind") or "")
                    if kind == "progress" and callable(on_progress):
                        event = message.get("event")
                        on_progress(event if isinstance(event, dict) else {})
                    elif kind == "started":
                        try:
                            process._chack_process_group_id = int(message.get("process_group_id") or 0)
                        except (AttributeError, TypeError, ValueError):
                            pass
                        if callable(on_process_started) and not process_started_notified:
                            process_started_notified = True
                            started_pid = int(message.get("pid") or process.pid or 0)
                            started_pgid = int(
                                message.get("process_group_id")
                                or getattr(process, "_chack_process_group_id", 0)
                                or started_pid
                            )
                            try:
                                on_process_started(started_pid, started_pgid)
                            except TypeError:
                                # Preserve compatibility with the original
                                # one-argument callback used by integrations.
                                on_process_started(started_pid)
                    elif kind in {"result", "error"}:
                        latest_message = message
            if cancel_event.is_set() and process.is_alive():
                _terminate_registered_child(process)
            if not process.is_alive():
                _capture_external_termination_info()
                break
            # A child may close its pipe while its interpreter is still
            # unwinding. Continue supervising its process state without spinning
            # on a permanently readable HUP pipe.
            if not connection_open:
                time.sleep(0.05)
        process.join(timeout=0.5)
        _capture_external_termination_info()
        while parent_connection.poll():
            try:
                message = parent_connection.recv()
            except (EOFError, OSError):
                break
            if isinstance(message, dict) and str(message.get("kind") or "") in {"result", "error"}:
                latest_message = message
        if cancel_event.is_set():
            return {
                "cancelled": True,
                "finished_at": time.time(),
                "process_pid": int(process.pid or 0),
                "process_exitcode": process.exitcode,
                "process_group_id": getattr(process, "_chack_process_group_id", None),
                "termination": dict(termination_info),
            }
        if latest_message and latest_message.get("kind") == "result":
            return {
                "output": latest_message.get("output"),
                "finished_at": latest_message.get("finished_at") or time.time(),
                "process_pid": int(process.pid or 0),
                "process_exitcode": process.exitcode,
                "process_group_id": getattr(process, "_chack_process_group_id", None),
                "termination": dict(termination_info),
            }
        if latest_message and latest_message.get("kind") == "error":
            return {
                "error": str(latest_message.get("error") or "Researcher child failed."),
                "finished_at": latest_message.get("finished_at") or time.time(),
                "process_pid": int(process.pid or 0),
                "process_exitcode": process.exitcode,
                "process_group_id": getattr(process, "_chack_process_group_id", None),
                "termination": dict(termination_info),
            }
        return {
            "error": f"Researcher child exited without a terminal result (exitcode={process.exitcode}).",
            "finished_at": time.time(),
            "process_pid": int(process.pid or 0),
            "process_exitcode": process.exitcode,
            "process_group_id": getattr(process, "_chack_process_group_id", None),
            "termination": dict(termination_info),
        }
    finally:
        _capture_external_termination_info()
        if process.is_alive():
            _terminate_researcher_process(process, grace_seconds=termination_grace_seconds)
            _capture_external_termination_info()
        unregister_process(registration)
        try:
            parent_connection.close()
        except Exception:
            pass
        try:
            child_connection.close()
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass


def _normalized_artifact_root(path: str) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve())
    except Exception:
        return value


def _research_writer_started(evidence_dir: str) -> None:
    root = _normalized_artifact_root(evidence_dir)
    if not root:
        return
    with _RESEARCH_WRITER_LOCK:
        _ACTIVE_RESEARCH_WRITERS[root] = int(_ACTIVE_RESEARCH_WRITERS.get(root, 0)) + 1


def _research_writer_finished(evidence_dir: str) -> None:
    root = _normalized_artifact_root(evidence_dir)
    if not root:
        return
    pending_cleanup: bool | None = None
    with _RESEARCH_WRITER_LOCK:
        remaining = max(0, int(_ACTIVE_RESEARCH_WRITERS.get(root, 0)) - 1)
        if remaining:
            _ACTIVE_RESEARCH_WRITERS[root] = remaining
        else:
            _ACTIVE_RESEARCH_WRITERS.pop(root, None)
            pending_cleanup = _PENDING_RESEARCH_CLEANUPS.pop(root, None)
    if pending_cleanup is not None:
        cleanup_research_artifacts(root, save_artifacts=bool(pending_cleanup))


def _cleanup_research_artifacts_when_idle(path: str, *, save_artifacts: bool) -> None:
    """Clean a temporary workspace only after every late child writer has stopped."""
    root = _normalized_artifact_root(path)
    if not root:
        return
    with _RESEARCH_WRITER_LOCK:
        if int(_ACTIVE_RESEARCH_WRITERS.get(root, 0)) > 0:
            _PENDING_RESEARCH_CLEANUPS[root] = bool(save_artifacts)
            return
    cleanup_research_artifacts(root, save_artifacts=save_artifacts)


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
            "kind": raw_job.get("kind"),
            "created_at": raw_job.get("created_at"),
            "save_artifacts": bool(raw_job.get("save_artifacts")),
            "max_parallel": raw_job.get("max_parallel"),
            "evidence_dir": raw_job.get("evidence_dir"),
            "expected_task_count": raw_job.get("expected_task_count"),
            "tasks": {
                task_id: {
                    k: v for k, v in task.items()
                    if k not in {"future", "cancel_event", "deadline_timer"}
                    and not str(k).startswith("_")
                }
                for task_id, task in (raw_job.get("tasks") or {}).items()
            },
        }


def _researcher_artifact_count(evidence_dir: str, researcher: str) -> int:
    root = Path(str(evidence_dir or "")).expanduser()
    short = normalize_researcher_name(researcher)
    candidate = root / short if short else root
    try:
        if not candidate.is_dir():
            return 0
        return sum(1 for path in candidate.rglob("*") if path.is_file())
    except Exception:
        return 0


def _async_completion_event_if_terminal_locked(job: dict[str, Any] | None) -> threading.Event | None:
    tasks = (job or {}).get("tasks") or {}
    expected = int((job or {}).get("expected_task_count") or 0)
    if expected > 0 and len(tasks) == expected and all(
            _async_task_is_terminal(row)
            for row in tasks.values()
        ):
        event = (job or {}).get("completion_event")
        return event if isinstance(event, threading.Event) else None
    return None


def _persist_async_job_ledger(job_id: str) -> None:
    """Persist compact child state without prompts, raw outputs, or runtime objects."""
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        if not job:
            return
        evidence_dir = str(job.get("evidence_dir") or "").strip()
        tasks: list[dict[str, Any]] = []
        for task in (job.get("tasks") or {}).values():
            tasks.append(
                {
                    key: task.get(key)
                    for key in (
                        "task_id",
                        "researcher",
                        "researcher_tool",
                        "status",
                        "created_at",
                        "started_at",
                        "finished_at",
                        "last_progress_at",
                        "deadline_at",
                        "deadline_seconds",
                        "current_tool",
                        "artifact_count",
                        "failure_reason",
                        "latest_action",
                        "execution_active",
                        "process_pid",
                        "process_group_id",
                        "process_exitcode",
                        "process_alive_after",
                        "descendant_pids_after_term",
                        "descendant_pids_after",
                        "termination",
                    )
                }
            )
        payload = {
            "job_id": str(job.get("job_id") or job_id),
            "kind": str(job.get("kind") or "async"),
            "created_at": job.get("created_at"),
            "updated_at": time.time(),
            "complete": bool(tasks) and all(_async_task_is_terminal(task) for task in tasks),
            "tasks": sorted(tasks, key=lambda row: str(row.get("task_id") or "")),
        }
    if not evidence_dir:
        return
    try:
        ledger_dir = Path(evidence_dir).expanduser() / "researcher_jobs"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        path = ledger_dir / f"{_async_output_name(job_id)}.json"
        temporary = ledger_dir / f".{path.name}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp"
        temporary.write_text(_compact_json(payload), encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        return


def _async_refresh_artifact_counts(job_id: str) -> None:
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        if not job:
            return
        evidence_dir = str(job.get("evidence_dir") or "")
        researchers = {
            str(task_id): str(task.get("researcher") or "")
            for task_id, task in (job.get("tasks") or {}).items()
        }
    counts = {
        task_id: _researcher_artifact_count(evidence_dir, researcher)
        for task_id, researcher in researchers.items()
    }
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        for task_id, count in counts.items():
            task = (job or {}).get("tasks", {}).get(task_id)
            if task is not None:
                task["artifact_count"] = int(count)


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
    with _ASYNC_RESEARCH_LOCK:
        for job_id in job_ids:
            job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
            tasks = (job or {}).get("tasks") or {}
            if tasks and any(not _async_task_is_unwound_terminal(task) for task in tasks.values()):
                return True
    return False


def _async_nonterminal_job_ids(job_ids: list[str]) -> list[str]:
    """Return jobs that are not logically terminal to the caller.

    Cancellation and deadline expiry are published immediately so polling can
    return and late output cannot revive a task. Physical process/writer
    unwinding is tracked separately by ``_async_unwound_job_ids``.
    """
    pending: list[str] = []
    with _ASYNC_RESEARCH_LOCK:
        for job_id in job_ids:
            normalized = str(job_id or "").strip()
            if not normalized:
                continue
            job = _ASYNC_RESEARCH_JOBS.get(normalized)
            tasks = (job or {}).get("tasks") or {}
            if tasks and any(not _async_task_is_terminal(task) for task in tasks.values()):
                pending.append(normalized)
    return pending


def _async_unwound_job_ids(job_ids: list[str]) -> list[str]:
    """Return jobs whose children or evidence writers are still unwinding."""
    pending: list[str] = []
    with _ASYNC_RESEARCH_LOCK:
        for job_id in job_ids:
            normalized = str(job_id or "").strip()
            if not normalized:
                continue
            job = _ASYNC_RESEARCH_JOBS.get(normalized)
            tasks = (job or {}).get("tasks") or {}
            if tasks and any(not _async_task_is_unwound_terminal(task) for task in tasks.values()):
                pending.append(normalized)
    return pending


def _wait_for_async_jobs_terminal(job_ids: list[str], deadline: float) -> list[str]:
    """Wait for owned async jobs to become logically terminal.

    A terminal cancellation/deadline is visible immediately; this function must
    not block on a provider call that the physical supervisor is still killing.
    Use ``_wait_for_async_jobs_unwound`` only for bounded cleanup decisions.
    """
    pending = _async_nonterminal_job_ids(job_ids)
    while pending:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            break
        # Wake periodically to re-check every job: one slow job must not hide
        # another job that completed while its event was being awaited.
        wait_seconds = 5 if remaining == float("inf") else min(5, int(remaining))
        if wait_seconds > 0:
            _async_wait_for_completion(pending[0], wait_seconds)
        else:
            time.sleep(min(0.1, remaining))
        pending = _async_nonterminal_job_ids(job_ids)
    return pending


def _wait_for_async_jobs_unwound(job_ids: list[str], deadline: float) -> list[str]:
    """Wait boundedly for physical child/writer cleanup after terminality."""
    pending = _async_unwound_job_ids(job_ids)
    while pending:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
        pending = _async_unwound_job_ids(job_ids)
    return pending


def _async_job_ids_for_evidence_dir(evidence_dir: str) -> list[str]:
    """Return every async job owned by one administrator evidence workspace.

    The Agents SDK may invoke a function tool in a copied ``contextvars``
    context.  In that case the run-local accounting ContextVar can differ from
    the one used by the administrator finalizer, even though the process-wide
    async job store still contains the completed job.  The evidence workspace
    is created uniquely for one administrator run, so it is a safe ownership
    key and lets finalization/cleanup harvest jobs without relying on ContextVar
    propagation.
    """
    root = str(evidence_dir or "").strip()
    if not root:
        return []
    found: list[tuple[float, str]] = []
    with _ASYNC_RESEARCH_LOCK:
        for job_id, job in _ASYNC_RESEARCH_JOBS.items():
            if str(job.get("evidence_dir") or "").strip() != root:
                continue
            try:
                created_at = float(job.get("created_at") or 0.0)
            except (TypeError, ValueError):
                created_at = 0.0
            found.append((created_at, str(job_id)))
    return [job_id for _created_at, job_id in sorted(found)]


def _async_submit(fn, *args):
    return _ASYNC_RESEARCH_EXECUTOR.submit(fn, *args)


def _async_mark_task_running_or_cancelled(job_id: str, task_id: str, tool_name: str, started_at: float) -> bool:
    should_run = False
    completion_event = None
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        task_deadline = float(task.get("deadline_at") or 0.0) if task else 0.0
        if task and task_deadline > 0 and started_at >= task_deadline:
            task["status"] = "deadline_exceeded"
            task["deadline_exceeded"] = True
            task["cancel_requested"] = True
            task["finished_at"] = started_at
            task["failure_reason"] = "Researcher deadline elapsed before it acquired an execution slot."
            task["latest_action"] = "deadline exceeded before process launch"
            task["execution_active"] = False
            task["last_activity_at"] = started_at
            task["last_progress_at"] = started_at
            completion_event = _async_completion_event_if_terminal_locked(job)
        elif task and (
            task.get("cancel_requested")
            or str(task.get("status") or "") in _RESEARCHER_TERMINAL_STATUSES
        ):
            if str(task.get("status") or "") not in _RESEARCHER_TERMINAL_STATUSES:
                task["status"] = "cancelled"
                task["finished_at"] = started_at
                task["failure_reason"] = "Researcher cancelled before start."
            task["started_at"] = task.get("started_at") or started_at
            task["execution_active"] = False
            task["last_activity_at"] = started_at
            task["last_progress_at"] = started_at
            task["latest_action"] = str(task.get("status") or "cancelled")
            completion_event = _async_completion_event_if_terminal_locked(job)
        elif task:
            task["status"] = "running"
            task["started_at"] = started_at
            task["execution_active"] = True
            task["last_activity_at"] = started_at
            task["last_progress_at"] = started_at
            task["current_tool"] = tool_name
            task["latest_action"] = f"running {tool_name}"
            should_run = True
    if isinstance(completion_event, threading.Event):
        completion_event.set()
    _persist_async_job_ledger(job_id)
    return should_run


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
        now = time.time()
        task["last_activity_at"] = now
        task["last_progress_at"] = now
        task["current_tool"] = str(tool or task.get("current_tool") or "")
        task["latest_action"] = f"{event_type} {tool}".strip()
        if event_type == "tool_started" and tool:
            live_counts = task.setdefault("live_tool_call_counts", {})
            live_counts[tool] = int(live_counts.get(tool, 0)) + 1
    _persist_async_job_ledger(job_id)


def _async_register_task(job_id: str, task_id: str, task: dict[str, Any]) -> None:
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        if not job:
            return
        job.setdefault("tasks", {})[task_id] = task
    _persist_async_job_ledger(job_id)


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
        response = normalize_researcher_response_payload(parsed)
        response.setdefault("researcher_tool", tool_name)
    else:
        response = researcher_response_from_output(tool_name, result.get("output"))
    if response is None:
        response = {
            "research_worked": False,
            "failure_reason": "Researcher did not return parseable JSON.",
            "overall_summary": "The unparseable researcher response was preserved in the paired raw output file.",
            "findings": [],
            "gaps": ["The researcher response could not be parsed into the configured structured output."],
            "open_topics": [],
            "full_research_review": str(result.get("output") or ""),
            "researcher_tool": tool_name,
        }
    root = Path(str(evidence_dir or "")).expanduser()
    try:
        output_dir = root / "researcher_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"async_{_async_output_name(task_id)}_{_async_output_name(tool_name)}"
        (output_dir / f"{stem}.json").write_text(_compact_json(response), encoding="utf-8")
        if result.get("output") is not None:
            raw_value = result.get("output")
            raw_text = raw_value if isinstance(raw_value, str) else _compact_json(raw_value)
            (output_dir / f"{stem}.raw.txt").write_text(raw_text, encoding="utf-8")
    except Exception:
        return


def _batch_result_projection(result: dict[str, Any]) -> dict[str, Any]:
    """Project one synchronous child result onto the administrator control plane."""
    projected: dict[str, Any] = {
        key: result.get(key)
        for key in ("researcher", "researcher_tool", "status", "task_id")
        if result.get(key) not in (None, "")
    }
    parsed = result.get("parsed_response")
    if isinstance(parsed, dict):
        projected["digest"] = compact_researcher_digest(parsed)
        counts = parsed.get("tool_call_counts") or result.get("tool_call_counts") or {}
        total = parsed.get("total_tool_calls")
        if total is None:
            total = result.get("total_tool_calls")
    else:
        counts = result.get("tool_call_counts") or {}
        total = result.get("total_tool_calls")
    if isinstance(counts, dict) and counts:
        projected["tool_call_counts"] = {
            str(name): int(value)
            for name, value in sorted(counts.items())
            if str(name).strip() and int(value or 0) > 0
        }
    if total is not None:
        projected["total_tool_calls"] = int(total or 0)
    if result.get("finished_at") is not None:
        projected["finished_at"] = result.get("finished_at")
    return projected


def _batch_result_is_useful(result: dict[str, Any]) -> bool:
    parsed = result.get("parsed_response")
    return isinstance(parsed, dict) and _response_has_useful_evidence(parsed)


def _persist_batch_researcher_output(
    evidence_dir: str,
    batch_id: str,
    result: dict[str, Any],
) -> None:
    """Persist a synchronous-batch result before projecting it for the parent.

    ``run_researchers_batch`` is a parent-facing control-plane tool.  Its return
    value must never contain both the exact provider output and the full parsed
    response, but finalization still needs the lossless response for validation
    and the filesystem must retain both representations for audit/recovery.
    """
    if not evidence_dir or not isinstance(result, dict):
        return
    tool_name = str(result.get("researcher_tool") or "").strip()
    if not tool_name:
        return
    parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else None
    if parsed is not None:
        response = normalize_researcher_response_payload(parsed)
        response.setdefault("researcher_tool", tool_name)
    else:
        response = researcher_response_from_output(tool_name, result.get("output"))
    if response is None:
        response = {
            "research_worked": False,
            "failure_reason": str(result.get("error") or "Researcher did not return parseable JSON.")[:500],
            "overall_summary": "The researcher returned unparseable output; the exact response is preserved in the paired raw file.",
            "findings": [],
            "gaps": ["The researcher response could not be parsed into the configured structured output."],
            "open_topics": [],
            "full_research_review": "",
            "researcher_tool": tool_name,
        }
    root = Path(str(evidence_dir)).expanduser()
    try:
        output_dir = root / "researcher_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"batch_{_async_output_name(batch_id)}_{_async_output_name(str(result.get('task_id') or result.get('researcher') or 'researcher'))}_{_async_output_name(tool_name)}"
        (output_dir / f"{stem}.json").write_text(_compact_json(response), encoding="utf-8")
        if result.get("output") is not None:
            raw_value = result.get("output")
            raw_text = raw_value if isinstance(raw_value, str) else _compact_json(raw_value)
            (output_dir / f"{stem}.raw.txt").write_text(raw_text, encoding="utf-8")
    except Exception:
        return


def _async_mark_task_done(job_id: str, task_id: str, future: Any) -> None:
    try:
        result = future.result()
        if not isinstance(result, dict):
            result = {"output": result}
        if result.get("cancelled"):
            proposed_status = "cancelled"
            proposed_error = ""
        elif result.get("error"):
            proposed_status = "error"
            proposed_error = str(result.get("error") or "Researcher failed.")
        elif not isinstance(result.get("parsed_response"), dict):
            proposed_status = "error"
            proposed_error = "Researcher did not return parseable final researcher JSON."
        else:
            proposed_status = "done"
            proposed_error = ""
    except Exception as exc:
        result = {}
        proposed_status = "cancelled" if future.cancelled() else "error"
        proposed_error = f"{type(exc).__name__}: {exc}"

    evidence_dir = ""
    researcher = ""
    tool_name = ""
    terminal_status = ""
    deadline_timer = None
    writer_registered = False
    termination_metadata: dict[str, Any] = {}
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        evidence_dir = str((job or {}).get("evidence_dir") or "")
        if task:
            researcher = str(task.get("researcher") or "")
            tool_name = str(task.get("researcher_tool") or result.get("researcher_tool") or "")
            if isinstance(result.get("termination"), dict):
                termination_metadata = deepcopy(result["termination"])
            terminal_status = str(task.get("status") or "")
            deadline_timer = task.get("deadline_timer")
            deadline_at = float(task.get("deadline_at") or 0.0)
            if terminal_status not in _RESEARCHER_TERMINAL_STATUSES:
                if deadline_at > 0 and time.time() >= deadline_at:
                    terminal_status = "deadline_exceeded"
                    task["deadline_exceeded"] = True
                    task["status"] = terminal_status
                    task["failure_reason"] = str(task.get("failure_reason") or task.get("error") or "Researcher child exceeded its deadline.")
                else:
                    task["completion_claimed"] = True
    if isinstance(deadline_timer, threading.Timer):
        deadline_timer.cancel()

    # Persist every returned payload before publishing its terminal state. Parseable
    # success, unparseable failure, cancellation diagnostics, and late post-deadline
    # output all remain auditable; only a status="done" response counts as coverage.
    if result:
        _persist_async_researcher_output(evidence_dir, task_id, tool_name, result)

    artifact_count = _researcher_artifact_count(evidence_dir, researcher)
    completion_event = None
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_id)
        task = job["tasks"].get(task_id) if job else None
        if task:
            now = time.time()
            existing_status = str(task.get("status") or terminal_status)
            writer_registered = bool(task.get("writer_registered"))
            task["writer_registered"] = False
            task["execution_active"] = False
            task["current_tool"] = ""
            task["artifact_count"] = artifact_count
            task["last_activity_at"] = now
            task["last_progress_at"] = max(float(task.get("last_progress_at") or 0.0), now)
            task["completion_claimed"] = False
            if termination_metadata:
                task["termination"] = termination_metadata
                for key in (
                    "process_pid",
                    "process_group_id",
                    "process_exitcode",
                    "process_alive_after",
                    "descendant_pids_after_term",
                    "descendant_pids_after",
                ):
                    if termination_metadata.get(key) is not None:
                        task[key] = termination_metadata[key]
            if existing_status in {"deadline_exceeded", "cancelled"}:
                task["status"] = existing_status
                task["late_finished_at"] = now
                task["latest_action"] = f"{existing_status}; worker unwound"
                task.pop("result", None)
            else:
                task["status"] = proposed_status
                task["finished_at"] = now
                task["latest_action"] = proposed_status
                if proposed_error:
                    task["error"] = proposed_error
                    task["failure_reason"] = proposed_error
                elif proposed_status == "cancelled":
                    task["failure_reason"] = str(task.get("failure_reason") or "Researcher cancelled.")
                if result and proposed_status in {"done", "error", "cancelled"}:
                    # Keep diagnostics losslessly even when the result is invalid or
                    # cancelled. Status remains non-success, so it cannot count as
                    # researcher coverage or revive a timed-out task.
                    task["result"] = result
                else:
                    task.pop("result", None)
            completion_event = _async_completion_event_if_terminal_locked(job)
    if isinstance(completion_event, threading.Event):
        completion_event.set()
    _persist_async_job_ledger(job_id)
    if writer_registered:
        _research_writer_finished(evidence_dir)


def _async_request_task_deadline(
    job_id: str,
    task_id: str,
    cancel_event: threading.Event,
    timeout_seconds: int,
) -> None:
    """Publish a logical deadline and request physical child termination."""
    should_cancel = False
    future = None
    completion_event = None
    evidence_dir = ""
    researcher = ""
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        task = (job or {}).get("tasks", {}).get(str(task_id or "").strip())
        if (
            not task
            or str(task.get("status") or "") in _RESEARCHER_TERMINAL_STATUSES
            or bool(task.get("completion_claimed"))
        ):
            return
        now = time.time()
        evidence_dir = str((job or {}).get("evidence_dir") or "")
        researcher = str(task.get("researcher") or "")
        task["deadline_exceeded"] = True
        task["cancel_requested"] = True
        task["deadline_seconds"] = int(timeout_seconds)
        task["error"] = f"Researcher child exceeded its {int(timeout_seconds)}s deadline."
        task["failure_reason"] = task["error"]
        task["status"] = "deadline_exceeded"
        task["finished_at"] = now
        task["latest_action"] = "deadline exceeded; cancellation requested"
        task["last_activity_at"] = now
        task["last_progress_at"] = max(float(task.get("last_progress_at") or 0.0), now)
        future = task.get("future")
        should_cancel = True
        completion_event = _async_completion_event_if_terminal_locked(job)
    artifact_count = _researcher_artifact_count(evidence_dir, researcher)
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        task = (job or {}).get("tasks", {}).get(str(task_id or "").strip())
        if task is not None:
            task["artifact_count"] = artifact_count
    if future is not None:
        future.cancel()
    if should_cancel:
        request_cancel(cancel_event)
    if isinstance(completion_event, threading.Event):
        completion_event.set()
    _persist_async_job_ledger(job_id)


def _async_task_health(task: dict[str, Any], *, now: float | None = None) -> str:
    """Return a deterministic operational health label without guessing success."""
    current = float(now if now is not None else time.time())
    status = str(task.get("status") or "unknown")
    execution_active = bool(task.get("execution_active"))
    if status in _RESEARCHER_TERMINAL_STATUSES:
        if execution_active:
            return "unwinding"
        if status == "done":
            return "succeeded"
        if status == "cancelled":
            return "cancelled"
        return "failed"
    try:
        deadline_at = float(task.get("deadline_at") or 0.0)
    except (TypeError, ValueError):
        deadline_at = 0.0
    if deadline_at and deadline_at <= current:
        return "deadline_due"
    if deadline_at and deadline_at - current <= 120:
        return "deadline_near"
    if status == "queued":
        return "waiting"
    try:
        last_progress = float(task.get("last_progress_at") or task.get("started_at") or current)
    except (TypeError, ValueError):
        last_progress = current
    idle_seconds = max(0.0, current - last_progress)
    tool_name = str(task.get("researcher_tool") or "")
    stale_after = 900 if tool_name in _BROWSER_RESEARCHER_TOOLS else 300
    if status == "running" and idle_seconds >= stale_after:
        # This is an observation, not proof that the provider is dead. The hard
        # deadline remains authoritative, especially for browser researchers.
        return "no_recent_progress"
    return "healthy"


def _async_cancel_task(
    job_id: str,
    task_id: str,
    *,
    reason: str,
    allow_running_browser: bool = False,
) -> dict[str, Any]:
    """Cancel one task while preserving siblings and terminal-state accounting."""
    job_key = str(job_id or "").strip()
    task_key = str(task_id or "").strip()
    clean_reason = " ".join(str(reason or "").split())[:500]
    future = None
    timer = None
    cancel_event = None
    completion_event = None
    was_active = False
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(job_key)
        if not job:
            return {"job_found": False, "job_id": job_id, "error": "Unknown async researcher job id."}
        task = (job.get("tasks") or {}).get(task_key)
        if not task:
            return {
                "job_found": True,
                "task_found": False,
                "job_id": job_id,
                "task_id": task_id,
                "error": "Unknown researcher task id for this job.",
            }
        status = str(task.get("status") or "unknown")
        was_active = bool(task.get("execution_active"))
        tool_name = str(task.get("researcher_tool") or "")
        if status in _RESEARCHER_TERMINAL_STATUSES:
            return {
                "job_found": True,
                "task_found": True,
                "job_id": job_id,
                "task_id": task_id,
                "status": status,
                "execution_active": was_active,
                "cancellation_requested": False,
                "already_terminal": True,
            }
        if was_active and tool_name in _BROWSER_RESEARCHER_TOOLS and not allow_running_browser:
            return {
                "job_found": True,
                "task_found": True,
                "job_id": job_id,
                "task_id": task_id,
                "status": status,
                "health": _async_task_health(task),
                "cancellation_requested": False,
                "protected": True,
                "error": (
                    "Running ChatGPT browser researchers are protected from model-initiated cancellation; "
                    "their configured hard deadline or the outer caller owns termination."
                ),
            }
        now = time.time()
        task["cancel_requested"] = True
        task["status"] = "cancelled"
        task["finished_at"] = now
        task["last_activity_at"] = now
        task["last_progress_at"] = max(float(task.get("last_progress_at") or 0.0), now)
        task["failure_reason"] = f"Researcher cancelled by administrator: {clean_reason}"
        task["latest_action"] = "cancelled; worker termination requested" if was_active else "cancelled before start"
        future = task.get("future")
        timer = task.get("deadline_timer")
        cancel_event = task.get("cancel_event")
        completion_event = _async_completion_event_if_terminal_locked(job)
    if isinstance(timer, threading.Timer):
        timer.cancel()
    if future is not None:
        future.cancel()
    process_kill_requested = False
    if isinstance(cancel_event, threading.Event):
        process_kill_requested = request_cancel(cancel_event)
    if isinstance(completion_event, threading.Event):
        completion_event.set()
    _persist_async_job_ledger(job_key)
    return {
        "job_found": True,
        "task_found": True,
        "job_id": job_id,
        "task_id": task_id,
        "status": "cancelled",
        "execution_active": was_active,
        "cancellation_requested": True,
        "process_kill_requested": process_kill_requested,
        "reason": clean_reason,
    }


def _async_cancel_job(job_id: str) -> dict[str, Any]:
    cancelled: list[str] = []
    cancellation_requested: list[str] = []
    already_finished: list[str] = []
    process_kill_requested: list[str] = []
    cancel_events: list[tuple[str, threading.Event]] = []
    futures: list[Any] = []
    timers: list[threading.Timer] = []
    completion_event = None
    with _ASYNC_RESEARCH_LOCK:
        job = _ASYNC_RESEARCH_JOBS.get(str(job_id or "").strip())
        if not job:
            return {"job_found": False, "job_id": job_id, "error": "Unknown async researcher job id."}
        for task_id, task in (job.get("tasks") or {}).items():
            if task.get("status") in _RESEARCHER_TERMINAL_STATUSES:
                already_finished.append(task_id)
                continue
            now = time.time()
            was_active = bool(task.get("execution_active"))
            task["cancel_requested"] = True
            task["status"] = "cancelled"
            task["finished_at"] = now
            task["last_activity_at"] = now
            task["last_progress_at"] = max(float(task.get("last_progress_at") or 0.0), now)
            task["failure_reason"] = "Researcher cancellation requested."
            task["latest_action"] = "cancelled; worker termination requested" if was_active else "cancelled before start"
            future = task.get("future")
            if future is not None:
                futures.append(future)
            timer = task.get("deadline_timer")
            if isinstance(timer, threading.Timer):
                timers.append(timer)
            cancel_event = task.get("cancel_event")
            if isinstance(cancel_event, threading.Event):
                cancel_events.append((task_id, cancel_event))
            if was_active:
                cancellation_requested.append(task_id)
            else:
                cancelled.append(task_id)
        completion_event = _async_completion_event_if_terminal_locked(job)
    for timer in timers:
        timer.cancel()
    # Future.cancel() may synchronously invoke callbacks that acquire the job lock;
    # it must therefore never be called while _ASYNC_RESEARCH_LOCK is held.
    for future in futures:
        future.cancel()
    for task_id, cancel_event in cancel_events:
        if request_cancel(cancel_event):
            process_kill_requested.append(task_id)
    if isinstance(completion_event, threading.Event):
        completion_event.set()
    _persist_async_job_ledger(job_id)
    return {
        "job_found": True,
        "job_id": job_id,
        "cancelled": cancelled,
        "cancellation_requested": cancellation_requested,
        "process_kill_requested": process_kill_requested,
        "already_finished": already_finished,
        "note": "Cancellation is terminal immediately; registered subprocess trees and remote browser jobs are terminated while in-process cleanup unwinds.",
    }


def _shutdown_async_research_jobs(*, timeout_seconds: float = 15.0) -> None:
    """Physically unwind MCP-owned async researchers before process exit.

    Async management intentionally uses daemon supervisor threads so a blocked
    provider cannot hold the MCP process open forever.  That safety property has
    a sharp edge: normal MCP shutdown can otherwise terminate those threads
    before their done callbacks persist ``execution_active=false``.  The parent
    process cannot repair the in-memory job state because the queue lives in the
    MCP process, so shutdown must request process-group termination, wait for the
    supervisors to reap, and publish the final ledger from here.

    This is bounded cleanup, not a provider timeout extension.  A task is only
    reconciled as inactive after its future completed; an unresolved future is
    deliberately left marked ``execution_active`` rather than being reported as
    successful or safely unwound without evidence.
    """
    try:
        limit = max(0.1, float(timeout_seconds or 0.0))
    except (TypeError, ValueError):
        limit = 15.0
    jobs: list[tuple[str, list[tuple[str, Any, Any, Any, Any]]]] = []
    with _ASYNC_RESEARCH_LOCK:
        for job_id, job in _ASYNC_RESEARCH_JOBS.items():
            rows: list[tuple[str, Any, Any, Any, Any]] = []
            for task_id, task in (job.get("tasks") or {}).items():
                if not bool(task.get("execution_active")):
                    continue
                rows.append(
                    (
                        str(task_id),
                        task.get("future"),
                        task.get("cancel_event"),
                        task.get("deadline_timer"),
                        str(task.get("status") or "unknown"),
                    )
                )
                now = time.time()
                if str(task.get("status") or "") not in _RESEARCHER_TERMINAL_STATUSES:
                    task["status"] = "cancelled"
                    task["finished_at"] = now
                    task["failure_reason"] = "MCP process shutdown requested researcher cancellation."
                task["cancel_requested"] = True
                task["last_activity_at"] = now
                task["last_progress_at"] = max(float(task.get("last_progress_at") or 0.0), now)
                task["latest_action"] = "MCP shutdown; physical termination requested"
            if rows:
                jobs.append((str(job_id), rows))

    for job_id, rows in jobs:
        for _task_id, future, cancel_event, timer, _status in rows:
            if isinstance(timer, threading.Timer):
                timer.cancel()
            if future is not None:
                try:
                    future.cancel()
                except Exception:
                    pass
            if isinstance(cancel_event, threading.Event):
                try:
                    request_cancel(cancel_event)
                except Exception:
                    pass
        _persist_async_job_ledger(job_id)

    deadline = time.monotonic() + limit
    while jobs and time.monotonic() < deadline:
        with _ASYNC_RESEARCH_LOCK:
            pending = [
                (job_id, task_id)
                for job_id, _rows in jobs
                for task_id, task in ((_ASYNC_RESEARCH_JOBS.get(job_id) or {}).get("tasks") or {}).items()
                if bool(task.get("execution_active"))
            ]
        if not pending:
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    # Callbacks normally persist this transition.  Persist once more from the
    # shutdown owner so the final file is durable even when the last callback
    # raced the interpreter's exit sequence.
    for job_id, _rows in jobs:
        _persist_async_job_ledger(job_id)


def _shutdown_sync_research_batches(*, timeout_seconds: float = 15.0) -> None:
    """Cancel MCP-owned synchronous batches during parent/process shutdown."""
    try:
        limit = max(0.1, float(timeout_seconds or 0.0))
    except (TypeError, ValueError):
        limit = 15.0
    with _SYNC_RESEARCH_LOCK:
        jobs = list(_SYNC_RESEARCH_BATCHES.items())
    shutdown_deadline = time.monotonic() + limit
    for batch_id, runtime in jobs:
        state_lock = runtime.get("state_lock")
        states = runtime.get("states") or {}
        cancel_events: list[threading.Event] = []
        now = time.time()
        try:
            with state_lock:
                for state in states.values():
                    if not isinstance(state, dict):
                        continue
                    if str(state.get("status") or "") in _RESEARCHER_TERMINAL_STATUSES and not state.get(
                        "execution_active"
                    ):
                        continue
                    if str(state.get("status") or "") not in _RESEARCHER_TERMINAL_STATUSES:
                        state["status"] = "cancelled"
                        state["finished_at"] = now
                        state["failure_reason"] = "MCP parent shutdown requested batch cancellation."
                    state["latest_action"] = "MCP parent shutdown; physical termination requested"
                    state["last_progress_at"] = max(float(state.get("last_progress_at") or 0.0), now)
                    cancel_event = state.get("cancel_event")
                    if isinstance(cancel_event, threading.Event):
                        cancel_events.append(cancel_event)
        except Exception:
            continue
        for cancel_event in cancel_events:
            try:
                request_cancel(cancel_event)
            except Exception:
                pass
        # request_cancel() performs the owned TERM -> grace -> KILL sequence;
        # wait for each supervisor callback to publish its physical settlement
        # instead of declaring execution_active=false merely because the MCP
        # process is about to exit.
        while time.monotonic() < shutdown_deadline:
            try:
                with state_lock:
                    active = any(bool(state.get("execution_active")) for state in states.values())
            except Exception:
                active = False
            if not active:
                break
            time.sleep(min(0.05, max(0.0, shutdown_deadline - time.monotonic())))
        try:
            with state_lock:
                for state in states.values():
                    if not isinstance(state, dict) or not state.get("execution_active"):
                        continue
                    try:
                        pid = int(state.get("process_pid") or 0)
                    except (TypeError, ValueError):
                        pid = 0
                    try:
                        pgid = int(state.get("process_group_id") or 0)
                    except (TypeError, ValueError):
                        pgid = 0
                    group_members = _live_process_group_members(pgid) if pgid > 1 else []
                    pid_live = False
                    if pid > 1:
                        try:
                            stat = (Path("/proc") / str(pid) / "stat").read_text(
                                encoding="utf-8", errors="replace"
                            )
                            close = stat.rfind(")")
                            fields = stat[close + 2 :].split() if close >= 0 else []
                            pid_live = bool(fields and fields[0] != "Z")
                        except (OSError, ValueError):
                            pid_live = False
                    if not group_members and not pid_live:
                        state["execution_active"] = False
                        state["current_tool"] = ""
                        termination = dict(state.get("termination") or {})
                        termination.update(
                            {
                                "shutdown_requested": True,
                                "verified_no_process_group_members": True,
                                "process_alive_after": False,
                                "descendant_pids_after": [],
                            }
                        )
                        state["termination"] = termination
                    else:
                        state["latest_action"] = "MCP shutdown could not yet prove physical settlement"
        except Exception:
            pass
        persist = runtime.get("persist")
        if callable(persist):
            try:
                persist()
            except Exception:
                pass
        try:
            with state_lock:
                settled = not any(bool(state.get("execution_active")) for state in states.values())
        except Exception:
            settled = False
        if settled:
            with _SYNC_RESEARCH_LOCK:
                _SYNC_RESEARCH_BATCHES.pop(str(batch_id), None)


def _shutdown_all_research_jobs(*, timeout_seconds: float = 15.0) -> None:
    """Bounded shutdown for both async and synchronous MCP researcher jobs."""
    _shutdown_sync_research_batches(timeout_seconds=timeout_seconds)
    _shutdown_async_research_jobs(timeout_seconds=timeout_seconds)


atexit.register(_shutdown_all_research_jobs)

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

# Nested researcher calls are independent once each child has its own
# ContextVars/artifact subfolder. Keep a conservative cap so a required full
# researcher run finishes within the administrator's runtime without flooding
# the provider, browser broker, or MCP server.
MAX_RESEARCHER_PARALLELISM = 4

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
Prefer `start_researchers_async` and completion-aware `poll_researchers_async` waits for long work. Poll once immediately after launch. `poll_researchers_async` is status-only by default (`include_outputs=false`): this preserves every full result outside your conversation while returning only lifecycle, health, heartbeat, tool counts, artifact counts, and failures. Do not set `include_outputs=true` during routine polling; that compatibility option injects each finished child's bounded digest again on every poll. A valid result contains only `research_worked`, `failure_reason`, `overall_summary`, `findings[{claim,summary}]`, `gaps`, and `open_topics`; unparseable results fall back to raw text. Treat open topics as optional leads, not mandatory tasks: launch a follow-up only when it can materially improve the requested conclusion within the remaining budget. Ordinary jobs normally use 30-120 second waits; ChatGPT browser jobs use 300-600 seconds and may take tens of minutes or up to 180 minutes. Queued/starting for a few minutes is not failure.
Use `list_researcher_jobs` if you lose a job id. Use `get_researcher_task` for one child's current diagnostics. When a child is done, call `get_researcher_result` with `view=summary` first. Read the lossless `parsed` or exact `raw` view page by page using `next_offset` whenever detailed evidence, citations, contradictions, provenance, or omitted context matters. Full reviews and evidence artifacts remain available even though summaries and status polls omit them.
Use `cancel_researcher_task` to stop only a duplicated/stale ordinary child while preserving siblings, and `cancel_researchers_async` only when the whole ordinary job is no longer useful. Use `retry_researcher_task` at most once and only for a concrete transient failure or material missing source family; it privately reuses the original prompt. A `no_recent_progress` health label is a warning to inspect, not proof of death.
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
        status = str(task.get("status") or result.get("status") or "unparsed")
        parsed_ok = parsed is not None or researcher_response_from_output(tool_name, result.get("output")) is not None
        if status == "done" and parsed_ok:
            continue
        if status not in {"done", "error", "cancelled", "deadline_exceeded", "unparsed"} and not result.get("output") and not task.get("error"):
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
        failure = _researcher_failure_record(
            tool_name,
            status=str(row.get("status") or "unparsed"),
            output=row.get("output"),
            error=str(row.get("error") or ""),
        )
        if failure is not None:
            failures.append(failure)
    for row in payload.get("errors") or []:
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("researcher_tool") or "").strip()
        if not tool_name:
            researcher = normalize_researcher_name(str(row.get("researcher") or ""))
            tool_name = RESEARCHER_REGISTRY.get(researcher, ("", ""))[1]
        failure = _researcher_failure_record(
            tool_name,
            status=str(row.get("status") or "error"),
            error=str(row.get("error") or ""),
        )
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
        if str(task.get("status") or "") != "done":
            continue
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else None
        # Poll projections deliberately contain the digest fields but omit the full
        # review. Harvest the canonical response from async state/files instead of
        # collecting this bounded copy as a second researcher result.
        if (
            parsed is not None
            and "overall_summary" in parsed
            and "full_research_review" not in parsed
        ):
            continue
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
            response.setdefault("researcher_tool", tool_name)
            counts = task.get("tool_call_counts")
            if not isinstance(counts, dict):
                counts = result.get("tool_call_counts")
            if not isinstance(counts, dict):
                counts = response.get("tool_call_counts")
            if isinstance(counts, dict):
                response["tool_call_counts"] = deepcopy(counts)
            if task.get("total_tool_calls") is not None:
                total_calls = task.get("total_tool_calls")
            elif result.get("total_tool_calls") is not None:
                total_calls = result.get("total_tool_calls")
            else:
                total_calls = response.get("total_tool_calls")
            if total_calls is not None:
                response["total_tool_calls"] = int(total_calls or 0)
            elif isinstance(counts, dict):
                response["total_tool_calls"] = int(sum(int(value or 0) for value in counts.values()))
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


def _researcher_responses_from_batch_output_files(evidence_dir: str) -> list[dict[str, Any]]:
    """Recover full synchronous-batch responses from the owned data plane."""
    root = Path(str(evidence_dir or "")).expanduser()
    output_dir = root / "researcher_outputs"
    if not output_dir.is_dir():
        return []
    responses: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("batch_*.json")):
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
        parsed_row = row.get("parsed_response")
        parsed_is_digest_only = (
            isinstance(parsed_row, dict)
            and "full_research_review" not in parsed_row
            and "final_research_review" not in parsed_row
        )
        # The synchronous batch transport is digest-only. Recover the canonical
        # full response from batch_*.json in the owned workspace instead.
        if parsed_is_digest_only:
            continue
        if isinstance(parsed_row, dict):
            response = deepcopy(parsed_row)
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


def _response_has_useful_evidence(
    response: dict[str, Any],
    *,
    evidence_dir: str = "",
) -> bool:
    """Return whether a terminal researcher response contains usable evidence."""
    if not isinstance(response, dict) or response.get("research_worked") is not True:
        return False
    if str(response.get("failure_reason") or "").strip():
        return False
    full_review = str(response.get("full_research_review") or "").strip()
    overall_summary = str(response.get("overall_summary") or "").strip()
    placeholder = {
        "placeholder",
        "summary",
        "conclusion",
        "conclusions",
        "n/a",
        "none",
        "ok",
    }
    if overall_summary.casefold() in placeholder or full_review.casefold() in placeholder:
        return False
    findings = response.get("findings")
    if not isinstance(findings, list) or not any(
        isinstance(item, dict)
        and str(item.get("claim") or "").strip()
        and str(item.get("summary") or "").strip()
        for item in findings
    ):
        return False

    # A digest-shaped object is not evidence. Success requires a non-trivial
    # complete review, or a verifiable preserved artifact record when a provider
    # intentionally puts the complete evidence only on disk.
    has_full_review = len(full_review) >= 20
    has_verified_artifact = False
    artifact_path = str(response.get("evidence_data_path") or evidence_dir or "").strip()
    artifact_rows = response.get("key_artifacts")
    if artifact_path and isinstance(artifact_rows, list):
        root = Path(artifact_path).expanduser()
        if root.is_dir():
            for row in artifact_rows:
                if not isinstance(row, dict):
                    continue
                filename = str(row.get("filename") or "").strip()
                if not filename:
                    continue
                candidate = (root / filename).resolve()
                try:
                    candidate.relative_to(root.resolve())
                except ValueError:
                    continue
                if candidate.is_file() and candidate.stat().st_size > 0:
                    has_verified_artifact = True
                    break
    return bool(has_full_review or has_verified_artifact)


def _researcher_response_is_terminal_and_parseable(response: dict[str, Any]) -> bool:
    """Compatibility predicate for normalized terminal researcher payloads."""
    return isinstance(response, dict) and response.get("research_worked") is True


def _administrator_synthesis_is_valid(payload: dict[str, Any]) -> bool:
    """Require a real administrator synthesis before allowing success."""
    if not isinstance(payload, dict):
        return False
    conclusions = str(payload.get("administrator_conclusions") or "").strip()
    if len(conclusions) < 40:
        return False
    normalized = re.sub(r"\s+", " ", conclusions).casefold()
    return normalized not in {"summary", "conclusions", "n/a", "none", "ok"}


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


def _persist_researcher_step_raw_outputs(evidence_dir: str, steps: list[Any]) -> list[str]:
    """Persist exact direct/batch researcher tool outputs outside LLM-facing payloads."""

    root = Path(str(evidence_dir or "")).expanduser()
    researcher_tools = {tool_name for _short, (_attr, tool_name) in RESEARCHER_REGISTRY.items()}
    output_dir = root / "researcher_outputs"
    written: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        for step in steps or []:
            tool_name = _step_tool_name(step)
            if tool_name not in researcher_tools and tool_name != "run_researchers_batch":
                continue
            output = _step_tool_output(step)
            if output in (None, ""):
                continue
            index += 1
            raw_text = output if isinstance(output, str) else _compact_json(output)
            filename = f"raw_step_{index:03d}_{_safe_output_name(tool_name, 'researcher')}.raw.txt"
            (output_dir / filename).write_text(raw_text, encoding="utf-8")
            written.append(str(Path("researcher_outputs") / filename))
    except Exception:
        return written
    return written


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
        # Never pass through an unparseable administrator result: callers must
        # receive an explicit fail-closed outcome rather than infer success from
        # a non-empty model response.
        payload = {
            "research_worked": False,
            "failure_reason": "Researcher administrator did not return parseable JSON.",
            "administrator_conclusions": "",
        }
    raw_responses = list(researcher_responses or []) + _researcher_responses_from_steps(steps)
    raw_responses.extend(_researcher_responses_from_batch_output_files(evidence_dir))
    responses = _dedupe_researcher_responses(
        [
            normalize_researcher_response_payload(response)
            for response in raw_responses
            if isinstance(response, dict)
        ]
    )
    # Full responses are used internally for deterministic validation and are
    # written to the owned filesystem. The administrator/model boundary receives
    # only bounded digests; raw output and full parsed reviews never cross it.
    response_digests = [compact_researcher_digest(response) for response in responses]
    payload["researcher_responses"] = response_digests
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
            if _response_has_useful_evidence(response)
        }
        missing_tools = [name for name in required_tools if name not in successful_tools]
        payload["required_researchers"] = required_tools
        payload["required_researchers_satisfied"] = not missing_tools
        if missing_tools:
            message = "Required researchers did not complete successfully: " + ", ".join(missing_tools) + "."
            existing_reason = str(payload.get("failure_reason") or "").strip()
            payload["research_worked"] = False
            payload["failure_reason"] = f"{existing_reason} {message}".strip()

    # `research_worked=true` is a claim about the complete run, not about the
    # administrator having emitted a schema-shaped object. Require at least one
    # terminal, parseable researcher response with substantive evidence and a
    # non-trivial administrator synthesis. Failed/partial researchers remain in
    # the diagnostic fields, but cannot themselves satisfy this gate.
    if payload.get("research_worked") is True:
        valid_responses = [
            response for response in responses if _response_has_useful_evidence(response)
        ]
        invalid_terminal_failures = [
            failure
            for failure in payload["researcher_failures"]
            if str(failure.get("status") or "").strip().lower()
            in {"queued", "running", "cancelling", "unknown", "unwinding"}
        ]
        reasons: list[str] = []
        if not responses:
            reasons.append("No terminal researcher response was collected.")
        elif not valid_responses:
            reasons.append("No terminal researcher response contained useful findings or evidence.")
        if not _administrator_synthesis_is_valid(payload):
            reasons.append("Administrator conclusions are missing or are not a substantive synthesis.")
        if invalid_terminal_failures:
            reasons.append("At least one researcher was still non-terminal when the result was finalized.")
        if reasons:
            payload["research_worked"] = False
            existing_reason = str(payload.get("failure_reason") or "").strip()
            payload["failure_reason"] = " ".join(
                part for part in [existing_reason, *reasons] if part
            )

    if payload.get("research_worked") is not True and not str(payload.get("failure_reason") or "").strip():
        payload["failure_reason"] = "Researcher administrator run did not satisfy the evidence and synthesis requirements."
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
        _persist_researcher_step_raw_outputs(evidence_dir, steps)
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
        records = list(researcher_responses or [])
        records.extend(
            {
                "research_worked": False,
                "failure_reason": str(failure.get("failure_reason") or "")[:500],
                "overall_summary": "This researcher did not complete successfully; inspect its failure metadata and any preserved raw output.",
                "findings": [],
                "gaps": ["The researcher did not complete a parseable full evidence review."],
                "open_topics": [],
                "full_research_review": "",
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

        raw_files = [
            str(Path("researcher_outputs") / path.name)
            for path in sorted(researcher_dir.glob("*.raw.txt"))
        ]
        unused_raw = list(raw_files)
        batch_raw = next((path for path in raw_files if "run_researchers_batch" in path), "")
        manifest: list[dict[str, Any]] = []
        for response, structured_path in zip(records, researcher_files):
            tool_name = str(response.get("researcher_tool") or "researcher").strip()
            safe_tool = _safe_output_name(tool_name, "researcher")
            raw_path = next((path for path in unused_raw if safe_tool in Path(path).name), "")
            if raw_path:
                unused_raw.remove(raw_path)
            elif batch_raw:
                raw_path = batch_raw
            row: dict[str, Any] = {
                "researcher_tool": tool_name,
                "structured_path": structured_path,
                "format": "full_researcher_response_v1",
                "full_research_review_available": bool(response.get("full_research_review")),
            }
            if raw_path:
                row["raw_path"] = raw_path
            manifest.append(row)

        output_files: dict[str, Any] = {
            "administrator_output": "admin_output.json",
            "researcher_outputs": researcher_files,
            "researcher_output_manifest": manifest,
        }
        if raw_files:
            output_files["raw_researcher_outputs"] = raw_files
        admin_copy = deepcopy(admin_payload)
        admin_copy["output_files"] = deepcopy(output_files)
        _write_compact_json(admin_file, admin_copy)
        return output_files
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
        runtime_cap_minutes: int = 90,
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
        # Standalone administrators retain the historical 90-minute cap. The
        # shared researcher queue passes its larger, explicit per-research
        # runtime here so a long browser research cannot be cut short by a
        # hidden administrator-only limit.
        self.runtime_cap_minutes = max(0, int(runtime_cap_minutes or 0))
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
                    f"- {short} via `start_researchers_async` (request `{exposed_tool}`): authenticated ChatGPT Deep Research browser; full response and artifacts"
                )
                continue
            if short == "prochatgpt":
                lines.append(
                    f"- {short} via `start_researchers_async` (request `{exposed_tool}`): authenticated ChatGPT Pro browser; full response and artifacts"
                )
                continue
            if short == "chatgptxhigh":
                lines.append(
                    f"- {short} via `start_researchers_async` (request `{exposed_tool}`): authenticated ChatGPT Extra High browser; full response and artifacts"
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
            lines.append(f"- {short} via `run_researchers_batch` (request `{exposed_tool}`): {capability}")
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
        required = set(self.required_researchers).intersection(enabled)

        @function_tool(name_override="run_researchers_batch")
        def run_researchers_batch(
            requests_json: str,
            save_artifacts: bool = False,
            max_parallel: int = 4,
        ) -> str:
            """Run several relevant specialized researchers with bounded parallelism.

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
                max_parallel: Maximum concurrent child researchers, capped at four. Use a
                    higher value only when the independent requests are safe to overlap.

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

            required_available = {
                short
                for short in required
                if RESEARCHER_REGISTRY[short][1] in tools_by_name
            }
            missing_required = sorted(required_available - {row["researcher"] for row in normalized})
            if missing_required:
                errors.append(
                    {
                        "error": (
                            "This administrator run has required researchers. Include every required "
                            "researcher in the same batch before synthesis: "
                            + ", ".join(missing_required)
                            + "."
                        )
                    }
                )
                return _compact_json({"batch_worked": False, "errors": errors, "results": []})

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

            try:
                requested_parallel = int(max_parallel or MAX_RESEARCHER_PARALLELISM)
            except (TypeError, ValueError):
                requested_parallel = MAX_RESEARCHER_PARALLELISM
            worker_count = max(1, min(requested_parallel, MAX_RESEARCHER_PARALLELISM, len(normalized)))

            child_timeout_seconds = max(
                1,
                int(
                    getattr(
                        self.config,
                        "researcher_administrator_child_timeout_seconds",
                        2100,
                    )
                    or 2100
                ),
            )
            batch_started = time.monotonic()
            parent_deadline = _CURRENT_RESEARCH_DEADLINE.get()
            if parent_deadline is None:
                parent_deadline = _researcher_deadline_from_environment()

            evidence_dir = research_artifacts_master_root() or research_artifacts_root()
            batch_id = f"research-batch-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            effective_child_timeout = child_timeout_seconds
            if parent_deadline is not None:
                effective_child_timeout = max(
                    1,
                    min(child_timeout_seconds, int(max(0.0, parent_deadline - batch_started))),
                )
            batch_deadline = batch_started + effective_child_timeout

            state_lock = threading.Lock()
            created_at = time.time()
            states: dict[int, dict[str, Any]] = {
                index: {
                    "task_id": f"task-{index}",
                    "researcher": normalized[index]["researcher"],
                    "researcher_tool": normalized[index]["tool_name"],
                    "status": "queued",
                    "cancel_event": threading.Event(),
                    "created_at": created_at,
                    "started_at": None,
                    "finished_at": None,
                    "last_progress_at": created_at,
                    "deadline_at": created_at + effective_child_timeout,
                    "deadline_seconds": effective_child_timeout,
                    "current_tool": "",
                    "artifact_count": 0,
                    "failure_reason": "",
                    "execution_active": False,
                    "latest_action": "queued",
                }
                for index in range(len(normalized))
            }

            def _persist_batch_ledger() -> None:
                if not evidence_dir:
                    return
                with state_lock:
                    rows = [
                        {
                            key: state.get(key)
                            for key in (
                                "task_id",
                                "researcher",
                                "researcher_tool",
                                "status",
                                "created_at",
                                "started_at",
                                "finished_at",
                                "last_progress_at",
                                "deadline_at",
                                "deadline_seconds",
                                "current_tool",
                                "artifact_count",
                                "failure_reason",
                                "latest_action",
                                "execution_active",
                                "process_pid",
                                "process_group_id",
                                "process_exitcode",
                                "process_alive_after",
                                "descendant_pids_after_term",
                                "descendant_pids_after",
                                "termination",
                            )
                        }
                        for _index, state in sorted(states.items())
                    ]
                try:
                    ledger_dir = Path(evidence_dir).expanduser() / "researcher_jobs"
                    ledger_dir.mkdir(parents=True, exist_ok=True)
                    path = ledger_dir / f"{_async_output_name(batch_id)}.json"
                    temporary = ledger_dir / f".{path.name}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp"
                    temporary.write_text(
                        _compact_json(
                            {
                                "job_id": batch_id,
                                "kind": "sync_batch",
                                "created_at": created_at,
                                "updated_at": time.time(),
                                "complete": bool(rows) and all(
                                    str(row.get("status") or "") in _RESEARCHER_TERMINAL_STATUSES
                                    for row in rows
                                ),
                                "tasks": rows,
                            }
                        ),
                        encoding="utf-8",
                    )
                    os.replace(temporary, path)
                except Exception:
                    return

            with _SYNC_RESEARCH_LOCK:
                _SYNC_RESEARCH_BATCHES[batch_id] = {
                    "state_lock": state_lock,
                    "states": states,
                    "persist": _persist_batch_ledger,
                    "evidence_dir": evidence_dir,
                }

            def _maybe_unregister_sync_batch() -> None:
                with state_lock:
                    active = any(bool(state.get("execution_active")) for state in states.values())
                if not active:
                    with _SYNC_RESEARCH_LOCK:
                        _SYNC_RESEARCH_BATCHES.pop(batch_id, None)

            def _request_child_timeout(index: int, reason: str) -> None:
                cancel_event = None
                with state_lock:
                    state = states[index]
                    if (
                        state["status"] in _RESEARCHER_TERMINAL_STATUSES
                        or bool(state.get("completion_claimed"))
                    ):
                        return
                    now = time.time()
                    state["status"] = "deadline_exceeded"
                    state["finished_at"] = now
                    state["last_progress_at"] = max(float(state.get("last_progress_at") or 0.0), now)
                    state["failure_reason"] = str(reason or "Researcher child exceeded its deadline.")
                    state["latest_action"] = "deadline exceeded; cancellation requested"
                    state["artifact_count"] = _researcher_artifact_count(
                        evidence_dir,
                        str(state.get("researcher") or ""),
                    )
                    cancel_event = state["cancel_event"]
                _persist_batch_ledger()
                if isinstance(cancel_event, threading.Event):
                    request_cancel(cancel_event)

            def _run_one(index: int, row: dict[str, str], context: contextvars.Context) -> dict[str, Any]:
                state = states[index]
                cancel_event: threading.Event = state["cancel_event"]

                def _record_progress(event_type: str, payload: dict[str, Any]) -> None:
                    now = time.time()
                    tool = str(payload.get("tool") or "")
                    with state_lock:
                        if state["status"] in _RESEARCHER_TERMINAL_STATUSES:
                            return
                        state["last_progress_at"] = now
                        state["current_tool"] = tool or str(state.get("current_tool") or row["tool_name"])
                        state["latest_action"] = f"{event_type} {tool}".strip()
                    _persist_batch_ledger()

                def _inner() -> dict[str, Any]:
                    tool_name = row["tool_name"]
                    now = time.time()
                    preflight_error: dict[str, Any] | None = None
                    with state_lock:
                        if (
                            state["status"] in _RESEARCHER_TERMINAL_STATUSES
                            or cancel_event.is_set()
                            or now >= float(state.get("deadline_at") or 0.0)
                        ):
                            if state["status"] not in _RESEARCHER_TERMINAL_STATUSES:
                                state["status"] = "deadline_exceeded"
                                state["finished_at"] = now
                                state["failure_reason"] = "Researcher child deadline elapsed before it acquired an execution slot."
                            preflight_error = {
                                "researcher": row["researcher"],
                                "researcher_tool": tool_name,
                                "status": str(state["status"]),
                                "error": str(state.get("failure_reason") or "Researcher did not start before its deadline."),
                            }
                        else:
                            state["status"] = "running"
                            state["started_at"] = now
                            state["last_progress_at"] = now
                            state["current_tool"] = tool_name
                            state["execution_active"] = True
                            state["latest_action"] = f"running {tool_name}"
                    _persist_batch_ledger()
                    if preflight_error is not None:
                        return preflight_error
                    _research_writer_started(evidence_dir)
                    cancel_token = set_cancellation_event(cancel_event)
                    log_token = set_log_context(_chack_tool_progress_callback=_record_progress)
                    try:
                        def _process_started(pid: int, pgid: int = 0) -> None:
                            with state_lock:
                                state["process_pid"] = int(pid or 0)
                                state["process_group_id"] = int(pgid or pid or 0)
                                state["child_execution_boundary"] = "process"
                                state["latest_action"] = f"running {tool_name} in process {int(pid or 0)}"
                            _persist_batch_ledger()

                        output_result = _run_researcher_in_process(
                            tools_by_name[tool_name],
                            {"prompt": row["prompt"], "save_artifacts": bool(save_artifacts)},
                            evidence_dir=evidence_dir,
                            cancel_event=cancel_event,
                            termination_grace_seconds=float(
                                getattr(
                                    self.config,
                                    "researcher_administrator_child_termination_grace_seconds",
                                    _DEFAULT_PROCESS_TERMINATION_GRACE_SECONDS,
                                )
                                or _DEFAULT_PROCESS_TERMINATION_GRACE_SECONDS
                            ),
                            on_process_started=_process_started,
                            on_progress=lambda event: _record_progress(
                                str(event.get("event") or "research_progress"),
                                event,
                            ),
                        )
                        with state_lock:
                            for key in (
                                "process_pid",
                                "process_group_id",
                                "process_exitcode",
                                "process_alive_after",
                                "descendant_pids_after_term",
                                "descendant_pids_after",
                                "termination",
                            ):
                                if key in output_result:
                                    state[key] = output_result.get(key)
                            deadline_at = float(state.get("deadline_at") or 0.0)
                            timed_out = (
                                state["status"] == "deadline_exceeded"
                                or time.time() >= deadline_at
                            )
                            if not timed_out and not output_result.get("cancelled"):
                                state["completion_claimed"] = True
                            timer = state.get("deadline_timer")
                        if isinstance(timer, threading.Timer):
                            timer.cancel()
                        if timed_out:
                            return {
                                "researcher": row["researcher"],
                                "researcher_tool": tool_name,
                                "status": "deadline_exceeded",
                                "error": str(state.get("failure_reason") or "Researcher child exceeded its deadline."),
                            }
                        if output_result.get("cancelled"):
                            return {
                                "researcher": row["researcher"],
                                "researcher_tool": tool_name,
                                "status": "cancelled",
                                "error": "Researcher was cancelled by the administrator.",
                            }
                        if output_result.get("error"):
                            return {
                                "researcher": row["researcher"],
                                "researcher_tool": tool_name,
                                "status": "error",
                                "error": str(output_result.get("error") or "Researcher child failed."),
                            }
                        output = output_result.get("output")
                        parsed = researcher_response_from_output(tool_name, output)
                        if parsed is None:
                            with state_lock:
                                state["status"] = "error"
                                state["failure_reason"] = "Researcher did not return parseable final researcher JSON."
                                state["finished_at"] = time.time()
                                state["latest_action"] = "error: unparseable researcher result"
                            return {
                                "researcher": row["researcher"],
                                "researcher_tool": tool_name,
                                "status": "error",
                                "error": "Researcher did not return parseable final researcher JSON.",
                                "output": output,
                            }
                        result: dict[str, Any] = {
                            "researcher": row["researcher"],
                            "researcher_tool": tool_name,
                            "status": "done",
                            "output": output,
                            "parsed_response": parsed,
                        }
                        with state_lock:
                            state["status"] = "done"
                            state["finished_at"] = time.time()
                            state["completion_claimed"] = False
                            state["failure_reason"] = ""
                            state["latest_action"] = "done"
                        return result
                    except Exception as exc:
                        with state_lock:
                            timed_out = state["status"] == "deadline_exceeded"
                            if not timed_out:
                                state["status"] = "error"
                                state["failure_reason"] = f"{type(exc).__name__}: {exc}"
                                state["finished_at"] = time.time()
                                state["latest_action"] = "error"
                            state["completion_claimed"] = False
                        return {
                            "researcher": row["researcher"],
                            "researcher_tool": tool_name,
                            "status": "deadline_exceeded" if timed_out else "error",
                            "error": str(state.get("failure_reason") or f"{type(exc).__name__}: {exc}"),
                        }
                    finally:
                        with state_lock:
                            timer = state.get("deadline_timer")
                        if isinstance(timer, threading.Timer):
                            timer.cancel()
                        reset_log_context(log_token)
                        reset_cancellation_event(cancel_token)
                        with state_lock:
                            state["execution_active"] = False
                            state["current_tool"] = ""
                            state["artifact_count"] = _researcher_artifact_count(
                                evidence_dir,
                                row["researcher"],
                            )
                            state["last_progress_at"] = max(
                                float(state.get("last_progress_at") or 0.0),
                                time.time(),
                            )
                        _persist_batch_ledger()
                        _research_writer_finished(evidence_dir)
                        _maybe_unregister_sync_batch()

                return context.run(_inner)

            for index, state in states.items():
                timer = threading.Timer(
                    effective_child_timeout,
                    _request_child_timeout,
                    args=(
                        index,
                        f"Researcher child exceeded its {effective_child_timeout}s deadline.",
                    ),
                )
                timer.daemon = True
                state["deadline_timer"] = timer
                timer.start()
            _persist_batch_ledger()

            results: list[dict[str, Any]] = []
            futures: dict[int, Any] = {}
            executor = _DaemonThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="researcher-batch",
            )
            try:
                futures = {
                    index: executor.submit(_run_one, index, row, contextvars.copy_context())
                    for index, row in enumerate(normalized)
                }
                pending = set(futures)
                reported: set[int] = set()
                while pending:
                    for index in list(pending):
                        future = futures[index]
                        with state_lock:
                            state = dict(states[index])
                        if state["status"] == "deadline_exceeded" and not future.done():
                            future.cancel()
                            errors.append(
                                {
                                    "researcher": normalized[index]["researcher"],
                                    "researcher_tool": normalized[index]["tool_name"],
                                    "status": "deadline_exceeded",
                                    "error": state["failure_reason"] or "Researcher child exceeded its deadline.",
                                    "started_at": state["started_at"],
                                    "deadline_at": state["deadline_at"],
                                }
                            )
                            reported.add(index)
                            pending.remove(index)
                            continue
                        if not future.done():
                            continue
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {
                                "researcher": normalized[index]["researcher"],
                                "researcher_tool": normalized[index]["tool_name"],
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        if result.get("status") == "done":
                            # Keep the exact provider response in the owned data
                            # plane. The value returned by this control-plane tool
                            # is projected below and must not carry raw + parsed
                            # copies into the administrator context.
                            result.setdefault("task_id", states[index].get("task_id"))
                            _persist_batch_researcher_output(evidence_dir, batch_id, result)
                            results.append(result)
                        else:
                            result.setdefault("task_id", states[index].get("task_id"))
                            _persist_batch_researcher_output(evidence_dir, batch_id, result)
                            errors.append(
                                {
                                    key: value
                                    for key, value in result.items()
                                    if key in {"researcher", "researcher_tool", "status", "error"}
                                }
                            )
                        reported.add(index)
                        pending.remove(index)
                    if not pending:
                        break
                    if time.monotonic() >= batch_deadline:
                        for index in list(pending):
                            _request_child_timeout(
                                index,
                                "Researcher batch reached its parent administrator deadline.",
                            )
                            future = futures[index]
                            future.cancel()
                            with state_lock:
                                state = dict(states[index])
                            if index not in reported:
                                errors.append(
                                    {
                                        "researcher": normalized[index]["researcher"],
                                        "researcher_tool": normalized[index]["tool_name"],
                                        "status": "deadline_exceeded",
                                        "error": state["failure_reason"] or "Researcher batch deadline exceeded.",
                                        "started_at": state["started_at"],
                                        "deadline_at": state["deadline_at"],
                                    }
                                )
                                reported.add(index)
                        pending.clear()
                        break
                    time.sleep(0.05)
            finally:
                # Never use the context-manager form here: it performs an implicit
                # shutdown(wait=True), which was the original batch deadlock.
                executor.shutdown(wait=False, cancel_futures=True)
            results.sort(key=lambda item: str(item.get("researcher_tool") or ""))
            errors.sort(key=lambda item: (str(item.get("researcher_tool") or ""), str(item.get("status") or "error")))

            # Persist the lossless provider output before projecting anything across
            # the administrator/model boundary. The batch return is control-plane
            # traffic: it may contain one bounded digest per successful child, but
            # never the exact raw output and full parsed response together.
            for result in results:
                _persist_batch_researcher_output(evidence_dir, batch_id, result)

            _persist_batch_ledger()
            with state_lock:
                task_states = [
                    {
                        key: state.get(key)
                        for key in (
                            "task_id",
                            "researcher",
                            "researcher_tool",
                            "status",
                            "created_at",
                            "started_at",
                            "finished_at",
                            "last_progress_at",
                            "deadline_at",
                            "deadline_seconds",
                            "current_tool",
                            "artifact_count",
                            "failure_reason",
                            "execution_active",
                            "process_pid",
                            "process_group_id",
                            "process_exitcode",
                            "process_alive_after",
                            "descendant_pids_after_term",
                            "descendant_pids_after",
                            "termination",
                        )
                    }
                    for _index, state in sorted(states.items())
                ]
            projected_results = [_batch_result_projection(result) for result in results]
            batch_worked = any(_batch_result_is_useful(result) for result in results)
            batch_complete = bool(results) and not errors and all(
                _batch_result_is_useful(result) for result in results
            )
            return _compact_json(
                {
                    "batch_id": batch_id,
                    "batch_worked": batch_worked,
                    "batch_complete": batch_complete,
                    "child_timeout_seconds": effective_child_timeout,
                    "tasks": task_states,
                    "results": projected_results,
                    "errors": errors,
                }
            )

        tool = run_researchers_batch
        tool.description = (
            f"{tool.description}\n\n"
            "Parameters: requests_json is a JSON array of objects with researcher and prompt; "
            "each prompt must be >=500 characters and relevant to that specific researcher. "
            "Set save_artifacts true when source/detail artifacts should be preserved. "
            f"max_parallel is capped at {MAX_RESEARCHER_PARALLELISM}; child ContextVars and per-researcher artifact folders keep concurrent requests isolated.\n"
            "Output: Compact JSON containing lifecycle tasks, bounded digests, and errors. "
            "Exact raw output and complete parsed responses are persisted under the owned "
            "researcher_outputs directory and retrieved explicitly through result tools."
        )
        return tool

    def _normalize_researcher_requests(
        self,
        requests_json: str,
        *,
        enabled: set[str],
        tools_by_name: dict[str, Any],
        enforce_budget: bool = True,
        required_researchers: Optional[set[str]] = None,
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

        required_available = {
            short
            for short in (required_researchers or set())
            if short in enabled and RESEARCHER_REGISTRY[short][1] in tools_by_name
        }
        missing_required = sorted(required_available - {row["researcher"] for row in normalized})
        if missing_required:
            errors.append(
                {
                    "error": (
                        "This administrator run has required researchers. Include every required "
                        "researcher in the same async request before synthesis: "
                        + ", ".join(missing_required)
                        + "."
                    )
                }
            )
            return [], errors

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

    def _build_async_tools(
        self,
        tools_by_name: dict[str, Any],
        enabled_researchers: list[str],
        *,
        artifact_root: str = "",
    ):
        enabled = set(enabled_researchers)
        async_required = {
            short
            for short in self.required_researchers
            if short in {"deepchatgpt", "prochatgpt", "chatgptxhigh"}
        }
        # Async tools can be invoked through the shared MCP server, where the
        # caller's ContextVars and process environment are not guaranteed to be
        # present. Capture the root in this closure instead of inferring it from
        # ambient state at invocation time.
        owner_artifact_root = str(artifact_root or "").strip()

        def _owned_job_ids() -> list[str]:
            if owner_artifact_root:
                return _async_job_ids_for_evidence_dir(owner_artifact_root)
            # Fallback for direct/legacy use without an artifact root. Normal
            # administrator runs always have a unique workspace.
            return list(dict.fromkeys(str(job_id) for job_id in self._launched_async_job_ids if str(job_id)))

        def _owned_job_snapshot(job_id: str) -> dict[str, Any] | None:
            job_key = str(job_id or "").strip()
            if not job_key or job_key not in set(_owned_job_ids()):
                return None
            return _async_job_snapshot(job_key)

        def _already_launched_researchers() -> set[str]:
            """Return researcher types already launched by this workspace.

            Required browser researchers are enforced as one complete initial
            wave. Once that gate has been met, focused follow-ups and the
            single-task retry tool must not be forced to relaunch every required
            browser researcher and duplicate cost.
            """
            launched: set[str] = set()
            for owned_job_id in _owned_job_ids():
                snapshot = _async_job_snapshot(owned_job_id) or {}
                for task in (snapshot.get("tasks") or {}).values():
                    short = normalize_researcher_name(str(task.get("researcher") or ""))
                    if short:
                        launched.add(short)
            return launched

        def _not_owned(job_id: str) -> str:
            return _compact_json(
                {
                    "job_found": False,
                    "job_id": job_id,
                    "error": "Researcher job was not found or is not owned by this administrator workspace.",
                }
            )

        def _available_result_views(task: dict[str, Any]) -> list[str]:
            result = task.get("result") if isinstance(task.get("result"), dict) else {}
            views = ["summary"] if result else []
            if result.get("parsed_response") is not None:
                views.append("parsed")
            if result.get("output") is not None:
                views.append("raw")
            return views

        def _poll_result_projection(result: dict[str, Any]) -> dict[str, Any]:
            """Expose one bounded digest, never the full/raw and parsed copies together."""
            parsed = result.get("parsed_response")
            if isinstance(parsed, dict):
                digest = compact_researcher_digest(parsed)
                digest.pop("researcher_tool", None)
                return {"parsed_response": digest}
            if result.get("output") is not None:
                raw_value = result.get("output")
                raw_text = raw_value if isinstance(raw_value, str) else _compact_json(raw_value)
                limit = 2_000
                projected: dict[str, Any] = {"output": raw_text[:limit]}
                if len(raw_text) > limit:
                    projected.update(
                        {
                            "output_truncated": True,
                            "raw_total_chars": len(raw_text),
                            "raw_view_available": True,
                        }
                    )
                return projected
            return {}

        def _compact_task_status(task_id: str, task: dict[str, Any], *, now: float) -> dict[str, Any]:
            created_at = float(task.get("created_at") or now)
            last_activity = float(task.get("last_activity_at") or created_at)
            row = {
                "task_id": task_id,
                "researcher": task.get("researcher", ""),
                "researcher_tool": task.get("researcher_tool", ""),
                "status": task.get("status", "unknown"),
                "health": _async_task_health(task, now=now),
                "execution_active": bool(task.get("execution_active")),
                "current_tool": task.get("current_tool", ""),
                "latest_action": task.get("latest_action", ""),
                "last_progress_at": task.get("last_progress_at"),
                "deadline_at": task.get("deadline_at"),
                "elapsed_seconds": round(max(0.0, now - created_at), 3),
                "idle_seconds": round(max(0.0, now - last_activity), 3),
                "artifact_count": int(task.get("artifact_count") or 0),
                "failure_reason": task.get("failure_reason", ""),
                "result_available": bool(task.get("result")),
                "available_result_views": _available_result_views(task),
            }
            if isinstance(task.get("termination"), dict):
                row["termination"] = deepcopy(task["termination"])
            for key in (
                "retry_count",
                "retried_from_job_id",
                "retried_from_task_id",
                "retry_spawned_job_id",
                "retry_spawned_task_id",
            ):
                if task.get(key) is not None:
                    row[key] = task.get(key)
            return row

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

        child_timeout_seconds = max(
            1,
            int(
                getattr(
                    self.config,
                    "researcher_administrator_child_timeout_seconds",
                    2100,
                )
                or 2100
            ),
        )

        def _run_one(
            job_id: str,
            task_id: str,
            tool_name: str,
            prompt: str,
            save_artifacts: bool,
            semaphore: threading.Semaphore,
            cancel_event: threading.Event,
            evidence_dir: str,
        ) -> dict[str, Any]:
            with semaphore:
                started_at = time.time()
                if cancel_event.is_set():
                    return {"researcher_tool": tool_name, "cancelled": True, "finished_at": started_at}
                if not _async_mark_task_running_or_cancelled(job_id, task_id, tool_name, started_at):
                    return {"researcher_tool": tool_name, "cancelled": True, "finished_at": started_at}
                _research_writer_started(evidence_dir)
                with _ASYNC_RESEARCH_LOCK:
                    job = _ASYNC_RESEARCH_JOBS.get(job_id)
                    task = (job or {}).get("tasks", {}).get(task_id)
                    if task is not None:
                        task["writer_registered"] = True
                log_token = set_log_context(
                    _chack_tool_progress_callback=lambda event_type, payload: _record_progress(
                        job_id,
                        task_id,
                        event_type,
                        payload,
                    )
                )
                cancel_token = set_cancellation_event(cancel_event)
                artifact_token = set_research_artifact_context(evidence_dir, evidence_dir)
                try:
                    def _process_started(pid: int, pgid: int = 0) -> None:
                        with _ASYNC_RESEARCH_LOCK:
                            job = _ASYNC_RESEARCH_JOBS.get(job_id)
                            task = (job or {}).get("tasks", {}).get(task_id)
                            if task is not None:
                                task["process_pid"] = int(pid or 0)
                                task["process_group_id"] = int(pgid or pid or 0)
                                task["child_execution_boundary"] = "process"
                                task["latest_action"] = f"running {tool_name} in process {int(pid or 0)}"
                        _persist_async_job_ledger(job_id)

                    output_result = _run_researcher_in_process(
                        tools_by_name[tool_name],
                        {"prompt": prompt, "save_artifacts": bool(save_artifacts)},
                        evidence_dir=evidence_dir,
                        cancel_event=cancel_event,
                        termination_grace_seconds=float(
                            getattr(
                                self.config,
                                "researcher_administrator_child_termination_grace_seconds",
                                _DEFAULT_PROCESS_TERMINATION_GRACE_SECONDS,
                            )
                            or _DEFAULT_PROCESS_TERMINATION_GRACE_SECONDS
                        ),
                        on_process_started=_process_started,
                        on_progress=lambda event: _record_progress(
                            job_id,
                            task_id,
                            str(event.get("event") or "research_progress"),
                            event,
                        ),
                    )
                    if output_result.get("cancelled"):
                        return {
                            "researcher_tool": tool_name,
                            **output_result,
                            "cancelled": True,
                        }
                    if output_result.get("error"):
                        return {
                            "researcher_tool": tool_name,
                            **output_result,
                        }
                    output = output_result.get("output")
                finally:
                    reset_research_artifact_context(artifact_token)
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

        def _run_one_in_context(run_context: contextvars.Context, *args) -> dict[str, Any]:
            """Run one task with a private copy of the administrator's ContextVars."""
            return run_context.run(_run_one, *args)

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
            metadata plus recent tool telemetry events. Once finished, status polling
            only advertises result availability; use get_researcher_result to retrieve
            one bounded summary or lossless page at a time.

            Args:
                requests_json: Compact JSON array of objects with researcher and prompt.
                    Each prompt must be detailed and at least 500 characters.
                save_artifacts: Preserve evidence folders for these researchers.
                max_parallel: Maximum concurrent child researchers, capped at four. Use a
                    higher value only when the independent requests are safe to overlap.

            Output: Compact JSON with async_started, job_id, task ids, and validation errors.
            """
            # Pay the one-time forkserver/spawn import cost before creating task
            # deadlines.  A cold process server is infrastructure setup, not
            # researcher work, and must not consume the child's hard deadline.
            try:
                _warm_researcher_process_context()
            except Exception as exc:
                return _compact_json(
                    {
                        "async_started": False,
                        "errors": [
                            {
                                "error": f"Researcher process isolation could not be initialized: {type(exc).__name__}: {exc}"
                            }
                        ],
                        "job_id": "",
                        "tasks": [],
                    }
                )
            pending_async_required = async_required - _already_launched_researchers()
            normalized, errors = self._normalize_researcher_requests(
                requests_json,
                enabled=enabled,
                tools_by_name=tools_by_name,
                required_researchers=pending_async_required,
            )
            if not normalized:
                return _compact_json({"async_started": False, "errors": errors, "job_id": "", "tasks": []})
            job_id = f"research-job-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            self._launched_async_job_ids.append(job_id)
            try:
                requested_parallel = int(max_parallel or MAX_RESEARCHER_PARALLELISM)
            except (TypeError, ValueError):
                requested_parallel = MAX_RESEARCHER_PARALLELISM
            parallel_limit = max(1, min(requested_parallel, MAX_RESEARCHER_PARALLELISM, len(normalized)))
            semaphore = threading.Semaphore(parallel_limit)
            evidence_dir = owner_artifact_root or research_artifacts_master_root() or research_artifacts_root()
            if not evidence_dir:
                evidence_dir = (
                    os.environ.get("CHACK_RESEARCH_MASTER_DIR", "").strip()
                    or os.environ.get("CHACK_RESEARCH_DATA_DIR", "").strip()
                )
            parent_context = contextvars.copy_context()
            job: dict[str, Any] = {
                "job_id": job_id,
                "kind": "async",
                "created_at": time.time(),
                "save_artifacts": bool(save_artifacts),
                "max_parallel": parallel_limit,
                "evidence_dir": evidence_dir,
                "completion_event": threading.Event(),
                "expected_task_count": len(normalized),
                "tasks": {},
            }
            task_rows: list[dict[str, str]] = []
            prepared_tasks: list[tuple[str, dict[str, str], threading.Event]] = []
            _async_job_store(job_id, job)

            effective_child_timeout = child_timeout_seconds
            parent_deadline = _CURRENT_RESEARCH_DEADLINE.get()
            if parent_deadline is None:
                parent_deadline = _researcher_deadline_from_environment()
            if parent_deadline is not None:
                effective_child_timeout = max(
                    1,
                    min(child_timeout_seconds, int(max(0.0, parent_deadline - time.monotonic()))),
                )

            # Register every task before submitting any future. A fast first future
            # must never make a partially populated job appear complete and leave
            # its completion event permanently set while later tasks still run.
            for index, row in enumerate(normalized):
                task_id = f"task-{index}-{uuid.uuid4().hex[:6]}"
                cancel_event = threading.Event()
                created_at = time.time()
                deadline_timer = threading.Timer(
                    effective_child_timeout,
                    _async_request_task_deadline,
                    args=(job_id, task_id, cancel_event, effective_child_timeout),
                )
                deadline_timer.daemon = True
                task = {
                    "task_id": task_id,
                    "researcher": row["researcher"],
                    "researcher_tool": row["tool_name"],
                    "_request_prompt": row["prompt"],
                    "status": "queued",
                    "created_at": created_at,
                    "started_at": None,
                    "finished_at": None,
                    "last_progress_at": created_at,
                    "deadline_at": created_at + effective_child_timeout,
                    "deadline_seconds": effective_child_timeout,
                    "current_tool": "",
                    "artifact_count": 0,
                    "failure_reason": "",
                    "execution_active": False,
                    "latest_action": "queued",
                    "last_activity_at": created_at,
                    "cancel_event": cancel_event,
                    "deadline_timer": deadline_timer,
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
                task_context = parent_context.copy()
                future = _async_submit(
                    _run_one_in_context,
                    task_context,
                    job_id,
                    task_id,
                    row["tool_name"],
                    row["prompt"],
                    bool(save_artifacts),
                    semaphore,
                    cancel_event,
                    evidence_dir,
                )
                _async_set_task_future(job_id, task_id, future)
                future.add_done_callback(lambda fut, jid=job_id, tid=task_id: _task_done(jid, tid, fut))
                with _ASYNC_RESEARCH_LOCK:
                    stored_job = _ASYNC_RESEARCH_JOBS.get(job_id)
                    stored_task = (stored_job or {}).get("tasks", {}).get(task_id)
                    timer = (stored_task or {}).get("deadline_timer")
                if isinstance(timer, threading.Timer):
                    timer.start()
            return _compact_json(
                {
                    "async_started": True,
                    "job_id": job_id,
                    "tasks": task_rows,
                    "errors": errors,
                    "max_parallel": parallel_limit,
                    "next_step": (
                        "Call poll_researchers_async with this job_id immediately. Then use completion-aware "
                        "wait_seconds=300-1800 for ChatGPT browser jobs, or 30-120 for ordinary researchers. "
                        "The wait returns early on completion. Tasks run with up to "
                        f"{parallel_limit} concurrent workers in this process; each child has isolated "
                        "ContextVars and a per-researcher evidence folder. Queued/running for 1-2 "
                        "minutes can be normal. Cancel only when stale, irrelevant, or near the runtime limit."
                    ),
                }
            )

        @function_tool(name_override="list_researcher_jobs")
        def list_researcher_jobs(include_terminal: bool = True, max_jobs: int = 50) -> str:
            """List every async researcher job owned by this administrator workspace.

            Use this after context compaction or whenever a job id was forgotten. The
            response is status-only and never embeds researcher outputs.

            Args:
                include_terminal: Include jobs whose tasks are all logically terminal.
                max_jobs: Maximum newest jobs to return, clamped to 1-100.

            Output: Compact JSON with job ids and per-task operational status/health.
            """
            limit = max(1, min(int(max_jobs or 50), 100))
            rows: list[dict[str, Any]] = []
            now = time.time()
            for job_id in _owned_job_ids():
                _async_refresh_artifact_counts(job_id)
                job = _owned_job_snapshot(job_id)
                if not job:
                    continue
                tasks = [
                    _compact_task_status(task_id, task, now=now)
                    for task_id, task in sorted((job.get("tasks") or {}).items())
                ]
                complete = bool(tasks) and all(_async_task_is_terminal(task) for task in tasks)
                if complete and not include_terminal:
                    continue
                rows.append(
                    {
                        "job_id": job_id,
                        "created_at": job.get("created_at"),
                        "complete": complete,
                        "execution_active": any(bool(task.get("execution_active")) for task in tasks),
                        "max_parallel": job.get("max_parallel"),
                        "tasks": tasks,
                    }
                )
            rows = rows[-limit:]
            return _compact_json({"jobs": rows, "count": len(rows), "outputs_included": False})

        @function_tool(name_override="get_researcher_task")
        def get_researcher_task(job_id: str, task_id: str, recent_event_limit: int = 10) -> str:
            """Inspect one child task's status, heartbeat, deadline, health, and events.

            This never includes the researcher output. Use get_researcher_result only
            after result_available becomes true.
            """
            if not _owned_job_snapshot(job_id):
                return _not_owned(job_id)
            _async_refresh_artifact_counts(str(job_id or "").strip())
            job = _owned_job_snapshot(job_id)
            if not job:
                return _not_owned(job_id)
            task_key = str(task_id or "").strip()
            task = (job.get("tasks") or {}).get(task_key)
            if not task:
                return _compact_json(
                    {
                        "job_found": True,
                        "task_found": False,
                        "job_id": job_id,
                        "task_id": task_id,
                        "error": "Unknown researcher task id for this job.",
                    }
                )
            row = _compact_task_status(task_key, task, now=time.time())
            row.update(
                {
                    "created_at": task.get("created_at"),
                    "started_at": task.get("started_at"),
                    "finished_at": task.get("finished_at"),
                    "deadline_seconds": task.get("deadline_seconds", child_timeout_seconds),
                }
            )
            if task.get("error"):
                row["error"] = task.get("error")
            result = task.get("result") if isinstance(task.get("result"), dict) else {}
            counts = result.get("tool_call_counts") or task.get("live_tool_call_counts") or {}
            if counts:
                row["tool_call_counts"] = dict(sorted(counts.items()))
                row["total_tool_calls"] = int(result.get("total_tool_calls") or sum(int(v) for v in counts.values()))
            event_limit = max(0, min(int(recent_event_limit or 0), 50))
            if event_limit:
                row["recent_events"] = (task.get("recent_events") or [])[-event_limit:]
            return _compact_json(
                {
                    "job_found": True,
                    "task_found": True,
                    "job_id": job_id,
                    "task": row,
                    "outputs_included": False,
                }
            )

        @function_tool(name_override="poll_researchers_async")
        def poll_researchers_async(job_id: str, include_outputs: bool = False, wait_seconds: int = 0) -> str:
            """Poll an asynchronous researcher job.

            Args:
                job_id: The id returned by start_researchers_async.
                include_outputs: Defaults to false. If explicitly true, include one
                    bounded digest per parsed completed task, otherwise raw text when
                    parsing failed, never both and never the full review; prefer
                    get_researcher_result for one bounded result at a time.
                wait_seconds: Optional completion-aware seconds to wait before polling,
                    clamped to 0-2100. Use 300-1800 for ChatGPT browser jobs and
                    30-120 for ordinary researchers. The call returns early when every
                    task reaches a terminal state.

            Output: Compact JSON with job status, per-task status/latest_action/timing, and
            elapsed_seconds, idle_seconds since the last observed event, recent tool
            events/live call counts while running, plus completed researcher
            results/tool_call_counts when available.
            """
            job_key = str(job_id or "").strip()
            job = _owned_job_snapshot(job_key)
            if not job:
                return _not_owned(job_id)
            wait = max(0, min(int(wait_seconds or 0), 2100))
            wait_started = time.monotonic()
            if wait:
                initial_tasks = (job.get("tasks") or {}).values()
                already_complete = bool(job.get("tasks")) and all(
                    str(task.get("status") or "") in _RESEARCHER_TERMINAL_STATUSES
                    for task in initial_tasks
                )
                if not already_complete:
                    _async_wait_for_completion(job_key, wait)
            waited = round(time.monotonic() - wait_started, 3)
            _async_refresh_artifact_counts(job_key)
            job = _owned_job_snapshot(job_key)
            if not job:
                return _not_owned(job_id)
            tasks = []
            now = time.time()
            for task_id, task in sorted((job.get("tasks") or {}).items()):
                row = _compact_task_status(task_id, task, now=now)
                row["started_at"] = task.get("started_at")
                row["finished_at"] = task.get("finished_at")
                row["deadline_seconds"] = task.get("deadline_seconds", child_timeout_seconds)
                if task.get("error"):
                    row["error"] = task.get("error")
                result = task.get("result") or {}
                if result:
                    if result.get("tool_call_counts") is not None:
                        row["tool_call_counts"] = result.get("tool_call_counts") or {}
                    if result.get("total_tool_calls") is not None:
                        row["total_tool_calls"] = result.get("total_tool_calls") or 0
                    if include_outputs:
                        projected_result = _poll_result_projection(result)
                        if projected_result:
                            row["result"] = projected_result
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
            complete = bool(tasks) and all(_async_task_is_terminal(task) for task in tasks)
            if complete:
                next_step = (
                    "Use get_researcher_result on each relevant done task, preferably view=summary first; "
                    "request parsed/raw pages only when needed, then synthesize or launch a focused follow-up."
                )
            elif any(s == "running" for s in statuses):
                next_step = (
                    "Some researchers are running. Continue with completion-aware wait_seconds=300-1800; cancel only duplicated, clearly stalled, or no-longer-useful tasks."
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
                    "outputs_included": bool(include_outputs),
                    "next_step": next_step,
                }
            )

        @function_tool(name_override="get_researcher_result")
        def get_researcher_result(
            job_id: str,
            task_id: str,
            view: str = "summary",
            offset: int = 0,
            max_chars: int = 8000,
        ) -> str:
            """Read exactly one completed researcher result without bloating status polls.

            Args:
                job_id: Job id returned by start/list_researcher_jobs.
                task_id: Exact task id returned by start/list/poll.
                view: metadata, summary, parsed, or raw. Start with metadata/summary.
                    parsed and raw are lossless and paginated.
                offset: Character offset for the selected view.
                max_chars: Page size, clamped to 500-12000 characters.

            Output: Compact metadata or one content page with next_offset, total_chars,
            complete, and SHA-256 so every page can be retrieved without data loss.
            """
            job = _owned_job_snapshot(job_id)
            if not job:
                return _not_owned(job_id)
            task = (job.get("tasks") or {}).get(str(task_id or "").strip())
            if not task:
                return _compact_json(
                    {
                        "job_found": True,
                        "task_found": False,
                        "job_id": job_id,
                        "task_id": task_id,
                        "error": "Unknown researcher task id for this job.",
                    }
                )
            result = task.get("result") if isinstance(task.get("result"), dict) else {}
            parsed = result.get("parsed_response")
            raw_value = result.get("output")
            raw_text = "" if raw_value is None else (
                raw_value if isinstance(raw_value, str) else _compact_json(raw_value)
            )
            parsed_text = "" if parsed is None else _compact_json(parsed)
            available_views = _available_result_views(task)
            metadata = {
                "job_found": True,
                "task_found": True,
                "job_id": job_id,
                "task_id": task_id,
                "researcher": task.get("researcher", ""),
                "researcher_tool": task.get("researcher_tool", ""),
                "status": task.get("status", "unknown"),
                "health": _async_task_health(task),
                "result_available": bool(result),
                "available_views": available_views,
                "tool_call_counts": result.get("tool_call_counts") or {},
                "total_tool_calls": int(result.get("total_tool_calls") or 0),
                "failure_reason": task.get("failure_reason", ""),
                "view_char_counts": {
                    **({"raw": len(raw_text)} if raw_value is not None else {}),
                    **({"parsed": len(parsed_text)} if parsed is not None else {}),
                },
            }
            selected_view = str(view or "summary").strip().lower()
            if selected_view == "metadata":
                return _compact_json(metadata)
            if not result:
                metadata["error"] = "This task has no completed result. Inspect status/failure_reason or wait for completion."
                return _compact_json(metadata)
            if selected_view == "raw":
                if raw_value is None:
                    metadata["error"] = "Raw output is unavailable for this result."
                    return _compact_json(metadata)
                content = raw_text
            elif selected_view == "parsed":
                if parsed is None:
                    metadata["error"] = "Parsed researcher JSON is unavailable; use raw or inspect failure metadata."
                    return _compact_json(metadata)
                content = parsed_text
            elif selected_view == "summary":
                parsed_row = parsed if isinstance(parsed, dict) else {}
                summary = compact_researcher_digest(parsed_row)
                summary["lossless_views"] = [name for name in ("parsed", "raw") if name in available_views]
                content = _compact_json(summary)
            else:
                metadata["error"] = "view must be one of: metadata, summary, parsed, raw."
                return _compact_json(metadata)
            start = max(0, min(int(offset or 0), len(content)))
            page_size = max(500, min(int(max_chars or 8000), 12000))
            end = min(len(content), start + page_size)
            return _compact_json(
                {
                    **metadata,
                    "view": selected_view,
                    "offset": start,
                    "next_offset": end if end < len(content) else None,
                    "total_chars": len(content),
                    "complete": end >= len(content),
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "content": content[start:end],
                }
            )

        @function_tool(name_override="cancel_researcher_task")
        def cancel_researcher_task(job_id: str, task_id: str, reason: str) -> str:
            """Cancel one queued/running ordinary researcher while preserving siblings.

            A concrete reason of at least 20 characters is required. Running ChatGPT
            browser tasks are protected and remain owned by their hard deadline; queued
            browser tasks may still be cancelled before they start.
            """
            if not _owned_job_snapshot(job_id):
                return _not_owned(job_id)
            clean_reason = " ".join(str(reason or "").split())
            if len(clean_reason) < 20:
                return _compact_json(
                    {
                        "job_found": True,
                        "task_found": True,
                        "job_id": job_id,
                        "task_id": task_id,
                        "cancellation_requested": False,
                        "error": "Cancellation reason must contain at least 20 characters.",
                    }
                )
            return _compact_json(
                _async_cancel_task(job_id, task_id, reason=clean_reason)
            )

        @function_tool(name_override="retry_researcher_task")
        def retry_researcher_task(job_id: str, task_id: str, reason: str) -> str:
            """Retry one failed/cancelled/deadline child once using its original prompt.

            The original prompt is retained privately and is never returned by status
            tools. A material retry reason of at least 80 characters is required. A
            successful task cannot be retried, and each task lineage gets at most one
            automatic retry to prevent runaway cost.
            """
            if not _owned_job_snapshot(job_id):
                return _not_owned(job_id)
            clean_reason = " ".join(str(reason or "").split())
            if len(clean_reason) < 80:
                return _compact_json(
                    {
                        "job_found": True,
                        "task_found": True,
                        "job_id": job_id,
                        "task_id": task_id,
                        "retry_started": False,
                        "error": "Retry reason must contain at least 80 characters describing the material gap or transient failure.",
                    }
                )
            job_key = str(job_id or "").strip()
            task_key = str(task_id or "").strip()
            with _ASYNC_RESEARCH_LOCK:
                raw_job = _ASYNC_RESEARCH_JOBS.get(job_key)
                raw_task = (raw_job or {}).get("tasks", {}).get(task_key)
                if not raw_task:
                    return _compact_json(
                        {
                            "job_found": True,
                            "task_found": False,
                            "job_id": job_id,
                            "task_id": task_id,
                            "retry_started": False,
                            "error": "Unknown researcher task id for this job.",
                        }
                    )
                status = str(raw_task.get("status") or "unknown")
                result = raw_task.get("result") if isinstance(raw_task.get("result"), dict) else {}
                parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else {}
                retryable = status in {"error", "cancelled", "deadline_exceeded"} or parsed.get("research_worked") is False
                if not retryable:
                    return _compact_json(
                        {
                            "job_found": True,
                            "task_found": True,
                            "job_id": job_id,
                            "task_id": task_id,
                            "status": status,
                            "retry_started": False,
                            "error": "Only failed, cancelled, deadline-exceeded, or research_worked=false tasks may be retried.",
                        }
                    )
                retry_count = int(raw_task.get("retry_count") or 0)
                if retry_count >= 1 or bool(raw_task.get("_retry_claimed")):
                    return _compact_json(
                        {
                            "job_found": True,
                            "task_found": True,
                            "job_id": job_id,
                            "task_id": task_id,
                            "retry_started": False,
                            "error": "This task lineage already used or claimed its one automatic retry.",
                        }
                    )
                original_prompt = str(raw_task.get("_request_prompt") or "").strip()
                researcher = str(raw_task.get("researcher") or "").strip()
                save_artifacts = bool((raw_job or {}).get("save_artifacts"))
                if not original_prompt or not researcher:
                    return _compact_json(
                        {
                            "job_found": True,
                            "task_found": True,
                            "job_id": job_id,
                            "task_id": task_id,
                            "retry_started": False,
                            "error": "Original researcher request is unavailable for a lossless retry.",
                        }
                    )
                raw_task["_retry_claimed"] = True
            retry_prompt = f"{original_prompt}\n\nDuplicate reason: Retry the original task because {clean_reason}"
            launched_text = self._invoke_tool_sync(
                start_researchers_async,
                {
                    "requests_json": _compact_json([{"researcher": researcher, "prompt": retry_prompt}]),
                    "save_artifacts": save_artifacts,
                    "max_parallel": 1,
                },
            )
            try:
                launched = json.loads(str(launched_text or "{}"))
            except json.JSONDecodeError:
                launched = {"async_started": False, "error": "Retry launcher returned non-JSON output."}
            if not bool(launched.get("async_started")):
                with _ASYNC_RESEARCH_LOCK:
                    source = ((_ASYNC_RESEARCH_JOBS.get(job_key) or {}).get("tasks") or {}).get(task_key)
                    if source is not None:
                        source.pop("_retry_claimed", None)
                return _compact_json(
                    {
                        "retry_started": False,
                        "retried_from_job_id": job_id,
                        "retried_from_task_id": task_id,
                        "launcher": launched,
                    }
                )
            retry_job_id = str(launched.get("job_id") or "")
            retry_tasks = launched.get("tasks") if isinstance(launched.get("tasks"), list) else []
            retry_task_id = str((retry_tasks[0] if retry_tasks else {}).get("task_id") or "")
            with _ASYNC_RESEARCH_LOCK:
                source = ((_ASYNC_RESEARCH_JOBS.get(job_key) or {}).get("tasks") or {}).get(task_key)
                retry_task = ((_ASYNC_RESEARCH_JOBS.get(retry_job_id) or {}).get("tasks") or {}).get(retry_task_id)
                if source is not None:
                    source["retry_spawned_job_id"] = retry_job_id
                    source["retry_spawned_task_id"] = retry_task_id
                    source["latest_action"] = "one retry launched"
                if retry_task is not None:
                    retry_task["retry_count"] = retry_count + 1
                    retry_task["retried_from_job_id"] = job_key
                    retry_task["retried_from_task_id"] = task_key
            _persist_async_job_ledger(job_key)
            return _compact_json(
                {
                    **launched,
                    "retry_started": True,
                    "retried_from_job_id": job_id,
                    "retried_from_task_id": task_id,
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
            if not _owned_job_snapshot(job_id):
                return _not_owned(job_id)
            return _compact_json(_async_cancel_job(job_id))

        for tool in (
            start_researchers_async,
            list_researcher_jobs,
            get_researcher_task,
            poll_researchers_async,
            get_researcher_result,
            cancel_researcher_task,
            retry_researcher_task,
            cancel_researchers_async,
        ):
            tool.description = f"{tool.description}\n\nOutput: Compact JSON."
        return [
            start_researchers_async,
            list_researcher_jobs,
            get_researcher_task,
            poll_researchers_async,
            get_researcher_result,
            cancel_researcher_task,
            retry_researcher_task,
            cancel_researchers_async,
        ]

    def _build_subagent_tools(self, enabled_researchers: list[str], artifact_root: str = ""):
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

        # Every ordinary researcher must enter through the supervised batch
        # boundary. Exposing its raw tool here lets the model bypass the
        # process-group deadline and is unsafe for long calls such as travel.
        # Browser researchers are already async-only for the same reason.
        tools = [
            tool
            for tool in all_tools
            if self._name_of_tool(tool) == "task_steps_manager"
        ]
        if synchronous_researchers:
            synchronous_tools = {
                name: tool for name, tool in tools_by_name.items()
                if name in synchronous_tool_names
            }
            tools.append(self._build_batch_tool(synchronous_tools, synchronous_researchers))
        async_tools = self._build_async_tools(
            tools_by_name,
            enabled_researchers,
            artifact_root=artifact_root,
        )
        tools.extend(async_tools)
        add_research_artifact_tools(tools, self.config, root=artifact_root)
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
            runtime_cap_minutes=self.runtime_cap_minutes,
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

        # Pin the administrator's artifact tools to its shared master folder.
        # Child researchers still receive ContextVar-scoped tools for their own
        # subfolders, but their execution must never redirect the administrator's
        # list/read/grep operations to the last child workspace.
        # Warm once per administrator run, before the model's hard deadline
        # starts. The child runner itself also tolerates direct callers.
        _warm_researcher_process_context()
        tools = self._build_subagent_tools(enabled_researchers, artifact_root=master_dir)
        if not tools:
            return "ERROR: no researcher tools available for researcher_administrator."

        def _owned_async_job_ids() -> list[str]:
            """Merge run-local accounting with jobs found by workspace ownership."""
            seen: set[str] = set()
            merged: list[str] = []
            for job_id in [
                *self._launched_async_job_ids,
                *_async_job_ids_for_evidence_dir(master_dir),
            ]:
                normalized = str(job_id or "").strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    merged.append(normalized)
            return merged

        # The administrator's model can stop after writing a synthesis even
        # though a completion-aware researcher job is still running.  Do not
        # harvest that partial snapshot: the queue's required-researcher check
        # must observe terminal task results, and the evidence writer must not
        # outlive the queue call.  The deadline includes the model run itself,
        # preserving the configured queue/administrator runtime cap.
        # Reserve a final synthesis window for the administrator itself. The
        # administrator model receives normal budget warnings and, on MCP
        # backends, can inspect `check_budget_status`; an instruction alone is
        # not a hard guarantee, so child researchers are bounded in code.
        synthesis_reserve_minutes = max(
            0,
            int(
                getattr(
                    self.config,
                    "researcher_administrator_synthesis_reserve_minutes",
                    _DEFAULT_SYNTHESIS_RESERVE_MINUTES,
                )
                or _DEFAULT_SYNTHESIS_RESERVE_MINUTES
            ),
        )
        admin_runtime_seconds = int(effective_runtime_minutes) * 60
        synthesis_reserve_seconds = min(
            admin_runtime_seconds,
            synthesis_reserve_minutes * 60,
        )
        researcher_deadline = (
            time.monotonic() + max(0, admin_runtime_seconds - synthesis_reserve_seconds)
            if int(effective_runtime_minutes or 0) > 0
            else float("inf")
        )
        admin_deadline = (
            time.monotonic() + admin_runtime_seconds
            if int(effective_runtime_minutes or 0) > 0
            else float("inf")
        )

        def _harvest_async_jobs() -> list[dict[str, str]]:
            job_ids = _owned_async_job_ids()
            pending = _wait_for_async_jobs_terminal(job_ids, researcher_deadline)
            if pending:
                # At the researcher deadline, stop queued/running work and
                # leave the reserved window to the administrator's synthesis.
                # fail-closed and keep any partial artifacts for diagnosis.
                for job_id in pending:
                    _async_cancel_job(job_id)
                pending = _wait_for_async_jobs_terminal(job_ids, time.monotonic() + 5)
            if not pending:
                return []
            failures: list[dict[str, str]] = []
            for job_id in pending:
                snapshot = _async_job_snapshot(job_id) or {}
                for task in (snapshot.get("tasks") or {}).values():
                    status = str(task.get("status") or "unknown")
                    if status in _RESEARCHER_TERMINAL_STATUSES:
                        continue
                    failures.append(
                        {
                            "researcher_tool": str(task.get("researcher_tool") or "unknown"),
                            "status": "deadline_exceeded",
                            "task_id": str(task.get("task_id") or ""),
                            "failure_reason": (
                                f"Async researcher job {job_id} remained non-terminal at the "
                                f"administrator deadline (status={status})."
                            ),
                        }
                    )
            return failures

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
        researcher_window_minutes = (
            max(0, int(effective_runtime_minutes) - synthesis_reserve_minutes)
            if int(effective_runtime_minutes or 0) > 0
            else 0
        )
        time_budget_line = (
            f"Time budget: administrator hard cap is {int(effective_runtime_minutes)} minutes; "
            f"researcher phase has a hard stop after {researcher_window_minutes} minutes, "
            f"leaving {synthesis_reserve_minutes} minutes reserved for your own synthesis. "
            "Do not launch new researchers once the researcher phase ends; collect terminal results and conclude.\n"
            if int(effective_runtime_minutes or 0) > 0
            else "Time budget: no administrator runtime cap is configured for this run.\n"
        )
        master_line = (
            f"Evidence workspace (preserved; runtime appends the path): {master_dir}\n"
            if save_artifacts
            else f"Evidence workspace (temporary; do not report its path): {master_dir}\n"
        )
        required_line = (
            "Required successful researchers: "
            + ", ".join(self.required_researchers)
            + ". This is a hard orchestration gate, not a suggestion: across the initial "
            "short-work batch and long-work async request(s), every listed researcher must "
            "be included in the initial orchestration wave. After that gate is met, targeted "
            "follow-ups and one-task retries may run without duplicating every required researcher. "
            "Launch and await each; `research_worked` must be false if any fails.\n"
            if self.required_researchers
            else ""
        )

        admin_context = (
            "\n\n### RUN CONFIGURATION\n"
            f"Available tools: {available_line}. Do not call anything absent from this list.\n"
            f"{chatgpt_priority_line}"
            f"{required_line}"
            f"{budget_line}"
            f"{time_budget_line}"
            f"{master_line}"
            "Researchers share evidence in `<workspace>/<researcher-short-name>`; use artifact list/read/grep tools when inspection is useful.\n"
            "Researcher capabilities:\n"
            + "\n".join(capability_lines)
            + f"\nExecution supports up to {MAX_RESEARCHER_PARALLELISM} concurrent child researchers. Use "
            "`run_researchers_batch` for short work and "
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
                # The administrator's researcher queue may execute in a separate
                # MCP process where _CURRENT_RESEARCH_DEADLINE is absent. Export
                # the same deadline through the child environment so sync and
                # async orchestration keep one hard wall-clock budget.
                _RESEARCHER_DEADLINE_EPOCH_ENV: (
                    str(
                        time.time()
                        + max(0.0, researcher_deadline - time.monotonic())
                    )
                    if researcher_deadline != float("inf")
                    else ""
                ),
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

        def _recover_synthesis_only(
            failed_output: str,
            completed_responses: list[dict[str, Any]],
        ) -> str:
            """Retry only the administrator synthesis after a provider final error.

            The researcher calls have already completed (or been recorded as
            failures).  Keep this recovery bounded and digest-only: it must not
            relaunch a researcher, receive raw/full researcher output, or extend
            the administrator's hard deadline.
            """
            if admin_deadline != float("inf") and time.monotonic() >= admin_deadline - 10:
                return ""
            digests = [
                compact_researcher_digest(response)
                for response in completed_responses
                if isinstance(response, dict)
            ]
            if not digests:
                return ""
            recovery_prompt = (
                "The previous administrator provider call ended with a final-result error. "
                "The researcher work is already complete and is listed below as bounded digests. "
                "Do not call tools, do not launch or retry researchers, and do not ask for raw/full "
                "researcher output. Produce exactly one JSON object matching the administrator output "
                "schema. Write a substantive evidence-weighted administrator_conclusions synthesis "
                "with concrete claims, contradictions, confidence, and gaps. Set research_worked true "
                "only if every required researcher digest is successful; otherwise set it false and "
                "explain the missing researcher.\n\n"
                f"Previous provider diagnostic (not evidence): {str(failed_output or '')[:1200]}\n\n"
                "BOUNDed researcher digests:\n"
                + json.dumps(digests, ensure_ascii=False, separators=(",", ":"))
            )
            recovery_overrides = {
                "agent": {
                    "output_schema_json": researcher_administrator_output_schema(
                        preserve_artifacts=save_artifacts
                    ),
                    "output_schema_name": "researcher_administrator_recovery_result",
                    "output_schema_strict": True,
                    "max_runtime_minutes": max(
                        1,
                        min(
                            int(effective_runtime_minutes or 1),
                            max(1, int((admin_deadline - time.monotonic()) / 60))
                            if admin_deadline != float("inf")
                            else int(effective_runtime_minutes or 1),
                        ),
                    ),
                    "max_cost_usd": effective_cost_usd,
                    "sub_action": "researcher_administrator_recovery",
                },
                "session": {
                    "max_turns": max(2, min(8, int(effective_max_turns or 8))),
                    "memory_max_messages": parent_memory_max_messages,
                    "memory_reset_to_messages": parent_memory_reset_to_messages,
                    "long_term_memory_enabled": False,
                },
                "tools": {"max_tools_used": 0},
                "env": {
                    "CHACK_RESEARCH_MASTER_DIR": master_dir,
                    "CHACK_RESEARCH_DATA_DIR": master_dir,
                    "CHACK_RESEARCH_SAVE_ARTIFACTS": "1" if save_artifacts else "0",
                },
            }
            recovery_config = build_subagent_config(
                self.config,
                model_name=model_name,
                model_provider=self.model_provider,
                max_turns=max(2, min(8, int(effective_max_turns or 8))),
                system_prompt=_ADMINISTRATOR_SYSTEM_PROMPT,
                overrides=recovery_overrides,
            )
            try:
                recovery_result = Chack(recovery_config).run(
                    session_id=create_subagent_session_id(
                        "researcher_administrator_recovery", parent_root_session_id
                    ),
                    text=recovery_prompt,
                    min_tools_used_override=0,
                    max_tools_used_override=0,
                    enable_self_critique=False,
                    require_task_steps_manager_init_first=False,
                    tools_override=[],
                    system_prompt_override=recovery_config.system_prompt,
                    usage_session_id=parent_task_session_id,
                )
                recovered = str(recovery_result.output or "").strip()
                parsed_recovered = _json_from_output(recovered)
                if (
                    isinstance(parsed_recovered, dict)
                    and parsed_recovered.get("research_worked") is True
                    and _administrator_synthesis_is_valid(parsed_recovered)
                ):
                    return recovered
            except Exception:
                return ""
            return ""

        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("researcher_administrator", parent_root_session_id)

        prev_master = os.environ.get("CHACK_RESEARCH_MASTER_DIR")
        from chack_agent import Chack
        chack = Chack(config)
        artifact_context_tokens = set_research_artifact_context(master_dir, master_dir)
        collector_token, researcher_responses = begin_researcher_response_collection()
        research_deadline_token = _CURRENT_RESEARCH_DEADLINE.set(
            researcher_deadline if researcher_deadline != float("inf") else None
        )
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
                async_deadline_failures = _harvest_async_jobs()
                combined_responses = list(researcher_responses or [])
                combined_responses.extend(_researcher_responses_from_async_jobs(_owned_async_job_ids()))
                combined_responses.extend(_researcher_responses_from_async_output_files(master_dir))
                combined_failures = _researcher_failures_from_async_jobs(_owned_async_job_ids())
                combined_failures.extend(async_deadline_failures)
                combined_tool_counts = _researcher_call_counts_from_async_jobs(_owned_async_job_ids())
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
                async_deadline_failures = _harvest_async_jobs()
                combined_responses = list(researcher_responses or [])
                combined_responses.extend(_researcher_responses_from_async_jobs(_owned_async_job_ids()))
                combined_responses.extend(_researcher_responses_from_async_output_files(master_dir))
                combined_failures = _researcher_failures_from_async_jobs(_owned_async_job_ids())
                combined_failures.extend(async_deadline_failures)
                combined_tool_counts = _researcher_call_counts_from_async_jobs(_owned_async_job_ids())
                combined_tool_counts.update(self._launched_researcher_tool_counts())
                recovered_output = _recover_synthesis_only(output, combined_responses)
                if recovered_output:
                    return finalize_researcher_administrator_output(
                        recovered_output,
                        evidence_dir=master_dir,
                        save_artifacts=save_artifacts,
                        researcher_responses=combined_responses,
                        researcher_failures=combined_failures,
                        tool_counts=combined_tool_counts,
                        steps=result.all_steps,
                        required_researchers=self.required_researchers,
                    )
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
            async_deadline_failures = _harvest_async_jobs()
            tool_counts = result.tool_counts.copy()
            tool_counts.update(_researcher_call_counts_from_async_jobs(_owned_async_job_ids()))
            tool_counts.update(self._launched_researcher_tool_counts())
            combined_responses = list(researcher_responses or [])
            combined_responses.extend(_researcher_responses_from_async_jobs(_owned_async_job_ids()))
            combined_responses.extend(_researcher_responses_from_async_output_files(master_dir))
            combined_failures = _researcher_failures_from_async_jobs(_owned_async_job_ids())
            combined_failures.extend(async_deadline_failures)
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
            _CURRENT_RESEARCH_DEADLINE.reset(research_deadline_token)
            end_researcher_response_collection(collector_token)
            # Timed-out children may still be unwinding registered subprocesses or
            # cancellation-aware HTTP calls. Defer deletion until the last writer
            # exits; the final writer performs the pending cleanup exactly once.
            _cleanup_research_artifacts_when_idle(
                master_dir,
                save_artifacts=save_artifacts,
            )
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
