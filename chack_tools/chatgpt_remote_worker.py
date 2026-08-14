"""Outbound-only workstation worker for cloud-brokered ChatGPT Web jobs.

The worker never opens a listening port. It leases a bounded number of jobs over
HTTPS, runs each in an isolated tab and per-job evidence directory, heartbeats
partial progress, and posts terminal results. Its execution backend is forced to
``local`` to prevent recursive submission through the broker.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import fcntl
import json
import logging
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, cast

from .chatgpt_async_client import ChatGPTAsyncApiClient, ChatGPTAsyncApiError
from .chatgpt_research_agents import (
    _XHIGH_COMPAT_PROMPT_PREFIX,
    ChatGPTWebResearchAgentTool,
    Mode,
)
from .config import ToolsConfig

LOG = logging.getLogger("chack-chatgpt-worker")
_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

_PUBLIC_METADATA_FIELDS = {
    "mode",
    "started_at",
    "finished_at",
    "answer_chars",
    "terminal_state",
    "stage",
    "forced_answer",
    "execution_backend",
    "prior_browser_submission_detected",
    "selected_effort",
    "selected_power",
    "selector_ui",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        return ""


def _write_secure_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _sanitized_error(exc: Exception, cdp_url: str) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = message.replace(str(Path.home()), "<home>")
    message = re.sub(r"/home/[^/\s]+", "<home>", message)
    message = re.sub(r"/tmp/[^\s]+", "<temporary-path>", message)
    if cdp_url:
        message = message.replace(cdp_url, "<local-cdp>")
    return message[:1000]


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return broker-safe metadata without local paths or conversation URLs."""
    return {
        key: value
        for key, value in metadata.items()
        if key in _PUBLIC_METADATA_FIELDS and isinstance(value, (str, int, float, bool))
    }


class ChatGPTRemoteWorker:
    def __init__(self) -> None:
        api_url = os.environ.get("CHACK_CHATGPT_ASYNC_API_URL", "").strip()
        worker_secret = os.environ.get("CHACK_CHATGPT_ASYNC_WORKER_SECRET", "").strip()
        self.client = ChatGPTAsyncApiClient(
            api_url,
            worker_secret,
            request_timeout_seconds=int(os.environ.get("CHACK_CHATGPT_ASYNC_REQUEST_TIMEOUT_SECONDS", "30")),
        )
        self.cdp_url = os.environ.get("CHACK_CHATGPT_CDP_URL", "http://127.0.0.1:9226").strip()
        self.worker_id = os.environ.get("CHACK_CHATGPT_WORKER_ID", "").strip() or f"{socket.gethostname()}-{os.getpid()}"
        self.poll_seconds = max(2, int(os.environ.get("CHACK_CHATGPT_WORKER_POLL_SECONDS", "10")))
        self.heartbeat_seconds = max(10, int(os.environ.get("CHACK_CHATGPT_WORKER_HEARTBEAT_SECONDS", "30")))
        # Five is the deliberately tested workstation/browser ceiling. Keep a
        # hard cap even if a service environment is accidentally set higher.
        self.concurrency = max(1, min(5, int(os.environ.get("CHACK_CHATGPT_WORKER_CONCURRENCY", "1"))))
        self.state_root = Path(
            os.environ.get("CHACK_CHATGPT_WORKER_STATE_DIR", "~/.local/state/chack-chatgpt-worker")
        ).expanduser()
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._completion_lock = threading.Lock()

    def _config(self, mode: str, output_timeout_seconds: int) -> ToolsConfig:
        kwargs: dict[str, Any] = {
            "chatgpt_execution_backend": "local",
            "chatgpt_cdp_url": self.cdp_url,
            "chatgpt_research_poll_seconds": int(os.environ.get("CHACK_CHATGPT_BROWSER_POLL_SECONDS", "15")),
        }
        timeout_fields = {
            "pro": "chatgpt_pro_timeout_seconds",
            "xhigh": "chatgpt_xhigh_timeout_seconds",
            "deep": "chatgpt_deep_timeout_seconds",
        }
        field_name = timeout_fields.get(mode)
        if not field_name:
            raise ValueError(f"Unsupported ChatGPT worker mode: {mode}")
        kwargs[field_name] = output_timeout_seconds
        return ToolsConfig(**kwargs)

    def _pending_path(self, job_id: str) -> Path:
        return self.state_root / "jobs" / job_id / "pending-completion.json"

    def _send_or_save_completion(self, job_id: str, completion: dict[str, Any]) -> None:
        with self._completion_lock:
            pending = self._pending_path(job_id)
            try:
                self.client.complete(job_id, **completion)
                pending.unlink(missing_ok=True)
                LOG.info("job=%s completion accepted status=%s", job_id, completion.get("status"))
            except ChatGPTAsyncApiError as exc:
                _write_secure_json(pending, completion)
                if exc.error_code == "lease_lost":
                    rejected = pending.with_name("rejected-completion.json")
                    pending.replace(rejected)
                    LOG.error("job=%s completion rejected because lease was lost; local output preserved at %s", job_id, rejected)
                    return
                LOG.error("job=%s completion upload failed; preserved for retry (%s)", job_id, exc.error_code or type(exc).__name__)

    def flush_pending(self) -> None:
        with self._completion_lock:
            jobs_root = self.state_root / "jobs"
            if not jobs_root.exists():
                return
            for path in jobs_root.glob("*/pending-completion.json"):
                completion = _read_json(path)
                if not completion:
                    continue
                job_id = path.parent.name
                try:
                    self.client.complete(job_id, **completion)
                    path.unlink(missing_ok=True)
                    LOG.info("job=%s pending completion delivered", job_id)
                except ChatGPTAsyncApiError as exc:
                    if exc.error_code == "lease_lost":
                        path.replace(path.with_name("rejected-completion.json"))
                        LOG.error("job=%s pending completion rejected because lease was lost", job_id)
                    else:
                        LOG.warning("job=%s pending completion still unavailable", job_id)

    def _heartbeat_loop(
        self,
        *,
        stop: threading.Event,
        cancelled: threading.Event,
        job_id: str,
        lease_id: str,
        run_state_path: Path,
        partial_path: Path,
    ) -> None:
        last_partial = ""
        while not stop.wait(self.heartbeat_seconds):
            state = _read_json(run_state_path)
            partial = _read_text(partial_path)
            changed_partial = partial if partial and partial != last_partial else ""
            try:
                response = self.client.heartbeat(
                    job_id,
                    lease_id=lease_id,
                    stage=str(state.get("terminal_state") or state.get("stage") or "browser_running"),
                    answer_chars=max(len(partial), int(state.get("answer_chars") or 0)),
                    partial_result=changed_partial,
                )
                if changed_partial:
                    last_partial = partial
                if response.get("cancel_requested"):
                    cancelled.set()
            except ChatGPTAsyncApiError as exc:
                LOG.warning("job=%s heartbeat failed (%s)", job_id, exc.error_code or type(exc).__name__)

    def process(self, lease: dict[str, Any]) -> None:
        job_id = str(lease.get("job_id") or "")
        lease_id = str(lease.get("lease_id") or "")
        if not _JOB_ID_RE.fullmatch(job_id) or not lease_id:
            LOG.error("broker returned a lease with an invalid job or lease id")
            return
        job_dir = self.state_root / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = job_dir / ".execution.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            os.chmod(lock_path, 0o600)
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                LOG.error("job=%s duplicate active execution refused", job_id)
                self._send_or_save_completion(
                    job_id,
                    {
                        "lease_id": lease_id,
                        "status": "FAILED",
                        "metadata": {"terminal_state": "duplicate_active_execution"},
                        "error_code": "DUPLICATE_ACTIVE_EXECUTION",
                        "error_message": "Another worker execution already owns this job; duplicate browser submission was refused.",
                    },
                )
                return
            try:
                self._process_locked(lease)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _process_locked(self, lease: dict[str, Any]) -> None:
        job_id = str(lease.get("job_id") or "")
        lease_id = str(lease.get("lease_id") or "")
        mode = str(lease.get("mode") or "")
        prompt = str(lease.get("prompt") or "")
        if mode == "pro" and prompt.startswith(_XHIGH_COMPAT_PROMPT_PREFIX):
            mode = "xhigh"
            prompt = prompt[len(_XHIGH_COMPAT_PROMPT_PREFIX):]
        default_timeout = 4500 if mode == "deep" else 5400
        output_timeout = int(lease.get("output_timeout_seconds") or default_timeout)
        if not job_id or not lease_id or mode not in {"pro", "xhigh", "deep"} or not prompt:
            LOG.error("broker returned an invalid lease payload")
            return

        job_dir = self.state_root / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        request_path = job_dir / "request.txt"
        request_path.write_text(prompt, encoding="utf-8")
        os.chmod(request_path, 0o600)
        run_state_path = job_dir / "chatgpt-run.json"
        partial_path = job_dir / f"chatgpt-{mode}-partial.md"
        submission_marker = job_dir / "browser-submission-attempted.json"

        if lease.get("cancel_requested"):
            self._send_or_save_completion(
                job_id,
                {
                    "lease_id": lease_id,
                    "status": "CANCELLED",
                    "metadata": {"mode": mode, "terminal_state": "cancelled_before_launch"},
                    "error_code": "CANCELLED_BEFORE_LAUNCH",
                },
            )
            return

        existing_state = _read_json(run_state_path)
        if int(lease.get("attempt") or 1) > 1 and (existing_state.get("conversation_url") or submission_marker.exists()):
            metadata = _public_metadata(existing_state)
            metadata.update(
                {
                    "mode": mode,
                    "execution_backend": "local_worker",
                    "prior_browser_submission_detected": True,
                }
            )
            self._send_or_save_completion(
                job_id,
                {
                    "lease_id": lease_id,
                    "status": "FAILED",
                    "partial_result": _read_text(partial_path),
                    "metadata": metadata,
                    "error_code": "AMBIGUOUS_PRIOR_BROWSER_SUBMISSION",
                    "error_message": "A prior browser submission was detected; automatic resubmission was refused to avoid duplicate paid work.",
                },
            )
            return

        helper = ChatGPTWebResearchAgentTool(self._config(mode, output_timeout), mode=cast(Mode, mode))
        stop = threading.Event()
        cancelled = threading.Event()
        heartbeats = threading.Thread(
            target=self._heartbeat_loop,
            kwargs={
                "stop": stop,
                "cancelled": cancelled,
                "job_id": job_id,
                "lease_id": lease_id,
                "run_state_path": run_state_path,
                "partial_path": partial_path,
            },
            name=f"chatgpt-heartbeat-{job_id}",
            daemon=True,
        )
        try:
            self.client.heartbeat(job_id, lease_id=lease_id, stage="launching_browser", answer_chars=0)
            heartbeats.start()
            LOG.info("job=%s mode=%s launching local ChatGPT browser executor", job_id, mode)
            _write_secure_json(
                submission_marker,
                {"job_id": job_id, "lease_id": lease_id, "mode": mode, "attempted_at": time.time()},
            )
            answer, _conversation_url, metadata = helper._browser_research(
                prompt,
                run_state_path=run_state_path,
                partial_path=partial_path,
            )
            metadata = _public_metadata(dict(metadata or {}))
            metadata.update({"execution_backend": "local_worker"})
            completion_status = "CANCELLED" if cancelled.is_set() else "SUCCEEDED"
            self._send_or_save_completion(
                job_id,
                {
                    "lease_id": lease_id,
                    "status": completion_status,
                    "result": answer if completion_status == "SUCCEEDED" else "",
                    "partial_result": answer if completion_status == "CANCELLED" else "",
                    "metadata": metadata,
                    "error_code": "CANCEL_REQUESTED" if completion_status == "CANCELLED" else "",
                },
            )
        except Exception as exc:
            state = _read_json(run_state_path)
            partial = _read_text(partial_path)
            terminal_state = str(state.get("terminal_state") or "").lower()
            status = "TIMED_OUT" if terminal_state in {"timeout", "forcing_answer"} else "FAILED"
            error_code = "BROWSER_OUTPUT_TIMEOUT" if status == "TIMED_OUT" else "BROWSER_EXECUTION_FAILED"
            self._send_or_save_completion(
                job_id,
                {
                    "lease_id": lease_id,
                    "status": status,
                    "partial_result": partial,
                    "metadata": _public_metadata(state),
                    "error_code": error_code,
                    "error_message": _sanitized_error(exc, self.cdp_url),
                },
            )
        finally:
            stop.set()
            if heartbeats.is_alive():
                heartbeats.join(timeout=5)

    def run_once(self) -> bool:
        self.flush_pending()
        lease = self.client.lease(worker_id=self.worker_id)
        if not lease:
            return False
        self.process(lease)
        return True

    def run_forever(self) -> None:
        self._run_pool(stop_when_idle=False)

    @staticmethod
    def _collect_finished(active: set[Future[None]]) -> None:
        for future in tuple(active):
            if not future.done():
                continue
            active.remove(future)
            try:
                future.result()
            except Exception:
                # ``process`` contains per-job failure conversion, but keep the
                # pool alive if a truly unexpected programming error escapes.
                LOG.exception("unexpected parallel worker job failure")

    def _run_pool(self, *, stop_when_idle: bool) -> None:
        LOG.info(
            "worker=%s outbound ChatGPT worker started concurrency=%d",
            self.worker_id,
            self.concurrency,
        )
        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="chatgpt-browser-job") as executor:
            active: set[Future[None]] = set()
            while True:
                self._collect_finished(active)
                try:
                    self.flush_pending()
                    queue_empty = False
                    while len(active) < self.concurrency:
                        lease = self.client.lease(worker_id=self.worker_id)
                        if not lease:
                            queue_empty = True
                            break
                        active.add(executor.submit(self.process, lease))
                except ChatGPTAsyncApiError as exc:
                    LOG.warning("broker unavailable (%s); retrying", exc.error_code or type(exc).__name__)
                    queue_empty = True
                except Exception:
                    LOG.exception("unexpected worker leasing failure")
                    queue_empty = True

                if stop_when_idle and queue_empty and not active:
                    return
                if active:
                    wait(active, timeout=self.poll_seconds, return_when=FIRST_COMPLETED)
                    self._collect_finished(active)
                else:
                    time.sleep(self.poll_seconds)

    def run_until_idle(self) -> None:
        """Process all currently available jobs, then return (test/maintenance helper)."""
        self._run_pool(stop_when_idle=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Outbound-only worker for cloud-brokered ChatGPT Web jobs")
    parser.add_argument("--once", action="store_true", help="Lease at most one job and exit")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = ChatGPTRemoteWorker()
    if args.once:
        worker.run_once()
        return 0
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
