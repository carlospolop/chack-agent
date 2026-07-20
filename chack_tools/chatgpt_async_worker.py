"""Outbound-only workstation worker for the authenticated ChatGPT job broker."""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

from .chatgpt_async_client import ChatGPTAsyncApiClient
from .chatgpt_research_agents import ChatGPTWebResearchAgentTool
from .config import ToolsConfig


class BrokerWorkerError(RuntimeError):
    """A safe, credential-free broker worker failure."""


class ChatGPTAsyncWorker:
    """Lease remote jobs, execute them on local authenticated Chrome, and return results."""

    def __init__(
        self,
        *,
        api_url: str,
        worker_secret: str,
        cdp_url: str = "http://127.0.0.1:9226",
        worker_id: str = "",
        poll_seconds: int = 10,
        heartbeat_seconds: int = 45,
    ):
        try:
            self.client = ChatGPTAsyncApiClient(api_url, worker_secret)
        except ValueError as exc:
            raise BrokerWorkerError(str(exc)) from exc
        self.cdp_url = cdp_url.strip() or "http://127.0.0.1:9226"
        self.worker_id = (worker_id.strip() or f"workstation-{socket.gethostname()}")[:100]
        self.poll_seconds = max(2, int(poll_seconds))
        self.heartbeat_seconds = max(10, int(heartbeat_seconds))

    @classmethod
    def from_environment(cls) -> "ChatGPTAsyncWorker":
        return cls(
            api_url=os.environ.get("CHACK_CHATGPT_ASYNC_API_URL", ""),
            worker_secret=os.environ.get("CHACK_CHATGPT_ASYNC_WORKER_SECRET", ""),
            cdp_url=os.environ.get("CHACK_CHATGPT_CDP_URL", "http://127.0.0.1:9226"),
            worker_id=os.environ.get("CHACK_CHATGPT_WORKER_ID", ""),
            poll_seconds=int(os.environ.get("CHACK_CHATGPT_WORKER_POLL_SECONDS", "10")),
            heartbeat_seconds=int(os.environ.get("CHACK_CHATGPT_WORKER_HEARTBEAT_SECONDS", "45")),
        )

    @staticmethod
    def _progress_snapshot(run_state_path: Path, partial_path: Path) -> tuple[str, int, str]:
        stage = "browser_running"
        answer_chars = 0
        partial = ""
        try:
            state = json.loads(run_state_path.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                stage = str(state.get("terminal_state") or stage)[:200]
                answer_chars = max(0, int(state.get("answer_chars") or 0))
        except Exception:
            pass
        try:
            partial = partial_path.read_text(encoding="utf-8").strip()
            answer_chars = max(answer_chars, len(partial))
        except Exception:
            pass
        return stage, answer_chars, partial

    def _heartbeat_loop(
        self,
        *,
        job_id: str,
        lease_id: str,
        run_state_path: Path,
        partial_path: Path,
        stopped: threading.Event,
        cancelled: threading.Event,
    ) -> None:
        while not stopped.wait(self.heartbeat_seconds):
            stage, answer_chars, partial = self._progress_snapshot(run_state_path, partial_path)
            try:
                response = self.client.heartbeat(
                    job_id,
                    lease_id=lease_id,
                    stage=stage,
                    answer_chars=answer_chars,
                    partial_result=partial,
                )
                if response.get("cancel_requested"):
                    cancelled.set()
            except Exception as exc:
                print(json.dumps({"event": "chatgpt_worker_heartbeat_failed", "job_id": job_id, "error": type(exc).__name__}), flush=True)

    def _complete(self, job_id: str, payload: dict[str, Any]) -> None:
        self.client.complete(job_id, **payload)

    def execute_lease(self, lease: dict[str, Any]) -> str:
        job_id = str(lease.get("job_id") or "")
        lease_id = str(lease.get("lease_id") or "")
        mode = str(lease.get("mode") or "").lower()
        prompt = str(lease.get("prompt") or "")
        timeout_seconds = max(60, int(lease.get("output_timeout_seconds") or (1800 if mode == "pro" else 4500)))
        if not job_id.startswith("job_") or not lease_id or mode not in {"pro", "deep"} or len(prompt.strip()) < 20:
            raise BrokerWorkerError("The broker returned an invalid lease.")

        config = ToolsConfig(
            chatgpt_cdp_url=self.cdp_url,
            chatgpt_pro_timeout_seconds=timeout_seconds if mode == "pro" else None,
            chatgpt_deep_timeout_seconds=timeout_seconds if mode == "deep" else None,
        )
        helper = ChatGPTWebResearchAgentTool(config, mode=mode)
        with tempfile.TemporaryDirectory(prefix="chack-chatgpt-worker-") as temporary:
            root = Path(temporary)
            run_state_path = root / "chatgpt-run.json"
            partial_path = root / "partial.md"
            stopped = threading.Event()
            cancelled = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                kwargs={
                    "job_id": job_id,
                    "lease_id": lease_id,
                    "run_state_path": run_state_path,
                    "partial_path": partial_path,
                    "stopped": stopped,
                    "cancelled": cancelled,
                },
                daemon=True,
            )
            heartbeat.start()
            try:
                answer, _conversation_url, metadata = helper._browser_research(
                    prompt,
                    run_state_path=run_state_path,
                    partial_path=partial_path,
                )
                status = "CANCELLED" if cancelled.is_set() else "SUCCEEDED"
                self._complete(
                    job_id,
                    {
                        "lease_id": lease_id,
                        "status": status,
                        "result": answer if status == "SUCCEEDED" else "",
                        "partial_result": answer if status == "CANCELLED" else "",
                        "metadata": {
                            **{key: value for key, value in metadata.items() if key != "conversation_url"},
                            "execution_backend": "local_worker",
                        },
                        "error_code": "CANCEL_REQUESTED" if status == "CANCELLED" else "",
                        "error_message": "The client cancelled this research job." if status == "CANCELLED" else "",
                    },
                )
            except Exception as exc:
                stage, _answer_chars, partial = self._progress_snapshot(run_state_path, partial_path)
                status = "TIMED_OUT" if stage == "timeout" or "deadline" in str(exc).lower() else "FAILED"
                self._complete(
                    job_id,
                    {
                        "lease_id": lease_id,
                        "status": status,
                        "result": "",
                        "partial_result": partial,
                        "metadata": {
                            "mode": mode,
                            "terminal_state": stage,
                            "execution_backend": "local_worker",
                            "answer_chars": len(partial),
                        },
                        "error_code": type(exc).__name__[:100],
                        "error_message": str(exc)[:1000],
                    },
                )
            finally:
                stopped.set()
                heartbeat.join(timeout=5)
        return job_id

    def run_once(self) -> bool:
        lease = self.client.lease(worker_id=self.worker_id)
        if not lease:
            return False
        job_id = self.execute_lease(lease)
        print(json.dumps({"event": "chatgpt_worker_job_completed", "job_id": job_id}), flush=True)
        return True

    def run_forever(self) -> None:
        print(json.dumps({"event": "chatgpt_worker_started", "worker_id": self.worker_id}), flush=True)
        while True:
            try:
                if not self.run_once():
                    time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                return
            except Exception as exc:
                print(json.dumps({"event": "chatgpt_worker_error", "error": type(exc).__name__}), flush=True)
                time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the outbound ChatGPT browser worker for the authenticated broker.")
    parser.add_argument("--once", action="store_true", help="Lease at most one job, then exit.")
    args = parser.parse_args()
    worker = ChatGPTAsyncWorker.from_environment()
    if args.once:
        worker.run_once()
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
