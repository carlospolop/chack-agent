"""HTTP client for the authenticated asynchronous ChatGPT broker."""

from __future__ import annotations

import threading
import time
import urllib.parse
from typing import Any

import requests


class ChatGPTAsyncApiError(RuntimeError):
    """A transport or application error returned by the async broker."""

    def __init__(self, message: str, *, status_code: int = 0, error_code: str = ""):
        super().__init__(message)
        self.status_code = int(status_code or 0)
        self.error_code = str(error_code or "")


class ChatGPTAsyncApiClient:
    def __init__(
        self,
        base_url: str,
        secret: str,
        *,
        request_timeout_seconds: int = 30,
        session: requests.Session | None = None,
    ):
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.secret = str(secret or "").strip()
        parsed = urllib.parse.urlparse(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("ChatGPT async API URL must be a clean HTTPS origin")
        if not self.secret:
            raise ValueError("ChatGPT async API secret is required")
        self.request_timeout_seconds = max(5, int(request_timeout_seconds or 30))
        # ``requests.Session`` does not promise thread safety. The outbound
        # workstation worker can execute several browser jobs concurrently, so
        # give every worker/heartbeat thread its own connection pool. Tests or
        # callers that explicitly inject a session retain the old behaviour.
        self.session = session
        self._thread_local = threading.local()

    def _http_session(self) -> requests.Session:
        if self.session is not None:
            return self.session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt, delay in enumerate((1, 2, 5), start=1):
            try:
                response = self._http_session().request(
                    method,
                    url,
                    headers={
                        "authorization": f"Bearer {self.secret}",
                        "content-type": "application/json",
                        "user-agent": "chack-agent-chatgpt-async/1",
                    },
                    json=json_body,
                    timeout=self.request_timeout_seconds,
                    allow_redirects=False,
                )
                if response.status_code in expected:
                    if response.status_code == 204 or not response.content:
                        return {}
                    payload = response.json()
                    return payload if isinstance(payload, dict) else {}
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                error_code = str(payload.get("error") or "")
                if response.status_code not in {429, 500, 502, 503, 504}:
                    raise ChatGPTAsyncApiError(
                        f"ChatGPT async API returned HTTP {response.status_code} ({error_code or 'request_failed'})",
                        status_code=response.status_code,
                        error_code=error_code,
                    )
                last_error = ChatGPTAsyncApiError(
                    f"ChatGPT async API transient HTTP {response.status_code}",
                    status_code=response.status_code,
                    error_code=error_code,
                )
            except ChatGPTAsyncApiError:
                raise
            except requests.RequestException as exc:
                last_error = exc
            if attempt < 3:
                time.sleep(delay)
        raise ChatGPTAsyncApiError(
            f"ChatGPT async API request failed after retries ({type(last_error).__name__})",
            status_code=getattr(last_error, "status_code", 0),
            error_code=getattr(last_error, "error_code", "transport_error"),
        )

    def submit(self, *, mode: str, prompt: str, idempotency_key: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/chatgpt/jobs",
            json_body={"mode": mode, "prompt": prompt, "idempotency_key": idempotency_key},
            expected=(200, 202),
        )

    def status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/chatgpt/jobs/{job_id}")

    def result(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/chatgpt/jobs/{job_id}/result")

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/chatgpt/jobs/{job_id}", expected=(200, 202))

    def lease(self, *, worker_id: str) -> dict[str, Any] | None:
        payload = self._request(
            "POST",
            "/v1/chatgpt/worker/lease",
            json_body={"worker_id": worker_id},
            expected=(200, 204),
        )
        return payload or None

    def heartbeat(
        self,
        job_id: str,
        *,
        lease_id: str,
        stage: str,
        answer_chars: int,
        partial_result: str = "",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "lease_id": lease_id,
            "stage": stage,
            "answer_chars": max(0, int(answer_chars or 0)),
        }
        if partial_result:
            body["partial_result"] = partial_result
        return self._request(
            "POST",
            f"/v1/chatgpt/worker/jobs/{job_id}/heartbeat",
            json_body=body,
        )

    def complete(
        self,
        job_id: str,
        *,
        lease_id: str,
        status: str,
        result: str = "",
        partial_result: str = "",
        metadata: dict[str, Any] | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/chatgpt/worker/jobs/{job_id}/complete",
            json_body={
                "lease_id": lease_id,
                "status": status,
                "result": result,
                "partial_result": partial_result,
                "metadata": metadata or {},
                "error_code": error_code,
                "error_message": error_message,
            },
        )
