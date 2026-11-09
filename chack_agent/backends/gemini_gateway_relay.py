"""Local relay between the Gemini CLI and an LLM gateway's Gemini API.

LiteLLM translates its Gemini API for every model it serves, but when it
streams (``:streamGenerateContent``) a model that is not Gemini its
``usageMetadata`` is all zeros, so a run's tokens and cost are lost. The same
request without streaming (``:generateContent``) reports the real usage.

The relay listens on 127.0.0.1 and forwards every request to the gateway
unchanged, except that a streaming request is sent as ``:generateContent`` and
its JSON answer is returned to the CLI as one server-sent event. Gateway errors
pass through with their status and body.

The relay also keeps the token usage of every answer it relays, so a run that
is stopped before the CLI prints its closing stats still counts the calls it
made. A CLI that goes away while the relay answers it (a cancelled or timed-out
run) is no error: the answer is dropped quietly.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_LOGGER = logging.getLogger("chack.gemini_gateway_relay")

_STREAM_METHOD = ":streamGenerateContent"
_UNARY_METHOD = ":generateContent"
_CHUNK_BYTES = 64 * 1024
# Errors of writing to a client that has gone away.
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

# Headers that describe one hop, or that the relay sets itself.
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
    "content-encoding",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect goes back to the CLI as it is, like any other answer.
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _upstream_target(base_url: str, path: str) -> tuple[str, bool]:
    """Gateway URL for a CLI request path, and whether it was a streaming call."""

    parsed = urllib.parse.urlsplit(path)
    request_path = parsed.path
    query = parsed.query
    streaming = request_path.endswith(_STREAM_METHOD)
    if streaming:
        request_path = request_path[: -len(_STREAM_METHOD)] + _UNARY_METHOD
        query = urllib.parse.urlencode(
            [
                (name, value)
                for name, value in urllib.parse.parse_qsl(query, keep_blank_values=True)
                if name != "alt"
            ]
        )
    url = base_url.rstrip("/") + request_path
    if query:
        url += "?" + query
    return url, streaming


def _usage_of(answer: Any) -> dict[str, int] | None:
    """Token usage of one Gemini API answer, counted the way the Gemini CLI counts it."""

    metadata = answer.get("usageMetadata") if isinstance(answer, dict) else None
    if not isinstance(metadata, dict):
        return None
    return {
        "input_tokens": int(metadata.get("promptTokenCount", 0) or 0),
        "output_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
        "cached_tokens": int(metadata.get("cachedContentTokenCount", 0) or 0),
    }


def _sse_event(body: bytes) -> bytes:
    # The Gemini SDK reads each event as one ``data:`` line, so the JSON
    # must not span lines.
    payload = json.dumps(json.loads(body), ensure_ascii=False, separators=(",", ":"))
    return b"data: " + payload.encode("utf-8") + b"\r\n\r\n"


class GeminiGatewayRelay:
    """Serves the Gemini CLI on 127.0.0.1 for one gateway Gemini API base URL."""

    def __init__(self, upstream_base_url: str, *, timeout_seconds: float | None = None) -> None:
        self._upstream_base_url = str(upstream_base_url or "").rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._usage_lock = threading.Lock()
        self._usage: list[dict[str, int]] = []

    def take_usage(self) -> list[dict[str, int]]:
        """Usage of the answers relayed since the last call, one entry per answer."""
        with self._usage_lock:
            usage, self._usage = self._usage, []
        return usage

    def _record_usage(self, body: bytes) -> None:
        try:
            usage = _usage_of(json.loads(body))
        except ValueError:
            return
        if usage is None:
            return
        with self._usage_lock:
            self._usage.append(usage)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Gemini gateway relay is not running")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> "GeminiGatewayRelay":
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                _LOGGER.debug("relay %s", format % args)

            def _relay(self) -> None:
                try:
                    relay._forward(self)
                except _CLIENT_GONE:
                    # The CLI was stopped (cancelled, timed out) while it waited.
                    self.close_connection = True
                    _LOGGER.debug("Gemini CLI went away before the relay answered %s", self.path)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _relay

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="chack-gemini-gateway-relay",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "GeminiGatewayRelay":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        url, streaming = _upstream_target(self._upstream_base_url, handler.path)
        length = int(handler.headers.get("content-length") or 0)
        body = handler.rfile.read(length) if length else None
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() not in _HOP_HEADERS
        }
        if streaming:
            headers = {name: value for name, value in headers.items() if name.lower() != "accept"}
            headers["Accept"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=handler.command)
        try:
            response = self._opener.open(request, timeout=self._timeout_seconds)
        except urllib.error.HTTPError as exc:
            response = exc
        except Exception as exc:
            _LOGGER.warning("Gemini gateway relay could not reach %s: %s", url, exc)
            error = {"error": {"code": 502, "status": "UNAVAILABLE", "message": f"gateway unreachable: {exc}"}}
            self._send(handler, 502, {"Content-Type": "application/json"}, json.dumps(error).encode("utf-8"))
            return
        with response:
            status = int(response.status if hasattr(response, "status") else response.code)
            response_headers = {
                name: value
                for name, value in response.headers.items()
                if name.lower() not in _HOP_HEADERS
            }
            if urllib.parse.urlsplit(url).path.endswith(_UNARY_METHOD) and 200 <= status < 300:
                # A model answer: read it whole to count its tokens before the CLI gets it.
                raw = response.read()
                self._record_usage(raw)
                if not streaming:
                    self._send(handler, status, response_headers, raw)
                    return
                try:
                    event = _sse_event(raw)
                except ValueError:
                    _LOGGER.warning("Gemini gateway relay got a non-JSON answer from %s", url)
                    self._send(handler, 502, response_headers, raw)
                    return
                response_headers = {
                    name: value
                    for name, value in response_headers.items()
                    if name.lower() != "content-type"
                }
                response_headers["Content-Type"] = "text/event-stream"
                self._send(handler, status, response_headers, event)
                return
            self._send_streamed(handler, status, response_headers, response)

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, headers: dict[str, str], body: bytes) -> None:
        handler.send_response(status)
        for name, value in headers.items():
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _send_streamed(
        handler: BaseHTTPRequestHandler, status: int, headers: dict[str, str], response: Any
    ) -> None:
        handler.send_response(status)
        for name, value in headers.items():
            handler.send_header(name, value)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        while True:
            chunk = response.read1(_CHUNK_BYTES) if hasattr(response, "read1") else response.read(_CHUNK_BYTES)
            if not chunk:
                break
            handler.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            handler.wfile.flush()
        handler.wfile.write(b"0\r\n\r\n")
