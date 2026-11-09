"""The relay between the Gemini CLI and a gateway's Gemini API."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chack_agent.backends.gemini_gateway_relay import GeminiGatewayRelay

FAKE_KEY = "fake-gateway-key"
ANSWER = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "line one\nline two"}]}}],
    "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30, "totalTokenCount": 150},
}


@pytest.fixture
def gateway():
    """A fake gateway: answers ``:generateContent`` and records every request."""

    seen: list[dict] = []
    answers: dict[str, tuple[int, bytes]] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            seen.append(
                {"method": self.command, "path": self.path, "headers": self.headers, "body": body}
            )
            status, payload = answers.get(self.path.split("?")[0].rsplit(":", 1)[-1], (200, b""))
            if not payload:
                payload = json.dumps(ANSWER, indent=2).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = _answer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/gemini/", seen, answers
    finally:
        server.shutdown()
        server.server_close()


def _post(url, body=b'{"contents": []}'):
    request = urllib.request.Request(
        url,
        data=body,
        headers={"x-goog-api-key": FAKE_KEY, "Content-Type": "application/json", "X-Team": "red"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def test_a_streaming_request_is_answered_from_the_unstreamed_call(gateway):
    base_url, seen, _ = gateway
    with GeminiGatewayRelay(base_url) as relay:
        status, headers, body = _post(
            f"{relay.base_url}/v1beta/models/bakkt/glm-5.2:streamGenerateContent?alt=sse&trace=1"
        )

    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    # One event, its JSON on a single line, as the Gemini SDK parses it.
    assert body.startswith(b"data: ") and body.endswith(b"\r\n\r\n")
    assert body.count(b"\n") == 2
    assert json.loads(body[len(b"data: "):].strip()) == ANSWER

    (upstream,) = seen
    assert upstream["path"] == "/gemini/v1beta/models/bakkt/glm-5.2:generateContent?trace=1"
    assert upstream["method"] == "POST"
    assert upstream["body"] == b'{"contents": []}'
    assert upstream["headers"]["x-goog-api-key"] == FAKE_KEY
    assert upstream["headers"]["X-Team"] == "red"


def test_other_requests_are_forwarded_unchanged(gateway):
    base_url, seen, _ = gateway
    with GeminiGatewayRelay(base_url) as relay:
        status, headers, body = _post(f"{relay.base_url}/v1beta/models/bakkt/glm-5.2:countTokens?x=1")

    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body) == ANSWER
    assert seen[0]["path"] == "/gemini/v1beta/models/bakkt/glm-5.2:countTokens?x=1"


@pytest.mark.parametrize("method", ["streamGenerateContent", "countTokens"])
def test_gateway_errors_keep_their_status_and_body(gateway, method):
    base_url, _, answers = gateway
    error = b'{"error": {"code": 400, "message": "content blocked by policy"}}'
    answers["generateContent"] = (400, error)
    answers["countTokens"] = (400, error)
    with GeminiGatewayRelay(base_url) as relay:
        status, _, body = _post(f"{relay.base_url}/v1beta/models/m:{method}?alt=sse")

    assert status == 400
    assert body == error


def test_the_relay_stops_listening_when_closed(gateway):
    base_url, _, _ = gateway
    relay = GeminiGatewayRelay(base_url).start()
    url = relay.base_url
    relay.close()

    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url + "/v1beta/models", timeout=2)
