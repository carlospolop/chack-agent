"""The relay between the Gemini CLI and a gateway's Gemini API."""
from __future__ import annotations

import json
import threading
import time
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


def test_unsupported_generation_fields_are_removed_only_after_explicit_rejections():
    seen = []

    class Gateway(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            data = json.loads(body)
            seen.append(data)
            if "temperature" in data.get("generationConfig", {}):
                status = 400
                answer = {"error": {"message": "This model doesn't support the temperature field. Remove temperature and try again."}}
            elif "topP" in data.get("generationConfig", {}):
                status = 400
                answer = {"error": {"message": "This model doesn't support the topP field. Remove topP and try again."}}
            else:
                status = 200
                answer = ANSWER
            payload = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = json.dumps({
            "contents": [{"parts": [{"text": "hi"}]}],
            "generationConfig": {"temperature": 1, "topP": 0.95, "maxOutputTokens": 128},
        }).encode()
        with GeminiGatewayRelay(f"http://127.0.0.1:{server.server_address[1]}") as relay:
            status, headers, response = _post(
                f"{relay.base_url}/v1beta/models/bakkt/kimi-k3.1:streamGenerateContent?alt=sse",
                body,
            )
            usage = relay.take_usage()
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    assert json.loads(response[len(b"data: "):].strip()) == ANSWER
    assert len(seen) == 3
    assert seen[1]["generationConfig"] == {"topP": 0.95, "maxOutputTokens": 128}
    assert seen[2]["generationConfig"] == {"maxOutputTokens": 128}
    assert seen[2]["contents"] == seen[0]["contents"]
    assert usage == [{"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}]


def test_the_relay_stops_listening_when_closed(gateway):
    base_url, _, _ = gateway
    relay = GeminiGatewayRelay(base_url).start()
    url = relay.base_url
    relay.close()

    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url + "/v1beta/models", timeout=2)


def test_the_relay_counts_each_model_answer_it_passes_on(gateway):
    base_url, _, answers = gateway
    with GeminiGatewayRelay(base_url) as relay:
        _post(f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse")
        _post(f"{relay.base_url}/v1beta/models/m:generateContent")
        # Not model answers: a token count, and a refused call.
        _post(f"{relay.base_url}/v1beta/models/m:countTokens")
        answers["generateContent"] = (429, b'{"error": {"code": 429, "message": "slow down"}}')
        _post(f"{relay.base_url}/v1beta/models/m:generateContent")
        counted = relay.take_usage()
        again = relay.take_usage()

    one = {"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}
    assert counted == [one, one]
    assert again == []


def test_a_cli_that_goes_away_mid_answer_is_dropped_quietly(capsys, caplog):
    """The relay was still writing an answer when the CLI was killed (a timed
    out or cancelled run): no traceback, as production logged after a timeout."""

    import socket

    sent = threading.Event()
    big = b"x" * (8 * 1024 * 1024)

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self.send_response(200)
            self.send_header("content-type", "application/octet-stream")
            self.send_header("content-length", str(len(big)))
            self.end_headers()
            try:
                self.wfile.write(big)
            except OSError:
                pass
            sent.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen_errors = ""
    try:
        with caplog.at_level("DEBUG", logger="chack.gemini_gateway_relay"):
            with GeminiGatewayRelay(f"http://127.0.0.1:{server.server_address[1]}") as relay:
                port = int(relay.base_url.rsplit(":", 1)[1])
                client = socket.create_connection(("127.0.0.1", port))
                body = b'{"contents": []}'
                client.sendall(
                    b"POST /v1beta/models/m:countTokens HTTP/1.1\r\nHost: x\r\n"
                    b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)
                )
                client.recv(1024)
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                client.close()
                # Wait until the relay has given up on this client, one way or the other.
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    seen_errors += capsys.readouterr().err
                    if "Traceback" in seen_errors or "went away" in caplog.text:
                        break
                    time.sleep(0.05)
    finally:
        server.shutdown()
        server.server_close()

    seen_errors += capsys.readouterr().err
    assert "Traceback" not in seen_errors
    assert "Exception occurred during processing" not in seen_errors
    assert "went away" in caplog.text


def _slow_gateway(delay, status=200, payload=None):
    """A gateway whose ``:generateContent`` answers after ``delay`` seconds."""

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            time.sleep(delay)
            answer = payload if payload is not None else json.dumps(ANSWER).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stream_with_timing(relay_url):
    """POSTs a streaming call; returns status, content type, seconds to headers, and body chunks."""

    import http.client

    port = int(relay_url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1beta/models/bakkt/glm-5.2:streamGenerateContent?alt=sse",
        body=b'{"contents": []}',
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    headers_after = time.monotonic() - started
    chunks = []
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        chunks.append(chunk)
    connection.close()
    return response.status, response.getheader("Content-Type"), headers_after, chunks


def test_a_slow_answer_keeps_the_cli_connection_alive_until_it_arrives():
    """The Gemini CLI drops a call whose headers take over a minute; a slow
    model answer gets its headers early and blank lines until the event."""

    server = _slow_gateway(1.5)
    try:
        relay = GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            keepalive_after_seconds=0.2,
            keepalive_every_seconds=0.2,
        )
        with relay:
            status, content_type, headers_after, chunks = _stream_with_timing(relay.base_url)
            usage = relay.take_usage()
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert content_type == "text/event-stream"
    assert headers_after < 1.0
    body = b"".join(chunks)
    # Keepalives come before the event, each in its own chunk, and are only whitespace.
    assert len(chunks) >= 3
    keepalive, _, event = body.partition(b"data: ")
    assert keepalive and not keepalive.strip()
    assert event.endswith(b"\r\n\r\n")
    assert json.loads(event) == ANSWER
    assert usage == [{"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}]


@pytest.mark.parametrize(
    "gateway_status, gateway_body, expected_code",
    [
        (429, b'{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "slow down"}}', 429),
        (503, b"upstream overloaded", 503),
        (500, b'{"error": {"message": "no code here"}}', 500),
    ],
)
def test_a_slow_gateway_error_reaches_the_cli_as_an_error_with_its_status(
    gateway_status, gateway_body, expected_code
):
    server = _slow_gateway(0.6, gateway_status, gateway_body)
    try:
        relay = GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            keepalive_after_seconds=0.1,
            keepalive_every_seconds=0.1,
        )
        with relay:
            status, _, _, chunks = _stream_with_timing(relay.base_url)
            usage = relay.take_usage()
    finally:
        server.shutdown()
        server.server_close()

    # The headers already said 200; the CLI parses the last chunk as an API error.
    assert status == 200
    error = json.loads(chunks[-1])["error"]
    assert error["code"] == expected_code
    assert usage == []


def test_a_fast_answer_keeps_its_own_status():
    server = _slow_gateway(0, 429, b'{"error": {"code": 429, "message": "slow down"}}')
    try:
        with GeminiGatewayRelay(f"http://127.0.0.1:{server.server_address[1]}") as relay:
            status, _, body = _post(f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 429
    assert json.loads(body)["error"]["message"] == "slow down"


def test_a_connection_the_gateway_drops_is_tried_again(monkeypatch):
    import socket

    import chack_agent.backends.gemini_gateway_relay as relay_module

    monkeypatch.setattr(relay_module, "_RECONNECT_DELAY_SECONDS", 0.01)
    calls = []

    class Flaky(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            calls.append(self.path)
            if len(calls) == 1:
                # Hang up without answering, as a gateway that closes the connection.
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            payload = json.dumps(ANSWER).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with GeminiGatewayRelay(f"http://127.0.0.1:{server.server_address[1]}") as relay:
            status, _, body = _post(f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert json.loads(body[len(b"data: "):]) == ANSWER
    assert len(calls) == 2


def test_an_unreachable_gateway_is_tried_a_bounded_number_of_times(monkeypatch):
    import socket

    import chack_agent.backends.gemini_gateway_relay as relay_module

    monkeypatch.setattr(relay_module, "_RECONNECT_DELAY_SECONDS", 0.01)
    attempts = []

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()

    with GeminiGatewayRelay(f"http://127.0.0.1:{closed_port}") as relay:
        opener_open = relay._opener.open

        def counting_open(request, timeout=None):
            attempts.append(request.full_url)
            return opener_open(request, timeout=timeout)

        monkeypatch.setattr(relay._opener, "open", counting_open)
        status, _, body = _post(f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse")

    assert status == 502
    assert json.loads(body)["error"]["code"] == 502
    assert len(attempts) == relay_module._CONNECTION_ATTEMPTS
