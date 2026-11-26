"""The relay between the Gemini CLI and a gateway's Gemini API."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chack_agent.backends import transient_retry
from chack_agent.backends.gemini_gateway_relay import GeminiGatewayRelay


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    # Transient gateway failures are retried; keep the backoff short in tests.
    monkeypatch.setattr(transient_retry, "BASE_DELAY_SECONDS", 0.01)

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


def _stream_with_timing(relay_url, body=b'{"contents": []}'):
    """POSTs a streaming call; returns status, content type, seconds to headers, and body chunks."""

    import http.client

    port = int(relay_url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1beta/models/bakkt/glm-5.2:streamGenerateContent?alt=sse",
        body=body,
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

    monkeypatch.setattr(transient_retry, "BASE_DELAY_SECONDS", 0.01)
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

    monkeypatch.setattr(transient_retry, "BASE_DELAY_SECONDS", 0.01)
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

    # The retries are spent: the error comes inside the stream, which the CLI does not retry again.
    assert status == 200
    assert json.loads(body)["error"]["code"] == 502
    assert len(attempts) == transient_retry.DEFAULT_ATTEMPTS


def test_calls_naming_another_model_go_to_the_runs_model(gateway):
    base_url, seen, _ = gateway
    with GeminiGatewayRelay(base_url, model="bakkt/glm-5.2") as relay:
        own = _post(f"{relay.base_url}/v1beta/models/bakkt/glm-5.2:streamGenerateContent?alt=sse")
        compression = _post(f"{relay.base_url}/v1beta/models/gemini-3-pro-preview:generateContent")
        loop_check = _post(f"{relay.base_url}/v1beta/models/gemini-3-flash-preview:streamGenerateContent?alt=sse")

    assert [reply[0] for reply in (own, compression, loop_check)] == [200, 200, 200]
    assert json.loads(compression[2]) == ANSWER
    assert [request["path"] for request in seen] == [
        "/gemini/v1beta/models/bakkt/glm-5.2:generateContent",
        "/gemini/v1beta/models/bakkt/glm-5.2:generateContent",
        "/gemini/v1beta/models/bakkt/glm-5.2:generateContent",
    ]


def test_without_a_run_model_the_cli_model_is_kept(gateway):
    base_url, seen, _ = gateway
    with GeminiGatewayRelay(base_url) as relay:
        _post(f"{relay.base_url}/v1beta/models/gemini-3-pro-preview:generateContent")

    assert seen[0]["path"] == "/gemini/v1beta/models/gemini-3-pro-preview:generateContent"


def _silent_gateway(answer_from_call):
    """A gateway that never answers its first ``answer_from_call - 1`` model calls.

    Returns the server, the calls it got, and an event that releases the silent ones.
    """

    calls = []
    release = threading.Event()

    class Silent(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            calls.append(self.path)
            if len(calls) < answer_from_call:
                release.wait(30)
                return
            answer = json.dumps(ANSWER).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Silent)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls, release


def _stop(server, release):
    release.set()
    server.shutdown()
    server.server_close()


def test_a_model_call_the_gateway_never_answers_is_sent_again_and_its_answer_reaches_the_cli():
    server, calls, release = _silent_gateway(answer_from_call=2)
    try:
        relay = GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=3840,
            request_timeout_seconds=0.6,
            keepalive_after_seconds=0.1,
            keepalive_every_seconds=0.1,
        )
        with relay:
            started = time.monotonic()
            status, _, _, chunks = _stream_with_timing(relay.base_url)
            elapsed = time.monotonic() - started
            usage = relay.take_usage()
    finally:
        _stop(server, release)

    assert status == 200
    keepalive, _, event = b"".join(chunks).partition(b"data: ")
    assert not keepalive.strip()
    assert json.loads(event) == ANSWER
    assert len(calls) == 2
    assert elapsed < 3
    assert usage == [{"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}]



def _racing_gateway(steps):
    """A gateway whose n-th model call follows ``steps[n]`` (the last step repeats):
    a number answers after that many seconds, with the request's text in the answer;
    ``"hang"`` never answers and notes when the relay hangs up on it, or ``(status, delay)``
    answers that error status after the delay.

    Returns the server, each call's start time, and ``{call index: hang-up time}``.
    """
    calls = []
    closed = {}
    lock = threading.Lock()

    class Racing(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = self.rfile.read(int(self.headers.get("content-length") or 0))
            with lock:
                index = len(calls)
                calls.append(time.monotonic())
            step = steps[min(index, len(steps) - 1)]
            if step == "hang":
                self.connection.settimeout(30)
                try:
                    self.connection.recv(1)
                except OSError:
                    pass
                closed[index] = time.monotonic()
                self.close_connection = True
                return
            status = 200
            if isinstance(step, tuple):
                status, step = step
            time.sleep(step)
            text = json.loads(request or b"{}").get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
            answer = json.loads(ANSWER_BYTES)
            answer["candidates"][0]["content"]["parts"][0]["text"] = f"answer to {text}"
            payload = json.dumps(answer).encode() if status == 200 else b'{"error": {"code": %d}}' % status
            try:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                closed[index] = time.monotonic()  # the relay hung up first

    server = ThreadingHTTPServer(("127.0.0.1", 0), Racing)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls, closed


def _hedged_stream(steps, limit, body=b'{"contents": [{"parts": [{"text": "q"}]}]}'):
    """One streaming call through a relay with a ``limit`` s call limit (hedge at half of it)."""
    server, calls, closed = _racing_gateway(steps)
    try:
        relay = GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=3840,
            request_timeout_seconds=limit,
            keepalive_after_seconds=0.1,
            keepalive_every_seconds=0.1,
        )
        with relay:
            started = time.monotonic()
            status, _, _, chunks = _stream_with_timing(relay.base_url, body)
            elapsed = time.monotonic() - started
            # Leave the relay time to hang up on the other request.
            time.sleep(0.3)
            usage = relay.take_usage()
    finally:
        server.shutdown()
        server.server_close()
    return status, b"".join(chunks), elapsed, started, calls, closed, usage


def _call_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "chack.gemini_gateway_relay.calls"]


def test_a_first_request_answering_after_the_hedge_point_wins_and_the_hedge_is_closed(caplog):
    """A slow but viable first request is no longer thrown away at the hedge point."""
    with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"):
        status, body, elapsed, started, calls, closed, usage = _hedged_stream([1.3, "hang"], limit=2.0)

    assert status == 200
    assert json.loads(body.partition(b"data: ")[2])["candidates"][0]["content"]["parts"][0]["text"] == "answer to q"
    # The hedge went out at half the limit, alongside the first request.
    assert len(calls) == 2
    assert 0.9 < calls[1] - calls[0] < 1.2
    # The answer came at the first request's latency, not later.
    assert elapsed < 1.3 + 0.4
    # The hedge was hung up as soon as the first request answered.
    assert 1 in closed and closed[1] - started < 1.3 + 0.4
    assert usage == [{"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}]
    lines = _call_lines(caplog)
    assert any(line.startswith("Gemini gateway hedge outcome=primary_won status=200 ") for line in lines)
    assert any("outcome=cancelled" in line for line in lines)
    assert not any("outcome=timeout" in line for line in lines)


def test_a_hung_first_request_is_overtaken_by_the_hedge(caplog):
    with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"):
        status, body, elapsed, started, calls, closed, usage = _hedged_stream(["hang", 0.3], limit=2.0)

    assert status == 200
    assert json.loads(body.partition(b"data: ")[2])["candidates"][0]["content"]["parts"][0]["text"] == "answer to q"
    assert len(calls) == 2
    # Answered at the hedge point (1 s) plus the hedge's own latency.
    assert 1.2 < elapsed < 1.3 + 0.4
    # The hung first request was hung up then, not left running to the deadline.
    assert 0 in closed and closed[0] - started < 1.3 + 0.4
    assert len(usage) == 1
    lines = _call_lines(caplog)
    assert any(line.startswith("Gemini gateway hedge outcome=hedge_won status=200 ") for line in lines)
    assert all(len(line) < 300 for line in lines)


def test_a_call_whose_requests_both_hang_times_out_at_its_unchanged_deadline(caplog):
    with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"):
        status, body, elapsed, started, calls, closed, usage = _hedged_stream(["hang"], limit=0.6)

    # The deadline is the hedge point (0.3 s) plus the whole 0.6 s limit, as before the hedge.
    assert 0.85 < elapsed < 1.4
    assert len(calls) == 2
    error = json.loads(body.strip())["error"]
    assert error["code"] == 504 and error["status"] == "DEADLINE_EXCEEDED"
    assert status == 200  # keepalives went out first
    # Neither request outlived the deadline upstream.
    assert sorted(closed) == [0, 1]
    assert all(at - started < 0.9 + 0.3 for at in closed.values())
    assert usage == []
    lines = _call_lines(caplog)
    assert sum("outcome=timeout status=None" in line for line in lines) == 2
    assert any(line.startswith("Gemini gateway hedge outcome=both_timeout status=None ") for line in lines)


def test_a_first_request_failing_after_the_hedge_point_leaves_the_answer_to_the_hedge(caplog):
    with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"):
        status, body, elapsed, _, calls, _, _ = _hedged_stream([(500, 1.2), 0.5], limit=2.0)

    assert status == 200
    assert b"answer to q" in body
    assert len(calls) == 2
    assert any(line.startswith("Gemini gateway hedge outcome=hedge_won ") for line in _call_lines(caplog))


def test_a_5xx_on_the_first_request_is_retried_as_before_without_a_hedge(caplog):
    with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"):
        status, body, _, _, calls, _, usage = _hedged_stream([(503, 0.0), 0.0], limit=2.0)

    assert status == 200
    assert b"answer to q" in body
    assert len(calls) == 2
    assert len(usage) == 1
    lines = _call_lines(caplog)
    assert lines[0].startswith("Gemini gateway call outcome=http_error status=503 attempt=1/4 ")
    assert lines[1].startswith("Gemini gateway call outcome=answered status=200 attempt=2/4 ")
    assert not any("hedge" in line for line in lines)


def test_concurrent_hedged_calls_each_get_their_own_answer():
    seen = {}
    lock = threading.Lock()

    class Gateway(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            text = request["contents"][0]["parts"][0]["text"]
            with lock:
                seen[text] = seen.get(text, 0) + 1
                first = seen[text] == 1
            number = int(text.rsplit("-", 1)[1])
            if first and number % 2:
                # Odd calls hang on their first request and are answered by the hedge.
                self.connection.settimeout(30)
                try:
                    self.connection.recv(1)
                except OSError:
                    pass
                return
            time.sleep(0.05 * (number % 3))
            answer = json.loads(ANSWER_BYTES)
            answer["candidates"][0]["content"]["parts"][0]["text"] = f"answer to {text}"
            answer["usageMetadata"]["promptTokenCount"] = 1000 + number
            payload = json.dumps(answer).encode()
            try:
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = {}
    try:
        relay = GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=3840,
            request_timeout_seconds=1.0,
            keepalive_after_seconds=0.1,
            keepalive_every_seconds=0.1,
        )
        with relay:

            def one(number):
                body = json.dumps({"contents": [{"parts": [{"text": f"call-{number}"}]}]}).encode()
                status, _, _, chunks = _stream_with_timing(relay.base_url, body)
                results[number] = (status, json.loads(b"".join(chunks).partition(b"data: ")[2]))

            threads = [threading.Thread(target=one, args=(number,)) for number in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            usage = relay.take_usage()
    finally:
        server.shutdown()
        server.server_close()

    assert sorted(results) == list(range(8))
    for number, (status, answer) in results.items():
        assert status == 200
        assert answer["candidates"][0]["content"]["parts"][0]["text"] == f"answer to call-{number}"
        assert answer["usageMetadata"]["promptTokenCount"] == 1000 + number
    # Only the hung calls were hedged, each once.
    assert seen == {f"call-{number}": 2 if number % 2 else 1 for number in range(8)}
    assert sorted(entry["input_tokens"] for entry in usage) == [1000 + number for number in range(8)]


@pytest.mark.parametrize("keepalive_after", [0.1, 5.0])
def test_a_gateway_that_never_answers_fails_the_cli_call_fast_with_a_deadline_error(keepalive_after):
    server, calls, release = _silent_gateway(answer_from_call=99)
    try:
        relay = GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=3840,
            request_timeout_seconds=0.4,
            keepalive_after_seconds=keepalive_after,
            keepalive_every_seconds=0.1,
        )
        with relay:
            started = time.monotonic()
            status, _, _, chunks = _stream_with_timing(relay.base_url)
            elapsed = time.monotonic() - started
    finally:
        _stop(server, release)

    # Two calls of at most 0.4 s each, not the run's 3840 s.
    assert elapsed < 3
    assert len(calls) == 2
    error = json.loads(b"".join(chunks).strip())["error"]
    assert error["code"] == 504
    assert error["status"] == "DEADLINE_EXCEEDED"
    # Once keepalives have gone out the CLI reads the error from the stream; otherwise from the status.
    assert status == (200 if keepalive_after < 0.4 else 504)


def test_the_call_limit_comes_from_the_environment_and_never_exceeds_the_runs(monkeypatch):
    from chack_agent.backends import gemini_gateway_relay as relay_module

    monkeypatch.setenv(relay_module.REQUEST_TIMEOUT_ENV, "0.3")
    server, calls, release = _silent_gateway(answer_from_call=99)
    try:
        with GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}", timeout_seconds=3840
        ) as relay:
            started = time.monotonic()
            status, _, body = _post(f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse")
            elapsed = time.monotonic() - started
    finally:
        _stop(server, release)
    assert status == 504 and json.loads(body)["error"]["code"] == 504
    assert elapsed < 3

    monkeypatch.delenv(relay_module.REQUEST_TIMEOUT_ENV)
    assert relay_module.gateway_call_timeout_seconds(3840) == relay_module.DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert relay_module.gateway_call_timeout_seconds(120) == 120


def test_an_unanswered_unstreamed_call_fails_at_the_call_limit():
    server, calls, release = _silent_gateway(answer_from_call=99)
    try:
        with GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=3840,
            request_timeout_seconds=0.4,
        ) as relay:
            started = time.monotonic()
            status, _, body = _post(f"{relay.base_url}/v1beta/models/m:countTokens")
            elapsed = time.monotonic() - started
    finally:
        _stop(server, release)

    assert status == 502
    assert json.loads(body)["error"]["code"] == 502
    assert elapsed < 3


def _conversation(turns, system="Follow the scan rules. " * 400):
    contents = [{"role": "user", "parts": [{"text": "Start. " + "context " * 300}]}]
    for turn in range(turns):
        contents += [{"role": "model", "parts": [{"text": f"answer {turn}"}]},
                     {"role": "user", "parts": [{"text": f"tool result {turn} " + "output " * 200}]}]
    return json.dumps({"contents": contents, "systemInstruction": {"parts": [{"text": system}]},
                       "tools": [{"functionDeclarations": [{"name": "exec"}]}]}).encode()


def _answer(prompt_tokens, cached=None):
    metadata = {"promptTokenCount": prompt_tokens, "candidatesTokenCount": 5}
    if cached is not None:
        metadata["cachedContentTokenCount"] = cached
    return json.dumps({"candidates": [], "usageMetadata": metadata}).encode()


def test_a_gateway_that_omits_cached_tokens_gets_them_estimated_from_repeated_prefixes(gateway):
    base_url, seen, answers = gateway
    url_path = "/v1beta/models/bakkt/deepseek:streamGenerateContent?alt=sse"
    counted = []
    with GeminiGatewayRelay(base_url, model="bakkt/deepseek") as relay:
        for turn, tokens in enumerate((10_000, 11_500, 13_000)):
            answers["generateContent"] = (200, _answer(tokens))
            _post(relay.base_url + url_path, _conversation(turn))
        # A different task's prompt shares nothing with the earlier ones.
        answers["generateContent"] = (200, _answer(9_000))
        _post(relay.base_url + url_path, _conversation(0, system="Unrelated instructions. " * 400))
        counted = relay.take_usage()

    cached = [usage["cached_tokens"] for usage in counted]
    assert cached[0] == 0 and cached[3] == 0
    # Each growing turn is cached up to the previous prompt, in whole 128-token blocks.
    for earlier, usage in zip((10_000, 11_500), counted[1:3]):
        assert earlier - 256 < usage["cached_tokens"] <= earlier and usage["cached_tokens"] % 128 == 0


def test_a_gateway_that_reports_cached_tokens_is_trusted(gateway):
    base_url, _, answers = gateway
    url_path = "/v1beta/models/gemini-x:generateContent"
    with GeminiGatewayRelay(base_url, model="gemini-x") as relay:
        answers["generateContent"] = (200, _answer(10_000, cached=0))
        _post(relay.base_url + url_path, _conversation(0))
        answers["generateContent"] = (200, _answer(11_000, cached=4_096))
        _post(relay.base_url + url_path, _conversation(1))
        answers["generateContent"] = (200, _answer(12_000, cached=0))
        _post(relay.base_url + url_path, _conversation(2))
        counted = relay.take_usage()

    assert [usage["cached_tokens"] for usage in counted] == [0, 4_096, 0]


def test_prefix_cache_estimates_expire_and_skip_short_prompts():
    from chack_agent.backends.gemini_gateway_relay import PrefixCacheEstimator, _prompt_text

    now = [0.0]
    estimator = PrefixCacheEstimator(clock=lambda: now[0])
    key = ("https://gateway", "model")
    # Instructions and tools come first in the prompt a model reads, whatever the request's key order.
    first, second = _prompt_text(_conversation(0)), _prompt_text(_conversation(1))
    assert estimator.estimate(key, first, 10_000) == 0
    assert estimator.estimate(key, second, 11_000) > 9_000
    # Another model's prompts never count.
    assert estimator.estimate(("https://gateway", "other"), second, 11_000) == 0
    now[0] = 3_600.0
    assert estimator.estimate(key, second, 11_000) == 0
    assert estimator.estimate(key, "short prompt", 500) == 0
    assert estimator.estimate(key, "short prompt", 500) == 0


def _session_prompt(session, turns):
    contents = [{"role": "user", "parts": [{"text": f"Task {session}. " + "context " * 300}]}]
    for turn in range(turns):
        contents += [{"role": "model", "parts": [{"text": f"answer {turn}"}]},
                     {"role": "user", "parts": [{"text": f"tool result {session}/{turn} " + "output " * 600}]}]
    return json.dumps({"systemInstruction": {"parts": [{"text": "Follow the scan rules. " * 400}]},
                       "contents": contents}, separators=(",", ":"))


def _interleaved_estimates(estimator, sessions, turns, tag):
    """Cached-token estimates of each session's turns, every turn of all sessions called in round-robin."""
    key = ("https://gateway", "model")
    tokens = lambda turn: 2_000 + 1_000 * turn
    estimates = {session: [] for session in range(sessions)}
    for turn in range(turns):
        for session in range(sessions):
            text = _session_prompt(session, turn)
            estimates[session].append(estimator.estimate(key, text, tokens(turn), tag(session)))
    return estimates


def test_concurrent_sessions_keep_their_own_prefix_estimates():
    from chack_agent.backends.gemini_gateway_relay import PrefixCacheEstimator

    # 40 runs share the process: between two turns of one run, 39 other prompts arrive.
    sessions, turns = 40, 4
    scoped = _interleaved_estimates(PrefixCacheEstimator(clock=lambda: 0.0), sessions, turns, lambda s: f"run-{s}")
    for session, estimates in scoped.items():
        assert estimates[0] < 2_000
        # Each later turn repeats the run's previous prompt, whose tokens it was.
        for turn in range(1, turns):
            earlier = 2_000 + 1_000 * (turn - 1)
            assert earlier - 256 < estimates[turn] <= earlier

    # Untagged calls only match the other runs' shared instructions, never the run's own history.
    shared = _interleaved_estimates(PrefixCacheEstimator(clock=lambda: 0.0), sessions, turns, lambda s: None)
    assert all(shared[s][t] < scoped[s][t] for s in range(sessions) for t in range(1, turns))


def test_session_prefix_history_expires_and_forgets_the_least_recent_runs():
    from chack_agent.backends.gemini_gateway_relay import _CACHE_SESSIONS, PrefixCacheEstimator

    now = [0.0]
    estimator = PrefixCacheEstimator(clock=lambda: now[0])
    key = ("https://gateway", "model")
    estimator.estimate(key, _session_prompt("first", 3), 6_000, "first")
    for session in range(_CACHE_SESSIONS):
        estimator.estimate(key, _session_prompt(session, 3), 6_000, session)
    # The first run was pushed out by more recent ones; a recent one still matches its own prompt.
    assert estimator.estimate(key, _session_prompt("first", 4), 7_000, "first") < 5_000
    assert estimator.estimate(key, _session_prompt(_CACHE_SESSIONS - 1, 4), 7_000, _CACHE_SESSIONS - 1) > 5_700
    now[0] = 3_600.0
    assert estimator.estimate(key, _session_prompt(_CACHE_SESSIONS - 1, 5), 8_000, _CACHE_SESSIONS - 1) == 0


def test_an_answer_without_text_or_tool_call_logs_why_without_its_content(gateway, caplog):
    base_url, _, answers = gateway
    url_path = "/v1beta/models/bakkt/deepseek:streamGenerateContent?alt=sse"
    request = json.dumps({"contents": [], "generationConfig": {"maxOutputTokens": 8192}}).encode()
    empty = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "secret plan", "thought": True}, {"text": " "}]},
                "finishReason": "MAX_TOKENS",
            }
        ],
        "usageMetadata": {"promptTokenCount": 90_000, "candidatesTokenCount": 8_192, "thoughtsTokenCount": 8_000},
    }
    tool_call = {
        "candidates": [{"content": {"parts": [{"functionCall": {"name": "exec", "args": {}}}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }
    with caplog.at_level("WARNING", logger="chack.gemini_gateway_relay"):
        with GeminiGatewayRelay(base_url, model="bakkt/deepseek") as relay:
            answers["generateContent"] = (200, json.dumps(empty).encode())
            status, _, _ = _post(relay.base_url + url_path, request)
            answers["generateContent"] = (200, json.dumps(tool_call).encode())
            _post(relay.base_url + url_path, request)
            answers["generateContent"] = (200, b"")
            _post(relay.base_url + url_path, request)
            counted = relay.take_usage()

    assert status == 200 and len(counted) == 3
    warnings = [record.getMessage() for record in caplog.records if "no text or tool call" in record.getMessage()]
    assert len(warnings) == 1
    assert "finish_reason=MAX_TOKENS" in warnings[0]
    assert "input_tokens=90000" in warnings[0] and "output_tokens=8192" in warnings[0]
    assert "thinking_tokens=8000" in warnings[0] and "max_output_tokens=8192" in warnings[0]
    assert "secret plan" not in caplog.text


def test_the_last_agent_turn_answer_shape_is_kept_without_content_and_helper_calls_counted(gateway):
    base_url, _, answers = gateway
    url_path = "/v1beta/models/m:generateContent"
    turn = json.dumps({"contents": [], "tools": [{"functionDeclarations": [{"name": "exec"}]}]}).encode()
    helper = json.dumps({"contents": []}).encode()
    tool_call = {
        "candidates": [
            {
                "content": {"parts": [{"text": "hidden reasoning", "thought": True}, {"functionCall": {"name": "exec", "args": {"c": "id"}}}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 20, "thoughtsTokenCount": 7},
    }
    with GeminiGatewayRelay(base_url, model="m") as relay:
        assert relay.last_answer_shape() is None
        answers["generateContent"] = (200, json.dumps(tool_call).encode())
        _post(relay.base_url + url_path, turn)
        answers["generateContent"] = (200, json.dumps(ANSWER).encode())
        _post(relay.base_url + url_path, helper)
        _post(relay.base_url + url_path, helper)
        shape = relay.last_answer_shape()

    assert shape == {
        "finish_reason": "STOP",
        "function_call": 1,
        "text": 0,
        "thought": 1,
        "empty": 0,
        "input_tokens": 500,
        "output_tokens": 20,
        "thinking_tokens": 7,
        "helper_answers_after": 2,
    }
    assert "hidden reasoning" not in json.dumps(shape) and "exec" not in json.dumps(shape)


UPSTREAM_TIMEOUT = b'{"error":{"message":"upstream request timeout","code":504}}'


def _scripted_gateway(script, delay=0.0):
    """A gateway answering each model call with the next step of ``script``:
    ``(status, body)``, or ``"cut"`` for an answer whose body stops short.
    Steps run out into the last one."""

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            calls.append(self.path)
            step = script[min(len(calls), len(script)) - 1]
            time.sleep(delay)
            if step == "cut":
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", "5000")
                self.end_headers()
                self.wfile.write(b'{"candidates": [')
                self.wfile.flush()
                self.close_connection = True
                return
            status, payload = step
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


def _stream_call(script, *, delay=0.0, keepalive_after=20.0):
    server, calls = _scripted_gateway(script, delay)
    try:
        with GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            keepalive_after_seconds=keepalive_after,
            keepalive_every_seconds=keepalive_after,
        ) as relay:
            if keepalive_after < 20:
                status, _, _, chunks = _stream_with_timing(relay.base_url)
                body = b"".join(chunks)
            else:
                status, _, body = _post(f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse")
            usage = relay.take_usage()
    finally:
        server.shutdown()
        server.server_close()
    return status, body, usage, calls


ANSWER_BYTES = json.dumps(ANSWER).encode()


@pytest.mark.parametrize("failure", [(504, UPSTREAM_TIMEOUT), (502, b"bad gateway"), (503, b"overloaded"), "cut"])
def test_a_transient_gateway_failure_is_retried_and_the_answer_reaches_the_cli(failure, caplog):
    with caplog.at_level("WARNING", logger="chack.gemini_gateway_relay"):
        status, body, usage, calls = _stream_call([failure, (200, ANSWER_BYTES)])

    assert status == 200
    assert json.loads(body[len(b"data: "):]) == ANSWER
    assert len(calls) == 2
    # Only the answer that arrived is counted.
    assert usage == [{"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}]
    assert "attempt 1/4; retrying in" in caplog.text
    assert "(16 request bytes)" in caplog.text


def test_each_failed_or_retried_call_is_logged_with_its_size_latency_and_tokens_but_no_content(caplog):
    error_500 = json.dumps({"error": {"code": 500, "message": "boom"}}).encode()
    with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"):
        _stream_call([(504, UPSTREAM_TIMEOUT), (200, ANSWER_BYTES)])
        _stream_call([(500, error_500)])

    lines = [r.getMessage() for r in caplog.records if r.name == "chack.gemini_gateway_relay.calls"]
    assert len(lines) == 3
    assert lines[0].startswith("Gemini gateway call outcome=http_error status=504 attempt=1/4 elapsed_seconds=")
    assert "request_bytes=16 prompt_tokens=None completion_tokens=None" in lines[0]
    assert lines[1].startswith("Gemini gateway call outcome=answered status=200 attempt=2/4 elapsed_seconds=")
    assert "request_bytes=16 prompt_tokens=120 completion_tokens=30" in lines[1]
    assert lines[2].startswith("Gemini gateway call outcome=http_error status=500 attempt=1/4 elapsed_seconds=")
    assert all(len(line) < 300 for line in lines)


@pytest.mark.parametrize(
    ("delay", "timeout", "method", "status", "outcome", "minimum_elapsed"),
    [
        (0.0, 2.0, "streamGenerateContent", 200, "answered", 0.0),
        (0.2, 2.0, "streamGenerateContent", 200, "answered", 0.18),
        (0.5, 0.15, "generateContent", 502, "timeout", 0.13),
    ],
    ids=["fast-answer", "slow-answer", "timeout"],
)
def test_gateway_call_logs_observed_latency_and_usage(
    caplog, delay, timeout, method, status, outcome, minimum_elapsed,
):
    server, calls = _scripted_gateway([(200, ANSWER_BYTES)], delay)
    private_body = b'{"contents":[{"parts":[{"text":"private prompt"}]}]}'
    try:
        with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"), GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            request_timeout_seconds=timeout,
            keepalive_after_seconds=0.05,
            keepalive_every_seconds=0.05,
        ) as relay:
            actual_status, _, answer = _post(
                f"{relay.base_url}/v1beta/models/m:{method}?private=private-query",
                private_body,
            )
    finally:
        server.shutdown()
        server.server_close()

    assert actual_status == status
    assert len(calls) == 1
    if outcome == "answered":
        assert b"data: " in answer
    records = [r for r in caplog.records if r.name == "chack.gemini_gateway_relay.calls"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    line = records[0].getMessage()
    fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
    assert fields["outcome"] == outcome
    assert fields["status"] == ("200" if outcome == "answered" else "None")
    assert fields["attempt"].startswith("1/")
    assert int(fields["request_bytes"]) == len(private_body)
    assert float(fields["elapsed_seconds"]) >= minimum_elapsed
    assert fields["prompt_tokens"] == ("120" if outcome == "answered" else "None")
    assert fields["completion_tokens"] == ("30" if outcome == "answered" else "None")
    assert all(value not in line for value in (FAKE_KEY, "private prompt", "private-query", "line one"))


def test_a_504_after_the_keepalive_headers_is_retried_instead_of_failing_the_turn():
    """The production failure: a slow call's headers went out, then the gateway
    answered 504; the CLI used to get it as a status-less error and end the run."""

    status, body, usage, calls = _stream_call(
        [(504, UPSTREAM_TIMEOUT), (200, ANSWER_BYTES)], delay=0.4, keepalive_after=0.1
    )

    assert status == 200
    keepalive, _, event = body.partition(b"data: ")
    assert not keepalive.strip()
    assert json.loads(event) == ANSWER
    assert len(calls) == 2
    assert usage and usage[0]["input_tokens"] == 120


def test_a_persistent_504_is_retried_a_bounded_number_of_times_then_reaches_the_cli():
    status, body, usage, calls = _stream_call([(504, UPSTREAM_TIMEOUT)])

    assert len(calls) == transient_retry.DEFAULT_ATTEMPTS
    # Inside the stream, so the CLI ends the turn on it instead of retrying the call again.
    assert status == 200
    error = json.loads(body)["error"]
    assert error["code"] == 504 and error["message"] == "upstream request timeout"
    assert usage == []


@pytest.mark.parametrize("status_code", [400, 404, 429, 500])
def test_other_gateway_errors_are_not_retried(status_code):
    error = json.dumps({"error": {"code": status_code, "message": "no"}}).encode()
    status, body, _, calls = _stream_call([(status_code, error), (200, ANSWER_BYTES)])

    assert status == status_code
    assert len(calls) == 1


def test_unstreamed_model_calls_are_retried_too():
    server, calls = _scripted_gateway([(504, UPSTREAM_TIMEOUT), (200, ANSWER_BYTES)])
    try:
        with GeminiGatewayRelay(f"http://127.0.0.1:{server.server_address[1]}") as relay:
            status, _, body = _post(f"{relay.base_url}/v1beta/models/m:generateContent")
            usage = relay.take_usage()
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert json.loads(body) == ANSWER
    assert len(calls) == 2
    assert len(usage) == 1


def test_retries_back_off_with_jitter_within_their_caps(caplog):
    sleeps = []
    values = iter([0.0, 1.0, 0.5, 0.25])
    retry = transient_retry.TransientRetry(
        "call", logger=logging.getLogger("test.retry"), attempts=4,
        max_wait_seconds=1000, sleep=sleeps.append, rng=lambda: next(values),
    )
    base = transient_retry.BASE_DELAY_SECONDS

    with caplog.at_level("WARNING", logger="test.retry"):
        assert [retry.wait("HTTP 504") for _ in range(4)] == [True, True, True, False]

    # Retry n waits between half and all of base * 2**(n-1).
    assert sleeps == pytest.approx([base / 2, 2 * base, 4 * base * 0.75])
    assert "HTTP 504" in caplog.text and "attempt 3/4; retrying in" in caplog.text
    assert "attempt 4/4; no attempts left" in caplog.text


def test_jittered_delays_differ_and_stay_in_range():
    delays = {transient_retry.backoff_delay(3) for _ in range(50)}
    ceiling = 4 * transient_retry.BASE_DELAY_SECONDS
    assert len(delays) > 1
    assert all(ceiling / 2 <= delay <= ceiling for delay in delays)
    assert transient_retry.backoff_delay(30) <= transient_retry.MAX_DELAY_SECONDS


def test_retries_stop_at_the_total_wait_and_the_deadline():
    sleeps = []
    capped = transient_retry.TransientRetry(
        "call", logger=logging.getLogger("test.retry"), attempts=10,
        max_wait_seconds=3 * transient_retry.BASE_DELAY_SECONDS, sleep=sleeps.append, rng=lambda: 1.0,
    )
    while capped.wait("HTTP 503"):
        pass
    assert sum(sleeps) <= 3 * transient_retry.BASE_DELAY_SECONDS
    assert capped.attempt < 10

    due = transient_retry.TransientRetry(
        "call", logger=logging.getLogger("test.retry"), attempts=10,
        deadline=time.monotonic(), sleep=sleeps.append,
    )
    assert due.wait("HTTP 503") is False


def test_retry_after_lengthens_the_backoff_up_to_its_cap():
    sleeps = []
    retry = transient_retry.TransientRetry(
        "call", logger=logging.getLogger("test.retry"), attempts=4,
        max_wait_seconds=1000, sleep=sleeps.append, rng=lambda: 0.0,
    )
    retry.wait("HTTP 503", retry_after=7.0)
    retry.wait("HTTP 503", retry_after=10_000)
    assert sleeps == [7.0, transient_retry.MAX_DELAY_SECONDS]


def test_once_a_call_spent_its_retries_the_clis_own_resends_get_one_attempt_until_the_gateway_recovers():
    server, calls = _scripted_gateway([(504, UPSTREAM_TIMEOUT)] * 6 + [(200, ANSWER_BYTES)] + [(504, UPSTREAM_TIMEOUT), (200, ANSWER_BYTES)])
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with GeminiGatewayRelay(url) as relay:
            stream = f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse"
            first = _post(stream)
            assert len(calls) == transient_retry.DEFAULT_ATTEMPTS
            second = _post(stream)
            third = _post(stream)
            assert len(calls) == transient_retry.DEFAULT_ATTEMPTS + 2
            # A re-send that gets an answer restores the full allowance.
            assert _post(stream)[0] == 200
            assert _post(stream)[0] == 200
    finally:
        server.shutdown()
        server.server_close()

    assert [json.loads(reply[2])["error"]["code"] for reply in (first, second, third)] == [504, 504, 504]
    assert len(calls) == transient_retry.DEFAULT_ATTEMPTS + 2 + 1 + 2


def test_a_hung_retry_of_the_first_request_is_hung_up_once_the_hedge_answers(caplog):
    """A retry started near the hedge point must not run on (and be billed)
    after the call already has its answer."""

    calls = []
    retry_closed = threading.Event()
    closed_at = []

    class Gateway(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            calls.append(time.monotonic())
            if len(calls) == 1:
                # A transient failure late in the first attempt's 2 s limit: the retry starts with under 1 s left.
                time.sleep(1.2)
                payload, status = b"overloaded", 503
            elif len(calls) == 2:
                # The retry gets no answer; note when the relay hangs up on it.
                self.connection.settimeout(10)
                try:
                    self.connection.recv(1)
                except OSError:
                    pass
                closed_at.append(time.monotonic())
                retry_closed.set()
                return
            else:
                payload, status = ANSWER_BYTES, 200
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        relay = GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=3840,
            request_timeout_seconds=4.0,
            keepalive_after_seconds=0.1,
            keepalive_every_seconds=0.1,
        )
        with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"), relay:
            started = time.monotonic()
            status, _, _, chunks = _stream_with_timing(relay.base_url)
            assert retry_closed.wait(5)
    finally:
        server.shutdown()
        server.server_close()

    # The call is hedged at half its limit and that answer reaches the CLI.
    assert status == 200
    assert json.loads(b"".join(chunks).partition(b"data: ")[2]) == ANSWER
    assert len(calls) == 3
    # The first request's hung retry was hung up once the hedge answered (~2 s), not left running.
    assert closed_at[0] - started < 2.0 + 0.5
    # It is logged as cancelled by the relay, which is no gateway 502.
    lines = [r.getMessage() for r in caplog.records if r.name == "chack.gemini_gateway_relay.calls"]
    assert any("outcome=cancelled status=None" in line for line in lines)
    assert not any("status=502" in line for line in lines)


def test_a_call_whose_caller_stopped_waiting_starts_no_new_retry():
    abandoned = threading.Event()
    retry = transient_retry.TransientRetry(
        "call", logger=logging.getLogger("test.retry"), attempts=10,
        abandoned=abandoned, sleep=lambda _delay: abandoned.set(),
    )
    # Abandoned during the backoff: no further attempt.
    assert retry.wait("HTTP 503") is False
    assert retry.attempt == 1
    # Already abandoned: no backoff either.
    slept = []
    retry = transient_retry.TransientRetry(
        "call", logger=logging.getLogger("test.retry"), attempts=10,
        abandoned=abandoned, sleep=slept.append,
    )
    assert retry.wait("HTTP 503") is False
    assert slept == []


def test_the_relays_own_socket_timeout_is_logged_as_a_timeout_not_a_gateway_502(caplog):
    server, calls, release = _silent_gateway(answer_from_call=99)
    try:
        with caplog.at_level("INFO", logger="chack.gemini_gateway_relay.calls"), GeminiGatewayRelay(
            f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=3840,
            request_timeout_seconds=0.4,
        ) as relay:
            status, _, _ = _post(f"{relay.base_url}/v1beta/models/m:countTokens")
    finally:
        _stop(server, release)

    # The CLI still gets the same 502 and the call is not retried.
    assert status == 502
    assert len(calls) == 1
    lines = [r.getMessage() for r in caplog.records if r.name == "chack.gemini_gateway_relay.calls"]
    assert len(lines) == 1
    assert lines[0].startswith("Gemini gateway call outcome=timeout status=None attempt=1/")


@pytest.mark.parametrize("every, expected_lines", [(0.0, 2), (3600.0, 0)])
def test_answered_calls_report_progress_at_most_once_per_interval(gateway, caplog, every, expected_lines):
    base_url, _, _ = gateway
    with caplog.at_level("INFO", logger="chack.gemini_gateway_relay"):
        with GeminiGatewayRelay(base_url, progress_every_seconds=every) as relay:
            for _ in range(2):
                _post(f"{relay.base_url}/v1beta/models/m:streamGenerateContent?alt=sse", b'{"secret": 1}')

    lines = [r.getMessage() for r in caplog.records if "model calls in the last" in r.getMessage()]
    assert len(lines) == expected_lines
    if lines:
        assert lines[-1].startswith("Gemini gateway relay answered 1 model calls")
        assert "(2 in total)" in lines[-1] and "request_bytes=13 prompt_tokens=120" in lines[-1]
    assert "secret" not in caplog.text
