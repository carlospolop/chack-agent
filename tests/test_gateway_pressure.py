"""A relay that absorbs gateway timeouts, rate limits and 5xx tells pressure listeners."""
from __future__ import annotations

import json
import time

import pytest

from chack_agent.backends import gateway_pressure, transient_retry
from chack_agent.backends.gemini_gateway_relay import GeminiGatewayRelay

from test_gemini_gateway_relay import ANSWER, _post, _silent_gateway, _slow_gateway, _stop, _stream_with_timing


@pytest.fixture
def heard(monkeypatch):
    monkeypatch.setattr(transient_retry, "BASE_DELAY_SECONDS", 0.01)
    events: list[tuple[str, str]] = []

    def listener(kind, detail):
        events.append((kind, detail))

    gateway_pressure.add_listener(listener)
    try:
        yield events
    finally:
        gateway_pressure.remove_listener(listener)


def test_a_call_the_gateway_never_answers_is_reported_as_a_timeout_even_when_the_retry_succeeds(heard):
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
            status, _, _, chunks = _stream_with_timing(relay.base_url)
    finally:
        _stop(server, release)

    assert status == 200
    assert json.loads(b"".join(chunks).partition(b"data: ")[2]) == ANSWER
    deadline = time.monotonic() + 2
    while not heard and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [kind for kind, _ in heard] == [gateway_pressure.TIMEOUT]


@pytest.mark.parametrize(
    "status, kind", [(429, gateway_pressure.RATE_LIMITED), (503, gateway_pressure.UNAVAILABLE)]
)
def test_rate_limits_and_unavailable_answers_are_reported(heard, status, kind):
    server = _slow_gateway(0, status, b'{"error": {"code": %d, "message": "busy"}}' % status)
    try:
        with GeminiGatewayRelay(f"http://127.0.0.1:{server.server_address[1]}") as relay:
            got, _, _ = _post(f"{relay.base_url}/v1beta/models/m:generateContent")
    finally:
        server.shutdown()
        server.server_close()

    assert got == status
    assert heard and {k for k, _ in heard} == {kind}


def test_a_good_answer_reports_nothing_and_a_failing_listener_never_breaks_the_call(heard):
    def broken(kind, detail):
        raise RuntimeError("listener bug")

    gateway_pressure.add_listener(broken)
    server = _slow_gateway(0, 503, b'{"error": {"code": 503, "message": "busy"}}')
    ok = _slow_gateway(0)
    try:
        with GeminiGatewayRelay(f"http://127.0.0.1:{ok.server_address[1]}") as relay:
            assert _post(f"{relay.base_url}/v1beta/models/m:generateContent")[0] == 200
        assert heard == []
        with GeminiGatewayRelay(f"http://127.0.0.1:{server.server_address[1]}") as relay:
            assert _post(f"{relay.base_url}/v1beta/models/m:generateContent")[0] == 503
    finally:
        gateway_pressure.remove_listener(broken)
        for s in (server, ok):
            s.shutdown()
            s.server_close()
    assert heard
