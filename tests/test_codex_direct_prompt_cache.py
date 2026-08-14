from __future__ import annotations

import json

from chack_agent.backends import codex_backend
from chack_agent.backends.codex_backend import CodexExecutor, _RawResult


def _executor() -> CodexExecutor:
    executor = object.__new__(CodexExecutor)
    executor._runtime_env_json = "{}"
    executor._uses_openrouter_route = False
    executor._model_name = "gpt-5.6-sol"
    executor._cacheable_developer_prompt = "stable context " * 1000
    executor._prompt_cache_prefix_key = "chack-stable-prefix"
    executor._allowed_tools_json = "[]"
    executor._thread_id = None
    executor._conversation = []
    executor._use_codex_access_token = True
    executor._codex_access_token = "unused-test-token"
    executor._use_existing_codex_auth_file = False
    executor._existing_codex_auth_file = ""
    executor._openai_api_key = ""
    executor._output_schema_strict = True
    executor._output_schema_json = ""
    executor._thinking_effort = "high"
    executor._sub_action = "threat_modeler"
    executor._direct_cache_credentials = lambda: (
        "chatgpt",
        "test-bearer-token",
        "test-account-id",
    )
    return executor


class _Response:
    status_code = 200
    text = ""

    def __init__(self) -> None:
        self.closed = False

    def iter_lines(self):
        events = [
            {
                "type": "response.output_text.delta",
                "delta": '{"checks":[]}',
            },
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 12000,
                        "output_tokens": 50,
                        "input_tokens_details": {
                            "cached_tokens": 9984,
                            "cache_write_tokens": 0,
                        },
                    }
                },
            },
        ]
        for event in events:
            yield f"data: {json.dumps(event)}".encode()
        yield b"data: [DONE]"

    def close(self) -> None:
        self.closed = True


class _FailedResponse(_Response):
    def iter_lines(self):
        event = {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "provider_failure",
                    "message": "Exact nested provider failure",
                }
            },
        }
        yield f"data: {json.dumps(event)}".encode()
        yield b"data: [DONE]"


def test_chatgpt_direct_request_uses_stable_key_and_session_without_unsupported_fields():
    first = _executor()
    second = _executor()

    first_url, first_headers, first_body = first._direct_cache_request("round one")
    second_url, second_headers, second_body = second._direct_cache_request("round two")

    assert first_url == "https://chatgpt.com/backend-api/codex/responses"
    assert first_headers["session_id"] == second_headers["session_id"]
    assert first_body["prompt_cache_key"] == second_body["prompt_cache_key"]
    assert first_body["input"][0] == second_body["input"][0]
    assert first_body["input"][1] != second_body["input"][1]
    assert "prompt_cache_options" not in first_body
    assert "prompt_cache_breakpoint" not in first_body["input"][0]["content"][0]
    assert first_headers["ChatGPT-Account-ID"] == "test-account-id"
    assert first_headers["originator"] == "codex_cli_rs"


def test_api_key_direct_request_uses_documented_explicit_cache_fields():
    executor = _executor()
    executor._direct_cache_credentials = lambda: ("api_key", "sk-test", "")

    url, headers, body = executor._direct_cache_request("changing suffix")

    assert url == "https://api.openai.com/v1/responses"
    assert headers["Authorization"] == "Bearer sk-test"
    assert "ChatGPT-Account-ID" not in headers
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }


def test_direct_response_reports_real_cache_read_telemetry(monkeypatch):
    executor = _executor()
    response = _Response()
    captured = {}

    def fake_post(url, *, headers, json, stream, timeout):
        captured.update(
            {
                "url": url,
                "headers": headers,
                "body": json,
                "stream": stream,
                "timeout": timeout,
            }
        )
        return response

    usage_calls = []
    monkeypatch.setattr(codex_backend.requests, "post", fake_post)
    monkeypatch.setattr(
        codex_backend,
        "report_live_usage",
        lambda *args, **kwargs: usage_calls.append((args, kwargs)),
    )

    output, steps, raw = executor._run_direct_cached_response("round two")

    assert output == '{"checks":[]}'
    assert steps == []
    usage = raw.raw_responses[0]["usage"]
    assert usage["input_tokens"] == 12000
    assert usage["input_tokens_details"]["cached_tokens"] == 9984
    assert usage["input_tokens_details"]["cache_write_tokens"] == 0
    assert usage_calls[0][1]["cached_prompt_tokens"] == 9984
    assert captured["headers"]["session_id"]
    assert captured["stream"] is True
    assert response.closed is True


def test_direct_transport_falls_back_to_codex_cli_on_provider_error():
    executor = _executor()
    executor._ensure_codex_home_and_config = lambda: None
    executor._should_use_direct_prompt_cache = lambda: True
    executor._run_direct_cached_response = lambda prompt: (
        "ERROR: Codex direct cached request failed (status=400)",
        [],
        _RawResult(raw_responses=[]),
    )
    executor._run_codex_once = lambda prompt, allow_api_key_fallback: (
        "cli fallback",
        [],
        _RawResult(raw_responses=[]),
    )

    assert executor._run_codex("prompt")[0] == "cli fallback"


def test_direct_transport_retries_transient_overload_before_cli_fallback(monkeypatch):
    executor = _executor()
    executor._ensure_codex_home_and_config = lambda: None
    executor._should_use_direct_prompt_cache = lambda: True
    executor._runtime_env_value = (
        lambda name, default="": "3"
        if name == "CHACK_CODEX_DIRECT_CACHE_MAX_ATTEMPTS"
        else default
    )
    results = iter(
        [
            (
                "ERROR: Codex direct cached request failed: "
                "Our servers are currently overloaded. Please try again later.",
                [],
                _RawResult(raw_responses=[]),
            ),
            (
                "ERROR: Codex direct cached request failed: ChunkedEncodingError: "
                "Response ended prematurely",
                [],
                _RawResult(raw_responses=[]),
            ),
            ("cached success", [], _RawResult(raw_responses=[])),
        ]
    )
    executor._run_direct_cached_response = lambda prompt: next(results)
    executor._direct_cache_retry_delay = lambda attempt: 0
    executor._run_codex_once = lambda prompt, allow_api_key_fallback: (
        "unexpected CLI fallback",
        [],
        _RawResult(raw_responses=[]),
    )
    monkeypatch.setattr(codex_backend.time, "sleep", lambda seconds: None)

    assert executor._run_codex("prompt")[0] == "cached success"


def test_direct_response_surfaces_nested_response_failure(monkeypatch):
    executor = _executor()
    monkeypatch.setattr(
        codex_backend.requests,
        "post",
        lambda *args, **kwargs: _FailedResponse(),
    )

    output, steps, raw = executor._run_direct_cached_response("round one")

    assert output == (
        "ERROR: Codex direct cached request failed: Exact nested provider failure"
    )
    assert steps == []
    assert raw.raw_responses == []
