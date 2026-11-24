"""The real opencode CLI, run through Chack against a fake OpenAI-compatible
gateway that fails with 504: opencode's own retry carries the session through
a transient failure, and a persistent one ends the run with its usage counted.

Set ``OPENCODE_CLI_PATH`` if ``opencode`` is not on ``PATH``.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import threading

import pytest

from chack_agent import llm_gateway
from chack_agent.backends.opencode_cli_backend import build_executor
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

OPENCODE = os.environ.get("OPENCODE_CLI_PATH") or shutil.which("opencode")
pytestmark = pytest.mark.skipif(not OPENCODE, reason="opencode CLI not installed")

MODEL = "corp/glm-5.2"
UPSTREAM_TIMEOUT = (504, {"error": {"message": "upstream request timeout", "code": 504}})


def _chunk(delta=None, finish=None, usage=None):
    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": MODEL,
             "choices": [] if usage else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        chunk["usage"] = {"prompt_tokens": usage, "completion_tokens": 20, "total_tokens": usage + 20}
    return chunk


def _stream(*chunks):
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"


def _text_answer(text, prompt_tokens):
    return _stream(_chunk({"role": "assistant", "content": text}), _chunk(finish="stop"), _chunk(usage=prompt_tokens))


def _session_gateway(after_tool_call):
    """A gateway for one agent session: the first agent call asks for Chack's
    exec tool; calls after its result follow ``after_tool_call`` steps
    (``(status, payload)``, or ``"finished"``), whose last step repeats."""

    agent_calls: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            tools = [tool["function"]["name"] for tool in request.get("tools") or []]
            if not tools:
                # opencode's own helper calls (titles): answer plainly.
                return self._send(200, _text_answer("ok", 10))
            agent_calls.append(self.path)
            if not any(message.get("role") == "tool" for message in request.get("messages") or []):
                exec_tool = next(name for name in tools if name.endswith("exec"))
                call = {"index": 0, "id": "call_1", "type": "function",
                        "function": {"name": exec_tool, "arguments": json.dumps({"command": "echo tool-ran"})}}
                return self._send(200, _stream(
                    _chunk({"role": "assistant", "tool_calls": [call]}), _chunk(finish="tool_calls"), _chunk(usage=1000)
                ))
            step = after_tool_call[min(len(agent_calls) - 1, len(after_tool_call)) - 1]
            if step == "finished":
                return self._send(200, _text_answer("finished", 1500))
            status, payload = step
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _send(self, status, body, content_type="text/event-stream"):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, agent_calls


class _Catalog:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _run_session(monkeypatch, tmp_path, after_tool_call):
    server, agent_calls = _session_gateway(after_tool_call)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    for name in list(os.environ):
        if name.startswith(("LLM_GATEWAY_", "OPENCODE_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENCODE_CLI_PATH", OPENCODE)
    monkeypatch.setenv("CHACK_OPENCODE_HOME_BASE", str(tmp_path / "opencode-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHACK_EXEC_CWD", str(tmp_path))
    llm_gateway._CATALOG_CACHE.clear()
    catalog = {"data": [{"model_name": MODEL, "litellm_params": {"model": "zai/glm-5.2"},
                         "model_info": {"max_input_tokens": 200_000}}]}
    monkeypatch.setattr(llm_gateway, "_http_get", lambda url, headers=None, timeout=None: _Catalog(catalog))
    config = ChackConfig(
        model=ModelConfig(primary=MODEL, provider="opencode", api_key_type="gateway"),
        agent=AgentConfig(main_action="test", sub_action="opencode"),
        session=SessionConfig(max_turns=6, long_term_memory_enabled=False),
        tools=ToolsConfig(task_steps_manager_enabled=False, exec_enabled=True),
        credentials=CredentialsConfig(
            llm_gateway_type="litellm", llm_gateway_base_url=f"{base_url}/v1", llm_gateway_api_key="sk-test"
        ),
        logging=LoggingConfig(level="WARNING"),
        system_prompt="You are the test agent.",
        env={},
    )
    try:
        result = build_executor(
            config, system_prompt="You are the test agent.", max_turns=6,
            memory_max_messages=10, memory_reset_to_messages=5,
        ).invoke({"input": "Run the check, then say finished."})
    finally:
        server.shutdown()
        llm_gateway._CATALOG_CACHE.clear()
    usage = [response["usage"] for response in result["raw_result"].raw_responses]
    return result, usage, agent_calls


def test_a_gateway_504_mid_session_is_retried_by_opencode_and_the_session_finishes(monkeypatch, tmp_path):
    result, usage, agent_calls = _run_session(monkeypatch, tmp_path, [UPSTREAM_TIMEOUT, "finished"])

    assert "finished" in result["output"]
    assert any("exec" in step[0].tool for step in result["intermediate_steps"])
    # Tool call, the 504, and the retried call that answered.
    assert len(agent_calls) == 3
    assert sorted(entry["input_tokens"] for entry in usage) == [1000, 1500]


def test_a_persistent_gateway_504_ends_the_opencode_session_with_its_usage_recorded(monkeypatch, tmp_path):
    result, usage, agent_calls = _run_session(monkeypatch, tmp_path, [UPSTREAM_TIMEOUT])

    assert result["output"].startswith("ERROR")
    assert "upstream request timeout" in result["output"]
    # The tool-call answer before the failure was billed and is reported.
    assert sum(entry["input_tokens"] for entry in usage) == 1000
    # opencode 1.19.5 retries a failed model call 5 times (session/retry.ts RETRY_MAX_RETRIES).
    assert len(agent_calls) - 1 == 6
