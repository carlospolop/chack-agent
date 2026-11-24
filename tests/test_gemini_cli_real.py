"""The real Gemini CLI, run through Chack against a local capture server: the\nsystem prompt it sends and how it reaches Gemini models on an LLM gateway."""

from __future__ import annotations

import http.server
import json
import os
import shutil
import threading

import pytest

from chack_agent import llm_gateway
from chack_agent.backends.gemini_cli_backend import build_executor
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

GEMINI = os.environ.get("GEMINI_CLI_PATH") or shutil.which("gemini")
pytestmark = pytest.mark.skipif(not GEMINI, reason="Gemini CLI not installed")

SYSTEM_PROMPT = "You are the canary auditor."


class _Requests(list):
    sent: list


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Requests the CLI sends to the model API; every one is refused."""
    requests = _Requests()
    sent: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            if "generatecontent" in self.path.lower():
                requests.append(json.loads(body or b"{}"))
                sent.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"code":400,"message":"captured","status":"INVALID_ARGUMENT"}}')

        do_GET = do_POST

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_CLI_PATH", GEMINI)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("CAPTURE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-home"))
    # The CLI must not pick up the developer's own login or project files.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "GEMINI.md").write_text("canary project notes\n")
    for name in [n for n in os.environ if n.startswith("LLM_GATEWAY_")]:
        monkeypatch.delenv(name)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GEMINI_SYSTEM_MD"):
        monkeypatch.delenv(name, raising=False)
    requests.sent = sent  # type: ignore[attr-defined]
    yield requests
    server.shutdown()


def _first_request(captured, **tools):
    config = ChackConfig(
        model=ModelConfig(primary="gemini-3-flash", provider="gemini"),
        agent=AgentConfig(main_action="test", sub_action="gemini"),
        session=SessionConfig(max_turns=2, long_term_memory_enabled=False),
        tools=ToolsConfig(task_steps_manager_enabled=False, exec_enabled=True, **tools),
        credentials=CredentialsConfig(gemini_api_key="fake-gemini-key"),
        logging=LoggingConfig(level="WARNING"),
        system_prompt=SYSTEM_PROMPT,
        env={},
    )
    executor = build_executor(
        config, system_prompt=SYSTEM_PROMPT, max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
    )
    executor.invoke({"input": "Say hello."})
    assert captured, "the Gemini CLI sent no request to the model API"
    request = captured[0]
    system = " ".join(part.get("text", "") for part in request["systemInstruction"]["parts"])
    tools_sent = {
        declaration["name"]
        for group in request.get("tools", [])
        for declaration in group.get("functionDeclarations") or []
    }
    tools_sent |= {key for group in request.get("tools", []) for key in group if key != "functionDeclarations"}
    return system, tools_sent, json.dumps(request["contents"])


def test_the_cli_sends_chacks_system_prompt_instead_of_its_own(captured):
    system, tools, contents = _first_request(captured)

    assert system.strip() == SYSTEM_PROMPT
    assert "mcp_chack_tools_exec" in tools
    # The task is sent once, without the system prompt repeated in it.
    assert "Say hello." in contents
    assert SYSTEM_PROMPT not in contents
    assert "canary project notes" not in contents + system


def test_a_litellm_gateway_serves_gemini_models_to_the_cli(captured, monkeypatch):
    llm_gateway._CATALOG_CACHE.clear()
    catalog = {
        "data": [
            {
                "model_name": "corp/gemini-3.5-flash",
                "litellm_params": {"model": "vertex_ai/gemini-3.5-flash"},
                "model_info": {},
            }
        ]
    }

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return catalog

    monkeypatch.setattr(llm_gateway, "_http_get", lambda url, headers=None, timeout=None: _Response())
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL")
    config = ChackConfig(
        model=ModelConfig(primary="gemini-3.5-flash", provider="gemini", api_key_type="gateway"),
        agent=AgentConfig(main_action="test", sub_action="gemini"),
        session=SessionConfig(max_turns=2, long_term_memory_enabled=False),
        tools=ToolsConfig(task_steps_manager_enabled=False, exec_enabled=True),
        credentials=CredentialsConfig(
            llm_gateway_type="litellm",
            llm_gateway_base_url=os.environ["CAPTURE_URL"] + "/v1",
            llm_gateway_api_key="sk-gateway-test",
            llm_gateway_headers={"X-Team": "red"},
        ),
        logging=LoggingConfig(level="WARNING"),
        system_prompt=SYSTEM_PROMPT,
        env={},
    )
    executor = build_executor(
        config, system_prompt=SYSTEM_PROMPT, max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
    )

    executor.invoke({"input": "Say hello."})
    llm_gateway._CATALOG_CACHE.clear()

    assert captured, "the Gemini CLI sent no request to the gateway"
    request = captured.sent[0]
    assert request["path"].startswith("/v1beta/models/corp/gemini-3.5-flash:")
    assert request["headers"]["x-goog-api-key"] == "sk-gateway-test"
    assert request["headers"]["x-team"] == "red"

def _litellm_config(model: str, base_url: str, **credentials) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary=model, provider="gemini", api_key_type="gateway"),
        agent=AgentConfig(main_action="test", sub_action="gemini"),
        session=SessionConfig(max_turns=2, long_term_memory_enabled=False),
        tools=ToolsConfig(task_steps_manager_enabled=False, exec_enabled=True),
        credentials=CredentialsConfig(
            llm_gateway_base_url=base_url + "/v1", llm_gateway_api_key="sk-gateway-test", **credentials
        ),
        logging=LoggingConfig(level="WARNING"),
        system_prompt=SYSTEM_PROMPT,
        env={},
    )

def _litellm_catalog(monkeypatch, name: str, upstream: str) -> None:
    llm_gateway._CATALOG_CACHE.clear()
    catalog = {"data": [{"model_name": name, "litellm_params": {"model": upstream}, "model_info": {}}]}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return catalog

    monkeypatch.setattr(llm_gateway, "_http_get", lambda url, headers=None, timeout=None: _Response())

def test_a_litellm_gateway_serves_other_models_to_the_cli_through_its_gemini_api(captured, monkeypatch):
    _litellm_catalog(monkeypatch, "corp/kimi-k2.5", "openai/kimi-k2.5")
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL")
    config = _litellm_config("corp/kimi-k2.5", os.environ["CAPTURE_URL"], llm_gateway_type="litellm")

    build_executor(
        config, system_prompt=SYSTEM_PROMPT, max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
    ).invoke({"input": "Say hello."})
    llm_gateway._CATALOG_CACHE.clear()

    assert captured, "the Gemini CLI sent no request to the gateway"
    assert captured.sent[0]["path"].startswith("/v1beta/models/corp/kimi-k2.5:")
    assert captured.sent[0]["headers"]["x-goog-api-key"] == "sk-gateway-test"

def test_a_custom_gateway_serves_the_cli_on_the_gemini_api_it_names(captured, monkeypatch):
    _litellm_catalog(monkeypatch, "corp/gemini-3.5-flash", "corp/gemini-3.5-flash")
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL")
    config = _litellm_config(
        "corp/gemini-3.5-flash",
        "http://127.0.0.1:9",
        llm_gateway_type="custom",
        llm_gateway_gemini_base_url=os.environ["CAPTURE_URL"] + "/google",
    )

    build_executor(
        config, system_prompt=SYSTEM_PROMPT, max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
    ).invoke({"input": "Say hello."})
    llm_gateway._CATALOG_CACHE.clear()

    assert captured, "the Gemini CLI sent no request to the gateway"
    assert captured.sent[0]["path"].startswith("/google/v1beta/models/corp/gemini-3.5-flash:")

def test_a_gateway_answer_without_token_usage_is_reported(monkeypatch, tmp_path, caplog):
    """A translated Gemini API answer with zero usage must report unknown cost."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            answer = {
                "candidates": [{"content": {"parts": [{"text": "hello"}], "role": "model"},
                                "finishReason": "STOP", "index": 0}],
                "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
            }
            body = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_CLI_PATH", GEMINI)
    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    _litellm_catalog(monkeypatch, "corp/glm-5.2", "openai/glm-5.2")
    config = _litellm_config(
        "corp/glm-5.2", f"http://127.0.0.1:{server.server_address[1]}", llm_gateway_type="litellm"
    )
    try:
        with caplog.at_level("WARNING", logger="chack.gemini_cli_backend"):
            result = build_executor(
                config, system_prompt=SYSTEM_PROMPT, max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
            ).invoke({"input": "Say hello."})
    finally:
        server.shutdown()
        llm_gateway._CATALOG_CACHE.clear()

    assert "hello" in str(result.get("output"))
    assert "reported no token usage for corp/glm-5.2" in caplog.text


def _session_gateway(after_tool_call):
    """A LiteLLM-style Gemini API for one agent session: the first model call
    asks for Chack's exec tool, later calls follow ``after_tool_call`` steps
    (``(status, payload)``), whose last step repeats."""

    calls: list[str] = []

    def _usage(prompt):
        return {"promptTokenCount": prompt, "candidatesTokenCount": 20, "totalTokenCount": prompt + 20}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            calls.append(self.path)
            declarations = [
                declaration
                for tool in request.get("tools") or []
                for declaration in tool.get("functionDeclarations") or []
            ]
            if not declarations:
                # A helper call of the CLI's own (routing, summaries): answer it plainly.
                status, payload = 200, {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                                                        "finishReason": "STOP"}], "usageMetadata": _usage(10)}
            elif not any(
                part.get("functionResponse")
                for content in request.get("contents") or []
                for part in content.get("parts") or []
            ):
                exec_tool = next(d["name"] for d in declarations if d["name"].endswith("exec"))
                status, payload = 200, {
                    "candidates": [{"content": {"role": "model", "parts": [
                        {"functionCall": {"name": exec_tool, "args": {"command": "echo tool-ran"}}}]},
                        "finishReason": "STOP"}],
                    "usageMetadata": _usage(1000),
                }
            else:
                agent_calls = sum(1 for path in calls) - 1
                status, payload = after_tool_call[min(agent_calls, len(after_tool_call)) - 1]
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


_UPSTREAM_TIMEOUT = (504, b'{"error":{"message":"upstream request timeout","code":504}}')


def _run_session(monkeypatch, tmp_path, after_tool_call):
    from chack_agent.backends import transient_retry

    monkeypatch.setattr(transient_retry, "BASE_DELAY_SECONDS", 0.05)
    server, calls = _session_gateway(after_tool_call)
    monkeypatch.setenv("GEMINI_CLI_PATH", GEMINI)
    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHACK_EXEC_CWD", str(tmp_path))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    _litellm_catalog(monkeypatch, "corp/flash", "openai/flash")
    config = _litellm_config("corp/flash", f"http://127.0.0.1:{server.server_address[1]}", llm_gateway_type="litellm")
    config.session.max_turns = 6
    try:
        result = build_executor(
            config, system_prompt=SYSTEM_PROMPT, max_turns=6, memory_max_messages=10, memory_reset_to_messages=5
        ).invoke({"input": "Run the check, then say finished."})
    finally:
        server.shutdown()
        llm_gateway._CATALOG_CACHE.clear()
    usage = [response["usage"] for response in result["raw_result"].raw_responses]
    return result, usage, calls


def test_a_gateway_504_mid_session_is_retried_and_the_session_finishes(monkeypatch, tmp_path):
    finished = {"candidates": [{"content": {"role": "model", "parts": [{"text": "finished"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 20}}
    result, usage, _ = _run_session(monkeypatch, tmp_path, [_UPSTREAM_TIMEOUT, (200, finished)])

    assert "finished" in result["output"]
    assert any("exec" in step[0].tool for step in result["intermediate_steps"])
    # Both answers the model gave are counted; the 504 is not.
    assert sorted(entry["input_tokens"] for entry in usage if entry["input_tokens"] >= 1000) == [1000, 1500]


def test_a_persistent_gateway_504_ends_the_session_with_its_usage_recorded(monkeypatch, tmp_path):
    from chack_agent.backends import transient_retry

    result, usage, calls = _run_session(monkeypatch, tmp_path, [_UPSTREAM_TIMEOUT])

    assert result["output"].startswith("ERROR: Gemini")
    assert "upstream request timeout" in result["output"]
    # The tool-call answer before the failure was billed and is reported.
    assert sum(entry["input_tokens"] for entry in usage) >= 1000
    # After the tool call, the failing call got the relay's attempts; the CLI's
    # own re-sends of it (three in Gemini CLI 0.67) got one attempt each.
    failing_call_attempts = len(calls) - 1
    assert transient_retry.DEFAULT_ATTEMPTS <= failing_call_attempts <= 2 * transient_retry.DEFAULT_ATTEMPTS


def test_a_run_ended_by_a_gateway_504_is_resumed_in_the_same_cli_session_by_a_later_run(monkeypatch, tmp_path):
    from chack_agent.backends import transient_retry

    monkeypatch.setattr(transient_retry, "BASE_DELAY_SECONDS", 0.05)
    finished = {"candidates": [{"content": {"role": "model", "parts": [{"text": "finished"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 20}}
    after_tool_call = [_UPSTREAM_TIMEOUT]
    server, _ = _session_gateway(after_tool_call)
    monkeypatch.setenv("GEMINI_CLI_PATH", GEMINI)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHACK_EXEC_CWD", str(tmp_path))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    _litellm_catalog(monkeypatch, "corp/flash", "openai/flash")
    config = _litellm_config("corp/flash", f"http://127.0.0.1:{server.server_address[1]}", llm_gateway_type="litellm")
    config.session.max_turns = 6

    def _executor():
        return build_executor(
            config, system_prompt=SYSTEM_PROMPT, max_turns=6, memory_max_messages=10, memory_reset_to_messages=5
        )

    try:
        monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "first-process"))
        first = _executor()
        failed = first.invoke({"input": "Run the check, then say finished."})
        handle = first.cli_session_handle()

        # A later process, with a home of its own, gets the handle and the gateway has recovered.
        after_tool_call[:] = [(200, finished)]
        monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "second-process"))
        second = _executor()
        assert second.adopt_cli_session(handle)
        resumed = second.invoke({"input": "Run the check, then say finished."})
    finally:
        server.shutdown()
        llm_gateway._CATALOG_CACHE.clear()

    assert failed["output"].startswith("ERROR: Gemini")
    assert any("exec" in step[0].tool for step in failed["intermediate_steps"])
    assert handle and handle["backend"] == "gemini" and handle["home"].startswith(str(tmp_path / "first-process"))
    assert "finished" in resumed["output"]
    # The resumed session already held the tool call and its result, so the check is not run again.
    assert not any("exec" in step[0].tool for step in resumed["intermediate_steps"])
    assert second.cli_session_handle()["session_id"] == handle["session_id"]


def test_a_handle_to_a_session_the_cli_no_longer_has_starts_a_new_session(monkeypatch, tmp_path):
    finished = {"candidates": [{"content": {"role": "model", "parts": [{"text": "finished"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 20}}
    server, _ = _session_gateway([(200, finished)])
    monkeypatch.setenv("GEMINI_CLI_PATH", GEMINI)
    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHACK_EXEC_CWD", str(tmp_path))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    _litellm_catalog(monkeypatch, "corp/flash", "openai/flash")
    config = _litellm_config("corp/flash", f"http://127.0.0.1:{server.server_address[1]}", llm_gateway_type="litellm")
    config.session.max_turns = 6
    emptied = tmp_path / "emptied-home"
    emptied.mkdir()
    try:
        executor = build_executor(
            config, system_prompt=SYSTEM_PROMPT, max_turns=6, memory_max_messages=10, memory_reset_to_messages=5
        )
        assert executor.adopt_cli_session(
            {"backend": "gemini", "session_id": "0f6c2f6e-1111-4b7a-9d55-3c1d2a7e9b10", "home": str(emptied)}
        )
        result = executor.invoke({"input": "Run the check, then say finished."})
    finally:
        server.shutdown()
        llm_gateway._CATALOG_CACHE.clear()

    # The task ran whole in a new session: the check, then the answer.
    assert "finished" in result["output"]
    assert any("exec" in step[0].tool for step in result["intermediate_steps"])
    assert executor.cli_session_handle()["session_id"] != "0f6c2f6e-1111-4b7a-9d55-3c1d2a7e9b10"
