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
