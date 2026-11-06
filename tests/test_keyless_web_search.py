"""Keyless web search for backends whose model has no hosted search."""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from chack_agent.backends import gateway_responses_backend, langgraph_backend, openai_compaction_backend
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)
from chack_tools import exa_web_search


@pytest.fixture
def exa_server(monkeypatch):
    """A stand-in for Exa's MCP endpoint that records each call."""
    calls: list[dict] = []
    replies: list[tuple[int, str, str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            calls.append({"path": self.path, "body": body})
            status, content_type, payload = replies.pop(0) if replies else (200, "application/json", "{}")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(exa_web_search, "EXA_MCP_URL", f"http://127.0.0.1:{server.server_address[1]}/mcp")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    yield calls, replies
    server.shutdown()


def _event_stream(result: dict) -> str:
    # Exa streams its answer; the text may hold raw control characters.
    message = json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).replace("\\t", "\t")
    return f"event: message\ndata: {message}\n\n"


def test_search_returns_the_result_text(exa_server):
    calls, replies = exa_server
    replies.append((200, "text/event-stream", _event_stream(
        {"content": [{"type": "text", "text": "Title: requests\tv2\nURL: https://pypi.org/project/requests/"}]}
    )))

    text = exa_web_search.web_search("latest requests release", num_results=50)

    assert text == "Title: requests\tv2\nURL: https://pypi.org/project/requests/"
    (call,) = calls
    assert call["body"]["params"]["name"] == "web_search_exa"
    # Result counts are capped and a missing objective falls back to the query.
    assert call["body"]["params"]["arguments"] == {
        "query": "latest requests release",
        "numResults": 10,
        "objective": "latest requests release",
    }


def test_an_api_key_is_sent_when_configured(exa_server, monkeypatch):
    calls, replies = exa_server
    replies.append((200, "text/event-stream", _event_stream({"content": [{"type": "text", "text": "ok"}]})))
    monkeypatch.setenv("EXA_API_KEY", "k&1")

    exa_web_search.web_search("q")

    assert calls[0]["path"] == "/mcp?exaApiKey=k%261"


def test_failures_come_back_as_errors_the_model_can_read(exa_server):
    _calls, replies = exa_server
    replies.append((200, "application/json", json.dumps(
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Invalid arguments"}}
    )))
    replies.append((200, "text/event-stream", _event_stream(
        {"isError": True, "content": [{"type": "text", "text": "page unreachable"}]}
    )))
    replies.append((429, "application/json", "{}"))

    assert exa_web_search.web_search("q") == "ERROR: Invalid arguments"
    assert exa_web_search.web_fetch(["https://example.com"]) == "ERROR: page unreachable"
    assert exa_web_search.web_search("q").startswith("ERROR: the free web search is rate limited")


def test_fetch_bounds_what_it_asks_for(exa_server):
    calls, replies = exa_server
    replies.append((200, "text/event-stream", _event_stream({"content": [{"type": "text", "text": "page"}]})))

    assert exa_web_search.web_fetch([f"https://example.com/{i}" for i in range(6)]).startswith("ERROR")
    assert exa_web_search.web_fetch(["  ", ""]).startswith("ERROR")
    assert exa_web_search.web_search("   ").startswith("ERROR")
    assert exa_web_search.web_fetch([" https://example.com "], max_characters=10**6) == "page"

    (call,) = calls
    assert call["body"]["params"]["arguments"] == {"urls": ["https://example.com"], "maxCharacters": 20000}


def _config(provider: str, **tools) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary="gpt-5", provider=provider),
        agent=AgentConfig(main_action="test", sub_action="test"),
        session=SessionConfig(),
        tools=ToolsConfig(task_steps_manager_enabled=False, **tools),
        credentials=CredentialsConfig(openai_api_key="sk-test", openrouter_api_key="sk-or-test"),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )


def _offered_tools(backend, provider: str, **tools) -> set[str]:
    executor = backend.build_executor(
        _config(provider, **tools),
        system_prompt="system",
        max_turns=2,
        memory_max_messages=10,
        memory_reset_to_messages=5,
    )
    if hasattr(executor, "_function_tools_by_name"):
        return set(executor._function_tools_by_name)
    return {tool.name for tool in executor.agent.tools}


@pytest.mark.parametrize(
    "backend, provider",
    [
        (openai_compaction_backend, "openai"),
        (gateway_responses_backend, "openrouter"),
        (langgraph_backend, "langgraph"),
    ],
)
def test_api_backends_get_keyless_search_unless_switched_off(backend, provider):
    web = {"web_search", "web_fetch"}

    assert _offered_tools(backend, provider) >= web
    assert not _offered_tools(backend, provider, native_web_search_enabled=False) & web
    assert not _offered_tools(backend, provider, keyless_web_search_enabled=False) & web
