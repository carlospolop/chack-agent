"""Web tools of the backends that call the model API directly."""

from __future__ import annotations

import http.server
import json
import os
import threading

import pytest
from agents import FunctionTool, WebSearchTool

from chack_agent.backends import gateway_responses_backend, hosted_web_search, langgraph_backend, openai_compaction_backend
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


@pytest.fixture
def model_server(monkeypatch):
    """An OpenAI-compatible model API that answers "done" and records each request."""
    requests_seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            requests_seen.append({"path": self.path, "body": body})
            if self.path.endswith("/chat/completions"):
                reply = {
                    "id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "done"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            else:
                reply = {
                    "id": "r1", "object": "response", "created_at": 0, "model": body["model"],
                    "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                    "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                                "content": [{"type": "output_text", "text": "done", "annotations": []}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                              "input_tokens_details": {"cached_tokens": 0},
                              "output_tokens_details": {"reasoning_tokens": 0}},
                }
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENAI_BASE_URL", "CHACK_GATEWAY_HOSTED_WEB_SEARCH"):
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("LLM_GATEWAY_"):
            monkeypatch.delenv(name, raising=False)
    hosted_web_search._probe_cache.clear()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests_seen
    hosted_web_search._probe_cache.clear()
    server.shutdown()


def _config(provider: str, primary: str = "gpt-5", tools: dict | None = None, **credentials) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary=primary, provider=provider),
        agent=AgentConfig(main_action="test", sub_action="test"),
        session=SessionConfig(),
        tools=ToolsConfig(task_steps_manager_enabled=False, **(tools or {})),
        credentials=CredentialsConfig(openai_api_key="sk-test", openrouter_api_key="sk-or-test", **credentials),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )


def _executor(backend, config):
    return backend.build_executor(
        config, system_prompt="system", max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
    )


def _function_tools(executor) -> set[str]:
    if hasattr(executor, "_function_tools_by_name"):
        return set(executor._function_tools_by_name)
    return {tool.name for tool in executor.agent.tools if isinstance(tool, FunctionTool)}


def _hosted_tools(executor) -> list:
    agent = getattr(executor, "agent", None)  # chat-completions runs have no hosted tools
    return [tool for tool in getattr(agent, "tools", []) if isinstance(tool, WebSearchTool)]


def _sent_tool_types(request: dict) -> list[str]:
    return [tool.get("type") for tool in request["body"].get("tools") or []]


def _sent_function_names(request: dict) -> set[str]:
    names = set()
    for tool in request["body"].get("tools") or []:
        if tool.get("type") == "function":
            names.add(tool.get("name") or (tool.get("function") or {}).get("name"))
    return names


WEB = {"web_search", "web_fetch"}


def test_openai_runs_get_openais_own_web_search_and_no_keyless_tools(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    executor = _executor(openai_compaction_backend, _config("openai"))

    assert not _function_tools(executor) & WEB
    assert len(_hosted_tools(executor)) == 1


def test_openai_compatible_endpoints_get_the_keyless_tools(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    executor = _executor(openai_compaction_backend, _config("openai"))

    assert _function_tools(executor) >= WEB
    assert not _hosted_tools(executor)


@pytest.mark.parametrize("backend, provider", [(gateway_responses_backend, "openrouter"), (langgraph_backend, "langgraph")])
def test_openrouter_models_with_native_search_use_openrouters_web_tools(model_server, backend, provider):
    base_url, seen = model_server
    executor = _executor(backend, _config(provider, "openai/gpt-5-mini", openrouter_base_url=base_url))

    assert executor.invoke({"input": "hi"})["output"] == "done"

    sent = seen[0]
    assert "openrouter:web_search" in _sent_tool_types(sent)
    assert "openrouter:web_fetch" in _sent_tool_types(sent)
    assert not _sent_function_names(sent) & WEB
    # Server tool results bypass chack's output limits, so they are capped.
    fetch = next(t for t in sent["body"]["tools"] if t.get("type") == "openrouter:web_fetch")
    assert fetch["parameters"]["max_content_tokens"] > 0


@pytest.mark.parametrize("backend, provider", [(gateway_responses_backend, "openrouter"), (langgraph_backend, "langgraph")])
def test_openrouter_models_without_native_search_search_keylessly(model_server, backend, provider):
    base_url, seen = model_server
    executor = _executor(backend, _config(provider, "z-ai/glm-4.6", openrouter_base_url=base_url))

    executor.invoke({"input": "hi"})

    sent = seen[0]
    # OpenRouter would bill a search for this model; its free page reading stays.
    assert "openrouter:web_search" not in _sent_tool_types(sent)
    assert "openrouter:web_fetch" in _sent_tool_types(sent)
    assert "web_search" in _sent_function_names(sent)
    assert "web_fetch" not in _sent_function_names(sent)


def _gateway_config(base_url: str, tools: dict | None = None) -> ChackConfig:
    return _config(
        "gateway", "corp/claude-haiku", tools,
        llm_gateway_type="custom", llm_gateway_base_url=base_url, llm_gateway_api_key="sk-gw",
    )


def test_gateways_that_serve_web_search_get_it_once_checked(monkeypatch):
    probes = []

    def fake_post(url, json=None, headers=None, timeout=None):
        probes.append({"url": url, "body": json, "headers": headers})
        return _FakeHttpResponse(200, {"id": "r1", "output": [{"type": "web_search_call", "status": "completed"}]})

    monkeypatch.setattr(hosted_web_search.requests, "post", fake_post)
    monkeypatch.delenv("CHACK_GATEWAY_HOSTED_WEB_SEARCH", raising=False)
    hosted_web_search._probe_cache.clear()

    first = _executor(gateway_responses_backend, _gateway_config("http://gw.test/v1"))
    second = _executor(gateway_responses_backend, _gateway_config("http://gw.test/v1"))

    for executor in (first, second):
        assert len(_hosted_tools(executor)) == 1
        assert "web_fetch" in _function_tools(executor)
        assert "web_search" not in _function_tools(executor)
    (probe,) = probes
    assert probe["url"] == "http://gw.test/v1/responses"
    assert probe["body"]["tools"] == [{"type": "web_search"}]
    assert probe["body"]["tool_choice"] == "required"
    assert probe["headers"]["Authorization"] == "Bearer sk-gw"


@pytest.mark.parametrize(
    "status, payload",
    [
        (400, {"error": {"message": "web search is not authorized"}}),
        (200, {"error": {"message": "web search is not authorized"}}),
        # A search cut off before it ran proves nothing.
        (200, {"output": [{"type": "web_search_call", "status": "incomplete"}]}),
        # The model answered without searching: the tool was dropped.
        (200, {"output": [{"type": "message", "content": []}]}),
    ],
)
def test_gateways_that_refuse_web_search_get_the_keyless_tools(monkeypatch, status, payload):
    monkeypatch.setattr(hosted_web_search.requests, "post", lambda *a, **k: _FakeHttpResponse(status, payload))
    monkeypatch.delenv("CHACK_GATEWAY_HOSTED_WEB_SEARCH", raising=False)
    hosted_web_search._probe_cache.clear()

    executor = _executor(gateway_responses_backend, _gateway_config("http://gw.test/v1"))

    assert not _hosted_tools(executor)
    assert _function_tools(executor) >= WEB


@pytest.mark.parametrize("forced, hosted", [("on", True), ("off", False)])
def test_the_gateway_check_can_be_skipped(monkeypatch, forced, hosted):
    def no_probe(*args, **kwargs):
        raise AssertionError("the gateway should not be probed")

    monkeypatch.setattr(hosted_web_search.requests, "post", no_probe)
    monkeypatch.setenv("CHACK_GATEWAY_HOSTED_WEB_SEARCH", forced)

    executor = _executor(gateway_responses_backend, _gateway_config("http://gw.test/v1"))

    assert bool(_hosted_tools(executor)) is hosted
    assert ("web_search" in _function_tools(executor)) is not hosted


@pytest.mark.parametrize(
    "backend, provider, primary",
    [
        (openai_compaction_backend, "openai", "gpt-5"),
        (gateway_responses_backend, "openrouter", "openai/gpt-5-mini"),
        (gateway_responses_backend, "openrouter", "z-ai/glm-4.6"),
        (langgraph_backend, "langgraph", "openai/gpt-5-mini"),
    ],
)
def test_turning_web_off_removes_every_web_tool(model_server, backend, provider, primary):
    base_url, seen = model_server
    executor = _executor(
        backend, _config(provider, primary, {"native_web_search_enabled": False}, openrouter_base_url=base_url)
    )

    assert not _function_tools(executor) & WEB
    assert not _hosted_tools(executor)
    if provider != "openai":
        executor.invoke({"input": "hi"})
        assert not [t for t in _sent_tool_types(seen[0]) if str(t).startswith("openrouter:")]


def test_turning_keyless_search_off_keeps_only_the_providers_tools(model_server):
    base_url, seen = model_server
    executor = _executor(
        gateway_responses_backend,
        _config("openrouter", "z-ai/glm-4.6", {"keyless_web_search_enabled": False}, openrouter_base_url=base_url),
    )

    executor.invoke({"input": "hi"})

    assert not _function_tools(executor) & WEB
    assert _sent_tool_types(seen[0]).count("openrouter:web_fetch") == 1


class _FakeHttpResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload
