"""OpenAI-compatible backends retry a gateway's transient failures a bounded number of times."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import openai._base_client as openai_base_client
import pytest
from langchain_core.messages import HumanMessage

from chack_agent import llm_gateway
from chack_agent.backends import transient_retry
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "corp/flash",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "finished"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
}
MODELS = {"object": "list", "data": []}
UPSTREAM_TIMEOUT = (504, {"error": {"message": "upstream request timeout", "code": 504}})


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch):
    monkeypatch.setattr(openai_base_client, "INITIAL_RETRY_DELAY", 0.01)
    monkeypatch.setattr(openai_base_client, "MAX_RETRY_DELAY", 0.05)


@pytest.fixture
def gateway():
    """A fake OpenAI-compatible gateway answering each request with the next scripted step."""

    script: list = []
    calls: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            calls.append(self.path)
            status, payload = script[min(len(calls), len(script)) - 1]
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _answer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", script, calls
    finally:
        server.shutdown()
        server.server_close()


def _gateway_config(base_url: str, provider: str = "gateway") -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary="corp/flash", provider=provider, api_key_type="gateway"),
        agent=AgentConfig(),
        session=SessionConfig(long_term_memory_enabled=False),
        tools=ToolsConfig(),
        credentials=CredentialsConfig(
            llm_gateway_type="custom", llm_gateway_base_url=base_url, llm_gateway_api_key="sk-test"
        ),
        logging=LoggingConfig(level="WARNING"),
        system_prompt="test",
        env={},
    )


@pytest.fixture
def no_catalog(monkeypatch):
    llm_gateway._CATALOG_CACHE.clear()

    class _Empty:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    monkeypatch.setattr(llm_gateway, "_http_get", lambda url, headers=None, timeout=None: _Empty())
    yield
    llm_gateway._CATALOG_CACHE.clear()


def _list_models(client):
    async def call():
        return await client.models.list()

    return asyncio.run(call())


def _gateway_responses_client(base_url):
    from chack_agent.backends.gateway_responses_backend import _configure_client_factory

    _, new_client = _configure_client_factory(_gateway_config(base_url))
    return new_client()


def _openai_client(base_url, monkeypatch):
    from chack_agent.backends.openai_compaction_backend import _configure_client_factory

    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = _gateway_config(base_url, provider="openai")
    return _configure_client_factory(config)()


@pytest.mark.parametrize("client_for", ["gateway_responses", "openai"])
def test_a_gateway_504_is_retried_and_the_request_succeeds(gateway, no_catalog, monkeypatch, caplog, client_for):
    base_url, script, calls = gateway
    script[:] = [UPSTREAM_TIMEOUT, (200, MODELS)]
    client = (
        _gateway_responses_client(base_url) if client_for == "gateway_responses" else _openai_client(base_url, monkeypatch)
    )

    with caplog.at_level("WARNING"):
        page = _list_models(client)

    assert page.data == []
    assert len(calls) == 2
    assert "answered 504 on attempt 1/4; the client retries it" in caplog.text


def test_a_persistent_504_is_retried_a_bounded_number_of_times(gateway, no_catalog, caplog):
    base_url, script, calls = gateway
    script[:] = [UPSTREAM_TIMEOUT]
    client = _gateway_responses_client(base_url)

    with caplog.at_level("WARNING"), pytest.raises(openai.InternalServerError):
        _list_models(client)

    assert len(calls) == transient_retry.DEFAULT_ATTEMPTS
    assert "answered 504 on attempt 4/4" in caplog.text


def test_a_client_error_is_not_retried(gateway, no_catalog):
    base_url, script, calls = gateway
    script[:] = [(400, {"error": {"message": "bad request"}}), (200, MODELS)]
    client = _gateway_responses_client(base_url)

    with pytest.raises(openai.BadRequestError):
        _list_models(client)

    assert len(calls) == 1


def test_the_langgraph_model_retries_a_gateway_504_and_gets_the_answer(gateway, no_catalog, caplog):
    from chack_agent.backends.langgraph_backend import build_executor

    base_url, script, calls = gateway
    script[:] = [UPSTREAM_TIMEOUT, (200, COMPLETION)]
    config = _gateway_config(base_url, provider="langgraph")
    executor = build_executor(
        config, system_prompt="test", max_turns=4, memory_max_messages=10,
        memory_reset_to_messages=5, tools_override=[],
    )

    with caplog.at_level("WARNING"):
        reply = executor._model_with_tools.invoke([HumanMessage(content="hi")])

    assert reply.content == "finished"
    assert [path for path in calls] == ["/v1/chat/completions"] * 2
    assert "answered 504 on attempt 1/4" in caplog.text
