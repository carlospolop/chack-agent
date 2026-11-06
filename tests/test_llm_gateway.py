import json
import os
import tomllib

import pytest

from chack_agent import llm_gateway
from chack_agent.agent import Chack
from chack_agent.backends import build_executor
from chack_agent.backends.claude_code_backend import build_executor as build_claude_executor
from chack_agent.backends.codex_backend import build_executor as build_codex_executor
from chack_agent.backends.gemini_cli_backend import build_executor as build_gemini_executor
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)
from chack_agent.llm_gateway import get_gateway_route, list_gateway_models
from chack_agent.pricing import _package_pricing_path


GATEWAY_URL = "https://gateway.example.test/v1"
GATEWAY_KEY = "sk-gateway-test"

# Shape of a LiteLLM proxy's /v1/model/info answer, trimmed to what matters.
LITELLM_MODEL_INFO = {
    "data": [
        {
            "model_name": "corp/opus-5.5",
            "litellm_params": {"model": "vertex_ai/claude-opus-5-5"},
            "model_info": {
                "max_input_tokens": 1_000_000,
                "input_cost_per_token": 4e-06,
                "cache_read_input_token_cost": 2e-07,
                "cache_creation_input_token_cost": 5e-06,
                "output_cost_per_token": 2e-05,
            },
        },
        {
            "model_name": "corp/claude-opus-5.5",
            "litellm_params": {"model": "vertex_ai/claude-opus-5-5"},
            "model_info": {"max_input_tokens": 1_000_000},
        },
        {
            "model_name": "corp/claude-haiku-5.5",
            "litellm_params": {"model": "vertex_ai/claude-haiku-5-5"},
            "model_info": {"input_cost_per_token": 1e-07, "output_cost_per_token": 5e-07},
        },
        {
            "model_name": "corp/claude-sonnet-5.5",
            "litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-5-5-v1:0"},
            "model_info": {},
        },
        {
            "model_name": "corp/gpt-6-sol",
            "litellm_params": {"model": "bedrock_mantle/openai.gpt-6-sol"},
            "model_info": {
                "max_input_tokens": 922_000,
                "input_cost_per_token": 2.2e-06,
                "cache_read_input_token_cost": 2.2e-07,
                "output_cost_per_token": 1.1e-05,
            },
        },
        {
            "model_name": "corp/gpt-5.6-sol",
            "litellm_params": {"model": "global.openai.gpt-5.6-sol"},
            "model_info": {},
        },
        {
            "model_name": "corp/auto",
            "litellm_params": {"model": "auto_router/complexity_router"},
            "model_info": {},
        },
    ]
}


# Shape of the OpenAI-compatible /v1/models answer of other gateways.
OPENAI_MODELS = {
    "data": [
        {"id": "anthropic/claude-opus-5.5", "object": "model", "context_length": 1_000_000},
        {"id": "openai/gpt-6-sol", "object": "model"},
        {"id": "my-claude", "object": "model"},
    ]
}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("LLM_GATEWAY_", "OPENROUTER_", "CLAUDE_", "ANTHROPIC_", "CODEX_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHACK_CODEX_HOME_BASE", str(tmp_path / "codex"))
    monkeypatch.setenv("CHACK_CLAUDE_HOME_BASE", str(tmp_path / "claude"))
    llm_gateway._CATALOG_CACHE.clear()

    def _offline(url, headers=None, timeout=None):
        raise ConnectionError(f"network disabled in tests: {url}")

    monkeypatch.setattr(llm_gateway, "_http_get", _offline)
    yield
    llm_gateway._CATALOG_CACHE.clear()


@pytest.fixture
def catalog_requests(monkeypatch):
    calls = []

    def _fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": dict(headers or {})})
        if url.endswith("/model/info"):
            return _FakeResponse(LITELLM_MODEL_INFO)
        if url.endswith("/models"):
            return _FakeResponse(OPENAI_MODELS)
        return _FakeResponse({"error": "not found"}, status=404)

    monkeypatch.setattr(llm_gateway, "_http_get", _fake_get)
    return calls


def _config(provider, primary, *, gateway_type="litellm", api_key_type="", **credentials):
    defaults = dict(
        llm_gateway_type=gateway_type,
        llm_gateway_base_url=GATEWAY_URL,
        llm_gateway_api_key=GATEWAY_KEY,
    )
    defaults.update(credentials)
    return ChackConfig(
        model=ModelConfig(primary=primary, provider=provider, api_key_type=api_key_type),
        agent=AgentConfig(main_action="test", sub_action="gateway"),
        session=SessionConfig(max_turns=2),
        tools=ToolsConfig(),
        credentials=CredentialsConfig(**defaults),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )


def _build(builder, config):
    return builder(
        config,
        system_prompt="system",
        max_turns=2,
        memory_max_messages=10,
        memory_reset_to_messages=5,
    )


# --- route selection -------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "primary", "api_key_type"),
    [
        ("gateway", "corp/opus-5.5", ""),
        ("codex", "gateway/corp/opus-5.5", ""),
        ("claude", "corp/opus-5.5", "gateway"),
        ("claude", "corp/opus-5.5", "litellm"),
    ],
)
def test_each_selector_routes_through_the_gateway(provider, primary, api_key_type, catalog_requests):
    config = _config(provider, primary)
    config.model.api_key_type = api_key_type

    route = get_gateway_route(config)

    assert route is not None
    assert route.model_name == "corp/opus-5.5"
    assert route.base_url == GATEWAY_URL
    assert route.anthropic_base_url == "https://gateway.example.test"
    assert route.api_key == GATEWAY_KEY
    assert route.context_window == 1_000_000


def test_vendor_models_without_a_selector_do_not_use_the_gateway(catalog_requests):
    assert get_gateway_route(_config("claude", "claude-opus-5-5")) is None
    assert catalog_requests == []


def test_gateway_selected_without_credentials_fails_clearly():
    config = _config("codex", "gateway/gpt-6-sol", llm_gateway_api_key="")

    with pytest.raises(ValueError, match="LLM_GATEWAY_API_KEY"):
        get_gateway_route(config)


def test_unknown_gateway_type_is_rejected():
    with pytest.raises(ValueError, match="LLM_GATEWAY_TYPE"):
        get_gateway_route(_config("gateway", "m", gateway_type="lite-llm"))


def test_gateway_settings_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://env-gateway.test/v1/")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "env-key")
    monkeypatch.setenv("LLM_GATEWAY_HEADERS", '{"X-Team": "red"}')
    monkeypatch.setenv("LLM_GATEWAY_ANTHROPIC_BASE_URL", "https://env-gateway.test/anthropic")
    config = _config(
        "gateway",
        "my-model",
        gateway_type="",
        llm_gateway_base_url="",
        llm_gateway_api_key="",
    )

    route = get_gateway_route(config)

    assert route.kind == "custom"
    assert route.base_url == "https://env-gateway.test/v1"
    assert route.anthropic_base_url == "https://env-gateway.test/anthropic"
    assert route.api_key == "env-key"
    assert route.headers == {"X-Team": "red"}
    assert route.model_name == "my-model"


def test_malformed_header_json_is_reported(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_HEADERS", "X-Team: red")

    with pytest.raises(ValueError, match="LLM_GATEWAY_HEADERS"):
        get_gateway_route(_config("gateway", "m", gateway_type="custom"))


def test_openrouter_routes_keep_their_own_credentials():
    config = _config(
        "codex",
        "openrouter/openai/gpt-6-sol",
        openrouter_api_key="or-key",
        openrouter_app_name="chack",
    )

    route = get_gateway_route(config)

    assert route.is_openrouter
    assert route.model_name == "openai/gpt-6-sol"
    assert route.api_key == "or-key"
    assert route.headers == {"X-Title": "chack"}


# --- model names -----------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        # Vendor IDs map to the shortest gateway alias serving that upstream.
        ("claude-opus-5-5", "corp/opus-5.5"),
        ("claude-haiku-5-5", "corp/claude-haiku-5.5"),
        ("gpt-6-sol", "corp/gpt-6-sol"),
        ("gpt-5.6-sol", "corp/gpt-5.6-sol"),
        ("gateway/claude-opus-5-5", "corp/opus-5.5"),
        # Gateway names are used as given.
        ("corp/claude-opus-5.5", "corp/claude-opus-5.5"),
        ("corp/auto", "corp/auto"),
    ],
)
def test_litellm_catalog_maps_vendor_model_ids(requested, expected, catalog_requests):
    assert get_gateway_route(_config("gateway", requested)).model_name == expected


def test_catalog_is_read_once_with_the_gateway_key(catalog_requests):
    for model in ("claude-opus-5-5", "gpt-6-sol", "corp/auto"):
        get_gateway_route(_config("gateway", model))

    assert len(catalog_requests) == 1
    assert catalog_requests[0]["url"] == f"{GATEWAY_URL}/model/info"
    assert catalog_requests[0]["headers"]["Authorization"] == f"Bearer {GATEWAY_KEY}"


def test_explicit_model_map_wins_over_the_catalog(catalog_requests):
    config = _config("gateway", "claude-opus-5-5", llm_gateway_model_map={"claude-opus-5-5": "corp/claude-opus-5.5"})

    assert get_gateway_route(config).model_name == "corp/claude-opus-5.5"


@pytest.mark.parametrize("gateway_type", ["litellm", "custom"])
def test_models_the_key_cannot_use_are_rejected_with_the_allowed_list(gateway_type, catalog_requests):
    with pytest.raises(ValueError, match="not available to this gateway API key") as raised:
        get_gateway_route(_config("gateway", "claude-sonnet-9", gateway_type=gateway_type))

    message = str(raised.value)
    assert "claude-sonnet-9" in message
    assert ("corp/gpt-6-sol" if gateway_type == "litellm" else "openai/gpt-6-sol") in message


def test_model_map_targets_must_be_allowed_too(catalog_requests):
    config = _config("gateway", "opus", llm_gateway_model_map={"opus": "corp/opus-9"})

    with pytest.raises(ValueError, match=r"opus \(as corp/opus-9\)"):
        get_gateway_route(config)


def test_unavailable_role_models_are_rejected_before_a_run(catalog_requests):
    config = _config("gemini", "gateway/claude-opus-5-5")
    config.model.cli = "claude-sonnet-9"

    with pytest.raises(ValueError, match="claude-sonnet-9"):
        _build(build_gemini_executor, config)


def test_built_in_role_aliases_the_key_cannot_use_fall_back_to_the_primary_model(catalog_requests):
    config = _config("gemini", "gateway/claude-opus-5-5", gateway_type="custom")
    config.model.travel = "BEST_CHEAPEST"
    config.model.cli = "BEST_QUALITY"

    model = _build(build_gemini_executor, config)._config.model

    assert model.travel == "anthropic/claude-opus-5.5"
    assert model.cli == "openai/gpt-6-sol"


def test_custom_gateways_map_vendor_ids_through_their_model_list(catalog_requests):
    route = get_gateway_route(_config("gateway", "claude-opus-5-5", gateway_type="custom"))

    assert route.model_name == "anthropic/claude-opus-5.5"
    assert route.context_window == 1_000_000
    assert [call["url"] for call in catalog_requests] == [f"{GATEWAY_URL}/models"]


def test_allowed_models_are_listed_for_the_configured_key(catalog_requests):
    litellm_models = list_gateway_models(_config("gateway", ""))
    custom_models = list_gateway_models(_config("gateway", "", gateway_type="custom"))

    assert [model.name for model in litellm_models][:2] == ["corp/auto", "corp/claude-haiku-5.5"]
    assert len(litellm_models) == 7
    assert [model.name for model in list_gateway_models(_config("gateway", "").credentials)] == [
        model.name for model in litellm_models
    ]
    assert [model.name for model in custom_models] == [
        "anthropic/claude-opus-5.5",
        "my-claude",
        "openai/gpt-6-sol",
    ]


def test_unreachable_catalog_falls_back_to_given_names_and_backs_off(monkeypatch):
    calls = []

    def _failing_get(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse({"error": "forbidden"}, status=403)

    monkeypatch.setattr(llm_gateway, "_http_get", _failing_get)

    first = get_gateway_route(_config("gateway", "claude-opus-5-5"))
    second = get_gateway_route(_config("gateway", "gpt-6-sol"))

    assert (first.model_name, second.model_name) == ("claude-opus-5-5", "gpt-6-sol")
    assert first.context_window == 0
    assert len(calls) == 1


# --- Codex CLI -------------------------------------------------------------


def _codex_config_toml(executor):
    executor._ensure_codex_home_and_config()
    with open(os.path.join(executor._codex_home, "config.toml"), "rb") as handle:
        return tomllib.load(handle)


def test_codex_runs_its_cli_against_the_gateway(catalog_requests):
    config = _config("codex", "gpt-6-sol", api_key_type="gateway", llm_gateway_headers={"X-Team": "red"})

    executor = build_executor(
        config,
        system_prompt="system",
        max_turns=2,
        memory_max_messages=10,
        memory_reset_to_messages=5,
    )

    assert executor.__class__.__name__ == "CodexExecutor"
    assert executor._model_name == "corp/gpt-6-sol"
    parsed = _codex_config_toml(executor)
    assert parsed["model"] == "corp/gpt-6-sol"
    assert parsed["model_provider"] == "chack_gateway"
    assert parsed["model_context_window"] == 922_000
    provider = parsed["model_providers"]["chack_gateway"]
    assert provider["base_url"] == GATEWAY_URL
    assert provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is False
    header_env = provider["env_http_headers"]["X-Team"]

    env = executor._build_env()
    assert env[provider["env_key"]] == GATEWAY_KEY
    assert env[header_env] == "red"
    assert GATEWAY_KEY not in open(os.path.join(executor._codex_home, "config.toml")).read()
    assert not os.path.exists(os.path.join(executor._codex_home, "auth.json"))


def test_codex_on_a_gateway_is_not_offered_openai_hosted_web_search(catalog_requests):
    executor = _build(build_codex_executor, _config("codex", "gpt-6-sol", api_key_type="gateway"))
    executor._ensure_codex_home_and_config()

    command = executor._build_command()

    assert 'web_search="disabled"' in command
    assert "cli_daybreak" not in " ".join(command)


def test_codex_gateway_route_ignores_a_chatgpt_token(monkeypatch, catalog_requests):
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "chatgpt-token")
    executor = _build(build_codex_executor, _config("codex", "gpt-6-sol", api_key_type="gateway"))

    env = executor._build_env()

    assert not executor._use_codex_access_token
    assert "CODEX_ACCESS_TOKEN" not in env


def test_codex_tool_server_inherits_the_gateway(catalog_requests):
    config = _config("codex", "gpt-6-sol", api_key_type="gateway")
    config.tools.exec_enabled = True
    executor = _build(build_codex_executor, config)

    executor._ensure_codex_home_and_config()
    executor._write_codex_explicit_mcp_env(executor._build_env())
    with open(os.path.join(executor._codex_home, "config.toml"), "rb") as handle:
        parsed = tomllib.load(handle)

    tool_env = parsed["mcp_servers"]["chack_tools"]["env"]
    assert tool_env["LLM_GATEWAY_API_KEY"] == GATEWAY_KEY
    assert tool_env["LLM_GATEWAY_BASE_URL"] == GATEWAY_URL


# --- Claude Code CLI -------------------------------------------------------


def test_claude_runs_its_cli_against_the_gateway(monkeypatch, catalog_requests):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-direct")
    config = _config("claude", "claude-opus-5-5", api_key_type="gateway", llm_gateway_headers={"X-Team": "red"})

    executor = _build(build_claude_executor, config)
    env = executor._build_env()

    assert executor.__class__.__name__ == "ClaudeCodeExecutor"
    assert executor._model_name == "corp/opus-5.5"
    assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example.test"
    assert env["ANTHROPIC_AUTH_TOKEN"] == GATEWAY_KEY
    assert env["ANTHROPIC_API_KEY"] == ""
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Team: red"
    # Subagents and background calls must use models the gateway serves.
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "corp/opus-5.5"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "corp/opus-5.5"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "corp/claude-haiku-5.5"
    assert "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS" not in env


def test_claude_turns_off_betas_only_for_bedrock_served_models(catalog_requests):
    config = _config("claude", "claude-sonnet-5-5", api_key_type="gateway")

    env = _build(build_claude_executor, config)._build_env()

    assert env["ANTHROPIC_MODEL"] == "corp/claude-sonnet-5.5"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"


def test_claude_background_model_falls_back_to_the_selected_model(catalog_requests):
    config = _config("claude", "my-claude", gateway_type="custom", api_key_type="gateway")

    env = _build(build_claude_executor, config)._build_env()

    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "my-claude"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "my-claude"


def test_claude_on_a_gateway_cannot_call_anthropic_hosted_web_search(catalog_requests):
    config = _config("claude", "claude-opus-5-5", api_key_type="gateway")
    config.tools.exec_enabled = True
    executor = _build(build_claude_executor, config)

    command = executor._build_command("prompt")

    builtin_tools = command[command.index("--tools") + 1].split(",")
    assert "WebSearch" not in builtin_tools
    assert "WebFetch" in builtin_tools
    assert "--betas" not in command


def test_claude_compacts_within_the_gateway_context_window(catalog_requests):
    config = _config("claude", "corp/claude-haiku-5.5", api_key_type="gateway")
    config.model.max_context_tokens = 400_000
    llm_gateway._CATALOG_CACHE.clear()
    # The haiku entry reports no window, so the configured budget is kept.
    assert _build(build_claude_executor, config)._build_env()["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "400000"

    config.model.primary = "claude-opus-5-5"
    config.model.max_context_tokens = 2_000_000
    env = _build(build_claude_executor, config)._build_env()

    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "1000000"


# --- Responses API backend -------------------------------------------------


def test_gateway_provider_uses_the_responses_backend(catalog_requests):
    config = _config("gateway", "claude-opus-5-5", llm_gateway_headers={"X-Team": "red"})

    executor = build_executor(
        config,
        system_prompt="system",
        max_turns=2,
        memory_max_messages=10,
        memory_reset_to_messages=5,
    )

    assert executor.__class__.__module__ == "chack_agent.backends.gateway_responses_backend"
    model = executor.agent.model
    assert model.model == "corp/opus-5.5"
    assert str(model._client.base_url).rstrip("/") == GATEWAY_URL
    assert model._client.api_key == GATEWAY_KEY
    assert model._client.default_headers["X-Team"] == "red"


def test_cli_without_gateway_support_moves_to_the_responses_backend(catalog_requests):
    executor = _build(build_gemini_executor, _config("gemini", "gateway/claude-opus-5-5"))

    assert executor.__class__.__module__ == "chack_agent.backends.gateway_responses_backend"
    assert executor._config.model.provider == "gateway"
    assert executor.agent.model.model == "corp/opus-5.5"


# --- cost ------------------------------------------------------------------


def test_litellm_prices_cost_gateway_model_names(monkeypatch, catalog_requests):
    monkeypatch.setenv("CHACK_PRICING", _package_pricing_path())
    chack = Chack(_config("codex", "gpt-6-sol", api_key_type="gateway"))

    # 22 input + 8 output tokens on corp/opus-5.5: 22*$4/M + 8*$20/M.
    opus = chack._estimate_model_cost(chack._pricing, "corp/opus-5.5", 22, 8)
    cached = chack._estimate_model_cost(chack._pricing, "corp/gpt-6-sol", 1_000_000, 0, cached_prompt_tokens=1_000_000)

    configured = chack._estimate_model_cost(chack._pricing, "gpt-6-sol", 1_000_000, 0)

    assert opus == pytest.approx(0.000248)
    assert configured == pytest.approx(2.2)
    assert cached == pytest.approx(0.22)
    assert "corp/auto" not in chack._pricing.models


class _FakeRunResult:
    def __init__(self, transcript, response_id):
        self.final_output = "done"
        self.new_items = []
        self.raw_responses = []
        self.last_response_id = response_id
        self._transcript = transcript

    def to_input_list(self):
        return list(self._transcript)


def _record_runs(monkeypatch, backend):
    runs = []

    def _run_sync(agent, input_items, **kwargs):
        runs.append({"input": list(input_items), "previous_response_id": kwargs.get("previous_response_id")})
        transcript = list(input_items) + [
            {"type": "function_call", "call_id": f"call_{len(runs)}", "name": "exec", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"call_{len(runs)}", "output": "42"},
            {"role": "assistant", "content": "done"},
        ]
        return _FakeRunResult(transcript, f"resp_{len(runs)}")

    monkeypatch.setattr(backend.Runner, "run_sync", _run_sync)
    return runs


@pytest.mark.parametrize(
    ("primary", "provider", "chained"),
    [
        # Gateways that translate the Responses API rebuild chained turns lossily.
        ("claude-opus-5-5", "gateway", False),
        ("openrouter/openai/gpt-6-sol", "openrouter", True),
    ],
)
def test_follow_up_runs_resend_history_unless_the_provider_stores_responses(
    monkeypatch, catalog_requests, primary, provider, chained
):
    from chack_agent.backends import gateway_responses_backend as backend

    config = _config(provider, primary, openrouter_api_key="or-key")
    executor = _build(backend.build_executor, config)
    runs = _record_runs(monkeypatch, backend)

    executor.invoke({"input": "first"})
    executor.invoke({"input": "second"})

    second = runs[1]
    if chained:
        assert second["previous_response_id"] == "resp_1"
        assert second["input"] == [{"role": "user", "content": "second"}]
    else:
        assert second["previous_response_id"] is None
        assert [item.get("content") for item in second["input"] if item.get("role") == "user"] == ["first", "second"]
        assert {"type": "function_call_output", "call_id": "call_1", "output": "42"} in second["input"]


def test_chained_requests_keep_outputs_for_calls_stored_with_the_previous_response():
    from chack_agent.backends.gateway_responses_backend import GatewayResponsesModel

    outputs = [{"type": "function_call_output", "call_id": "call_9", "output": "42"}]

    chained = GatewayResponsesModel._prepare_input_items(outputs, "resp_8")
    unchained = GatewayResponsesModel._prepare_input_items(outputs, None)

    assert chained == outputs
    assert all(item.get("type") != "function_call_output" for item in unchained)
