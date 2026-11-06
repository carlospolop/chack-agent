"""Codex and Claude Code CLIs on an OpenRouter key, gateway env loading, and launch-hook isolation."""

from __future__ import annotations

import os
import textwrap
import tomllib

import pytest

from chack_agent import llm_gateway
from chack_agent.backends import build_executor, claude_code_backend, codex_backend
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
    load_config,
)
from chack_agent.llm_gateway import openrouter_model_id
from chack_agent.provider_launch_hooks import set_provider_pre_launch_hook

OPENROUTER_KEY = "or-test-key"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("LLM_GATEWAY_", "OPENROUTER_", "CLAUDE_", "ANTHROPIC_", "CODEX_", "OPENAI_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHACK_CODEX_HOME_BASE", str(tmp_path / "codex"))
    monkeypatch.setenv("CHACK_CLAUDE_HOME_BASE", str(tmp_path / "claude"))
    llm_gateway._CATALOG_CACHE.clear()

    def _offline(url, headers=None, timeout=None):
        raise ConnectionError(f"network disabled in tests: {url}")

    monkeypatch.setattr(llm_gateway, "_http_get", _offline)
    set_provider_pre_launch_hook(None)
    yield
    set_provider_pre_launch_hook(None)
    llm_gateway._CATALOG_CACHE.clear()


def _config(provider, primary, api_key_type, **credentials):
    return ChackConfig(
        model=ModelConfig(primary=primary, provider=provider, api_key_type=api_key_type),
        agent=AgentConfig(main_action="test", sub_action="openrouter-cli"),
        session=SessionConfig(max_turns=2),
        tools=ToolsConfig(),
        credentials=CredentialsConfig(**credentials),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )


def _build(config):
    return build_executor(
        config, system_prompt="system", max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("gpt-6-sol", "openai/gpt-6-sol"),
        ("claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"),
        ("claude-haiku-5-5", "anthropic/claude-haiku-5.5"),
        ("openrouter/google/gemini-2.5-flash", "google/gemini-2.5-flash"),
        ("anthropic/claude-opus-5.5", "anthropic/claude-opus-5.5"),
    ],
)
def test_vendor_model_names_map_to_openrouter_ids(requested, expected):
    assert openrouter_model_id(requested) == expected


def test_codex_runs_its_cli_against_openrouter(monkeypatch):
    executor = _build(_config("codex", "gpt-6-sol", "openrouter", openrouter_api_key=OPENROUTER_KEY))

    assert executor.__class__.__name__ == "CodexExecutor"
    executor._ensure_codex_home_and_config()
    config_path = os.path.join(executor._codex_home, "config.toml")
    with open(config_path, "rb") as handle:
        parsed = tomllib.load(handle)
    provider = parsed["model_providers"][parsed["model_provider"]]
    env = executor._build_env()

    assert parsed["model"] == "openai/gpt-6-sol"
    assert provider["base_url"] == "https://openrouter.ai/api/v1"
    assert provider["wire_api"] == "responses"
    assert env[provider["env_key"]] == OPENROUTER_KEY
    assert OPENROUTER_KEY not in open(config_path, encoding="utf-8").read()


def test_claude_runs_its_cli_against_openrouter(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")
    executor = _build(_config("claude", "claude-sonnet-5-5", "openrouter", openrouter_api_key=OPENROUTER_KEY))

    env = executor._build_env()

    assert executor.__class__.__name__ == "ClaudeCodeExecutor"
    assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert env["ANTHROPIC_AUTH_TOKEN"] == OPENROUTER_KEY
    assert env["ANTHROPIC_API_KEY"] == ""
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert env["ANTHROPIC_MODEL"] == "anthropic/claude-sonnet-5.5"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"].startswith("anthropic/")


@pytest.mark.parametrize("api_key_type", ["openrouter", "gateway"])
def test_launch_hook_login_does_not_replace_the_routed_key_for_codex(monkeypatch, api_key_type):
    set_provider_pre_launch_hook(lambda provider: {"access_token": f"{provider}-login"})
    config = _config(
        "codex",
        "gpt-6-sol",
        api_key_type,
        openrouter_api_key=OPENROUTER_KEY,
        llm_gateway_type="custom",
        llm_gateway_base_url="https://gateway.example.test/v1",
        llm_gateway_api_key="gw-key",
    )
    executor = _build(config)

    executor._refresh_provider_credentials()
    env = executor._build_env()

    assert not executor._use_codex_access_token
    assert "codex-login" not in env.values()


@pytest.mark.parametrize("api_key_type", ["openrouter", "gateway"])
def test_launch_hook_login_does_not_replace_the_routed_key_for_claude(monkeypatch, api_key_type):
    set_provider_pre_launch_hook(lambda provider: {"access_token": f"{provider}-login"})
    config = _config(
        "claude",
        "claude-sonnet-5-5",
        api_key_type,
        openrouter_api_key=OPENROUTER_KEY,
        llm_gateway_type="custom",
        llm_gateway_base_url="https://gateway.example.test/v1",
        llm_gateway_api_key="gw-key",
    )
    executor = _build(config)
    executor._claude_session_id = "routed-session"

    executor._refresh_provider_credentials()
    env = executor._build_env()

    assert executor._claude_access_token != "claude-login"
    # A login change would discard the session; the routed key never changed.
    assert executor._claude_session_id == "routed-session"
    assert "claude-login" not in env.values()
    assert env["ANTHROPIC_AUTH_TOKEN"] in {OPENROUTER_KEY, "gw-key"}


def _write_yaml(tmp_path, body):
    path = tmp_path / "agent.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def test_load_config_fills_gateway_credentials_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", " gw-env-key ")
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "https://gateway.example.test/v1")
    monkeypatch.setenv("LLM_GATEWAY_TYPE", "litellm")
    monkeypatch.setenv("LLM_GATEWAY_HEADERS", '{"X-Team": "red"}')
    path = _write_yaml(
        tmp_path,
        """
        system_prompt: test
        agent:
          primary: gpt-6-sol
          provider: codex
          api_key_type: gateway
          main_action: test
          sub_action: run
        credentials:
          llm_gateway_type: custom
        """,
    )

    config = load_config(path)

    assert config.credentials.llm_gateway_api_key == "gw-env-key"
    assert config.credentials.llm_gateway_base_url == "https://gateway.example.test/v1"
    # A value written in the file wins over the environment.
    assert config.credentials.llm_gateway_type == "custom"
    assert config.credentials.llm_gateway_headers == {"X-Team": "red"}


def test_load_config_rejects_malformed_gateway_headers(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_GATEWAY_HEADERS", "not json")
    path = _write_yaml(
        tmp_path,
        """
        system_prompt: test
        agent:
          primary: gpt-6-sol
          provider: codex
          main_action: test
          sub_action: run
        """,
    )

    with pytest.raises(ValueError, match="LLM_GATEWAY_HEADERS"):
        load_config(path)
