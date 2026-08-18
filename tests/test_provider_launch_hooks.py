from __future__ import annotations

import pytest

from chack_agent.backends import claude_code_backend, codex_backend
from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor
from chack_agent.backends.codex_backend import CodexExecutor
from chack_agent.provider_launch_hooks import (
    run_provider_pre_launch_hook,
    set_provider_pre_launch_hook,
)


@pytest.fixture(autouse=True)
def _reset_provider_hook():
    set_provider_pre_launch_hook(None)
    yield
    set_provider_pre_launch_hook(None)


def test_hook_dispatches_only_supported_provider_names():
    calls = []
    set_provider_pre_launch_hook(
        lambda provider: calls.append(provider) or {"access_token": f"{provider}-token"}
    )

    assert run_provider_pre_launch_hook("codex") == {"access_token": "codex-token"}
    assert run_provider_pre_launch_hook("claude") == {"access_token": "claude-token"}
    assert run_provider_pre_launch_hook("openai") == {}
    assert calls == ["codex", "claude"]


def test_codex_executor_applies_rotated_token_and_discards_account_thread(monkeypatch):
    executor = object.__new__(CodexExecutor)
    executor._codex_access_token = "old-codex"
    executor._openai_api_key = "old-codex"
    executor._use_codex_access_token = True
    executor._use_existing_codex_auth_file = False
    executor._existing_codex_auth_file = ""
    executor._thread_id = "old-account-thread"
    executor._codex_home = None
    monkeypatch.setattr(
        codex_backend,
        "run_provider_pre_launch_hook",
        lambda provider: {"access_token": "new-codex", "provider": provider},
    )

    executor._refresh_provider_credentials()

    assert executor._codex_access_token == "new-codex"
    assert executor._openai_api_key == "new-codex"
    assert executor._thread_id is None


def test_claude_executor_applies_only_claude_hook_and_starts_fresh_session(monkeypatch):
    executor = object.__new__(ClaudeCodeExecutor)
    executor._claude_access_token = "old-claude"
    executor._claude_session_id = "old-account-session"
    requested = []
    monkeypatch.setattr(
        claude_code_backend,
        "run_provider_pre_launch_hook",
        lambda provider: requested.append(provider) or {"access_token": "new-claude"},
    )

    executor._refresh_provider_credentials()

    assert requested == ["claude"]
    assert executor._claude_access_token == "new-claude"
    assert executor._claude_session_id is None


def test_codex_executor_discards_an_invalidated_oauth_token(monkeypatch):
    executor = object.__new__(CodexExecutor)
    executor._codex_access_token = "expired-codex"
    executor._openai_api_key = "fallback-openai-key"
    executor._fallback_openai_api_key = "fallback-openai-key"
    executor._use_codex_access_token = True
    executor._use_existing_codex_auth_file = False
    executor._existing_codex_auth_file = ""
    executor._thread_id = "expired-account-thread"
    executor._codex_home = None
    monkeypatch.setattr(
        codex_backend,
        "run_provider_pre_launch_hook",
        lambda provider: {"clear_access_token": "true"},
    )

    executor._refresh_provider_credentials()

    assert executor._codex_access_token == ""
    assert executor._openai_api_key == "fallback-openai-key"
    assert executor._use_codex_access_token is False
    assert executor._thread_id is None


def test_claude_executor_discards_an_invalidated_oauth_token(monkeypatch):
    executor = object.__new__(ClaudeCodeExecutor)
    executor._claude_access_token = "invalid-claude"
    executor._claude_session_id = "invalid-account-session"
    monkeypatch.setattr(
        claude_code_backend,
        "run_provider_pre_launch_hook",
        lambda provider: {"clear_access_token": "true"},
    )

    executor._refresh_provider_credentials()

    assert executor._claude_access_token == ""
    assert executor._claude_session_id is None
