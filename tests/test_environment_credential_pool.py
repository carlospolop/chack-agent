from __future__ import annotations

import json

import pytest

from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor
from chack_agent.backends.codex_backend import CodexExecutor, _RawResult as CodexRawResult
from chack_agent.environment_credential_pool import (
    reset_environment_credential_pool_for_tests,
    rotate_environment_credentials,
)
from chack_agent.provider_launch_hooks import (
    run_provider_pre_launch_hook,
    set_provider_pre_launch_hook,
)


@pytest.fixture(autouse=True)
def _reset_pool(monkeypatch: pytest.MonkeyPatch):
    set_provider_pre_launch_hook(None)
    reset_environment_credential_pool_for_tests()
    for name in (
        "CODEX_TOKEN_POOL_JSON",
        "CODEX_ACCESS_TOKEN",
        "CHACK_CODEX_ACCESS_TOKEN",
        "CLAUDE_TOKEN_POOL_JSON",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_ACCESS_TOKEN",
        "CHACK_PROVIDER_POOL_RETRY_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    set_provider_pre_launch_hook(None)
    reset_environment_credential_pool_for_tests()


def test_default_hook_uses_and_rotates_codex_environment_pool(monkeypatch):
    monkeypatch.setenv(
        "CODEX_TOKEN_POOL_JSON",
        json.dumps([{"access_token": "codex-a"}, {"access_token": "codex-b"}]),
    )
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-a")

    assert run_provider_pre_launch_hook("codex") == {"access_token": "codex-a"}
    assert rotate_environment_credentials("codex", "codex-a") == {
        "access_token": "codex-b"
    }
    assert run_provider_pre_launch_hook("codex") == {"access_token": "codex-b"}
    assert rotate_environment_credentials("codex", "codex-a") == {
        "access_token": "codex-b"
    }


def test_exhausted_pool_does_not_immediately_cycle(monkeypatch):
    monkeypatch.setenv(
        "CLAUDE_TOKEN_POOL_JSON",
        json.dumps([{"token": "claude-a"}, {"token": "claude-b"}]),
    )
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-a")

    assert rotate_environment_credentials("claude", "claude-a") == {
        "access_token": "claude-b"
    }
    assert rotate_environment_credentials("claude", "claude-b") == {}


def test_codex_auth_failure_retries_with_next_pool_token(monkeypatch):
    monkeypatch.setenv(
        "CODEX_TOKEN_POOL_JSON",
        json.dumps([{"access_token": "codex-a"}, {"access_token": "codex-b"}]),
    )
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-a")
    executor = object.__new__(CodexExecutor)
    executor._use_codex_access_token = True
    executor._codex_access_token = "codex-a"
    executor._fallback_openai_api_key = ""
    executor._openai_api_key = "codex-a"
    executor._thread_id = "old-thread"
    executor._use_existing_codex_auth_file = False
    executor._existing_codex_auth_file = ""
    executor._codex_home = None
    calls = []
    executor._run_codex_once = lambda prompt, allow_api_key_fallback: (
        calls.append((prompt, allow_api_key_fallback))
        or ("ok", [], CodexRawResult(raw_responses=[]))
    )

    result = executor._maybe_retry_with_api_key(
        "prompt",
        ("ERROR: quota exceeded", [], CodexRawResult(raw_responses=[])),
        True,
        codex_exec_failed=True,
    )

    assert result[0] == "ok"
    assert executor._codex_access_token == "codex-b"
    assert executor._thread_id is None
    assert calls == [("prompt", True)]


def test_claude_auth_failure_retries_with_next_pool_token(monkeypatch):
    monkeypatch.setenv(
        "CLAUDE_TOKEN_POOL_JSON",
        json.dumps([{"token": "claude-a"}, {"token": "claude-b"}]),
    )
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-a")
    executor = object.__new__(ClaudeCodeExecutor)
    executor._claude_access_token = "claude-a"
    executor._claude_session_id = "old-session"
    executor._uses_openrouter_route = False
    executor._anthropic_api_key = ""
    executor._refresh_provider_credentials = lambda: None
    calls = []
    executor._run_claude_once = lambda prompt: (
        calls.append(prompt) or ("ok", [], object())
    )

    result = executor._run_claude("prompt")

    assert result[0] == "ok"
    assert executor._claude_access_token == "claude-a"
    assert calls == ["prompt"]

    calls.clear()
    executor._run_claude_once = lambda prompt: (
        calls.append(prompt)
        or (
            "ERROR: session limit reached" if len(calls) == 1 else "ok",
            [],
            object(),
        )
    )
    result = executor._run_claude("prompt")
    assert result[0] == "ok"
    assert executor._claude_access_token == "claude-b"
    assert executor._claude_session_id is None
    assert calls == ["prompt", "prompt"]
