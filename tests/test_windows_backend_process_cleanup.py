from __future__ import annotations

from types import SimpleNamespace

from chack_agent.backends import claude_code_backend, codex_backend


def test_codex_windows_cleanup_uses_taskkill(monkeypatch):
    calls = []
    process = SimpleNamespace(pid=123, stdout=None, kill=lambda: calls.append("fallback"))
    monkeypatch.setattr(codex_backend.os, "name", "nt")
    monkeypatch.setattr(
        codex_backend.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    codex_backend._terminate_process_tree(process)

    assert calls[0][0] == ["taskkill", "/PID", "123", "/T", "/F"]
    assert "fallback" not in calls


def test_claude_windows_cleanup_uses_taskkill(monkeypatch):
    calls = []
    process = SimpleNamespace(pid=456, kill=lambda: calls.append("fallback"))
    monkeypatch.setattr(claude_code_backend.os, "name", "nt")
    monkeypatch.setattr(
        claude_code_backend.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    claude_code_backend._terminate_process_tree(process)

    assert calls[0][0] == ["taskkill", "/PID", "456", "/T", "/F"]
    assert "fallback" not in calls
