from __future__ import annotations

import pytest

from chack_tools.config import ToolsConfig
from chack_tools.exec_tool import ExecTool


def test_exec_tool_rejects_runtime_denied_commands(monkeypatch):
    monkeypatch.setenv(
        "CHACK_EXEC_DENY_REGEX",
        r"(?:^|[;&|]\s*|\s)(?:\S*/)?adb(?:\s+[^;&|]*)?\s(?:emu\s+kill|(?:shell\s+)?reboot|kill-server)(?:\s|$)",
    )

    with pytest.raises(PermissionError, match="runtime execution policy"):
        ExecTool(ToolsConfig(exec_enabled=True)).run(
            "adb -s emulator-5556 shell reboot -p"
        )


def test_exec_tool_allows_commands_outside_runtime_policy(monkeypatch):
    monkeypatch.setenv("CHACK_EXEC_DENY_REGEX", r"adb.*reboot")

    output = ExecTool(ToolsConfig(exec_enabled=True)).run("printf safe")

    assert output == "safe"


def test_exec_tool_rejects_invalid_runtime_policy(monkeypatch):
    monkeypatch.setenv("CHACK_EXEC_DENY_REGEX", "[")

    with pytest.raises(ValueError, match="Invalid CHACK_EXEC_DENY_REGEX"):
        ExecTool(ToolsConfig(exec_enabled=True)).run("printf safe")
