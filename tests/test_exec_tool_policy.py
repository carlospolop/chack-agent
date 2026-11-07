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


def _mcp_server_env(monkeypatch, caller_env: dict[str, str]) -> dict[str, str]:
    """The environment a Chack MCP server runs with for this caller."""
    from chack_agent.backends.tool_payloads import augment_subprocess_pythonpath

    env = dict(caller_env)
    augment_subprocess_pythonpath(env)
    for name in ("PYTHONPATH", "CHACK_CALLER_PYTHONPATH_JSON"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


@pytest.mark.parametrize("caller_pythonpath", ["/opt/project/lib", None])
def test_exec_commands_get_the_callers_pythonpath_not_the_runtime_import_roots(
    monkeypatch, tmp_path, caller_pythonpath
):
    (tmp_path / "project-lib").mkdir()
    caller = {} if caller_pythonpath is None else {"PYTHONPATH": caller_pythonpath}
    server_env = _mcp_server_env(monkeypatch, caller)
    # The server itself imports from the runtime's roots...
    assert server_env["PYTHONPATH"] != (caller_pythonpath or "")

    output = ExecTool(ToolsConfig(exec_enabled=True)).run(
        'printf "%s|%s" "${PYTHONPATH-unset}" "${CHACK_CALLER_PYTHONPATH_JSON-unset}"'
    )

    # ...but a shell command sees exactly what the caller had.
    assert output == f"{caller_pythonpath or 'unset'}|unset"


def test_a_python_launched_by_exec_does_not_import_the_runtime_packages(monkeypatch, tmp_path):
    shadow = tmp_path / "runtime-site"
    shadow.mkdir()
    (shadow / "chack_shadow_probe.py").write_text("VALUE = 'runtime'\n")
    monkeypatch.syspath_prepend(str(shadow))
    _mcp_server_env(monkeypatch, {})

    output = ExecTool(ToolsConfig(exec_enabled=True)).run(
        'python3 -c "import importlib.util as u; print(u.find_spec(\'chack_shadow_probe\') is None)"'
    )

    assert output == "True"


def test_exec_outside_a_chack_mcp_server_keeps_the_process_environment(monkeypatch):
    monkeypatch.delenv("CHACK_CALLER_PYTHONPATH_JSON", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/kept/as/is")

    output = ExecTool(ToolsConfig(exec_enabled=True)).run('printf "%s" "$PYTHONPATH"')

    assert output == "/kept/as/is"


def test_exec_commands_do_not_export_traces(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example")
    monkeypatch.setenv("TRACEPARENT", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
    monkeypatch.setenv("CHACK_EXEC_TEST_KEEP", "kept")

    output = ExecTool(ToolsConfig(exec_enabled=True)).run(
        'printf "%s|%s|%s" "${OTEL_EXPORTER_OTLP_ENDPOINT-unset}" "${TRACEPARENT-unset}" "$CHACK_EXEC_TEST_KEEP"'
    )

    assert output == "unset|unset|kept"
