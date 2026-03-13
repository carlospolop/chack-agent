import json
from pathlib import Path

from chack_agent.backends.playwright_mcp import playwright_mcp_server_config
from chack_agent.backends.playwright_mcp import playwright_mcp_result_to_text
from chack_agent.backends.playwright_mcp import playwright_mcp_server_instance
from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor
from chack_agent.backends.langgraph_backend import _load_playwright_mcp_tools
from chack_agent.backends.openai_compaction_backend import _build_mcp_servers as build_openai_mcp_servers
from chack_agent.backends.openrouter_openai_backend import _build_mcp_servers as build_openrouter_mcp_servers
from chack_agent.config import AgentConfig, ChackConfig, CredentialsConfig, LoggingConfig, ModelConfig, SessionConfig, ToolsConfig


def _build_claude_executor(tmp_path: Path, tools_config_json: str) -> ClaudeCodeExecutor:
    return ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=10,
        _memory_reset_to=5,
        _base_system_prompt="system",
        _model_name="test-model",
        _max_turns=5,
        _claude_cli_path="/bin/echo",
        _tools_config_json=tools_config_json,
        _allowed_tools_json="[]",
        _serialized_tools_override_b64="",
        _serialized_tools_append_b64="",
        _model_provider="claude",
        _default_model="",
        _social_network_model="",
        _scientific_model="",
        _websearcher_model="",
        _tester_model="",
        _subchack_model="",
        _social_network_max_turns=5,
        _scientific_max_turns=5,
        _websearcher_max_turns=5,
        _tester_max_turns=5,
        _subchack_max_turns=5,
        _min_tools_used=0,
        _max_tools_used=0,
        _require_task_steps_manager_init_first=True,
        _output_schema_json="",
        _claude_home=str(tmp_path),
    )


def test_playwright_mcp_server_config_shape(monkeypatch):
    monkeypatch.setattr(
        "chack_agent.backends.playwright_mcp.playwright_mcp_browser_executable_path",
        lambda: None,
    )
    monkeypatch.setattr(
        "chack_agent.backends.playwright_mcp.playwright_mcp_needs_no_sandbox",
        lambda: False,
    )
    assert playwright_mcp_server_config() == {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
    }


def _build_config(*, provider: str, playwright_enabled: bool) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary="gpt-5", provider=provider),
        agent=AgentConfig(),
        session=SessionConfig(),
        tools=ToolsConfig(playwright_enabled=playwright_enabled),
        credentials=CredentialsConfig(),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )


def test_playwright_mcp_server_instance_name():
    server = playwright_mcp_server_instance()
    assert server.name == "playwright"


def test_playwright_mcp_server_config_uses_browser_executable(monkeypatch):
    monkeypatch.setattr(
        "chack_agent.backends.playwright_mcp.playwright_mcp_browser_executable_path",
        lambda: "/tmp/playwright-chromium",
    )
    monkeypatch.setattr(
        "chack_agent.backends.playwright_mcp.playwright_mcp_needs_no_sandbox",
        lambda: False,
    )
    assert playwright_mcp_server_config() == {
        "command": "npx",
        "args": [
            "-y",
            "@playwright/mcp@latest",
            "--executable-path",
            "/tmp/playwright-chromium",
        ],
    }


def test_playwright_mcp_server_config_adds_no_sandbox(monkeypatch):
    monkeypatch.setattr(
        "chack_agent.backends.playwright_mcp.playwright_mcp_browser_executable_path",
        lambda: "/tmp/playwright-chromium",
    )
    monkeypatch.setattr(
        "chack_agent.backends.playwright_mcp.playwright_mcp_needs_no_sandbox",
        lambda: True,
    )
    assert playwright_mcp_server_config()["args"][-1] == "--no-sandbox"


def test_playwright_mcp_result_to_text_joins_text_blocks():
    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Result:
        content = [_TextBlock("first"), _TextBlock("second")]
        structuredContent = None

    assert playwright_mcp_result_to_text(_Result()) == "first\n\nsecond"


def test_claude_settings_include_playwright_mcp_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend.playwright_mcp_is_available",
        lambda: True,
    )
    executor = _build_claude_executor(
        tmp_path,
        json.dumps({"playwright_enabled": True}),
    )

    executor._write_claude_settings(str(tmp_path))
    payload = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))

    assert payload["mcpServers"]["playwright"] == playwright_mcp_server_config()


def test_claude_settings_skip_playwright_mcp_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend.playwright_mcp_is_available",
        lambda: True,
    )
    executor = _build_claude_executor(
        tmp_path,
        json.dumps({"playwright_enabled": False}),
    )

    executor._write_claude_settings(str(tmp_path))
    payload = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))

    assert "playwright" not in payload["mcpServers"]


def test_openai_backend_includes_playwright_mcp_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "chack_agent.backends.openai_compaction_backend.playwright_mcp_is_available",
        lambda: True,
    )
    servers = build_openai_mcp_servers(_build_config(provider="openai", playwright_enabled=True))
    assert len(servers) == 1
    assert getattr(servers[0], "name", "") == "playwright"


def test_openrouter_backend_includes_playwright_mcp_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "chack_agent.backends.openrouter_openai_backend.playwright_mcp_is_available",
        lambda: True,
    )
    servers = build_openrouter_mcp_servers(_build_config(provider="openrouter", playwright_enabled=True))
    assert len(servers) == 1
    assert getattr(servers[0], "name", "") == "playwright"


def test_langgraph_loads_playwright_mcp_tools_when_enabled(monkeypatch):
    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name
            self.description = f"{name} description"
            self.inputSchema = {"type": "object", "properties": {}}

    monkeypatch.setattr(
        "chack_agent.backends.langgraph_backend.playwright_mcp_is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "chack_agent.backends.langgraph_backend.playwright_mcp_list_tools",
        lambda: [_Tool("browser_navigate")],
    )

    tools = _load_playwright_mcp_tools(_build_config(provider="langgraph", playwright_enabled=True))
    assert [getattr(tool, "name", "") for tool in tools] == ["browser_navigate"]
