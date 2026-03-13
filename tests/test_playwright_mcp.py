import json
from pathlib import Path

from chack_agent.backends.playwright_mcp import playwright_mcp_server_config
from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor


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


def test_playwright_mcp_server_config_shape():
    assert playwright_mcp_server_config() == {
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
    }


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
