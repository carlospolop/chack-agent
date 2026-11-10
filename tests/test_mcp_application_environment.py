"""Custom MCP tools retain explicitly selected application resource settings."""
import json
import subprocess
import sys

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from chack_agent.backends import codex_backend
from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor
from chack_agent.backends.gemini_cli_backend import GeminiCliExecutor
from chack_agent.backends.copilot_cli_backend import CopilotCliExecutor
from chack_agent.backends.tool_payloads import configured_mcp_env_names
from chack_agent.config import (
    AgentConfig, ChackConfig, CredentialsConfig, LoggingConfig,
    ModelConfig, SessionConfig, ToolsConfig,
)


def _child_settings(env):
    return json.loads(subprocess.check_output(
        [sys.executable, '-c', 'import os,json;print(json.dumps({k:os.environ.get(k) for k in ["APP_GRAPH_BUCKET","APP_TABLE","UNSELECTED","UNSET"]}))'],
        env=env, text=True,
    ))


@pytest.mark.parametrize('backend,method', [
    (ClaudeCodeExecutor, '_mcp_env_map'),
    (GeminiCliExecutor, '_gemini_mcp_env_map'),
    (CopilotCliExecutor, '_copilot_mcp_env_map'),
])
def test_mcp_child_receives_selected_current_resource_settings(backend, method):
    executor = object.__new__(backend)
    executor._tools_config_json = json.dumps({'mcp_env_vars': ['APP_GRAPH_BUCKET', 'APP_TABLE', 'UNSET']})
    executor._gemini_api_key = ''
    source = {'APP_GRAPH_BUCKET': 'bucket-a', 'APP_TABLE': 'table-a', 'UNSELECTED': 'private'}
    executor._build_env = lambda: source
    assert _child_settings(getattr(executor, method)()) == {
        'APP_GRAPH_BUCKET': 'bucket-a', 'APP_TABLE': 'table-a', 'UNSET': None, 'UNSELECTED': None,
    }
    source['APP_TABLE'] = 'table-b'
    assert _child_settings(getattr(executor, method)())['APP_TABLE'] == 'table-b'


def test_codex_explicit_mcp_config_transports_selected_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('CHACK_CODEX_HOME_BASE', str(tmp_path / 'homes'))
    # CLI discovery is the outer boundary. No model or network is used.
    monkeypatch.setattr(codex_backend, '_codex_single_agent_catalog', lambda *a, **k: None)
    config = ChackConfig(
        model=ModelConfig(primary='gpt-6-sol', provider='codex'),
        agent=AgentConfig(), session=SessionConfig(),
        tools=ToolsConfig(mcp_env_vars=['APP_GRAPH_BUCKET','APP_TABLE','UNSET']),
        credentials=CredentialsConfig(openai_api_key='test'), logging=LoggingConfig(),
        system_prompt='probe', env={},
    )
    executor = codex_backend.build_executor(config, system_prompt='probe', max_turns=2,
        memory_max_messages=10, memory_reset_to_messages=5)
    # Ensure a custom-tool server is rendered without invoking a CLI.
    monkeypatch.setattr(executor, '_has_configured_tools', lambda: True)
    executor._ensure_codex_home_and_config()
    source = {'APP_GRAPH_BUCKET':'bucket-a','APP_TABLE':'table-a','UNSELECTED':'private'}
    executor._write_codex_explicit_mcp_env(source)
    from pathlib import Path
    path = Path(executor._codex_home) / 'config.toml'
    env = tomllib.loads(path.read_text())['mcp_servers']['chack_tools']['env']
    assert _child_settings(env) == {
        'APP_GRAPH_BUCKET':'bucket-a','APP_TABLE':'table-a','UNSET':None,'UNSELECTED':None,
    }
    source['APP_TABLE']='table-b'
    executor._write_codex_explicit_mcp_env(source)
    env = tomllib.loads(path.read_text())['mcp_servers']['chack_tools']['env']
    assert _child_settings(env)['APP_TABLE']=='table-b'
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('names', [['BAD=NAME'], ['*'], 'APP_TABLE', ['X'] * 65])
def test_invalid_environment_selection_is_rejected(names):
    with pytest.raises(ValueError):
        configured_mcp_env_names(json.dumps({'mcp_env_vars':names}))
