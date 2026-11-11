"""Behavioral contracts for the Kimi Code and Qwen Code adapters.

The fake executable emits each CLI's documented JSONL shape. No model or
external service is involved, so this checks routing, isolation, parsing,
session resume, usage and the native-tool restrictions on every test run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from chack_agent.backends import build_executor
from chack_agent.backends.code_cli_backend import CodeCliExecutor
from chack_agent.config import (
    AgentConfig, ChackConfig, CredentialsConfig, LoggingConfig,
    ModelConfig, SessionConfig, ToolsConfig, resolve_backend_type,
    resolve_config_aliases,
)


@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cli = tmp_path / "fake-cli"
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['FAKE_CALLS'], 'a') as f:\n"
        " f.write(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(),"
        " 'allowed':json.loads(os.environ['CHACK_ALLOWED_TOOLS_JSON'])})+'\\n')\n"
        "def emit(value): print(json.dumps(value), flush=True)\n"
        "if os.environ.get('FAKE_FAIL'):\n"
        " print('diagnostic from stderr', file=sys.stderr, flush=True); sys.exit(2)\n"
        "if os.environ['FAKE_KIND']=='kimi':\n"
        " emit({'role':'assistant','content':'working','tool_calls':[{'id':'call-1','function':"
        "{'name':'mcp__chack_tools__exec','arguments':'{\"command\":\"printf OK\"}'}}]})\n"
        " emit({'role':'tool','tool_call_id':'call-1','content':'OK'})\n"
        " emit({'role':'assistant','content':os.environ.get('FAKE_CONTENT','OK')})\n"
        " emit({'role':'meta','type':'system.usage','usage':{'input_tokens':12,'output_tokens':2}})\n"
        " emit({'role':'meta','type':'session.resume_hint','session_id':'session_test'})\n"
        "else:\n"
        " emit({'type':'system','subtype':'init','session_id':'qwen-test'})\n"
        " emit({'type':'assistant','message':{'content':[{'type':'tool_use','id':'call-1',"
        "'name':'mcp__chack_tools__exec','input':{'command':'printf OK'}}]}})\n"
        " emit({'type':'user','message':{'content':[{'type':'tool_result',"
        "'tool_use_id':'call-1','content':'OK'}]}})\n"
        " emit({'type':'assistant','message':{'content':[{'type':'text','text':'intermediate'}]}})\n"
        " emit({'type':'result','subtype':'success','session_id':'qwen-test','result':os.environ.get('FAKE_CONTENT','OK'),"
        "'usage':{'input_tokens':15,'output_tokens':2,'cache_read_input_tokens':3}})\n"
        " if os.environ.get('FAKE_HANG_AFTER_RESULT'):\n"
        "  import time; time.sleep(30)\n"
    )
    cli.chmod(0o755)
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("CHACK_KIMI_HOME", str(tmp_path / "kimi-home"))
    monkeypatch.setenv("CHACK_QWEN_HOME", str(tmp_path / "qwen-home"))
    monkeypatch.setenv("KIMI_CLI_PATH", str(cli))
    monkeypatch.setenv("QWEN_CLI_PATH", str(cli))
    monkeypatch.setenv("QWEN_BASE_URL", "https://models.example.test/v1")
    monkeypatch.chdir(tmp_path)
    return cli


def config(backend: str, *, exec_enabled: bool = True) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary="test-model", provider=backend),
        agent=AgentConfig(require_task_steps_manager_init_first=False),
        session=SessionConfig(max_turns=4),
        tools=ToolsConfig(
            exec_enabled=exec_enabled, task_steps_manager_enabled=False,
            native_web_search_enabled=False,
        ),
        credentials=CredentialsConfig(kimi_api_key="test-kimi", qwen_api_key="test-qwen"),
        logging=LoggingConfig(), system_prompt="Be concise.", env={},
    )


@pytest.mark.parametrize("backend,alias", [
    ("kimi", "kimi-code"), ("qwen", "qwen_code"),
])
def test_routing_tool_events_and_resume(fake_cli: Path, monkeypatch: pytest.MonkeyPatch, backend: str, alias: str) -> None:
    monkeypatch.setenv("FAKE_KIND", backend)
    cfg = config(alias)
    assert resolve_backend_type(cfg) == backend
    executor = build_executor(
        cfg, system_prompt="Be concise.", max_turns=4,
        memory_max_messages=4, memory_reset_to_messages=2,
    )
    assert isinstance(executor, CodeCliExecutor)
    first = executor.invoke({"input": "First"})
    second = executor.invoke({"input": "Second"})
    assert first["output"] == second["output"] == "OK"
    assert [step[0].tool for step in first["intermediate_steps"]] == ["exec"]
    assert first["raw_result"].raw_responses[0]["usage"]["input_tokens"] > 0
    assert len(executor._conversation) == 4
    compacted = executor.compact_for_resume("keep the answer")
    assert compacted.attempted and compacted.succeeded
    assert compacted.method == ("/compact" if backend == "kimi" else "/compress")
    calls = [json.loads(x) for x in (fake_cli.parent / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 3
    assert "exec" in calls[0]["allowed"]
    assert Path(calls[0]["cwd"]).name == "workspace"
    assert not list(Path(calls[0]["cwd"]).glob("*.py"))
    if backend == "kimi":
        assert "--agent-file" in calls[0]["args"]
        assert "--agent-file" not in calls[1]["args"]
        assert calls[1]["args"][-2:] == ["--session", "session_test"]
        assert "/compact keep the answer" in calls[2]["args"]
    else:
        assert "--system-prompt" in calls[0]["args"]
        assert calls[1]["args"][-2:] == ["--resume", "qwen-test"]
        assert "/compress" in calls[2]["args"]


@pytest.mark.parametrize("backend", ["kimi", "qwen"])
def test_native_host_tools_removed_and_exec_switch_honored(
    fake_cli: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    monkeypatch.setenv("FAKE_KIND", backend)
    executor = build_executor(
        config(backend, exec_enabled=False), system_prompt="System.",
        max_turns=4, memory_max_messages=4, memory_reset_to_messages=2,
    )
    executor._ensure_home()
    assert "exec" not in executor._allowed_tools
    home = Path(executor._home)
    if backend == "kimi":
        cfg = (home / "config.toml").read_text()
        assert 'enabled = ["mcp__chack_tools__*"]' in cfg
        assert "Bash" not in cfg
        agent = (home / "chack-agent.md").read_text()
        assert "mcp__chack_tools__*" in agent
    else:
        cfg = json.loads((home / "settings.json").read_text())
        assert "run_shell_command" in cfg["tools"]["disabled"]
        assert "read_file" in cfg["tools"]["disabled"]
        assert "tool_search" in cfg["tools"]["disabled"]
        assert cfg["tools"]["core"]
        assert cfg["model"]["reasoningEffort"] == "high"
    assert os.stat(home).st_mode & 0o077 == 0


def test_qwen_requires_explicit_endpoint_for_direct_key(fake_cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_BASE_URL")
    with pytest.raises(ValueError, match="QWEN_BASE_URL"):
        build_executor(
            config("qwen"), system_prompt="System.", max_turns=2,
            memory_max_messages=4, memory_reset_to_messages=2,
        )


def test_qwen_keeps_native_web_when_enabled(fake_cli: Path) -> None:
    cfg = config("qwen")
    cfg.tools.native_web_search_enabled = True
    executor = build_executor(
        cfg, system_prompt="System.", max_turns=2,
        memory_max_messages=4, memory_reset_to_messages=2,
    )
    executor._ensure_home()
    settings = json.loads((Path(executor._home) / "settings.json").read_text())
    assert {"web_search", "web_fetch"}.issubset(settings["tools"]["core"])
    assert not {"web_search", "web_fetch"}.intersection(settings["tools"]["disabled"])
    assert "run_shell_command" in settings["tools"]["disabled"]


def test_qwen_omits_unsupported_reasoning_field_for_claude_gateway_model(fake_cli: Path) -> None:
    cfg = config("qwen")
    cfg.model.primary = "bakkt/claude-opus-5.5"
    executor = build_executor(
        cfg, system_prompt="System.", max_turns=2,
        memory_max_messages=4, memory_reset_to_messages=2,
    )
    executor._ensure_home()
    settings = json.loads((Path(executor._home) / "settings.json").read_text())
    assert settings["model"] == {}


def test_qwen_cli_caps_native_tool_calls_as_well_as_mcp_calls(fake_cli: Path) -> None:
    cfg = config("qwen")
    cfg.tools.max_tools_used = 4
    executor = build_executor(
        cfg, system_prompt="System.", max_turns=2,
        memory_max_messages=4, memory_reset_to_messages=2,
    )
    command = executor._command("Hello")
    assert command[command.index("--max-tool-calls") + 1] == str(
        4 + sum(name in executor._allowed_tools for name in ("task_steps_manager", "check_budget_status"))
    )


@pytest.mark.parametrize("backend", ["kimi", "qwen"])
def test_schema_output_is_validated_and_normalized(
    fake_cli: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    monkeypatch.setenv("FAKE_KIND", backend)
    monkeypatch.setenv("FAKE_CONTENT", '```json\n{"answer": 47}\n```')
    cfg = config(backend)
    cfg.agent.output_schema_json = {
        "type": "object", "properties": {"answer": {"type": "integer"}},
        "required": ["answer"], "additionalProperties": False,
    }
    executor = build_executor(
        cfg, system_prompt="System.", max_turns=2,
        memory_max_messages=4, memory_reset_to_messages=2,
    )
    result = executor.invoke({"input": "Return JSON."})
    assert json.loads(result["output"]) == {"answer": 47}


@pytest.mark.parametrize("backend", ["kimi", "qwen"])
def test_role_model_defaults_inherit_explicit_cli_model(backend: str) -> None:
    cfg = resolve_config_aliases(config(backend))
    assert cfg.model.primary == "test-model"
    assert cfg.model.scientific == "test-model"
    assert cfg.model.cli == "test-model"
    cfg.model.primary = "BEST_QUALITY"
    with pytest.raises(ValueError, match="explicit agent.primary"):
        resolve_config_aliases(cfg)


def test_cli_failure_reports_stderr(fake_cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_KIND", "kimi")
    monkeypatch.setenv("FAKE_FAIL", "1")
    executor = build_executor(
        config("kimi"), system_prompt="System.", max_turns=2,
        memory_max_messages=2, memory_reset_to_messages=1,
    )
    result = executor.invoke({"input": "Hello"})
    assert "diagnostic from stderr" in result["output"]


def test_qwen_returns_terminal_result_when_post_result_drain_stalls(
    fake_cli: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_KIND", "qwen")
    monkeypatch.setenv("FAKE_HANG_AFTER_RESULT", "1")
    executor = build_executor(
        config("qwen"), system_prompt="System.", max_turns=2,
        memory_max_messages=2, memory_reset_to_messages=1,
    )
    started = time.monotonic()
    result = executor.invoke({"input": "Hello"})
    assert result["output"] == "OK"
    assert time.monotonic() - started < 10
