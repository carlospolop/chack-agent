"""What the installed Claude Code CLI actually sends for a Chack run.

The CLI is pointed at a local server that records the first model request and
answers it with an error, so no model is called.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chack_agent.backends.claude_code_backend import build_executor
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

pytestmark = pytest.mark.skipif(shutil.which("claude") is None, reason="Claude Code CLI not installed")


def _capture_first_request(tmp_path, monkeypatch, model: dict | None = None, **tools) -> dict:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            if self.command == "POST" and "/messages" in self.path and body:
                requests.append(json.loads(body))
            message = b'{"type":"error","error":{"type":"invalid_request_error","message":"captured"}}'
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)

        do_GET = do_POST = _answer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("CHACK_CLAUDE_HOME_BASE", str(tmp_path / "claude-homes"))
        workdir = tmp_path / "work"
        workdir.mkdir(parents=True)
        config = ChackConfig(
            model=ModelConfig(**{"primary": "claude-haiku-5-5", "provider": "claude", **(model or {})}),
            agent=AgentConfig(main_action="test", sub_action="test"),
            session=SessionConfig(max_turns=2),
            tools=ToolsConfig(**{"exec_enabled": True, "task_steps_manager_enabled": False, **tools}),
            credentials=CredentialsConfig(anthropic_api_key="sk-ant-test"),
            logging=LoggingConfig(),
            system_prompt="system",
            env={},
        )
        executor = build_executor(
            config, system_prompt="system", max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
        )
        executor._ensure_claude_home_and_settings()
        env = executor._build_env()
        env.update(
            {
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}",
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "CLAUDE_CODE_MAX_RETRIES": "0",
            }
        )
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        subprocess.run(
            executor._build_command("Run printf alpha"),
            input="Run printf alpha",
            env=env,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        server.shutdown()
    assert requests, "Claude Code sent no model request"
    return max(requests, key=lambda request: len(request.get("tools") or []))


def test_a_run_sends_its_tools_and_no_unused_builtin_ones(tmp_path, monkeypatch):
    request = _capture_first_request(tmp_path, monkeypatch)

    tools = {tool["name"] for tool in request.get("tools") or []}
    assert "mcp__chack_tools__exec" in tools
    assert {"Read", "Edit", "Write"} <= tools
    # Chack's exec tool is the run's only shell.
    assert "Bash" not in tools
    assert not tools & {"Agent", "Task", "Workflow", "Monitor", "CronCreate", "EnterWorktree", "SendMessage"}

    text = json.dumps(request)
    assert "persistent file-based memory" not in text
    assert "gitStatus" not in text
    assert "Co-Authored-By" not in text
    assert "Run printf alpha" in text


def test_web_search_can_be_turned_off(tmp_path, monkeypatch):
    request = _capture_first_request(tmp_path, monkeypatch, native_web_search_enabled=False)

    tools = {tool["name"] for tool in request.get("tools") or []}
    assert not tools & {"WebSearch", "WebFetch"}
    assert "mcp__chack_tools__exec" in tools


def test_a_run_without_exec_gets_no_shell_or_host_file_tools(tmp_path, monkeypatch):
    request = _capture_first_request(tmp_path, monkeypatch, exec_enabled=False, scientific_enabled=True)

    tools = {tool["name"] for tool in request.get("tools") or []}
    assert not tools & {"Bash", "Read", "Edit", "Write", "Glob", "Grep", "mcp__chack_tools__exec"}
    assert "Run printf alpha" in json.dumps(request)


def test_prompt_caching_marks_the_request_unless_the_model_turns_it_off(tmp_path, monkeypatch):
    cached = _capture_first_request(tmp_path / "cached", monkeypatch)
    assert "cache_control" in json.dumps(cached)

    uncached = _capture_first_request(tmp_path / "uncached", monkeypatch, model={"prompt_cache_enabled": False})
    assert "cache_control" not in json.dumps(uncached)
