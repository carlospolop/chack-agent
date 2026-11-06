"""What the installed Gemini CLI actually sends for a Chack run.

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

from chack_agent.backends.gemini_cli_backend import build_executor
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

_GEMINI = shutil.which(os.environ.get("GEMINI_CLI_PATH", "") or "gemini")

pytestmark = pytest.mark.skipif(_GEMINI is None, reason="Gemini CLI not installed")


def _capture_first_request(tmp_path, monkeypatch, **tools) -> dict:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            if self.command == "POST" and "generatecontent" in self.path.lower() and body:
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
        monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-homes"))
        monkeypatch.setenv("GEMINI_CLI_PATH", _GEMINI)
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        config = ChackConfig(
            model=ModelConfig(primary="gemini-3-flash", provider="gemini"),
            agent=AgentConfig(main_action="test", sub_action="test"),
            session=SessionConfig(max_turns=2),
            tools=ToolsConfig(exec_enabled=True, task_steps_manager_enabled=False, **tools),
            credentials=CredentialsConfig(gemini_api_key="test-key"),
            logging=LoggingConfig(),
            system_prompt="system",
            env={},
        )
        executor = build_executor(
            config, system_prompt="system", max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
        )
        executor._ensure_gemini_home_and_config()
        executor._write_gemini_settings(executor._gemini_home, tools_cwd=str(workdir))
        env = executor._build_env()
        env["GOOGLE_GEMINI_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}"
        done = subprocess.run(
            executor._build_command("Run printf alpha"),
            env=env,
            cwd=executor._gemini_workspace_dir(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        server.shutdown()
    assert requests, f"Gemini CLI sent no model request: {done.stdout[-1500:]}"
    return requests[0]


def _tool_names(request: dict) -> set[str]:
    return {
        declaration["name"]
        for group in request.get("tools") or []
        for declaration in group.get("functionDeclarations") or []
    }


def test_a_run_can_use_the_chack_tools(tmp_path, monkeypatch):
    request = _capture_first_request(tmp_path, monkeypatch)

    tools = _tool_names(request)
    assert "mcp_chack_tools_exec" in tools
    assert "google_web_search" in tools
    # Gemini's own file and shell tools would work in its empty launch directory.
    assert not tools & {"run_shell_command", "read_file", "write_file", "activate_skill", "invoke_agent"}

    text = json.dumps(request)
    assert "<available_skills>" not in text
    assert "Run printf alpha" in text


def test_web_search_can_be_turned_off(tmp_path, monkeypatch):
    tools = _tool_names(_capture_first_request(tmp_path, monkeypatch, native_web_search_enabled=False))

    assert "google_web_search" not in tools
    assert "mcp_chack_tools_exec" in tools
