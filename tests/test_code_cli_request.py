"""Capture the real first model request from each installed CLI.

The endpoint is a loopback HTTP server which always returns an error. This
checks the tool registry and wire configuration without using credentials.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from chack_agent.backends import build_executor
from chack_agent.config import (
    AgentConfig, ChackConfig, CredentialsConfig, LoggingConfig,
    ModelConfig, SessionConfig, ToolsConfig,
)


@pytest.mark.parametrize("backend", ["kimi", "qwen"])
def test_real_cli_first_request_has_chack_tools_only(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = shutil.which(os.environ.get(f"{backend.upper()}_CLI_PATH", "") or backend)
    if executable is None:
        pytest.skip(f"{backend} Code CLI is not installed")
    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if body:
                try:
                    captured.append(json.loads(body))
                except ValueError:
                    pass
            answer = b'{"error":{"message":"request captured","type":"invalid_request_error"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def do_GET(self):
            answer = b'{"data":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv(f"CHACK_{backend.upper()}_HOME", str(tmp_path / "home"))
        monkeypatch.setenv(f"{backend.upper()}_CLI_PATH", executable)
        monkeypatch.setenv(f"{backend.upper()}_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.chdir(tmp_path)
        config = ChackConfig(
            ModelConfig(primary="local-test-model", provider=backend),
            AgentConfig(require_task_steps_manager_init_first=False),
            SessionConfig(max_turns=2),
            ToolsConfig(exec_enabled=True, task_steps_manager_enabled=False,
                        native_web_search_enabled=False),
            CredentialsConfig(kimi_api_key="test-key", qwen_api_key="test-key"),
            LoggingConfig(), "Answer briefly.", {},
        )
        executor = build_executor(
            config, system_prompt="Answer briefly.", max_turns=2,
            memory_max_messages=2, memory_reset_to_messages=1,
        )
        executor._ensure_home()
        done = subprocess.run(
            executor._command("Reply OK."), env=executor._build_env(),
            cwd=Path(executor._home) / "workspace",
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        )
        assert captured, f"{backend} sent no model request: {done.stderr[-500:]} {done.stdout[-500:]}"
        tools = captured[0].get("tools") or []
        names = {
            item.get("function", {}).get("name") or item.get("name")
            for item in tools if isinstance(item, dict)
        }
        assert any("chack_tools" in str(name) and str(name).endswith("exec") for name in names)
        assert all("chack_tools" in str(name) for name in names), names
        assert not names.intersection({
            "Bash", "run_shell_command", "shell", "read_file", "write_file", "edit",
        })
    finally:
        server.shutdown()
