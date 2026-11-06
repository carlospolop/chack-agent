"""What a chack run through the real Claude Code CLI sends to the model."""

from __future__ import annotations

import http.server
import json
import shutil
import threading

import pytest

from chack_agent import Chack
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
)
from chack_tools.config import ToolsConfig

pytestmark = pytest.mark.skipif(shutil.which("claude") is None, reason="Claude Code CLI not installed")


@pytest.fixture
def captured_requests():
    """A model endpoint that records each request and rejects it."""
    bodies: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            if "/messages" in self.path:
                bodies.append(json.loads(self.rfile.read(length)))
            else:
                self.rfile.read(length)
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"type":"error","error":{"type":"invalid_request_error","message":"captured"}}')

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api", bodies
    server.shutdown()


def _first_request(tmp_path, monkeypatch, captured_requests) -> dict:
    base_url, bodies = captured_requests
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "CLAUDE.md").write_text("canary-host-instruction\n")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "CLAUDE.md").write_text("canary-repo-instruction\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workdir)
    config = ChackConfig(
        model=ModelConfig(primary="claude-haiku-5-5", provider="claude", api_key_type="openrouter"),
        agent=AgentConfig(thinking_effort="low", main_action="test", sub_action="claude"),
        session=SessionConfig(max_turns=2, long_term_memory_enabled=False),
        tools=ToolsConfig(task_steps_manager_enabled=False, exec_enabled=True),
        credentials=CredentialsConfig(openrouter_api_key="sk-or-test", openrouter_base_url=base_url),
        logging=LoggingConfig(level="WARNING"),
        system_prompt="You are a precise assistant.",
        env={},
    )

    Chack(config).run(
        "claude-request-test",
        "Run `printf alpha` with exec.",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )

    assert bodies, "Claude Code sent no model request"
    return bodies[0]


def test_a_run_sends_only_the_tools_and_instructions_it_uses(tmp_path, monkeypatch, captured_requests):
    request = _first_request(tmp_path, monkeypatch, captured_requests)

    tools = {tool["name"] for tool in request["tools"]}
    text = json.dumps(request)
    assert "mcp__chack_tools__exec" in tools
    # chack's exec is the shell; only file and web built-ins remain.
    assert tools - {"mcp__chack_tools__exec", "mcp__chack_tools__check_budget_status"} <= {
        "Read", "Edit", "Write", "ToolSearch", "WebFetch", "DeferredToolPlaceholder",
    }
    assert "canary-host-instruction" not in text
    assert "canary-repo-instruction" not in text
    assert "persistent file-based memory" not in text
    assert "Attribution for git commits" not in text
    # The run's own prompt and task still arrive.
    assert "CHACK RUNTIME" in text
    assert "printf alpha" in text
