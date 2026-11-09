"""A cancelled Gemini CLI or Copilot CLI run stops at once, like Codex, Claude Code
and opencode runs, and a Gemini gateway run counts the calls it made before.

The CLIs are fake: small Python scripts that act like the real ones (ask the
model through the relay, print their JSON events, start a child process like
an MCP server, then wait), so no model is called.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chack_agent.backends import copilot_cli_backend, gemini_cli_backend
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)
from chack_agent.live_cost_state import reset_active_usage_sink, set_active_usage_sink
from chack_tools.cancellation import request_cancel, reset_cancellation_event, set_cancellation_event

GATEWAY_MODEL = "bakkt/gemini-3.6-flash"
ANSWER = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "working on it"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 1234,
        "candidatesTokenCount": 56,
        "cachedContentTokenCount": 200,
        "totalTokenCount": 1290,
    },
}
# Long enough that only a stopped run returns in time.
RUN_JOIN_SECONDS = 20

# A CLI that, once started, asks the model one question (through the URL the
# backend gives it), prints what Gemini CLI prints, starts a child that lives
# as long as it is not killed, says it is ready, and then waits quietly, as a
# CLI does while the model works on its next answer.
FAKE_CLI = r'''#!{python}
import json, os, subprocess, sys, time, urllib.request
state = {state!r}
base = os.environ.get("GOOGLE_GEMINI_BASE_URL", "")
if base:
    request = urllib.request.Request(
        base.rstrip("/") + "/v1beta/models/{model}:streamGenerateContent?alt=sse",
        data=b'{{"contents": []}}',
        headers={{"Content-Type": "application/json", "x-goog-api-key": "fake-key"}},
        method="POST",
    )
    urllib.request.urlopen(request, timeout=10).read()
print(json.dumps({{"type": "init", "session_id": "fake-session"}}), flush=True)
print(json.dumps({{"type": "message", "role": "assistant", "content": "working on it"}}), flush=True)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
with open(os.path.join(state, "child.pid"), "w") as handle:
    handle.write(str(child.pid))
with open(os.path.join(state, "ready"), "w") as handle:
    handle.write("1")
time.sleep(300)
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of a killed parent may linger as a zombie until reaped.
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
        return waited == 0
    except ChildProcessError:
        return False


def _wait_for(path, seconds: float = RUN_JOIN_SECONDS) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def fake_cli(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    script = tmp_path / "fake-cli"
    script.write_text(FAKE_CLI.format(python=sys.executable, state=str(state), model=GATEWAY_MODEL))
    script.chmod(0o755)
    return script, state


@pytest.fixture
def gateway():
    """A gateway that answers unstreamed calls with ``ANSWER``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            body = json.dumps(ANSWER).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/gw"
    finally:
        server.shutdown()
        server.server_close()


def _config(provider: str, model: str) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary=model, provider=provider),
        agent=AgentConfig(main_action="test", sub_action="test"),
        session=SessionConfig(max_turns=2),
        tools=ToolsConfig(exec_enabled=False, task_steps_manager_enabled=False),
        credentials=CredentialsConfig(gemini_api_key="fake-key"),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-homes"))
    monkeypatch.setenv("CHACK_COPILOT_HOME", str(tmp_path / "copilot-homes"))
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "fake-github-token")
    for name in ("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "CHACK_GEMINI_EXEC_TIMEOUT_SECONDS",
                 "CHACK_COPILOT_EXEC_TIMEOUT_SECONDS", "GOOGLE_GEMINI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _gemini_executor(monkeypatch, script, gateway_url: str):
    monkeypatch.setenv("GEMINI_CLI_PATH", str(script))
    executor = gemini_cli_backend.build_executor(
        _config("gemini", GATEWAY_MODEL),
        system_prompt="system",
        max_turns=2,
        memory_max_messages=10,
        memory_reset_to_messages=5,
    )
    executor._gemini_base_url = gateway_url
    return executor


def _copilot_executor(monkeypatch, script):
    monkeypatch.setenv("COPILOT_CLI_PATH", str(script))
    return copilot_cli_backend.build_executor(
        _config("copilot", "gpt-5.6"),
        system_prompt="system",
        max_turns=2,
        memory_max_messages=10,
        memory_reset_to_messages=5,
    )


def _start_run(run, event):
    """Run ``run`` on a thread under ``event``; returns the thread and its result box."""

    box: dict = {"reports": []}

    def target():
        cancel_token = set_cancellation_event(event)
        sink_token = set_active_usage_sink(
            lambda model, prompt, completion, cached, write: box["reports"].append(
                (model, prompt, completion, cached)
            )
        )
        try:
            box["result"] = run()
        except BaseException as exc:  # surfaced by the assertions
            box["error"] = exc
        finally:
            reset_active_usage_sink(sink_token)
            reset_cancellation_event(cancel_token)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


def test_a_cancelled_gemini_gateway_run_stops_and_counts_the_calls_it_made(
    cli_env, monkeypatch, fake_cli, gateway
):
    script, state = fake_cli
    executor = _gemini_executor(monkeypatch, script, gateway)
    event = threading.Event()

    thread, box = _start_run(lambda: executor._run_gemini("Prove it."), event)
    assert _wait_for(state / "ready"), "the fake Gemini CLI never got its answer"
    child_pid = int((state / "child.pid").read_text())
    request_cancel(event)
    thread.join(RUN_JOIN_SECONDS)

    assert not thread.is_alive(), "the Gemini run kept going after it was cancelled"
    assert "error" not in box, box.get("error")
    output, _steps, raw = box["result"]
    assert output == "ERROR: Gemini execution cancelled."
    # The one answer the run got is counted, though the CLI never printed its stats.
    assert box["reports"] == [(GATEWAY_MODEL, 1234, 56, 200)]
    assert [response["usage"]["input_tokens"] for response in raw.raw_responses] == [1234]
    assert not _alive(child_pid), "the Gemini CLI's child process was left running"


def test_a_cancelled_copilot_run_stops_and_kills_its_processes(cli_env, monkeypatch, fake_cli):
    script, state = fake_cli
    executor = _copilot_executor(monkeypatch, script)
    event = threading.Event()

    thread, box = _start_run(lambda: executor._run_copilot("Prove it."), event)
    assert _wait_for(state / "ready"), "the fake Copilot CLI never started"
    child_pid = int((state / "child.pid").read_text())
    request_cancel(event)
    thread.join(RUN_JOIN_SECONDS)

    assert not thread.is_alive(), "the Copilot run kept going after it was cancelled"
    assert "error" not in box, box.get("error")
    output, _steps, _raw = box["result"]
    assert output == "ERROR: Copilot execution cancelled."
    assert not _alive(child_pid), "the Copilot CLI's child process was left running"


@pytest.mark.parametrize("backend", ["gemini", "copilot"])
def test_a_quiet_cli_still_stops_at_its_deadline(cli_env, monkeypatch, fake_cli, gateway, backend):
    script, state = fake_cli
    if backend == "gemini":
        monkeypatch.setenv("CHACK_GEMINI_EXEC_TIMEOUT_SECONDS", "3")
        executor = _gemini_executor(monkeypatch, script, gateway)
        run = lambda: executor._run_gemini("Prove it.")  # noqa: E731
        expected = "ERROR: Gemini execution timed out after 3s."
    else:
        monkeypatch.setenv("CHACK_COPILOT_EXEC_TIMEOUT_SECONDS", "3")
        executor = _copilot_executor(monkeypatch, script)
        run = lambda: executor._run_copilot("Prove it.")  # noqa: E731
        expected = "ERROR: Copilot execution timed out after 3s."

    thread, box = _start_run(run, threading.Event())
    thread.join(RUN_JOIN_SECONDS)

    assert not thread.is_alive(), "a CLI that printed nothing more outlived its deadline"
    assert box["result"][0] == expected
    assert not _alive(int((state / "child.pid").read_text()))


@pytest.mark.parametrize(
    ("backend", "start_message"),
    [("gemini", "Starting Gemini CLI process"), ("copilot", "Starting Copilot CLI process")],
)
def test_a_cli_run_without_its_own_timeout_gets_the_agents_runtime_budget(
    cli_env, monkeypatch, caplog, backend, start_message, tmp_path
):
    # Dynamic's poc-agent has max_runtime_minutes: 45, like Codex runs get.
    monkeypatch.setenv("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "2700")
    missing = str(tmp_path / "no-such-cli")
    if backend == "gemini":
        monkeypatch.setenv("GEMINI_CLI_PATH", missing)
        executor = gemini_cli_backend.build_executor(
            _config("gemini", "gemini-3-flash"), system_prompt="system", max_turns=2,
            memory_max_messages=10, memory_reset_to_messages=5,
        )
        run = lambda: executor._run_gemini("hi")  # noqa: E731
    else:
        monkeypatch.setenv("COPILOT_CLI_PATH", missing)
        executor = _copilot_executor(monkeypatch, missing)
        run = lambda: executor._run_copilot("hi")  # noqa: E731

    with caplog.at_level("INFO"):
        output, _steps, _raw = run()

    assert "was not found" in output
    starts = [record.getMessage() for record in caplog.records if start_message in record.getMessage()]
    assert starts and "timeout_seconds=3000" in starts[-1]


@pytest.mark.parametrize("backend", ["gemini", "copilot"])
def test_an_explicit_cli_timeout_still_wins_over_the_runtime_budget(cli_env, monkeypatch, caplog, backend, tmp_path):
    monkeypatch.setenv("CHACK_BUDGET_MAX_RUNTIME_SECONDS", "2700")
    missing = str(tmp_path / "no-such-cli")
    if backend == "gemini":
        monkeypatch.setenv("CHACK_GEMINI_EXEC_TIMEOUT_SECONDS", "600")
        monkeypatch.setenv("GEMINI_CLI_PATH", missing)
        executor = gemini_cli_backend.build_executor(
            _config("gemini", "gemini-3-flash"), system_prompt="system", max_turns=2,
            memory_max_messages=10, memory_reset_to_messages=5,
        )
        run = lambda: executor._run_gemini("hi")  # noqa: E731
    else:
        monkeypatch.setenv("CHACK_COPILOT_EXEC_TIMEOUT_SECONDS", "600")
        executor = _copilot_executor(monkeypatch, missing)
        run = lambda: executor._run_copilot("hi")  # noqa: E731

    with caplog.at_level("INFO"):
        run()

    assert any("timeout_seconds=600" in record.getMessage() for record in caplog.records)
