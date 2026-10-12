from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from chack_agent.backends import codex_backend as codex_backend_module
from chack_agent.backends.codex_backend import (
    CodexExecutor,
    _cleanup_isolated_codex_home,
    _preview_text,
)


def test_windows_pipe_reader_handles_buffered_subprocess_lines(monkeypatch):
    monkeypatch.setattr(
        codex_backend_module,
        "_uses_windows_pipe_reader",
        lambda: True,
    )
    stream = io.StringIO("first\nsecond\n")

    assert codex_backend_module._readline_when_ready(stream, 1.0) == "first\n"
    assert codex_backend_module._readline_when_ready(stream, 1.0) == "second\n"
    assert codex_backend_module._readline_when_ready(stream, 1.0) == ""
    assert id(stream) not in codex_backend_module._PIPE_READERS


def test_linux_pipe_reader_does_not_block_deadlines_on_a_partial_line(monkeypatch):
    monkeypatch.setattr(codex_backend_module, "_uses_windows_pipe_reader", lambda: False)
    with subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); "
            "time.sleep(0.8); sys.stdout.write(' line\\n'); sys.stdout.flush()",
        ],
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        started = time.monotonic()
        assert codex_backend_module._readline_when_ready(process.stdout, 0.1) is None
        assert time.monotonic() - started < 0.4
        assert codex_backend_module._readline_when_ready(process.stdout, 1.5) == "partial line\n"
        assert codex_backend_module._readline_when_ready(process.stdout, 1.0) == ""
        assert process.wait(timeout=2) == 0


def test_linux_pipe_reader_drains_lines_buffered_by_text_stream(monkeypatch):
    monkeypatch.setattr(codex_backend_module, "_uses_windows_pipe_reader", lambda: False)
    with subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys,time; sys.stdout.write('first\\nsecond\\n'); "
            "sys.stdout.flush(); time.sleep(0.8)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        assert codex_backend_module._readline_when_ready(process.stdout, 1.0) == "first\n"
        assert codex_backend_module._readline_when_ready(process.stdout, 0.1) == "second\n"
        assert codex_backend_module._readline_when_ready(process.stdout, 1.5) == ""
        assert process.wait(timeout=2) == 0


def test_web_research_silence_has_a_separate_cap_from_slow_browser_research():
    resolve = codex_backend_module._codex_silence_limit_seconds
    env = {
        "CHACK_RESEARCHER_LONG_TOOL_SILENCE_SECONDS": "7800",
        "CHACK_WEBSEARCHER_LONG_TOOL_SILENCE_SECONDS": "900",
        "CHACK_RESEARCHER_SILENCE_SECONDS": "300",
    }
    assert resolve("webresearcher", saw_output=True, long_tool=True, runtime_env=env) == 900
    assert resolve("webresearcher", saw_output=True, long_tool=False, runtime_env=env) == 300
    assert resolve("scientific", saw_output=True, long_tool=True, runtime_env=env) == 7800
    assert resolve("researcher_administrator", saw_output=True, long_tool=True,
                   runtime_env={"CHACK_RESEARCH_ADMIN_LONG_TOOL_SILENCE_SECONDS": "7800"}) == 7800
    assert resolve("verifier", saw_output=True, long_tool=True, runtime_env=env) == 0


def test_web_research_silent_tool_call_terminates_owned_codex_process(monkeypatch):
    process_box = {}
    script = (
        "import json,time; "
        "print(json.dumps({'type':'thread.started','thread_id':'test-silence'}),flush=True); "
        "print(json.dumps({'type':'item.started','item':{'type':'mcp_tool_call'}}),flush=True); "
        "time.sleep(10)"
    )
    real_popen = subprocess.Popen

    def launch(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        if args and args[0] and args[0][0] == sys.executable:
            process_box["process"] = process
        return process

    monkeypatch.setattr(codex_backend_module.subprocess, "Popen", launch)
    executor = object.__new__(CodexExecutor)
    executor._model_name = "fake-model"
    executor._model_provider = "fake-provider"
    executor._sub_action = "webresearcher"
    executor._thread_id = ""
    executor._codex_home = ""
    executor._runtime_env_json = json.dumps({
        "CHACK_CODEX_EXEC_TIMEOUT_SECONDS": "8",
        "CHACK_CODEX_STARTUP_SILENCE_SECONDS": "2",
        "CHACK_WEBSEARCHER_LONG_TOOL_SILENCE_SECONDS": "1",
    })
    monkeypatch.setattr(executor, "_build_command", lambda: [sys.executable, "-u", "-c", script])
    monkeypatch.setattr(executor, "_build_env", lambda: os.environ.copy())
    monkeypatch.setattr(executor, "_write_codex_explicit_mcp_env", lambda _env: None)
    errors = []
    monkeypatch.setattr(executor, "_log_codex_failure", lambda code, **_kw: errors.append(code))
    started = time.monotonic()
    try:
        result, _steps, _raw = executor._run_codex_once("test", allow_api_key_fallback=False)
        assert time.monotonic() - started < 5
        assert "no progress output for 1s (role=webresearcher, active_tool=True)" in result
        assert "codex_cli_silence_watchdog" in errors
        assert process_box["process"].poll() is not None
        assert id(process_box["process"].stdout) not in codex_backend_module._PIPE_READERS
    finally:
        process = process_box.get("process")
        if process and process.poll() is None:
            codex_backend_module._terminate_process_tree(process)
            process.wait(timeout=2)


def test_failure_preview_preserves_start_and_terminal_error():
    text = "startup:" + ("x" * 100) + ":terminal provider error"

    preview = _preview_text(text, max_chars=80)

    assert len(preview) == 80
    assert preview.startswith("startup:")
    assert preview.endswith(":terminal provider error")
    assert "truncated middle" in preview


def test_cleanup_isolated_codex_home_removes_only_child(tmp_path: Path):
    home_base = tmp_path / "chack"
    target = home_base / "session-1"
    target.mkdir(parents=True)
    (target / "config.toml").write_text("model = 'test'", encoding="utf-8")

    assert _cleanup_isolated_codex_home(str(target), str(home_base)) is True
    assert not target.exists()
    assert home_base.is_dir()


def test_cleanup_isolated_codex_home_refuses_base_and_outside(tmp_path: Path):
    home_base = tmp_path / "chack"
    outside = tmp_path / "outside"
    home_base.mkdir()
    outside.mkdir()

    assert _cleanup_isolated_codex_home(str(home_base), str(home_base)) is False
    assert _cleanup_isolated_codex_home(str(outside), str(home_base)) is False
    assert home_base.is_dir()
    assert outside.is_dir()


def test_executor_cleanup_resets_runtime_paths(tmp_path: Path):
    home_base = tmp_path / "chack"
    target = home_base / "session-2"
    target.mkdir(parents=True)
    executor = object.__new__(CodexExecutor)
    executor._runtime_env_json = (
        '{"CHACK_CODEX_HOME_BASE": "' + str(home_base) + '"}'
    )
    executor._codex_home = str(target)
    executor._output_schema_path = str(target / "output-schema.json")

    assert executor.cleanup_runtime_artifacts() is True
    assert executor._codex_home is None
    assert executor._output_schema_path is None
    assert home_base.is_dir()
