from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

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


def test_administrator_reads_only_active_async_child_artifacts(tmp_path: Path):
    ledger_dir = tmp_path / "researcher_jobs"
    ledger_dir.mkdir()
    child_dir = tmp_path / "news_media"
    child_dir.mkdir()
    manifest = child_dir / "_artifact_manifest.jsonl"
    manifest.write_text('{"tool":"bing"}\n', encoding="utf-8")
    ledger = ledger_dir / "one.json"
    ledger.write_text(json.dumps({
        "kind": "async",
        "tasks": [{"status": "running", "execution_active": True,
                   "researcher": "news_media"}],
    }), encoding="utf-8")

    active, mtime = codex_backend_module._active_async_child_artifact_mtime(str(tmp_path))
    assert active is True
    assert mtime == manifest.stat().st_mtime

    ledger.write_text(json.dumps({
        "kind": "async",
        "tasks": [{"status": "done", "execution_active": False,
                   "researcher": "news_media"}],
    }), encoding="utf-8")
    assert codex_backend_module._active_async_child_artifact_mtime(str(tmp_path)) == (False, 0.0)
    ledger.write_text("{incomplete", encoding="utf-8")
    assert codex_backend_module._active_async_child_artifact_mtime(str(tmp_path)) == (False, 0.0)


def test_admin_silence_watchdog_follows_writing_child_then_stops_on_stale_child(
    monkeypatch, tmp_path: Path,
):
    ledger_dir = tmp_path / "researcher_jobs"
    ledger_dir.mkdir()
    (ledger_dir / "one.json").write_text(json.dumps({
        "kind": "async",
        "tasks": [{"status": "running", "execution_active": True,
                   "researcher": "news_media"}],
    }), encoding="utf-8")
    folder = tmp_path / "news_media"
    folder.mkdir()
    manifest = folder / "_artifact_manifest.jsonl"
    script = (
        "import json,time; "
        "print(json.dumps({'type':'thread.started','thread_id':'child-progress'}),flush=True); "
        "print(json.dumps({'type':'item.started','item':{'type':'mcp_tool_call'}}),flush=True); "
        "time.sleep(2.0); "
        "print(json.dumps({'type':'item.completed','item':{'type':'agent_message',"
        "'text':'admin done'}}),flush=True); "
        "print(json.dumps({'type':'turn.completed','usage':{}}),flush=True)"
    )
    executor = object.__new__(CodexExecutor)
    executor._model_name = "fake-model"
    executor._model_provider = "fake-provider"
    executor._sub_action = "researcher_administrator"
    executor._thread_id = ""
    executor._codex_home = ""
    executor._runtime_env_json = json.dumps({
        "CHACK_CODEX_EXEC_TIMEOUT_SECONDS": "8",
        "CHACK_RESEARCH_ADMIN_LONG_TOOL_SILENCE_SECONDS": "1",
        "CHACK_RESEARCH_ADMIN_SILENCE_SECONDS": "1",
        "CHACK_RESEARCH_MASTER_DIR": str(tmp_path),
    })
    monkeypatch.setattr(executor, "_build_command", lambda: [sys.executable, "-u", "-c", script])
    monkeypatch.setattr(executor, "_build_env", lambda: os.environ.copy())
    monkeypatch.setattr(executor, "_write_codex_explicit_mcp_env", lambda _env: None)
    errors = []
    monkeypatch.setattr(executor, "_log_codex_failure", lambda code, **_kw: errors.append(code))

    def write_child_progress():
        for index in range(4):
            time.sleep(0.4)
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"tool": "bing", "number": index}) + "\n")

    writer = threading.Thread(target=write_child_progress)
    writer.start()
    try:
        output, _steps, _raw = executor._run_codex_once("test", allow_api_key_fallback=False)
        assert "admin done" in output
        assert "codex_cli_silence_watchdog" not in errors
    finally:
        writer.join(timeout=3)

    silent_script = script.replace("time.sleep(2.0)", "time.sleep(10)")
    monkeypatch.setattr(executor, "_build_command", lambda: [sys.executable, "-u", "-c", silent_script])
    output, _steps, _raw = executor._run_codex_once("test", allow_api_key_fallback=False)
    assert "no progress output for 1s" in output
    assert "codex_cli_silence_watchdog" in errors


def test_aborted_admin_marks_only_stopped_children_cancelled(tmp_path: Path):
    ledger_dir = tmp_path / "researcher_jobs"
    ledger_dir.mkdir()
    ledger = ledger_dir / "one.json"
    ledger.write_text(json.dumps({
        "kind": "async", "complete": False,
        "tasks": [
            {"researcher": "news_media", "status": "running", "execution_active": True,
             "process_pid": 99999999, "artifact_count": 242},
            {"researcher": "websearcher", "status": "done", "execution_active": False},
            {"researcher": "scientific", "status": "running", "execution_active": True,
             "process_pid": os.getpid()},
        ],
    }), encoding="utf-8")
    assert codex_backend_module._reconcile_aborted_admin_children(
        str(tmp_path), "administrator silence watchdog"
    ) == 1
    state = json.loads(ledger.read_text(encoding="utf-8"))
    news, web, scientific = state["tasks"]
    assert news["status"] == "cancelled"
    assert news["execution_active"] is False
    assert "administrator silence watchdog" in news["failure_reason"]
    assert news["artifact_count"] == 242
    assert news["finished_at"] > 0
    assert web["status"] == "done"
    assert scientific["status"] == "running"
    assert state["complete"] is False
    assert codex_backend_module._reconcile_aborted_admin_children(
        str(tmp_path), "administrator silence watchdog"
    ) == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal and /proc process-tree check")
def test_process_tree_cleanup_kills_term_ignoring_descendant_after_parent_exits():
    child_script = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    )
    parent_script = (
        "import subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-u','-c',{child_script!r}]); "
        "print(child.pid,flush=True); child.wait()"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", parent_script],
        stdout=subprocess.PIPE, text=True, start_new_session=True,
    )
    child_pid = 0
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
        time.sleep(0.2)
        assert codex_backend_module._pid_is_running(child_pid)
        assert child_pid in codex_backend_module._descendant_pids(process.pid)
        codex_backend_module._terminate_process_tree(process)
        process.wait(timeout=3)
        assert not codex_backend_module._pid_is_running(child_pid)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        if child_pid and codex_backend_module._pid_is_running(child_pid):
            os.kill(child_pid, 9)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal and /proc process-tree check")
def test_admin_watchdog_reconciles_real_stopped_child_and_preserves_artifacts(
    monkeypatch, tmp_path: Path,
):
    child_script = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    )
    codex_script = (
        "import json,os,subprocess,sys,time; from pathlib import Path; "
        "root=Path(sys.argv[1]); child=subprocess.Popen([sys.executable,'-u','-c',"
        f"{child_script!r}]); "
        "ledger=root/'researcher_jobs'; ledger.mkdir(); "
        "folder=root/'news_media'; folder.mkdir(); "
        "(folder/'_artifact_manifest.jsonl').write_text('preserved evidence'); "
        "(ledger/'one.json').write_text(json.dumps({'kind':'async','complete':False,"
        "'tasks':[{'researcher':'news_media','status':'running',"
        "'execution_active':True,'process_pid':child.pid,'artifact_count':1}]})); "
        "print(json.dumps({'type':'thread.started','thread_id':'admin-reconcile'}),flush=True); "
        "print(json.dumps({'type':'item.started','item':{'type':'mcp_tool_call'}}),flush=True); "
        "time.sleep(10)"
    )
    executor = object.__new__(CodexExecutor)
    executor._model_name = "fake-model"
    executor._model_provider = "fake-provider"
    executor._sub_action = "researcher_administrator"
    executor._thread_id = ""
    executor._codex_home = ""
    executor._runtime_env_json = json.dumps({
        "CHACK_CODEX_EXEC_TIMEOUT_SECONDS": "8",
        "CHACK_RESEARCH_ADMIN_LONG_TOOL_SILENCE_SECONDS": "1",
        "CHACK_RESEARCH_MASTER_DIR": str(tmp_path),
    })
    monkeypatch.setattr(executor, "_build_command", lambda: [
        sys.executable, "-u", "-c", codex_script, str(tmp_path),
    ])
    monkeypatch.setattr(executor, "_build_env", lambda: os.environ.copy())
    monkeypatch.setattr(executor, "_write_codex_explicit_mcp_env", lambda _env: None)
    monkeypatch.setattr(executor, "_log_codex_failure", lambda *_args, **_kw: None)
    pid = 0
    try:
        output, _steps, _raw = executor._run_codex_once("test", allow_api_key_fallback=False)
        assert "no progress output for 1s" in output
        state = json.loads((tmp_path / "researcher_jobs" / "one.json").read_text())
        task = state["tasks"][0]
        pid = task["process_pid"]
        assert task["status"] == "cancelled"
        assert state["complete"] is True
        assert task["artifact_count"] == 1
        assert not codex_backend_module._pid_is_running(pid)
        assert (tmp_path / "news_media" / "_artifact_manifest.jsonl").read_text() == "preserved evidence"
    finally:
        if pid and codex_backend_module._pid_is_running(pid):
            os.kill(pid, 9)


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
