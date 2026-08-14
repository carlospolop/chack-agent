import os
import json
import asyncio
import subprocess
import sys
import time
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.research_artifacts import (
    cleanup_research_artifacts,
    reset_research_artifact_context,
    set_research_artifact_context,
)
from chack_tools.cancellation import (
    current_cancellation_event,
    register_process,
    request_cancel,
    reset_cancellation_event,
    set_cancellation_event,
    unregister_process,
)
from chack_tools.researcher_administrator_agent import (
    _ADMINISTRATOR_SYSTEM_PROMPT,
    RESEARCHER_REGISTRY,
    ResearcherAdministratorAgentTool,
    normalize_researcher_name,
)
from chack_tools.telemetry import run_with_tool_logging
from chack_tools import subagent_config as sc


def _tool_names(tools):
    return {
        str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")
        for tool in tools
    }


def _wait_for_file(path: Path, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    candidate = Path(path)
    while time.monotonic() < deadline:
        if candidate.exists():
            return True
        time.sleep(0.02)
    return candidate.exists()


def test_chatgpt_researchers_are_never_cancelled_for_elapsed_time():
    prompt = _ADMINISTRATOR_SYSTEM_PROMPT
    assert "Prefer `start_researchers_async`" in prompt
    assert "Never use `wait(..., terminate=true)`" in prompt
    assert "configured hard timeout" in prompt
    assert "up to 180 minutes" in prompt


def test_administrator_runtime_cap_is_explicit_and_configurable():
    default = ResearcherAdministratorAgentTool(
        ToolsConfig(),
        model_provider="openai",
        fallback_model="m",
    )
    queue = ResearcherAdministratorAgentTool(
        ToolsConfig(),
        model_provider="openai",
        fallback_model="m",
        runtime_cap_minutes=180,
    )

    assert default.runtime_cap_minutes == 90
    assert queue.runtime_cap_minutes == 180


def test_administrator_synthesis_reserve_defaults_to_five_minutes():
    assert ToolsConfig().researcher_administrator_synthesis_reserve_minutes == 5


def test_async_jobs_can_be_harvested_by_unique_evidence_workspace():
    import chack_tools.researcher_administrator_agent as admin_mod

    evidence_dir = "/tmp/chack-ledger-workspace-test"
    job_id = "research-job-owned-by-workspace"
    admin_mod._async_job_store(
        job_id,
        {
            "job_id": job_id,
            "created_at": time.time(),
            "evidence_dir": evidence_dir,
            "tasks": {},
        },
    )

    assert admin_mod._async_job_ids_for_evidence_dir(evidence_dir) == [job_id]
    assert admin_mod._async_job_ids_for_evidence_dir(evidence_dir + "-other") == []


def test_async_harvest_waits_for_nonterminal_job_before_return():
    import chack_tools.researcher_administrator_agent as admin_mod

    job_id = "research-job-harvest-waits"
    completion = threading.Event()
    admin_mod._async_job_store(
        job_id,
        {
            "job_id": job_id,
            "created_at": time.time(),
            "completion_event": completion,
            "expected_task_count": 1,
            "tasks": {
                "task-1": {
                    "task_id": "task-1",
                    "researcher_tool": "scientific_research",
                    "status": "running",
                }
            },
        },
    )

    def finish_job():
        time.sleep(0.05)
        with admin_mod._ASYNC_RESEARCH_LOCK:
            admin_mod._ASYNC_RESEARCH_JOBS[job_id]["tasks"]["task-1"]["status"] = "done"
        completion.set()

    worker = threading.Thread(target=finish_job)
    worker.start()
    try:
        pending = admin_mod._wait_for_async_jobs_terminal([job_id], time.monotonic() + 2)
    finally:
        worker.join(timeout=2)
        with admin_mod._ASYNC_RESEARCH_LOCK:
            admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)

    assert pending == []


def test_administrator_harvests_workspace_owned_job_without_contextvar_ids(monkeypatch, tmp_path):
    import chack_agent
    import chack_tools.researcher_administrator_agent as admin_mod

    job_id = "research-job-workspace-harvest"
    tool_name = "scientific_research"

    class FakeTool:
        def __init__(self, name):
            self.name = name

    class FakeChack:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            admin_mod._async_job_store(
                job_id,
                {
                    "job_id": job_id,
                    "created_at": time.time(),
                    "evidence_dir": str(tmp_path),
                    "tasks": {
                        "task-0": {
                            "task_id": "task-0",
                            "researcher": "scientific",
                            "researcher_tool": tool_name,
                            "status": "done",
                            "result": {
                                "researcher_tool": tool_name,
                                "parsed_response": {
                                    "research_worked": True,
                                    "failure_reason": "",
                                    "final_research_review": "workspace-harvested review",
                                    "tool_call_counts": {"search_europe_pmc": 2},
                                    "total_tool_calls": 2,
                                },
                            },
                        }
                    },
                },
            )
            return SimpleNamespace(
                output=json.dumps(
                    {
                        "research_worked": True,
                        "failure_reason": "",
                        "administrator_conclusions": (
                            "The workspace-harvested scientific review provides substantive evidence "
                            "for the administrator synthesis and records the relevant limitations."
                        ),
                    },
                    separators=(",", ":"),
                ),
                tool_counts=Counter(),
                all_steps=[],
            )

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
        required_researchers=["scientific"],
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda _enabled, **_kwargs: [FakeTool("task_steps_manager")],
    )

    output = helper._run_single(
        "Research a scientific question with primary sources, exact dates, contradictions, and evidence gaps. " * 12,
        {
            "max_turns": 20,
            "max_runtime_minutes": 0,
            "remaining_runtime_minutes": 0,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "memory_max_messages": 8,
            "memory_reset_to_messages": 8,
            "session_id": "workspace-harvest-test",
            "research_master_dir": str(tmp_path),
        },
        save_artifacts=False,
    )
    payload = json.loads(output)

    assert payload["research_worked"] is True
    assert payload["required_researchers_satisfied"] is True
    assert payload["researcher_call_counts"] == {tool_name: 1}
    assert payload["researcher_responses"][0]["researcher_tool"] == tool_name


def test_administrator_system_prompt_is_compact_and_has_one_first_wave_policy():
    prompt = _ADMINISTRATOR_SYSTEM_PROMPT

    assert len(prompt) < 5_000
    assert "Researchers are blind to one another" in prompt
    assert "Repeat a researcher only for a specific unresolved source gap or contradiction" in prompt
    assert "normally run 3-5" not in prompt
    assert "key_artifacts" not in prompt
    assert "CHACK_RESEARCH_DATA_DIR" not in prompt


def test_administrator_registered_only_when_enabled():
    off = AgentsToolset(ToolsConfig(scientific_enabled=True), model_provider="openai", default_model="m")
    assert "researcher_administrator" not in _tool_names(off.tools)

    on = AgentsToolset(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        default_model="m",
    )
    assert "researcher_administrator" in _tool_names(on.tools)


def test_chatgpt_researchers_are_structurally_async_only():
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        deepchatgpt_enabled=True,
        prochatgpt_enabled=True,
        chatgptxhigh_enabled=True,
        chatgpt_cdp_url="http://127.0.0.1:9226",
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["deepchatgpt", "prochatgpt", "chatgptxhigh"],
    )

    inner = _tool_names(helper._build_subagent_tools(helper._enabled_researchers()))
    assert "start_researchers_async" in inner
    assert "poll_researchers_async" in inner
    assert "list_researcher_jobs" in inner
    assert "get_researcher_task" in inner
    assert "get_researcher_result" in inner
    assert "cancel_researcher_task" in inner
    assert "retry_researcher_task" in inner
    assert "deepchatgpt_researcher" not in inner
    assert "prochatgpt_researcher" not in inner
    assert "chatgptxhigh" not in inner
    assert "run_researchers_batch" not in inner
    # Whole-job cancellation remains exposed even when the selected researchers
    # are browser-backed. The administrator must retain an explicit control-plane
    # path; browser jobs only reject impatient/model-driven cancellation in their
    # worker policy, not the management capability itself.
    assert "cancel_researchers_async" in inner


def test_administrator_allowlist_force_enables_researchers():
    # business is not globally enabled but the allowlist must still grant it.
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        researcher_administrator_researchers=["scientific", "business"],
        scientific_enabled=True,
    )
    helper = ResearcherAdministratorAgentTool(cfg, model_provider="openai", fallback_model="m", researchers=["scientific", "business"])
    assert helper._enabled_researchers() == ["scientific", "business"]

    inner = _tool_names(helper._build_subagent_tools(helper._enabled_researchers()))
    assert inner == {
        "run_researchers_batch",
        "start_researchers_async",
        "poll_researchers_async",
        "cancel_researchers_async",
        "list_researcher_jobs",
        "get_researcher_task",
        "get_researcher_result",
        "cancel_researcher_task",
        "retry_researcher_task",
        "list_research_artifacts",
        "read_research_artifact",
        "grep_research_artifacts",
        "delete_research_artifact",
        "register_research_artifact",
        "task_steps_manager",
    }
    # The administrator never gets low-level search tools or other orchestrators.
    assert "search_arxiv" not in inner
    assert "subchack_researcher" not in inner
    assert "researcher_administrator" not in inner


def test_administrator_artifact_tools_stay_pinned_when_child_context_changes(tmp_path):
    master_dir = tmp_path / "master"
    child_dir = master_dir / "websearcher"
    wrong_dir = tmp_path / "wrong-context"
    master_dir.mkdir()
    child_dir.mkdir()
    wrong_dir.mkdir()
    (master_dir / "master-evidence.txt").write_text("master evidence", encoding="utf-8")
    (child_dir / "child-evidence.txt").write_text("child evidence", encoding="utf-8")
    (wrong_dir / "wrong-evidence.txt").write_text("wrong context", encoding="utf-8")

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    list_tool = next(
        tool for tool in helper._build_subagent_tools(["scientific"], artifact_root=str(master_dir))
        if getattr(tool, "name", "") == "list_research_artifacts"
    )

    context_tokens = set_research_artifact_context(str(wrong_dir), str(wrong_dir))
    try:
        from agents.tool_context import ToolContext
        from agents.usage import Usage

        raw_args = json.dumps({"glob": "*", "max_results": 20})
        tool_context = ToolContext(
            context=None,
            usage=Usage(),
            tool_name="list_research_artifacts",
            tool_call_id="pinned-root-test",
            tool_arguments=raw_args,
        )
        output = asyncio.run(
            list_tool.on_invoke_tool(
                tool_context,
                raw_args,
            )
        )
    finally:
        reset_research_artifact_context(context_tokens)

    assert "master-evidence.txt" in output
    assert "wrong-evidence.txt" not in output


def test_async_tools_capture_explicit_administrator_artifact_root(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    class FakeResearcher:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "root-capture test",
                    "tool_call_counts": {"search_europe_pmc": 1},
                    "total_tool_calls": 1,
                }
            )

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    root = str(tmp_path / "administrator-root")
    tools = helper._build_async_tools(
        {"scientific_research": FakeResearcher()},
        ["scientific"],
        artifact_root=root,
    )
    start_tool = next(tool for tool in tools if getattr(tool, "name", "") == "start_researchers_async")
    prompt = "Investigate this scientific question with primary sources, exact dates, contradictions, and evidence gaps. " * 12
    output = helper._invoke_tool_sync(
        start_tool,
        {
            "requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}]),
            "save_artifacts": True,
            "max_parallel": 1,
        },
    )
    payload = json.loads(output)
    job_id = payload["job_id"]
    try:
        assert admin_mod._async_job_get(job_id)["evidence_dir"] == root
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)


def test_sync_researcher_batch_deadline_returns_without_executor_shutdown_wait():
    class SlowResearcher:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            # Deliberately ignores the cooperative event: the parent must still
            # return a terminal timeout instead of waiting for this coroutine.
            await asyncio.sleep(4)
            return "late output"

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            researcher_administrator_enabled=True,
            scientific_enabled=True,
            researcher_administrator_child_timeout_seconds=1,
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    batch = helper._build_batch_tool(
        {"scientific_research": SlowResearcher()},
        ["scientific"],
    )
    prompt = "Investigate the scientific claim with primary sources, exact dates, contradictions, and evidence gaps. " * 12
    started = time.monotonic()
    output = helper._invoke_tool_sync(
        batch,
        {
            "requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}]),
            "save_artifacts": False,
            "max_parallel": 1,
        },
    )
    elapsed = time.monotonic() - started
    payload = json.loads(output)

    assert elapsed < 3
    assert payload["batch_worked"] is False
    assert payload["batch_complete"] is False
    assert payload["errors"][0]["status"] == "deadline_exceeded"
    assert "deadline" in payload["errors"][0]["error"].lower()
    task = payload["tasks"][0]
    assert task["status"] == "deadline_exceeded"
    assert task["started_at"] is not None
    assert task["last_progress_at"] is not None
    assert task["deadline_at"] > task["started_at"]
    assert task["artifact_count"] == 0
    assert task["failure_reason"]
    assert task["execution_active"] is True


def test_sync_batch_uses_exported_deadline_when_mcp_contextvar_is_missing(monkeypatch):
    """The MCP reconstruction must retain the administrator's reserved window."""
    class SlowResearcher:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            await asyncio.sleep(4)
            return "late output"

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            researcher_administrator_enabled=True,
            scientific_enabled=True,
            researcher_administrator_child_timeout_seconds=30,
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    monkeypatch.setenv(
        "CHACK_RESEARCHER_ADMIN_RESEARCHER_DEADLINE_EPOCH",
        str(time.time() + 1.2),
    )
    batch = helper._build_batch_tool(
        {"scientific_research": SlowResearcher()},
        ["scientific"],
    )
    prompt = "Investigate the scientific claim with primary sources, exact dates, contradictions, and evidence gaps. " * 12
    payload = json.loads(
        helper._invoke_tool_sync(
            batch,
            {
                "requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}]),
                "save_artifacts": False,
                "max_parallel": 1,
            },
        )
    )

    assert payload["child_timeout_seconds"] == 1
    assert payload["batch_worked"] is False
    assert payload["errors"][0]["status"] == "deadline_exceeded"


def test_daemon_executor_does_not_join_blocked_worker_at_process_exit():
    code = "\n".join(
        [
            "import threading",
            "from chack_tools.researcher_administrator_agent import _DaemonThreadPoolExecutor",
            "release = threading.Event()",
            "executor = _DaemonThreadPoolExecutor(max_workers=1)",
            "executor.submit(release.wait, 60)",
            "executor.shutdown(wait=False, cancel_futures=True)",
            "print('parent-returned')",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        # Importing the full agents stack can take several seconds under parallel CI
        # load. Keep this comfortably below the 60-second blocked worker while
        # avoiding a harness-only timeout before interpreter shutdown is observed.
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "parent-returned" in completed.stdout


def test_isolated_researcher_ignoring_term_is_killed_with_descendant_group(tmp_path):
    """Exercise the real child supervisor, process group, TERM, and KILL path."""
    import chack_tools.researcher_administrator_agent as admin_mod

    root = tmp_path / "term-kill-acceptance"
    root.mkdir()
    child_started = root / "child.started"
    descendant_pid_file = root / "descendant.pid"
    descendant_pgid_file = root / "descendant.pgid"

    class IgnoringTermResearcher:
        name = "scientific_research"

        def __init__(self, marker_root):
            self.marker_root = str(marker_root)

        async def on_invoke_tool(self, _ctx, _raw_args):
            import subprocess
            import sys

            descendant_code = "\n".join(
                [
                    "import os, signal, sys, time",
                    "from pathlib import Path",
                    "root = Path(sys.argv[1])",
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                    "(root / 'descendant.pid').write_text(str(os.getpid()))",
                    "(root / 'descendant.pgid').write_text(str(os.getpgrp()))",
                    "while True: time.sleep(1)",
                ]
            )
            subprocess.Popen(
                [sys.executable, "-c", descendant_code, self.marker_root],
                close_fds=True,
            )
            Path(self.marker_root, "child.started").write_text(str(os.getpid()), encoding="utf-8")
            while True:
                await asyncio.sleep(1)

    cancel_event = threading.Event()
    cancel_token = set_cancellation_event(cancel_event)
    trigger = threading.Thread(
        target=lambda: (
            _wait_for_file(child_started, 10)
            and _wait_for_file(descendant_pid_file, 10)
            and _wait_for_file(descendant_pgid_file, 10)
            and request_cancel(cancel_event)
        ),
        daemon=True,
    )
    started_at = time.monotonic()
    try:
        trigger.start()
        result = admin_mod._run_researcher_in_process(
            IgnoringTermResearcher(root),
            {"prompt": "Investigate this claim with primary evidence and explicit limitations."},
            evidence_dir=str(root),
            cancel_event=cancel_event,
            termination_grace_seconds=0.25,
        )
    finally:
        request_cancel(cancel_event)
        trigger.join(timeout=2)
        reset_cancellation_event(cancel_token)

    elapsed = time.monotonic() - started_at
    termination = result["termination"]
    descendant_pid = int(descendant_pid_file.read_text(encoding="utf-8"))
    descendant_pgid = int(descendant_pgid_file.read_text(encoding="utf-8"))

    assert result["cancelled"] is True
    assert termination["term_sent"] is True
    assert termination["kill_sent"] is True
    assert termination["process_group_id"] == descendant_pgid
    assert termination["process_alive_after"] is False
    assert termination["descendant_pids_after"] == []
    assert elapsed >= 0.20

    gone_deadline = time.monotonic() + 5
    while Path(f"/proc/{descendant_pid}").exists() and time.monotonic() < gone_deadline:
        time.sleep(0.05)
    assert not Path(f"/proc/{descendant_pid}").exists()
    assert admin_mod._live_process_group_members(descendant_pgid) == []


def test_administrator_empty_allowlist_uses_globally_enabled():
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        scientific_enabled=True,
        webresearcher_enabled=True,  # legacy alias for websearcher
    )
    helper = ResearcherAdministratorAgentTool(cfg, model_provider="openai", fallback_model="m")
    assert set(helper._enabled_researchers()) == {"scientific", "websearcher"}


def test_required_researchers_must_be_enabled_for_administrator():
    cfg = ToolsConfig(researcher_administrator_enabled=True, websearcher_enabled=True)
    with pytest.raises(ValueError, match="required_researchers must be enabled"):
        ResearcherAdministratorAgentTool(
            cfg,
            model_provider="openai",
            fallback_model="m",
            researchers=["websearcher"],
            required_researchers=["prochatgpt"],
        )


def test_administrator_tracks_required_researchers():
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        websearcher_enabled=True,
        prochatgpt_enabled=True,
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["websearcher", "prochatgpt"],
        required_researchers=["websearcher", "prochatgpt"],
    )
    assert helper.required_researchers == ["websearcher", "prochatgpt"]


def test_administrator_run_accounting_is_isolated_across_concurrent_calls(monkeypatch):
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        deepchatgpt_enabled=True,
        prochatgpt_enabled=True,
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["deepchatgpt", "prochatgpt"],
    )
    barrier = threading.Barrier(2)
    observed: dict[str, dict[str, int]] = {}

    def fake_scoped(prompt, ctx, save_artifacts=False):
        short = "prochatgpt" if prompt == "run-a" else "deepchatgpt"
        helper._launched_researcher_counts[short] += 1
        helper._launched_async_job_ids.append(f"job-{short}")
        barrier.wait(timeout=5)
        observed[prompt] = dict(helper._launched_researcher_counts)
        assert helper._launched_async_job_ids == [f"job-{short}"]
        return prompt

    monkeypatch.setattr(helper, "_run_single_scoped", fake_scoped)
    outputs: dict[str, str] = {}
    threads = [
        threading.Thread(target=lambda: outputs.__setitem__("run-a", helper._run_single("run-a", {}))),
        threading.Thread(target=lambda: outputs.__setitem__("run-b", helper._run_single("run-b", {}))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert outputs == {"run-a": "run-a", "run-b": "run-b"}
    assert observed == {
        "run-a": {"prochatgpt": 1},
        "run-b": {"deepchatgpt": 1},
    }


def test_administrator_error_output_keeps_exact_attempted_researcher_count(monkeypatch, tmp_path):
    import chack_agent

    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        prochatgpt_enabled=True,
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["prochatgpt"],
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda _enabled, **_kwargs: [SimpleNamespace(name="task_steps_manager")],
    )

    class FakeChack:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            helper._launched_researcher_counts["prochatgpt"] += 1
            return SimpleNamespace(
                output="ERROR: synthetic administrator failure",
                tool_counts=Counter(),
                all_steps=[],
            )

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)
    output = helper._run_single(
        "Research an evidence-heavy medical question with exact sources. " * 12,
        {
            "max_turns": 20,
            "max_runtime_minutes": 0,
            "remaining_runtime_minutes": 0,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "research_master_dir": str(tmp_path / "admin-error"),
        },
        save_artifacts=True,
    )
    payload = json.loads(output)

    assert payload["research_worked"] is False
    assert payload["researcher_call_counts"] == {"prochatgpt_researcher": 1}
    assert payload["total_researcher_calls"] == 1
    assert "synthetic administrator failure" in payload["failure_reason"]


def test_administrator_deduplicates_overlapping_launch_count_observations(monkeypatch, tmp_path):
    import chack_agent

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda _enabled, **_kwargs: [SimpleNamespace(name="task_steps_manager")],
    )

    class FakeChack:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            # The same physical launch is visible both in model telemetry and in
            # the pre-spawn reservation ledger. It must still be reported once.
            helper._launched_researcher_counts["scientific"] += 1
            return SimpleNamespace(
                output=json.dumps(
                    {
                        "research_worked": True,
                        "failure_reason": "",
                        "administrator_conclusions": (
                            "One supervised scientific worker completed; overlapping telemetry "
                            "sources describe that same launch rather than separate calls."
                        ),
                    }
                ),
                tool_counts=Counter({"scientific_research": 1}),
                all_steps=[],
            )

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)
    output = helper._run_single(
        "Research a bounded scientific question using primary evidence and explicit caveats. " * 12,
        {
            "max_turns": 20,
            "max_runtime_minutes": 0,
            "remaining_runtime_minutes": 0,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "research_master_dir": str(tmp_path / "count-observations"),
        },
        save_artifacts=True,
    )
    payload = json.loads(output)

    assert payload["researcher_call_counts"] == {"scientific_research": 1}
    assert payload["total_researcher_calls"] == 1


def test_cancelled_async_then_successful_replacement_has_exact_terminal_accounting(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    output_dir = tmp_path / "researcher_outputs"
    output_dir.mkdir()
    cancelled = {
        "research_worked": False,
        "failure_reason": "Researcher did not return parseable JSON.",
        "overall_summary": "The cancelled attempt produced no usable response.",
        "findings": [],
        "gaps": ["The worker was deliberately cancelled before evidence collection."],
        "open_topics": [],
        "full_research_review": "",
        "researcher_tool": "scientific_research",
    }
    successful = {
        "research_worked": True,
        "failure_reason": "",
        "overall_summary": "The replacement completed with primary scientific evidence.",
        "findings": [
            {
                "claim": "The replacement researcher completed the requested evidence review.",
                "summary": (
                    "It returned substantive primary-source findings after the intentionally "
                    "cancelled lifecycle probe had physically unwound."
                ),
            }
        ],
        "gaps": [],
        "open_topics": [],
        "full_research_review": "Primary-source review with methods, contradictions, and caveats. " * 20,
        "researcher_tool": "scientific_research",
        "tool_call_counts": {"search_europe_pmc": 3},
        "total_tool_calls": 3,
    }
    (output_dir / "async_cancelled_scientific_research.json").write_text(
        json.dumps(cancelled), encoding="utf-8"
    )
    (output_dir / "async_replacement_scientific_research.json").write_text(
        json.dumps(successful), encoding="utf-8"
    )
    steps = [
        {
            "tool": "poll_researchers_async",
            "output": json.dumps(
                {
                    "complete": True,
                    "outputs_included": False,
                    "tasks": [
                        {
                            "task_id": "task-cancelled",
                            "researcher": "scientific",
                            "researcher_tool": "scientific_research",
                            "status": "cancelled",
                            "execution_active": False,
                            "latest_action": "cancelled; worker unwound",
                            "tool_call_counts": {"scientific_research": 1},
                            "total_tool_calls": 1,
                        }
                    ],
                }
            ),
        },
        {
            "tool": "poll_researchers_async",
            "output": json.dumps(
                {
                    "complete": True,
                    "outputs_included": False,
                    "tasks": [
                        {
                            "task_id": "task-replacement",
                            "researcher": "scientific",
                            "researcher_tool": "scientific_research",
                            "status": "done",
                            "execution_active": False,
                            "latest_action": "done",
                            "result_available": True,
                            "tool_call_counts": {"search_europe_pmc": 3},
                            "total_tool_calls": 3,
                        }
                    ],
                }
            ),
        },
    ]

    responses = admin_mod._researcher_responses_from_async_output_files(str(tmp_path))
    payload = json.loads(
        admin_mod.finalize_researcher_administrator_output(
            json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "administrator_conclusions": (
                        "The replacement supplied substantive evidence after the lifecycle probe "
                        "was cancelled and physically settled, so the final synthesis is supported."
                    ),
                }
            ),
            evidence_dir=str(tmp_path),
            save_artifacts=True,
            researcher_responses=responses,
            researcher_failures=[],
            tool_counts=Counter({"scientific_research": 2}),
            steps=steps,
            required_researchers=["scientific"],
        )
    )

    assert payload["research_worked"] is True
    assert payload["required_researchers_satisfied"] is True
    assert len(payload["researcher_responses"]) == 1
    assert payload["researcher_responses"][0]["research_worked"] is True
    assert payload["researcher_failures"] == [
        {
            "researcher_tool": "scientific_research",
            "status": "cancelled",
            "failure_reason": "cancelled; worker unwound",
            "task_id": "task-cancelled",
            "tool_call_counts": {"scientific_research": 1},
            "total_tool_calls": 1,
        }
    ]
    assert payload["researcher_call_counts"] == {"scientific_research": 2}
    assert payload["total_researcher_calls"] == 2
    assert payload["researcher_tool_call_counts"] == {"search_europe_pmc": 3}


def test_administrator_capability_map_lists_internal_researcher_tools():
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        researcher_administrator_researchers=["websearcher", "scientific"],
        scientific_enabled=True,
        websearcher_enabled=True,
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["websearcher", "scientific"],
    )

    lines = helper._researcher_capability_lines(helper._enabled_researchers())
    text = "\n".join(lines)

    assert "- websearcher via `run_researchers_batch` (request `websearcher_research`):" in text
    assert "fetch_url_text" in text
    assert "web_archive_search" in text
    assert "- scientific via `run_researchers_batch` (request `scientific_research`):" in text
    assert "search_arxiv" in text
    assert "download_pmc_full_text" in text


def test_administrator_prioritizes_deep_and_pro_chatgpt_before_xhigh():
    instruction = ResearcherAdministratorAgentTool._chatgpt_priority_instruction(
        ["websearcher", "chatgptxhigh", "prochatgpt", "deepchatgpt"]
    )

    assert "`deepchatgpt_researcher`" in instruction
    assert "`prochatgpt_researcher`" in instruction
    assert "start every enabled one immediately" in instruction
    assert "`start_researchers_async`" in instruction
    assert "chatgptxhigh" not in instruction


def test_administrator_prioritizes_xhigh_when_deep_and_pro_are_unavailable():
    instruction = ResearcherAdministratorAgentTool._chatgpt_priority_instruction(
        ["websearcher", "chatgptxhigh"]
    )

    assert "neither `deepchatgpt_researcher` nor `prochatgpt_researcher` is available" in instruction
    assert "`chatgptxhigh` is enabled" in instruction
    assert "start it immediately" in instruction
    assert "`start_researchers_async`" in instruction


def test_administrator_has_no_chatgpt_priority_without_browser_researchers():
    assert ResearcherAdministratorAgentTool._chatgpt_priority_instruction(
        ["websearcher", "scientific"]
    ) == ""


def test_administrator_forces_try_harder_for_child_researchers(monkeypatch):
    import chack_tools.agents_toolset as at

    captured = {}

    class FakeTool:
        def __init__(self, name):
            self.name = name

    class SpyToolset:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.tools = [
                FakeTool("scientific_research"),
                FakeTool("task_steps_manager"),
            ]

    monkeypatch.setattr(at, "AgentsToolset", SpyToolset)

    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        researcher_administrator_researchers=["scientific"],
        scientific_enabled=True,
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
        self_critique_enabled=False,
        self_critique_rounds=0,
    )

    helper._build_subagent_tools(helper._enabled_researchers())

    assert helper.self_critique_enabled is False
    assert helper.self_critique_rounds == 0
    assert captured["self_critique_enabled"] is True
    assert captured["self_critique_rounds"] == 1


def test_administrator_preserves_configured_child_try_harder_rounds(monkeypatch):
    import chack_tools.agents_toolset as at

    captured = {}

    class FakeTool:
        def __init__(self, name):
            self.name = name

    class SpyToolset:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.tools = [
                FakeTool("scientific_research"),
                FakeTool("task_steps_manager"),
            ]

    monkeypatch.setattr(at, "AgentsToolset", SpyToolset)

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
        self_critique_enabled=True,
        self_critique_rounds=3,
    )

    helper._build_subagent_tools(helper._enabled_researchers())

    assert captured["self_critique_enabled"] is True
    assert captured["self_critique_rounds"] == 3


def test_administrator_prompt_includes_compact_tool_map_and_usage_audit(monkeypatch):
    import chack_agent

    captured = {}

    class FakeTool:
        def __init__(self, name):
            self.name = name

    class FakeChack:
        def __init__(self, config):
            captured["config"] = config

        def run(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output=json.dumps(
                    {
                        "research_worked": True,
                        "failure_reason": "",
                        "administrator_conclusions": "ok",
                    },
                    separators=(",", ":"),
                ),
                tool_counts=Counter(),
                all_steps=[],
            )

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            researcher_administrator_enabled=True,
            researcher_administrator_max_tools_used=3,
            websearcher_enabled=True,
            scientific_enabled=True,
            prochatgpt_enabled=True,
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=["websearcher", "scientific", "prochatgpt"],
        self_critique_rounds=2,
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda enabled, **_kwargs: [
            FakeTool("websearcher_research"),
            FakeTool("scientific_research"),
            FakeTool("prochatgpt_researcher"),
            FakeTool("task_steps_manager"),
            FakeTool("run_researchers_batch"),
            FakeTool("start_researchers_async"),
            FakeTool("poll_researchers_async"),
            FakeTool("cancel_researchers_async"),
        ],
    )
    monkeypatch.setattr(
        helper,
        "_researcher_capability_lines",
        lambda enabled: [
            "- websearcher via `websearcher_research`: fetch_url_text, web_archive_search",
            "- scientific via `scientific_research`: search_arxiv, download_pmc_full_text",
        ],
    )

    prompt = (
        "Research a complex safety controversy across open web and scientific literature. "
        "Include primary sources, source preservation, disagreements, exact entities, dates, "
        "and evidence gaps. " * 8
    )
    out = helper._run_single(
        prompt,
        {
            "max_turns": 20,
            "max_runtime_minutes": 0,
            "remaining_runtime_minutes": 0,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "memory_max_messages": 8,
            "memory_reset_to_messages": 8,
            "session_id": "admin-prompt-test",
        },
        save_artifacts=False,
    )
    payload = json.loads(out)
    sent_prompt = captured["text"]

    # This fixture intentionally has no terminal researcher response and uses
    # a placeholder synthesis. The fail-closed gate must reject it; the test
    # itself is about the prompt contract below.
    assert payload["research_worked"] is False
    assert captured["max_tools_used_override"] == 20
    assert "Researcher-call budget: 3 launches" in sent_prompt
    assert "management polls/status do not count" in sent_prompt
    assert "Researcher capabilities:" in sent_prompt
    assert "- websearcher via `websearcher_research`: fetch_url_text, web_archive_search" in sent_prompt
    assert "- scientific via `scientific_research`: search_arxiv, download_pmc_full_text" in sent_prompt
    assert "Compare `tool_call_counts` with the capabilities above" in sent_prompt
    assert "focused follow-up only for a material missing source/tool family" in sent_prompt
    assert "try-harder self-critique for 2 round(s)" in sent_prompt
    assert "start_researchers_async" in sent_prompt
    assert "poll_researchers_async" in sent_prompt
    assert "ChatGPT browser 300-600s" in sent_prompt
    assert "`prochatgpt_researcher` is enabled" in sent_prompt
    assert "start every enabled one immediately" in sent_prompt
    assert "### Evidence collection" not in sent_prompt
    assert len(sent_prompt) - len(prompt) < 4_000


def test_administrator_prompt_exposes_research_and_synthesis_time_windows(monkeypatch):
    import chack_agent

    captured = {}

    class FakeChack:
        def __init__(self, _config):
            pass

        def run(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output=json.dumps(
                    {
                        "research_worked": True,
                        "failure_reason": "",
                        "administrator_conclusions": "ok",
                    }
                ),
                tool_counts=Counter(),
                all_steps=[],
            )

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    monkeypatch.setattr(helper, "_build_subagent_tools", lambda _enabled, **_kwargs: [SimpleNamespace(name="scientific_research")])
    monkeypatch.setattr(helper, "_researcher_capability_lines", lambda _enabled: [])

    helper._run_single(
        "Research this topic with primary sources, contradictions, dates, and evidence gaps. " * 12,
        {
            "max_turns": 20,
            "max_runtime_minutes": 60,
            "remaining_runtime_minutes": 60,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "memory_max_messages": 8,
            "memory_reset_to_messages": 8,
            "session_id": "admin-time-budget-test",
        },
    )

    assert "administrator hard cap is 60 minutes" in captured["text"]
    assert "researcher phase has a hard stop after 55 minutes" in captured["text"]
    assert "leaving 5 minutes reserved for your own synthesis" in captured["text"]


def test_useful_evidence_rejects_placeholder_full_review_even_with_artifacts(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    evidence = tmp_path / "evidence.txt"
    evidence.write_text("real source evidence", encoding="utf-8")
    response = {
        "research_worked": True,
        "failure_reason": "",
        "overall_summary": "A substantive bounded summary.",
        "findings": [{"claim": "A concrete claim with enough detail", "summary": "A substantive finding summary with caveats and provenance."}],
        "full_research_review": "placeholder",
        "evidence_data_path": str(tmp_path),
        "key_artifacts": [{"filename": evidence.name}],
    }

    assert admin_mod._response_has_useful_evidence(response) is False


def test_administrator_timeout_returns_preserved_artifact_paths(monkeypatch, tmp_path):
    import chack_agent

    class FakeTool:
        def __init__(self, name):
            self.name = name

    class FakeChack:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            raise TimeoutError("synthetic admin timeout")

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, websearcher_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["websearcher"],
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda enabled, **_kwargs: [
            FakeTool("websearcher_research"),
            FakeTool("task_steps_manager"),
            FakeTool("run_researchers_batch"),
            FakeTool("start_researchers_async"),
            FakeTool("poll_researchers_async"),
            FakeTool("cancel_researchers_async"),
        ],
    )

    prompt = "Research timeout artifact preservation with enough detail and concrete sources. " * 10
    out = helper._run_single(
        prompt,
        {
            "max_turns": 20,
            "max_runtime_minutes": 0,
            "remaining_runtime_minutes": 0,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "memory_max_messages": 8,
            "memory_reset_to_messages": 8,
            "session_id": "admin-timeout-test",
            "research_master_dir": str(tmp_path),
        },
        save_artifacts=True,
    )
    payload = json.loads(out)

    assert payload["research_worked"] is False
    assert "TimeoutError: synthetic admin timeout" in payload["failure_reason"]
    assert payload["evidence_data_path"] == str(tmp_path)
    assert payload["output_files"]["administrator_output"] == "admin_output.json"
    saved = json.loads((tmp_path / "admin_output.json").read_text(encoding="utf-8"))
    assert saved["research_worked"] is False
    assert saved["evidence_data_path"] == str(tmp_path)


def test_administrator_timeout_harvests_completed_async_researchers(monkeypatch, tmp_path):
    import chack_agent
    import chack_tools.researcher_administrator_agent as admin_mod

    job_id = "research-job-timeout-harvest"
    admin_mod._async_job_store(
        job_id,
        {
            "job_id": job_id,
            "created_at": time.time(),
            "tasks": {
                "task-1": {
                    "task_id": "task-1",
                    "researcher": "scientific",
                    "researcher_tool": "scientific_research",
                    "status": "done",
                    "result": {
                        "researcher_tool": "scientific_research",
                        "parsed_response": {
                            "research_worked": True,
                            "failure_reason": "",
                            "final_research_review": "science review",
                            "tool_call_counts": {"search_europe_pmc": 2},
                            "total_tool_calls": 2,
                        },
                    },
                }
            },
        },
    )

    class FakeTool:
        def __init__(self, name):
            self.name = name

    class FakeChack:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            helper._launched_async_job_ids = [job_id]
            raise TimeoutError("synthetic admin timeout")

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda enabled, **_kwargs: [
            FakeTool("scientific_research"),
            FakeTool("task_steps_manager"),
            FakeTool("start_researchers_async"),
            FakeTool("poll_researchers_async"),
            FakeTool("cancel_researchers_async"),
        ],
    )

    out = helper._run_single(
        "Research timeout harvesting with source-backed scientific evidence. " * 10,
        {
            "max_turns": 20,
            "max_runtime_minutes": 0,
            "remaining_runtime_minutes": 0,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "memory_max_messages": 8,
            "memory_reset_to_messages": 8,
            "session_id": "admin-timeout-harvest-test",
            "research_master_dir": str(tmp_path),
        },
        save_artifacts=True,
    )
    payload = json.loads(out)

    assert payload["research_worked"] is False
    assert len(payload["researcher_responses"]) == 1
    assert payload["researcher_responses"][0]["researcher_tool"] == "scientific_research"
    assert payload["researcher_tool_call_counts"] == {"search_europe_pmc": 2}
    assert payload["researcher_call_counts"] == {"scientific_research": 1}


def test_administrator_timeout_harvests_persisted_async_researcher_files(monkeypatch, tmp_path):
    import chack_agent

    class FakeTool:
        def __init__(self, name):
            self.name = name

    class FakeChack:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            output_dir = tmp_path / "researcher_outputs"
            output_dir.mkdir(parents=True)
            (output_dir / "async_task-1_legal_research.json").write_text(
                json.dumps(
                    {
                        "research_worked": True,
                        "failure_reason": "",
                        "final_research_review": "legal review",
                        "researcher_tool": "legal_research",
                        "tool_call_counts": {"boe_law_search": 1},
                        "total_tool_calls": 1,
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            raise TimeoutError("synthetic admin timeout")

    monkeypatch.setattr(chack_agent, "Chack", FakeChack)

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, legal_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["legal"],
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda enabled, **_kwargs: [
            FakeTool("legal_research"),
            FakeTool("task_steps_manager"),
            FakeTool("start_researchers_async"),
            FakeTool("poll_researchers_async"),
            FakeTool("cancel_researchers_async"),
        ],
    )

    out = helper._run_single(
        "Research timeout harvesting from persisted async output files. " * 10,
        {
            "max_turns": 20,
            "max_runtime_minutes": 0,
            "remaining_runtime_minutes": 0,
            "max_cost_usd": 0,
            "remaining_cost_usd": 0,
            "memory_max_messages": 8,
            "memory_reset_to_messages": 8,
            "session_id": "admin-timeout-file-harvest-test",
            "research_master_dir": str(tmp_path),
        },
        save_artifacts=True,
    )
    payload = json.loads(out)

    assert payload["research_worked"] is False
    assert len(payload["researcher_responses"]) == 1
    assert payload["researcher_responses"][0]["researcher_tool"] == "legal_research"
    assert payload["researcher_tool_call_counts"] == {"boe_law_search": 1}
    assert payload["researcher_call_counts"] == {"legal_research": 1}


def test_administrator_managed_researchers_get_artifact_file_tools():
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        researcher_administrator_researchers=["websearcher", "scientific", "cli"],
        websearcher_enabled=True,
        scientific_enabled=True,
        cli_enabled=True,
        websearcher_fetch_url_text_enabled=True,
        scientific_arxiv_enabled=True,
        cli_exec_enabled=True,
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["websearcher", "scientific", "cli"],
    )

    for short in helper._enabled_researchers():
        names = {helper._name_of_tool(tool) for tool in helper._build_capability_tools_for_researcher(short)}
        assert "list_research_artifacts" in names, short
        assert "read_research_artifact" in names, short
        assert "grep_research_artifacts" in names, short
        assert "delete_research_artifact" in names, short


def test_administrator_async_research_tools_start_and_poll():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )

    class FakeResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, ctx, raw_args):
            payload = json.loads(raw_args)
            assert payload["save_artifacts"] is False
            run_with_tool_logging("search_arxiv", {"query": "async science"}, lambda: "ok")
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "async science review",
                    "tool_call_counts": {"search_arxiv": 1},
                    "total_tool_calls": 1,
                },
                separators=(",", ":"),
            )

    tools = helper._build_async_tools({"scientific_research": FakeResearchTool()}, ["scientific"])
    by_name = {tool.name: tool for tool in tools}
    long_prompt = "Research async scientific evidence with exact sources and caveats. " * 12

    start = helper._invoke_tool_sync(
        by_name["start_researchers_async"],
        {
            "requests_json": json.dumps(
                [{"researcher": "scientific", "prompt": long_prompt}],
                separators=(",", ":"),
            ),
            "save_artifacts": False,
        },
    )
    started = json.loads(start)
    assert started["async_started"] is True
    assert started["max_parallel"] == 1
    assert "up to 1 concurrent workers" in started["next_step"]
    job_id = started["job_id"]

    poll_started = time.monotonic()
    polled = helper._invoke_tool_sync(
        by_name["poll_researchers_async"],
        {"job_id": job_id, "include_outputs": False, "wait_seconds": 900},
    )
    poll_elapsed = time.monotonic() - poll_started
    payload = json.loads(polled)

    assert payload["complete"] is True
    assert poll_elapsed < 2
    assert "next_step" in payload
    assert payload["requested_wait_seconds"] == 900
    assert 0 <= payload["waited_seconds"] <= poll_elapsed + 0.1
    task = payload["tasks"][0]
    assert task["status"] == "done"
    assert "idle_seconds" in task
    assert task["tool_call_counts"] == {"search_arxiv": 1}
    assert task["total_tool_calls"] == 1
    assert any(event["tool"] == "search_arxiv" for event in task["recent_events"])


def test_async_researchers_keep_the_administrator_artifact_context_isolated(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            researcher_administrator_enabled=True,
            news_media_enabled=True,
            chatgptxhigh_enabled=True,
            chatgpt_cdp_url="http://127.0.0.1:9226",
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=["news_media", "chatgptxhigh"],
    )
    master_dir = str(tmp_path / "administrator-workspace")
    class FakeResearchTool:
        def __init__(self, name, marker_root):
            self.name = name
            self.marker_root = str(marker_root)

        async def on_invoke_tool(self, ctx, raw_args):
            from chack_tools.research_artifacts import (
                research_artifacts_master_root,
                research_artifacts_root,
            )

            record = {
                "name": self.name,
                "data_dir": research_artifacts_root(),
                "master_dir": research_artifacts_master_root(),
            }
            record_dir = Path(self.marker_root) / "ipc_seen"
            record_dir.mkdir(parents=True, exist_ok=True)
            (record_dir / f"{self.name}.json").write_text(
                json.dumps(record, separators=(",", ":")),
                encoding="utf-8",
            )
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": f"{self.name} review",
                    "tool_call_counts": {},
                    "total_tool_calls": 0,
                },
                separators=(",", ":"),
            )

    tools = helper._build_async_tools(
        {
            "news_media_research": FakeResearchTool("news_media_research", master_dir),
            "chatgptxhigh": FakeResearchTool("chatgptxhigh", master_dir),
        },
        ["news_media", "chatgptxhigh"],
    )
    by_name = {tool.name: tool for tool in tools}
    prompts = [
        {"researcher": "news_media", "prompt": "Research current media evidence with primary sources and exact timestamps. " * 12},
        {"researcher": "chatgptxhigh", "prompt": "Research the same question with independent browser evidence and caveats. " * 12},
    ]
    context_tokens = set_research_artifact_context(master_dir, master_dir)
    try:
        started = json.loads(
            helper._invoke_tool_sync(
                by_name["start_researchers_async"],
                {
                    "requests_json": json.dumps(prompts, separators=(",", ":")),
                    "save_artifacts": False,
                },
            )
        )
        assert started["async_started"] is True
        job = admin_mod._async_job_get(started["job_id"])
        assert job is not None
        assert job["evidence_dir"] == master_dir
        payload = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": started["job_id"], "include_outputs": False, "wait_seconds": 900},
            )
        )
    finally:
        reset_research_artifact_context(context_tokens)

    assert payload["complete"] is True
    assert started["max_parallel"] == 2
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((Path(master_dir) / "ipc_seen").glob("*.json"))
    ]
    assert {row["name"] for row in records} == {"news_media_research", "chatgptxhigh"}
    assert all(row["data_dir"] == master_dir for row in records)
    assert all(row["master_dir"] == master_dir for row in records)


def test_required_researcher_async_request_rejects_missing_researchers():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True, business_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific", "business"],
        required_researchers=["scientific", "business"],
    )
    prompt = "Research this required source family with primary evidence, exact dates, contradictions, and limitations. " * 12

    normalized, errors = helper._normalize_researcher_requests(
        json.dumps([{"researcher": "scientific", "prompt": prompt}]),
        enabled={"scientific", "business"},
        tools_by_name={"scientific_research": object(), "business_research": object()},
        required_researchers={"scientific", "business"},
    )

    assert normalized == []
    assert any("business" in str(error.get("error")) for error in errors)


def test_required_researcher_batch_request_rejects_missing_researchers():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True, business_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific", "business"],
        required_researchers=["scientific", "business"],
    )

    class FakeResearchTool:
        def __init__(self, name):
            self.name = name

        async def on_invoke_tool(self, ctx, raw_args):
            return json.dumps({"research_worked": True, "failure_reason": "", "final_research_review": "ok"})

    tool = helper._build_batch_tool(
        {
            "scientific_research": FakeResearchTool("scientific_research"),
            "business_research": FakeResearchTool("business_research"),
        },
        ["scientific", "business"],
    )
    prompt = "Research this required source family with primary evidence, exact dates, contradictions, and limitations. " * 12
    output = helper._invoke_tool_sync(
        tool,
        {
            "requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}]),
            "save_artifacts": False,
            "max_parallel": 2,
        },
    )
    payload = json.loads(output)

    assert payload["batch_worked"] is False
    assert any("business" in str(error.get("error")) for error in payload["errors"])


def test_required_researcher_mode_hides_direct_sync_tools():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True, business_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific", "business"],
        required_researchers=["scientific", "business"],
    )

    names = _tool_names(helper._build_subagent_tools(helper._enabled_researchers()))

    assert "run_researchers_batch" in names
    assert "scientific_research" not in names
    assert "business_research" not in names


def test_async_poll_unknown_job_returns_without_waiting():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    tools = helper._build_async_tools({}, ["scientific"])
    poll = {tool.name: tool for tool in tools}["poll_researchers_async"]

    started = time.monotonic()
    payload = json.loads(
        helper._invoke_tool_sync(
            poll,
            {"job_id": "missing-job", "include_outputs": False, "wait_seconds": 900},
        )
    )

    assert time.monotonic() - started < 0.5
    assert payload["job_found"] is False


def test_async_completion_waits_for_every_preregistered_task(tmp_path):
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    marker_root = tmp_path / "preregistered-ipc"
    marker_root.mkdir()
    second_started = marker_root / "second.started"
    release_second = marker_root / "second.release"

    class FakeResearchTool:
        name = "scientific_research"

        def __init__(self, root):
            self.root = str(root)

        async def on_invoke_tool(self, ctx, raw_args):
            request = json.loads(str(raw_args or "{}"))
            prompt = str(request.get("prompt") or "")
            if "SECOND-REQUEST-MARKER" in prompt:
                Path(self.root, "second.started").touch()
                deadline = time.monotonic() + 10
                while not Path(self.root, "second.release").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                review = "second review after IPC release"
            else:
                review = "first review completed before the second task was released"
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": review,
                    "tool_call_counts": {},
                    "total_tool_calls": 0,
                },
                separators=(",", ":"),
            )

    tools = helper._build_async_tools(
        {"scientific_research": FakeResearchTool(marker_root)},
        ["scientific"],
    )
    by_name = {tool.name: tool for tool in tools}
    prompts = [
        {"researcher": "scientific", "prompt": "FIRST-REQUEST-MARKER First independent evidence review with primary sources and caveats. " * 12},
        {
            "researcher": "scientific",
            "prompt": (
                "SECOND-REQUEST-MARKER Second independent evidence review with direct sources and limitations. " * 12
                + " Duplicate reason: This controlled concurrency test intentionally launches a second "
                "scientific task to verify preregistration and completion ordering across materially "
                "independent work while preserving the production duplicate guard."
            ),
        },
    ]
    started = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {"requests_json": json.dumps(prompts, separators=(",", ":")), "save_artifacts": False},
        )
    )
    assert _wait_for_file(second_started, 10)

    wait_started = time.monotonic()
    incomplete = json.loads(
        helper._invoke_tool_sync(
            by_name["poll_researchers_async"],
            {"job_id": started["job_id"], "include_outputs": False, "wait_seconds": 1},
        )
    )
    assert time.monotonic() - wait_started >= 0.8
    assert incomplete["complete"] is False
    assert sorted(task["status"] for task in incomplete["tasks"]) == ["done", "running"]

    release_second.touch()
    final_started = time.monotonic()
    complete = json.loads(
        helper._invoke_tool_sync(
            by_name["poll_researchers_async"],
            {"job_id": started["job_id"], "include_outputs": False, "wait_seconds": 900},
        )
    )
    assert time.monotonic() - final_started < 2
    assert complete["complete"] is True


def test_async_completion_is_signaled_after_result_persistence(monkeypatch):
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    persistence_started = threading.Event()
    release_persistence = threading.Event()

    def blocking_persist(*_args, **_kwargs):
        persistence_started.set()
        release_persistence.wait(10)

    monkeypatch.setattr(
        "chack_tools.researcher_administrator_agent._persist_async_researcher_output",
        blocking_persist,
    )

    class FakeResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, ctx, raw_args):
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "durable review",
                    "tool_call_counts": {},
                    "total_tool_calls": 0,
                },
                separators=(",", ":"),
            )

    tools = helper._build_async_tools({"scientific_research": FakeResearchTool()}, ["scientific"])
    by_name = {tool.name: tool for tool in tools}
    prompt = "Research persistence ordering with direct evidence and explicit limitations. " * 12
    started = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {
                "requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}], separators=(",", ":")),
                "save_artifacts": True,
            },
        )
    )
    assert persistence_started.wait(10)

    while_persisting = json.loads(
        helper._invoke_tool_sync(
            by_name["poll_researchers_async"],
            {"job_id": started["job_id"], "include_outputs": False, "wait_seconds": 0},
        )
    )
    assert while_persisting["complete"] is False
    assert while_persisting["tasks"][0]["status"] == "running"

    release_persistence.set()
    complete = json.loads(
        helper._invoke_tool_sync(
            by_name["poll_researchers_async"],
            {"job_id": started["job_id"], "include_outputs": False, "wait_seconds": 900},
        )
    )
    assert complete["complete"] is True


def test_async_management_tools_are_status_only_by_default_and_lossless(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    review = "primary evidence and contradiction analysis; " * 180
    raw_output = json.dumps(
        {
            "research_worked": True,
            "failure_reason": "",
            "final_research_review": review,
            "key_artifacts": [],
            "tool_call_counts": {"search_arxiv": 2, "fetch_url_text": 1},
            "total_tool_calls": 3,
        },
        separators=(",", ":"),
    )

    class FakeResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            return raw_output

    root = str(tmp_path / "owned-workspace")
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    tools = helper._build_async_tools(
        {"scientific_research": FakeResearchTool()},
        ["scientific"],
        artifact_root=root,
    )
    by_name = {tool.name: tool for tool in tools}
    prompt = "Investigate this scientific claim using primary evidence, dates, contradictions, and limitations. " * 10
    started = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {"requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}])},
        )
    )
    job_id = started["job_id"]
    task_id = started["tasks"][0]["task_id"]
    try:
        # No include_outputs argument: status polling must remain compact.
        polled = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": job_id, "wait_seconds": 10},
            )
        )
        assert polled["complete"] is True
        assert polled["tasks"][0]["status"] == "done"
        assert polled["tasks"][0]["result_available"] is True
        assert polled["tasks"][0]["health"] == "succeeded"
        assert "result" not in polled["tasks"][0]
        assert len(json.dumps(polled)) < len(raw_output)

        listed = json.loads(helper._invoke_tool_sync(by_name["list_researcher_jobs"], {}))
        assert [row["job_id"] for row in listed["jobs"]] == [job_id]
        assert listed["jobs"][0]["tasks"][0]["task_id"] == task_id

        inspected = json.loads(
            helper._invoke_tool_sync(
                by_name["get_researcher_task"],
                {"job_id": job_id, "task_id": task_id},
            )
        )
        assert inspected["task"]["result_available"] is True
        assert inspected["outputs_included"] is False
        assert "result" not in inspected["task"]

        metadata = json.loads(
            helper._invoke_tool_sync(
                by_name["get_researcher_result"],
                {"job_id": job_id, "task_id": task_id, "view": "metadata"},
            )
        )
        assert metadata["result_available"] is True
        assert set(metadata["available_views"]) >= {"raw", "parsed"}
        assert metadata["total_tool_calls"] == 3

        chunks = []
        offset = 0
        while True:
            page = json.loads(
                helper._invoke_tool_sync(
                    by_name["get_researcher_result"],
                    {
                        "job_id": job_id,
                        "task_id": task_id,
                        "view": "raw",
                        "offset": offset,
                        "max_chars": 1000,
                    },
                )
            )
            chunks.append(page["content"])
            if page["complete"]:
                break
            offset = page["next_offset"]
        assert "".join(chunks) == raw_output

        # Explicit compatibility path: include exactly one canonical representation.
        # Parsed JSON wins; do not duplicate it as both raw text and parsed content,
        # and keep runtime metadata on the task row rather than inside result again.
        full = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": job_id, "include_outputs": True},
            )
        )
        full_task = full["tasks"][0]
        assert set(full_task["result"]) == {"parsed_response"}
        assert "output" not in full_task["result"]
        projected = full_task["result"]["parsed_response"]
        assert set(projected) == {
            "research_worked",
            "failure_reason",
            "overall_summary",
            "findings",
            "gaps",
            "open_topics",
        }
        assert len(projected["overall_summary"]) <= 1000
        assert projected["findings"]
        assert projected["open_topics"] == []
        assert "full_research_review" not in projected
        assert "final_research_review" not in projected
        assert "key_artifacts" not in projected
        assert "tool_call_counts" not in projected
        assert "total_tool_calls" not in projected
        assert full_task["tool_call_counts"] == {"search_arxiv": 2, "fetch_url_text": 1}
        assert full_task["total_tool_calls"] == 3
        # The complete review is not copied into the poll at all; lossless raw and
        # parsed views above remain available on demand.
        assert review not in json.dumps(full, ensure_ascii=False)
        assert admin_mod._researcher_responses_from_poll_output(json.dumps(full)) == []
        recovered = admin_mod._researcher_responses_from_async_jobs([job_id])
        assert recovered[0]["full_research_review"] == review
        assert recovered[0]["tool_call_counts"] == {"search_arxiv": 2, "fetch_url_text": 1}
        assert recovered[0]["total_tool_calls"] == 3
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            job = admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)
        for task in ((job or {}).get("tasks") or {}).values():
            timer = task.get("deadline_timer")
            if timer is not None:
                timer.cancel()


def test_async_management_tools_reject_foreign_workspace_jobs(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    owned_root = str(tmp_path / "owned")
    tools = helper._build_async_tools({}, ["scientific"], artifact_root=owned_root)
    by_name = {tool.name: tool for tool in tools}
    foreign_job = "research-job-foreign"
    foreign_task = "task-foreign"
    admin_mod._async_job_store(
        foreign_job,
        {
            "job_id": foreign_job,
            "created_at": time.time(),
            "evidence_dir": str(tmp_path / "foreign"),
            "expected_task_count": 1,
            "tasks": {
                foreign_task: {
                    "task_id": foreign_task,
                    "researcher": "scientific",
                    "researcher_tool": "scientific_research",
                    "status": "done",
                    "execution_active": False,
                    "result": {"output": "secret foreign output"},
                }
            },
        },
    )
    try:
        listed = json.loads(helper._invoke_tool_sync(by_name["list_researcher_jobs"], {}))
        assert listed["jobs"] == []
        for tool_name, args in (
            ("poll_researchers_async", {"job_id": foreign_job}),
            ("get_researcher_task", {"job_id": foreign_job, "task_id": foreign_task}),
            ("get_researcher_result", {"job_id": foreign_job, "task_id": foreign_task}),
            (
                "cancel_researcher_task",
                {"job_id": foreign_job, "task_id": foreign_task, "reason": "Foreign task must remain isolated."},
            ),
            (
                "retry_researcher_task",
                {
                    "job_id": foreign_job,
                    "task_id": foreign_task,
                    "reason": "Foreign task retries must remain isolated from this administrator workspace completely.",
                },
            ),
            ("cancel_researchers_async", {"job_id": foreign_job}),
        ):
            payload = json.loads(helper._invoke_tool_sync(by_name[tool_name], args))
            assert payload.get("job_found") is False
            assert "owned" in str(payload.get("error") or "").lower()
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            admin_mod._ASYNC_RESEARCH_JOBS.pop(foreign_job, None)


def test_retry_researcher_task_reuses_private_prompt_once(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    counter_path = tmp_path / "retry-call-counter.txt"

    class FlakyResearchTool:
        name = "scientific_research"

        def __init__(self, counter):
            self.counter = str(counter)

        async def on_invoke_tool(self, _ctx, _raw_args):
            counter = Path(self.counter)
            try:
                calls = int(counter.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                calls = 0
            calls += 1
            counter.write_text(str(calls), encoding="utf-8")
            if calls == 1:
                return "ERROR: transient provider failure"
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "retry succeeded with primary evidence",
                    "tool_call_counts": {},
                    "total_tool_calls": 0,
                }
            )

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    root = str(tmp_path / "retry-workspace")
    by_name = {
        tool.name: tool
        for tool in helper._build_async_tools(
            {"scientific_research": FlakyResearchTool(counter_path)},
            ["scientific"],
            artifact_root=root,
        )
    }
    private_marker = "PRIVATE-ORIGINAL-REQUEST-MARKER"
    prompt = (private_marker + " investigate primary scientific sources and contradictions. ") * 12
    started = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {"requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}])},
        )
    )
    source_job_id = started["job_id"]
    source_task_id = started["tasks"][0]["task_id"]
    all_job_ids = [source_job_id]
    try:
        source = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": source_job_id, "wait_seconds": 10, "include_outputs": True},
            )
        )
        assert source["tasks"][0]["status"] == "error"
        assert source["tasks"][0]["result"] == {"output": "ERROR: transient provider failure"}
        assert private_marker not in json.dumps(source)
        failed_raw = list((Path(root) / "researcher_outputs").glob("async_*.raw.txt"))
        assert len(failed_raw) == 1
        assert failed_raw[0].read_text(encoding="utf-8") == "ERROR: transient provider failure"

        retried = json.loads(
            helper._invoke_tool_sync(
                by_name["retry_researcher_task"],
                {
                    "job_id": source_job_id,
                    "task_id": source_task_id,
                    "reason": (
                        "The first provider call returned a transient transport failure before collecting any "
                        "sources, so one identical retry can materially recover the missing evidence."
                    ),
                },
            )
        )
        assert retried["retry_started"] is True
        retry_job_id = retried["job_id"]
        all_job_ids.append(retry_job_id)
        retry = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": retry_job_id, "wait_seconds": 10},
            )
        )
        assert retry["tasks"][0]["status"] == "done"
        assert retry["tasks"][0]["retried_from_job_id"] == source_job_id
        assert private_marker not in json.dumps(retry)

        duplicate_retry = json.loads(
            helper._invoke_tool_sync(
                by_name["retry_researcher_task"],
                {
                    "job_id": source_job_id,
                    "task_id": source_task_id,
                    "reason": (
                        "A second retry should be rejected even if this explanation is intentionally long enough "
                        "to satisfy validation, because every task lineage has a strict one-retry budget."
                    ),
                },
            )
        )
        assert duplicate_retry["retry_started"] is False
        assert "already used" in duplicate_retry["error"]
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            jobs = [admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None) for job_id in all_job_ids]
        for job in jobs:
            for task in ((job or {}).get("tasks") or {}).values():
                timer = task.get("deadline_timer")
                if timer is not None:
                    timer.cancel()


def test_async_poll_bounds_unparseable_raw_output_and_keeps_lossless_view(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    full_raw_output = "unparseable-provider-output-" * 2_000

    class UnparseableResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            return full_raw_output

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    root = str(tmp_path / "bounded-raw-workspace")
    by_name = {
        tool.name: tool
        for tool in helper._build_async_tools(
            {"scientific_research": UnparseableResearchTool()},
            ["scientific"],
            artifact_root=root,
        )
    }
    prompt = "Investigate primary scientific evidence, contradictions, scope, dates, and provenance. " * 10
    started = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {"requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}])},
        )
    )
    job_id = started["job_id"]
    task_id = started["tasks"][0]["task_id"]
    try:
        polled = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": job_id, "wait_seconds": 10, "include_outputs": True},
            )
        )
        projected = polled["tasks"][0]["result"]
        assert polled["tasks"][0]["status"] == "error"
        assert len(projected["output"]) == 2_000
        assert projected["output_truncated"] is True
        assert projected["raw_total_chars"] == len(full_raw_output)
        assert projected["raw_view_available"] is True
        assert len(json.dumps(polled)) < 8_000

        raw_page = json.loads(
            helper._invoke_tool_sync(
                by_name["get_researcher_result"],
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "view": "raw",
                    "offset": 0,
                    "max_chars": 12_000,
                },
            )
        )
        assert raw_page["total_chars"] == len(full_raw_output)
        assert raw_page["content"] == full_raw_output[:12_000]
        assert raw_page["next_offset"] == 12_000
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            job = admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)
        for task in ((job or {}).get("tasks") or {}).values():
            timer = task.get("deadline_timer")
            if timer is not None:
                timer.cancel()


def test_async_orchestrator_blocks_unjustified_duplicate_researchers(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    counter_path = tmp_path / "duplicate-launch-count.txt"

    class CountingResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            try:
                count = int(counter_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                count = 0
            counter_path.write_text(str(count + 1), encoding="utf-8")
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "overall_summary": "The focused scientific pass completed.",
                    "findings": [],
                    "gaps": [],
                    "open_topics": [],
                    "full_research_review": "A complete evidence review was produced.",
                    "tool_call_counts": {},
                    "total_tool_calls": 0,
                }
            )

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    root = str(tmp_path / "duplicate-guard-workspace")
    by_name = {
        tool.name: tool
        for tool in helper._build_async_tools(
            {"scientific_research": CountingResearchTool()},
            ["scientific"],
            artifact_root=root,
        )
    }
    base_prompt = (
        "Investigate one bounded scientific evidence slice using primary sources, provenance, "
        "contradictions, dates, limitations, and explicit uncertainty. "
    ) * 6
    requests = [
        {"researcher": "scientific", "prompt": f"{base_prompt} Focus number {index}."}
        for index in range(4)
    ]
    job_ids: list[str] = []
    try:
        started = json.loads(
            helper._invoke_tool_sync(
                by_name["start_researchers_async"],
                {"requests_json": json.dumps(requests), "max_parallel": 4},
            )
        )
        assert started["async_started"] is True
        assert len(started["tasks"]) == 1
        assert len(started["errors"]) == 3
        assert all("duplicate researcher launch blocked" in row["error"] for row in started["errors"])
        job_ids.append(started["job_id"])
        first_poll = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": started["job_id"], "wait_seconds": 10},
            )
        )
        assert first_poll["tasks"][0]["status"] == "done"
        assert counter_path.read_text(encoding="utf-8") == "1"

        blocked = json.loads(
            helper._invoke_tool_sync(
                by_name["start_researchers_async"],
                {"requests_json": json.dumps([requests[1]])},
            )
        )
        assert blocked["async_started"] is False
        assert blocked["tasks"] == []
        assert "duplicate researcher launch blocked" in blocked["errors"][0]["error"]

        justified_prompt = (
            f"{base_prompt}\nDuplicate reason: The first pass covered only clinical efficacy; "
            "this follow-up must inspect a materially different primary-source family, resolve "
            "a specific contradiction, and collect evidence absent from the original result."
        )
        followup = json.loads(
            helper._invoke_tool_sync(
                by_name["start_researchers_async"],
                {
                    "requests_json": json.dumps(
                        [{"researcher": "scientific", "prompt": justified_prompt}]
                    )
                },
            )
        )
        assert followup["async_started"] is True
        assert followup["errors"] == []
        job_ids.append(followup["job_id"])
        followup_poll = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": followup["job_id"], "wait_seconds": 10},
            )
        )
        assert followup_poll["tasks"][0]["status"] == "done"
        assert counter_path.read_text(encoding="utf-8") == "2"
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            jobs = [admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None) for job_id in job_ids]
        for job in jobs:
            for task in ((job or {}).get("tasks") or {}).values():
                timer = task.get("deadline_timer")
                if timer is not None:
                    timer.cancel()


def test_required_browser_initial_wave_allows_focused_followup(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    class ImmediateBrowserResearchTool:
        def __init__(self, name: str) -> None:
            self.name = name

        async def on_invoke_tool(self, _ctx, _raw_args):
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "overall_summary": "Primary evidence was collected successfully.",
                    "findings": [],
                    "gaps": [],
                    "open_topics": [],
                    "full_research_review": "Complete browser research evidence.",
                    "tool_call_counts": {"chatgpt_web": 1},
                    "total_tool_calls": 1,
                }
            )

    enabled = ["deepchatgpt", "prochatgpt"]
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            researcher_administrator_enabled=True,
            deepchatgpt_enabled=True,
            prochatgpt_enabled=True,
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=enabled,
        required_researchers=enabled,
    )
    root = str(tmp_path / "required-browser-workspace")
    by_name = {
        tool.name: tool
        for tool in helper._build_async_tools(
            {
                "deepchatgpt_researcher": ImmediateBrowserResearchTool("deepchatgpt_researcher"),
                "prochatgpt_researcher": ImmediateBrowserResearchTool("prochatgpt_researcher"),
            },
            enabled,
            artifact_root=root,
        )
    }
    deep_prompt = "Run deep browser research with primary sources, dates, contradictions, and provenance. " * 10
    pro_prompt = "Run Pro browser research with primary sources, dates, contradictions, and provenance. " * 10
    job_ids: list[str] = []
    try:
        incomplete = json.loads(
            helper._invoke_tool_sync(
                by_name["start_researchers_async"],
                {
                    "requests_json": json.dumps(
                        [{"researcher": "deepchatgpt", "prompt": deep_prompt}]
                    )
                },
            )
        )
        assert incomplete["async_started"] is False
        assert "prochatgpt" in json.dumps(incomplete)

        initial = json.loads(
            helper._invoke_tool_sync(
                by_name["start_researchers_async"],
                {
                    "requests_json": json.dumps(
                        [
                            {"researcher": "deepchatgpt", "prompt": deep_prompt},
                            {"researcher": "prochatgpt", "prompt": pro_prompt},
                        ]
                    )
                },
            )
        )
        assert initial["async_started"] is True
        job_ids.append(initial["job_id"])
        initial_poll = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": initial["job_id"], "wait_seconds": 10},
            )
        )
        assert {task["status"] for task in initial_poll["tasks"]} == {"done"}

        focused_followup_prompt = (
            f"{deep_prompt}\nDuplicate reason: The initial browser pass left a material source-family gap; "
            "this focused follow-up must inspect different primary documents, resolve a concrete "
            "contradiction, and avoid repeating evidence already collected by the first wave."
        )
        followup = json.loads(
            helper._invoke_tool_sync(
                by_name["start_researchers_async"],
                {
                    "requests_json": json.dumps(
                        [{"researcher": "deepchatgpt", "prompt": focused_followup_prompt}]
                    ),
                    "max_parallel": 1,
                },
            )
        )
        assert followup["async_started"] is True
        job_ids.append(followup["job_id"])
        followup_poll = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": followup["job_id"], "wait_seconds": 10},
            )
        )
        assert followup_poll["tasks"][0]["status"] == "done"
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            jobs = [admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None) for job_id in job_ids]
        for job in jobs:
            for task in ((job or {}).get("tasks") or {}).values():
                timer = task.get("deadline_timer")
                if timer is not None:
                    timer.cancel()


def test_administrator_duplicate_researcher_guard_blocks_unjustified_repeat():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, websearcher_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["websearcher"],
    )

    class FakeResearchTool:
        name = "websearcher_research"
        description = "fake web researcher"

        async def on_invoke_tool(self, ctx, raw_args):
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "ok",
                },
                separators=(",", ":"),
            )

    tool = helper._guard_researcher_tool(FakeResearchTool(), "websearcher")
    prompt = "Research this topic carefully with sources and disconfirming evidence. " * 10
    first = helper._invoke_tool_sync(tool, {"prompt": prompt, "save_artifacts": False})
    second = helper._invoke_tool_sync(tool, {"prompt": prompt, "save_artifacts": False})
    justified = helper._invoke_tool_sync(
        tool,
        {
            "prompt": (
                prompt
                + "\nDuplicate reason: The first pass used only broad web search and missed direct regulatory pages, archived pages, and fetched source text that materially affect the answer."
            ),
            "save_artifacts": False,
        },
    )

    assert json.loads(first)["research_worked"] is True
    assert second.startswith("ERROR: duplicate researcher launch blocked")
    assert json.loads(justified)["research_worked"] is True


def test_administrator_async_cancel_terminates_registered_running_process(tmp_path):
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    root = tmp_path / "cancel-process"
    root.mkdir()
    started = root / "started"
    release = root / "release"

    class FakeResearchTool:
        name = "scientific_research"

        def __init__(self, marker_root):
            self.marker_root = str(marker_root)

        async def on_invoke_tool(self, _ctx, _raw_args):
            assert current_cancellation_event() is not None
            Path(self.marker_root, "started").touch()
            cancel_event = current_cancellation_event()
            deadline = time.monotonic() + 30
            while (
                not Path(self.marker_root, "release").exists()
                and not (cancel_event is not None and cancel_event.is_set())
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            if cancel_event is not None and cancel_event.is_set():
                return "ERROR: fake researcher cancelled"
            return "{}"

    tools = helper._build_async_tools(
        {"scientific_research": FakeResearchTool(root)},
        ["scientific"],
        artifact_root=str(root),
    )
    by_name = {tool.name: tool for tool in tools}
    long_prompt = "Research cancellation of a long-running scientific researcher with exact status. " * 10
    start = helper._invoke_tool_sync(
        by_name["start_researchers_async"],
        {
            "requests_json": json.dumps([{"researcher": "scientific", "prompt": long_prompt}], separators=(",", ":")),
            "save_artifacts": False,
        },
    )
    job_id = json.loads(start)["job_id"]
    assert _wait_for_file(started, 10)

    cancelled = json.loads(helper._invoke_tool_sync(by_name["cancel_researchers_async"], {"job_id": job_id}))
    assert cancelled["cancellation_requested"]

    payload = {}
    for _ in range(200):
        payload = json.loads(helper._invoke_tool_sync(by_name["poll_researchers_async"], {"job_id": job_id}))
        task = payload.get("tasks", [{}])[0]
        if (
            payload.get("complete")
            and task.get("execution_active") is False
            and isinstance(task.get("termination"), dict)
        ):
            break
        time.sleep(0.05)
    assert payload["complete"] is True
    assert payload["tasks"][0]["status"] == "cancelled"
    assert payload["tasks"][0]["execution_active"] is False
    assert payload["tasks"][0]["termination"]["term_sent"] is True
    assert payload["tasks"][0]["termination"]["process_alive_after"] is False


def test_async_cancel_kills_term_ignoring_grandchild_in_private_process_group(tmp_path):
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    root = tmp_path / "cancel-grandchild"
    root.mkdir()
    pid_path = root / "grandchild.pid"

    class GrandchildResearchTool:
        name = "scientific_research"

        def __init__(self, marker: Path):
            self.marker = str(marker)

        async def on_invoke_tool(self, _ctx, _raw_args):
            code = (
                "import os,signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({self.marker!r}, 'w').write(str(os.getpid())); "
                "time.sleep(60)"
            )
            subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            cancel_event = current_cancellation_event()
            while not (cancel_event is not None and cancel_event.is_set()):
                await asyncio.sleep(0.02)
            return "ERROR: parent worker cancelled"

    def pid_is_active(pid: int) -> bool:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            close = stat.rfind(")")
            return close >= 0 and stat[close + 2 :].split()[0] != "Z"
        except (OSError, IndexError):
            return False

    tools = helper._build_async_tools(
        {"scientific_research": GrandchildResearchTool(pid_path)},
        ["scientific"],
        artifact_root=str(root),
    )
    by_name = {tool.name: tool for tool in tools}
    prompt = "Investigate process-tree cancellation with exact evidence, adversarial cases, and portable fallbacks. " * 10
    started = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {
                "requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}]),
                "save_artifacts": False,
            },
        )
    )
    job_id = started["job_id"]
    grandchild_pid = 0
    try:
        assert _wait_for_file(pid_path, 10)
        grandchild_pid = int(pid_path.read_text(encoding="utf-8"))
        assert pid_is_active(grandchild_pid)

        cancelled = json.loads(
            helper._invoke_tool_sync(by_name["cancel_researchers_async"], {"job_id": job_id})
        )
        assert len(cancelled["cancellation_requested"]) == 1

        payload = {}
        for _ in range(200):
            payload = json.loads(
                helper._invoke_tool_sync(by_name["poll_researchers_async"], {"job_id": job_id})
            )
            task = payload.get("tasks", [{}])[0]
            if payload.get("complete") and task.get("execution_active") is False:
                break
            time.sleep(0.05)
        task = payload["tasks"][0]
        termination = task["termination"]
        assert task["status"] == "cancelled"
        assert termination["term_sent"] is True
        assert termination["kill_sent"] is True
        assert grandchild_pid in termination["descendant_pids_after_term"]
        assert termination["descendant_pids_after"] == []
        assert termination["process_alive_after"] is False
        deadline = time.monotonic() + 3
        while pid_is_active(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_is_active(grandchild_pid) is False
    finally:
        if grandchild_pid and pid_is_active(grandchild_pid):
            try:
                os.kill(grandchild_pid, 9)
            except ProcessLookupError:
                pass


def test_isolated_worker_cgroup_kills_setsided_term_ignoring_grandchild(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    probe_cgroup = admin_mod._create_researcher_cgroup()
    if not probe_cgroup:
        pytest.skip("writable cgroup-v2 delegation is unavailable")
    assert admin_mod._remove_empty_researcher_cgroup(probe_cgroup) is True

    marker = tmp_path / "escaped-grandchild.pid"
    cancel_event = threading.Event()

    class EscapingResearchTool:
        name = "scientific_research"

        def __init__(self, marker_path: Path):
            self.marker_path = str(marker_path)

        async def on_invoke_tool(self, _ctx, _raw_args):
            code = (
                "import os,signal,time; "
                "os.setsid(); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({self.marker_path!r}, 'w').write(str(os.getpid())); "
                "time.sleep(60)"
            )
            subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            child_cancel = current_cancellation_event()
            while not (child_cancel is not None and child_cancel.is_set()):
                await asyncio.sleep(0.02)
            return "ERROR: parent worker cancelled"

    def pid_is_active(pid: int) -> bool:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            close = stat.rfind(")")
            return close >= 0 and stat[close + 2 :].split()[0] != "Z"
        except (OSError, IndexError):
            return False

    def request_cancel_after_spawn() -> None:
        if _wait_for_file(marker, 10):
            cancel_event.set()

    canceller = threading.Thread(target=request_cancel_after_spawn, daemon=True)
    canceller.start()
    escaped_pid = 0
    try:
        result = admin_mod._run_researcher_in_process(
            EscapingResearchTool(marker),
            {"prompt": "Research escaped descendants with complete evidence. " * 10},
            evidence_dir=str(tmp_path),
            cancel_event=cancel_event,
            termination_grace_seconds=0.5,
        )
        canceller.join(timeout=2)
        assert marker.is_file()
        escaped_pid = int(marker.read_text(encoding="utf-8"))
        termination = result["termination"]
        assert result["cancelled"] is True
        assert termination["cgroup_kill_sent"] is True
        assert termination["cgroup_populated_after"] is False
        assert termination["cgroup_removed"] is True
        deadline = time.monotonic() + 3
        while pid_is_active(escaped_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_is_active(escaped_pid) is False
    finally:
        cancel_event.set()
        if escaped_pid and pid_is_active(escaped_pid):
            try:
                os.kill(escaped_pid, 9)
            except ProcessLookupError:
                pass


def test_successful_worker_cleans_daemonized_descendant_before_return(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    probe_cgroup = admin_mod._create_researcher_cgroup()
    if not probe_cgroup:
        pytest.skip("writable cgroup-v2 delegation is unavailable")
    assert admin_mod._remove_empty_researcher_cgroup(probe_cgroup) is True

    marker = tmp_path / "successful-daemon.pid"

    class ReturningResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            code = (
                "import os,signal,time; "
                "os.setsid(); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({str(marker)!r}, 'w').write(str(os.getpid())); "
                "time.sleep(60)"
            )
            subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            while not marker.exists():
                await asyncio.sleep(0.02)
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "findings": [
                        {
                            "claim": "The researcher returned before its accidental daemon.",
                            "summary": "The supervisor must clean the inherited execution boundary before publishing success.",
                        }
                    ],
                    "full_research_review": "substantive evidence " * 100,
                }
            )

    def pid_is_active(pid: int) -> bool:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            close = stat.rfind(")")
            return close >= 0 and stat[close + 2 :].split()[0] != "Z"
        except (OSError, IndexError):
            return False

    daemon_pid = 0
    try:
        result = admin_mod._run_researcher_in_process(
            ReturningResearchTool(),
            {"prompt": "Return a complete result while testing descendant cleanup. " * 10},
            evidence_dir=str(tmp_path),
            cancel_event=threading.Event(),
            termination_grace_seconds=0.2,
        )
        daemon_pid = int(marker.read_text(encoding="utf-8"))
        assert "cancelled" not in result
        assert json.loads(result["output"])["research_worked"] is True
        assert result["termination"]["cgroup_kill_sent"] is True
        assert result["termination"]["cgroup_populated_after"] is False
        assert result["termination"]["cgroup_removed"] is True
        assert pid_is_active(daemon_pid) is False
    finally:
        if daemon_pid and pid_is_active(daemon_pid):
            try:
                os.kill(daemon_pid, 9)
            except ProcessLookupError:
                pass


def test_mcp_shutdown_reconciles_active_async_task_before_exit(tmp_path):
    """MCP shutdown must not leave a terminal task physically active in its ledger."""
    import chack_tools.researcher_administrator_agent as admin_mod

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    root = tmp_path / "mcp-shutdown"
    root.mkdir()
    started = root / "started"

    class BlockingResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            started.touch()
            cancel_event = current_cancellation_event()
            while not (cancel_event is not None and cancel_event.is_set()):
                await asyncio.sleep(0.02)
            return "ERROR: shutdown cancellation"

    tools = helper._build_async_tools(
        {"scientific_research": BlockingResearchTool()},
        ["scientific"],
        artifact_root=str(root),
    )
    by_name = {tool.name: tool for tool in tools}
    prompt = "Investigate this scientific question with primary sources, exact dates, contradictions, and evidence gaps. " * 10
    launched = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {"requests_json": json.dumps([{"researcher": "scientific", "prompt": prompt}])},
        )
    )
    job_id = launched["job_id"]
    try:
        assert _wait_for_file(started, 10)
        admin_mod._shutdown_async_research_jobs(timeout_seconds=10)
        payload = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": job_id},
            )
        )
        task = payload["tasks"][0]
        assert payload["complete"] is True
        assert task["status"] == "cancelled"
        assert task["execution_active"] is False
        assert task["termination"]["term_sent"] is True
        assert task["termination"]["process_alive_after"] is False
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            job = admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)
        for task in ((job or {}).get("tasks") or {}).values():
            timer = task.get("deadline_timer")
            if timer is not None:
                timer.cancel()


def test_administrator_can_cancel_one_ordinary_task_without_cancelling_siblings(tmp_path):
    root = tmp_path / "sibling-cancel"
    root.mkdir()

    class BlockingResearcher:
        def __init__(self, short: str, marker_root: Path):
            self.short = short
            self.marker_root = str(marker_root)
            self.name = f"{short}_research"

        async def on_invoke_tool(self, _ctx, _raw_args):
            Path(self.marker_root, f"{self.short}.started").touch()
            cancel_event = current_cancellation_event()
            deadline = time.monotonic() + 30
            while (
                not Path(self.marker_root, f"{self.short}.release").exists()
                and not (cancel_event is not None and cancel_event.is_set())
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            if cancel_event is not None and cancel_event.is_set():
                return "ERROR: task cancelled"
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": f"{self.short} completed",
                    "tool_call_counts": {},
                    "total_tool_calls": 0,
                }
            )

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            researcher_administrator_enabled=True,
            scientific_enabled=True,
            business_enabled=True,
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific", "business"],
    )
    tools = helper._build_async_tools(
        {
            "scientific_research": BlockingResearcher("scientific", root),
            "business_research": BlockingResearcher("business", root),
        },
        ["scientific", "business"],
        artifact_root=str(tmp_path),
    )
    by_name = {tool.name: tool for tool in tools}
    prompt = "Investigate this claim with primary sources, exact dates, contradictions, and limitations. " * 10
    launched = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {
                "requests_json": json.dumps(
                    [
                        {"researcher": "scientific", "prompt": prompt},
                        {"researcher": "business", "prompt": prompt},
                    ]
                ),
                "max_parallel": 2,
            },
        )
    )
    job_id = launched["job_id"]
    task_by_researcher = {row["researcher"]: row["task_id"] for row in launched["tasks"]}
    try:
        assert _wait_for_file(root / "scientific.started", 10)
        assert _wait_for_file(root / "business.started", 10)
        cancelled = json.loads(
            helper._invoke_tool_sync(
                by_name["cancel_researcher_task"],
                {
                    "job_id": job_id,
                    "task_id": task_by_researcher["scientific"],
                    "reason": "Scientific request duplicated another source pass and is no longer needed.",
                },
            )
        )
        assert cancelled["cancellation_requested"] is True
        assert cancelled["task_id"] == task_by_researcher["scientific"]
        status = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": job_id},
            )
        )
        rows = {row["researcher"]: row for row in status["tasks"]}
        assert rows["scientific"]["status"] == "cancelled"
        assert rows["business"]["status"] == "running"
    finally:
        (root / "scientific.release").touch()
        (root / "business.release").touch()


def test_running_browser_task_is_protected_from_model_initiated_target_cancel(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    root = str(tmp_path / "browser-workspace")
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, chatgptxhigh_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["chatgptxhigh"],
    )
    by_name = {
        tool.name: tool
        for tool in helper._build_async_tools({}, ["chatgptxhigh"], artifact_root=root)
    }
    job_id = "research-job-browser-protected"
    task_id = "task-browser"
    cancel_event = threading.Event()
    now = time.time()
    admin_mod._async_job_store(
        job_id,
        {
            "job_id": job_id,
            "kind": "async",
            "created_at": now,
            "evidence_dir": root,
            "expected_task_count": 1,
            "tasks": {
                task_id: {
                    "task_id": task_id,
                    "researcher": "chatgptxhigh",
                    "researcher_tool": "chatgptxhigh",
                    "status": "running",
                    "execution_active": True,
                    "created_at": now,
                    "started_at": now,
                    "last_activity_at": now,
                    "last_progress_at": now,
                    "deadline_at": now + 1800,
                    "cancel_event": cancel_event,
                }
            },
        },
    )
    try:
        cancelled = json.loads(
            helper._invoke_tool_sync(
                by_name["cancel_researcher_task"],
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "reason": "The browser task appears slow, but this alone must never terminate it prematurely.",
                },
            )
        )
        assert cancelled["protected"] is True
        assert cancelled["cancellation_requested"] is False
        assert cancel_event.is_set() is False
        with admin_mod._ASYNC_RESEARCH_LOCK:
            assert admin_mod._ASYNC_RESEARCH_JOBS[job_id]["tasks"][task_id]["status"] == "running"
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)


def test_async_task_health_is_deterministic_and_separates_terminal_from_execution():
    import chack_tools.researcher_administrator_agent as admin_mod

    now = 10_000.0
    assert admin_mod._async_task_health(
        {
            "status": "running",
            "execution_active": True,
            "researcher_tool": "scientific_research",
            "started_at": now - 500,
            "last_progress_at": now - 301,
            "deadline_at": now + 1000,
        },
        now=now,
    ) == "no_recent_progress"
    assert admin_mod._async_task_health(
        {
            "status": "running",
            "execution_active": True,
            "researcher_tool": "chatgptxhigh",
            "started_at": now - 500,
            "last_progress_at": now - 301,
            "deadline_at": now + 1000,
        },
        now=now,
    ) == "healthy"
    assert admin_mod._async_task_health(
        {"status": "deadline_exceeded", "execution_active": True},
        now=now,
    ) == "unwinding"
    assert admin_mod._async_task_health(
        {"status": "done", "execution_active": False},
        now=now,
    ) == "succeeded"


def test_async_deadline_is_terminal_immediately_and_late_result_is_rejected(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod
    from concurrent.futures import Future

    job_id = "research-job-deadline-terminal"
    task_id = "task-deadline"
    completion = threading.Event()
    cancel_event = threading.Event()
    future = Future()
    future.set_running_or_notify_cancel()
    admin_mod._async_job_store(
        job_id,
        {
            "job_id": job_id,
            "created_at": time.time(),
            "evidence_dir": str(tmp_path),
            "completion_event": completion,
            "expected_task_count": 1,
            "tasks": {
                task_id: {
                    "task_id": task_id,
                    "researcher": "scientific",
                    "researcher_tool": "scientific_research",
                    "status": "running",
                    "execution_active": True,
                    "cancel_event": cancel_event,
                    "future": future,
                }
            },
        },
    )
    try:
        admin_mod._async_request_task_deadline(job_id, task_id, cancel_event, 1)
        snapshot = admin_mod._async_job_snapshot(job_id)
        task = snapshot["tasks"][task_id]
        assert task["status"] == "deadline_exceeded"
        assert task["execution_active"] is True
        assert task["finished_at"] > 0
        assert completion.is_set()
        assert admin_mod._wait_for_async_jobs_terminal([job_id], time.monotonic() + 0.2) == []
        assert admin_mod._async_jobs_have_nonterminal_tasks([job_id]) is True

        future.set_result(
            {
                "researcher_tool": "scientific_research",
                "parsed_response": {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "late output must not count",
                },
            }
        )
        admin_mod._async_mark_task_done(job_id, task_id, future)
        snapshot = admin_mod._async_job_snapshot(job_id)
        task = snapshot["tasks"][task_id]
        assert task["status"] == "deadline_exceeded"
        assert task["execution_active"] is False
        assert "result" not in task
        assert admin_mod._researcher_responses_from_async_jobs([job_id]) == []
        failures = admin_mod._researcher_failures_from_async_jobs([job_id])
        assert failures[0]["status"] == "deadline_exceeded"
        assert admin_mod._async_jobs_have_nonterminal_tasks([job_id]) is False
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)


def test_async_child_deadline_includes_time_waiting_for_parallel_slot(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod

    root = tmp_path / "slot-deadline"
    root.mkdir()

    class BlockingScientific:
        name = "scientific_research"

        def __init__(self, marker_root: Path):
            self.marker_root = str(marker_root)

        async def on_invoke_tool(self, _ctx, _raw_args):
            Path(self.marker_root, "scientific.started").touch()
            cancel_event = current_cancellation_event()
            deadline = time.monotonic() + 30
            while (
                not Path(self.marker_root, "release").exists()
                and not (cancel_event is not None and cancel_event.is_set())
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            return "ERROR: released after deadline"

    class FastBusiness:
        name = "business_research"

        def __init__(self, marker_root: Path):
            self.marker_root = str(marker_root)

        async def on_invoke_tool(self, _ctx, _raw_args):
            Path(self.marker_root, "business.started").touch()
            return "{}"

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            researcher_administrator_enabled=True,
            scientific_enabled=True,
            business_enabled=True,
            researcher_administrator_child_timeout_seconds=1,
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific", "business"],
    )
    tools = helper._build_async_tools(
        {
            "scientific_research": BlockingScientific(root),
            "business_research": FastBusiness(root),
        },
        ["scientific", "business"],
        artifact_root=str(tmp_path),
    )
    by_name = {tool.name: tool for tool in tools}
    prompt = "Investigate this claim with primary evidence, dates, contradictions, and explicit limitations. " * 10
    started = json.loads(
        helper._invoke_tool_sync(
            by_name["start_researchers_async"],
            {
                "requests_json": json.dumps(
                    [
                        {"researcher": "scientific", "prompt": prompt},
                        {"researcher": "business", "prompt": prompt},
                    ],
                    separators=(",", ":"),
                ),
                "save_artifacts": True,
                "max_parallel": 1,
            },
        )
    )
    job_id = started["job_id"]
    try:
        assert _wait_for_file(root / "scientific.started", 10)
        polled = json.loads(
            helper._invoke_tool_sync(
                by_name["poll_researchers_async"],
                {"job_id": job_id, "include_outputs": False, "wait_seconds": 3},
            )
        )
        assert polled["complete"] is True
        assert {task["status"] for task in polled["tasks"]} == {"deadline_exceeded"}
        assert not (root / "business.started").exists()
    finally:
        (root / "release").touch()
        deadline = time.monotonic() + 5
        while admin_mod._async_jobs_have_nonterminal_tasks([job_id]) and time.monotonic() < deadline:
            time.sleep(0.05)
        with admin_mod._ASYNC_RESEARCH_LOCK:
            job = admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)
        for task in ((job or {}).get("tasks") or {}).values():
            timer = task.get("deadline_timer")
            if timer is not None:
                timer.cancel()


def test_artifact_cleanup_is_deferred_until_late_writer_finishes(monkeypatch):
    import chack_tools.researcher_administrator_agent as admin_mod

    monkeypatch.delenv("CHACK_RESEARCH_MASTER_DIR", raising=False)
    master = sc.create_research_master_dir("deferred-cleanup")
    token = set_research_artifact_context(master, master)
    evidence = Path(master, "scientific", "late.txt")
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("still writing", encoding="utf-8")
    try:
        admin_mod._research_writer_started(master)
        admin_mod._cleanup_research_artifacts_when_idle(master, save_artifacts=False)
        assert evidence.exists()
        admin_mod._research_writer_finished(master)
        assert not Path(master).exists()
    finally:
        reset_research_artifact_context(token)
        cleanup_research_artifacts(master, save_artifacts=False)


def test_cancellation_event_terminates_all_registered_processes():
    cancel_event = threading.Event()
    token = set_cancellation_event(cancel_event)
    try:
        killed: list[str] = []

        class FakeProcess:
            def __init__(self, name):
                self.name = name

        reg1 = register_process(FakeProcess("one"), lambda proc: killed.append(proc.name))
        reg2 = register_process(FakeProcess("two"), lambda proc: killed.append(proc.name))

        assert request_cancel(cancel_event) is True
        assert sorted(killed) == ["one", "two"]
        unregister_process(reg1)
        unregister_process(reg2)
    finally:
        reset_cancellation_event(token)


def test_async_task_done_persists_researcher_output_file(tmp_path):
    import chack_tools.researcher_administrator_agent as admin_mod
    from concurrent.futures import Future

    job_id = "research-job-persist-output"
    task_id = "task-1"
    admin_mod._async_job_store(
        job_id,
        {
            "job_id": job_id,
            "created_at": time.time(),
            "evidence_dir": str(tmp_path),
            "tasks": {
                task_id: {
                    "task_id": task_id,
                    "researcher": "cli",
                    "researcher_tool": "cli_research",
                    "status": "running",
                }
            },
        },
    )
    future = Future()
    raw_output = json.dumps(
        {
            "research_worked": True,
            "failure_reason": "",
            "final_research_review": "cli review",
            "tool_call_counts": {"exec": 2},
            "total_tool_calls": 2,
        },
        separators=(",", ":"),
    )
    future.set_result(
        {
            "researcher_tool": "cli_research",
            "output": raw_output,
            "parsed_response": json.loads(raw_output),
        }
    )

    admin_mod._async_mark_task_done(job_id, task_id, future)

    files = list((tmp_path / "researcher_outputs").glob("async_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["researcher_tool"] == "cli_research"
    assert payload["full_research_review"] == "cli review"
    assert payload["tool_call_counts"] == {"exec": 2}
    raw_files = list((tmp_path / "researcher_outputs").glob("async_*.raw.txt"))
    assert len(raw_files) == 1
    assert raw_files[0].read_text(encoding="utf-8") == raw_output


def test_normalize_researcher_name_accepts_aliases_and_tool_names():
    assert normalize_researcher_name("web") == "websearcher"
    assert normalize_researcher_name("Social-Network") == "social_network"
    assert normalize_researcher_name("scientific_research") == "scientific"
    assert normalize_researcher_name("unknown") == "unknown"
    # Every registry short-name normalizes to itself.
    for short in RESEARCHER_REGISTRY:
        assert normalize_researcher_name(short) == short


def test_master_dir_shares_per_type_folder(monkeypatch):
    monkeypatch.delenv("CHACK_RESEARCH_MASTER_DIR", raising=False)
    master = sc.create_research_master_dir("sess-test")
    try:
        monkeypatch.setenv("CHACK_RESEARCH_MASTER_DIR", master)
        first = sc.create_subagent_evidence_dir("scientific", "sess-test")
        second = sc.create_subagent_evidence_dir("scientific", "sess-test")
        assert first == second == os.path.join(master, "scientific")
        assert sc.create_subagent_evidence_dir("business", "sess-test") == os.path.join(master, "business")
    finally:
        cleanup_research_artifacts(master, save_artifacts=False)


def test_cleanup_guard_preserves_subfolders_but_admin_owns_master(monkeypatch):
    monkeypatch.delenv("CHACK_RESEARCH_MASTER_DIR", raising=False)
    master = sc.create_research_master_dir("sess-guard")
    monkeypatch.setenv("CHACK_RESEARCH_MASTER_DIR", master)
    sub = sc.create_subagent_evidence_dir("scientific", "sess-guard")
    Path(sub, "paper.txt").write_text("evidence", encoding="utf-8")

    # A sub-researcher must not delete its per-type folder while siblings run.
    cleanup_research_artifacts(sub, save_artifacts=False)
    assert Path(sub, "paper.txt").exists()

    # The administrator owns the master folder and may clean the whole tree.
    cleanup_research_artifacts(master, save_artifacts=False)
    assert not Path(master).exists()


def test_master_aware_instruction_forces_persistence(monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_MASTER_DIR", "/tmp/chack-research-data/x/administrator-1")
    text = sc.append_evidence_dir_instruction(
        "please research topic",
        "/tmp/chack-research-data/x/administrator-1/scientific",
        "Start now.",
        save_artifacts=False,
    )
    assert "shared by every researcher of your type" in text
    assert "temporary" in text
    assert "do not include evidence paths" in text

    preserved = sc.append_evidence_dir_instruction(
        "please research topic",
        "/tmp/chack-research-data/x/administrator-1/scientific",
        "Start now.",
        save_artifacts=True,
    )
    assert "requested preserved evidence files" in preserved


def test_run_rejects_when_no_researchers_enabled():
    cfg = ToolsConfig(researcher_administrator_enabled=True)  # nothing enabled
    helper = ResearcherAdministratorAgentTool(cfg, model_provider="openai", fallback_model="m")
    long_prompt = "x" * 600
    out = helper.run(long_prompt)
    assert "no researchers enabled" in out


def test_administrator_agent_dict_overrides_models_and_turns():
    import chack_tools.agents_toolset as at

    captured = {}
    original = at.ResearcherAdministratorAgentTool

    class Spy(original):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    at.ResearcherAdministratorAgentTool = Spy
    try:
        cfg = ToolsConfig(
            researcher_administrator_enabled=True,
            researcher_administrator_researchers=["scientific", "business"],
            researcher_administrator_agent={
                "model": "gpt-test-admin",
                "max_turns": 123,
                "researcher_models": {"scientific": "SCI_MODEL", "business": "BIZ_MODEL"},
                "researcher_max_turns": {"scientific": 40},
            },
        )
        at.AgentsToolset(cfg, model_provider="openai", default_model="primary-x")
    finally:
        at.ResearcherAdministratorAgentTool = original

    assert captured["max_turns"] == 123
    assert captured["model_name"] == "gpt-test-admin"
    assert captured["researcher_model_overrides"] == {"scientific": "SCI_MODEL", "business": "BIZ_MODEL"}
    assert captured["researcher_max_turns_overrides"] == {"scientific": 40}


def test_administrator_per_researcher_model_overrides_with_aliases():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True),
        model_provider="openai",
        fallback_model="gpt-4o",
        researchers=["scientific", "websearcher"],
        researcher_model_overrides={"scientific": "MODEL_SCI", "web": "MODEL_WEB"},
        researcher_max_turns_overrides={"scientific": 45},
    )
    # "web" alias normalizes to the websearcher short-name.
    assert helper.researcher_model_overrides == {"scientific": "MODEL_SCI", "websearcher": "MODEL_WEB"}
    assert helper._model_for("scientific", "DEFAULT") == "MODEL_SCI"
    assert helper._model_for("websearcher", "DEFAULT") == "MODEL_WEB"
    assert helper._model_for("business", "DEFAULT") == "DEFAULT"
    assert helper._max_turns_for("scientific", 30) == 45
    assert helper._max_turns_for("websearcher", 30) == 30


def test_no_save_keeps_sibling_data_until_admin_finishes(monkeypatch, tmp_path):
    # In no-save mode, sub-researchers must not delete their per-type folder so
    # later same-type researchers can read earlier researchers' downloads.
    monkeypatch.delenv("CHACK_RESEARCH_MASTER_DIR", raising=False)
    master = sc.create_research_master_dir("nosave")
    monkeypatch.setenv("CHACK_RESEARCH_MASTER_DIR", master)

    first = sc.create_subagent_evidence_dir("scientific", "nosave")
    Path(first, "batch1.txt").write_text("old data", encoding="utf-8")
    # First researcher's own cleanup runs with save_artifacts=False mid-run.
    cleanup_research_artifacts(first, save_artifacts=False)
    assert Path(first, "batch1.txt").exists()

    # A later same-type researcher shares the folder and sees the old data.
    second = sc.create_subagent_evidence_dir("scientific", "nosave")
    assert second == first
    assert Path(second, "batch1.txt").exists()

    # Only when the administrator finishes (no-save) is the whole tree removed.
    cleanup_research_artifacts(master, save_artifacts=False)
    assert not Path(master).exists()


def test_administrator_output_schema_shape():
    from chack_tools.researcher_administrator_agent import researcher_administrator_output_schema

    schema = researcher_administrator_output_schema(preserve_artifacts=True)

    props = schema["properties"]
    assert set(props) == {"research_worked", "failure_reason", "administrator_conclusions"}
    assert schema["required"] == [
        "research_worked",
        "failure_reason",
        "administrator_conclusions",
    ]


def test_administrator_output_schema_omits_artifacts_when_not_preserved():
    from chack_tools.researcher_administrator_agent import researcher_administrator_output_schema

    schema = researcher_administrator_output_schema(preserve_artifacts=False)
    props = schema["properties"]
    assert "evidence_data_path" not in props
    assert "key_artifacts" not in props
    assert "researchers_executed" not in props


def test_administrator_finalizer_appends_researcher_outputs_and_usage():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    output = (
        '{"research_worked":true,"failure_reason":"",'
        '"administrator_conclusions":"summary"}'
    )
    researcher_response = {
        "research_worked": True,
        "failure_reason": "",
        "overall_summary": "Compact web research conclusion.",
        "findings": [{
            "claim": "The web evidence supports the investigated claim",
            "summary": "The researcher found direct web evidence relevant to the investigated claim and retained every citation, contradiction, and caveat in the complete review.",
        }],
        "gaps": [],
        "open_topics": [],
        "full_research_review": "web review",
        "researcher_tool": "websearcher_research",
        "tool_call_counts": {
            "search_google_web": 2,
            "fetch_url_text": 1,
        },
        "total_tool_calls": 3,
    }

    final = finalize_researcher_administrator_output(
        output,
        evidence_dir="/tmp/evidence",
        save_artifacts=True,
        researcher_responses=[researcher_response],
        tool_counts=Counter({"websearcher_research": 2, "scientific_research": 1, "task_steps_manager": 4}),
        steps=[],
    )
    payload = json.loads(final)

    assert payload["administrator_conclusions"] == "summary"
    assert payload["evidence_data_path"] == "/tmp/evidence"
    assert payload["researcher_responses"][0]["researcher_tool"] == "websearcher_research"
    assert "full_research_review" not in payload["researcher_responses"][0]
    assert payload["researcher_tool_call_counts"] == {
        "fetch_url_text": 1,
        "search_google_web": 2,
    }
    assert payload["researcher_call_counts"] == {
        "scientific_research": 1,
        "websearcher_research": 2,
    }
    assert payload["total_researcher_calls"] == 3


def test_administrator_finalizer_counts_partial_artifact_researcher(tmp_path):
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    web_dir = tmp_path / "websearcher" / "web-pages"
    web_dir.mkdir(parents=True)
    (web_dir / "source.txt").write_text("evidence", encoding="utf-8")
    (tmp_path / "websearcher" / "_artifact_manifest.jsonl").write_text(
        '{"filename":"web-pages/source.txt","source_url":"https://example.com",'
        '"provenance":"web-pages:example","tool":"fetch_url_text","kind":"web-pages","label":"example"}\n',
        encoding="utf-8",
    )

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir=str(tmp_path),
        save_artifacts=True,
        researcher_responses=[],
        researcher_failures=[],
        tool_counts=Counter(),
        steps=[],
    )
    payload = json.loads(final)

    assert payload["researcher_call_counts"] == {"websearcher_research": 1}
    assert payload["researcher_failures"][0]["researcher_tool"] == "websearcher_research"
    assert payload["researcher_failures"][0]["status"] == "partial_artifacts_without_result"
    assert "fetch_url_text:1" in payload["researcher_failures"][0]["failure_reason"]
    assert payload["researcher_tool_call_counts"] == {"fetch_url_text": 1}


def test_administrator_finalizer_counts_cancelled_researcher_artifact_manifest(tmp_path):
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    product_dir = tmp_path / "product" / "web-pages"
    product_dir.mkdir(parents=True)
    (product_dir / "source.txt").write_text("evidence", encoding="utf-8")
    (tmp_path / "product" / "_artifact_manifest.jsonl").write_text(
        '{"filename":"web-pages/source.txt","source_url":"https://example.com",'
        '"provenance":"web-pages:example","tool":"fetch_url_text","kind":"web-pages","label":"example"}\n',
        encoding="utf-8",
    )

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir=str(tmp_path),
        save_artifacts=True,
        researcher_responses=[],
        researcher_failures=[
            {
                "researcher_tool": "product_research",
                "status": "cancelled",
                "failure_reason": "ERROR: Codex exec failed (exit=-15).",
                "task_id": "task-0",
            }
        ],
        tool_counts=Counter({"product_research": 1}),
        steps=[],
    )
    payload = json.loads(final)

    assert payload["researcher_call_counts"] == {"product_research": 1}
    assert payload["researcher_tool_call_counts"] == {"fetch_url_text": 1}
    assert payload["researcher_failures"][0]["tool_call_counts"] == {"fetch_url_text": 1}
    assert payload["output_files"]["researcher_outputs"] == ["researcher_outputs/001_product_research.json"]
    saved = json.loads((tmp_path / "researcher_outputs" / "001_product_research.json").read_text(encoding="utf-8"))
    assert saved["researcher_tool"] == "product_research"
    assert saved["status"] == "cancelled"
    assert saved["tool_call_counts"] == {"fetch_url_text": 1}


def test_administrator_finalizer_tolerates_runtime_notice_suffix():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    output = (
        '{"research_worked":false,"failure_reason":"blocked",'
        '"administrator_conclusions":"summary"}\n\n======\n[Admin Notice] Runtime budget is low.'
    )
    final = finalize_researcher_administrator_output(
        output,
        evidence_dir="/tmp/evidence",
        save_artifacts=True,
        researcher_responses=[],
        tool_counts=Counter(),
        steps=[],
    )
    payload = json.loads(final)

    assert payload["research_worked"] is False
    assert payload["evidence_data_path"] == "/tmp/evidence"
    assert payload["researcher_responses"] == []


def test_administrator_finalizer_omits_evidence_path_when_not_preserved():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir="/tmp/evidence",
        save_artifacts=False,
        researcher_responses=[],
        tool_counts=Counter(),
        steps=[],
    )
    payload = json.loads(final)

    assert "evidence_data_path" not in payload
    assert payload["researcher_responses"] == []
    assert payload["researcher_tool_call_counts"] == {}
    assert payload["researcher_call_counts"] == {}


def test_administrator_finalizer_requires_success_from_each_required_researcher():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    responses = [
        {
            "research_worked": True,
            "failure_reason": "",
            "overall_summary": "The web researcher returned a substantive evidence-backed review covering the investigated claim and its material caveats.",
            "findings": [{
                "claim": "The investigated web claim is supported by directly inspectable primary evidence",
                "summary": "The researcher compared directly inspectable sources, recorded the relevant dates and provenance, and retained material contradictions and uncertainty in the complete review for downstream synthesis.",
            }],
            "full_research_review": "The web review records the directly inspectable sources, dates, provenance, contradictions, and limitations relevant to the investigated claim.",
            "researcher_tool": "websearcher_research",
        },
        {
            "research_worked": True,
            "failure_reason": "",
            "overall_summary": "The Pro researcher returned a substantive evidence-backed review covering the investigated claim and its material caveats.",
            "findings": [{
                "claim": "The investigated claim has corroborating evidence in the independent Pro review",
                "summary": "The researcher compared independent source material, recorded the relevant dates and provenance, and retained material contradictions and uncertainty in the complete review for downstream synthesis.",
            }],
            "full_research_review": "The Pro review records the independent source material, dates, provenance, contradictions, and limitations relevant to the investigated claim.",
            "researcher_tool": "prochatgpt_researcher",
        },
    ]
    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"The administrator compared both independent reviews, separated corroborated observations from inference, and recorded the remaining uncertainty and evidence limitations before reaching this synthesis."}',
        evidence_dir="/tmp/evidence",
        save_artifacts=False,
        researcher_responses=responses,
        tool_counts=Counter(),
        steps=[],
        required_researchers=["websearcher", "prochatgpt"],
    )
    payload = json.loads(final)

    assert payload["research_worked"] is True
    assert payload["required_researchers_satisfied"] is True
    assert payload["required_researchers"] == ["websearcher_research", "prochatgpt_researcher"]


def test_administrator_finalizer_fails_when_required_researcher_did_not_succeed():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir="/tmp/evidence",
        save_artifacts=False,
        researcher_responses=[{
            "research_worked": True,
            "failure_reason": "",
            "final_research_review": "web evidence",
            "researcher_tool": "websearcher_research",
        }],
        researcher_failures=[{
            "researcher_tool": "prochatgpt_researcher",
            "status": "failed",
            "failure_reason": "browser failed",
        }],
        tool_counts=Counter(),
        steps=[],
        required_researchers=["websearcher", "prochatgpt"],
    )
    payload = json.loads(final)

    assert payload["research_worked"] is False
    assert payload["required_researchers_satisfied"] is False
    assert "prochatgpt_researcher" in payload["failure_reason"]


def test_administrator_finalizer_writes_admin_and_researcher_output_files(tmp_path):
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    researcher_response = {
        "research_worked": True,
        "failure_reason": "",
        "overall_summary": "Compact web research conclusion.",
        "findings": [{
            "claim": "The web evidence supports the investigated claim",
            "summary": "The researcher found direct web evidence relevant to the investigated claim and retained every citation, contradiction, and caveat in the complete review.",
        }],
        "gaps": [],
        "open_topics": [],
        "full_research_review": "web review",
        "researcher_tool": "websearcher_research",
        "tool_call_counts": {"fetch_url_text": 1},
        "total_tool_calls": 1,
    }

    raw_output = json.dumps(researcher_response, separators=(",", ":"))
    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir=str(tmp_path),
        save_artifacts=True,
        researcher_responses=[researcher_response],
        tool_counts=Counter({"websearcher_research": 1}),
        steps=[{"tool": "websearcher_research", "output": raw_output}],
    )
    payload = json.loads(final)

    assert payload["output_files"]["administrator_output"] == "admin_output.json"
    assert payload["output_files"]["researcher_outputs"] == ["researcher_outputs/001_websearcher_research.json"]
    assert json.loads((tmp_path / "admin_output.json").read_text(encoding="utf-8"))["administrator_conclusions"] == "summary"
    assert json.loads((tmp_path / "researcher_outputs" / "001_websearcher_research.json").read_text(encoding="utf-8")) == researcher_response
    raw_paths = payload["output_files"]["raw_researcher_outputs"]
    assert raw_paths == ["researcher_outputs/raw_step_001_websearcher_research.raw.txt"]
    assert (tmp_path / raw_paths[0]).read_text(encoding="utf-8") == raw_output
    manifest = payload["output_files"]["researcher_output_manifest"]
    assert manifest == [{
        "researcher_tool": "websearcher_research",
        "structured_path": "researcher_outputs/001_websearcher_research.json",
        "format": "full_researcher_response_v1",
        "full_research_review_available": True,
        "raw_path": raw_paths[0],
    }]
    saved_admin = json.loads((tmp_path / "admin_output.json").read_text(encoding="utf-8"))
    assert saved_admin["output_files"] == payload["output_files"]


def test_administrator_finalizer_parses_prefixed_structured_tool_output():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    researcher_json = json.dumps(
        {
            "research_worked": True,
            "failure_reason": "",
            "final_research_review": "nested review",
            "tool_call_counts": {"WebSearch": 2, "fetch_url_text": 1},
            "total_tool_calls": 3,
        }
    )

    class Action:
        tool = "mcp__chack_tools__websearcher_research"
        tool_input = {
            "result": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "result": researcher_json,
                            "researcher_tool": "websearcher_research",
                        }
                    ),
                }
            ]
        }

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir="/tmp/evidence",
        save_artifacts=True,
        researcher_responses=[],
        tool_counts=Counter({"mcp__chack_tools__websearcher_research": 1}),
        steps=[(Action(), None)],
    )
    payload = json.loads(final)

    assert len(payload["researcher_responses"]) == 1
    assert payload["researcher_responses"][0]["researcher_tool"] == "websearcher_research"
    assert payload["researcher_tool_call_counts"] == {
        "WebSearch": 2,
        "fetch_url_text": 1,
    }
    assert payload["researcher_call_counts"] == {"websearcher_research": 1}


def test_administrator_finalizer_parses_batch_tool_output():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    batch_output = {
        "batch_worked": True,
        "errors": [],
        "results": [
            {
                "researcher": "websearcher",
                "researcher_tool": "websearcher_research",
                "parsed_response": {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "web review",
                    "tool_call_counts": {"search_google_web": 2},
                    "total_tool_calls": 2,
                },
            },
            {
                "researcher": "scientific",
                "researcher_tool": "scientific_research",
                "parsed_response": {
                    "research_worked": True,
                    "failure_reason": "",
                    "final_research_review": "science review",
                    "tool_call_counts": {"search_pmc_full_text": 1},
                    "total_tool_calls": 1,
                },
            },
        ],
    }

    class Action:
        tool = "run_researchers_batch"
        tool_input = {"result": json.dumps(batch_output)}

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir="/tmp/evidence",
        save_artifacts=True,
        researcher_responses=[],
        tool_counts=Counter({"run_researchers_batch": 1}),
        steps=[(Action(), None)],
    )
    payload = json.loads(final)

    assert len(payload["researcher_responses"]) == 2
    assert payload["researcher_call_counts"] == {
        "scientific_research": 1,
        "websearcher_research": 1,
    }
    assert payload["researcher_tool_call_counts"] == {
        "search_google_web": 2,
        "search_pmc_full_text": 1,
    }


def test_administrator_finalizer_parses_async_poll_tool_output():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    poll_output = {
        "job_found": True,
        "job_id": "research-job-test",
        "complete": True,
        "tasks": [
            {
                "task_id": "task-1",
                "researcher": "religious",
                "researcher_tool": "religious_research",
                "status": "done",
                "result": {
                    "researcher_tool": "religious_research",
                    "parsed_response": {
                        "research_worked": True,
                        "failure_reason": "",
                        "final_research_review": "religious review",
                        "tool_call_counts": {"sefaria_search": 2, "quran_search": 1},
                        "total_tool_calls": 3,
                    },
                },
            },
            {
                "task_id": "task-2",
                "researcher": "scientific",
                "researcher_tool": "scientific_research",
                "status": "done",
                "result": {
                    "researcher_tool": "scientific_research",
                    "parsed_response": {
                        "research_worked": True,
                        "failure_reason": "",
                        "final_research_review": "science review",
                        "tool_call_counts": {"search_europe_pmc": 1},
                        "total_tool_calls": 1,
                    },
                },
            },
        ],
    }

    class Action:
        tool = "poll_researchers_async"
        tool_input = {"result": json.dumps(poll_output)}

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir="/tmp/evidence",
        save_artifacts=True,
        researcher_responses=[],
        tool_counts=Counter({"poll_researchers_async": 3}),
        steps=[(Action(), None)],
    )
    payload = json.loads(final)

    assert len(payload["researcher_responses"]) == 2
    assert payload["researcher_call_counts"] == {
        "religious_research": 1,
        "scientific_research": 1,
    }
    assert payload["researcher_tool_call_counts"] == {
        "quran_search": 1,
        "search_europe_pmc": 1,
        "sefaria_search": 2,
    }


def test_async_job_call_counts_include_unparsed_researcher_tasks():
    import chack_tools.researcher_administrator_agent as admin_mod

    job_id = "job-count-unparsed"
    try:
        admin_mod._async_job_store(
            job_id,
            {
                "job_id": job_id,
                "tasks": {
                    "task-1": {
                        "researcher_tool": "websearcher_research",
                        "status": "done",
                        "result": {"output": "ERROR: failed before JSON"},
                    },
                    "task-2": {
                        "researcher_tool": "product_research",
                        "status": "cancelled",
                    },
                },
            },
        )

        counts = admin_mod._researcher_call_counts_from_async_jobs([job_id])
    finally:
        with admin_mod._ASYNC_RESEARCH_LOCK:
            admin_mod._ASYNC_RESEARCH_JOBS.pop(job_id, None)

    assert counts == Counter({"websearcher_research": 1, "product_research": 1})


def test_finalizer_reports_unparsed_researcher_failures_from_poll_output():
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    poll_output = {
        "tasks": [
            {
                "task_id": "task-kg",
                "researcher": "knowledge_graph",
                "researcher_tool": "knowledge_graph_research",
                "status": "done",
                "result": {"output": "ERROR: failed before JSON"},
            }
        ]
    }

    class Action:
        tool = "poll_researchers_async"
        tool_input = {"result": json.dumps(poll_output)}

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir="/tmp/evidence",
        save_artifacts=False,
        researcher_responses=[],
        tool_counts=Counter({"poll_researchers_async": 1}),
        steps=[(Action(), None)],
    )
    payload = json.loads(final)

    assert payload["researcher_responses"] == []
    assert payload["researcher_failures"] == [
        {
            "researcher_tool": "knowledge_graph_research",
            "status": "done",
            "failure_reason": "ERROR: failed before JSON",
            "task_id": "task-kg",
        }
    ]
    assert payload["researcher_call_counts"] == {"knowledge_graph_research": 1}
