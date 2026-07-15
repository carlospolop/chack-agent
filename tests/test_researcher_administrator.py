import os
import json
import time
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.research_artifacts import cleanup_research_artifacts
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


def test_chatgpt_researchers_are_never_cancelled_for_elapsed_time():
    prompt = _ADMINISTRATOR_SYSTEM_PROMPT
    assert "Prefer `start_researchers_async`" in prompt
    assert "Never use `wait(..., terminate=true)`" in prompt
    assert "configured hard timeout" in prompt
    assert "45-90 minutes" in prompt


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
        chatgpt_cdp_url="http://127.0.0.1:9226",
    )
    helper = ResearcherAdministratorAgentTool(
        cfg,
        model_provider="openai",
        fallback_model="m",
        researchers=["deepchatgpt", "prochatgpt"],
    )

    inner = _tool_names(helper._build_subagent_tools(helper._enabled_researchers()))
    assert "start_researchers_async" in inner
    assert "poll_researchers_async" in inner
    assert "deepchatgpt_researcher" not in inner
    assert "prochatgpt_researcher" not in inner
    assert "run_researchers_batch" not in inner
    assert "cancel_researchers_async" not in inner


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
        "scientific_research",
        "business_research",
        "run_researchers_batch",
        "start_researchers_async",
        "poll_researchers_async",
        "cancel_researchers_async",
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


def test_administrator_empty_allowlist_uses_globally_enabled():
    cfg = ToolsConfig(
        researcher_administrator_enabled=True,
        scientific_enabled=True,
        webresearcher_enabled=True,  # legacy alias for websearcher
    )
    helper = ResearcherAdministratorAgentTool(cfg, model_provider="openai", fallback_model="m")
    assert set(helper._enabled_researchers()) == {"scientific", "websearcher"}


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

    assert "- websearcher via `websearcher_research`:" in text
    assert "fetch_url_text" in text
    assert "web_archive_search" in text
    assert "- scientific via `scientific_research`:" in text
    assert "search_arxiv" in text
    assert "download_pmc_full_text" in text


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
        ),
        model_provider="openai",
        fallback_model="m",
        researchers=["websearcher", "scientific"],
        self_critique_rounds=2,
    )
    monkeypatch.setattr(
        helper,
        "_build_subagent_tools",
        lambda enabled: [
            FakeTool("websearcher_research"),
            FakeTool("scientific_research"),
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

    assert payload["research_worked"] is True
    assert captured["max_tools_used_override"] == 20
    assert "budget for this run: 3 total `*_research` calls" in sent_prompt
    assert "not on management polls/status checks" in sent_prompt
    assert "Internal tools available to each researcher in this run:" in sent_prompt
    assert "- websearcher via `websearcher_research`: fetch_url_text, web_archive_search" in sent_prompt
    assert "- scientific via `scientific_research`: search_arxiv, download_pmc_full_text" in sent_prompt
    assert "compare its code-added tool_call_counts against the capability map" in sent_prompt
    assert "skipped or barely used" in sent_prompt
    assert "relaunch that researcher with explicit missing tool names" in sent_prompt
    assert "try-harder self-critique for 2 round(s)" in sent_prompt
    assert "start_researchers_async" in sent_prompt
    assert "poll_researchers_async" in sent_prompt
    assert "ChatGPT Pro/Deep use 300-600 seconds" in sent_prompt
    assert "recent_events" in sent_prompt
    assert "idle_seconds" in sent_prompt
    assert "cancel_researchers_async" in sent_prompt


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
        lambda enabled: [
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
        lambda enabled: [
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
        lambda enabled: [
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
    assert "one at a time" in started["next_step"]
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
    assert payload["waited_seconds"] == 900
    task = payload["tasks"][0]
    assert task["status"] == "done"
    assert "idle_seconds" in task
    assert task["tool_call_counts"] == {"search_arxiv": 1}
    assert task["total_tool_calls"] == 1
    assert any(event["tool"] == "search_arxiv" for event in task["recent_events"])


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


def test_administrator_async_cancel_terminates_registered_running_process():
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(researcher_administrator_enabled=True, scientific_enabled=True),
        model_provider="openai",
        fallback_model="m",
        researchers=["scientific"],
    )
    registered = threading.Event()
    release = threading.Event()
    terminated = threading.Event()

    class FakeProcess:
        killed = False

    process = FakeProcess()

    def terminate(proc):
        proc.killed = True
        terminated.set()
        release.set()

    class FakeResearchTool:
        name = "scientific_research"

        async def on_invoke_tool(self, ctx, raw_args):
            assert current_cancellation_event() is not None
            token = register_process(process, terminate)
            registered.set()
            try:
                release.wait(10)
            finally:
                unregister_process(token)
            return "ERROR: fake researcher cancelled" if process.killed else "{}"

    tools = helper._build_async_tools({"scientific_research": FakeResearchTool()}, ["scientific"])
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
    assert registered.wait(10)

    cancelled = json.loads(helper._invoke_tool_sync(by_name["cancel_researchers_async"], {"job_id": job_id}))
    assert cancelled["cancellation_requested"]
    assert terminated.wait(10)
    assert process.killed is True

    payload = {}
    for _ in range(20):
        payload = json.loads(helper._invoke_tool_sync(by_name["poll_researchers_async"], {"job_id": job_id}))
        if payload.get("complete"):
            break
        time.sleep(0.05)
    assert payload["complete"] is True
    assert payload["tasks"][0]["status"] == "cancelled"


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
    future.set_result(
        {
            "researcher_tool": "cli_research",
            "parsed_response": {
                "research_worked": True,
                "failure_reason": "",
                "final_research_review": "cli review",
                "tool_call_counts": {"exec": 2},
                "total_tool_calls": 2,
            },
        }
    )

    admin_mod._async_mark_task_done(job_id, task_id, future)

    files = list((tmp_path / "researcher_outputs").glob("async_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["researcher_tool"] == "cli_research"
    assert payload["tool_call_counts"] == {"exec": 2}


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
        "final_research_review": "web review",
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
    assert payload["researcher_responses"] == [researcher_response]
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


def test_administrator_finalizer_writes_admin_and_researcher_output_files(tmp_path):
    from chack_tools.researcher_administrator_agent import finalize_researcher_administrator_output

    researcher_response = {
        "research_worked": True,
        "failure_reason": "",
        "final_research_review": "web review",
        "researcher_tool": "websearcher_research",
        "tool_call_counts": {"fetch_url_text": 1},
        "total_tool_calls": 1,
    }

    final = finalize_researcher_administrator_output(
        '{"research_worked":true,"failure_reason":"","administrator_conclusions":"summary"}',
        evidence_dir=str(tmp_path),
        save_artifacts=True,
        researcher_responses=[researcher_response],
        tool_counts=Counter({"websearcher_research": 1}),
        steps=[],
    )
    payload = json.loads(final)

    assert payload["output_files"]["administrator_output"] == "admin_output.json"
    assert payload["output_files"]["researcher_outputs"] == ["researcher_outputs/001_websearcher_research.json"]
    assert json.loads((tmp_path / "admin_output.json").read_text(encoding="utf-8"))["administrator_conclusions"] == "summary"
    assert json.loads((tmp_path / "researcher_outputs" / "001_websearcher_research.json").read_text(encoding="utf-8")) == researcher_response


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
