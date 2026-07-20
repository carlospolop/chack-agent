import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLUGIN_PATH = (
    Path(__file__).parents[1]
    / "integrations"
    / "hermes"
    / "chack-research-usage"
    / "__init__.py"
)


def _load_plugin():
    name = "test_chack_research_usage_plugin"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin():
    return _load_plugin()


def _queue_result(usage):
    inner = json.dumps({"researches": [], "researcher_usage": usage})
    outer = json.dumps({"result": inner, "structuredContent": {"result": inner}})
    return (
        '<untrusted_tool_result source="mcp__chack_agent__researcher_queue">\n'
        "external data\n"
        f"{outer}\n"
        "</untrusted_tool_result>"
    )


def _observe(plugin, *, session, turn, call_id, usage, platform="telegram"):
    plugin._post_tool_call(
        tool_name="mcp__chack_agent__researcher_queue",
        result=_queue_result(usage),
        session_id=session,
        task_id=f"task-{session}",
        turn_id=turn,
        tool_call_id=call_id,
        status="ok",
        platform=platform,
    )


def test_telegram_final_response_aggregates_multiple_queue_calls(plugin):
    plugin._pre_llm_call(session_id="telegram-session", task_id="task-tg", turn_id="turn-1")
    _observe(
        plugin,
        session="telegram-session",
        turn="turn-1",
        call_id="call-1",
        usage={
            "administrator_calls": 1,
            "researcher_call_counts": {
                "prochatgpt_researcher": 2,
                "websearcher_research": 1,
            },
            "complete": True,
        },
    )
    _observe(
        plugin,
        session="telegram-session",
        turn="turn-1",
        call_id="call-2",
        usage={
            "administrator_calls": 2,
            "researcher_call_counts": {
                "deepchatgpt_researcher": 1,
                "websearcher_research": 1,
            },
            "complete": True,
        },
    )

    transformed = plugin._transform_llm_output(
        response_text="Final research answer.",
        session_id="telegram-session",
        platform="telegram",
    )

    assert transformed == (
        "Final research answer.\n\n"
        "_Chack usage: queue x2 · admin x3 · chatgptpro x2 · "
        "chatgptdeep x1 · webresearcher x2_"
    )
    assert plugin._transform_llm_output(
        response_text="Duplicate delivery", session_id="telegram-session"
    ) is None


def test_cron_final_response_gets_same_footer(plugin):
    plugin._pre_llm_call(
        session_id="cron_job_20260720", task_id="cron-task", turn_id="cron-turn", platform="cron"
    )
    _observe(
        plugin,
        session="cron_job_20260720",
        turn="cron-turn",
        call_id="cron-call",
        platform="cron",
        usage={
            "administrator_calls": 1,
            "researcher_call_counts": {"deepchatgpt_researcher": 1},
            "complete": True,
        },
    )

    transformed = plugin._transform_llm_output(
        response_text="Cron report", session_id="cron_job_20260720", platform="cron"
    )

    assert transformed.endswith(
        "_Chack usage: queue x1 · admin x1 · chatgptdeep x1_"
    )


def test_duplicate_post_tool_event_is_counted_once(plugin):
    plugin._pre_llm_call(session_id="s", task_id="t", turn_id="turn")
    usage = {
        "administrator_calls": 1,
        "researcher_call_counts": {"prochatgpt_researcher": 1},
        "complete": True,
    }
    _observe(plugin, session="s", turn="turn", call_id="same", usage=usage)
    _observe(plugin, session="s", turn="turn", call_id="same", usage=usage)

    transformed = plugin._transform_llm_output(response_text="Done", session_id="s")

    assert transformed.endswith("_Chack usage: queue x1 · admin x1 · chatgptpro x1_")


def test_no_queue_call_leaves_response_unchanged(plugin):
    plugin._pre_llm_call(session_id="plain", task_id="plain-task", turn_id="plain-turn")
    assert plugin._transform_llm_output(response_text="Plain answer", session_id="plain") is None


def test_incomplete_queue_accounting_is_never_presented_as_exact(plugin):
    plugin._pre_llm_call(session_id="failed", task_id="failed-task", turn_id="failed-turn")
    plugin._post_tool_call(
        tool_name="mcp__chack_agent__researcher_queue",
        result="ERROR: queue timed out",
        session_id="failed",
        task_id="failed-task",
        turn_id="failed-turn",
        tool_call_id="failed-call",
        status="error",
    )

    transformed = plugin._transform_llm_output(response_text="Research failed", session_id="failed")

    assert transformed.endswith(
        "_Chack usage (partial accounting): queue x1 · admin x0_"
    )


def test_old_queue_payload_can_recover_counts_from_safe_admin_artifact(plugin, tmp_path, monkeypatch):
    allowed = tmp_path / "chack-research-data"
    admin_path = allowed / "researcher-queues" / "q" / "researches" / "r" / "admin_output.json"
    admin_path.parent.mkdir(parents=True)
    admin_path.write_text(
        json.dumps({"researcher_call_counts": {"websearcher_research": 2}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin, "_ALLOWED_ARTIFACT_ROOTS", (allowed.resolve(),))
    old_payload = {
        "researches": [
            {
                "conclusions": "legacy result",
                "output_files": {"json": str(admin_path)},
            }
        ]
    }
    plugin._pre_llm_call(session_id="legacy", task_id="legacy-task", turn_id="legacy-turn")
    plugin._post_tool_call(
        tool_name="researcher_queue",
        result=json.dumps(old_payload),
        session_id="legacy",
        task_id="legacy-task",
        turn_id="legacy-turn",
        tool_call_id="legacy-call",
        status="ok",
    )

    transformed = plugin._transform_llm_output(response_text="Legacy answer", session_id="legacy")

    assert transformed.endswith(
        "_Chack usage: queue x1 · admin x1 · webresearcher x2_"
    )
