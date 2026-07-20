import json
import threading
import time

from agents import function_tool

from chack_tools.parallel_research_tool import get_parallel_research_tool
from chack_tools.researcher_administrator_agent import ResearcherAdministratorAgentTool
from chack_tools.tool_usage_state import (
    STORE,
    reset_active_usage_session,
    set_active_usage_session,
)


def _invoke(tool, payload):
    return json.loads(ResearcherAdministratorAgentTool._invoke_tool_sync(tool, payload))


def test_parallel_research_runs_selected_researchers_concurrently():
    lock = threading.Lock()
    active = 0
    max_active = 0

    def make_tool(name):
        @function_tool(name_override=name)
        def researcher(prompt: str, save_artifacts: bool = False) -> str:
            """Return deterministic test research."""
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.08)
            with lock:
                active -= 1
            return json.dumps({"worked": True, "prompt_chars": len(prompt)})

        return researcher

    researchers = [
        make_tool("travel_research"),
        make_tool("websearcher_research"),
        make_tool("news_media_research"),
        make_tool("social_network_research"),
    ]
    tool = get_parallel_research_tool(researchers, max_requests=4)
    prompt = "Detailed evidence request. " + ("x" * 520)
    session_id = "parallel-research-concurrency-test"
    STORE.reset_session(session_id)
    token = set_active_usage_session(session_id)
    try:
        payload = _invoke(tool, {
            "requests_json": json.dumps([
                {"researcher": "travel", "prompt": prompt},
                {"researcher": "websearcher", "prompt": prompt},
                {"researcher": "news_media", "prompt": prompt},
                {"researcher": "social_network", "prompt": prompt},
            ]),
            "max_parallel": 4,
        })
    finally:
        reset_active_usage_session(token)

    assert payload["worked"] is True
    assert [item["researcher"] for item in payload["results"]] == [
        "travel", "websearcher", "news_media", "social_network",
    ]
    assert max_active == 4
    assert STORE.snapshot(session_id) == {
        "travel_research": 1,
        "websearcher_research": 1,
        "news_media_research": 1,
        "social_network_research": 1,
    }
    STORE.clear(session_id)


def test_parallel_research_rejects_short_prompts_and_more_than_four_requests():
    @function_tool(name_override="travel_research")
    def travel(prompt: str, save_artifacts: bool = False) -> str:
        """Return deterministic test research."""
        return "unused"

    tool = get_parallel_research_tool([travel], max_requests=4)
    short = _invoke(tool, {
        "requests_json": json.dumps([{"researcher": "travel", "prompt": "short"}]),
    })
    assert short["worked"] is False
    assert "at least 500 characters" in short["errors"][0]["error"]

    prompt = "x" * 500
    too_many = _invoke(tool, {
        "requests_json": json.dumps([
            {"researcher": "travel", "prompt": prompt} for _ in range(5)
        ]),
    })
    assert too_many["worked"] is False
    assert "At most 4" in too_many["errors"][0]
