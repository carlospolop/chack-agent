"""Keyed search tools are offered only when their key can serve a request."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext
from agents.usage import Usage

import chack_tools.serpapi_keys as sk
from chack_tools import exa_web_search
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.search_availability import SERPAPI_TOOL_NAMES, keyless_web_search_note
from chack_tools.social_network_agent import SocialNetworkAgentTool
from chack_tools.task_steps_manager_state import reset_active_context, set_active_context
from chack_tools.websearcher_agent import WebSearcherAgentTool

_REAL_ACCOUNT_CHECK = sk._query_account_exhausted

_KEY_ENV = (
    "SERPAPI_API_KEY",
    "BRAVE_API_KEY",
    "FORUMSCOUT_API_KEY",
    "BOOKING_API_TOKEN",
    "BOOKING_AFFILIATE_ID",
    "AMADEUS_CLIENT_ID",
    "AMADEUS_CLIENT_SECRET",
    "OPENTRIPMAP_API_KEY",
    "TICKETMASTER_API_KEY",
    "EXA_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for name in _KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERPAPI_EXHAUSTION_CHECK_ENABLED", "1")


class _Account:
    """Stand-in for serpapi.com/account.json: searches left per key."""

    def __init__(self, monkeypatch, searches_left: dict[str, int]):
        self.searches_left = searches_left
        self.calls: list[str] = []
        self.clock = 0.0
        monkeypatch.setattr(sk, "_query_account_exhausted", _REAL_ACCOUNT_CHECK)
        monkeypatch.setattr(sk, "_now", lambda: self.clock)
        monkeypatch.setattr(sk.requests, "get", self._get)

    def _get(self, url, params=None, timeout=None):
        assert url == sk._SERPAPI_ACCOUNT_URL, f"unexpected network call to {url}"
        key = params["api_key"]
        self.calls.append(key)
        payload = {"total_searches_left": self.searches_left[key]}
        return SimpleNamespace(status_code=200, json=lambda: payload)


def _web_toolset() -> set[str]:
    config = ToolsConfig(
        brave_enabled=True,
        serpapi_google_web_enabled=True,
        serpapi_bing_web_enabled=True,
        serpapi_bing_copilot_enabled=True,
        websearcher_google_ai_mode_enabled=True,
        scientific_google_scholar_enabled=True,
        social_network_google_news_enabled=True,
        social_network_tiktok_web_enabled=True,
        social_network_forum_search_enabled=True,
        travel_booking_enabled=True,
        task_steps_manager_enabled=False,
    )
    return {tool.name for tool in AgentsToolset(config, model_provider="openai", default_model="m").tools}


def _websearcher_tools() -> list:
    helper = WebSearcherAgentTool(
        ToolsConfig(task_steps_manager_enabled=False), model_provider="openai", fallback_model="m"
    )
    return helper._build_subagent_tools()


def test_exhausted_serpapi_keys_register_no_serpapi_tools_and_a_keyless_search(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "k1,k2")
    _Account(monkeypatch, {"k1": 0, "k2": 0})

    names = _web_toolset()

    assert not names & SERPAPI_TOOL_NAMES
    # No Brave key, no ForumScout key, no Booking credentials either.
    assert not names & {"brave_search", "forum_search", "find_booking_cities"}
    assert "web_search" in names


def test_a_usable_serpapi_key_registers_the_serpapi_tools(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "dead,live")
    _Account(monkeypatch, {"dead": 0, "live": 250})

    names = _web_toolset()

    assert {
        "search_google_web",
        "search_bing_web",
        "search_bing_copilot",
        "search_google_ai_mode",
        "search_google_scholar",
        "search_google_news",
        "tiktok_web_search",
    } <= names
    assert "web_search" not in names


def test_serpapi_tools_return_on_a_later_run_once_the_quota_renews(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "k1")
    account = _Account(monkeypatch, {"k1": 0})
    assert not {tool.name for tool in _websearcher_tools()} & SERPAPI_TOOL_NAMES

    # Same verdict while the exhaustion is cached: no extra account lookups.
    account.searches_left["k1"] = 5000
    assert not {tool.name for tool in _websearcher_tools()} & SERPAPI_TOOL_NAMES
    assert account.calls == ["k1"]

    account.clock += sk._exhausted_ttl_seconds() + 1
    names = {tool.name for tool in _websearcher_tools()}

    assert {"search_google_web", "search_bing_web", "search_google_ai_mode", "search_bing_copilot"} <= names
    assert "web_search" not in names


def test_brave_search_is_registered_only_with_a_brave_key(monkeypatch):
    without_key = {tool.name for tool in _websearcher_tools()}
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    with_key = {tool.name for tool in _websearcher_tools()}

    assert "brave_search" not in without_key
    assert "brave_search" in with_key
    # A paid engine works, so the keyless fallback stays out.
    assert "web_search" not in with_key
    assert "brave_search" in _web_toolset()


def test_websearcher_without_paid_engines_keeps_keyless_search_and_is_told(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "k1")
    _Account(monkeypatch, {"k1": 0})

    tools = _websearcher_tools()
    names = {tool.name for tool in tools}

    assert {"web_search", "fetch_url_text", "gdelt_news_search"} <= names
    assert keyless_web_search_note(tools)

    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    assert not keyless_web_search_note(_websearcher_tools())


def test_keyless_search_respects_its_switch(monkeypatch):
    from chack_agent.config import ToolsConfig as AgentToolsConfig

    helper = WebSearcherAgentTool(
        AgentToolsConfig(task_steps_manager_enabled=False, keyless_web_search_enabled=False),
        model_provider="openai",
        fallback_model="m",
    )

    assert "web_search" not in {tool.name for tool in helper._build_subagent_tools()}


def test_social_agent_drops_tools_whose_keys_are_missing(monkeypatch):
    helper = SocialNetworkAgentTool(
        ToolsConfig(task_steps_manager_enabled=False), model_provider="openai", fallback_model="m"
    )
    bare = {tool.name for tool in helper._build_subagent_tools()}
    monkeypatch.setenv("FORUMSCOUT_API_KEY", "fs")
    monkeypatch.setenv("SERPAPI_API_KEY", "serp")
    keyed = {tool.name for tool in helper._build_subagent_tools()}

    assert not bare & {"forum_search", "reddit_posts_search", "x_search", "search_google_news", "tiktok_web_search"}
    assert "mastodon_search" in bare
    assert {"forum_search", "reddit_posts_search", "x_search", "search_google_news", "tiktok_web_search"} <= keyed


# ── Keyless web search: per-run cache, pacing, rate limits ─────────────────────


class _Exa:
    """Records Exa MCP calls and answers with queued (status, text) replies."""

    def __init__(self, monkeypatch):
        self.calls: list[dict] = []
        self.replies: list[tuple[int, str]] = []
        self.sleeps: list[float] = []
        self.clock = 1000.0
        monkeypatch.setattr(exa_web_search.requests, "post", self._post)
        monkeypatch.setattr(exa_web_search.time, "monotonic", lambda: self.clock)
        monkeypatch.setattr(exa_web_search.time, "sleep", self._sleep)
        monkeypatch.setattr(exa_web_search, "_next_search_at", 0.0)

    def _sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds

    def _post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(json["params"]["arguments"])
        status, text = self.replies.pop(0) if self.replies else (200, "Title: result")
        body = __import__("json").dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}]}})
        return SimpleNamespace(
            status_code=status,
            text=body,
            headers={"content-type": "application/json"},
            raise_for_status=lambda: None,
        )


def _search(tool, **args) -> str:
    raw = json.dumps(args)
    ctx = ToolContext(context=None, usage=Usage(), tool_name=tool.name, tool_call_id="t", tool_arguments=raw)
    return asyncio.run(tool.on_invoke_tool(ctx, raw))


def _keyless_search_tool():
    return next(tool for tool in exa_web_search.get_web_search_tools() if tool.name == "web_search")


def _in_session(session_id: str):
    return set_active_context(session_id, "Run 1")


def test_identical_keyless_searches_in_a_run_are_served_from_cache(monkeypatch):
    exa = _Exa(monkeypatch)
    tool = _keyless_search_tool()
    tokens = _in_session("run-a")
    try:
        first = _search(tool, query="jquery 3.7.0 advisory")
        again = _search(tool, query="  JQuery 3.7.0   advisory ")
    finally:
        reset_active_context(tokens)
    tokens = _in_session("run-b")
    try:
        other_run = _search(tool, query="jquery 3.7.0 advisory")
    finally:
        reset_active_context(tokens)

    assert first == again == other_run == "Title: result"
    # One request for run-a, one for run-b: nothing is shared across runs.
    assert len(exa.calls) == 2
    # A freshly built tool list (a new run) starts with an empty cache.
    _search(_keyless_search_tool(), query="jquery 3.7.0 advisory")
    assert len(exa.calls) == 3


def test_keyless_searches_are_paced_and_overload_is_refused(monkeypatch):
    exa = _Exa(monkeypatch)
    monkeypatch.setenv("CHACK_KEYLESS_SEARCH_MIN_INTERVAL_SECONDS", "2")
    monkeypatch.setenv("CHACK_KEYLESS_SEARCH_MAX_WAIT_SECONDS", "3")

    exa_web_search.web_search("one")
    exa_web_search.web_search("two")
    assert exa.sleeps == [2.0]

    # Two searches already booked ahead: the next would wait 4s > 3s max.
    monkeypatch.setattr(exa_web_search, "_next_search_at", exa.clock + 4)
    assert exa_web_search.web_search("three") == exa_web_search.RATE_LIMITED
    assert len(exa.calls) == 2


def test_rate_limited_keyless_search_fails_once_and_is_not_cached(monkeypatch):
    exa = _Exa(monkeypatch)
    monkeypatch.setenv("CHACK_KEYLESS_SEARCH_MIN_INTERVAL_SECONDS", "0")
    exa.replies.append((429, ""))
    tool = _keyless_search_tool()

    limited = _search(tool, query="q")
    retried = _search(tool, query="q")

    assert limited.startswith("ERROR: rate limited")
    assert "retry later or use other sources" in limited
    assert retried == "Title: result"
    assert len(exa.calls) == 2
