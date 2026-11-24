"""fetch_url_text: identical fetches in a run are served once; permanent HTTP
failures fail fast and are not fetched again."""

from __future__ import annotations

import asyncio
import json

import pytest
from agents.tool_context import ToolContext
from agents.usage import Usage

from chack_tools import open_research_sources as ors
from chack_tools.config import ToolsConfig
from chack_tools.task_steps_manager_state import reset_active_context, set_active_context

_PAGE = "<html><body><p>" + "OSV advisory text for the vulnerable package version. " * 4 + "</p></body></html>"


class _Response:
    def __init__(self, url: str, status: int, body: str = _PAGE):
        self.url = url
        self.status_code = status
        self.headers = {"content-type": "text/html"}
        self.encoding = "utf-8"
        self._body = body.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(response=self)

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        pass


class _Web:
    """Answers each URL with a fixed status and counts the requests."""

    def __init__(self, monkeypatch, tmp_path, statuses: dict[str, int]):
        self.statuses = statuses
        self.requests: list[str] = []
        monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(ors, "_is_public_https_url", lambda url: url.startswith("https://"))
        monkeypatch.setattr(ors.requests, "get", self._get)

    def _get(self, url, **_kwargs):
        self.requests.append(url)
        return _Response(url, self.statuses.get(url.split("#")[0], 200))


def _fetch(tool, url: str, timeout_seconds: int = 30) -> str:
    raw = json.dumps({"url": url, "timeout_seconds": timeout_seconds})
    ctx = ToolContext(context=None, usage=Usage(), tool_name=tool.name, tool_call_id="t", tool_arguments=raw)
    return asyncio.run(tool.on_invoke_tool(ctx, raw))


def _tool():
    return ors.get_fetch_url_text_tool(ors.OpenResearchTool(ToolsConfig()))


@pytest.fixture
def run_a():
    tokens = set_active_context("run-a", "Run 1")
    yield
    reset_active_context(tokens)


def test_identical_fetches_in_a_run_hit_the_network_once(monkeypatch, tmp_path, run_a):
    web = _Web(monkeypatch, tmp_path, {})
    tool = _tool()
    url = "https://api.osv.dev/v1/vulns/GHSA-xxxx"

    first = _fetch(tool, url)
    again = _fetch(tool, url, timeout_seconds=60)
    same_page = _fetch(tool, "https://API.osv.dev/v1/vulns/GHSA-xxxx#details")

    assert first.startswith("SUCCESS")
    assert first in again and first in same_page
    assert web.requests == [url]


def test_the_fetch_cache_is_not_shared_across_runs(monkeypatch, tmp_path, run_a):
    web = _Web(monkeypatch, tmp_path, {})
    tool = _tool()
    url = "https://example.com/page"

    _fetch(tool, url)
    tokens = set_active_context("run-b", "Run 1")
    try:
        _fetch(tool, url)
    finally:
        reset_active_context(tokens)
    # A tool list built for a new run starts empty too.
    _fetch(_tool(), url)

    assert web.requests == [url, url, url]


@pytest.mark.parametrize("status", [401, 404, 405, 410, 422])
def test_permanent_http_failures_fail_fast_and_are_not_refetched(monkeypatch, tmp_path, run_a, status):
    url = "https://example.com/missing"
    web = _Web(monkeypatch, tmp_path, {url: status})
    tool = _tool()

    first = _fetch(tool, url)
    again = _fetch(tool, url)

    assert first.startswith(f"ERROR: URL returned HTTP {status}")
    assert ors.FETCH_PERMANENT_NOTE in first
    assert first in again
    assert web.requests == [url]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_http_failures_stay_retryable(monkeypatch, tmp_path, run_a, status):
    url = "https://example.com/busy"
    web = _Web(monkeypatch, tmp_path, {url: status})
    tool = _tool()

    first = _fetch(tool, url)
    web.statuses[url] = 200
    second = _fetch(tool, url)

    assert first.startswith(f"ERROR: URL returned HTTP {status}")
    assert ors.FETCH_PERMANENT_NOTE not in first
    assert second.startswith("SUCCESS")
    assert web.requests == [url, url]


def test_timeouts_stay_retryable(monkeypatch, tmp_path, run_a):
    web = _Web(monkeypatch, tmp_path, {})
    calls = {"n": 0}

    def flaky(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ors.requests.exceptions.Timeout()
        return web._get(url, **kwargs)

    monkeypatch.setattr(ors.requests, "get", flaky)
    tool = _tool()

    assert _fetch(tool, "https://example.com/slow") == "ERROR: URL fetch timed out"
    assert _fetch(tool, "https://example.com/slow").startswith("SUCCESS")


def test_the_fetch_cache_is_bounded(monkeypatch, tmp_path, run_a):
    web = _Web(monkeypatch, tmp_path, {})
    monkeypatch.setattr(ors, "FETCH_CACHE_ENTRIES", 2)
    tool = _tool()

    for name in ("a", "b", "c", "a"):
        _fetch(tool, f"https://example.com/{name}")

    # "a" was evicted by "c", so it is fetched again.
    assert web.requests == [f"https://example.com/{name}" for name in ("a", "b", "c", "a")]
