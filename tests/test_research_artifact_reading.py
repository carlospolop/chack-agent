"""Reading saved research artifacts: bounded slices that can always continue,
targeted windows, and no re-sent text for identical repeats."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from agents.tool_context import ToolContext
from agents.usage import Usage

from chack_tools import open_research_sources as ors
from chack_tools.config import ToolsConfig
from chack_tools.research_artifacts import ResearchArtifactsTool, get_research_artifact_tools
from chack_tools.task_steps_manager_state import reset_active_context, set_active_context


def _page(sections: int) -> str:
    parts = ["<html><head><title>Advisory</title><style>p{}</style></head><body><nav>Home | Docs</nav>"]
    for i in range(sections):
        parts.append(
            f"<h2>Section {i}</h2>\n<p>Paragraph {i} explains the behaviour of\n component {i} "
            f"in detail, with enough words to look like a real page. {'filler ' * 20}</p>"
        )
    parts.append("<p>CVE-2099-0001 affects version 3.7.0 only.</p></body></html>")
    return "".join(parts)


class _Response:
    def __init__(self, url, body, content_type="text/html"):
        self.url, self.status_code, self.encoding = url, 200, "utf-8"
        self.headers = {"content-type": content_type}
        self._body = body.encode()

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        pass


@pytest.fixture
def evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ors, "_is_public_https_url", lambda url: True)
    tokens = set_active_context("run-a", "Run 1")
    yield tmp_path
    reset_active_context(tokens)


def _fetch(monkeypatch, evidence, body, content_type="text/html") -> str:
    monkeypatch.setattr(ors.requests, "get", lambda url, **kw: _Response(url, body, content_type))
    out = ors.OpenResearchTool(ToolsConfig()).fetch_url_text("https://example.com/advisory")
    assert out.startswith("SUCCESS"), out
    path = next(line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("Saved text:"))
    return str(__import__("pathlib").Path(path).relative_to(evidence))


def _tools():
    return {tool.name: tool for tool in get_research_artifact_tools(ResearchArtifactsTool(ToolsConfig()))}


def _call(tool, **args) -> str:
    raw = json.dumps(args)
    ctx = ToolContext(context=None, usage=Usage(), tool_name=tool.name, tool_call_id="t", tool_arguments=raw)
    return asyncio.run(tool.on_invoke_tool(ctx, raw))


def _line_numbers(text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"^(\d+): ", text, re.M)]


def test_fetched_pages_are_saved_one_block_per_line(monkeypatch, evidence):
    rel = _fetch(monkeypatch, evidence, _page(5))
    lines = (evidence / rel).read_text().splitlines()

    assert len(lines) > 10
    assert "CVE-2099-0001 affects version 3.7.0 only." in lines
    assert not any("p{}" in line for line in lines)


def test_fetched_json_is_saved_one_value_per_line(monkeypatch, evidence):
    payload = {"id": "GHSA-1", "affected": [{"package": "jquery", "versions": ["3.7.0", "3.7.1"]}], "details": "x" * 50}
    rel = _fetch(monkeypatch, evidence, json.dumps(payload, separators=(",", ":")), "application/json")
    saved = (evidence / rel).read_text()

    assert len(saved.splitlines()) > 5
    assert json.loads(saved) == payload


def test_reading_from_the_top_is_bounded_and_every_line_stays_reachable(monkeypatch, evidence):
    rel = _fetch(monkeypatch, evidence, _page(120))
    total = len((evidence / rel).read_text().splitlines())
    read = _tools()["read_research_artifact"]

    seen: list[int] = []
    start = 1
    calls = 0
    while True:
        out = _call(read, path=rel, start_line=start)
        calls += 1
        assert len(out) < 6600
        numbers = _line_numbers(out)
        assert numbers[0] == start
        seen.extend(numbers)
        follow = re.search(r"start_line=(\d+)", out)
        if not follow:
            break
        start = int(follow.group(1))
        assert start == numbers[-1] + 1

    assert calls > 1
    assert seen == list(range(1, total + 1))


def test_around_text_returns_the_matching_section_without_repeated_lines(monkeypatch, evidence):
    rel = _fetch(monkeypatch, evidence, _page(120))
    read = _tools()["read_research_artifact"]

    out = _call(read, path=rel, around_text="CVE-2099-0001", context_lines=3)
    numbers = _line_numbers(out)

    assert "affects version 3.7.0" in out
    assert len(numbers) == len(set(numbers)) <= 4
    assert len(out) < 1500

    many = _call(read, path=rel, around_text="Paragraph 1", context_lines=5)
    numbers = _line_numbers(many)
    assert len(numbers) == len(set(numbers))
    assert len(many) < 6600


def test_long_lines_are_windowed_around_the_match(evidence):
    (evidence / "blob.txt").write_text("a" * 50_000 + " NEEDLE here " + "b" * 50_000)
    tools = _tools()

    around = _call(tools["read_research_artifact"], path="blob.txt", around_text="needle")
    grep = _call(tools["grep_research_artifacts"], pattern="NEEDLE")

    for out in (around, grep):
        assert "NEEDLE here" in out
        assert len(out) < 1000


def test_grep_reports_when_it_stops_early(evidence):
    (evidence / "log.txt").write_text("\n".join(f"hit {i}" for i in range(100)))
    grep = _tools()["grep_research_artifacts"]

    out = _call(grep, pattern="hit")
    everything = _call(grep, pattern="hit", max_matches=500, max_chars=100_000)

    assert len(re.findall(r"^log\.txt:", out, re.M)) == 20
    assert "more exist" in out
    assert len(re.findall(r"^log\.txt:", everything, re.M)) == 100
    assert "more exist" not in everything


def test_identical_repeats_get_a_notice_once_then_the_text_again(evidence):
    (evidence / "page.txt").write_text("\n".join(f"line {i} of the evidence page" for i in range(200)))
    tools = _tools()
    read = tools["read_research_artifact"]

    first = _call(read, path="page.txt", start_line=1, end_line=50)
    notice = _call(read, path="page.txt", start_line=1, end_line=50)
    again = _call(read, path="page.txt", start_line=1, end_line=50)

    assert _line_numbers(first) == list(range(1, 51))
    assert _line_numbers(notice) == [] and len(notice) < 300
    assert again == first

    # A changed file is served in full, never answered with a notice.
    (evidence / "page.txt").write_text("\n".join(f"line {i} updated" for i in range(200)))
    changed = _call(read, path="page.txt", start_line=1, end_line=50)
    assert "updated" in changed

    # Another run (session) gets the text, not a notice.
    tokens = set_active_context("run-b", "Run 1")
    try:
        assert _call(read, path="page.txt", start_line=1, end_line=50) == changed
    finally:
        reset_active_context(tokens)


def test_identical_listing_of_an_unchanged_folder_is_not_resent(evidence):
    for i in range(30):
        (evidence / f"file-{i}.txt").write_text("x")
    listing = _tools()["list_research_artifacts"]

    first = _call(listing, glob="*", max_results=200)
    repeat = _call(listing, glob="*", max_results=200)
    (evidence / "new.txt").write_text("y")
    after_change = _call(listing, glob="*", max_results=200)

    assert "file-29.txt" in first
    assert "file-29.txt" not in repeat and len(repeat) < 300
    assert "new.txt" in after_change
