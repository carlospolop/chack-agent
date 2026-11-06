"""Keyless web search and page reading through Exa's hosted MCP server.

Backends that call the model API directly have no hosted web search of their
own. They get ``web_search`` and ``web_fetch`` instead, backed by
``https://mcp.exa.ai/mcp``: it needs no account (``EXA_API_KEY`` raises its
rate limits). The server answers stateless ``tools/call`` requests, so each
tool call is a single HTTP POST.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional
from urllib.parse import quote

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests

from .telemetry import run_with_tool_logging

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
MAX_SEARCH_RESULTS = 10
MAX_FETCH_URLS = 5
MAX_FETCH_CHARACTERS = 20000
DEFAULT_FETCH_CHARACTERS = 6000


def exa_mcp_url() -> str:
    key = os.environ.get("EXA_API_KEY", "").strip()
    return f"{EXA_MCP_URL}?exaApiKey={quote(key, safe='')}" if key else EXA_MCP_URL


def _rpc_message(response: requests.Response) -> dict:
    """The JSON-RPC message of a plain JSON or a server-sent-events reply."""
    body = response.text
    if "text/event-stream" in response.headers.get("content-type", ""):
        data = [line[5:].lstrip() for line in body.splitlines() if line.startswith("data:")]
        body = "\n".join(data)
    return json.loads(body, strict=False)


def call_exa_tool(name: str, arguments: dict, timeout_seconds: int = 30) -> str:
    response = requests.post(
        exa_mcp_url(),
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
        timeout=timeout_seconds,
    )
    if response.status_code == 429:
        return "ERROR: the free web search is rate limited right now; retry later or set EXA_API_KEY."
    response.raise_for_status()
    message = _rpc_message(response)
    if "error" in message:
        return f"ERROR: {message['error'].get('message', 'web search failed')}"
    result = message.get("result") or {}
    text = "\n".join(
        str(item.get("text", "")) for item in result.get("content") or [] if item.get("type") == "text"
    ).strip()
    if result.get("isError"):
        return f"ERROR: {text or 'web search failed'}"
    return text or "No results."


def web_search(query: str, num_results: Optional[int] = None, objective: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return "ERROR: query cannot be empty"
    count = max(1, min(int(num_results or 5), MAX_SEARCH_RESULTS))
    return call_exa_tool(
        "web_search_exa",
        {"query": query, "numResults": count, "objective": (objective or "").strip() or query},
    )


def web_fetch(urls: List[str], max_characters: Optional[int] = None) -> str:
    wanted = [url.strip() for url in urls or [] if url and url.strip()]
    if not wanted:
        return "ERROR: give at least one URL"
    if len(wanted) > MAX_FETCH_URLS:
        return f"ERROR: at most {MAX_FETCH_URLS} URLs per call"
    limit = max(1, min(int(max_characters or DEFAULT_FETCH_CHARACTERS), MAX_FETCH_CHARACTERS))
    return call_exa_tool("web_fetch_exa", {"urls": wanted, "maxCharacters": limit})


def get_web_search_tools() -> list:
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="web_search")
    def web_search_tool(query: str, num_results: Optional[int] = None, objective: str = "") -> str:
        """Search the web and return the best matching pages with their key passages.

        Args:
            query: What to look for, phrased as a description of the ideal page.
            num_results: How many results to return (1-10, default 5).
            objective: Optional goal of the search: which pages should rank first and which facts to pull.
        """
        tool_input = {"query": query, "num_results": num_results, "objective": objective}
        try:
            return run_with_tool_logging("web_search", tool_input, lambda: web_search(query, num_results, objective))
        except Exception as exc:
            return f"ERROR: web search failed ({exc})"

    @function_tool(name_override="web_fetch")
    def web_fetch_tool(urls: List[str], max_characters: Optional[int] = None) -> str:
        """Read web pages and return their main text.

        Args:
            urls: One to five URLs to read.
            max_characters: Characters to keep per page (default 6000, at most 20000).
        """
        tool_input = {"urls": urls, "max_characters": max_characters}
        try:
            return run_with_tool_logging("web_fetch", tool_input, lambda: web_fetch(urls, max_characters))
        except Exception as exc:
            return f"ERROR: fetching the pages failed ({exc})"

    return [web_search_tool, web_fetch_tool]
