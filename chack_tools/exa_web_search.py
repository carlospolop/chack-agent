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
import threading
import time
from typing import List, Optional
from urllib.parse import quote

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests

from .run_cache import RunScopedCache
from .telemetry import run_with_tool_logging

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
MAX_SEARCH_RESULTS = 10
MAX_FETCH_URLS = 5
MAX_FETCH_CHARACTERS = 20000
DEFAULT_FETCH_CHARACTERS = 6000
# Searches from one process are spaced at least this far apart; a search that
# would wait longer than the max wait is refused instead of queued.
_DEFAULT_SEARCH_MIN_INTERVAL_SECONDS = 1.0
_DEFAULT_SEARCH_MAX_WAIT_SECONDS = 10.0
SEARCH_CACHE_ENTRIES = 64
RATE_LIMITED = "ERROR: rate limited by the keyless web search (Exa); retry later or use other sources."

_pace_lock = threading.Lock()
_next_search_at = 0.0


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def _reserve_search_slot() -> Optional[float]:
    """Seconds to wait before the next search, or None when the wait is too long."""
    global _next_search_at
    interval = _float_env("CHACK_KEYLESS_SEARCH_MIN_INTERVAL_SECONDS", _DEFAULT_SEARCH_MIN_INTERVAL_SECONDS)
    max_wait = _float_env("CHACK_KEYLESS_SEARCH_MAX_WAIT_SECONDS", _DEFAULT_SEARCH_MAX_WAIT_SECONDS)
    with _pace_lock:
        now = time.monotonic()
        slot = max(now, _next_search_at)
        wait = slot - now
        if wait > max_wait:
            return None
        _next_search_at = slot + interval
    return wait


def exa_mcp_url() -> str:
    key = os.environ.get("EXA_API_KEY", "").strip()
    return f"{EXA_MCP_URL}?exaApiKey={quote(key, safe='')}" if key else EXA_MCP_URL


def _rpc_message(response: requests.Response) -> dict:
    """The JSON-RPC message of a plain JSON or a server-sent-events reply."""
    # Server-sent events and JSON are UTF-8; requests would decode a charset-less text/event-stream as ISO-8859-1,
    # garbling non-ASCII text.
    if "charset" not in response.headers.get("content-type", "").lower():
        response.encoding = "utf-8"
    body = response.text
    if "text/event-stream" in response.headers.get("content-type", ""):
        # Event lines end only at CR/LF; splitlines() would also cut a data line at a separator such as U+2028
        # or U+0085 inside the result text and drop the rest of the message.
        lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        data = [line[5:].lstrip() for line in lines if line.startswith("data:")]
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
        return RATE_LIMITED
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
    wait = _reserve_search_slot()
    if wait is None:
        return RATE_LIMITED
    if wait > 0:
        time.sleep(wait)
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

    # One cache per built tool list (one run), keyed by run scope too.
    search_cache = RunScopedCache(SEARCH_CACHE_ENTRIES)

    def cached_web_search(query: str, num_results: Optional[int], objective: str) -> str:
        key = (" ".join((query or "").lower().split()), int(num_results or 5), " ".join((objective or "").lower().split()))
        cached = search_cache.get(key)
        if cached is not None:
            return cached
        result = web_search(query, num_results, objective)
        if not result.startswith("ERROR:"):
            search_cache.put(key, result)
        return result

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
            return run_with_tool_logging(
                "web_search", tool_input, lambda: cached_web_search(query, num_results, objective)
            )
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
