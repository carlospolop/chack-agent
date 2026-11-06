"""Web search for model CLIs whose own hosted search cannot run.

Codex's and Claude Code's web search tools run on their vendor's API, so they
fail when the model is reached through a gateway or without the vendor login.
Those runs get Exa's hosted MCP server instead: it offers ``web_search_exa``
and ``web_fetch_exa`` and works without an account (``EXA_API_KEY`` raises its
rate limits). Backends that call the model API directly get the same search
as chack's own ``web_search``/``web_fetch`` tools.

``tools.native_web_search_enabled = false`` turns web search off everywhere.
``tools.keyless_web_search_enabled = false`` keeps the model's own hosted
search but never sends searches to Exa.
"""

from __future__ import annotations

from typing import Any

from chack_tools.exa_web_search import exa_mcp_url

FALLBACK_WEB_MCP_SERVER = "web_search"


def web_search_enabled(config: Any) -> bool:
    return bool(getattr(getattr(config, "tools", None), "native_web_search_enabled", True))


def keyless_web_search_enabled(config: Any) -> bool:
    """Whether a run without hosted web search may use the keyless Exa search."""
    tools = getattr(config, "tools", None)
    return web_search_enabled(config) and bool(getattr(tools, "keyless_web_search_enabled", True))


def fallback_web_mcp_url() -> str:
    return exa_mcp_url()
