"""Web search for model CLIs whose own hosted search cannot run.

Codex's and Claude Code's web search tools run on their vendor's API, so they
fail when the model is reached through a gateway or without the vendor login.
Those runs get Exa's hosted MCP server instead: it offers ``web_search_exa``
and ``web_fetch_exa`` and works without an account (``EXA_API_KEY`` raises its
rate limits). ``tools.native_web_search_enabled = false`` turns web search off
everywhere, this fallback included.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
FALLBACK_WEB_MCP_SERVER = "web_search"


def web_search_enabled(config: Any) -> bool:
    return bool(getattr(getattr(config, "tools", None), "native_web_search_enabled", True))


def fallback_web_mcp_url() -> str:
    key = os.environ.get("EXA_API_KEY", "").strip()
    return f"{EXA_MCP_URL}?exaApiKey={quote(key, safe='')}" if key else EXA_MCP_URL
