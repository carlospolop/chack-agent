"""Which web tools a run that calls the model API directly gets.

The provider's own hosted search and page reading come first: OpenAI's
``web_search`` tool, OpenRouter's ``openrouter:web_search`` and
``openrouter:web_fetch`` server tools, or a custom gateway's ``web_search``
when the gateway serves it for the model. Whatever the provider cannot do is
covered by chack's keyless ``web_search``/``web_fetch`` tools (Exa), so a run
with web enabled always has both.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from ..llm_gateway import GatewayRoute
from .web_search_fallback import keyless_web_search_enabled, web_search_enabled

# Models whose provider search OpenRouter runs natively; for every other model
# OpenRouter would bill an Exa search, which the keyless tool does for free.
_OPENROUTER_NATIVE_SEARCH = re.compile(
    r"^(?:openai/(?:gpt-4\.1|gpt-[5-9]|o[3-9])"
    r"|anthropic/claude-(?:3\.5-haiku|3\.7-sonnet|(?:opus|sonnet|haiku)-[4-9]|[4-9])"
    r"|google/gemini-[3-9]"
    r"|x-ai/grok-[4-9]"
    r"|perplexity/)"
)
# Server tool results enter the model's context unseen by chack's output
# limits, so they are bounded here.
_OPENROUTER_SEARCH_TOOL = {
    "type": "openrouter:web_search",
    "parameters": {"engine": "native", "max_results": 5, "search_context_size": "medium"},
}
_OPENROUTER_FETCH_TOOL = {
    # ``native`` falls back to OpenRouter's own free fetch, never a billed engine.
    "type": "openrouter:web_fetch",
    "parameters": {"engine": "native", "max_uses": 10, "max_content_tokens": 8000},
}

_PROBE_TTL_SECONDS = 3600
_probe_lock = threading.Lock()
_probe_cache: dict[tuple[str, str], tuple[float, bool]] = {}


@dataclass(frozen=True)
class WebToolPlan:
    # Tools the provider runs itself: raw tool dicts sent with each request.
    server_tools: tuple[dict, ...] = ()
    # OpenAI-style hosted ``web_search`` (the Agents SDK ``WebSearchTool``).
    hosted_web_search: bool = False
    # chack's keyless tools still needed: a subset of {"web_search", "web_fetch"}.
    keyless_tools: frozenset[str] = field(default_factory=frozenset)


def openrouter_has_native_search(model_name: str) -> bool:
    return bool(_OPENROUTER_NATIVE_SEARCH.match(str(model_name or "").strip().lower()))


def gateway_serves_hosted_web_search(route: GatewayRoute) -> bool:
    """Whether a custom gateway accepts the ``web_search`` tool for this model.

    Gateways pass it to the model's provider, which may not offer or allow it,
    and some only refuse once a search actually runs. So one small request
    forces a search and checks that it finished; the answer is kept for an
    hour. ``CHACK_GATEWAY_HOSTED_WEB_SEARCH=on|off`` skips the check.
    """
    forced = os.environ.get("CHACK_GATEWAY_HOSTED_WEB_SEARCH", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False
    key = (route.base_url, route.model_name)
    now = time.monotonic()
    with _probe_lock:
        cached = _probe_cache.get(key)
        if cached and now - cached[0] < _PROBE_TTL_SECONDS:
            return cached[1]
    try:
        response = requests.post(
            route.base_url.rstrip("/") + "/responses",
            json={
                "model": route.model_name,
                "input": "Search the web for example.com and reply with its page title.",
                "tools": [{"type": "web_search"}],
                "tool_choice": "required",
                # Room for the search to run: a cut-off search is never refused.
                "max_output_tokens": 200,
            },
            headers={"Authorization": f"Bearer {route.api_key}", **dict(route.headers)},
            timeout=60,
        )
        body = (response.json() or {}) if response.status_code == 200 else {}
        supported = not body.get("error") and any(
            item.get("type") == "web_search_call" and item.get("status") not in {"failed", "incomplete"}
            for item in body.get("output") or []
        )
    except Exception:
        supported = False
    with _probe_lock:
        _probe_cache[key] = (now, supported)
    return supported


def keyless_web_plan(config: Any) -> WebToolPlan:
    """Only chack's keyless tools, for providers that run no web tools."""
    if not keyless_web_search_enabled(config):
        return WebToolPlan()
    return WebToolPlan(keyless_tools=frozenset({"web_search", "web_fetch"}))


def plan_web_tools(config: Any, *, route: GatewayRoute | None, responses_api: bool) -> WebToolPlan:
    """The web tools for a run on ``route`` (``None``: OpenAI's own API).

    ``responses_api`` is false for chat-completions runs, which cannot carry
    OpenAI's hosted ``web_search`` tool.
    """
    if not web_search_enabled(config):
        return WebToolPlan()
    keyless = {"web_search", "web_fetch"} if keyless_web_search_enabled(config) else set()
    if route is None:
        # OpenAI's web_search also opens and reads pages.
        return WebToolPlan(hosted_web_search=True)
    if route.is_openrouter:
        server_tools = [_OPENROUTER_FETCH_TOOL]
        keyless.discard("web_fetch")
        if openrouter_has_native_search(route.model_name):
            server_tools.insert(0, _OPENROUTER_SEARCH_TOOL)
            keyless.discard("web_search")
        return WebToolPlan(server_tools=tuple(server_tools), keyless_tools=frozenset(keyless))
    if responses_api and gateway_serves_hosted_web_search(route):
        keyless.discard("web_search")
        return WebToolPlan(hosted_web_search=True, keyless_tools=frozenset(keyless))
    return keyless_web_plan(config)
