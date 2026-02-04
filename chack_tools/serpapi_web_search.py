from __future__ import annotations

from typing import Optional

import requests
from langchain_core.tools import StructuredTool

from .config import ToolsConfig


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_snippet(text: str, max_chars: int = 240) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


class SerpApiWebSearchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _api_key(self) -> str:
        return str(self.config.serpapi_api_key or "").strip()

    def _max_results(self, requested: Optional[int] = None) -> int:
        default_max = _coerce_int(getattr(self.config, "serpapi_web_max_results", 6), 6)
        if requested is None:
            return _clamp(default_max, 1, 10)
        return _clamp(_coerce_int(requested, default_max), 1, 10)

    def _request(self, params: dict, timeout_seconds: int = 20, max_results: Optional[int] = None) -> str:
        api_key = self._api_key()
        if not api_key:
            return "ERROR: SerpAPI key not configured."
        req_params = dict(params)
        req_params["api_key"] = api_key
        req_params["output"] = "json"
        try:
            response = requests.get("https://serpapi.com/search", params=req_params, timeout=timeout_seconds)
        except requests.exceptions.Timeout:
            return "ERROR: SerpAPI request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to SerpAPI"
        if response.status_code >= 400:
            body = (response.text or "").strip().replace("\n", " ")
            if len(body) > 220:
                body = body[:217] + "..."
            detail = f" ({body})" if body else ""
            return f"ERROR: SerpAPI returned HTTP {response.status_code}{detail}"
        try:
            payload = response.json()
        except ValueError:
            return "ERROR: SerpAPI returned invalid JSON"

        if isinstance(payload, dict) and payload.get("error"):
            return f"ERROR: SerpAPI error ({payload.get('error')})"
        results = payload.get("organic_results") if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return "ERROR: Unexpected SerpAPI response format"
        if not results:
            return f"SUCCESS: No SerpAPI results found for '{params.get('q', '')}'."

        shown = results[: self._max_results(max_results)]
        engine = str(params.get("engine", "serpapi"))
        lines = [f"SUCCESS: SerpAPI {engine} web results for '{params.get('q', '')}' (top {len(shown)}):"]
        for idx, item in enumerate(shown, start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "(no title)"
            url = item.get("link") or item.get("tracking_link") or ""
            snippet = _normalize_snippet(item.get("snippet") or item.get("description") or "")
            meta = []
            if item.get("source"):
                meta.append(str(item["source"]))
            if item.get("date"):
                meta.append(f"date: {item['date']}")
            if item.get("position"):
                meta.append(f"pos: {item['position']}")
            lines.append(f"{idx}. {title} - {url}")
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if snippet:
                lines.append(f"   {snippet}")
        return "\n".join(lines)

    def search_google_web(
        self,
        query: str,
        page: int = 1,
        num: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        max_results = self._max_results(num)
        page = max(1, _coerce_int(page, 1))
        return self._request(
            {
                "engine": "google",
                "q": query,
                "num": max_results,
                "start": (page - 1) * max_results,
            },
            timeout_seconds=timeout_seconds,
            max_results=max_results,
        )

    def search_bing_web(
        self,
        query: str,
        page: int = 1,
        count: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        max_results = self._max_results(count)
        page = max(1, _coerce_int(page, 1))
        return self._request(
            {
                "engine": "bing",
                "q": query,
                "count": max_results,
                "first": ((page - 1) * max_results) + 1,
            },
            timeout_seconds=timeout_seconds,
            max_results=max_results,
        )


def build_google_web_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = SerpApiWebSearchTool(config)

    def _search_google_web(
        query: str,
        page: int = 1,
        num: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search Google web results via SerpAPI.

        Args:
            query: Search query string.
            page: Result page (1+).
            num: Number of results (1-10). Defaults to config value.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.search_google_web(
            query=query,
            page=page,
            num=num,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="search_google_web",
        description=_search_google_web.__doc__ or "Search Google web results via SerpAPI.",
        func=_search_google_web,
    )


def build_bing_web_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = SerpApiWebSearchTool(config)

    def _search_bing_web(
        query: str,
        page: int = 1,
        count: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search Bing web results via SerpAPI.

        Args:
            query: Search query string.
            page: Result page (1+).
            count: Number of results (1-10). Defaults to config value.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.search_bing_web(
            query=query,
            page=page,
            count=count,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="search_bing_web",
        description=_search_bing_web.__doc__ or "Search Bing web results via SerpAPI.",
        func=_search_bing_web,
    )
