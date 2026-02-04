import os
import re
import time
import random
from typing import Optional

import requests
from langchain_core.tools import StructuredTool, tool

from .config import ToolsConfig


_FRESHNESS_PATTERN = re.compile(r"^\\d{4}-\\d{2}-\\d{2}to\\d{4}-\\d{2}-\\d{2}$")


def _normalize_freshness(value: str) -> Optional[str]:
    if not value:
        return None
    value = value.strip().lower()
    if value in {"pd", "pw", "pm", "py"}:
        return value
    if _FRESHNESS_PATTERN.match(value):
        return value
    return None


class BraveSearchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def search(
        self,
        query: str,
        count: Optional[int] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
        timeout_seconds: int = 20,
    ) -> str:
        return self._brave_search_impl(
            query=query,
            count=count,
            country=country,
            search_lang=search_lang,
            ui_lang=ui_lang,
            freshness=freshness,
            timeout_seconds=timeout_seconds,
        )

    def _brave_search_impl(
        self,
        query: str,
        count: Optional[int] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
        timeout_seconds: int = 20,
    ) -> str:
        api_key = self.config.brave_api_key or os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return "Brave API key not configured."
        if not query.strip():
            return "ERROR: Query cannot be empty"
        if count is None:
            count = self.config.brave_max_results
        if count < 1:
            count = 1
        if count > 20:
            count = 20
        normalized_freshness = _normalize_freshness(freshness) if freshness else None
        if freshness and not normalized_freshness:
            return (
                "ERROR: freshness must be one of pd, pw, pm, py, or a range like "
                "YYYY-MM-DDtoYYYY-MM-DD"
            )
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        params = {
            "q": query,
            "count": count,
        }
        if country:
            params["country"] = country
        if search_lang:
            params["search_lang"] = search_lang
        if ui_lang:
            params["ui_lang"] = ui_lang
        if normalized_freshness:
            params["freshness"] = normalized_freshness
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=timeout_seconds,
        )
        if response.status_code == 429:
            #Sleep random time from 0 to 10s and retry once
            time.sleep(random.uniform(0, 10))
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        web_results = payload.get("web", {}).get("results", [])
        results = []
        for entry in web_results[: self.config.brave_max_results]:
            title = entry.get("title") or "(no title)"
            url = entry.get("url") or ""
            snippet = entry.get("description") or ""
            results.append(f"- {title}: {url}\n  {snippet}")
        return "\n".join(results) if results else "No results."


def build_brave_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = BraveSearchTool(config)

    def _brave_search(
        query: str,
        count: Optional[int] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search Brave Search API and return a short list of results.

        Args:
            query: Search query string.
            count: Optional number of results to return (1-20).
            country: Optional country code (e.g., "US").
            search_lang: Optional search language (e.g., "en").
            ui_lang: Optional UI language (e.g., "en-US").
            freshness: Optional freshness filter (pd, pw, pm, py, or YYYY-MM-DDtoYYYY-MM-DD).
            timeout_seconds: Request timeout in seconds.
        """
        return helper._brave_search_impl(
            query=query,
            count=count,
            country=country,
            search_lang=search_lang,
            ui_lang=ui_lang,
            freshness=freshness,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="brave_search",
        description=_brave_search.__doc__ or "Search Brave Search API.",
        func=_brave_search,
    )
