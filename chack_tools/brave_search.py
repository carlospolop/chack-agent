import os
import re
import time
from datetime import datetime, timezone
import random
from typing import Optional

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests

from .config import ToolsConfig
from .telemetry import log_tool_started, log_tool_executed



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
        api_key = os.environ.get("BRAVE_API_KEY", "")
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
        return "\n".join(results)


def get_brave_search_tool(helper: BraveSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="brave_search")
    def brave_search(
        query: str,
        count: Optional[int] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search the web via Brave and return a short list of results.

        Use this to verify facts, commands, API behavior, and any information that may have changed.
        Prefer it for quick confirmation or a complementary source alongside Google/Bing.

        Args:
            query: Search query string.
            count: Optional number of results to return (1-20).
            country: Optional country code (e.g., "US").
            search_lang: Optional search language (e.g., "en").
            ui_lang: Optional UI language (e.g., "en-US").
            freshness: Optional freshness filter (pd, pw, pm, py, or YYYY-MM-DDtoYYYY-MM-DD).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "count": count,
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "freshness": freshness,
            "timeout_seconds": timeout_seconds,
        }
        start_ts = log_tool_started("brave_search", tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.search(
                query=query,
                count=count,
                country=country,
                search_lang=search_lang,
                ui_lang=ui_lang,
                freshness=freshness,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return f"ERROR: Brave search failed ({exc})"
        finally:
            end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            duration_ms = int((time.time() - start_time) * 1000)
            log_tool_executed(
                "brave_search",
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    return brave_search
