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

    def _request_payload(self, params: dict, timeout_seconds: int = 20):
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
        return payload

    def _request(self, params: dict, timeout_seconds: int = 20, max_results: Optional[int] = None) -> str:
        payload = self._request_payload(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
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

    @staticmethod
    def _extract_text_blocks(payload: dict) -> list[str]:
        blocks = payload.get("text_blocks") or payload.get("answer_blocks") or []
        out: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    text = str(
                        block.get("text")
                        or block.get("snippet")
                        or block.get("content")
                        or ""
                    ).strip()
                    if text:
                        out.append(text)
                elif isinstance(block, str) and block.strip():
                    out.append(block.strip())
        for key in ["answer", "chat_response", "response", "output"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
            elif isinstance(value, dict):
                text = str(value.get("text") or value.get("content") or "").strip()
                if text:
                    out.append(text)
        return out

    @staticmethod
    def _extract_reference_rows(payload: dict) -> list[dict]:
        refs = payload.get("references") or payload.get("citations") or payload.get("sources") or []
        rows: list[dict] = []
        if not isinstance(refs, list):
            refs = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            url = str(ref.get("link") or ref.get("url") or "").strip()
            if not url:
                continue
            rows.append(
                {
                    "title": ref.get("title") or ref.get("source") or "(no title)",
                    "url": url,
                    "snippet": ref.get("snippet") or ref.get("description") or "",
                    "source": ref.get("source") or "",
                }
            )
        if rows:
            return rows
        organic = payload.get("organic_results") or []
        if isinstance(organic, list):
            for item in organic:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("link") or item.get("tracking_link") or "").strip()
                if not url:
                    continue
                rows.append(
                    {
                        "title": item.get("title") or "(no title)",
                        "url": url,
                        "snippet": item.get("snippet") or item.get("description") or "",
                        "source": item.get("source") or "",
                    }
                )
        return rows

    def _format_ai_mode(self, engine: str, query: str, payload: dict) -> str:
        text_blocks = self._extract_text_blocks(payload)
        refs = self._extract_reference_rows(payload)
        if not text_blocks and not refs:
            return f"SUCCESS: No SerpAPI {engine} results found for '{query}'."
        lines = [f"SUCCESS: SerpAPI {engine} results for '{query}':"]
        if text_blocks:
            lines.append("Summary:")
            for block in text_blocks[:4]:
                lines.append(f"- {_normalize_snippet(block, max_chars=300)}")
        if refs:
            shown = refs[: self._max_results()]
            lines.append(f"References (top {len(shown)}):")
            for idx, ref in enumerate(shown, start=1):
                lines.append(f"{idx}. {ref['title']} - {ref['url']}")
                if ref.get("source"):
                    lines.append(f"   {ref['source']}")
                if ref.get("snippet"):
                    lines.append(f"   {_normalize_snippet(str(ref['snippet']))}")
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

    def search_google_ai_mode(
        self,
        query: str,
        timeout_seconds: int = 45,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        payload = self._request_payload(
            {"engine": "google_ai_mode", "q": query},
            timeout_seconds=timeout_seconds,
        )
        if isinstance(payload, str):
            return payload
        return self._format_ai_mode("google_ai_mode", query, payload)

    def search_bing_copilot(
        self,
        query: str,
        timeout_seconds: int = 45,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        payload = self._request_payload(
            {"engine": "bing_copilot", "q": query},
            timeout_seconds=timeout_seconds,
        )
        if isinstance(payload, str):
            return payload
        return self._format_ai_mode("bing_copilot", query, payload)


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


def build_google_ai_mode_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = SerpApiWebSearchTool(config)

    def _search_google_ai_mode(
        query: str,
        timeout_seconds: int = 45,
    ) -> str:
        """Search Google AI Mode via SerpAPI and return summary + references.

        Args:
            query: Search query string.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.search_google_ai_mode(
            query=query,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="search_google_ai_mode",
        description=_search_google_ai_mode.__doc__ or "Search Google AI Mode via SerpAPI.",
        func=_search_google_ai_mode,
    )


def build_bing_copilot_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = SerpApiWebSearchTool(config)

    def _search_bing_copilot(
        query: str,
        timeout_seconds: int = 45,
    ) -> str:
        """Search Bing Copilot via SerpAPI and return summary + references.

        Args:
            query: Search query string.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.search_bing_copilot(
            query=query,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="search_bing_copilot",
        description=_search_bing_copilot.__doc__ or "Search Bing Copilot via SerpAPI.",
        func=_search_bing_copilot,
    )
