import os
from typing import Any

import requests
from langchain_core.tools import StructuredTool

from .config import ToolsConfig
from .serpapi_keys import is_serpapi_rate_limited, shuffled_serpapi_keys


_FORUM_TIME_OPTIONS = {"", "hour", "day", "week", "month", "year"}
_INSTAGRAM_SORT_OPTIONS = {"recent", "top"}
_LINKEDIN_SORT_OPTIONS = {"date_posted", "relevance"}
_REDDIT_POSTS_SORT_OPTIONS = {"hot", "new", "relevance", "top"}
_REDDIT_COMMENTS_SORT_OPTIONS = {"created_utc", "score"}
_X_SORT_OPTIONS = {"Latest", "Top"}


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_snippet(text: str, max_chars: int = 220) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


class ForumScoutTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _api_key(self) -> str:
        return self.config.forumscout_api_key or os.environ.get("FORUMSCOUT_API_KEY", "")

    def _base_url(self) -> str:
        return (
            self.config.forumscout_base_url
            or os.environ.get("FORUMSCOUT_BASE_URL", "")
            or "https://forumscout.app"
        ).rstrip("/")

    def _serpapi_key(self) -> str:
        keys = shuffled_serpapi_keys(getattr(self.config, "serpapi_api_key", ""))
        return keys[0] if keys else ""

    def _request(
        self,
        endpoint: str,
        query: str,
        params: dict[str, Any],
        timeout_seconds: int = 20,
    ) -> str:
        api_key = self._api_key()
        if not api_key:
            return "ForumScout API key not configured."
        if not query.strip():
            return "ERROR: Query cannot be empty"

        headers = {
            "Accept": "application/json",
            "X-API-Key": api_key,
        }
        url = f"{self._base_url()}{endpoint}"
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout_seconds)
        except requests.exceptions.Timeout:
            return "ERROR: ForumScout request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to ForumScout"

        if response.status_code >= 400:
            body = (response.text or "").strip().replace("\n", " ")
            if len(body) > 220:
                body = body[:217] + "..."
            detail = f" ({body})" if body else ""
            return f"ERROR: ForumScout returned HTTP {response.status_code}{detail}"

        try:
            payload = response.json()
        except ValueError:
            return "ERROR: ForumScout returned invalid JSON"

        results = payload if isinstance(payload, list) else payload.get("results", [])
        if not isinstance(results, list):
            return "ERROR: Unexpected ForumScout response format"
        if not results:
            return f"SUCCESS: No ForumScout results found for '{query}'."

        max_results = _clamp(_coerce_int(self.config.forumscout_max_results, 6), 1, 20)
        shown = results[:max_results]
        lines = [f"SUCCESS: ForumScout results for '{query}' (top {len(shown)}):"]
        for idx, item in enumerate(shown, start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "(no title)"
            url = item.get("url") or ""
            snippet = _normalize_snippet(item.get("snippet") or "")
            meta = []
            if item.get("source"):
                meta.append(str(item["source"]))
            if item.get("author"):
                meta.append(f"author: {item['author']}")
            if item.get("date"):
                meta.append(f"date: {item['date']}")
            lines.append(f"{idx}. {title} - {url}")
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if snippet:
                lines.append(f"   {snippet}")
        return "\n".join(lines)

    def _serpapi_request(self, params: dict[str, Any], timeout_seconds: int = 20) -> str:
        api_keys = shuffled_serpapi_keys(getattr(self.config, "serpapi_api_key", ""))
        if not api_keys:
            return "ERROR: SerpAPI key not configured."
        payload = None
        for idx, api_key in enumerate(api_keys):
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
                if is_serpapi_rate_limited(response.status_code, body) and idx < len(api_keys) - 1:
                    continue
                detail = f" ({body})" if body else ""
                return f"ERROR: SerpAPI returned HTTP {response.status_code}{detail}"
            try:
                payload = response.json()
            except ValueError:
                return "ERROR: SerpAPI returned invalid JSON"
            if isinstance(payload, dict) and payload.get("error"):
                error_text = str(payload.get("error") or "")
                if is_serpapi_rate_limited(response.status_code, error_text) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI error ({error_text})"
            break
        if payload is None:
            return "ERROR: All configured SerpAPI keys are rate limited."
        engine = str(params.get("engine", "") or "").strip().lower()
        result_key_by_engine = {
            "google_news": "news_results",
            "google_forums": "organic_results",
        }
        result_key = result_key_by_engine.get(engine, "organic_results")
        results = payload.get(result_key) if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return "ERROR: Unexpected SerpAPI response format"
        if not results:
            return f"SUCCESS: No SerpAPI results found for '{params.get('q', '')}'."
        max_results = _clamp(_coerce_int(self.config.forumscout_max_results, 6), 1, 20)
        shown = results[:max_results]
        source = str(params.get("engine", "serpapi"))
        lines = [f"SUCCESS: SerpAPI {source} results for '{params.get('q', '')}' (top {len(shown)}):"]
        for idx, item in enumerate(shown, start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "(no title)"
            url = item.get("link") or item.get("serpapi_link") or ""
            snippet = _normalize_snippet(item.get("snippet") or "")
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

    def forum_search(
        self,
        query: str,
        time: str = "",
        country: str = "",
        page: int = 1,
        timeout_seconds: int = 20,
    ) -> str:
        if time not in _FORUM_TIME_OPTIONS:
            return "ERROR: time must be one of '', hour, day, week, month, year"
        if country and len(country.strip()) != 2:
            return "ERROR: country must be an ISO 3166-1 alpha-2 code (e.g., us)"
        page = max(1, _coerce_int(page, 1))
        params = {
            "keyword": query,
            "time": time,
            "country": country.lower(),
            "page": page,
        }
        return self._request("/api/forum_search", query=query, params=params, timeout_seconds=timeout_seconds)

    def linkedin_search(
        self,
        query: str,
        page: int = 1,
        sort_by: str = "date_posted",
        timeout_seconds: int = 20,
    ) -> str:
        if sort_by not in _LINKEDIN_SORT_OPTIONS:
            return "ERROR: sort_by must be one of date_posted, relevance"
        page = max(1, _coerce_int(page, 1))
        params = {"keyword": query, "page": page, "sort_by": sort_by}
        return self._request("/api/linkedin_search", query=query, params=params, timeout_seconds=timeout_seconds)

    def instagram_search(
        self,
        query: str,
        page: int = 1,
        sort_by: str = "recent",
        timeout_seconds: int = 20,
    ) -> str:
        if sort_by not in _INSTAGRAM_SORT_OPTIONS:
            return "ERROR: sort_by must be one of recent, top"
        page = max(1, _coerce_int(page, 1))
        params = {"keyword": query, "page": page, "sort_by": sort_by}
        return self._request("/api/instagram_search", query=query, params=params, timeout_seconds=timeout_seconds)

    def reddit_posts_search(
        self,
        query: str,
        page: int = 1,
        sort_by: str = "new",
        timeout_seconds: int = 20,
    ) -> str:
        if sort_by not in _REDDIT_POSTS_SORT_OPTIONS:
            return "ERROR: sort_by must be one of hot, new, relevance, top"
        page = max(1, _coerce_int(page, 1))
        params = {"keyword": query, "page": page, "sort_by": sort_by}
        return self._request(
            "/api/reddit_posts_search",
            query=query,
            params=params,
            timeout_seconds=timeout_seconds,
        )

    def reddit_comments_search(
        self,
        query: str,
        page: int = 1,
        sort_by: str = "created_utc",
        timeout_seconds: int = 20,
    ) -> str:
        if sort_by not in _REDDIT_COMMENTS_SORT_OPTIONS:
            return "ERROR: sort_by must be one of created_utc, score"
        page = max(1, _coerce_int(page, 1))
        params = {"keyword": query, "page": page, "sort_by": sort_by}
        return self._request(
            "/api/reddit_comments_search",
            query=query,
            params=params,
            timeout_seconds=timeout_seconds,
        )

    def x_search(
        self,
        query: str,
        page: int = 1,
        sort_by: str = "Latest",
        timeout_seconds: int = 20,
    ) -> str:
        if sort_by not in _X_SORT_OPTIONS:
            return "ERROR: sort_by must be one of Latest, Top"
        page = max(1, _coerce_int(page, 1))
        params = {"keyword": query, "page": page, "sort_by": sort_by}
        return self._request("/api/x_search", query=query, params=params, timeout_seconds=timeout_seconds)

    def search_google_forums(
        self,
        query: str,
        page: int = 1,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        page = max(1, _coerce_int(page, 1))
        return self._serpapi_request(
            {
                "engine": "google_forums",
                "q": query,
                "page": page,
            },
            timeout_seconds=timeout_seconds,
        )

    def search_google_news(
        self,
        query: str,
        page: int = 1,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        page = max(1, _coerce_int(page, 1))
        return self._serpapi_request(
            {
                "engine": "google_news",
                "q": query,
                "start": max(0, (page - 1) * 10),
            },
            timeout_seconds=timeout_seconds,
        )


def build_forum_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ForumScoutTool(config)

    def _forum_search(
        query: str,
        time: str = "",
        country: str = "",
        page: int = 1,
        timeout_seconds: int = 20,
    ) -> str:
        """Search forums via ForumScout.

        Args:
            query: Search keyword.
            time: One of '', hour, day, week, month, year.
            country: Optional ISO 3166-1 alpha-2 code (e.g., us).
            page: Page number (1+).
            timeout_seconds: Request timeout in seconds.
        """
        return helper.forum_search(
            query=query,
            time=time,
            country=country,
            page=page,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="forum_search",
        description=_forum_search.__doc__ or "Search forums via ForumScout.",
        func=_forum_search,
    )


def build_linkedin_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ForumScoutTool(config)

    def _linkedin_search(
        query: str,
        page: int = 1,
        sort_by: str = "date_posted",
        timeout_seconds: int = 20,
    ) -> str:
        """Search LinkedIn posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: One of date_posted, relevance.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.linkedin_search(
            query=query,
            page=page,
            sort_by=sort_by,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="linkedin_search",
        description=_linkedin_search.__doc__ or "Search LinkedIn posts via ForumScout.",
        func=_linkedin_search,
    )


def build_instagram_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ForumScoutTool(config)

    def _instagram_search(
        query: str,
        page: int = 1,
        sort_by: str = "recent",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Instagram posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: One of recent, top.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.instagram_search(
            query=query,
            page=page,
            sort_by=sort_by,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="instagram_search",
        description=_instagram_search.__doc__ or "Search Instagram posts via ForumScout.",
        func=_instagram_search,
    )


def build_reddit_posts_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ForumScoutTool(config)

    def _reddit_posts_search(
        query: str,
        page: int = 1,
        sort_by: str = "new",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Reddit posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: One of hot, new, relevance, top.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.reddit_posts_search(
            query=query,
            page=page,
            sort_by=sort_by,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="reddit_posts_search",
        description=_reddit_posts_search.__doc__ or "Search Reddit posts via ForumScout.",
        func=_reddit_posts_search,
    )


def build_reddit_comments_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ForumScoutTool(config)

    def _reddit_comments_search(
        query: str,
        page: int = 1,
        sort_by: str = "created_utc",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Reddit comments via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: One of created_utc, score.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.reddit_comments_search(
            query=query,
            page=page,
            sort_by=sort_by,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="reddit_comments_search",
        description=_reddit_comments_search.__doc__ or "Search Reddit comments via ForumScout.",
        func=_reddit_comments_search,
    )


def build_x_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ForumScoutTool(config)

    def _x_search(
        query: str,
        page: int = 1,
        sort_by: str = "Latest",
        timeout_seconds: int = 20,
    ) -> str:
        """Search X (Twitter) posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: One of Latest, Top.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.x_search(
            query=query,
            page=page,
            sort_by=sort_by,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="x_search",
        description=_x_search.__doc__ or "Search X (Twitter) posts via ForumScout.",
        func=_x_search,
    )
