import json
import os
import re
from typing import Any
from uuid import uuid4

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests

from .config import ToolsConfig
from .research_artifacts import record_research_artifact, record_research_json_artifact, research_artifacts_root
from .serpapi_keys import (
    is_serpapi_rate_limited,
    note_serpapi_response_error,
    usable_serpapi_keys,
)
from .telemetry import run_with_tool_logging



_FORUM_TIME_OPTIONS = {"", "hour", "day", "week", "month", "year"}
_INSTAGRAM_SORT_OPTIONS = {"recent", "top"}
_LINKEDIN_SORT_OPTIONS = {"date_posted", "relevance"}
_REDDIT_POSTS_SORT_OPTIONS = {"hot", "new", "relevance", "top"}
_REDDIT_COMMENTS_SORT_OPTIONS = {"created_utc", "score"}
_X_SORT_OPTIONS = {"Latest", "Top"}
_GOOGLE_TRENDS_DATA_TYPES = {
    "TIMESERIES",
    "GEO_MAP",
    "GEO_MAP_0",
    "RELATED_TOPICS",
    "RELATED_QUERIES",
}
_GOOGLE_TRENDS_GPROPS = {"", "images", "news", "froogle", "youtube"}
_GOOGLE_TRENDS_HOURS = {4, 24, 48, 168}


def _run_logged(tool: str, tool_input: dict, func):
    try:
        return run_with_tool_logging(tool, tool_input, func)
    except Exception as exc:
        return f"ERROR: {tool} failed ({exc})"


def _set_param_descriptions(tool: Any, descriptions: dict[str, str]):
    schema = getattr(tool, "params_json_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, description in descriptions.items():
            if isinstance(properties.get(name), dict):
                properties[name]["description"] = description
    return _with_forum_output(tool)


def _with_forum_output(tool: Any):
    current = str(getattr(tool, "description", "") or "").strip()
    if current and "Output:" not in current:
        tool.description = (
            f"{current}\n\n"
            "Parameters: Use the schema descriptions to provide platform search text, locale, pagination, sort/filter tokens, profile IDs, and request timeouts.\n"
            "Output: Returns compact SUCCESS/ERROR text with social posts, profiles, comments, news, video, trend, or forum results, including source URLs and pagination tokens when available."
        )
    return tool


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


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("title", "name", "query", "value", "text", "url", "link"):
            text = _as_text(value.get(key))
            if text:
                return text
    return ""


def _append_dict_fields(lines: list[str], item: dict[str, Any], fields: list[tuple[str, str]], indent: str = "   ") -> None:
    parts = []
    for key, label in fields:
        value = _as_text(item.get(key))
        if value:
            parts.append(f"{label}: {value}")
    if parts:
        lines.append(f"{indent}{' | '.join(parts)}")


def _item_url(item: dict[str, Any]) -> str:
    return _as_text(
        item.get("link")
        or item.get("url")
        or item.get("source_url")
        or item.get("serpapi_link")
        or item.get("video_link")
    )


def _first_list(payload: dict[str, Any], keys: list[str]) -> list:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _safe_filename(value: str, fallback: str = "social-data") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _artifact_dir(kind: str) -> str:
    root = research_artifacts_root()
    base = os.path.join(root, kind) if root else os.path.join("/tmp", "chack-social", kind)
    os.makedirs(base, exist_ok=True)
    return base


def _write_json_artifact(kind: str, label: str, payload: Any) -> str:
    path = os.path.join(_artifact_dir(kind), f"{_safe_filename(label)}_{uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    record_research_json_artifact(path, payload, provenance=f"{kind}:{label}", kind=kind, label=label)
    return path


def _write_text_artifact(kind: str, label: str, text: str) -> str:
    path = os.path.join(_artifact_dir(kind), f"{_safe_filename(label)}_{uuid4().hex}.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(text or ""))
    record_research_artifact(path, provenance=f"{kind}:{label}", kind=kind, label=label)
    return path


class ForumScoutTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _api_key(self) -> str:
        return os.environ.get("FORUMSCOUT_API_KEY", "")

    def _base_url(self) -> str:
        return os.environ.get("FORUMSCOUT_BASE_URL", "https://forumscout.app")

    def _serpapi_key(self) -> str:
        keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
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
            # Retry once on transient HTML/invalid payloads.
            try:
                response = requests.get(url, headers=headers, params=params, timeout=timeout_seconds)
                payload = response.json()
            except Exception:
                return "ERROR: ForumScout returned invalid JSON"

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return "ERROR: ForumScout returned invalid JSON"

        artifact = _write_json_artifact("forumscout", query, payload)
        results = payload if isinstance(payload, list) else payload.get("results", [])
        if not isinstance(results, list):
            return "ERROR: Unexpected ForumScout response format"
        if not results:
            return f"SUCCESS: No ForumScout results found for '{query}'.\nArtifact JSON: {artifact}"

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
        text_path = _write_text_artifact("forumscout", query, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def _serpapi_payload(self, params: dict[str, Any], timeout_seconds: int = 20) -> Any:
        api_keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if not api_keys:
            return "ERROR: No usable SerpAPI key (not configured or all keys exhausted)."
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
                note_serpapi_response_error(api_key, response.status_code, body)
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
                note_serpapi_response_error(api_key, response.status_code, error_text)
                if is_serpapi_rate_limited(response.status_code, error_text) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI error ({error_text})"
            break
        if payload is None:
            return "ERROR: All configured SerpAPI keys are rate limited."
        return payload

    def _serpapi_request(self, params: dict[str, Any], timeout_seconds: int = 20) -> str:
        payload = self._serpapi_payload(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact(
            f"serpapi-{str(params.get('engine', 'social'))}",
            str(params.get("q") or params.get("profile_id") or params.get("story_token") or "query"),
            payload,
        )
        engine = str(params.get("engine", "") or "").strip().lower()
        result_keys_by_engine = {
            "google_news": ["news_results"],
            "google_forums": ["organic_results"],
            "google_videos": ["video_results", "videos_results"],
        }
        result_keys = result_keys_by_engine.get(engine, ["organic_results"])
        results = _first_list(payload, result_keys) if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return "ERROR: Unexpected SerpAPI response format"
        query_label = (
            str(params.get("q") or "").strip()
            or str(params.get("story_token") or "").strip()
            or str(params.get("section_token") or "").strip()
            or str(params.get("topic_token") or "").strip()
            or str(params.get("publication_token") or "").strip()
            or str(params.get("kgmid") or "").strip()
        )
        if not results:
            return f"SUCCESS: No SerpAPI results found for '{query_label}'.\nArtifact JSON: {artifact}"
        max_results = _clamp(_coerce_int(self.config.forumscout_max_results, 6), 1, 20)
        shown = results[:max_results]
        source = str(params.get("engine", "serpapi"))
        lines = [f"SUCCESS: SerpAPI {source} results for '{query_label}' (top {len(shown)}):"]
        for idx, item in enumerate(shown, start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "(no title)"
            url = _item_url(item)
            snippet = _normalize_snippet(item.get("snippet") or "")
            meta = []
            source_value = item.get("source")
            if isinstance(source_value, dict):
                name = _as_text(source_value.get("name") or source_value.get("title"))
                authors = _as_text(source_value.get("authors"))
                if name:
                    meta.append(name)
                if authors:
                    meta.append(f"authors: {authors}")
            elif source_value:
                meta.append(str(source_value))
            if item.get("date"):
                meta.append(f"date: {item['date']}")
            if item.get("iso_date"):
                meta.append(f"iso_date: {item['iso_date']}")
            if item.get("displayed_link"):
                meta.append(str(item["displayed_link"]))
            if item.get("position"):
                meta.append(f"pos: {item['position']}")
            lines.append(f"{idx}. {title} - {url}")
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if snippet:
                lines.append(f"   {snippet}")
            sitelinks = item.get("sitelinks")
            if isinstance(sitelinks, list) and sitelinks:
                for sub in sitelinks[:3]:
                    if not isinstance(sub, dict):
                        continue
                    subtitle = _as_text(sub.get("title") or sub.get("snippet"))
                    suburl = _item_url(sub)
                    if subtitle or suburl:
                        lines.append(f"   - {subtitle} - {suburl}".rstrip())
        related = payload.get("related_searches") if isinstance(payload, dict) else None
        if isinstance(related, list) and related:
            lines.append("Related searches:")
            for item in related[:8]:
                if isinstance(item, dict):
                    query = _as_text(item.get("query") or item.get("title"))
                    link = _item_url(item)
                    if query:
                        suffix = f" - {link}" if link else ""
                        lines.append(f"- {query}{suffix}")
        menu_links = payload.get("menu_links") if isinstance(payload, dict) else None
        if isinstance(menu_links, list) and menu_links:
            lines.append("Menu links / tokens:")
            for item in menu_links[:8]:
                if not isinstance(item, dict):
                    continue
                title = _as_text(item.get("title") or item.get("name"))
                tokens = []
                for key in ("topic_token", "publication_token", "section_token", "story_token"):
                    value = _as_text(item.get(key))
                    if value:
                        tokens.append(f"{key}: {value}")
                if title or tokens:
                    suffix = f" | {' | '.join(tokens)}" if tokens else ""
                    lines.append(f"- {title}{suffix}")
        text_path = _write_text_artifact(
            f"serpapi-{source}",
            query_label or "query",
            "\n".join(lines),
        )
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
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
        gl: str = "",
        hl: str = "",
        location: str = "",
        tbs: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        page = max(1, _coerce_int(page, 1))
        params: dict[str, Any] = {
            "engine": "google_forums",
            "q": query,
            "start": (page - 1) * 10,
        }
        if gl.strip():
            params["gl"] = gl.strip().lower()
        if hl.strip():
            params["hl"] = hl.strip().lower()
        if location.strip():
            params["location"] = location.strip()
        if tbs.strip():
            params["tbs"] = tbs.strip()
        return self._serpapi_request(params, timeout_seconds=timeout_seconds)

    def search_google_news(
        self,
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        so: int = 0,
        topic_token: str = "",
        publication_token: str = "",
        section_token: str = "",
        story_token: str = "",
        kgmid: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        advanced_tokens = {
            "topic_token": topic_token,
            "publication_token": publication_token,
            "section_token": section_token,
            "story_token": story_token,
            "kgmid": kgmid,
        }
        has_advanced = any(str(value or "").strip() for value in advanced_tokens.values())
        if not query.strip() and not has_advanced:
            return "ERROR: Query or a Google News token is required"
        if query.strip() and has_advanced:
            return "ERROR: Google News q cannot be used together with topic/publication/section/story/kgmid tokens"
        if str(kgmid or "").strip():
            other_tokens = [
                topic_token,
                publication_token,
                section_token,
                story_token,
            ]
            if any(str(value or "").strip() for value in other_tokens):
                return "ERROR: Google News kgmid must be used alone without other advanced tokens"
        page = max(1, _coerce_int(page, 1))
        so = _clamp(_coerce_int(so, 0), 0, 1)
        params: dict[str, Any] = {
            "engine": "google_news",
        }
        if query.strip():
            params["q"] = query.strip()
            params["page"] = page
        if gl.strip():
            params["gl"] = gl.strip().lower()
        if hl.strip():
            params["hl"] = hl.strip().lower()
        if so and not str(kgmid or "").strip():
            params["so"] = so
        for key, value in advanced_tokens.items():
            if str(value or "").strip():
                params[key] = str(value).strip()
        return self._serpapi_request(params, timeout_seconds=timeout_seconds)

    def search_google_trends(
        self,
        query: str,
        data_type: str = "TIMESERIES",
        date: str = "today 12-m",
        geo: str = "",
        region: str = "",
        gprop: str = "",
        hl: str = "",
        tz: int = 420,
        category: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        data_type = str(data_type or "TIMESERIES").strip().upper()
        if data_type not in _GOOGLE_TRENDS_DATA_TYPES:
            return "ERROR: data_type must be one of TIMESERIES, GEO_MAP, GEO_MAP_0, RELATED_TOPICS, RELATED_QUERIES"
        gprop = str(gprop or "").strip().lower()
        if gprop not in _GOOGLE_TRENDS_GPROPS:
            return "ERROR: gprop must be one of '', images, news, froogle, youtube"
        if not str(query or "").strip():
            return "ERROR: Query cannot be empty"
        params: dict[str, Any] = {
            "engine": "google_trends",
            "q": str(query).strip(),
            "data_type": data_type,
            "date": str(date or "today 12-m").strip() or "today 12-m",
            "tz": _coerce_int(tz, 420),
        }
        if geo.strip():
            params["geo"] = geo.strip().upper()
        if region.strip():
            params["region"] = region.strip().upper()
        if gprop:
            params["gprop"] = gprop
        if hl.strip():
            params["hl"] = hl.strip().lower()
        if str(category or "").strip():
            params["cat"] = str(category).strip()
        payload = self._serpapi_payload(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return "ERROR: Unexpected SerpAPI response format"
        lines = self._format_google_trends(query=str(query), data_type=data_type, payload=payload).splitlines()
        artifact = _write_json_artifact("serpapi-google_trends", str(query), payload)
        text_path = _write_text_artifact("serpapi-google_trends", str(query), "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_google_trends_trending_now(
        self,
        geo: str = "US",
        hours: int = 24,
        category_id: str = "",
        only_active: bool = False,
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        hours = _coerce_int(hours, 24)
        if hours not in _GOOGLE_TRENDS_HOURS:
            return "ERROR: hours must be one of 4, 24, 48, 168"
        params: dict[str, Any] = {
            "engine": "google_trends_trending_now",
            "geo": str(geo or "US").strip().upper() or "US",
            "hours": hours,
        }
        if category_id.strip():
            params["category_id"] = category_id.strip()
        if only_active:
            params["only_active"] = "true"
        if hl.strip():
            params["hl"] = hl.strip().lower()
        payload = self._serpapi_payload(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return "ERROR: Unexpected SerpAPI response format"
        lines = self._format_trending_now(params["geo"], hours, payload).splitlines()
        artifact = _write_json_artifact("serpapi-google_trends_trending_now", params["geo"], payload)
        text_path = _write_text_artifact("serpapi-google_trends_trending_now", params["geo"], "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_google_videos(
        self,
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        location: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        page = max(1, _coerce_int(page, 1))
        params: dict[str, Any] = {
            "engine": "google_videos",
            "q": query,
            "start": (page - 1) * 10,
        }
        if gl.strip():
            params["gl"] = gl.strip().lower()
        if hl.strip():
            params["hl"] = hl.strip().lower()
        if location.strip():
            params["location"] = location.strip()
        return self._serpapi_request(params, timeout_seconds=timeout_seconds)

    def search_tiktok_web(
        self,
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: Query cannot be empty"
        return self._serpapi_request(
            {
                "engine": "google",
                "q": f"site:tiktok.com {query}",
                "start": (max(1, _coerce_int(page, 1)) - 1) * 10,
                **({"gl": gl.strip().lower()} if gl.strip() else {}),
                **({"hl": hl.strip().lower()} if hl.strip() else {}),
            },
            timeout_seconds=timeout_seconds,
        )

    def search_bluesky_web(
        self,
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: Query cannot be empty"
        return self._serpapi_request(
            {
                "engine": "google",
                "q": f"(site:bsky.app OR site:bsky.social) {query}",
                "start": (max(1, _coerce_int(page, 1)) - 1) * 10,
                **({"gl": gl.strip().lower()} if gl.strip() else {}),
                **({"hl": hl.strip().lower()} if hl.strip() else {}),
            },
            timeout_seconds=timeout_seconds,
        )

    def get_instagram_profile(
        self,
        profile_id: str,
        next_page_token: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        profile_id = str(profile_id or "").strip().strip("@")
        if not profile_id:
            return "ERROR: profile_id is required"
        params: dict[str, Any] = {
            "engine": "instagram_profile",
            "profile_id": profile_id,
        }
        if next_page_token.strip():
            params["next_page_token"] = next_page_token.strip()
        payload = self._serpapi_payload(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return "ERROR: Unexpected SerpAPI response format"
        lines = self._format_social_profile("Instagram", profile_id, payload).splitlines()
        artifact = _write_json_artifact("serpapi-instagram_profile", profile_id, payload)
        text_path = _write_text_artifact("serpapi-instagram_profile", profile_id, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_facebook_profile(
        self,
        profile_id: str,
        timeout_seconds: int = 20,
    ) -> str:
        profile_id = str(profile_id or "").strip().strip("@")
        if not profile_id:
            return "ERROR: profile_id is required"
        payload = self._serpapi_payload(
            {
                "engine": "facebook_profile",
                "profile_id": profile_id,
            },
            timeout_seconds=timeout_seconds,
        )
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return "ERROR: Unexpected SerpAPI response format"
        lines = self._format_social_profile("Facebook", profile_id, payload).splitlines()
        artifact = _write_json_artifact("serpapi-facebook_profile", profile_id, payload)
        text_path = _write_text_artifact("serpapi-facebook_profile", profile_id, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def _format_google_trends(self, query: str, data_type: str, payload: dict[str, Any]) -> str:
        lines = [f"SUCCESS: Google Trends {data_type} results for '{query}':"]
        timeline = payload.get("interest_over_time") or {}
        if isinstance(timeline, dict):
            timeline_data = timeline.get("timeline_data") or timeline.get("data") or []
            if isinstance(timeline_data, list) and timeline_data:
                lines.append("Interest over time:")
                for item in timeline_data[:20]:
                    if not isinstance(item, dict):
                        continue
                    date_label = _as_text(item.get("date") or item.get("time"))
                    values = item.get("values") or item.get("value") or []
                    if isinstance(values, list):
                        value_text = ", ".join(
                            _as_text(value.get("extracted_value") or value.get("value") or value)
                            if isinstance(value, dict)
                            else _as_text(value)
                            for value in values[:5]
                        )
                    else:
                        value_text = _as_text(values)
                    if date_label or value_text:
                        lines.append(f"- {date_label}: {value_text}".rstrip(": "))
        for section_key, label in (
            ("related_queries", "Related queries"),
            ("related_topics", "Related topics"),
        ):
            section = payload.get(section_key)
            if not isinstance(section, dict):
                continue
            for bucket in ("rising", "top"):
                items = section.get(bucket) or []
                if not isinstance(items, list) or not items:
                    continue
                lines.append(f"{label} ({bucket}):")
                for item in items[:10]:
                    if not isinstance(item, dict):
                        continue
                    topic = item.get("topic") if isinstance(item.get("topic"), dict) else {}
                    text = _as_text(item.get("query") or topic.get("title") or item.get("title"))
                    value = _as_text(item.get("value") or item.get("extracted_value"))
                    link = _item_url(item)
                    suffix = f" | value: {value}" if value else ""
                    if link:
                        suffix += f" | {link}"
                    if text:
                        lines.append(f"- {text}{suffix}")
        for key, label in (
            ("interest_by_region", "Interest by region"),
            ("compared_breakdown_by_region", "Compared breakdown by region"),
        ):
            section = payload.get(key)
            items = []
            if isinstance(section, dict):
                items = section.get("data") or section.get("regions") or []
            elif isinstance(section, list):
                items = section
            if not isinstance(items, list) or not items:
                continue
            lines.append(f"{label}:")
            for item in items[:15]:
                if not isinstance(item, dict):
                    continue
                location = _as_text(item.get("location") or item.get("geo") or item.get("name"))
                values = item.get("values") or item.get("value") or item.get("extracted_value")
                value_text = _as_text(values)
                if location:
                    lines.append(f"- {location}: {value_text}".rstrip(": "))
        if len(lines) == 1:
            lines.append("No trend blocks were returned.")
        return "\n".join(lines)

    def _format_trending_now(self, geo: str, hours: int, payload: dict[str, Any]) -> str:
        items = (
            payload.get("trending_searches")
            or payload.get("trending_now")
            or payload.get("trends")
            or payload.get("daily_searches")
            or payload.get("realtime_searches")
            or payload.get("stories")
            or []
        )
        if isinstance(items, dict):
            items = items.get("items") or items.get("data") or []
        if not isinstance(items, list):
            return "ERROR: Unexpected SerpAPI response format"
        if not items:
            return f"SUCCESS: No Google Trends Trending Now results found for {geo} in the past {hours} hours."
        max_results = _clamp(_coerce_int(self.config.forumscout_max_results, 6), 1, 20)
        lines = [f"SUCCESS: Google Trends Trending Now results for {geo} ({hours}h, top {min(len(items), max_results)}):"]
        for idx, item in enumerate(items[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            title = _as_text(item.get("query") or item.get("title") or item.get("name"))
            if not title and isinstance(item.get("queries"), list):
                title = _as_text(item["queries"][0])
            lines.append(f"{idx}. {title or '(no title)'}")
            _append_dict_fields(
                lines,
                item,
                [
                    ("search_volume", "search volume"),
                    ("increase_percentage", "increase"),
                    ("started", "started"),
                    ("active", "active"),
                    ("trend_status", "status"),
                ],
            )
            queries = item.get("queries") or item.get("related_queries") or []
            if isinstance(queries, list) and queries:
                query_text = ", ".join(_as_text(query) for query in queries[:8] if _as_text(query))
                if query_text:
                    lines.append(f"   queries: {query_text}")
            articles = item.get("articles") or item.get("news_articles") or item.get("news") or []
            if isinstance(articles, list) and articles:
                lines.append("   articles:")
                for article in articles[:3]:
                    if not isinstance(article, dict):
                        continue
                    article_title = _as_text(article.get("title"))
                    article_link = _item_url(article)
                    source = _as_text(article.get("source") or article.get("publication"))
                    source_text = f" ({source})" if source else ""
                    lines.append(f"   - {article_title}{source_text} - {article_link}".rstrip())
        return "\n".join(lines)

    def _format_social_profile(self, source: str, profile_id: str, payload: dict[str, Any]) -> str:
        profile = payload.get("profile_results") or payload.get("profile") or payload.get("account") or {}
        if not isinstance(profile, dict):
            return "ERROR: Unexpected SerpAPI response format"
        lines = [f"SUCCESS: {source} profile results for '{profile_id}':"]
        name = _as_text(profile.get("name") or profile.get("username") or profile.get("full_name") or profile_id)
        url = _item_url(profile)
        lines.append(f"Profile: {name} - {url}")
        _append_dict_fields(
            lines,
            profile,
            [
                ("id", "id"),
                ("verified", "verified"),
                ("followers", "followers"),
                ("follower_count", "followers"),
                ("following", "following"),
                ("following_count", "following"),
                ("posts", "posts"),
                ("posts_count", "posts"),
                ("description", "description"),
                ("bio", "bio"),
                ("category", "category"),
            ],
            indent="",
        )
        posts = (
            profile.get("posts")
            or profile.get("latest_posts")
            or payload.get("posts")
            or payload.get("posts_results")
            or []
        )
        if isinstance(posts, list) and posts:
            lines.append("Recent posts:")
            for idx, item in enumerate(posts[:8], start=1):
                if not isinstance(item, dict):
                    continue
                title = _as_text(item.get("title") or item.get("caption") or item.get("text") or item.get("description"))
                link = _item_url(item)
                lines.append(f"{idx}. {_normalize_snippet(title, 180) or '(post)'} - {link}")
                _append_dict_fields(
                    lines,
                    item,
                    [
                        ("date", "date"),
                        ("timestamp", "timestamp"),
                        ("likes", "likes"),
                        ("comments", "comments"),
                        ("views", "views"),
                    ],
                )
        pagination = payload.get("serpapi_pagination") or {}
        if isinstance(pagination, dict) and pagination.get("next_page_token"):
            lines.append(f"Next page token: {pagination['next_page_token']}")
        return "\n".join(lines)


def get_forum_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="forum_search")
    def forum_search(
        query: str,
        time: str = "",
        country: str = "",
        page: int = 1,
        timeout_seconds: int = 20,
    ) -> str:
        """Generic forum search via ForumScout.

        Args:
            query: Search keyword.
            time: Time filter (hour, day, week, month, year, or empty).
            country: ISO 3166-1 alpha-2 country code.
            page: Page number (1+).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "time": time,
            "country": country,
            "page": page,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "forum_search",
            tool_input,
            lambda: helper.forum_search(
                query=query,
                time=time,
                country=country,
                page=page,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(forum_search)


def get_linkedin_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="linkedin_search")
    def linkedin_search(
        query: str,
        page: int = 1,
        sort_by: str = "date_posted",
        timeout_seconds: int = 20,
    ) -> str:
        """Search LinkedIn posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: Sort order (date_posted, relevance).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "sort_by": sort_by,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "linkedin_search",
            tool_input,
            lambda: helper.linkedin_search(
                query=query,
                page=page,
                sort_by=sort_by,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(linkedin_search)


def get_instagram_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="instagram_search")
    def instagram_search(
        query: str,
        page: int = 1,
        sort_by: str = "recent",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Instagram posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: Sort order (recent, top).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "sort_by": sort_by,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "instagram_search",
            tool_input,
            lambda: helper.instagram_search(
                query=query,
                page=page,
                sort_by=sort_by,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(instagram_search)


def get_reddit_posts_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="reddit_posts_search")
    def reddit_posts_search(
        query: str,
        page: int = 1,
        sort_by: str = "new",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Reddit posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: Sort order (hot, new, relevance, top).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "sort_by": sort_by,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "reddit_posts_search",
            tool_input,
            lambda: helper.reddit_posts_search(
                query=query,
                page=page,
                sort_by=sort_by,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(reddit_posts_search)


def get_reddit_comments_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="reddit_comments_search")
    def reddit_comments_search(
        query: str,
        page: int = 1,
        sort_by: str = "created_utc",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Reddit comments via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: Sort order (created_utc, score).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "sort_by": sort_by,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "reddit_comments_search",
            tool_input,
            lambda: helper.reddit_comments_search(
                query=query,
                page=page,
                sort_by=sort_by,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(reddit_comments_search)


def get_x_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="x_search")
    def x_search(
        query: str,
        page: int = 1,
        sort_by: str = "Latest",
        timeout_seconds: int = 20,
    ) -> str:
        """Search X (Twitter) posts via ForumScout.

        Args:
            query: Search keyword.
            page: Page number (1+).
            sort_by: Sort order (Latest, Top).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "sort_by": sort_by,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "x_search",
            tool_input,
            lambda: helper.x_search(
                query=query,
                page=page,
                sort_by=sort_by,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(x_search)


def get_google_forums_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_forums")
    def search_google_forums(
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        location: str = "",
        tbs: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Google forums results via SerpAPI.

        Args:
            query: Search keyword.
            page: Page number (1+).
            gl: Optional country code (e.g. 'us').
            hl: Optional language code (e.g. 'en').
            location: Optional city/region/country search origin.
            tbs: Optional Google advanced time/filter string.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "gl": gl,
            "hl": hl,
            "location": location,
            "tbs": tbs,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_forums",
            tool_input,
            lambda: helper.search_google_forums(
                query=query,
                page=page,
                gl=gl,
                hl=hl,
                location=location,
                tbs=tbs,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(search_google_forums)


def get_google_news_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_news")
    def search_google_news(
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        so: int = 0,
        topic_token: str = "",
        publication_token: str = "",
        section_token: str = "",
        story_token: str = "",
        kgmid: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Google News results via SerpAPI.

        Args:
            query: Search keyword.
            page: Page number (1+).
            gl: Optional country code.
            hl: Optional language code.
            so: Sort order, 0 relevance or 1 date where supported.
            topic_token: Optional Google News topic token.
            publication_token: Optional Google News publication token.
            section_token: Optional Google News section token.
            story_token: Optional Google News story/full-coverage token.
            kgmid: Optional Google News Knowledge Graph ID. Must be used without q or other tokens.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "gl": gl,
            "hl": hl,
            "so": so,
            "topic_token": topic_token,
            "publication_token": publication_token,
            "section_token": section_token,
            "story_token": story_token,
            "kgmid": kgmid,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_news",
            tool_input,
            lambda: helper.search_google_news(
                query=query,
                page=page,
                gl=gl,
                hl=hl,
                so=so,
                topic_token=topic_token,
                publication_token=publication_token,
                section_token=section_token,
                story_token=story_token,
                kgmid=kgmid,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(search_google_news)


def get_google_trends_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_trends")
    def search_google_trends(
        query: str,
        data_type: str = "TIMESERIES",
        date: str = "today 12-m",
        geo: str = "",
        region: str = "",
        gprop: str = "",
        hl: str = "",
        tz: int = 420,
        category: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Google Trends via SerpAPI for non-AI interest and query expansion data.

        Args:
            query: Search term, topic ID, or up to 5 comma-separated terms for TIMESERIES comparison.
            data_type: TIMESERIES, GEO_MAP, GEO_MAP_0, RELATED_TOPICS, or RELATED_QUERIES.
            date: Trends date window, e.g. 'now 7-d', 'today 12-m', or '2025-01-01 2025-12-31'.
            geo: Optional region code such as US, ES, GB, or empty for worldwide.
            region: Optional breakdown region for GEO_MAP types, such as COUNTRY, REGION, DMA, CITY.
            gprop: Optional property: images, news, froogle, youtube, or empty for web search.
            hl: Optional language code.
            tz: Timezone offset in minutes.
            category: Optional Google Trends category id.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "data_type": data_type,
            "date": date,
            "geo": geo,
            "region": region,
            "gprop": gprop,
            "hl": hl,
            "tz": tz,
            "category": category,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_trends",
            tool_input,
            lambda: helper.search_google_trends(
                query=query,
                data_type=data_type,
                date=date,
                geo=geo,
                region=region,
                gprop=gprop,
                hl=hl,
                tz=tz,
                category=category,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(search_google_trends)


def get_google_trends_trending_now_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_trends_trending_now")
    def search_google_trends_trending_now(
        geo: str = "US",
        hours: int = 24,
        category_id: str = "",
        only_active: bool = False,
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Get currently rising Google searches via SerpAPI Trending Now.

        Args:
            geo: Region code, default US.
            hours: Lookback window: 4, 24, 48, or 168.
            category_id: Optional Google Trending Now category id.
            only_active: Whether to only return active trends.
            hl: Optional language code.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "geo": geo,
            "hours": hours,
            "category_id": category_id,
            "only_active": only_active,
            "hl": hl,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_trends_trending_now",
            tool_input,
            lambda: helper.search_google_trends_trending_now(
                geo=geo,
                hours=hours,
                category_id=category_id,
                only_active=only_active,
                hl=hl,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(search_google_trends_trending_now)


def get_google_videos_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_videos")
    def search_google_videos(
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        location: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Google Videos via SerpAPI for cross-platform video results.

        Args:
            query: Search keyword.
            page: Page number (1+).
            gl: Optional country code.
            hl: Optional language code.
            location: Optional city/region/country search origin.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "gl": gl,
            "hl": hl,
            "location": location,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_videos",
            tool_input,
            lambda: helper.search_google_videos(
                query=query,
                page=page,
                gl=gl,
                hl=hl,
                location=location,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(search_google_videos)


def get_instagram_profile_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_instagram_profile")
    def get_instagram_profile(
        profile_id: str,
        next_page_token: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Fetch an Instagram profile by handle/profile id via SerpAPI.

        Args:
            profile_id: Instagram handle/profile id, with or without @.
            next_page_token: Optional token from a previous profile call.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "profile_id": profile_id,
            "next_page_token": next_page_token,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "get_instagram_profile",
            tool_input,
            lambda: helper.get_instagram_profile(
                profile_id=profile_id,
                next_page_token=next_page_token,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(get_instagram_profile)


def get_tiktok_web_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="tiktok_web_search")
    def tiktok_web_search(
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search TikTok-indexed public pages via SerpAPI Google site search.

        This is a web/SERP lookup, not TikTok's restricted Display API.
        """
        tool_input = {
            "query": query,
            "page": page,
            "gl": gl,
            "hl": hl,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "tiktok_web_search",
            tool_input,
            lambda: helper.search_tiktok_web(
                query=query,
                page=page,
                gl=gl,
                hl=hl,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _set_param_descriptions(tiktok_web_search, {
        "query": "TikTok-related search text; the tool searches public TikTok pages indexed by Google, not TikTok's private API.",
        "page": "Google results page number to request for the TikTok site search.",
        "gl": "Optional Google country code for localized TikTok web results, such as us or es.",
        "hl": "Optional Google interface language code for TikTok web results, such as en or es.",
        "timeout_seconds": "Maximum seconds to wait for the SerpAPI Google site-search request.",
    })


def get_bluesky_web_search_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="bluesky_web_search")
    def bluesky_web_search(
        query: str,
        page: int = 1,
        gl: str = "",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search public Bluesky pages via SerpAPI Google site search.

        This fallback is used because the direct public Bluesky API can deny
        anonymous infrastructure requests.
        """
        tool_input = {
            "query": query,
            "page": page,
            "gl": gl,
            "hl": hl,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "bluesky_web_search",
            tool_input,
            lambda: helper.search_bluesky_web(
                query=query,
                page=page,
                gl=gl,
                hl=hl,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _set_param_descriptions(bluesky_web_search, {
        "query": "Bluesky-related search text; the tool searches public bsky.app pages indexed by Google.",
        "page": "Google results page number to request for the Bluesky site search.",
        "gl": "Optional Google country code for localized Bluesky web results, such as us or es.",
        "hl": "Optional Google interface language code for Bluesky web results, such as en or es.",
        "timeout_seconds": "Maximum seconds to wait for the SerpAPI Google site-search request.",
    })


def get_facebook_profile_tool(helper: ForumScoutTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_facebook_profile")
    def get_facebook_profile(profile_id: str, timeout_seconds: int = 20) -> str:
        """Fetch a Facebook profile/page by profile id via SerpAPI.

        Args:
            profile_id: Facebook profile/page id from the URL.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "profile_id": profile_id,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "get_facebook_profile",
            tool_input,
            lambda: helper.get_facebook_profile(
                profile_id=profile_id,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_forum_output(get_facebook_profile)
