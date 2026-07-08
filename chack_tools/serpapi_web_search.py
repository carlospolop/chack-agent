from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests

from .config import ToolsConfig
from .research_artifacts import record_research_artifact, record_research_json_artifact, research_artifacts_root
from .telemetry import log_tool_started, log_tool_executed, log_tool_error

from .serpapi_keys import (
    is_serpapi_rate_limited,
    note_serpapi_response_error,
    usable_serpapi_keys,
)


SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS = 45
SERPAPI_WEB_TIMEOUT_MIN_SECONDS = 45
SERPAPI_WEB_TIMEOUT_MAX_SECONDS = 120


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


def _serpapi_web_timeout(value) -> int:
    return _clamp(
        _coerce_int(value, SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS),
        SERPAPI_WEB_TIMEOUT_MIN_SECONDS,
        SERPAPI_WEB_TIMEOUT_MAX_SECONDS,
    )


def _is_no_results_error(error_text: str) -> bool:
    text = " ".join(str(error_text or "").lower().split())
    return "returned any results" in text or "no results" in text


def _safe_filename(value: str, fallback: str = "serpapi-web") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _artifact_dir(kind: str) -> str:
    root = research_artifacts_root()
    base = os.path.join(root, kind) if root else os.path.join("/tmp", "chack-serpapi-web", kind)
    os.makedirs(base, exist_ok=True)
    return base


def _write_json_artifact(kind: str, label: str, payload) -> str:
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


class SerpApiWebSearchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _api_key(self) -> str:
        keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        return keys[0] if keys else ""

    def _max_results(self, requested: Optional[int] = None) -> int:
        default_max = _coerce_int(getattr(self.config, "serpapi_web_max_results", 6), 6)
        if requested is None:
            return _clamp(default_max, 1, 10)
        return _clamp(_coerce_int(requested, default_max), 1, 10)

    def _request_payload(self, params: dict, timeout_seconds: int = SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS):
        api_keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if not api_keys:
            return "ERROR: No usable SerpAPI key (not configured or all keys exhausted)."
        timeout_seconds = _serpapi_web_timeout(timeout_seconds)
        last_error = "ERROR: SerpAPI request failed"
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
                if _is_no_results_error(error_text):
                    return {"organic_results": []}
                note_serpapi_response_error(api_key, response.status_code, error_text)
                if is_serpapi_rate_limited(response.status_code, error_text) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI error ({error_text})"
            return payload
        return last_error

    def _request(
        self,
        params: dict,
        timeout_seconds: int = SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS,
        max_results: Optional[int] = None,
    ) -> str:
        payload = self._request_payload(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact(str(params.get("engine", "serpapi")), str(params.get("q", "query")), payload)
        results = payload.get("organic_results") if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return "ERROR: Unexpected SerpAPI response format"
        if not results:
            return f"SUCCESS: No SerpAPI results found for '{params.get('q', '')}'.\nArtifact JSON: {artifact}"

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
        text_path = _write_text_artifact(str(params.get("engine", "serpapi")), str(params.get("q", "query")), "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    @staticmethod
    def _first_text(item: dict, keys: list[str]) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
        return ""

    @staticmethod
    def _append_link_rows(
        lines: list[str],
        heading: str,
        rows: list,
        *,
        limit: int,
        snippet_keys: list[str] | None = None,
    ) -> None:
        if not isinstance(rows, list) or not rows:
            return
        shown = [item for item in rows if isinstance(item, dict)][:limit]
        if not shown:
            return
        lines.append(f"{heading}:")
        for idx, item in enumerate(shown, start=1):
            title = item.get("title") or item.get("question") or item.get("query") or "(no title)"
            url = item.get("link") or item.get("url") or item.get("serpapi_link") or ""
            meta = []
            if item.get("source"):
                meta.append(str(item["source"]))
            if item.get("date"):
                meta.append(f"date: {item['date']}")
            if item.get("rating"):
                meta.append(f"rating: {item['rating']}")
            if item.get("reviews"):
                meta.append(f"reviews: {item['reviews']}")
            line = f"{idx}. {title}"
            if url:
                line += f" - {url}"
            lines.append(line)
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            snippet = SerpApiWebSearchTool._first_text(
                item,
                snippet_keys or ["snippet", "description", "answer"],
            )
            if snippet:
                lines.append(f"   {_normalize_snippet(snippet)}")

    def _append_google_structured_blocks(self, lines: list[str], payload: dict, *, limit: int) -> None:
        answer_box = payload.get("answer_box")
        if isinstance(answer_box, dict):
            values = [
                self._first_text(answer_box, ["title", "answer", "snippet", "snippet_highlighted_words"]),
                self._first_text(answer_box, ["source", "displayed_link"]),
            ]
            link = self._first_text(answer_box, ["link"])
            compact = [value for value in values if value]
            if compact or link:
                lines.append("Answer box:")
                if compact:
                    lines.append(f"- {_normalize_snippet(' | '.join(compact), max_chars=320)}")
                if link:
                    lines.append(f"- Source: {link}")

        knowledge_graph = payload.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            facts = []
            for key in ["title", "type", "description", "website"]:
                value = self._first_text(knowledge_graph, [key])
                if value:
                    facts.append(f"{key}: {_normalize_snippet(value, max_chars=180)}")
            source = knowledge_graph.get("source")
            if isinstance(source, dict) and source.get("link"):
                facts.append(f"source: {source['link']}")
            if facts:
                lines.append("Knowledge graph:")
                for fact in facts[:6]:
                    lines.append(f"- {fact}")

        self._append_link_rows(
            lines,
            "Top stories",
            payload.get("top_stories") or payload.get("news_results") or [],
            limit=min(limit, 5),
        )

        self._append_link_rows(
            lines,
            "Inline videos",
            payload.get("inline_videos") or payload.get("video_results") or [],
            limit=min(limit, 5),
            snippet_keys=["snippet", "duration", "platform"],
        )

        inline_images = payload.get("inline_images") or payload.get("images_results") or []
        if isinstance(inline_images, list) and inline_images:
            images = [item for item in inline_images if isinstance(item, dict)][: min(limit, 5)]
            if images:
                lines.append("Inline images:")
                for idx, item in enumerate(images, start=1):
                    title = item.get("title") or item.get("source") or "(image)"
                    url = item.get("original") or item.get("link") or item.get("thumbnail") or ""
                    source = item.get("source") or item.get("source_name") or ""
                    lines.append(f"{idx}. {title}" + (f" - {url}" if url else ""))
                    if source:
                        lines.append(f"   source: {source}")

        local_results = payload.get("local_results")
        places = []
        if isinstance(local_results, dict):
            places = local_results.get("places") or []
        elif isinstance(local_results, list):
            places = local_results
        self._append_link_rows(
            lines,
            "Local results",
            places,
            limit=min(limit, 5),
            snippet_keys=["address", "description", "phone"],
        )

        self._append_link_rows(
            lines,
            "Related questions",
            payload.get("related_questions") or [],
            limit=min(limit, 5),
            snippet_keys=["snippet", "answer"],
        )

        related_searches = payload.get("related_searches") or []
        if isinstance(related_searches, list):
            queries = []
            for item in related_searches:
                if not isinstance(item, dict):
                    continue
                query = self._first_text(item, ["query"])
                if query:
                    queries.append(query)
            if queries:
                lines.append("Related searches:")
                for query in queries[: min(limit, 8)]:
                    lines.append(f"- {query}")

    def _format_google_web(
        self,
        query: str,
        payload: dict,
        *,
        max_results: Optional[int] = None,
        include_structured: bool = True,
    ) -> str:
        results = payload.get("organic_results") if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return "ERROR: Unexpected SerpAPI response format"

        limit = self._max_results(max_results)
        shown = results[:limit]
        lines = [f"SUCCESS: SerpAPI google web results for '{query}' (top {len(shown)}):"]
        if shown:
            lines.append("Organic results:")
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

        if include_structured:
            self._append_google_structured_blocks(lines, payload, limit=limit)

        if len(lines) == 1:
            return f"SUCCESS: No SerpAPI results found for '{query}'."
        artifact = _write_json_artifact("google-web", query, payload)
        text_path = _write_text_artifact("google-web", query, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
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
        artifact = _write_json_artifact(engine, query, payload)
        text_path = _write_text_artifact(engine, query, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_google_web(
        self,
        query: str,
        page: int = 1,
        num: Optional[int] = None,
        gl: str = "",
        hl: str = "",
        location: str = "",
        tbs: str = "",
        include_structured: bool = True,
        timeout_seconds: int = SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        max_results = self._max_results(num)
        page = max(1, _coerce_int(page, 1))
        params = {
            "engine": "google",
            "q": query,
            "num": max_results,
            "start": (page - 1) * max_results,
        }
        optional_params = {
            "gl": gl,
            "hl": hl,
            "location": location,
            "tbs": tbs,
        }
        for key, value in optional_params.items():
            cleaned = str(value or "").strip()
            if cleaned:
                params[key] = cleaned
        payload = self._request_payload(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        return self._format_google_web(
            query,
            payload,
            max_results=max_results,
            include_structured=include_structured,
        )

    def search_bing_web(
        self,
        query: str,
        page: int = 1,
        count: Optional[int] = None,
        timeout_seconds: int = SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS,
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
        timeout_seconds: int = 100,
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


def _with_web_output(tool):
    current = str(getattr(tool, "description", "") or "").strip()
    if current and "Output:" not in current:
        tool.description = (
            f"{current}\n\n"
            "Parameters: Use the schema descriptions to provide the query, pagination, locale, location, date/filter strings, structured-block toggle, and request timeout.\n"
            "Output: Returns compact SUCCESS/ERROR text with search results, URLs, snippets, structured SERP blocks, or AI-mode answer/citation fields depending on the endpoint."
        )
    return tool


def get_google_web_search_tool(helper: SerpApiWebSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_web")
    def search_google_web(
        query: str,
        page: int = 1,
        num: Optional[int] = None,
        gl: str = "",
        hl: str = "",
        location: str = "",
        tbs: str = "",
        include_structured: bool = True,
        timeout_seconds: int = SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS,
    ) -> str:
        """Search Google web results via SerpAPI, including useful structured SERP blocks.

        Use when accuracy and recency matter (docs, error messages, product info).
        Prefer this as a primary web source and cross-check with Bing/Brave if needed.
        When present, the result also includes non-AI structured blocks such as answer box,
        knowledge graph, top stories, local results, related questions, and related searches.

        Args:
            query: Search query string.
            page: Result page (1+).
            num: Number of results (1-10). Defaults to config value.
            gl: Optional Google country code, e.g. us or es.
            hl: Optional Google language code, e.g. en or es.
            location: Optional SerpAPI location string, e.g. Austin, Texas, United States.
            tbs: Optional Google advanced filter string, commonly used for date filters.
            include_structured: Include high-signal structured SERP blocks when present.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "num": num,
            "gl": gl,
            "hl": hl,
            "location": location,
            "tbs": tbs,
            "include_structured": include_structured,
            "timeout_seconds": timeout_seconds,
        }
        start_ts = log_tool_started("search_google_web", tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.search_google_web(
                query=query,
                page=page,
                num=num,
                gl=gl,
                hl=hl,
                location=location,
                tbs=tbs,
                include_structured=include_structured,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                log_tool_error(
                    "search_google_web",
                    tool_input,
                    error=error,
                    trace=traceback.format_exc(),
                )
            except Exception:
                pass
            return f"ERROR: Google web search failed ({exc})"
        finally:
            end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            duration_ms = int((time.time() - start_time) * 1000)
            log_tool_executed(
                "search_google_web",
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    return _with_web_output(search_google_web)


def get_bing_web_search_tool(helper: SerpApiWebSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_bing_web")
    def search_bing_web(
        query: str,
        page: int = 1,
        count: Optional[int] = None,
        timeout_seconds: int = SERPAPI_WEB_TIMEOUT_DEFAULT_SECONDS,
    ) -> str:
        """Search Bing web results via SerpAPI.

        Use as a second source to cross-check findings and reduce search-engine bias.

        Args:
            query: Search query string.
            page: Result page (1+).
            count: Number of results (1-10). Defaults to config value.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "count": count,
            "timeout_seconds": timeout_seconds,
        }
        start_ts = log_tool_started("search_bing_web", tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.search_bing_web(
                query=query,
                page=page,
                count=count,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                log_tool_error(
                    "search_bing_web",
                    tool_input,
                    error=error,
                    trace=traceback.format_exc(),
                )
            except Exception:
                pass
            return f"ERROR: Bing web search failed ({exc})"
        finally:
            end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            duration_ms = int((time.time() - start_time) * 1000)
            log_tool_executed(
                "search_bing_web",
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    return _with_web_output(search_bing_web)


def get_google_ai_mode_tool(helper: SerpApiWebSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_ai_mode")
    def search_google_ai_mode(
        query: str,
        timeout_seconds: int = 45,
    ) -> str:
        """Search Google in AI mode via SerpAPI.

        Args:
            query: Search query string.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {"query": query, "timeout_seconds": timeout_seconds}
        start_ts = log_tool_started("search_google_ai_mode", tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.search_google_ai_mode(
                query=query,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                log_tool_error(
                    "search_google_ai_mode",
                    tool_input,
                    error=error,
                    trace=traceback.format_exc(),
                )
            except Exception:
                pass
            return f"ERROR: Google AI mode search failed ({exc})"
        finally:
            end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            duration_ms = int((time.time() - start_time) * 1000)
            log_tool_executed(
                "search_google_ai_mode",
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    return _with_web_output(search_google_ai_mode)


def get_bing_copilot_tool(helper: SerpApiWebSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_bing_copilot")
    def search_bing_copilot(
        query: str,
        timeout_seconds: int = 100,
    ) -> str:
        """Search Bing Copilot in AI mode via SerpAPI.

        Args:
            query: Search query string.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {"query": query, "timeout_seconds": timeout_seconds}
        start_ts = log_tool_started("search_bing_copilot", tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.search_bing_copilot(
                query=query,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                log_tool_error(
                    "search_bing_copilot",
                    tool_input,
                    error=error,
                    trace=traceback.format_exc(),
                )
            except Exception:
                pass
            return f"ERROR: Bing Copilot search failed ({exc})"
        finally:
            end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            duration_ms = int((time.time() - start_time) * 1000)
            log_tool_executed(
                "search_bing_copilot",
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    return _with_web_output(search_bing_copilot)
