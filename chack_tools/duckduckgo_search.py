from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

import requests
from langchain_core.tools import StructuredTool, tool

from .config import ToolsConfig


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results = []
        self._current_title = ""
        self._current_url = ""
        self._in_result = False
        self._in_title = False
        self._result_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div" and _has_class(attrs_dict, "result__body"):
            self._in_result = True
            self._current_title = ""
            self._current_url = ""
            self._result_depth = 1
        elif self._in_result and tag == "div":
            self._result_depth += 1
        if self._in_result and tag == "a" and _has_class(attrs_dict, "result__a"):
            self._in_title = True
            self._current_url = attrs_dict.get("href", "")

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        if tag == "div" and self._in_result:
            self._result_depth -= 1
            if self._result_depth <= 0:
                self._in_result = False
                if self._current_title and self._current_url:
                    self.results.append(
                        {
                            "title": self._current_title.strip(),
                            "url": _normalize_duckduckgo_url(self._current_url),
                        }
                    )
                self._current_title = ""
                self._current_url = ""
                self._result_depth = 0

    def handle_data(self, data):
        if self._in_title and data:
            self._current_title += data


def _normalize_duckduckgo_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("//"):
        url = f"https:{url}"
    if url.startswith("/"):
        url = f"https://duckduckgo.com{url}"
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    if "uddg" in query_params and query_params["uddg"]:
        return unquote(query_params["uddg"][0])
    return url


def _has_class(attrs: dict, class_name: str) -> bool:
    raw = attrs.get("class", "")
    if not raw:
        return False
    return class_name in str(raw).split()


class DuckDuckGoTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def search(self, query: str) -> str:
        return self._duckduckgo_search_impl(query=query)

    def _duckduckgo_search_impl(self, query: str, user_agent: Optional[str] = None) -> str:
        """Search DuckDuckGo and return a short list of results.

        Args:
            query: Search query string.
            user_agent: Optional custom User-Agent header override (useful to avoid blocks if the tool is not returning results).
        """
        max_results = self.config.duckduckgo_max_results
        if not query.strip():
            return "ERROR: Query cannot be empty"
        if max_results < 1:
            max_results = 1
        if max_results > 20:
            max_results = 20

        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        default_ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.3 Safari/605.1.15"
        )
        ua = user_agent or default_ua
        retry_ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

        try:
            response = requests.get(search_url, headers=headers, timeout=20)
            if response.status_code == 202:
                return "ERROR: DuckDuckGo returned HTTP 202 (likely blocked). Try different network/user_agent."
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: DuckDuckGo search timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to DuckDuckGo"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: DuckDuckGo returned HTTP {exc.response.status_code}"

        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        results = parser.results[:max_results]

        if not results and ua == default_ua:
            try:
                headers["User-Agent"] = retry_ua
                response = requests.get(search_url, headers=headers, timeout=20)
                if response.status_code == 202:
                    return "ERROR: DuckDuckGo returned HTTP 202 (likely blocked). Try different network/user_agent."
                response.raise_for_status()
            except requests.exceptions.Timeout:
                return "ERROR: DuckDuckGo search timed out"
            except requests.exceptions.ConnectionError:
                return "ERROR: Failed to connect to DuckDuckGo"
            except requests.exceptions.HTTPError as exc:
                return f"ERROR: DuckDuckGo returned HTTP {exc.response.status_code}"

            parser = _DuckDuckGoHTMLParser()
            parser.feed(response.text)
            results = parser.results[:max_results]

        if not results:
            return (
                f"SUCCESS: No DuckDuckGo results found for '{query}'. "
                "Try a different user_agent."
            )

        lines = [f"SUCCESS: DuckDuckGo results for '{query}' (top {len(results)}):"]
        for idx, result in enumerate(results, start=1):
            lines.append(f"{idx}. {result['title']} - {result['url']}")
        return "\n".join(lines)


def build_duckduckgo_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = DuckDuckGoTool(config)

    def _duckduckgo_search(query: str, user_agent: Optional[str] = None) -> str:
        """Search DuckDuckGo and return a short list of results.

        Args:
            query: Search query string.
            user_agent: Optional custom User-Agent header override (useful to avoid blocks if the tool is not returning results).
        """
        return helper._duckduckgo_search_impl(query=query, user_agent=user_agent)

    return StructuredTool.from_function(
        name="duckduckgo_search",
        description=_duckduckgo_search.__doc__ or "Search DuckDuckGo.",
        func=_duckduckgo_search,
    )
