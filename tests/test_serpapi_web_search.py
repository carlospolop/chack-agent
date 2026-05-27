from chack_tools.config import ToolsConfig
from chack_tools.serpapi_web_search import SerpApiWebSearchTool


class _Response:
    status_code = 200
    text = "{}"

    def json(self):
        return {
            "organic_results": [
                {
                    "title": "Result",
                    "link": "https://example.com/result",
                    "snippet": "Example result",
                }
            ]
        }


def test_serpapi_web_search_clamps_too_short_timeouts(monkeypatch):
    seen = {}

    def fake_get(url, *, params, timeout):
        seen["url"] = url
        seen["params"] = params
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.serpapi_web_search.requests.get", fake_get)

    tool = SerpApiWebSearchTool(ToolsConfig(serpapi_web_max_results=3))

    result = tool.search_google_web("next.js cve", timeout_seconds=10)

    assert result.startswith("SUCCESS: SerpAPI google web results")
    assert seen["url"] == "https://serpapi.com/search"
    assert seen["params"]["api_key"] == "test-key"
    assert seen["timeout"] == 45


def test_serpapi_web_search_clamps_excessive_timeouts(monkeypatch):
    seen = {}

    def fake_get(url, *, params, timeout):
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.serpapi_web_search.requests.get", fake_get)

    tool = SerpApiWebSearchTool(ToolsConfig())

    result = tool.search_bing_web("react advisory", timeout_seconds=999)

    assert result.startswith("SUCCESS: SerpAPI bing web results")
    assert seen["timeout"] == 120
