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


class _NoResultsResponse:
    status_code = 200
    text = "{}"

    def json(self):
        return {"error": "Google hasn't returned any results for this query."}


class _RichGoogleResponse:
    status_code = 200
    text = "{}"

    def json(self):
        return {
            "answer_box": {
                "title": "Example answer",
                "answer": "42",
                "link": "https://example.com/answer",
            },
            "knowledge_graph": {
                "title": "Example Entity",
                "type": "Organization",
                "website": "https://example.com",
                "description": "An example entity used for testing.",
            },
            "organic_results": [
                {
                    "title": "Result",
                    "link": "https://example.com/result",
                    "snippet": "Example result",
                }
            ],
            "top_stories": [
                {
                    "title": "Story",
                    "link": "https://news.example/story",
                    "source": "News Example",
                    "date": "Today",
                }
            ],
            "inline_videos": [
                {
                    "title": "Video",
                    "link": "https://video.example/watch",
                    "snippet": "Video context",
                }
            ],
            "inline_images": [
                {
                    "title": "Image",
                    "original": "https://images.example/original.jpg",
                    "source": "Images Example",
                }
            ],
            "local_results": {
                "places": [
                    {
                        "title": "Example Place",
                        "address": "1 Example St",
                        "rating": "4.8",
                        "reviews": "123",
                    }
                ]
            },
            "related_questions": [
                {
                    "question": "What is the example?",
                    "snippet": "A test question.",
                    "link": "https://example.com/question",
                }
            ],
            "related_searches": [
                {"query": "example related search"},
            ],
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


def test_serpapi_web_search_treats_no_results_as_success(monkeypatch):
    def fake_get(url, *, params, timeout):
        return _NoResultsResponse()

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.serpapi_web_search.requests.get", fake_get)

    tool = SerpApiWebSearchTool(ToolsConfig())

    result = tool.search_google_web("site:example.invalid no results")

    assert result == "SUCCESS: No SerpAPI results found for 'site:example.invalid no results'."
    assert "ERROR" not in result


def test_google_web_search_formats_structured_blocks_and_filters(monkeypatch):
    seen = {}

    def fake_get(url, *, params, timeout):
        seen["params"] = params
        return _RichGoogleResponse()

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.serpapi_web_search.requests.get", fake_get)

    tool = SerpApiWebSearchTool(ToolsConfig(serpapi_web_max_results=3))

    result = tool.search_google_web(
        "example entity",
        gl="us",
        hl="en",
        location="Austin, Texas, United States",
        tbs="qdr:m",
    )

    assert seen["params"]["gl"] == "us"
    assert seen["params"]["hl"] == "en"
    assert seen["params"]["location"] == "Austin, Texas, United States"
    assert seen["params"]["tbs"] == "qdr:m"
    assert "Organic results:" in result
    assert "Answer box:" in result
    assert "Knowledge graph:" in result
    assert "Top stories:" in result
    assert "Inline videos:" in result
    assert "Inline images:" in result
    assert "Local results:" in result
    assert "Related questions:" in result
    assert "Related searches:" in result
