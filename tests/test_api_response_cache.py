import json

import boto3
import pytest
import requests

from chack_tools import serpapi_keys
from chack_tools.api_response_cache import MAX_AGE_CAP_SECONDS
from chack_tools.brave_search import BraveSearchTool
from chack_tools.config import ToolsConfig
from chack_tools.serpapi_web_search import SerpApiWebSearchTool


SERPAPI_KEY = "serp-secret-key-123"
BRAVE_KEY = "brave-secret-key-456"


class _FakeS3:
    """In-memory S3 that misses like the real one (an exception, never None)."""

    def __init__(self):
        self.objects = {}
        self.down = False

    def get_object(self, Bucket, Key):
        if self.down:
            raise ConnectionError("s3 unreachable")
        if (Bucket, Key) not in self.objects:
            raise KeyError("NoSuchKey")
        body = self.objects[(Bucket, Key)]

        class _Body:
            def read(self_inner):
                return body

        return {"Body": _Body()}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        if self.down:
            raise ConnectionError("s3 unreachable")
        self.objects[(Bucket, Key)] = Body


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _serp_success(query="q"):
    return {
        "search_metadata": {"status": "Success"},
        "organic_results": [{"title": f"Result for {query}", "link": "https://example.com", "snippet": "s"}],
    }


@pytest.fixture
def s3(monkeypatch, tmp_path):
    fake = _FakeS3()
    created = []

    def _client(*args, **kwargs):
        created.append(kwargs)
        return fake

    monkeypatch.setattr(boto3, "client", _client)
    monkeypatch.setenv("SERPAPI_API_KEY", SERPAPI_KEY)
    monkeypatch.setenv("BRAVE_API_KEY", BRAVE_KEY)
    monkeypatch.setenv("CHACK_API_CACHE_BUCKET", "cache-bucket")
    monkeypatch.setenv("CHACK_API_CACHE_PREFIX", "/tenants/123/chack/")
    monkeypatch.delenv("CHACK_API_CACHE_MAX_AGE", raising=False)
    monkeypatch.setenv("CHACK_RESEARCH_MASTER_DIR", str(tmp_path))
    monkeypatch.setattr(serpapi_keys, "is_serpapi_key_exhausted", lambda key: False)
    fake.created = created
    return fake


@pytest.fixture
def live(monkeypatch):
    calls = []
    responses = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        return responses.pop(0) if len(responses) > 1 else responses[0]

    monkeypatch.setattr(requests, "get", _get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return calls, responses


def test_second_identical_search_is_served_from_cache(s3, live):
    calls, responses = live
    responses.append(_Response(200, _serp_success("acme")))
    tool = SerpApiWebSearchTool(ToolsConfig())

    first = tool._request_payload({"engine": "google", "q": "acme"})
    second = tool._request_payload({"engine": "google", "q": "acme"})

    assert len(calls) == 1
    assert second == first == _serp_success("acme")
    # A fresh tool (a relaunched agent) also reuses the entry.
    assert SerpApiWebSearchTool(ToolsConfig())._request_payload({"engine": "google", "q": "acme"}) == first
    assert len(calls) == 1


def test_different_query_engine_or_credential_misses(s3, live, monkeypatch):
    calls, responses = live
    responses.append(_Response(200, _serp_success()))
    tool = SerpApiWebSearchTool(ToolsConfig())

    tool._request_payload({"engine": "google", "q": "acme"})
    tool._request_payload({"engine": "google", "q": "other"})
    tool._request_payload({"engine": "bing", "q": "acme"})
    monkeypatch.setenv("SERPAPI_API_KEY", "a-different-key")
    tool._request_payload({"engine": "google", "q": "acme"})

    assert len(calls) == 4


def test_entries_never_contain_the_credential_and_stay_under_the_tenant_prefix(s3, live):
    _, responses = live
    responses.append(_Response(200, _serp_success()))
    SerpApiWebSearchTool(ToolsConfig())._request_payload({"engine": "google", "q": "acme"})

    assert len(s3.objects) == 1
    (bucket, key), body = next(iter(s3.objects.items()))
    assert bucket == "cache-bucket"
    assert key.startswith("tenants/123/chack/serpapi/")
    assert SERPAPI_KEY not in key and SERPAPI_KEY.encode() not in body


def test_other_tenant_prefix_does_not_see_the_entry(s3, live, monkeypatch):
    calls, responses = live
    responses.append(_Response(200, _serp_success()))
    SerpApiWebSearchTool(ToolsConfig())._request_payload({"engine": "google", "q": "acme"})
    monkeypatch.setenv("CHACK_API_CACHE_PREFIX", "tenants/456/chack")
    SerpApiWebSearchTool(ToolsConfig())._request_payload({"engine": "google", "q": "acme"})

    assert len(calls) == 2


def test_entry_older_than_sixteen_hours_is_refetched(s3, live, monkeypatch):
    calls, responses = live
    responses.append(_Response(200, _serp_success()))
    monkeypatch.setenv("CHACK_API_CACHE_MAX_AGE", str(7 * 24 * 3600))  # capped at 16h
    tool = SerpApiWebSearchTool(ToolsConfig())
    now = [1_000_000.0]
    tool._cache._clock = lambda: now[0]

    tool._request_payload({"engine": "google", "q": "acme"})
    now[0] += MAX_AGE_CAP_SECONDS - 60
    tool._request_payload({"engine": "google", "q": "acme"})
    assert len(calls) == 1
    now[0] += 120
    tool._request_payload({"engine": "google", "q": "acme"})
    assert len(calls) == 2


@pytest.mark.parametrize(
    "response",
    [
        _Response(429, {"error": "Your account has run out of searches."}),
        _Response(401, {"error": "Invalid API key."}),
        _Response(500, {"error": "Internal error"}),
        _Response(200, {"error": "Your account has run out of searches."}),
        _Response(200, {"search_metadata": {"status": "Processing"}, "organic_results": []}),
    ],
)
def test_errors_quota_and_incomplete_responses_are_not_cached(s3, live, response):
    calls, responses = live
    responses.append(response)
    tool = SerpApiWebSearchTool(ToolsConfig())

    tool._request_payload({"engine": "google", "q": "acme"})
    assert s3.objects == {}
    responses[0] = _Response(200, _serp_success())
    assert tool._request_payload({"engine": "google", "q": "acme"}) == _serp_success()

    assert len(calls) == 2
    assert len(s3.objects) == 1


def test_cache_is_off_when_bucket_is_unset(s3, live, monkeypatch):
    calls, responses = live
    responses.append(_Response(200, _serp_success()))
    monkeypatch.delenv("CHACK_API_CACHE_BUCKET")
    tool = SerpApiWebSearchTool(ToolsConfig())

    tool._request_payload({"engine": "google", "q": "acme"})
    tool._request_payload({"engine": "google", "q": "acme"})

    assert len(calls) == 2
    assert s3.objects == {} and s3.created == []


def test_settings_are_captured_when_the_tool_is_built(s3, live, monkeypatch):
    calls, responses = live
    responses.append(_Response(200, _serp_success()))
    tool = SerpApiWebSearchTool(ToolsConfig())
    monkeypatch.delenv("CHACK_API_CACHE_BUCKET")

    tool._request_payload({"engine": "google", "q": "acme"})
    tool._request_payload({"engine": "google", "q": "acme"})

    assert len(calls) == 1


def test_s3_outage_falls_back_to_live_calls(s3, live):
    calls, responses = live
    responses.append(_Response(200, _serp_success()))
    s3.down = True
    tool = SerpApiWebSearchTool(ToolsConfig())

    assert tool._request_payload({"engine": "google", "q": "acme"}) == _serp_success()
    assert tool._request_payload({"engine": "google", "q": "acme"}) == _serp_success()
    assert len(calls) == 2


def test_corrupt_entry_is_ignored_and_replaced(s3, live):
    calls, responses = live
    responses.append(_Response(200, _serp_success()))
    tool = SerpApiWebSearchTool(ToolsConfig())
    tool._request_payload({"engine": "google", "q": "acme"})
    key = next(iter(s3.objects))
    s3.objects[key] = b"not json"

    assert tool._request_payload({"engine": "google", "q": "acme"}) == _serp_success()
    assert tool._request_payload({"engine": "google", "q": "acme"}) == _serp_success()
    assert len(calls) == 2


def test_formatted_search_output_is_identical_on_a_hit(s3, live):
    calls, responses = live
    responses.append(_Response(200, _serp_success("acme")))
    tool = SerpApiWebSearchTool(ToolsConfig())

    first = tool.search_bing_web("acme")
    second = tool.search_bing_web("acme")

    strip = lambda text: [line for line in text.splitlines() if not line.startswith("Artifact")]
    assert len(calls) == 1
    assert strip(first) == strip(second)
    assert "Result for acme" in second


def _brave_success():
    return {"type": "search", "web": {"results": [{"title": "Brave hit", "url": "https://b.example", "description": "d"}]}}


def test_brave_hit_skips_the_live_call(s3, live):
    calls, responses = live
    responses.append(_Response(200, _brave_success()))
    tool = BraveSearchTool(ToolsConfig())

    first = tool._brave_search_impl(query="acme")
    second = BraveSearchTool(ToolsConfig())._brave_search_impl(query="acme")

    assert len(calls) == 1
    assert first == second and "Brave hit" in second
    assert all(BRAVE_KEY.encode() not in body for body in s3.objects.values())


def test_brave_rate_limit_is_not_cached(s3, live):
    calls, responses = live
    responses.extend([_Response(429, {"type": "ErrorResponse"}), _Response(429, {"type": "ErrorResponse"})])
    tool = BraveSearchTool(ToolsConfig())

    with pytest.raises(requests.HTTPError):
        tool._brave_search_impl(query="acme")
    responses[:] = [_Response(200, _brave_success())]
    assert "Brave hit" in tool._brave_search_impl(query="acme")

    assert len(calls) == 3
    assert len(s3.objects) == 1
