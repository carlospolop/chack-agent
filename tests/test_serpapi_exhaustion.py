import chack_tools.serpapi_keys as sk


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _setup(monkeypatch, verdicts, clock=None):
    """Stub the account endpoint and clock; return a call counter."""
    calls = {"n": 0}

    def fake_query(api_key):
        calls["n"] += 1
        return verdicts.get(api_key)

    monkeypatch.setattr(sk, "_query_account_exhausted", fake_query)
    if clock is not None:
        monkeypatch.setattr(sk, "_now", clock)
    monkeypatch.setenv("SERPAPI_EXHAUSTION_CHECK_ENABLED", "1")
    monkeypatch.delenv("SERPAPI_EXHAUSTED_CACHE_SECONDS", raising=False)
    monkeypatch.delenv("SERPAPI_OK_CACHE_SECONDS", raising=False)
    sk.reset_serpapi_exhaustion_cache()
    return calls


def test_usable_filters_exhausted_keys(monkeypatch):
    _setup(monkeypatch, {"good": False, "dead": True})
    assert sk.usable_serpapi_keys("good,dead") == ["good"]


def test_usable_empty_when_all_exhausted(monkeypatch):
    _setup(monkeypatch, {"a": True, "b": True})
    assert sk.usable_serpapi_keys("a,b") == []


def test_cache_prevents_requery(monkeypatch):
    calls = _setup(monkeypatch, {"k": False})
    assert sk.is_serpapi_key_exhausted("k") is False
    assert sk.is_serpapi_key_exhausted("k") is False
    assert calls["n"] == 1  # queried once, then served from cache


def test_exhausted_parked_longer_than_usable(monkeypatch):
    clock = _Clock()
    calls = _setup(monkeypatch, {"ok": False, "dead": True}, clock=clock)
    assert sk.is_serpapi_key_exhausted("ok") is False    # query 1 (15 min TTL)
    assert sk.is_serpapi_key_exhausted("dead") is True   # query 2 (1 day TTL)

    clock.t = 16 * 60  # 16 min: usable TTL (15 min) expired, exhausted TTL (1 day) not
    assert sk.is_serpapi_key_exhausted("dead") is True   # still cached, no re-query
    assert calls["n"] == 2
    assert sk.is_serpapi_key_exhausted("ok") is False    # re-checked
    assert calls["n"] == 3


def test_fail_open_when_check_inconclusive(monkeypatch):
    calls = _setup(monkeypatch, {"k": None})  # network/HTTP unknown
    assert sk.is_serpapi_key_exhausted("k") is False
    assert sk.is_serpapi_key_exhausted("k") is False     # cached usable, no re-query
    assert calls["n"] == 1


def test_disabled_bypasses_check(monkeypatch):
    calls = _setup(monkeypatch, {"dead": True})
    monkeypatch.setenv("SERPAPI_EXHAUSTION_CHECK_ENABLED", "0")
    assert sk.usable_serpapi_keys("dead") == ["dead"]
    assert calls["n"] == 0  # no account calls at all


def test_quota_exhausted_classification():
    assert sk.is_serpapi_quota_exhausted(200, "You have run out of searches this month") is True
    assert sk.is_serpapi_quota_exhausted(200, "Insufficient searches") is True
    assert sk.is_serpapi_quota_exhausted(200, "no searches left on this account") is True
    # A plain hourly rate limit must NOT be treated as quota exhaustion.
    assert sk.is_serpapi_quota_exhausted(429, "Rate limit reached, too many requests") is False
    assert sk.is_serpapi_quota_exhausted(429, "") is False


def test_note_marks_exhausted_only_on_quota(monkeypatch):
    _setup(monkeypatch, {})
    sk.note_serpapi_response_error("transient", 429, "too many requests")
    assert sk._cache_get("transient") is None  # not parked
    sk.note_serpapi_response_error("dead", 200, "You have run out of searches this month")
    assert sk._cache_get("dead") is True       # parked as exhausted


def test_web_search_tool_reports_all_exhausted(monkeypatch):
    from chack_tools.config import ToolsConfig
    from chack_tools.serpapi_web_search import SerpApiWebSearchTool

    _setup(monkeypatch, {"dead": True})
    monkeypatch.setenv("SERPAPI_API_KEY", "dead")
    out = SerpApiWebSearchTool(ToolsConfig())._request_payload({"q": "x", "engine": "google"})
    assert isinstance(out, str) and "all keys exhausted" in out


def test_web_search_tool_uses_only_a_key_with_usage_left(monkeypatch):
    from chack_tools.config import ToolsConfig
    import chack_tools.serpapi_web_search as sws

    _setup(monkeypatch, {"dead": True, "live": False})
    monkeypatch.setenv("SERPAPI_API_KEY", "dead,live")

    used = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"organic_results": []}

    def _fake_get(url, params=None, timeout=None):
        used["api_key"] = (params or {}).get("api_key")
        return _FakeResp()

    monkeypatch.setattr(sws.requests, "get", _fake_get)

    out = sws.SerpApiWebSearchTool(ToolsConfig())._request_payload({"q": "x", "engine": "google"})
    # The exhausted key is never spent; the request runs on the key with usage left.
    assert used["api_key"] == "live"
    assert isinstance(out, dict)
