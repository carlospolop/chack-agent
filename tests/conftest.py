import pytest

import chack_tools.serpapi_keys as serpapi_keys


@pytest.fixture(autouse=True)
def _no_serpapi_account_lookups(monkeypatch):
    """Tests never reach serpapi.com: the key check is inconclusive (key usable)
    unless a test stubs it, and no verdict leaks between tests."""
    monkeypatch.setattr(serpapi_keys, "_query_account_exhausted", lambda api_key: None)
    serpapi_keys.reset_serpapi_exhaustion_cache()
    yield
    serpapi_keys.reset_serpapi_exhaustion_cache()
