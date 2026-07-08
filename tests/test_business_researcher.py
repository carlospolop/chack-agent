import json

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.business_search import BusinessSearchTool
from chack_tools.config import ToolsConfig


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            response = type("Response", (), {"status_code": self.status_code})()
            raise requests.exceptions.HTTPError(response=response)


def _tool_names(tools):
    return {
        str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")
        for tool in tools
    }


def test_sec_company_search_writes_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))

    def fake_get(url, **kwargs):
        assert url == "https://www.sec.gov/files/company_tickers.json"
        return FakeResponse({
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        })

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    result = helper.search_sec_companies("apple")

    assert "Apple Inc." in result
    assert "CIK: 0000320193" in result
    artifacts = list((tmp_path / "sec").glob("company_tickers_apple_*.json"))
    assert len(artifacts) == 1


def test_sec_submissions_resolves_ticker_and_formats_filing(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    seen_urls = []

    def fake_get(url, **kwargs):
        seen_urls.append(url)
        if url == "https://www.sec.gov/files/company_tickers.json":
            return FakeResponse({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
        assert url == "https://data.sec.gov/submissions/CIK0000320193.json"
        return FakeResponse({
            "name": "Apple Inc.",
            "tickers": ["AAPL"],
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K"],
                    "accessionNumber": ["0000320193-25-000079", "0000320193-25-000078"],
                    "filingDate": ["2025-11-01", "2025-10-20"],
                    "reportDate": ["2025-09-27", "2025-10-20"],
                    "primaryDocument": ["aapl-20250927.htm", "aapl-8k.htm"],
                }
            },
        })

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    result = helper.get_sec_company_submissions("AAPL", form_filter="10-K", max_filings=1)

    assert seen_urls == [
        "https://www.sec.gov/files/company_tickers.json",
        "https://data.sec.gov/submissions/CIK0000320193.json",
    ]
    assert "10-K" in result
    assert "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm" in result
    assert list((tmp_path / "sec").glob("submissions_0000320193_*.json"))


def test_gleif_search_uses_fulltext_and_country_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return FakeResponse({
            "meta": {"pagination": {"total": 1}},
            "data": [{
                "id": "HWUPKR0MPOU8FGXBT394",
                "attributes": {
                    "lei": "HWUPKR0MPOU8FGXBT394",
                    "entity": {
                        "legalName": {"name": "APPLE INC."},
                        "legalAddress": {"country": "US"},
                        "jurisdiction": "US-CA",
                    },
                    "registration": {"status": "ISSUED"},
                },
            }],
        })

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    result = helper.search_gleif_lei("Apple Inc", country="us", max_results=2)

    assert captured["url"] == "https://api.gleif.org/api/v1/lei-records"
    assert captured["params"]["filter[fulltext]"] == "Apple Inc"
    assert captured["params"]["filter[entity.legalAddress.country]"] == "US"
    assert captured["params"]["page[size]"] == 2
    assert "HWUPKR0MPOU8FGXBT394" in result
    assert list((tmp_path / "gleif").glob("lei_search_Apple_Inc_*.json"))


def test_google_finance_uses_serpapi_params_and_writes_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return FakeResponse({
            "summary": {
                "title": "Apple Inc",
                "exchange": "NASDAQ",
                "price": "$200.00",
                "price_movement": {"percentage": "1.2%", "value": "2.4"},
            },
            "news_results": [{"title": "Apple market update", "link": "https://example.test/news"}],
        })

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    result = helper.search_google_finance("AAPL:NASDAQ", window="1Y", hl="en")

    assert captured["url"] == "https://serpapi.com/search"
    assert captured["params"]["engine"] == "google_finance"
    assert captured["params"]["q"] == "AAPL:NASDAQ"
    assert captured["params"]["window"] == "1Y"
    assert captured["params"]["api_key"] == "serp-key"
    assert "Apple Inc" in result
    assert "Apple market update" in result
    assert list((tmp_path / "google-finance").glob("AAPL_NASDAQ_*.json"))


def test_google_maps_and_reviews_use_expected_params(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    calls = []

    def fake_get(url, **kwargs):
        params = kwargs.get("params")
        calls.append(params)
        if params["engine"] == "google_maps":
            return FakeResponse({
                "local_results": [{
                    "title": "Example Cafe",
                    "rating": 4.6,
                    "reviews": 120,
                    "address": "1 Main St",
                    "phone": "555-0100",
                    "place_id": "place-1",
                    "data_id": "data-1",
                }]
            })
        return FakeResponse({
            "place_info": {"title": "Example Cafe", "rating": 4.6},
            "reviews": [{"rating": 5, "date": "2026-01-01", "text": "Great service"}],
        })

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    maps = helper.search_google_maps("coffee", location="Austin, Texas", gl="US", hl="en")
    reviews = helper.get_google_maps_reviews(data_id="data-1", place_id="place-1", sort_by="newestFirst")

    assert calls[0]["engine"] == "google_maps"
    assert calls[0]["q"] == "coffee"
    assert calls[0]["location"] == "Austin, Texas"
    assert calls[0]["z"] == "14"
    assert calls[1]["engine"] == "google_maps_reviews"
    assert calls[1]["data_id"] == "data-1"
    assert "place_id" not in calls[1]
    assert "Example Cafe" in maps
    assert "Great service" in reviews
    assert list((tmp_path / "google-maps").glob("coffee_*.json"))
    assert list((tmp_path / "google-maps-reviews").glob("data-1_*.json"))


def test_yelp_apple_maps_ads_and_shopping_params(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    calls = []

    def fake_get(url, **kwargs):
        params = kwargs.get("params")
        calls.append(params)
        engine = params["engine"]
        if engine == "yelp":
            return FakeResponse({"organic_results": [{"title": "Yelp Cafe", "place_ids": ["yp1"], "rating": 4.5}]})
        if engine == "apple_maps":
            return FakeResponse({"place_results": [{"title": "Apple Cafe", "muid": "m1", "rating": 4.4}]})
        if engine == "google_ads_transparency_center":
            return FakeResponse({"ad_creatives": [{"advertiser_id": "AR1", "format": "text", "text": "Buy now"}]})
        return FakeResponse({"shopping_results": [{"title": "Widget", "product_id": "p1", "price": "$9.99", "rating": 4.1}]})

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    yelp = helper.search_yelp_businesses("coffee", "New York, NY")
    apple = helper.search_apple_maps("coffee", "Austin, TX")
    ads = helper.search_google_ads_transparency(text="example.com", platform="SEARCH", creative_format="text")
    shopping = helper.search_google_shopping("wireless mouse", gl="us")

    assert calls[0]["engine"] == "yelp"
    assert calls[0]["find_desc"] == "coffee"
    assert calls[0]["find_loc"] == "New York, NY"
    assert calls[1]["engine"] == "apple_maps"
    assert calls[1]["query"] == "coffee"
    assert calls[1]["location"] == "Austin, TX"
    assert calls[2]["engine"] == "google_ads_transparency_center"
    assert calls[2]["text"] == "example.com"
    assert calls[2]["platform"] == "SEARCH"
    assert "region" not in calls[2]
    assert calls[3]["engine"] == "google_shopping"
    assert calls[3]["q"] == "wireless mouse"
    assert "Yelp Cafe" in yelp
    assert "place_id: yp1" in yelp
    assert "Apple Cafe" in apple
    assert "Buy now" in ads
    assert "Widget" in shopping


def test_google_ads_defaults_and_transparency_region_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    calls = []

    def fake_get(url, **kwargs):
        params = kwargs.get("params")
        calls.append(params)
        if params["engine"] == "google_ads":
            return FakeResponse({"shopping_results": [{"title": "Ad Product", "link": "https://example.test"}]})
        return FakeResponse({"ad_creatives": [{"advertiser": "Nike", "format": "text"}]})

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    ads = helper.search_google_ads("buy shoes")
    transparency = helper.search_google_ads_transparency(text="nike.com", region="US")

    assert calls[0]["engine"] == "google_ads"
    assert calls[0]["location"] == "United States"
    assert calls[1]["engine"] == "google_ads_transparency_center"
    assert calls[1]["region"] == "2840"
    assert "Ad Product" in ads
    assert "Nike" in transparency


def test_marketplace_and_tripadvisor_params(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    calls = []

    def fake_get(url, **kwargs):
        params = kwargs.get("params")
        calls.append(params)
        engine = params["engine"]
        if engine == "amazon":
            return FakeResponse({"organic_results": [{"title": "Amazon Widget", "asin": "B001", "price": "$10"}]})
        if engine == "walmart":
            return FakeResponse({"organic_results": [{"title": "Walmart Widget", "product_id": "W1", "price": "$11"}]})
        if engine == "ebay":
            return FakeResponse({"organic_results": [{"title": "eBay Widget", "item_id": "E1", "price": "$12"}]})
        if engine == "home_depot":
            return FakeResponse({"products": [{"title": "Depot Widget", "product_id": "H1", "price": "$13"}]})
        return FakeResponse({"places": [{"name": "Trip Hotel", "place_id": "T1", "rating": 4.2}]})

    monkeypatch.setattr("chack_tools.business_search.requests.get", fake_get)
    helper = BusinessSearchTool(ToolsConfig(business_max_results=5))

    amazon = helper.search_amazon_products("widget")
    walmart = helper.search_walmart_products("widget")
    ebay = helper.search_ebay_products("widget")
    depot = helper.search_home_depot_products("widget")
    tripadvisor = helper.search_tripadvisor("hotel")

    assert calls[0]["engine"] == "amazon"
    assert calls[0]["k"] == "widget"
    assert calls[1]["engine"] == "walmart"
    assert calls[1]["query"] == "widget"
    assert calls[2]["engine"] == "ebay"
    assert calls[2]["_nkw"] == "widget"
    assert calls[3]["engine"] == "home_depot"
    assert calls[3]["q"] == "widget"
    assert calls[4]["engine"] == "tripadvisor"
    assert calls[4]["q"] == "hotel"
    assert "Amazon Widget" in amazon
    assert "Walmart Widget" in walmart
    assert "eBay Widget" in ebay
    assert "Depot Widget" in depot
    assert "Trip Hotel" in tripadvisor


def test_business_researcher_is_registered_when_enabled():
    toolset = AgentsToolset(
        ToolsConfig(business_enabled=True),
        model_provider="openai",
        default_model="gpt-test",
    )

    assert "business_research" in _tool_names(toolset.tools)


def test_business_agent_includes_expanded_serpapi_tools(monkeypatch):
    from chack_tools.business_research_agent import BusinessResearchAgentTool

    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    helper = BusinessResearchAgentTool(
        ToolsConfig(business_enabled=True),
        model_provider="openai",
        fallback_model="gpt-test",
    )

    names = _tool_names(helper._build_subagent_tools())
    assert "search_opencorporates_companies" not in names
    assert "search_google_maps_businesses" in names
    assert "get_google_maps_reviews" in names
    assert "search_yelp_businesses" in names
    assert "search_google_ads_transparency" in names
    assert "search_google_shopping" in names
    assert "search_amazon_products" in names
    assert "search_walmart_products" in names
    assert "search_ebay_products" in names
    assert "search_home_depot_products" in names
    assert "search_tripadvisor" in names
