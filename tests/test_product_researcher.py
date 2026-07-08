import json

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.product_search import ProductSearchTool


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


def test_open_food_facts_search_and_product_write_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("params")))
        if url.endswith("/cgi/search.pl"):
            return FakeResponse({
                "products": [{
                    "product_name": "Nutella",
                    "brands": "Ferrero",
                    "code": "3017620422003",
                    "quantity": "400 g",
                    "nutriscore_grade": "e",
                    "ingredients_text": "sugar, palm oil, hazelnuts, cocoa",
                }]
            })
        assert url == "https://world.openfoodfacts.org/api/v2/product/3017620422003.json"
        return FakeResponse({
            "product": {
                "product_name": "Nutella",
                "brands": "Ferrero",
                "categories": "Spreads",
                "countries": "France, Spain",
                "ingredients_text": "sugar, palm oil, hazelnuts, cocoa",
            }
        })

    monkeypatch.setattr("chack_tools.product_search.requests.get", fake_get)
    helper = ProductSearchTool(ToolsConfig(product_max_results=5))

    search = helper.search_open_food_facts_products("nutella", max_results=2)
    product = helper.get_open_food_facts_product("3017620422003")

    assert calls[0][0] == "https://world.openfoodfacts.org/cgi/search.pl"
    assert calls[0][1]["search_terms"] == "nutella"
    assert calls[0][1]["page_size"] == 2
    assert "Nutella" in search
    assert "barcode: 3017620422003" in search
    assert "Countries: France, Spain" in product
    assert list((tmp_path / "open-food-facts").glob("search_nutella_*.json"))
    assert list((tmp_path / "open-food-facts").glob("barcode_3017620422003_*.json"))


def test_openfda_and_nvd_free_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("params"), kwargs.get("headers")))
        if "api.fda.gov" in url:
            return FakeResponse({
                "results": [{
                    "recalling_firm": "Example Foods",
                    "classification": "Class II",
                    "status": "Ongoing",
                    "recall_initiation_date": "20260601",
                    "product_description": "Example peanut butter",
                    "reason_for_recall": "Possible allergen issue",
                }]
            })
        if "cpes" in url:
            return FakeResponse({
                "products": [{
                    "cpe": {
                        "cpeName": "cpe:2.3:a:example:router:1.0:*:*:*:*:*:*:*",
                        "deprecated": False,
                        "titles": [{"title": "Example Router 1.0"}],
                    }
                }]
            })
        return FakeResponse({
            "vulnerabilities": [{
                "cve": {
                    "id": "CVE-2026-0001",
                    "published": "2026-01-01T00:00:00.000",
                    "lastModified": "2026-01-02T00:00:00.000",
                    "vulnStatus": "Analyzed",
                    "descriptions": [{"lang": "en", "value": "Example vulnerability in Example Router."}],
                    "metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "HIGH", "baseScore": 8.1}}]},
                }
            }]
        })

    monkeypatch.setattr("chack_tools.product_search.requests.get", fake_get)
    helper = ProductSearchTool(ToolsConfig(product_max_results=5))

    recalls = helper.search_openfda_recalls("peanut butter", product_type="food", max_results=2)
    cpe = helper.search_nvd_cpe("example router", max_results=2)
    cve = helper.search_nvd_cve("example router", max_results=2)

    assert calls[0][0] == "https://api.fda.gov/food/enforcement.json"
    assert calls[0][1]["search"] == 'product_description:"peanut butter"'
    assert calls[1][0] == "https://services.nvd.nist.gov/rest/json/cpes/2.0"
    assert calls[1][1]["keywordSearch"] == "example router"
    assert calls[2][0] == "https://services.nvd.nist.gov/rest/json/cves/2.0"
    assert "Example Foods" in recalls
    assert "Example Router 1.0" in cpe
    assert "CVE-2026-0001" in cve
    assert "HIGH 8.1" in cve
    assert list((tmp_path / "openfda").glob("food_peanut_butter_*.json"))
    assert list((tmp_path / "nvd-cpe").glob("cpe_example_router_*.json"))
    assert list((tmp_path / "nvd-cve").glob("cve_example_router_*.json"))


def test_google_lens_products_uses_serpapi_params(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return FakeResponse({
            "products": [{
                "title": "Similar Widget",
                "price": "$19.99",
                "source": "Example Shop",
                "link": "https://example.test/widget",
            }]
        })

    monkeypatch.setattr("chack_tools.product_search.requests.get", fake_get)
    helper = ProductSearchTool(ToolsConfig(product_max_results=5))

    result = helper.search_google_lens_products(
        "https://example.test/image.jpg",
        query="widget",
        country="us",
        hl="en",
        max_results=1,
    )

    assert captured["url"] == "https://serpapi.com/search"
    assert captured["params"]["engine"] == "google_lens"
    assert captured["params"]["url"] == "https://example.test/image.jpg"
    assert captured["params"]["type"] == "products"
    assert captured["params"]["q"] == "widget"
    assert captured["params"]["country"] == "us"
    assert captured["params"]["hl"] == "en"
    assert captured["params"]["api_key"] == "serp-key"
    assert "Similar Widget" in result
    assert list((tmp_path / "google-lens").glob("lens_https_example.test_image.jpg_*.json"))


def test_product_researcher_is_registered_when_enabled():
    toolset = AgentsToolset(
        ToolsConfig(product_enabled=True),
        model_provider="openai",
        default_model="gpt-test",
    )

    assert "product_research" in _tool_names(toolset.tools)


def test_product_agent_includes_expected_tools(monkeypatch):
    from chack_tools.product_research_agent import ProductResearchAgentTool

    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    helper = ProductResearchAgentTool(
        ToolsConfig(product_enabled=True),
        model_provider="openai",
        fallback_model="gpt-test",
    )

    names = _tool_names(helper._build_subagent_tools())
    assert "search_open_food_facts_products" in names
    assert "get_open_food_facts_product" in names
    assert "search_openfda_recalls" in names
    assert "search_nvd_cpe_products" in names
    assert "search_nvd_cve_vulnerabilities" in names
    assert "search_google_lens_products" in names
    assert "search_google_shopping" in names
    assert "search_amazon_products" in names
    assert "search_walmart_products" in names
    assert "search_ebay_products" in names
    assert "search_home_depot_products" in names
    assert "search_google_trends" in names
    assert "search_google_patents" in names
    assert "search_youtube_videos" in names
    assert "get_youtube_video_transcript" in names
