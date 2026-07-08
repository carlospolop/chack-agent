from chack_tools.config import ToolsConfig
from chack_tools.scientific_research_agent import ScientificResearchAgentTool
from chack_tools.scientific_search import ScientificSearchTool


class _Response:
    status_code = 200
    text = "{}"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_google_scholar_sends_advanced_params_and_formats_ids(monkeypatch):
    seen = {}

    def fake_get(url, *, params, timeout):
        seen["params"] = dict(params)
        return _Response(
            {
                "organic_results": [
                    {
                        "title": "A useful review",
                        "link": "https://example.com/paper",
                        "result_id": "SCHOLAR123",
                        "snippet": "Review paper snippet.",
                        "publication_info": {"summary": "A Researcher - Journal, 2024"},
                        "resources": [
                            {
                                "title": "PDF",
                                "link": "https://example.com/paper.pdf",
                                "file_format": "PDF",
                            }
                        ],
                        "inline_links": {
                            "cited_by": {"total": 42, "cites_id": "CITES123"},
                            "versions": {"total": 5, "cluster_id": "CLUSTER123"},
                            "serpapi_cite_link": "https://serpapi.com/search?engine=google_scholar_cite&q=SCHOLAR123",
                        },
                    }
                ]
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)

    tool = ScientificSearchTool(ToolsConfig(scientific_max_results=3))
    result = tool.search_google_scholar(
        query="machine learning cancer",
        start_year=2020,
        end_year=2024,
        sort_by_date=True,
        review_articles_only=True,
        exclude_citations=True,
        hl="en",
    )

    assert seen["params"]["engine"] == "google_scholar"
    assert seen["params"]["as_ylo"] == 2020
    assert seen["params"]["as_yhi"] == 2024
    assert seen["params"]["scisbd"] == "2"
    assert seen["params"]["as_rr"] == "1"
    assert seen["params"]["as_vis"] == "1"
    assert seen["params"]["hl"] == "en"
    assert "result_id: SCHOLAR123" in result
    assert "cites_id CITES123" in result
    assert "cluster CLUSTER123" in result
    assert "Cite lookup:" in result


def test_google_scholar_cite_formats_citations_and_exports(monkeypatch):
    def fake_get(url, *, params, timeout):
        assert params["engine"] == "google_scholar_cite"
        assert params["q"] == "SCHOLAR123"
        return _Response(
            {
                "citations": [
                    {"title": "MLA", "snippet": "MLA citation text"},
                    {"title": "BibTeX", "snippet": "@article{example}"},
                ],
                "links": [
                    {"name": "BibTeX", "link": "https://example.com/bibtex"},
                ],
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)

    tool = ScientificSearchTool(ToolsConfig())
    result = tool.search_google_scholar_cite("SCHOLAR123")

    assert "MLA citation text" in result
    assert "@article{example}" in result
    assert "BibTeX: https://example.com/bibtex" in result


def test_google_scholar_clamps_num_to_serpapi_max(monkeypatch):
    seen = {}

    def fake_get(url, *, params, timeout):
        seen["params"] = dict(params)
        return _Response({"organic_results": []})

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)

    tool = ScientificSearchTool(ToolsConfig(scientific_max_results=50))
    tool.search_google_scholar(query="large result request")

    assert seen["params"]["num"] == 20


def test_google_patents_details_formats_detail_payload(monkeypatch):
    def fake_get(url, *, params, timeout):
        assert params["engine"] == "google_patents_details"
        assert params["patent_id"] == "patent/US123/en"
        return _Response(
            {
                "patent_results": {
                    "title": "Scientific device patent",
                    "publication_number": "US123",
                    "assignee": "Example Labs",
                    "inventor": "A. Inventor",
                    "publication_date": "2024-01-01",
                    "pdf": "https://patents.example/US123.pdf",
                    "abstract": "A device for testing scientific examples.",
                    "claims": [{"text": "1. A device comprising a sensor."}],
                    "classifications": [{"code": "G01N", "description": "Investigating materials"}],
                    "citations": [{"publication_number": "US456", "title": "Prior patent"}],
                }
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)

    tool = ScientificSearchTool(ToolsConfig())
    result = tool.search_google_patents_details("patent/US123/en")

    assert "Scientific device patent" in result
    assert "Assignee: Example Labs" in result
    assert "PDF: https://patents.example/US123.pdf" in result
    assert "Claims:" in result
    assert "Classifications:" in result
    assert "Citations:" in result


def test_scientific_agent_includes_new_serpapi_tools():
    helper = ScientificResearchAgentTool(ToolsConfig(), model_provider="openai")
    names = [getattr(tool, "name", "") for tool in helper._build_subagent_tools()]

    assert "search_google_patents_details" in names
    assert "search_google_scholar_cite" in names
