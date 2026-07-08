from chack_tools.config import ToolsConfig
from chack_tools.scientific_search import ScientificSearchTool


class _Response:
    status_code = 200

    def __init__(self, *, json_payload=None, text=""):
        self._json_payload = json_payload
        self.text = text

    def json(self):
        return self._json_payload

    def raise_for_status(self):
        return None


def test_search_and_download_pmc_full_text(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, *, params=None, timeout=None, **kwargs):
        calls.append((url, params or {}))
        if "esearch.fcgi" in url:
            return _Response(json_payload={"esearchresult": {"idlist": ["12345"]}})
        if "esummary.fcgi" in url:
            return _Response(
                json_payload={
                    "result": {
                        "12345": {
                            "title": "PMC article",
                            "source": "PMC Journal",
                            "pubdate": "2026",
                            "articleids": [{"idtype": "pmcid", "value": "PMC12345"}],
                        }
                    }
                }
            )
        return _Response(text="<article><body><p>" + ("full text " * 80) + "</p></body></article>")

    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))

    tool = ScientificSearchTool(ToolsConfig())
    result = tool.search_pmc_full_text("test query")
    assert "PMC12345" in result
    assert "Full-text XML" in result

    download = tool.download_pmc_full_text("PMC12345")
    assert "SUCCESS: Downloaded PMC full text" in download
    assert "Saved XML:" in download
    assert "Saved text:" in download


def test_search_and_download_ncbi_bookshelf(monkeypatch, tmp_path):
    def fake_get(url, *, params=None, timeout=None, **kwargs):
        params = params or {}
        if "esearch.fcgi" in url:
            return _Response(json_payload={"esearchresult": {"idlist": ["67890"]}})
        if "esummary.fcgi" in url:
            return _Response(
                json_payload={
                    "result": {
                        "67890": {
                            "title": "NCBI book chapter",
                            "accessionid": "NBK67890",
                            "rtype": "chapter",
                            "pubdate": "2026",
                        }
                    }
                }
            )
        return _Response(text="<html><body><main>" + ("book content " * 80) + "</main></body></html>")

    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))

    tool = ScientificSearchTool(ToolsConfig())
    result = tool.search_ncbi_bookshelf("test query")
    assert "NBK67890" in result
    assert "report=reader" in result

    download = tool.download_ncbi_bookshelf("NBK67890")
    assert "SUCCESS: Downloaded NCBI Bookshelf full content" in download
    assert "Saved HTML:" in download
    assert "Saved text:" in download


def test_search_and_download_medrxiv_full_text(monkeypatch, tmp_path):
    jats_url = "https://www.medrxiv.org/content/early/2026/01/01/test.source.xml"

    def fake_get(url, *, timeout=None, headers=None, **kwargs):
        if "api.medrxiv.org" in url:
            return _Response(
                json_payload={
                    "collection": [
                        {
                            "doi": "10.1101/test",
                            "title": "Cancer machine learning preprint",
                            "authors": "A. Researcher",
                            "date": "2026-01-01",
                            "category": "epidemiology",
                            "abstract": "Cancer machine learning abstract",
                            "jatsxml": jats_url,
                        }
                    ]
                }
            )
        return _Response(text="<article><body><p>" + ("medrxiv full text " * 80) + "</p></body></article>")

    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))

    tool = ScientificSearchTool(ToolsConfig())
    result = tool.search_medrxiv_preprints(
        "cancer machine learning",
        start_date="2026-01-01",
        end_date="2026-01-31",
    )
    assert "Full-text JATS XML" in result
    assert jats_url in result

    download = tool.download_medrxiv_full_text(jats_url)
    assert "SUCCESS: Downloaded medRxiv full text" in download
    assert "Saved XML:" in download
    assert "Saved text:" in download
