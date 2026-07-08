import json

from chack_tools.config import ToolsConfig
from chack_tools.open_research_agents import build_religious_agent
from chack_tools.open_research_sources import OpenResearchTool


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


def test_gita_tools_save_compact_json_and_text(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        if url.endswith("/chapters"):
            return FakeResponse([
                {
                    "chapter_number": 2,
                    "verses_count": 72,
                    "translation": "Sankhya Yoga",
                    "meaning": {"en": "Transcendental Knowledge"},
                }
            ])
        if url.endswith("/chapter/2"):
            return FakeResponse({
                "chapter_number": 2,
                "verses_count": 72,
                "translation": "Sankhya Yoga",
                "summary": {"en": "A chapter summary."},
            })
        assert url.endswith("/slok/2/47")
        return FakeResponse({
            "_id": "BG2.47",
            "slok": "karma verse",
            "transliteration": "karma transliteration",
            "siva": {"author": "Swami Sivananda", "et": "English translation"},
        })

    monkeypatch.setattr("chack_tools.open_research_sources.requests.get", fake_get)
    helper = OpenResearchTool(ToolsConfig())

    chapters = helper.get_gita_chapters()
    chapter = helper.get_gita_chapter(2)
    verse = helper.get_gita_verse(2, 47)

    assert seen == [
        "https://vedicscriptures.github.io/chapters",
        "https://vedicscriptures.github.io/chapter/2",
        "https://vedicscriptures.github.io/slok/2/47",
    ]
    assert "Sankhya Yoga" in chapters
    assert "A chapter summary." in chapter
    assert "Swami Sivananda" in verse
    assert "Artifact JSON:" in verse
    assert "Artifact text:" in verse
    assert list((tmp_path / "gita-verse").glob("2.47_*.json"))
    assert list((tmp_path / "gita-verse").glob("2.47_*.txt"))


def test_hadith_tools_save_editions_collection_and_section(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        if url.endswith("/editions.json"):
            return FakeResponse({
                "bukhari": {
                    "name": "Sahih al Bukhari",
                    "collection": [{"name": "eng-bukhari", "language": "English"}],
                }
            })
        if url.endswith("/editions/eng-bukhari.json"):
            return FakeResponse({
                "metadata": {"name": "Sahih al Bukhari", "sections": {"1": "Revelation"}},
                "hadiths": [
                    {"hadithnumber": 1, "text": "Narrated example one."},
                    {"hadithnumber": 2, "text": "Narrated example two."},
                ],
            })
        assert url.endswith("/editions/eng-bukhari/1.json")
        return FakeResponse({
            "metadata": {"name": "Sahih al Bukhari", "section": {"1": "Revelation"}},
            "hadiths": [{"hadithnumber": 1, "text": "Actions are by intentions."}],
        })

    monkeypatch.setattr("chack_tools.open_research_sources.requests.get", fake_get)
    helper = OpenResearchTool(ToolsConfig())

    editions = helper.get_hadith_editions()
    collection = helper.get_hadith_collection("eng-bukhari", max_hadiths=1)
    search = helper.search_hadith("eng-bukhari", "example two")
    section = helper.get_hadith_section("eng-bukhari", 1)

    assert "eng-bukhari" in editions
    assert "showing/saving first 1" in collection
    assert "Narrated example two." not in collection
    assert "Narrated example two." in search
    assert list((tmp_path / "hadith-search").glob("eng-bukhari-example_two_*.json"))
    assert "Actions are by intentions." in section
    assert "Artifact JSON:" in section
    assert "Artifact text:" in section
    assert list((tmp_path / "hadith-section").glob("eng-bukhari-1_*.json"))
    assert list((tmp_path / "hadith-section").glob("eng-bukhari-1_*.txt"))


def test_suttacentral_tools_save_metadata_and_full_text(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        if url == "https://suttacentral.net/api/suttaplex/mn1":
            return FakeResponse([
                {
                    "acronym": "MN 1",
                    "original_title": "Mūlapariyāyasutta",
                    "translated_title": "The Root of All Things",
                    "root_lang_name": "Pāli",
                    "translations": [{"id": "mn1_translation-en-sujato", "lang": "en", "author_uid": "sujato"}],
                }
            ])
        assert url == (
            "https://raw.githubusercontent.com/suttacentral/bilara-data/published/"
            "translation/en/sujato/sutta/mn/mn1_translation-en-sujato.json"
        )
        return FakeResponse({"mn1:1.1": "So I have heard.", "mn1:1.2": "At one time..."})

    monkeypatch.setattr("chack_tools.open_research_sources.requests.get", fake_get)
    helper = OpenResearchTool(ToolsConfig())

    metadata = helper.get_suttacentral_suttaplex("mn1")
    text = helper.get_suttacentral_text("mn1")

    assert "MN 1" in metadata
    assert "Mūlapariyāyasutta" in metadata
    assert "So I have heard." in text
    assert "Artifact JSON:" in text
    assert "Artifact text:" in text
    assert list((tmp_path / "suttacentral-text").glob("mn1-translation-en-sujato_*.json"))
    assert list((tmp_path / "suttacentral-text").glob("mn1-translation-en-sujato_*.txt"))


def test_religious_agent_includes_expanded_primary_text_tools(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    helper = build_religious_agent(
        ToolsConfig(task_steps_manager_enabled=False),
        model_provider="openai",
        fallback_model="gpt-test",
    )

    names = _tool_names(helper._build_subagent_tools())

    assert "bible_passage_get" in names
    assert "sefaria_search" in names
    assert "quran_verse_get" in names
    assert "gita_verse_get" in names
    assert "hadith_search" in names
    assert "hadith_section_get" in names
    assert "suttacentral_text_get" in names
    assert "search_google_web" in names
