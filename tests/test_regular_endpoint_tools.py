from chack_agent.agent import Chack
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.open_research_agents import (
    build_data_statistics_agent,
    build_knowledge_graph_agent,
    build_legal_agent,
    build_news_media_agent,
    build_religious_agent,
    get_data_statistics_research_tool,
    get_knowledge_graph_research_tool,
    get_legal_research_tool,
    get_news_media_research_tool,
    get_religious_research_tool,
)
from chack_tools.scientific_research_agent import ScientificResearchAgentTool, get_scientific_research_tool
from chack_tools.social_network_agent import SocialNetworkAgentTool, get_social_network_research_tool
from chack_tools.business_research_agent import BusinessResearchAgentTool, get_business_research_tool
from chack_tools.product_research_agent import ProductResearchAgentTool, get_product_research_tool
from chack_tools.cli_research_agent import CliResearchAgentTool, get_cli_research_tool
from chack_tools.websearcher_agent import WebSearcherAgentTool
from chack_tools.websearcher_agent import get_websearcher_research_tool


def _tool_names(tools):
    return {
        str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")
        for tool in tools
    }


def test_regular_agent_endpoint_tools_disabled_by_default():
    toolset = AgentsToolset(
        ToolsConfig(task_steps_manager_enabled=False),
        model_provider="openai",
        default_model="gpt-test",
    )

    assert _tool_names(toolset.tools) == set()


def test_regular_agent_can_enable_new_endpoint_tools(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    config = ToolsConfig(
        task_steps_manager_enabled=False,
        open_research_fetch_url_text_enabled=True,
        open_research_web_archive_enabled=True,
        open_research_gdelt_enabled=True,
        serpapi_bing_copilot_enabled=True,
        scientific_crossref_enabled=True,
        scientific_clinicaltrials_enabled=True,
        scientific_biorxiv_enabled=True,
        scientific_retraction_watch_enabled=True,
        scientific_pubchem_enabled=True,
        social_network_mastodon_enabled=True,
        social_network_tiktok_web_enabled=True,
        social_network_bluesky_web_enabled=True,
        legal_boe_enabled=True,
        legal_federal_register_enabled=True,
        data_statistics_world_bank_enabled=True,
        knowledge_graph_wikidata_enabled=True,
        religious_bible_enabled=True,
        religious_sefaria_enabled=True,
        religious_quran_enabled=True,
        religious_gita_enabled=True,
        religious_hadith_enabled=True,
        religious_suttacentral_enabled=True,
        product_open_food_facts_enabled=True,
        product_openfda_enabled=True,
        product_nvd_enabled=True,
        product_cpsc_enabled=True,
        product_cisa_kev_enabled=True,
        product_osv_enabled=True,
        product_google_lens_enabled=True,
        business_sec_enabled=True,
        business_gleif_enabled=True,
        business_google_finance_enabled=True,
        business_google_maps_enabled=True,
        business_google_maps_reviews_enabled=True,
        business_yelp_enabled=True,
        business_tripadvisor_enabled=True,
    )
    toolset = AgentsToolset(config, model_provider="openai", default_model="gpt-test")
    names = _tool_names(toolset.tools)

    expected = {
        "fetch_url_text",
        "web_archive_search",
        "wayback_fetch",
        "gdelt_news_search",
        "search_bing_copilot",
        "crossref_search",
        "crossref_doi_lookup",
        "clinicaltrials_search",
        "clinicaltrial_get",
        "biorxiv_search",
        "biorxiv_download",
        "retraction_watch",
        "pubchem_search",
        "mastodon_search",
        "tiktok_web_search",
        "bluesky_web_search",
        "boe_law_search",
        "boe_law_metadata_get",
        "boe_law_text_download",
        "boe_aux_table_get",
        "federal_register_search",
        "world_bank_indicator",
        "wikidata_entity_search",
        "wikidata_sparql",
        "bible_passage_get",
        "sefaria_search",
        "sefaria_text_get",
        "quran_search",
        "quran_verse_get",
        "quran_chapters_get",
        "gita_chapters_get",
        "gita_chapter_get",
        "gita_verse_get",
        "hadith_editions_get",
        "hadith_search",
        "hadith_collection_get",
        "hadith_section_get",
        "suttacentral_suttaplex_get",
        "suttacentral_text_get",
        "search_open_food_facts_products",
        "get_open_food_facts_product",
        "search_openfda_recalls",
        "search_nvd_cpe_products",
        "search_nvd_cve_vulnerabilities",
        "cpsc_recalls_search",
        "cisa_kev_search",
        "osv_package_query",
        "search_google_lens_products",
        "search_sec_companies",
        "get_sec_company_submissions",
        "get_sec_company_facts",
        "search_gleif_lei",
        "get_gleif_lei_record",
        "search_google_finance",
        "search_google_maps_businesses",
        "get_google_maps_reviews",
        "search_yelp_businesses",
        "get_yelp_place",
        "get_yelp_reviews",
        "search_tripadvisor",
        "get_tripadvisor_place",
        "get_tripadvisor_reviews",
    }

    assert expected <= names
    assert len(names) == len(toolset.tools)


def test_new_endpoint_tools_have_emojis():
    expected = {
        "fetch_url_text": "🌐",
        "web_archive_search": "🕰️",
        "gdelt_news_search": "🌎",
        "crossref_search": "🔗",
        "clinicaltrials_search": "🧪",
        "boe_law_search": "🇪🇸",
        "world_bank_indicator": "🌍",
        "wikidata_sparql": "🕸️",
        "bible_passage_get": "✝️",
        "quran_verse_get": "📖",
        "gita_verse_get": "🕉️",
        "hadith_search": "☪️",
        "suttacentral_text_get": "☸️",
        "search_google_lens_products": "📷",
        "search_google_finance": "💹",
        "search_yelp_businesses": "🍽️",
        "search_tripadvisor": "🧭",
        "cli_research": "🧪",
        "list_research_artifacts": "📁",
        "read_research_artifact": "📄",
        "grep_research_artifacts": "🔍",
        "researcher_queue_create": "🗂️",
        "researcher_queue": "📥",
        "researcher_queue_status": "📡",
        "start_researchers_async": "🚀",
        "poll_researchers_async": "📡",
        "cancel_researchers_async": "🛑",
    }

    for tool_name, emoji in expected.items():
        assert Chack._tool_emoji(tool_name) == emoji


def test_all_enable_flags_register_unique_regular_tools(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    flags = {
        name: True
        for name in ToolsConfig.__dataclass_fields__
        if name.endswith("_enabled")
    }
    flags["task_steps_manager_enabled"] = False
    toolset = AgentsToolset(
        ToolsConfig(**flags),
        model_provider="openai",
        default_model="gpt-test",
    )
    names = [
        str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")
        for tool in toolset.tools
    ]

    assert len(names) == len(set(names))
    assert "search_bing_copilot" in names
    assert "websearcher_research" in names
    assert "social_network_research" in names
    assert "scientific_research" in names
    assert "business_research" in names
    assert "product_research" in names
    assert "legal_research" in names
    assert "data_statistics_research" in names
    assert "news_media_research" in names
    assert "knowledge_graph_research" in names
    assert "religious_research" in names
    assert "cli_research" in names


def test_researcher_flags_register_researcher_tools_without_endpoint_flags(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    toolset = AgentsToolset(
        ToolsConfig(
            task_steps_manager_enabled=False,
            websearcher_enabled=True,
            social_network_enabled=True,
            scientific_enabled=True,
            business_enabled=True,
            product_enabled=True,
            legal_enabled=True,
            data_statistics_enabled=True,
            news_media_enabled=True,
            knowledge_graph_enabled=True,
            religious_enabled=True,
            cli_enabled=True,
        ),
        model_provider="openai",
        default_model="gpt-test",
    )
    names = _tool_names(toolset.tools)

    assert {
        "websearcher_research",
        "social_network_research",
        "scientific_research",
        "business_research",
        "product_research",
        "legal_research",
        "data_statistics_research",
        "news_media_research",
        "knowledge_graph_research",
        "religious_research",
        "cli_research",
    } <= names
    assert "crossref_search" not in names
    assert "bible_passage_get" not in names
    assert "search_sec_companies" not in names


def test_open_specialist_researchers_have_unique_internal_toolsets(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    config = ToolsConfig(task_steps_manager_enabled=False)
    helpers = {
        "legal": build_legal_agent(config, model_provider="openai", fallback_model="gpt-test"),
        "data": build_data_statistics_agent(config, model_provider="openai", fallback_model="gpt-test"),
        "news": build_news_media_agent(config, model_provider="openai", fallback_model="gpt-test"),
        "kg": build_knowledge_graph_agent(config, model_provider="openai", fallback_model="gpt-test"),
        "religious": build_religious_agent(config, model_provider="openai", fallback_model="gpt-test"),
    }

    names_by_kind = {
        kind: [getattr(tool, "name", "") for tool in helper._build_subagent_tools()]
        for kind, helper in helpers.items()
    }

    for names in names_by_kind.values():
        assert len(names) == len(set(names))
        assert "list_research_artifacts" in names
        assert "read_research_artifact" in names
        assert "grep_research_artifacts" in names
    assert "boe_law_search" in names_by_kind["legal"]
    assert "federal_register_search" in names_by_kind["legal"]
    assert "run_shell_command" in names_by_kind["data"]
    assert "world_bank_indicator" in names_by_kind["data"]
    assert "search_google_news" in names_by_kind["news"]
    assert "wikidata_sparql" in names_by_kind["kg"]
    assert "bible_passage_get" in names_by_kind["religious"]
    assert "suttacentral_text_get" in names_by_kind["religious"]


def test_websearcher_researcher_has_unique_internal_toolset(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    helper = WebSearcherAgentTool(
        ToolsConfig(task_steps_manager_enabled=False),
        model_provider="openai",
        fallback_model="gpt-test",
    )

    names = [getattr(tool, "name", "") for tool in helper._build_subagent_tools()]

    assert len(names) == len(set(names))
    assert "fetch_url_text" in names
    assert "web_archive_search" in names
    assert "gdelt_news_search" in names
    assert "search_bing_copilot" in names
    assert "list_research_artifacts" in names
    assert "read_research_artifact" in names
    assert "grep_research_artifacts" in names


def test_researcher_agent_tools_have_schema_descriptions(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    config = ToolsConfig(task_steps_manager_enabled=False)
    kwargs = {"model_provider": "openai", "fallback_model": "gpt-test"}
    tools = [
        get_websearcher_research_tool(WebSearcherAgentTool(config, **kwargs)),
        get_scientific_research_tool(ScientificResearchAgentTool(config, **kwargs)),
        get_social_network_research_tool(SocialNetworkAgentTool(config, **kwargs)),
        get_business_research_tool(BusinessResearchAgentTool(config, **kwargs)),
        get_product_research_tool(ProductResearchAgentTool(config, **kwargs)),
        get_cli_research_tool(CliResearchAgentTool(config, **kwargs)),
        get_legal_research_tool(build_legal_agent(config, **kwargs)),
        get_data_statistics_research_tool(build_data_statistics_agent(config, **kwargs)),
        get_news_media_research_tool(build_news_media_agent(config, **kwargs)),
        get_knowledge_graph_research_tool(build_knowledge_graph_agent(config, **kwargs)),
        get_religious_research_tool(build_religious_agent(config, **kwargs)),
    ]

    for tool in tools:
        description = str(getattr(tool, "description", "") or "")
        properties = getattr(tool, "params_json_schema", {}).get("properties", {})
        assert len(description) > 40, tool.name
        assert "Output:" in description, tool.name
        assert "Parameters:" in description or "Args:" in description, tool.name
        assert properties["prompt"].get("description"), tool.name
        assert properties["save_artifacts"].get("description"), tool.name


def test_all_enabled_regular_tools_have_schema_descriptions(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    flags = {
        name: True
        for name in ToolsConfig.__dataclass_fields__
        if name.endswith("_enabled")
    }
    flags["task_steps_manager_enabled"] = False
    toolset = AgentsToolset(ToolsConfig(**flags), model_provider="openai", default_model="gpt-test")

    for tool in toolset.tools:
        name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
        description = str(getattr(tool, "description", "") or "")
        assert description, name
        assert "Output:" in description, name
        if (getattr(tool, "params_json_schema", {}) or {}).get("properties", {}):
            assert "Parameters:" in description or "Args:" in description, name
        for param_name, param_schema in (getattr(tool, "params_json_schema", {}) or {}).get("properties", {}).items():
            assert param_schema.get("description"), f"{name}.{param_name}"


def test_researcher_internal_tools_have_schema_descriptions(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    config = ToolsConfig(task_steps_manager_enabled=False)
    kwargs = {"model_provider": "openai", "fallback_model": "gpt-test"}
    helpers = [
        WebSearcherAgentTool(config, **kwargs),
        ScientificResearchAgentTool(config, **kwargs),
        SocialNetworkAgentTool(config, **kwargs),
        BusinessResearchAgentTool(config, **kwargs),
        ProductResearchAgentTool(config, **kwargs),
        CliResearchAgentTool(config, **kwargs),
        build_legal_agent(config, **kwargs),
        build_data_statistics_agent(config, **kwargs),
        build_news_media_agent(config, **kwargs),
        build_knowledge_graph_agent(config, **kwargs),
        build_religious_agent(config, **kwargs),
    ]

    for helper in helpers:
        for tool in helper._build_subagent_tools():
            name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
            description = str(getattr(tool, "description", "") or "")
            assert description, name
            assert "Output:" in description, name
            if (getattr(tool, "params_json_schema", {}) or {}).get("properties", {}):
                assert "Parameters:" in description or "Args:" in description, name
            for param_name, param_schema in (getattr(tool, "params_json_schema", {}) or {}).get("properties", {}).items():
                assert param_schema.get("description"), f"{name}.{param_name}"
