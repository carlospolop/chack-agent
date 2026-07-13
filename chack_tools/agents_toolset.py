import os
from dataclasses import replace

from .config import ToolsConfig
from .brave_search import BraveSearchTool, get_brave_search_tool
from .business_research_agent import BusinessResearchAgentTool, get_business_research_tool
from .business_search import (
    BusinessSearchTool,
    get_amazon_product_search_tool,
    get_amazon_product_tool,
    get_apple_maps_place_tool,
    get_apple_maps_search_tool,
    get_ebay_product_search_tool,
    get_ebay_product_tool,
    get_gleif_lei_record_tool,
    get_gleif_lei_search_tool,
    get_google_ads_search_tool,
    get_google_ads_transparency_tool,
    get_google_finance_markets_tool,
    get_google_finance_search_tool,
    get_google_immersive_product_tool,
    get_google_maps_reviews_tool,
    get_google_maps_search_tool,
    get_google_shopping_light_search_tool,
    get_google_shopping_search_tool,
    get_home_depot_product_search_tool,
    get_home_depot_product_tool,
    get_sec_company_facts_tool,
    get_sec_company_search_tool,
    get_sec_company_submissions_tool,
    get_tripadvisor_place_tool,
    get_tripadvisor_reviews_tool,
    get_tripadvisor_search_tool,
    get_walmart_product_search_tool,
    get_walmart_product_tool,
    get_yelp_business_search_tool,
    get_yelp_place_tool,
    get_yelp_reviews_tool,
)
from .exec_tool import ExecTool, get_exec_tool
from .forumscout_search import (
    ForumScoutTool,
    get_bluesky_web_search_tool,
    get_facebook_profile_tool,
    get_forum_search_tool,
    get_google_forums_search_tool,
    get_google_news_search_tool,
    get_google_trends_search_tool,
    get_google_trends_trending_now_tool,
    get_google_videos_search_tool,
    get_instagram_profile_tool,
    get_instagram_search_tool,
    get_linkedin_search_tool,
    get_reddit_comments_search_tool,
    get_reddit_posts_search_tool,
    get_tiktok_web_search_tool,
    get_x_search_tool,
)
from .open_research_sources import (
    OpenResearchTool,
    get_bible_passage_tool,
    get_biorxiv_download_tool,
    get_biorxiv_search_tool,
    get_boe_aux_table_tool,
    get_boe_law_metadata_tool,
    get_boe_law_search_tool,
    get_boe_law_text_download_tool,
    get_cisa_kev_search_tool,
    get_clinicaltrial_get_tool,
    get_clinicaltrials_search_tool,
    get_cpsc_recalls_search_tool,
    get_crossref_doi_lookup_tool,
    get_crossref_search_tool,
    get_federal_register_search_tool,
    get_fetch_url_text_tool,
    get_gdelt_news_search_tool,
    get_gita_chapter_tool,
    get_gita_chapters_tool,
    get_gita_verse_tool,
    get_hadith_collection_tool,
    get_hadith_editions_tool,
    get_hadith_search_tool,
    get_hadith_section_tool,
    get_mastodon_search_tool,
    get_osv_package_query_tool,
    get_pubchem_search_tool,
    get_quran_chapters_tool,
    get_quran_search_tool,
    get_quran_verse_tool,
    get_retraction_watch_tool,
    get_sefaria_search_tool,
    get_sefaria_text_tool,
    get_suttacentral_suttaplex_tool,
    get_suttacentral_text_tool,
    get_wayback_fetch_tool,
    get_web_archive_search_tool,
    get_wikidata_entity_search_tool,
    get_wikidata_sparql_tool,
    get_world_bank_indicator_tool,
)
from .pdf_text import PdfTextTool, get_pdf_text_tool
from .playwright_fetch import (
    PlaywrightFetchTool,
    get_playwright_fetch_tool,
    is_playwright_available,
)
from .product_research_agent import ProductResearchAgentTool, get_product_research_tool
from .product_search import (
    ProductSearchTool,
    get_google_lens_products_tool,
    get_nvd_cpe_search_tool,
    get_nvd_cve_search_tool,
    get_open_food_facts_product_tool,
    get_open_food_facts_search_tool,
    get_openfda_recalls_search_tool,
)
from .open_research_agents import (
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
from .researcher_administrator_agent import (
    ResearcherAdministratorAgentTool,
    get_researcher_administrator_tool,
)
from .scientific_research_agent import ScientificResearchAgentTool, get_scientific_research_tool
from .scientific_search import (
    ScientificSearchTool,
    get_arxiv_search_tool,
    get_europe_pmc_search_tool,
    get_google_patents_details_tool,
    get_google_patents_search_tool,
    get_google_scholar_cite_tool,
    get_google_scholar_search_tool,
    get_medrxiv_full_text_download_tool,
    get_medrxiv_preprint_search_tool,
    get_ncbi_bookshelf_download_tool,
    get_ncbi_bookshelf_search_tool,
    get_openalex_search_tool,
    get_plos_search_tool,
    get_pmc_full_text_download_tool,
    get_pmc_full_text_search_tool,
    get_semantic_scholar_search_tool,
    get_youtube_transcript_tool,
    get_youtube_video_details_tool,
    get_youtube_video_search_tool,
)
from .serpapi_web_search import (
    SerpApiWebSearchTool,
    get_google_web_search_tool,
    get_bing_web_search_tool,
    get_google_ai_mode_tool,
    get_bing_copilot_tool,
)
from .social_network_agent import SocialNetworkAgentTool, get_social_network_research_tool
from .task_steps_manager_tool import TaskStepsManagerTool, get_task_steps_manager_tool
from .websearcher_agent import WebSearcherAgentTool, get_websearcher_research_tool
from .cli_research_agent import CliResearchAgentTool, get_cli_research_tool
from .subchack_research_agent import SubChackResearchAgentTool, get_subchack_research_tool
from .serpapi_keys import has_serpapi_keys


class AgentsToolset:
    def __init__(
        self,
        config: ToolsConfig,
        model_provider: str = "",
        default_model: str = "",
        social_network_model: str = "CHEAP_BUT_QUALITY",
        scientific_model: str = "CHEAP_BUT_QUALITY",
        websearcher_model: str = "CHEAP_BUT_QUALITY",
        business_model: str = "CHEAP_BUT_QUALITY",
        product_model: str = "CHEAP_BUT_QUALITY",
        legal_model: str = "CHEAP_BUT_QUALITY",
        data_statistics_model: str = "CHEAP_BUT_QUALITY",
        news_media_model: str = "CHEAP_BUT_QUALITY",
        knowledge_graph_model: str = "CHEAP_BUT_QUALITY",
        religious_model: str = "CHEAP_BUT_QUALITY",
        cli_model: str = "CHEAP_BUT_QUALITY",
        subchack_model: str = "",
        researcher_administrator_model: str = "",
        social_network_max_turns: int = 30,
        scientific_max_turns: int = 30,
        websearcher_max_turns: int = 30,
        business_max_turns: int = 30,
        product_max_turns: int = 30,
        legal_max_turns: int = 30,
        data_statistics_max_turns: int = 30,
        news_media_max_turns: int = 30,
        knowledge_graph_max_turns: int = 30,
        religious_max_turns: int = 30,
        cli_max_turns: int = 30,
        subchack_max_turns: int = 30,
        researcher_administrator_max_turns: int = 100,
        self_critique_enabled: bool = False,
        self_critique_rounds: int = 0,
        # Backward-compatibility shim for older integrations that still pass a
        # `tool_profile` kwarg (e.g. CLI smoke tests).
        tool_profile: str = "",
    ):
        self.config = config
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.tool_profile = str(tool_profile or "").strip()
        self.default_model = self._resolve_alias(default_model, fallback="")
        self.social_network_model = self._resolve_alias(
            social_network_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.scientific_model = self._resolve_alias(
            scientific_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.websearcher_model = self._resolve_alias(
            websearcher_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.business_model = self._resolve_alias(
            business_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.product_model = self._resolve_alias(
            product_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.legal_model = self._resolve_alias(
            legal_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.data_statistics_model = self._resolve_alias(
            data_statistics_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.news_media_model = self._resolve_alias(
            news_media_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.knowledge_graph_model = self._resolve_alias(
            knowledge_graph_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.religious_model = self._resolve_alias(
            religious_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.cli_model = self._resolve_alias(
            cli_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.subchack_model = self._resolve_alias(
            subchack_model,
            fallback="",
        )
        self.researcher_administrator_model = self._resolve_alias(
            researcher_administrator_model,
            fallback="",
        )
        self.social_network_max_turns = social_network_max_turns
        self.scientific_max_turns = scientific_max_turns
        self.websearcher_max_turns = websearcher_max_turns
        self.business_max_turns = business_max_turns
        self.product_max_turns = product_max_turns
        self.legal_max_turns = legal_max_turns
        self.data_statistics_max_turns = data_statistics_max_turns
        self.news_media_max_turns = news_media_max_turns
        self.knowledge_graph_max_turns = knowledge_graph_max_turns
        self.religious_max_turns = religious_max_turns
        self.cli_max_turns = cli_max_turns
        self.subchack_max_turns = subchack_max_turns
        self.researcher_administrator_max_turns = researcher_administrator_max_turns
        self.self_critique_rounds = max(0, int(self_critique_rounds or 0))
        self.self_critique_enabled = bool(self_critique_enabled or self.self_critique_rounds > 0)
        self.tools = self._build_tools()

    def _resolve_alias(self, value: str, *, fallback: str) -> str:
        raw = str(value or "").strip() or fallback
        if not raw:
            return ""
        try:
            from chack_agent.model_aliases import resolve_model_alias

            return resolve_model_alias(raw, provider=self.model_provider)
        except Exception:
            return raw

    @staticmethod
    def _tool_name(tool) -> str:
        return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")

    def _dedupe_tools(self, tools):
        seen = set()
        unique = []
        for tool in tools:
            name = self._tool_name(tool)
            if name and name in seen:
                continue
            if name:
                seen.add(name)
            unique.append(tool)
        return unique

    def _build_tools(self):
        tools = []
        if self.config.exec_enabled:
            exec_helper = ExecTool(self.config)
            tools.append(get_exec_tool(exec_helper))

        if getattr(self.config, "task_steps_manager_enabled", True):
            task_helper = TaskStepsManagerTool(self.config)
            tools.append(get_task_steps_manager_tool(task_helper))

        if self.config.brave_enabled:
            brave_helper = BraveSearchTool(self.config)
            tools.append(get_brave_search_tool(brave_helper))

        if self.config.playwright_enabled and is_playwright_available():
            playwright_helper = PlaywrightFetchTool(self.config)
            tools.append(get_playwright_fetch_tool(playwright_helper))

        has_serpapi = has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if has_serpapi and (self.config.serpapi_google_web_enabled or self.config.websearcher_google_web_enabled):
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(get_google_web_search_tool(web_helper))

        if has_serpapi and (self.config.serpapi_bing_web_enabled or self.config.websearcher_bing_web_enabled):
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(get_bing_web_search_tool(web_helper))

        if has_serpapi and self.config.websearcher_google_ai_mode_enabled:
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(get_google_ai_mode_tool(web_helper))

        if has_serpapi and (self.config.serpapi_bing_copilot_enabled or self.config.websearcher_bing_copilot_enabled):
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(get_bing_copilot_tool(web_helper))

        if self.config.websearcher_brave_enabled:
            brave_helper = BraveSearchTool(self.config)
            tools.append(get_brave_search_tool(brave_helper))

        open_helper = OpenResearchTool(self.config)
        if self.config.open_research_fetch_url_text_enabled or self.config.websearcher_fetch_url_text_enabled:
            tools.append(get_fetch_url_text_tool(open_helper))

        if self.config.open_research_web_archive_enabled or self.config.websearcher_web_archive_enabled:
            tools.append(get_web_archive_search_tool(open_helper))
            tools.append(get_wayback_fetch_tool(open_helper))

        if self.config.open_research_gdelt_enabled or self.config.websearcher_gdelt_enabled:
            tools.append(get_gdelt_news_search_tool(open_helper))

        if self.config.open_research_federal_register_enabled or self.config.legal_federal_register_enabled:
            tools.append(get_federal_register_search_tool(open_helper))

        if self.config.open_research_world_bank_enabled or self.config.data_statistics_world_bank_enabled:
            tools.append(get_world_bank_indicator_tool(open_helper))

        if (
            self.config.open_research_wikidata_enabled
            or self.config.legal_wikidata_enabled
            or self.config.data_statistics_wikidata_enabled
            or self.config.knowledge_graph_wikidata_enabled
            or self.config.religious_wikidata_enabled
        ):
            tools.append(get_wikidata_entity_search_tool(open_helper))
            tools.append(get_wikidata_sparql_tool(open_helper))

        if self.config.legal_boe_enabled:
            tools.append(get_boe_law_search_tool(open_helper))
            tools.append(get_boe_law_metadata_tool(open_helper))
            tools.append(get_boe_law_text_download_tool(open_helper))
            tools.append(get_boe_aux_table_tool(open_helper))

        if self.config.religious_bible_enabled:
            tools.append(get_bible_passage_tool(open_helper))
        if self.config.religious_sefaria_enabled:
            tools.append(get_sefaria_search_tool(open_helper))
            tools.append(get_sefaria_text_tool(open_helper))
        if self.config.religious_quran_enabled:
            tools.append(get_quran_search_tool(open_helper))
            tools.append(get_quran_verse_tool(open_helper))
            tools.append(get_quran_chapters_tool(open_helper))
        if self.config.religious_gita_enabled:
            tools.append(get_gita_chapters_tool(open_helper))
            tools.append(get_gita_chapter_tool(open_helper))
            tools.append(get_gita_verse_tool(open_helper))
        if self.config.religious_hadith_enabled:
            tools.append(get_hadith_editions_tool(open_helper))
            tools.append(get_hadith_search_tool(open_helper))
            tools.append(get_hadith_collection_tool(open_helper))
            tools.append(get_hadith_section_tool(open_helper))
        if self.config.religious_suttacentral_enabled:
            tools.append(get_suttacentral_suttaplex_tool(open_helper))
            tools.append(get_suttacentral_text_tool(open_helper))

        scientific_helper = ScientificSearchTool(self.config)
        if self.config.scientific_arxiv_enabled:
            tools.append(get_arxiv_search_tool(scientific_helper))
        if self.config.scientific_europe_pmc_enabled:
            tools.append(get_europe_pmc_search_tool(scientific_helper))
        if self.config.scientific_pmc_full_text_enabled:
            tools.append(get_pmc_full_text_search_tool(scientific_helper))
            tools.append(get_pmc_full_text_download_tool(scientific_helper))
        if self.config.scientific_ncbi_bookshelf_enabled:
            tools.append(get_ncbi_bookshelf_search_tool(scientific_helper))
            tools.append(get_ncbi_bookshelf_download_tool(scientific_helper))
        if self.config.scientific_semantic_scholar_enabled:
            tools.append(get_semantic_scholar_search_tool(scientific_helper))
        if self.config.scientific_openalex_enabled:
            tools.append(get_openalex_search_tool(scientific_helper))
        if self.config.scientific_plos_enabled:
            tools.append(get_plos_search_tool(scientific_helper))
        if has_serpapi and self.config.scientific_google_patents_enabled:
            tools.append(get_google_patents_search_tool(scientific_helper))
        if has_serpapi and self.config.scientific_google_patents_details_enabled:
            tools.append(get_google_patents_details_tool(scientific_helper))
        if has_serpapi and self.config.scientific_google_scholar_enabled:
            tools.append(get_google_scholar_search_tool(scientific_helper))
        if has_serpapi and self.config.scientific_google_scholar_cite_enabled:
            tools.append(get_google_scholar_cite_tool(scientific_helper))
        if has_serpapi and self.config.scientific_youtube_search_enabled:
            tools.append(get_youtube_video_search_tool(scientific_helper))
        if has_serpapi and getattr(self.config, "scientific_youtube_details_enabled", False):
            tools.append(get_youtube_video_details_tool(scientific_helper))
        if has_serpapi and self.config.scientific_youtube_transcript_enabled:
            tools.append(get_youtube_transcript_tool(scientific_helper))
        if self.config.scientific_medrxiv_enabled:
            tools.append(get_medrxiv_preprint_search_tool(scientific_helper))
            tools.append(get_medrxiv_full_text_download_tool(scientific_helper))
        if self.config.scientific_crossref_enabled:
            tools.append(get_crossref_search_tool(open_helper))
            tools.append(get_crossref_doi_lookup_tool(open_helper))
        if self.config.scientific_clinicaltrials_enabled:
            tools.append(get_clinicaltrials_search_tool(open_helper))
            tools.append(get_clinicaltrial_get_tool(open_helper))
        if self.config.scientific_biorxiv_enabled:
            tools.append(get_biorxiv_search_tool(open_helper))
            tools.append(get_biorxiv_download_tool(open_helper))
        if self.config.scientific_retraction_watch_enabled:
            tools.append(get_retraction_watch_tool(open_helper))
        if self.config.scientific_pubchem_enabled:
            tools.append(get_pubchem_search_tool(open_helper))
        if self.config.scientific_pdf_text_enabled:
            pdf_helper = PdfTextTool(self.config)
            tools.append(get_pdf_text_tool(pdf_helper))
        if self.config.scientific_exec_enabled:
            exec_helper = ExecTool(self.config)
            tools.append(get_exec_tool(exec_helper))

        forum_helper = ForumScoutTool(self.config)
        if self.config.social_network_forum_search_enabled:
            tools.append(get_forum_search_tool(forum_helper))
        if self.config.social_network_linkedin_enabled:
            tools.append(get_linkedin_search_tool(forum_helper))
        if self.config.social_network_instagram_enabled:
            tools.append(get_instagram_search_tool(forum_helper))
        if self.config.social_network_reddit_posts_enabled:
            tools.append(get_reddit_posts_search_tool(forum_helper))
        if self.config.social_network_reddit_comments_enabled:
            tools.append(get_reddit_comments_search_tool(forum_helper))
        if self.config.social_network_x_enabled:
            tools.append(get_x_search_tool(forum_helper))
        if has_serpapi and self.config.social_network_google_forums_enabled:
            tools.append(get_google_forums_search_tool(forum_helper))
        if has_serpapi and self.config.social_network_google_news_enabled:
            tools.append(get_google_news_search_tool(forum_helper))
        if has_serpapi and self.config.social_network_google_trends_enabled:
            tools.append(get_google_trends_search_tool(forum_helper))
        if has_serpapi and self.config.social_network_google_trending_now_enabled:
            tools.append(get_google_trends_trending_now_tool(forum_helper))
        if has_serpapi and self.config.social_network_google_videos_enabled:
            tools.append(get_google_videos_search_tool(forum_helper))
        if has_serpapi and self.config.social_network_instagram_profile_enabled:
            tools.append(get_instagram_profile_tool(forum_helper))
        if has_serpapi and self.config.social_network_facebook_profile_enabled:
            tools.append(get_facebook_profile_tool(forum_helper))
        if self.config.social_network_tiktok_web_enabled:
            tools.append(get_tiktok_web_search_tool(forum_helper))
        if self.config.social_network_bluesky_web_enabled:
            tools.append(get_bluesky_web_search_tool(forum_helper))
        if self.config.social_network_mastodon_enabled:
            tools.append(get_mastodon_search_tool(open_helper))
        if has_serpapi and self.config.social_network_youtube_video_details_enabled:
            tools.append(get_youtube_video_search_tool(scientific_helper))
            tools.append(get_youtube_video_details_tool(scientific_helper))
            tools.append(get_youtube_transcript_tool(scientific_helper))

        business_helper = BusinessSearchTool(self.config)
        if self.config.business_sec_enabled:
            tools.append(get_sec_company_search_tool(business_helper))
            tools.append(get_sec_company_submissions_tool(business_helper))
            tools.append(get_sec_company_facts_tool(business_helper))
        if self.config.business_gleif_enabled:
            tools.append(get_gleif_lei_search_tool(business_helper))
            tools.append(get_gleif_lei_record_tool(business_helper))
        if self.config.business_cpsc_enabled:
            tools.append(get_cpsc_recalls_search_tool(open_helper))
        if self.config.business_federal_register_enabled:
            tools.append(get_federal_register_search_tool(open_helper))
        if self.config.business_wikidata_enabled:
            tools.append(get_wikidata_entity_search_tool(open_helper))
            tools.append(get_wikidata_sparql_tool(open_helper))
        if has_serpapi and self.config.business_google_finance_enabled:
            tools.append(get_google_finance_search_tool(business_helper))
        if has_serpapi and self.config.business_google_finance_markets_enabled:
            tools.append(get_google_finance_markets_tool(business_helper))
        if has_serpapi and self.config.business_google_maps_enabled:
            tools.append(get_google_maps_search_tool(business_helper))
        if has_serpapi and self.config.business_google_maps_reviews_enabled:
            tools.append(get_google_maps_reviews_tool(business_helper))
        if has_serpapi and self.config.business_yelp_enabled:
            tools.append(get_yelp_business_search_tool(business_helper))
            tools.append(get_yelp_place_tool(business_helper))
            tools.append(get_yelp_reviews_tool(business_helper))
        if has_serpapi and self.config.business_apple_maps_enabled:
            tools.append(get_apple_maps_search_tool(business_helper))
            tools.append(get_apple_maps_place_tool(business_helper))
        if has_serpapi and self.config.business_google_ads_enabled:
            tools.append(get_google_ads_search_tool(business_helper))
        if has_serpapi and self.config.business_google_ads_transparency_enabled:
            tools.append(get_google_ads_transparency_tool(business_helper))
        if has_serpapi and self.config.business_google_shopping_enabled:
            tools.append(get_google_shopping_search_tool(business_helper))
        if has_serpapi and self.config.business_google_shopping_light_enabled:
            tools.append(get_google_shopping_light_search_tool(business_helper))
        if has_serpapi and self.config.business_google_immersive_product_enabled:
            tools.append(get_google_immersive_product_tool(business_helper))
        if has_serpapi and self.config.business_amazon_enabled:
            tools.append(get_amazon_product_search_tool(business_helper))
            tools.append(get_amazon_product_tool(business_helper))
        if has_serpapi and self.config.business_walmart_enabled:
            tools.append(get_walmart_product_search_tool(business_helper))
            tools.append(get_walmart_product_tool(business_helper))
        if has_serpapi and self.config.business_ebay_enabled:
            tools.append(get_ebay_product_search_tool(business_helper))
            tools.append(get_ebay_product_tool(business_helper))
        if has_serpapi and self.config.business_home_depot_enabled:
            tools.append(get_home_depot_product_search_tool(business_helper))
            tools.append(get_home_depot_product_tool(business_helper))
        if has_serpapi and self.config.business_tripadvisor_enabled:
            tools.append(get_tripadvisor_search_tool(business_helper))
            tools.append(get_tripadvisor_place_tool(business_helper))
            tools.append(get_tripadvisor_reviews_tool(business_helper))
        if has_serpapi and self.config.business_google_web_enabled:
            tools.append(get_google_web_search_tool(SerpApiWebSearchTool(self.config)))
        if has_serpapi and self.config.business_bing_web_enabled:
            tools.append(get_bing_web_search_tool(SerpApiWebSearchTool(self.config)))
        if has_serpapi and self.config.business_google_news_enabled:
            tools.append(get_google_news_search_tool(forum_helper))
        if has_serpapi and self.config.business_google_trends_enabled:
            tools.append(get_google_trends_search_tool(forum_helper))
        if has_serpapi and self.config.business_google_patents_enabled:
            tools.append(get_google_patents_search_tool(scientific_helper))
        if has_serpapi and self.config.business_google_patents_details_enabled:
            tools.append(get_google_patents_details_tool(scientific_helper))

        product_helper = ProductSearchTool(self.config)
        if self.config.product_open_food_facts_enabled:
            tools.append(get_open_food_facts_search_tool(product_helper))
            tools.append(get_open_food_facts_product_tool(product_helper))
        if self.config.product_openfda_enabled:
            tools.append(get_openfda_recalls_search_tool(product_helper))
        if self.config.product_nvd_enabled:
            tools.append(get_nvd_cpe_search_tool(product_helper))
            tools.append(get_nvd_cve_search_tool(product_helper))
        if self.config.product_cpsc_enabled:
            tools.append(get_cpsc_recalls_search_tool(open_helper))
        if self.config.product_cisa_kev_enabled:
            tools.append(get_cisa_kev_search_tool(open_helper))
        if self.config.product_osv_enabled:
            tools.append(get_osv_package_query_tool(open_helper))
        if has_serpapi and self.config.product_google_lens_enabled:
            tools.append(get_google_lens_products_tool(product_helper))
        if has_serpapi and self.config.product_google_shopping_enabled:
            tools.append(get_google_shopping_search_tool(business_helper))
        if has_serpapi and self.config.product_google_shopping_light_enabled:
            tools.append(get_google_shopping_light_search_tool(business_helper))
        if has_serpapi and self.config.product_google_immersive_product_enabled:
            tools.append(get_google_immersive_product_tool(business_helper))
        if has_serpapi and self.config.product_amazon_enabled:
            tools.append(get_amazon_product_search_tool(business_helper))
            tools.append(get_amazon_product_tool(business_helper))
        if has_serpapi and self.config.product_walmart_enabled:
            tools.append(get_walmart_product_search_tool(business_helper))
            tools.append(get_walmart_product_tool(business_helper))
        if has_serpapi and self.config.product_ebay_enabled:
            tools.append(get_ebay_product_search_tool(business_helper))
            tools.append(get_ebay_product_tool(business_helper))
        if has_serpapi and self.config.product_home_depot_enabled:
            tools.append(get_home_depot_product_search_tool(business_helper))
            tools.append(get_home_depot_product_tool(business_helper))
        if has_serpapi and self.config.product_google_trends_enabled:
            tools.append(get_google_trends_search_tool(forum_helper))
        if has_serpapi and self.config.product_google_patents_enabled:
            tools.append(get_google_patents_search_tool(scientific_helper))
            tools.append(get_google_patents_details_tool(scientific_helper))
        if has_serpapi and self.config.product_youtube_enabled:
            tools.append(get_youtube_video_search_tool(scientific_helper))
            tools.append(get_youtube_video_details_tool(scientific_helper))
            tools.append(get_youtube_transcript_tool(scientific_helper))

        if self.config.websearcher_enabled or self.config.webresearcher_enabled:
            websearcher_helper = WebSearcherAgentTool(
                self.config,
                model_name=self.websearcher_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.websearcher_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_websearcher_research_tool(websearcher_helper))

        if self.config.cli_enabled:
            cli_helper = CliResearchAgentTool(
                self.config,
                model_name=self.cli_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.cli_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_cli_research_tool(cli_helper))

        if self.config.subchack_enabled:
            subchack_helper = SubChackResearchAgentTool(
                self.config,
                model_name=self.subchack_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.subchack_max_turns,
                social_network_model=self.social_network_model,
                scientific_model=self.scientific_model,
                websearcher_model=self.websearcher_model,
                business_model=self.business_model,
                product_model=self.product_model,
                legal_model=self.legal_model,
                data_statistics_model=self.data_statistics_model,
                news_media_model=self.news_media_model,
                knowledge_graph_model=self.knowledge_graph_model,
                religious_model=self.religious_model,
                cli_model=self.cli_model,
                social_network_max_turns=self.social_network_max_turns,
                scientific_max_turns=self.scientific_max_turns,
                websearcher_max_turns=self.websearcher_max_turns,
                business_max_turns=self.business_max_turns,
                product_max_turns=self.product_max_turns,
                legal_max_turns=self.legal_max_turns,
                data_statistics_max_turns=self.data_statistics_max_turns,
                news_media_max_turns=self.news_media_max_turns,
                knowledge_graph_max_turns=self.knowledge_graph_max_turns,
                religious_max_turns=self.religious_max_turns,
                cli_max_turns=self.cli_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_subchack_research_tool(subchack_helper))

        if getattr(self.config, "researcher_administrator_enabled", False):
            admin_agent_cfg = dict(getattr(self.config, "researcher_administrator_agent", {}) or {})
            # The admin model and its launched-researcher models can be managed
            # from the tools config (`researcher_administrator_agent`), which
            # propagates to every backend. They fall back to `model.*` keys.
            admin_model = (
                self._resolve_alias(str(admin_agent_cfg.get("model") or ""), fallback="")
                or self.researcher_administrator_model
            )
            admin_max_turns = int(admin_agent_cfg.get("max_turns") or 0) or self.researcher_administrator_max_turns
            administrator_helper = ResearcherAdministratorAgentTool(
                self.config,
                model_name=admin_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=admin_max_turns,
                researchers=list(
                    getattr(self.config, "researcher_administrator_researchers", []) or []
                ),
                researcher_model_overrides=admin_agent_cfg.get("researcher_models"),
                researcher_max_turns_overrides=admin_agent_cfg.get("researcher_max_turns"),
                social_network_model=self.social_network_model,
                scientific_model=self.scientific_model,
                websearcher_model=self.websearcher_model,
                business_model=self.business_model,
                product_model=self.product_model,
                legal_model=self.legal_model,
                data_statistics_model=self.data_statistics_model,
                news_media_model=self.news_media_model,
                knowledge_graph_model=self.knowledge_graph_model,
                religious_model=self.religious_model,
                cli_model=self.cli_model,
                social_network_max_turns=self.social_network_max_turns,
                scientific_max_turns=self.scientific_max_turns,
                websearcher_max_turns=self.websearcher_max_turns,
                business_max_turns=self.business_max_turns,
                product_max_turns=self.product_max_turns,
                legal_max_turns=self.legal_max_turns,
                data_statistics_max_turns=self.data_statistics_max_turns,
                news_media_max_turns=self.news_media_max_turns,
                knowledge_graph_max_turns=self.knowledge_graph_max_turns,
                religious_max_turns=self.religious_max_turns,
                cli_max_turns=self.cli_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_researcher_administrator_tool(administrator_helper))

        if getattr(self.config, "researcher_queue_enabled", False):
            from .researcher_queue_agent import (
                ResearcherQueueAgentTool,
                get_researcher_queue_create_tool,
                get_researcher_queue_tool,
                get_researcher_queue_status_tool,
            )

            queue_agent_cfg = dict(getattr(self.config, "researcher_queue_agent", {}) or {})
            queue_researchers = list(getattr(self.config, "researcher_queue_researchers", []) or [])
            queue_admin_model = (
                self._resolve_alias(str(queue_agent_cfg.get("model") or ""), fallback="")
                or self.researcher_administrator_model
            )
            queue_admin_max_turns = (
                int(queue_agent_cfg.get("max_turns") or 0) or self.researcher_administrator_max_turns
            )
            queue_admin_budget = (
                int(getattr(self.config, "researcher_queue_max_tools_used", 0) or 0)
                or self.config.researcher_administrator_max_tools_used
            )
            # The queue owns a private administrator (force-enabled) that researches
            # each merged request. Force it off recursion just like the standalone one.
            queue_admin_config = replace(
                self.config,
                researcher_administrator_enabled=True,
                researcher_administrator_researchers=queue_researchers,
                researcher_administrator_max_tools_used=queue_admin_budget,
                researcher_administrator_agent={
                    **dict(
                        getattr(self.config, "researcher_administrator_agent", {}) or {}
                    ),
                    **queue_agent_cfg,
                },
            )
            queue_admin = ResearcherAdministratorAgentTool(
                queue_admin_config,
                model_name=queue_admin_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=queue_admin_max_turns,
                researchers=queue_researchers,
                researcher_model_overrides=queue_agent_cfg.get("researcher_models"),
                researcher_max_turns_overrides=queue_agent_cfg.get("researcher_max_turns"),
                social_network_model=self.social_network_model,
                scientific_model=self.scientific_model,
                websearcher_model=self.websearcher_model,
                business_model=self.business_model,
                product_model=self.product_model,
                legal_model=self.legal_model,
                data_statistics_model=self.data_statistics_model,
                news_media_model=self.news_media_model,
                knowledge_graph_model=self.knowledge_graph_model,
                religious_model=self.religious_model,
                cli_model=self.cli_model,
                social_network_max_turns=self.social_network_max_turns,
                scientific_max_turns=self.scientific_max_turns,
                websearcher_max_turns=self.websearcher_max_turns,
                business_max_turns=self.business_max_turns,
                product_max_turns=self.product_max_turns,
                legal_max_turns=self.legal_max_turns,
                data_statistics_max_turns=self.data_statistics_max_turns,
                news_media_max_turns=self.news_media_max_turns,
                knowledge_graph_max_turns=self.knowledge_graph_max_turns,
                religious_max_turns=self.religious_max_turns,
                cli_max_turns=self.cli_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            queue_merge_model = (
                self._resolve_alias(
                    str(
                        queue_agent_cfg.get("merge_model")
                        or getattr(self.config, "researcher_queue_merge_model", "")
                        or ""
                    ),
                    fallback="",
                )
                or self.default_model
            )
            queue_helper = ResearcherQueueAgentTool(
                queue_admin,
                config=self.config,
                model_provider=self.model_provider,
                merge_model=queue_merge_model,
                fallback_model=self.default_model,
                window_seconds=int(getattr(self.config, "researcher_queue_window_seconds", 300) or 0),
                expected_participants=int(
                    getattr(self.config, "researcher_queue_expected_participants", 0) or 0
                ),
                max_requests_per_call=int(
                    getattr(self.config, "researcher_queue_max_requests_per_call", 5) or 5
                ),
                max_batch_requests=int(
                    getattr(self.config, "researcher_queue_max_batch_requests", 20) or 0
                ),
                max_wait_seconds=int(
                    getattr(self.config, "researcher_queue_max_wait_seconds", 5400) or 5400
                ),
                queue_id=str(getattr(self.config, "researcher_queue_id", "") or ""),
            )
            tools.append(get_researcher_queue_create_tool(queue_helper))
            tools.append(get_researcher_queue_tool(queue_helper))
            tools.append(get_researcher_queue_status_tool(queue_helper))

        if self.config.social_network_enabled:
            social_helper = SocialNetworkAgentTool(
                self.config,
                model_name=self.social_network_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.social_network_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_social_network_research_tool(social_helper))

        if self.config.scientific_enabled:
            scientific_helper = ScientificResearchAgentTool(
                self.config,
                model_name=self.scientific_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.scientific_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_scientific_research_tool(scientific_helper))

        if self.config.business_enabled:
            business_helper = BusinessResearchAgentTool(
                self.config,
                model_name=self.business_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.business_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_business_research_tool(business_helper))

        if self.config.product_enabled:
            product_helper = ProductResearchAgentTool(
                self.config,
                model_name=self.product_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.product_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_product_research_tool(product_helper))

        if self.config.legal_enabled:
            legal_helper = build_legal_agent(
                self.config,
                model_name=self.legal_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.legal_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_legal_research_tool(legal_helper))

        if self.config.data_statistics_enabled:
            data_helper = build_data_statistics_agent(
                self.config,
                model_name=self.data_statistics_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.data_statistics_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_data_statistics_research_tool(data_helper))

        if self.config.news_media_enabled:
            news_helper = build_news_media_agent(
                self.config,
                model_name=self.news_media_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.news_media_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_news_media_research_tool(news_helper))

        if self.config.knowledge_graph_enabled:
            kg_helper = build_knowledge_graph_agent(
                self.config,
                model_name=self.knowledge_graph_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.knowledge_graph_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_knowledge_graph_research_tool(kg_helper))

        if self.config.religious_enabled:
            religious_helper = build_religious_agent(
                self.config,
                model_name=self.religious_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.religious_max_turns,
                self_critique_enabled=self.self_critique_enabled,
                self_critique_rounds=self.self_critique_rounds,
            )
            tools.append(get_religious_research_tool(religious_helper))

        if self.config.pdf_text_enabled:
            pdf_helper = PdfTextTool(self.config)
            tools.append(get_pdf_text_tool(pdf_helper))

        return self._dedupe_tools(tools)
