from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolsConfig:
    task_steps_manager_enabled: bool = True

    exec_enabled: bool = False
    exec_cwd: str = ""
    exec_timeout_seconds: int = 60
    exec_max_output_chars: int = 5000

    brave_enabled: bool = False
    brave_api_key: str = ""
    brave_max_results: int = 10

    playwright_enabled: bool = False
    playwright_timeout_seconds: int = 30
    playwright_max_output_chars: int = 12000
    playwright_headless: bool = True

    social_network_enabled: bool = False
    forumscout_api_key: str = ""
    forumscout_max_results: int = 10

    serpapi_api_key: Any = ""
    serpapi_google_web_enabled: bool = False
    serpapi_bing_web_enabled: bool = False
    serpapi_bing_copilot_enabled: bool = False
    serpapi_web_max_results: int = 6
    open_research_max_results: int = 10
    open_research_fetch_url_text_enabled: bool = False
    open_research_web_archive_enabled: bool = False
    open_research_gdelt_enabled: bool = False
    open_research_federal_register_enabled: bool = False
    open_research_world_bank_enabled: bool = False
    open_research_wikidata_enabled: bool = False

    scientific_enabled: bool = False
    scientific_max_results: int = 10
    scientific_arxiv_enabled: bool = False
    scientific_europe_pmc_enabled: bool = False
    scientific_pmc_full_text_enabled: bool = False
    scientific_ncbi_bookshelf_enabled: bool = False
    scientific_semantic_scholar_enabled: bool = False
    scientific_openalex_enabled: bool = False
    scientific_plos_enabled: bool = False
    scientific_google_patents_enabled: bool = False
    scientific_google_patents_details_enabled: bool = False
    scientific_google_scholar_enabled: bool = False
    scientific_google_scholar_cite_enabled: bool = False
    scientific_youtube_search_enabled: bool = False
    scientific_youtube_details_enabled: bool = False
    scientific_youtube_transcript_enabled: bool = False
    scientific_medrxiv_enabled: bool = False
    scientific_crossref_enabled: bool = False
    scientific_clinicaltrials_enabled: bool = False
    scientific_biorxiv_enabled: bool = False
    scientific_retraction_watch_enabled: bool = False
    scientific_pubchem_enabled: bool = False
    scientific_pdf_text_enabled: bool = False
    scientific_exec_enabled: bool = False

    pdf_text_enabled: bool = False
    # When true, preserved research artifacts trigger an extra follow-up asking
    # the researcher to account for every saved evidence file in key_artifacts or
    # delete it. Keep true by default so preserved research folders are auditable.
    research_strict_artifact_manifest: bool = True

    websearcher_enabled: bool = False
    webresearcher_enabled: bool = False
    websearcher_brave_enabled: bool = False
    websearcher_google_web_enabled: bool = False
    websearcher_bing_web_enabled: bool = False
    websearcher_google_ai_mode_enabled: bool = False
    websearcher_bing_copilot_enabled: bool = False
    websearcher_web_archive_enabled: bool = False
    websearcher_gdelt_enabled: bool = False
    websearcher_fetch_url_text_enabled: bool = False

    business_enabled: bool = False
    business_max_results: int = 10
    business_sec_enabled: bool = False
    business_gleif_enabled: bool = False
    business_google_finance_enabled: bool = False
    business_google_finance_markets_enabled: bool = False
    business_google_web_enabled: bool = False
    business_bing_web_enabled: bool = False
    business_google_news_enabled: bool = False
    business_google_trends_enabled: bool = False
    business_google_patents_enabled: bool = False
    business_google_patents_details_enabled: bool = False
    business_google_maps_enabled: bool = False
    business_google_maps_reviews_enabled: bool = False
    business_yelp_enabled: bool = False
    business_apple_maps_enabled: bool = False
    business_google_ads_enabled: bool = False
    business_google_ads_transparency_enabled: bool = False
    business_google_shopping_enabled: bool = False
    business_google_shopping_light_enabled: bool = False
    business_google_immersive_product_enabled: bool = False
    business_amazon_enabled: bool = False
    business_walmart_enabled: bool = False
    business_ebay_enabled: bool = False
    business_home_depot_enabled: bool = False
    business_tripadvisor_enabled: bool = False
    business_cpsc_enabled: bool = False
    business_federal_register_enabled: bool = False
    business_wikidata_enabled: bool = False
    business_playwright_enabled: bool = False

    travel_enabled: bool = False
    travel_max_results: int = 10
    travel_google_flights_enabled: bool = False
    travel_google_travel_explore_enabled: bool = False
    travel_google_hotels_enabled: bool = False
    travel_google_hotels_reviews_enabled: bool = False
    travel_booking_enabled: bool = False
    travel_amadeus_enabled: bool = False
    travel_opentripmap_enabled: bool = False
    travel_open_meteo_enabled: bool = False
    travel_open_meteo_air_quality_enabled: bool = False
    travel_open_meteo_marine_enabled: bool = False
    travel_public_holidays_enabled: bool = False
    travel_ticketmaster_enabled: bool = False
    travel_frankfurter_enabled: bool = False
    travel_wikivoyage_enabled: bool = False
    travel_transitous_enabled: bool = False
    travel_agent: dict = field(default_factory=dict)

    product_enabled: bool = False
    product_max_results: int = 10
    product_serpapi_enabled: bool = False
    product_google_lens_enabled: bool = False
    product_open_food_facts_enabled: bool = False
    product_openfda_enabled: bool = False
    product_nvd_enabled: bool = False
    product_cpsc_enabled: bool = False
    product_cisa_kev_enabled: bool = False
    product_osv_enabled: bool = False
    product_google_shopping_enabled: bool = False
    product_google_shopping_light_enabled: bool = False
    product_google_immersive_product_enabled: bool = False
    product_amazon_enabled: bool = False
    product_walmart_enabled: bool = False
    product_ebay_enabled: bool = False
    product_home_depot_enabled: bool = False
    product_google_trends_enabled: bool = False
    product_google_patents_enabled: bool = False
    product_youtube_enabled: bool = False
    product_playwright_enabled: bool = False

    cli_enabled: bool = False
    cli_exec_enabled: bool = False
    cli_brave_enabled: bool = False
    cli_google_web_enabled: bool = False
    cli_agent: dict = field(default_factory=dict)

    # Authenticated ChatGPT Web tools. All attach to the same user-managed
    # Chrome CDP profile; each request runs in its own clean tab.
    deepchatgpt_enabled: bool = False
    prochatgpt_enabled: bool = False
    chatgptxhigh_enabled: bool = False
    chatgpt_cdp_url: str = ""
    # Total browser-output deadlines. Mode-specific values take precedence over
    # the deprecated shared timeout below. Zero/None means use the built-in
    # defaults: 90 minutes for Pro, 30 minutes for Extra High, and 75 minutes
    # for Deep Research.
    chatgpt_pro_timeout_seconds: Optional[int] = None
    chatgpt_xhigh_timeout_seconds: Optional[int] = None
    chatgpt_deep_timeout_seconds: Optional[int] = None
    chatgpt_research_timeout_seconds: int = 0  # deprecated shared fallback
    chatgpt_research_poll_seconds: int = 15
    # Execution backend: "auto" uses the async HTTPS broker when URL + secret
    # are configured and otherwise preserves the direct local-browser behavior.
    # The outbound workstation worker always overrides this to "local" so it
    # cannot recursively submit the job it has just leased.
    chatgpt_execution_backend: str = "auto"
    chatgpt_async_api_url: str = ""
    chatgpt_async_api_secret: str = ""
    chatgpt_async_poll_seconds: int = 10
    # XHigh's 1800-second output deadline plus the 300-second terminal/grace
    # window. Mode-specific code still caps stale overrides at timeout+grace.
    chatgpt_async_max_wait_seconds: int = 2100
    chatgpt_async_request_timeout_seconds: int = 30
    # Pro requests click "Answer now" this many seconds before their total
    # output deadline; this is part of, not added after, the Pro timeout.
    chatgpt_force_answer_grace_seconds: int = 300

    subchack_enabled: bool = False
    subchack_agent: dict = field(default_factory=dict)

    # Root-level dispatcher that lets the current agent select and launch up to
    # four of its enabled researcher tools concurrently. Every dispatched
    # researcher prompt is hard-validated to contain at least 500 characters.
    parallel_research_enabled: bool = False
    parallel_research_max_requests: int = 4

    researcher_administrator_enabled: bool = False
    # Researcher short-names the administrator may launch (e.g. ["scientific",
    # "business", "websearcher"]). Empty means "every researcher enabled above".
    researcher_administrator_researchers: list = field(default_factory=list)
    # Optional subset that must each return one successful result. This is useful
    # when the caller treats researcher selection as a requirement rather than an
    # allowlist. Empty preserves the administrator's relevance-based selection.
    researcher_administrator_required_researchers: list = field(default_factory=list)
    researcher_administrator_agent: dict = field(default_factory=dict)

    # Shared research queue. Several agents (threads in one process, or external
    # agents connected to one long-running MCP server) submit research requests;
    # requests are collected for a short window, near-duplicate ones are merged,
    # each merged request is researched once by a research administrator, and
    # each caller receives only the research results covering its own request(s).
    researcher_queue_enabled: bool = False
    # Stable logical queue id. All researcher_queue calls sharing this id batch
    # together and write to one shared evidence folder; distinct ids are fully
    # isolated (separate batches + folders). Set it per job so concurrent
    # jobs — even in one process — never mix their research results. Empty falls
    # back to the CHACK_RESEARCHER_QUEUE_ID env var, then the process-default queue.
    researcher_queue_id: str = ""
    # Seconds to wait for more queue callers before launching a batch. During this
    # window similar requests can be merged; set 0 to launch immediately.
    researcher_queue_window_seconds: int = 300
    researcher_queue_expected_participants: int = 0     # >0 flushes early once N callers have joined
    researcher_queue_max_requests_per_call: int = 5     # research prompts allowed per single call
    researcher_queue_max_batch_requests: int = 20       # hard ceiling of prompts per batch (0 = no ceiling)
    researcher_queue_max_parallel_researches: int = 0   # concurrent admin researches per batch (0 = one per merged request, all at once)
    researcher_queue_max_wait_seconds: int = 5400       # safety cap on the blocking call (90 min)
    researcher_queue_max_runtime_minutes: int = 60      # fixed runtime cap for each queued admin research (0 = no cap)
    researcher_queue_max_cost_usd: float = 5.0          # fixed cost cap for each queued admin research (0 = no cap)
    researcher_queue_merge_model: str = ""              # empty -> default model
    # Researcher short-names the queue's administrator may launch. Empty means
    # "every researcher enabled above" (same semantics as the administrator).
    researcher_queue_researchers: list = field(default_factory=list)
    # Optional subset of queue researchers that must each complete successfully
    # in every merged administrator request.
    researcher_queue_required_researchers: list = field(default_factory=list)
    researcher_queue_agent: dict = field(default_factory=dict)
    researcher_queue_max_tools_used: int = 0

    social_network_forum_search_enabled: bool = False
    social_network_linkedin_enabled: bool = False
    social_network_instagram_enabled: bool = False
    social_network_reddit_posts_enabled: bool = False
    social_network_reddit_comments_enabled: bool = False
    social_network_x_enabled: bool = False
    social_network_google_forums_enabled: bool = False
    social_network_google_news_enabled: bool = False
    social_network_google_trends_enabled: bool = False
    social_network_google_trending_now_enabled: bool = False
    social_network_google_videos_enabled: bool = False
    social_network_instagram_profile_enabled: bool = False
    social_network_facebook_profile_enabled: bool = False
    social_network_youtube_video_details_enabled: bool = False
    social_network_mastodon_enabled: bool = False
    social_network_tiktok_web_enabled: bool = False
    social_network_bluesky_web_enabled: bool = False

    social_network_agent: dict = field(default_factory=dict)
    scientific_agent: dict = field(default_factory=dict)
    websearcher_agent: dict = field(default_factory=dict)
    business_agent: dict = field(default_factory=dict)
    product_agent: dict = field(default_factory=dict)
    legal_enabled: bool = False
    legal_boe_enabled: bool = False
    data_statistics_enabled: bool = False
    news_media_enabled: bool = False
    knowledge_graph_enabled: bool = False
    religious_enabled: bool = False
    legal_federal_register_enabled: bool = False
    legal_wikidata_enabled: bool = False
    data_statistics_world_bank_enabled: bool = False
    data_statistics_wikidata_enabled: bool = False
    knowledge_graph_wikidata_enabled: bool = False
    religious_bible_enabled: bool = False
    religious_sefaria_enabled: bool = False
    religious_quran_enabled: bool = False
    religious_gita_enabled: bool = False
    religious_hadith_enabled: bool = False
    religious_suttacentral_enabled: bool = False
    religious_wikidata_enabled: bool = False
    legal_agent: dict = field(default_factory=dict)
    data_statistics_agent: dict = field(default_factory=dict)
    news_media_agent: dict = field(default_factory=dict)
    knowledge_graph_agent: dict = field(default_factory=dict)
    religious_agent: dict = field(default_factory=dict)

    deny_builtin_tools: list = field(default_factory=list)

    min_tools_used: int = 10
    max_tools_used: int = 0

    social_network_max_tools_used: int = 0
    scientific_max_tools_used: int = 0
    websearcher_max_tools_used: int = 0
    business_max_tools_used: int = 0
    travel_max_tools_used: int = 0
    product_max_tools_used: int = 0
    legal_max_tools_used: int = 0
    data_statistics_max_tools_used: int = 0
    news_media_max_tools_used: int = 0
    knowledge_graph_max_tools_used: int = 0
    religious_max_tools_used: int = 0
    cli_max_tools_used: int = 0
    subchack_max_tools_used: int = 0
    researcher_administrator_max_tools_used: int = 0
