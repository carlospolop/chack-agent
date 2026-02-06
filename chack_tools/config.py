from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolsConfig:
    exec_enabled: bool = False
    exec_timeout_seconds: int = 60
    exec_max_output_chars: int = 5000

    brave_enabled: bool = False
    brave_api_key: str = ""
    brave_max_results: int = 10

    social_network_enabled: bool = False
    forumscout_api_key: str = ""
    forumscout_max_results: int = 10

    serpapi_api_key: Any = ""
    serpapi_google_web_enabled: bool = False
    serpapi_bing_web_enabled: bool = False
    serpapi_web_max_results: int = 6

    scientific_enabled: bool = False
    scientific_max_results: int = 10
    scientific_arxiv_enabled: bool = False
    scientific_europe_pmc_enabled: bool = False
    scientific_semantic_scholar_enabled: bool = False
    scientific_openalex_enabled: bool = False
    scientific_plos_enabled: bool = False
    scientific_google_patents_enabled: bool = False
    scientific_google_scholar_enabled: bool = False
    scientific_youtube_search_enabled: bool = False
    scientific_youtube_transcript_enabled: bool = False
    scientific_pdf_text_enabled: bool = False
    scientific_exec_enabled: bool = False

    pdf_text_enabled: bool = False

    websearcher_enabled: bool = False
    websearcher_brave_enabled: bool = False
    websearcher_google_web_enabled: bool = False
    websearcher_bing_web_enabled: bool = False
    websearcher_google_ai_mode_enabled: bool = False

    tester_enabled: bool = False
    tester_exec_enabled: bool = False
    tester_brave_enabled: bool = False
    tester_google_web_enabled: bool = False
    tester_agent: dict = field(default_factory=dict)

    social_network_forum_search_enabled: bool = False
    social_network_linkedin_enabled: bool = False
    social_network_instagram_enabled: bool = False
    social_network_reddit_posts_enabled: bool = False
    social_network_reddit_comments_enabled: bool = False
    social_network_x_enabled: bool = False
    social_network_google_forums_enabled: bool = False
    social_network_google_news_enabled: bool = False

    social_network_agent: dict = field(default_factory=dict)
    scientific_agent: dict = field(default_factory=dict)
    websearcher_agent: dict = field(default_factory=dict)

    min_tools_used: int = 10
    max_tools_used: int = 0

    social_network_max_tools_used: int = 0
    scientific_max_tools_used: int = 0
    websearcher_max_tools_used: int = 0
    tester_max_tools_used: int = 0
