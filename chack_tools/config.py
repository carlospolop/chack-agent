from dataclasses import dataclass
from typing import Any


@dataclass
class ToolsConfig:
    exec_enabled: bool = True
    exec_timeout_seconds: int = 120
    exec_max_output_chars: int = 5000

    duckduckgo_enabled: bool = True
    duckduckgo_max_results: int = 6

    brave_enabled: bool = True
    brave_api_key: str = ""
    brave_max_results: int = 6

    forumscout_enabled: bool = True
    forumscout_api_key: str = ""
    forumscout_base_url: str = "https://forumscout.app"
    forumscout_max_results: int = 6

    serpapi_api_key: Any = ""
    serpapi_google_web_enabled: bool = True
    serpapi_bing_web_enabled: bool = True
    serpapi_web_max_results: int = 6

    scientific_enabled: bool = True
    scientific_max_results: int = 10

    pdf_text_enabled: bool = True

    websearcher_enabled: bool = True

    min_tools_used: int = 10
