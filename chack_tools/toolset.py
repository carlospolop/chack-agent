from .config import ToolsConfig
from .brave_search import build_brave_search_tool
from .exec_tool import exec_tool
from .pdf_text import build_pdf_text_tool
from .serpapi_web_search import build_bing_web_search_tool, build_google_web_search_tool
from .scientific_research_agent import build_scientific_research_tool
from .social_network_agent import build_social_network_research_tool
from .task_list_tool import build_task_list_tool
from .websearcher_agent import build_websearcher_research_tool
from .serpapi_keys import has_serpapi_keys


class Toolset:
    def __init__(
        self,
        config: ToolsConfig,
        tool_profile: str = "all",
        social_network_model: str = "",
        scientific_model: str = "",
        websearcher_model: str = "",
        social_network_max_turns: int = 30,
        scientific_max_turns: int = 30,
        websearcher_max_turns: int = 30,
    ):
        self.config = config
        self.tool_profile = tool_profile
        self.social_network_model = social_network_model
        self.scientific_model = scientific_model
        self.websearcher_model = websearcher_model
        self.social_network_max_turns = social_network_max_turns
        self.scientific_max_turns = scientific_max_turns
        self.websearcher_max_turns = websearcher_max_turns
        self.tools = self._build_tools()

    def _build_tools(self):
        tools = []
        if self.config.exec_enabled:
            tools.append(exec_tool)
        tools.append(build_task_list_tool(self.config))
        if self.config.brave_enabled:
            tools.append(build_brave_search_tool(self.config))
        has_serpapi = has_serpapi_keys(getattr(self.config, "serpapi_api_key", ""))
        if has_serpapi and self.config.serpapi_google_web_enabled:
            tools.append(build_google_web_search_tool(self.config))
        include_bing_web = self.tool_profile in {"all", "telegram"}
        if has_serpapi and self.config.serpapi_bing_web_enabled and include_bing_web:
            tools.append(build_bing_web_search_tool(self.config))
        if self.config.websearcher_enabled and (self.config.brave_enabled or has_serpapi):
            tools.append(
                build_websearcher_research_tool(
                    self.config,
                    model_name=self.websearcher_model,
                    max_turns=self.websearcher_max_turns,
                )
            )
        include_forumscout = self.tool_profile in {"all", "telegram"}
        if self.config.forumscout_enabled and include_forumscout:
            tools.append(
                build_social_network_research_tool(
                    self.config,
                    model_name=self.social_network_model,
                    max_turns=self.social_network_max_turns,
                )
            )
        include_scientific = self.tool_profile in {"all", "telegram"}
        if self.config.scientific_enabled and include_scientific:
            tools.append(
                build_scientific_research_tool(
                    self.config,
                    model_name=self.scientific_model,
                    max_turns=self.scientific_max_turns,
                )
            )
        include_pdf = self.tool_profile in {"all", "telegram"}
        if self.config.pdf_text_enabled and include_pdf:
            tools.append(build_pdf_text_tool(self.config))
        return tools
