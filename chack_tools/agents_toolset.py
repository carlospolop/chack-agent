import os

from .config import ToolsConfig
from .brave_search import BraveSearchTool, get_brave_search_tool
from .exec_tool import ExecTool, get_exec_tool
from .pdf_text import PdfTextTool, get_pdf_text_tool
from .scientific_research_agent import ScientificResearchAgentTool, get_scientific_research_tool
from .serpapi_web_search import (
    SerpApiWebSearchTool,
    get_google_web_search_tool,
    get_bing_web_search_tool
)
from .social_network_agent import SocialNetworkAgentTool, get_social_network_research_tool
from .task_steps_manager_tool import TaskStepsManagerTool, get_task_steps_manager_tool
from .websearcher_agent import WebSearcherAgentTool, get_websearcher_research_tool
from .tester_agent import TesterAgentTool, get_tester_agent_tool
from .serpapi_keys import has_serpapi_keys


class AgentsToolset:
    def __init__(
        self,
        config: ToolsConfig,
        tool_profile: str = "all",
        model_provider: str = "",
        default_model: str = "",
        social_network_model: str = "",
        scientific_model: str = "",
        websearcher_model: str = "",
        tester_model: str = "",
        social_network_max_turns: int = 30,
        scientific_max_turns: int = 30,
        websearcher_max_turns: int = 30,
        tester_max_turns: int = 30,
    ):
        self.config = config
        self.tool_profile = tool_profile
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.default_model = default_model
        self.social_network_model = social_network_model
        self.scientific_model = scientific_model
        self.websearcher_model = websearcher_model
        self.tester_model = tester_model
        self.social_network_max_turns = social_network_max_turns
        self.scientific_max_turns = scientific_max_turns
        self.websearcher_max_turns = websearcher_max_turns
        self.tester_max_turns = tester_max_turns
        self.tools = self._build_tools()

    def _build_tools(self):
        tools = []
        if self.config.exec_enabled:
            exec_helper = ExecTool(self.config)
            tools.append(get_exec_tool(exec_helper))

        task_helper = TaskStepsManagerTool(self.config)
        tools.append(get_task_steps_manager_tool(task_helper))

        if self.config.brave_enabled:
            brave_helper = BraveSearchTool(self.config)
            tools.append(get_brave_search_tool(brave_helper))

        has_serpapi = has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if has_serpapi and self.config.serpapi_google_web_enabled:
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(get_google_web_search_tool(web_helper))

        include_bing_web = self.tool_profile in {"all", "telegram"}
        if has_serpapi and self.config.serpapi_bing_web_enabled and include_bing_web:
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(get_bing_web_search_tool(web_helper))

        if self.config.websearcher_enabled:
            websearcher_helper = WebSearcherAgentTool(
                self.config,
                model_name=self.websearcher_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.websearcher_max_turns,
            )
            tools.append(get_websearcher_research_tool(websearcher_helper))

        if self.config.tester_enabled:
            tester_helper = TesterAgentTool(
                self.config,
                model_name=self.tester_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.tester_max_turns,
            )
            tools.append(get_tester_agent_tool(tester_helper))

        include_forumscout = self.tool_profile in {"all", "telegram"}
        if self.config.social_network_enabled and include_forumscout:
            social_helper = SocialNetworkAgentTool(
                self.config,
                model_name=self.social_network_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.social_network_max_turns,
            )
            tools.append(get_social_network_research_tool(social_helper))

        include_scientific = self.tool_profile in {"all", "telegram"}
        if self.config.scientific_enabled and include_scientific:
            scientific_helper = ScientificResearchAgentTool(
                self.config,
                model_name=self.scientific_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.scientific_max_turns,
            )
            tools.append(get_scientific_research_tool(scientific_helper))

        include_pdf = self.tool_profile in {"all", "telegram"}
        if self.config.pdf_text_enabled and include_pdf:
            pdf_helper = PdfTextTool(self.config)
            tools.append(get_pdf_text_tool(pdf_helper))

        return tools
