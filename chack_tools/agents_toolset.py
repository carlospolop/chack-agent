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
        model_provider: str = "",
        default_model: str = "",
        social_network_model: str = "CHEAP_BUT_QUALITY",
        scientific_model: str = "CHEAP_BUT_QUALITY",
        websearcher_model: str = "CHEAP_BUT_QUALITY",
        tester_model: str = "CHEAP_BUT_QUALITY",
        social_network_max_turns: int = 30,
        scientific_max_turns: int = 30,
        websearcher_max_turns: int = 30,
        tester_max_turns: int = 30,
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
        self.tester_model = self._resolve_alias(
            tester_model,
            fallback="CHEAP_BUT_QUALITY",
        )
        self.social_network_max_turns = social_network_max_turns
        self.scientific_max_turns = scientific_max_turns
        self.websearcher_max_turns = websearcher_max_turns
        self.tester_max_turns = tester_max_turns
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

        if has_serpapi and self.config.serpapi_bing_web_enabled:
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

        if self.config.social_network_enabled:
            social_helper = SocialNetworkAgentTool(
                self.config,
                model_name=self.social_network_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.social_network_max_turns,
            )
            tools.append(get_social_network_research_tool(social_helper))

        if self.config.scientific_enabled:
            scientific_helper = ScientificResearchAgentTool(
                self.config,
                model_name=self.scientific_model,
                fallback_model=self.default_model,
                model_provider=self.model_provider,
                max_turns=self.scientific_max_turns,
            )
            tools.append(get_scientific_research_tool(scientific_helper))

        if self.config.pdf_text_enabled:
            pdf_helper = PdfTextTool(self.config)
            tools.append(get_pdf_text_tool(pdf_helper))

        return tools
