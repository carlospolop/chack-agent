from langchain_core.tools import StructuredTool
from typing import Optional

from .brave_search import BraveSearchTool
from .config import ToolsConfig
from .serpapi_web_search import SerpApiWebSearchTool
from .subagent_runner import SubAgentRunner

try:
    from agents import function_tool
except ImportError:  # pragma: no cover
    function_tool = None


_WEBSEARCHER_AGENT_SYSTEM_PROMPT = """### ROLE
You are an autonomous Web Research Sub-Agent focused on extensive, evidence-based web research.

### OBJECTIVE
Use the available web tools to gather broad and deep evidence from multiple sources, then produce a concise, factual synthesis.

### OPERATING RULES
- Use multiple search engines (Brave + Google + Bing) and compare findings.
- Use AI-mode endpoints when useful to bootstrap a broad overview, but always ground conclusions with linked sources.
- Prioritize primary/original sources and include relevant URLs in your final answer.
- If sources conflict, explicitly call out the conflict and indicate which sources seem more reliable.
- Never invent facts.
- Never ask the user follow-up questions; do the best possible research autonomously.
"""


class WebSearcherAgentTool:
    def __init__(self, config: ToolsConfig, model_name: str = "", max_turns: int = 30):
        self.config = config
        self.brave = BraveSearchTool(config)
        self.web = SerpApiWebSearchTool(config)
        self.runner = SubAgentRunner(
            model_name=model_name,
            env_var_name="CHACK_WEBSEARCHER_AGENT_MODEL",
            max_turns=max(2, int(max_turns or 30)),
        )

    def _build_subagent_tools(self):
        if function_tool is None:
            raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")

        brave = self.brave
        web = self.web

        @function_tool(name_override="brave_search")
        def brave_search(
            query: str,
            count: Optional[int] = None,
            country: Optional[str] = None,
            search_lang: Optional[str] = None,
            ui_lang: Optional[str] = None,
            freshness: Optional[str] = None,
            timeout_seconds: int = 25,
        ) -> str:
            return brave.search(
                query=query,
                count=count,
                country=country,
                search_lang=search_lang,
                ui_lang=ui_lang,
                freshness=freshness,
                timeout_seconds=timeout_seconds,
            )

        @function_tool(name_override="search_google_web")
        def search_google_web(
            query: str,
            page: int = 1,
            num: Optional[int] = None,
            timeout_seconds: int = 25,
        ) -> str:
            return web.search_google_web(
                query=query,
                page=page,
                num=num,
                timeout_seconds=timeout_seconds,
            )

        @function_tool(name_override="search_bing_web")
        def search_bing_web(
            query: str,
            page: int = 1,
            count: Optional[int] = None,
            timeout_seconds: int = 25,
        ) -> str:
            return web.search_bing_web(
                query=query,
                page=page,
                count=count,
                timeout_seconds=timeout_seconds,
            )

        @function_tool(name_override="search_google_ai_mode")
        def search_google_ai_mode(
            query: str,
            timeout_seconds: int = 45,
        ) -> str:
            return web.search_google_ai_mode(
                query=query,
                timeout_seconds=timeout_seconds,
            )

        @function_tool(name_override="search_bing_copilot")
        def search_bing_copilot(
            query: str,
            timeout_seconds: int = 45,
        ) -> str:
            return web.search_bing_copilot(
                query=query,
                timeout_seconds=timeout_seconds,
            )

        return [
            brave_search,
            search_google_web,
            search_bing_web,
            search_google_ai_mode,
            search_bing_copilot,
        ]

    def run(self, prompt: str) -> str:
        has_brave = bool((self.config.brave_api_key or "").strip())
        has_serpapi = bool((self.config.serpapi_api_key or "").strip())
        if not has_brave and not has_serpapi:
            return "ERROR: Neither Brave API key nor SerpAPI key is configured."
        tools = self._build_subagent_tools()
        return self.runner.run(
            prompt=prompt,
            agent_name="Web Research Sub-Agent",
            system_prompt=_WEBSEARCHER_AGENT_SYSTEM_PROMPT,
            tools=tools,
        )


def build_websearcher_research_tool(
    config: ToolsConfig,
    model_name: str = "",
    max_turns: int = 30,
) -> StructuredTool:
    helper = WebSearcherAgentTool(
        config,
        model_name=model_name,
        max_turns=max_turns,
    )

    def _websearcher_research(prompt: str) -> str:
        """Run a dedicated web-research sub-agent for extensive web research.

        Args:
            prompt: Detailed research request for the sub-agent.
        """
        return helper.run(prompt=prompt)

    return StructuredTool.from_function(
        name="websearcher_research",
        description=_websearcher_research.__doc__ or "Run web-research sub-agent.",
        func=_websearcher_research,
    )
