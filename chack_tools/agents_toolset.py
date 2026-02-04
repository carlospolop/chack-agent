import os
import subprocess
from typing import Optional

from agents import function_tool

from .config import ToolsConfig
from .brave_search import BraveSearchTool
from .formatting import _truncate
from .forumscout_search import ForumScoutTool
from .pdf_text import PdfTextTool
from .serpapi_web_search import SerpApiWebSearchTool
from .scientific_search import ScientificSearchTool
from .scientific_research_agent import ScientificResearchAgentTool
from .social_network_agent import SocialNetworkAgentTool
from .task_list_tool import TaskListTool
from .websearcher_agent import WebSearcherAgentTool


def _exec_command(command: str) -> str:
    timeout = int(os.environ.get("CHACK_EXEC_TIMEOUT", "120"))
    max_chars = int(os.environ.get("CHACK_EXEC_MAX_OUTPUT", "4000"))
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip() or "(no output)"
    return _truncate(output, max_chars)


class AgentsToolset:
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

    @staticmethod
    def _make_brave_tool(helper: BraveSearchTool):
        @function_tool(name_override="brave_search")
        def brave_search(
            query: str,
            count: Optional[int] = None,
            country: Optional[str] = None,
            search_lang: Optional[str] = None,
            ui_lang: Optional[str] = None,
            freshness: Optional[str] = None,
            timeout_seconds: int = 20,
        ) -> str:
            """Search Brave Search API and return a short list of results.

            Args:
                query: Search query string.
                count: Optional number of results to return (1-20).
                country: Optional country code (e.g., "US").
                search_lang: Optional search language (e.g., "en").
                ui_lang: Optional UI language (e.g., "en-US").
                freshness: Optional freshness filter (pd, pw, pm, py, or YYYY-MM-DDtoYYYY-MM-DD).
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper._brave_search_impl(
                    query=query,
                    count=count,
                    country=country,
                    search_lang=search_lang,
                    ui_lang=ui_lang,
                    freshness=freshness,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: Brave search failed ({exc})"

        return brave_search

    @staticmethod
    def _make_google_web_tool(helper: SerpApiWebSearchTool):
        @function_tool(name_override="search_google_web")
        def search_google_web(
            query: str,
            page: int = 1,
            num: Optional[int] = None,
            timeout_seconds: int = 20,
        ) -> str:
            """Search Google web results via SerpAPI.

            Args:
                query: Search query string.
                page: Result page (1+).
                num: Number of results (1-10). Defaults to config value.
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.search_google_web(
                    query=query,
                    page=page,
                    num=num,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: Google web search failed ({exc})"

        return search_google_web

    @staticmethod
    def _make_bing_web_tool(helper: SerpApiWebSearchTool):
        @function_tool(name_override="search_bing_web")
        def search_bing_web(
            query: str,
            page: int = 1,
            count: Optional[int] = None,
            timeout_seconds: int = 20,
        ) -> str:
            """Search Bing web results via SerpAPI.

            Args:
                query: Search query string.
                page: Result page (1+).
                count: Number of results (1-10). Defaults to config value.
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.search_bing_web(
                    query=query,
                    page=page,
                    count=count,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: Bing web search failed ({exc})"

        return search_bing_web

    @staticmethod
    def _make_forum_tools(helper: ForumScoutTool):
        @function_tool(name_override="forum_search")
        def forum_search(
            query: str,
            time: str = "",
            country: str = "",
            page: int = 1,
            timeout_seconds: int = 20,
        ) -> str:
            """Search forums via ForumScout.

            Args:
                query: Search keyword.
                time: One of '', hour, day, week, month, year.
                country: Optional ISO 3166-1 alpha-2 code (e.g., us).
                page: Page number (1+).
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.forum_search(
                    query=query,
                    time=time,
                    country=country,
                    page=page,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: ForumScout forum_search failed ({exc})"

        @function_tool(name_override="linkedin_search")
        def linkedin_search(
            query: str,
            page: int = 1,
            sort_by: str = "date_posted",
            timeout_seconds: int = 20,
        ) -> str:
            """Search LinkedIn posts via ForumScout.

            Args:
                query: Search keyword.
                page: Page number (1+).
                sort_by: One of date_posted, relevance.
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.linkedin_search(
                    query=query,
                    page=page,
                    sort_by=sort_by,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: ForumScout linkedin_search failed ({exc})"

        @function_tool(name_override="instagram_search")
        def instagram_search(
            query: str,
            page: int = 1,
            sort_by: str = "recent",
            timeout_seconds: int = 20,
        ) -> str:
            """Search Instagram posts via ForumScout.

            Args:
                query: Search keyword.
                page: Page number (1+).
                sort_by: One of recent, top.
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.instagram_search(
                    query=query,
                    page=page,
                    sort_by=sort_by,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: ForumScout instagram_search failed ({exc})"

        @function_tool(name_override="reddit_posts_search")
        def reddit_posts_search(
            query: str,
            page: int = 1,
            sort_by: str = "new",
            timeout_seconds: int = 20,
        ) -> str:
            """Search Reddit posts via ForumScout.

            Args:
                query: Search keyword.
                page: Page number (1+).
                sort_by: One of hot, new, relevance, top.
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.reddit_posts_search(
                    query=query,
                    page=page,
                    sort_by=sort_by,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: ForumScout reddit_posts_search failed ({exc})"

        @function_tool(name_override="reddit_comments_search")
        def reddit_comments_search(
            query: str,
            page: int = 1,
            sort_by: str = "created_utc",
            timeout_seconds: int = 20,
        ) -> str:
            """Search Reddit comments via ForumScout.

            Args:
                query: Search keyword.
                page: Page number (1+).
                sort_by: One of created_utc, score.
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.reddit_comments_search(
                    query=query,
                    page=page,
                    sort_by=sort_by,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: ForumScout reddit_comments_search failed ({exc})"

        @function_tool(name_override="x_search")
        def x_search(
            query: str,
            page: int = 1,
            sort_by: str = "Latest",
            timeout_seconds: int = 20,
        ) -> str:
            """Search X (Twitter) posts via ForumScout.

            Args:
                query: Search keyword.
                page: Page number (1+).
                sort_by: One of Latest, Top.
                timeout_seconds: Request timeout in seconds.
            """
            try:
                return helper.x_search(
                    query=query,
                    page=page,
                    sort_by=sort_by,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: ForumScout x_search failed ({exc})"

        return [
            forum_search,
            linkedin_search,
            instagram_search,
            reddit_posts_search,
            reddit_comments_search,
            x_search,
        ]

    @staticmethod
    def _make_scientific_tools(helper: ScientificSearchTool):
        @function_tool(name_override="search_arxiv")
        def search_arxiv(query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
            """Search arXiv papers with direct PDF URLs."""
            try:
                return helper.search_arxiv(query=query, max_results=max_results, timeout_seconds=timeout_seconds)
            except Exception as exc:
                return f"ERROR: arXiv search failed ({exc})"

        @function_tool(name_override="search_europe_pmc")
        def search_europe_pmc(
            query: str,
            page: int = 1,
            page_size: int = 25,
            timeout_seconds: int = 20,
        ) -> str:
            """Search Europe PMC and return open-access papers with PDF URLs."""
            try:
                return helper.search_europe_pmc(
                    query=query,
                    page=page,
                    page_size=page_size,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: Europe PMC search failed ({exc})"

        @function_tool(name_override="search_semantic_scholar")
        def search_semantic_scholar(query: str, limit: int = 20, timeout_seconds: int = 20) -> str:
            """Search Semantic Scholar and return papers with open-access URLs."""
            try:
                return helper.search_semantic_scholar(query=query, limit=limit, timeout_seconds=timeout_seconds)
            except Exception as exc:
                return f"ERROR: Semantic Scholar search failed ({exc})"

        @function_tool(name_override="search_openalex")
        def search_openalex(
            query: str,
            page: int = 1,
            per_page: int = 10,
            timeout_seconds: int = 20,
        ) -> str:
            """Search OpenAlex and return works with open-access PDF URLs."""
            try:
                return helper.search_openalex(
                    query=query,
                    page=page,
                    per_page=per_page,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: OpenAlex search failed ({exc})"

        @function_tool(name_override="search_plos")
        def search_plos(query: str, rows: int = 20, start: int = 0, timeout_seconds: int = 20) -> str:
            """Search PLOS and return direct full-text PDF URLs."""
            try:
                return helper.search_plos(query=query, rows=rows, start=start, timeout_seconds=timeout_seconds)
            except Exception as exc:
                return f"ERROR: PLOS search failed ({exc})"

        return [
            search_arxiv,
            search_europe_pmc,
            search_semantic_scholar,
            search_openalex,
            search_plos,
        ]

    @staticmethod
    def _make_pdf_tool(helper: PdfTextTool):
        @function_tool(name_override="download_pdf_as_text")
        def download_pdf_as_text(
            url: str,
            max_chars: Optional[int] = None,
            timeout_seconds: int = 30,
        ) -> str:
            """Download a PDF URL and extract readable text."""
            try:
                return helper.download_pdf_as_text(
                    url=url,
                    max_chars=max_chars,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return f"ERROR: PDF extraction failed ({exc})"

        return download_pdf_as_text

    @staticmethod
    def _make_social_network_tool(helper: SocialNetworkAgentTool):
        @function_tool(name_override="social_network_research")
        def social_network_research(prompt: str) -> str:
            """Run a dedicated social-network sub-agent using ForumScout tools.

            Args:
                prompt: The research request for the nested social sub-agent.
            """
            try:
                return helper.run(prompt=prompt)
            except Exception as exc:
                return f"ERROR: social_network_research failed ({exc})"

        return social_network_research

    @staticmethod
    def _make_scientific_research_tool(helper: ScientificResearchAgentTool):
        @function_tool(name_override="scientific_research")
        def scientific_research(prompt: str) -> str:
            """Run a dedicated scientific-research sub-agent.

            Args:
                prompt: The scientific research request for the nested sub-agent.
            """
            try:
                return helper.run(prompt=prompt)
            except Exception as exc:
                return f"ERROR: scientific_research failed ({exc})"

        return scientific_research

    @staticmethod
    def _make_websearcher_tool(helper: WebSearcherAgentTool):
        @function_tool(name_override="websearcher_research")
        def websearcher_research(prompt: str) -> str:
            """Run a dedicated web-research sub-agent for extensive web research.

            Args:
                prompt: Detailed research request for the nested sub-agent.
            """
            try:
                return helper.run(prompt=prompt)
            except Exception as exc:
                return f"ERROR: websearcher_research failed ({exc})"

        return websearcher_research

    @staticmethod
    def _make_task_list_tool(helper: TaskListTool):
        @function_tool(name_override="task_list")
        def task_list(
            action: str,
            task_id: Optional[int] = None,
            text: str = "",
            status: str = "",
            tasks: str = "",
            notes: str = "",
        ) -> str:
            """Create and maintain the live per-request task plan shown to the user.

            Guardrails and behavior:
            - In each run, the first use of this tool MUST be `action="init"`.
            - Task-list calls DO NOT count toward the minimum non-task tool usage target.
            - The rendered board is persisted per request and split by run label
              (e.g. "Run 1" and "Run 2 (self-critique)").
            - Every mutating action updates the live board message in chat/thread.

            Actions and arguments:
            - `init`: initialize the current run list. Provide `tasks` as newline-separated items.
            - `list`: return the current full board.
            - `add`: add one task with `text` (optional `status`, `notes`).
            - `update`: update task `task_id` fields (`text`, `status`, `notes`).
            - `complete`: mark `task_id` as done (optional `notes`).
            - `delete`: remove `task_id`.
            - `clear`: clear all tasks in the current run.
            - `replace`: replace the current run list with newline-separated `tasks`.

            Status values:
            - `todo`, `doing`, `done`.
            """
            try:
                return helper.manage(
                    action=action,
                    task_id=task_id,
                    text=text,
                    status=status,
                    tasks=tasks,
                    notes=notes,
                )
            except Exception as exc:
                return f"ERROR: task list update failed ({exc})"

        return task_list

    def _build_tools(self):
        tools = []
        if self.config.exec_enabled:
            @function_tool(name_override="exec")
            def exec_tool(command: str) -> str:
                """Execute a shell command locally and return combined output.

                Args:
                    command: Shell command to execute.
                """
                return _exec_command(command)

            tools.append(exec_tool)
        task_helper = TaskListTool(self.config)
        tools.append(self._make_task_list_tool(task_helper))

        if self.config.brave_enabled:
            brave_helper = BraveSearchTool(self.config)
            tools.append(self._make_brave_tool(brave_helper))
        has_serpapi = bool((self.config.serpapi_api_key or "").strip())
        if has_serpapi and self.config.serpapi_google_web_enabled:
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(self._make_google_web_tool(web_helper))
        include_bing_web = self.tool_profile in {"all", "telegram"}
        if has_serpapi and self.config.serpapi_bing_web_enabled and include_bing_web:
            web_helper = SerpApiWebSearchTool(self.config)
            tools.append(self._make_bing_web_tool(web_helper))
        if self.config.websearcher_enabled and (self.config.brave_enabled or has_serpapi):
            websearcher_helper = WebSearcherAgentTool(
                self.config,
                model_name=self.websearcher_model,
                max_turns=self.websearcher_max_turns,
            )
            tools.append(self._make_websearcher_tool(websearcher_helper))

        include_forumscout = self.tool_profile in {"all", "telegram"}
        if self.config.forumscout_enabled and include_forumscout:
            social_helper = SocialNetworkAgentTool(
                self.config,
                model_name=self.social_network_model,
                max_turns=self.social_network_max_turns,
            )
            tools.append(self._make_social_network_tool(social_helper))

        include_scientific = self.tool_profile in {"all", "telegram"}
        if self.config.scientific_enabled and include_scientific:
            scientific_helper = ScientificResearchAgentTool(
                self.config,
                model_name=self.scientific_model,
                max_turns=self.scientific_max_turns,
            )
            tools.append(self._make_scientific_research_tool(scientific_helper))

        include_pdf = self.tool_profile in {"all", "telegram"}
        if self.config.pdf_text_enabled and include_pdf:
            pdf_helper = PdfTextTool(self.config)
            tools.append(self._make_pdf_tool(pdf_helper))

        return tools
