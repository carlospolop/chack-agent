import os
import time
from typing import Any, Optional

from .brave_search import BraveSearchTool, get_brave_search_tool
from .config import ToolsConfig
from .playwright_fetch import (
    PlaywrightFetchTool,
    get_playwright_fetch_tool,
    is_playwright_available,
)
from .open_research_sources import (
    OpenResearchTool,
    get_fetch_url_text_tool,
    get_gdelt_news_search_tool,
    get_web_archive_search_tool,
    get_wayback_fetch_tool,
)
from .serpapi_web_search import (
    SerpApiWebSearchTool,
    get_google_web_search_tool,
    get_bing_web_search_tool,
    get_google_ai_mode_tool,
    get_bing_copilot_tool,
)
from .serpapi_keys import has_serpapi_keys
from .task_steps_manager_tool import TaskStepsManagerTool, get_task_steps_manager_tool
from .research_artifacts import add_research_artifact_tools, cleanup_research_artifacts, reset_research_artifact_context, set_research_artifact_context
from .subagent_config import (
    OBJECTIVE_EVIDENCE_RULES,
    append_evidence_dir_instruction,
    append_research_tool_usage,
    build_subagent_config,
    create_subagent_evidence_dir,
    create_subagent_session_id,
    enforce_prompt_str_or_list_schema,
    inherit_subagent_limits,
    normalize_subagent_prompts,
    record_researcher_response,
    reconcile_research_artifacts,
    run_parallel_subagent_prompts,
    subagent_launch_block_reason,
)
from .task_steps_manager_state import current_session_id
from .telemetry import current_log_context, run_with_tool_logging

try:
    from agents import function_tool
except ImportError:
    function_tool = None


_WEBSEARCHER_AGENT_SYSTEM_PROMPT = """### RULES
- Your only job is web research: broad discovery, source preservation, historic/current page context, and a concise factual synthesis.
- Use all relevant web tools repeatedly until coverage is strong: compare Brave/Google/Bing results, use AI-mode only as a discovery bootstrap, and ground conclusions in linked sources.
- Fetch readable page text/HTML for concrete URLs; use Playwright for JavaScript-rendered pages; use web archives for deleted/changed pages, old versions, historic claims, and source preservation.
- Prefer primary/original sources, include relevant URLs, mention saved page/archive artifact filenames only when artifacts are preserved, and mention sources without naming internal tool names.

""" + OBJECTIVE_EVIDENCE_RULES


class WebSearcherAgentTool:
    def __init__(
        self,
        config: ToolsConfig,
        model_name: str = "",
        fallback_model: str = "",
        model_provider: str = "",
        max_turns: int = 30,
        self_critique_enabled: bool = False,
        self_critique_rounds: int = 0,
    ):
        self.config = config
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.max_turns = max(2, int(max_turns or 30))
        self.self_critique_rounds = max(0, int(self_critique_rounds or 0))
        self.self_critique_enabled = bool(self_critique_enabled or self.self_critique_rounds > 0)
        self.brave = BraveSearchTool(config)
        self.web = SerpApiWebSearchTool(config)
        self.open = OpenResearchTool(config)

    def _resolved_model(self) -> Optional[str]:
        configured = (self.model_name or "").strip()
        if configured:
            return configured
        fallback = (self.fallback_model or "").strip()
        return fallback or None

    def _build_subagent_tools(self):
        if function_tool is None:
            raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")

        tools = []
        if getattr(self.config, "task_steps_manager_enabled", True):
            task_helper = TaskStepsManagerTool(self.config)
            tools.append(get_task_steps_manager_tool(task_helper))
        tools.append(get_fetch_url_text_tool(self.open))
        tools.append(get_web_archive_search_tool(self.open))
        tools.append(get_wayback_fetch_tool(self.open))
        tools.append(get_gdelt_news_search_tool(self.open))
        tools.append(get_brave_search_tool(self.brave))
        if self.config.playwright_enabled and is_playwright_available():
            tools.append(get_playwright_fetch_tool(PlaywrightFetchTool(self.config)))

        has_serpapi = has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if has_serpapi:
            tools.append(get_google_web_search_tool(self.web))
            tools.append(get_bing_web_search_tool(self.web))
            tools.append(get_google_ai_mode_tool(self.web))
            tools.append(get_bing_copilot_tool(self.web))
        add_research_artifact_tools(tools, self.config)
        return tools

    def _run_single(self, prompt: str, ctx: dict[str, Any], save_artifacts: bool = False) -> str:
        has_brave = bool(os.environ.get("BRAVE_API_KEY", "").strip())
        has_serpapi = has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        brave_allowed = True
        serpapi_allowed = has_serpapi
        # The web researcher also has keyless archive, GDELT, and page-fetch tools.
        # Brave/SerpAPI are preferred for broad discovery but are no longer required
        # for source preservation or URL-specific research.

        tools = self._build_subagent_tools()
        model_name = self._resolved_model() or ""
        launch_block = subagent_launch_block_reason(
            parent_original_runtime_minutes=int(ctx.get("max_runtime_minutes") or 0),
            parent_remaining_runtime_minutes=float(ctx.get("remaining_runtime_minutes") or 0.0),
            parent_original_cost_usd=float(ctx.get("max_cost_usd") or 0.0),
            parent_remaining_cost_usd=float(ctx.get("remaining_cost_usd") or 0.0),
        )
        if launch_block:
            return launch_block
        effective_max_turns, effective_runtime_minutes, effective_cost_usd = inherit_subagent_limits(
            default_max_turns=self.max_turns,
            parent_max_turns=int(ctx.get("max_turns") or 0),
            parent_remaining_runtime_minutes=float(ctx.get("remaining_runtime_minutes") or 0.0),
            parent_remaining_cost_usd=float(ctx.get("remaining_cost_usd") or 0.0),
        )
        parent_memory_max_messages = max(1, int(ctx.get("memory_max_messages") or 8))
        parent_memory_reset_to_messages = max(1, int(ctx.get("memory_reset_to_messages") or parent_memory_max_messages))
        parent_root_session_id = str(ctx.get("session_id") or "").strip()
        evidence_dir = create_subagent_evidence_dir("websearcher", parent_root_session_id)
        prompt = append_evidence_dir_instruction(
            prompt,
            evidence_dir,
            "Now start the research navigating through the web using the tools given!",
            save_artifacts=save_artifacts,
        )
        overrides = {
            "agent": {
                "self_critique_enabled": self.self_critique_enabled,
                "self_critique_rounds": self.self_critique_rounds,
            },
            "session": {
                "max_turns": effective_max_turns,
                "memory_max_messages": parent_memory_max_messages,
                "memory_reset_to_messages": parent_memory_reset_to_messages,
                "long_term_memory_enabled": False,
                "long_term_memory_max_chars": 0,
                "long_term_memory_dir": "",
            },
            "tools": {
                "max_tools_used": self.config.websearcher_max_tools_used,
                "websearcher_enabled": True,
                "websearcher_brave_enabled": True,
                "websearcher_google_web_enabled": True,
                "websearcher_bing_web_enabled": True,
                "websearcher_google_ai_mode_enabled": True,
                "websearcher_bing_copilot_enabled": True,
                "websearcher_web_archive_enabled": True,
                "websearcher_gdelt_enabled": True,
                "websearcher_fetch_url_text_enabled": True,
                "brave_enabled": True,
                "serpapi_google_web_enabled": True,
                "serpapi_bing_web_enabled": True,
                "serpapi_bing_copilot_enabled": True,
                "exec_enabled": False,
                "pdf_text_enabled": False,
                "scientific_enabled": False,
                "social_network_enabled": False,
            },
            "env": {
                "CHACK_RESEARCH_DATA_DIR": evidence_dir,
                "CHACK_RESEARCH_SAVE_ARTIFACTS": "1" if save_artifacts else "0",
            },
        }
        overrides["agent"]["max_runtime_minutes"] = effective_runtime_minutes
        overrides["agent"]["max_cost_usd"] = effective_cost_usd
        main_action = str(ctx.get("main_action") or "").strip()
        if main_action:
            overrides["agent"]["main_action"] = main_action
        overrides["agent"]["sub_action"] = "webresearcher"
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=_WEBSEARCHER_AGENT_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("websearcher", parent_root_session_id)
        from chack_agent import Chack
        chack = Chack(config)
        artifact_context_tokens = set_research_artifact_context(
            evidence_dir,
            os.environ.get("CHACK_RESEARCH_MASTER_DIR", "").strip(),
        )
        try:
            result = chack.run(
                session_id=subagent_session_id,
                text=prompt,
                min_tools_used_override=0,
                max_tools_used_override=self.config.websearcher_max_tools_used,
                enable_self_critique=None,
                require_task_steps_manager_init_first=bool(
                    getattr(self.config, "task_steps_manager_enabled", True)
                ),
                tools_override=tools,
                system_prompt_override=config.system_prompt,
                usage_session_id=parent_task_session_id,
            )
            output = result.output.strip() if result.output else "ERROR: sub-agent returned an empty response."
            if output.startswith("ERROR:"):
                return output
            tool_counts = result.tool_counts.copy()

            def _run_followup(followup: str, output_schema_json=None) -> str:
                followup_result = chack.run(
                    session_id=subagent_session_id,
                    text=followup,
                    min_tools_used_override=0,
                    max_tools_used_override=self.config.websearcher_max_tools_used,
                    enable_self_critique=False,
                    self_critique_rounds_override=0,
                    require_task_steps_manager_init_first=False,
                    tools_override=tools,
                    system_prompt_override=config.system_prompt,
                    usage_session_id=parent_task_session_id,
                    output_schema_json_override=output_schema_json,
                )
                tool_counts.update(followup_result.tool_counts)
                return (followup_result.output or "").strip()

            output = reconcile_research_artifacts(
                output,
                evidence_dir=evidence_dir,
                save_artifacts=bool(save_artifacts and getattr(self.config, "research_strict_artifact_manifest", True)),
                run_followup=_run_followup,
            )
            return append_research_tool_usage(output, tool_counts)
        finally:
            cleanup_research_artifacts(evidence_dir, save_artifacts=save_artifacts)
            reset_research_artifact_context(artifact_context_tokens)

    def run(self, prompt: str | list[str], save_artifacts: bool = False) -> str:
        prompts, error = normalize_subagent_prompts(prompt, min_chars=500, max_prompts=3)
        if error:
            return error
        ctx = current_log_context()
        return run_parallel_subagent_prompts(
            prompts,
            lambda item: self._run_single(item, ctx, save_artifacts=save_artifacts),
        )


def get_websearcher_research_tool(
    helper: WebSearcherAgentTool,
):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="websearcher_research")
    def websearcher_research(prompt: str | list[str], save_artifacts: bool = False) -> str:
        """Run a dedicated web-research sub-agent for extensive web research.

        Use when you need broad, iterative web investigation without consuming your main context.
        The sub-agent uses Brave + Google + Bing (including AI-mode endpoints) to cross-validate.

        Args:
            prompt: A detailed web research request (string) or a list of up to 3 detailed requests. Each request must be at least 500 characters indicating the goals of the subagent, suggested process to obtain proper results, expected output, and relevant information to gather.
            save_artifacts: If true, preserve the evidence folder after the run and return it in the JSON result. If false, artifacts are temporary and deleted after the run.

        Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final review, and artifact folder path only when artifacts are preserved.
        """
        tool_input = {"prompt": prompt, "save_artifacts": save_artifacts}
        try:
            return run_with_tool_logging(
                "websearcher_research",
                tool_input,
                lambda: _run_and_record_researcher_response(
                    "websearcher_research",
                    helper.run(prompt=prompt, save_artifacts=save_artifacts),
                ),
            )
        except Exception as exc:
            return f"ERROR: websearcher_research failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(websearcher_research)
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: Provide prompt as one detailed request or up to 3 detailed requests; set save_artifacts true only when the evidence folder must be preserved.\n"
        "Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final review, and artifact folder path only when artifacts are preserved."
    )
    return tool


def _run_and_record_researcher_response(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output
