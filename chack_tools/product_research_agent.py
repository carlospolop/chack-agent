from __future__ import annotations

import os
import time
from typing import Any, Optional

from .business_search import (
    BusinessSearchTool,
    get_amazon_product_search_tool,
    get_amazon_product_tool,
    get_ebay_product_search_tool,
    get_ebay_product_tool,
    get_google_immersive_product_tool,
    get_google_shopping_light_search_tool,
    get_google_shopping_search_tool,
    get_home_depot_product_search_tool,
    get_home_depot_product_tool,
    get_walmart_product_search_tool,
    get_walmart_product_tool,
)
from .config import ToolsConfig
from .forumscout_search import ForumScoutTool, get_google_trends_search_tool
from .playwright_fetch import (
    PlaywrightFetchTool,
    get_playwright_fetch_tool,
    is_playwright_available,
)
from .product_search import (
    ProductSearchTool,
    get_google_lens_products_tool,
    get_nvd_cpe_search_tool,
    get_nvd_cve_search_tool,
    get_open_food_facts_product_tool,
    get_open_food_facts_search_tool,
    get_openfda_recalls_search_tool,
)
from .open_research_sources import (
    OpenResearchTool,
    get_cisa_kev_search_tool,
    get_cpsc_recalls_search_tool,
    get_fetch_url_text_tool,
    get_osv_package_query_tool,
)
from .scientific_search import (
    ScientificSearchTool,
    get_google_patents_details_tool,
    get_google_patents_search_tool,
    get_youtube_transcript_tool,
    get_youtube_video_details_tool,
    get_youtube_video_search_tool,
)
from .serpapi_keys import has_serpapi_keys
from .serpapi_web_search import (
    SerpApiWebSearchTool,
    get_bing_web_search_tool,
    get_google_web_search_tool,
)
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
from .task_steps_manager_tool import TaskStepsManagerTool, get_task_steps_manager_tool
from .research_artifacts import add_research_artifact_tools, cleanup_research_artifacts, reset_research_artifact_context, set_research_artifact_context
from .telemetry import current_log_context, run_with_tool_logging

try:
    from agents import function_tool
except ImportError:
    function_tool = None


_PRODUCT_AGENT_SYSTEM_PROMPT = """### RULES
- Your only job is product research: identifiers, variants, sellers, marketplaces, price/availability, safety, recalls, vulnerabilities, reviews, patents, videos, visual and web evidence.
- Use all relevant product tools repeatedly until coverage is strong. Preserve raw product, marketplace, recall, vulnerability, patent, video/transcript, image-search, and web artifacts for audit.
- When search/shopping/video/review results identify a source you rely on, fetch the underlying product page, recall notice, advisory, vulnerability page, marketplace listing, review page, or article with a content-access/detail tool before making that source part of the final review.
- In preserved-artifact runs, a search-only product report is not acceptable. Use `fetch_url_text` or a product/detail/download/transcript tool for the core source(s) you rely on; if every relevant source cannot be fetched or opened, return research_worked=false with the reason instead of synthesizing from snippets.
- Identify exact product identifiers when possible: barcode/GTIN/UPC/EAN, ASIN, SKU, product_id, CPE, CVE, patent IDs, model/part numbers, seller IDs, URLs, dates, countries, and marketplace domains.
- For food/CPG, check ingredients, nutrition, labels, and recalls; for software/electronics/connected devices, check vulnerability and product-identifier sources; for visual evidence, use image-search when an image URL is available.
- Treat shopping, marketplace, trend, review, video, patent, and web results as snapshots. Compare sources, label uncertainty, and state when a source does not cover the target.
- Mention sources, and mention artifact filenames only when artifacts are preserved, without naming internal tool names.
""" + OBJECTIVE_EVIDENCE_RULES


class ProductResearchAgentTool:
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
        self.product = ProductSearchTool(config)
        self.business = BusinessSearchTool(config)
        self.web = SerpApiWebSearchTool(config)
        self.forum = ForumScoutTool(config)
        self.scientific = ScientificSearchTool(config)
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

        tools.append(get_open_food_facts_search_tool(self.product))
        tools.append(get_open_food_facts_product_tool(self.product))
        tools.append(get_openfda_recalls_search_tool(self.product))
        tools.append(get_nvd_cpe_search_tool(self.product))
        tools.append(get_nvd_cve_search_tool(self.product))
        tools.append(get_cpsc_recalls_search_tool(self.open))
        tools.append(get_cisa_kev_search_tool(self.open))
        tools.append(get_osv_package_query_tool(self.open))

        has_serpapi = has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if has_serpapi:
            tools.append(get_google_lens_products_tool(self.product))
            tools.append(get_google_shopping_search_tool(self.business))
            tools.append(get_google_shopping_light_search_tool(self.business))
            tools.append(get_google_immersive_product_tool(self.business))
            tools.append(get_amazon_product_search_tool(self.business))
            tools.append(get_amazon_product_tool(self.business))
            tools.append(get_walmart_product_search_tool(self.business))
            tools.append(get_walmart_product_tool(self.business))
            tools.append(get_ebay_product_search_tool(self.business))
            tools.append(get_ebay_product_tool(self.business))
            tools.append(get_home_depot_product_search_tool(self.business))
            tools.append(get_home_depot_product_tool(self.business))
            tools.append(get_google_web_search_tool(self.web))
            tools.append(get_bing_web_search_tool(self.web))
            tools.append(get_google_trends_search_tool(self.forum))
            tools.append(get_google_patents_search_tool(self.scientific))
            tools.append(get_google_patents_details_tool(self.scientific))
            tools.append(get_youtube_video_search_tool(self.scientific))
            tools.append(get_youtube_video_details_tool(self.scientific))
            tools.append(get_youtube_transcript_tool(self.scientific))
        tools.append(get_fetch_url_text_tool(self.open))

        if self.config.playwright_enabled and is_playwright_available():
            tools.append(get_playwright_fetch_tool(PlaywrightFetchTool(self.config)))
        add_research_artifact_tools(tools, self.config)

        return tools

    def _run_single(self, prompt: str, ctx: dict[str, Any], save_artifacts: bool = False) -> str:
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
        evidence_dir = create_subagent_evidence_dir("product", parent_root_session_id)
        prompt = append_evidence_dir_instruction(
            prompt,
            evidence_dir,
            "Now start the product research checking all product, marketplace, safety, recall, vulnerability, review, trend, patent, video, and web tools given!",
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
                "product_enabled": True,
                "product_open_food_facts_enabled": True,
                "product_openfda_enabled": True,
                "product_nvd_enabled": True,
                "product_cpsc_enabled": True,
                "product_cisa_kev_enabled": True,
                "product_osv_enabled": True,
                "product_google_lens_enabled": True,
                "product_google_shopping_enabled": True,
                "product_google_shopping_light_enabled": True,
                "product_google_immersive_product_enabled": True,
                "product_amazon_enabled": True,
                "product_walmart_enabled": True,
                "product_ebay_enabled": True,
                "product_home_depot_enabled": True,
                "product_google_trends_enabled": True,
                "product_google_patents_enabled": True,
                "product_youtube_enabled": True,
                "product_playwright_enabled": True,
                "product_max_tools_used": self.config.product_max_tools_used,
                "max_tools_used": self.config.product_max_tools_used,
                "brave_enabled": False,
                "exec_enabled": False,
                "pdf_text_enabled": False,
                "business_enabled": False,
                "scientific_enabled": False,
                "social_network_enabled": False,
                "websearcher_enabled": False,
                "serpapi_google_web_enabled": True,
                "serpapi_bing_web_enabled": True,
                "scientific_google_patents_enabled": True,
                "scientific_google_patents_details_enabled": True,
                "scientific_youtube_search_enabled": True,
                "scientific_youtube_details_enabled": True,
                "scientific_youtube_transcript_enabled": True,
                "social_network_google_trends_enabled": True,
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
        overrides["agent"]["sub_action"] = "product"
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=_PRODUCT_AGENT_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("product", parent_root_session_id)
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
                max_tools_used_override=self.config.product_max_tools_used,
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
                    max_tools_used_override=self.config.product_max_tools_used,
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


def get_product_research_tool(helper: ProductResearchAgentTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="product_research")
    def product_research(prompt: str | list[str], save_artifacts: bool = False) -> str:
        """Run a dedicated product-research sub-agent.

        Use this tool to launch an autonomous product researcher for product identifiers,
        marketplaces, prices, sellers, reviews, safety/recalls, vulnerabilities, patents,
        videos/transcripts, trends, and web evidence.

        Args:
            prompt: A detailed product research request (string) or a list of up to 3 detailed requests. Each request must be at least 500 characters indicating target products, identifiers, markets, countries, timeframes, evidence to collect, expected output, and important caveats.
            save_artifacts: If true, preserve the evidence folder after the run and return it in the JSON result. If false, artifacts are temporary and deleted after the run.

        Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final product review, and artifact folder path only when artifacts are preserved.
        """
        tool_input = {"prompt": prompt, "save_artifacts": save_artifacts}
        try:
            return run_with_tool_logging(
                "product_research",
                tool_input,
                lambda: _run_and_record_researcher_response(
                    "product_research",
                    helper.run(prompt=prompt, save_artifacts=save_artifacts),
                ),
            )
        except Exception as exc:
            return f"ERROR: product_research failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(product_research)
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: Provide prompt as one detailed product request or up to 3 detailed requests; set save_artifacts true only when the evidence folder must be preserved.\n"
        "Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final product review, and artifact folder path only when artifacts are preserved."
    )
    return tool


def _run_and_record_researcher_response(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output
