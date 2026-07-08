from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from .config import ToolsConfig
from .exec_tool import ExecTool, get_controlled_shell_command_tool
from .forumscout_search import (
    ForumScoutTool,
    get_google_news_search_tool,
    get_google_trends_search_tool,
    get_google_trends_trending_now_tool,
    get_google_videos_search_tool,
)
from .open_research_sources import (
    OpenResearchTool,
    get_boe_aux_table_tool,
    get_boe_law_metadata_tool,
    get_boe_law_search_tool,
    get_boe_law_text_download_tool,
    get_bible_passage_tool,
    get_federal_register_search_tool,
    get_fetch_url_text_tool,
    get_gdelt_news_search_tool,
    get_gita_chapter_tool,
    get_gita_chapters_tool,
    get_gita_verse_tool,
    get_hadith_collection_tool,
    get_hadith_editions_tool,
    get_hadith_search_tool,
    get_hadith_section_tool,
    get_quran_chapters_tool,
    get_quran_search_tool,
    get_quran_verse_tool,
    get_sefaria_search_tool,
    get_sefaria_text_tool,
    get_suttacentral_suttaplex_tool,
    get_suttacentral_text_tool,
    get_web_archive_search_tool,
    get_wayback_fetch_tool,
    get_wikidata_entity_search_tool,
    get_wikidata_sparql_tool,
    get_world_bank_indicator_tool,
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


_LEGAL_PROMPT = """### RULES
- Your only job is legal/regulatory research: statutes, rules, notices, enforcement signals, regulatory documents, jurisdiction, provenance, dates, and source links.
- Use all relevant legal/regulatory tools repeatedly until coverage is strong. Prefer primary official records; treat web/news as leads unless corroborated.
- When search/news results identify a regulation, complaint, enforcement action, court record, agency page, or article you rely on, fetch/download the underlying source text before making it part of the final legal review.
- For Spanish law, use BOE consolidated-legislation search, metadata/analysis, auxiliary tables, and full consolidated text downloads as generic legal sources, not just latest-news monitoring.
- Track jurisdiction, agency/court, document type, publication/effective dates, docket/notice IDs, exact URLs, gaps, and contradictions.
- Mention sources, and mention artifact filenames only when artifacts are preserved, without naming internal tool names.
""" + OBJECTIVE_EVIDENCE_RULES


_DATA_STATS_PROMPT = """### RULES
- Your only job is data/statistics research: datasets, indicators, time series, entity IDs, raw JSON/CSV-like evidence, methods, units, geography, date ranges, update dates, and definitions.
- Use all relevant data/statistics tools repeatedly until coverage is strong. Prefer downloadable/queryable primary data over prose summaries.
- Use command execution when it helps fetch, inspect, parse, calculate from, or validate datasets and save useful outputs as artifacts.
- Synchronous command execution has hard timeouts. For long-running fetches/parsers/calculations, start them in the background with explicit log/output files, then monitor them with quick commands such as ps, tail, grep, cat, or checking exit/status files.
- Preserve conflicting datasets instead of smoothing them over; report differences, gaps, and likely causes only when evidence supports them.
- Mention sources, and mention artifact filenames only when artifacts are preserved, without naming internal tool names.
- IMPORTANT: NEVER UNDER ANY CONCEPT EXECUTE ANY POTENTIALLY DANGEROUS PROGRAM (MALWARE, VIRUS, C2, REV SHELL) UNDER ANY CIRCUNSTANCES
""" + OBJECTIVE_EVIDENCE_RULES


_NEWS_MEDIA_PROMPT = """### RULES
- Your only job is news/media-intelligence research: coverage, timelines, source domains, story clusters, media/video evidence, trend signals, archived pages, and original URLs.
- Use all relevant news/media tools repeatedly until coverage is strong. Prefer original reporting, direct documents, archived/source pages, and primary media over AI summaries.
- Treat trend/news APIs as discovery evidence, not proof by themselves; track timestamps, publishers, countries/languages, syndication, duplication, gaps, and contradictions.
- Mention sources, and mention artifact filenames only when artifacts are preserved, without naming internal tool names.
""" + OBJECTIVE_EVIDENCE_RULES


_KG_ENTITY_PROMPT = """### RULES
- Your only job is knowledge-graph/entity research: entity resolution, identifiers, aliases, relationships, official URLs, registry IDs, graph claims, and provenance.
- Use all relevant entity/graph tools repeatedly until coverage is strong. Prefer structured entity sources plus primary pages.
- For identifiable people, organizations, places, products, projects, or public datasets, start with structured entity search/SPARQL-style graph tools to collect IDs, aliases, relationships, coordinates, official URLs, and source claims; use web search/fetch mainly to verify or fill missing primary pages.
- Treat labels, descriptions, and graph statements as sourced claims, not guaranteed truth; preserve ambiguous candidates when evidence is insufficient.
- Mention sources, and mention artifact filenames only when artifacts are preserved, without naming internal tool names.
""" + OBJECTIVE_EVIDENCE_RULES


_RELIGIOUS_PROMPT = """### RULES
- Your only job is religious-text/context research: primary scriptures, translations, commentaries, references, editions, provenance, languages, and source links.
- Use all relevant religious/web/entity tools repeatedly until coverage is strong. Prefer primary-text APIs and inspectable artifacts; use web results only for discovery, context, and secondary commentary.
- Retrieve exact Bible references and translations; use Sefaria for Jewish texts; Quran.com/Quran Foundation for Quran search/verses/chapters; Bhagavad Gita tools for verses/commentaries; Hadith tools for editions/sections/search; SuttaCentral/Bilara for Buddhist metadata, Pali roots, and translations.
- For any central scripture, commentary, or tradition claim, retrieve at least the exact referenced passage or text record with a religious content tool when one is available; do not rely on memory or search snippets for primary text evidence.
- Stay comparative across traditions, denominations, schools, translations, and interpretations; track exact references, versions, languages, source URLs, gaps, and artifact filenames only when artifacts are preserved.
- Mention sources without naming internal tool names.
""" + OBJECTIVE_EVIDENCE_RULES


class OpenSpecialistResearchAgentTool:
    def __init__(
        self,
        config: ToolsConfig,
        *,
        kind: str,
        system_prompt: str,
        build_tools: Callable[["OpenSpecialistResearchAgentTool"], list[Any]],
        max_tools_attr: str,
        model_name: str = "",
        fallback_model: str = "",
        model_provider: str = "",
        max_turns: int = 30,
        self_critique_enabled: bool = False,
        self_critique_rounds: int = 0,
    ):
        self.config = config
        self.kind = kind
        self.system_prompt = system_prompt
        self._build_tools_callback = build_tools
        self.max_tools_attr = max_tools_attr
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.max_turns = max(2, int(max_turns or 30))
        self.self_critique_rounds = max(0, int(self_critique_rounds or 0))
        self.self_critique_enabled = bool(self_critique_enabled or self.self_critique_rounds > 0)
        self.open = OpenResearchTool(config)
        self.web = SerpApiWebSearchTool(config)
        self.forum = ForumScoutTool(config)

    def _resolved_model(self) -> Optional[str]:
        configured = (self.model_name or "").strip()
        if configured:
            return configured
        fallback = (self.fallback_model or "").strip()
        return fallback or None

    def _build_subagent_tools(self):
        if function_tool is None:
            raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")
        return self._build_tools_callback(self)

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
        evidence_dir = create_subagent_evidence_dir(self.kind, parent_root_session_id)
        prompt = append_evidence_dir_instruction(
            prompt,
            evidence_dir,
            f"Now start the {self.kind.replace('_', ' ')} research using all relevant tools given!",
            save_artifacts=save_artifacts,
        )
        max_tools = int(getattr(self.config, self.max_tools_attr, 0) or 0)
        overrides = {
            "agent": {
                "self_critique_enabled": self.self_critique_enabled,
                "self_critique_rounds": self.self_critique_rounds,
                "max_runtime_minutes": effective_runtime_minutes,
                "max_cost_usd": effective_cost_usd,
                "sub_action": self.kind,
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
                "max_tools_used": max_tools,
                "brave_enabled": False,
                "exec_enabled": self.kind == "data_statistics",
                "pdf_text_enabled": False,
                "scientific_enabled": False,
                "social_network_enabled": False,
                "websearcher_enabled": False,
                "serpapi_google_web_enabled": True,
                "serpapi_bing_web_enabled": True,
            },
            "env": {
                "CHACK_RESEARCH_DATA_DIR": evidence_dir,
                "CHACK_RESEARCH_SAVE_ARTIFACTS": "1" if save_artifacts else "0",
            },
        }
        main_action = str(ctx.get("main_action") or "").strip()
        if main_action:
            overrides["agent"]["main_action"] = main_action
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=self.system_prompt,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id(self.kind, parent_root_session_id)
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
                max_tools_used_override=max_tools,
                enable_self_critique=None,
                require_task_steps_manager_init_first=bool(getattr(self.config, "task_steps_manager_enabled", True)),
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
                    max_tools_used_override=max_tools,
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
        return run_parallel_subagent_prompts(prompts, lambda item: self._run_single(item, ctx, save_artifacts=save_artifacts))


def _base_tools(helper: OpenSpecialistResearchAgentTool) -> list[Any]:
    tools: list[Any] = []
    if getattr(helper.config, "task_steps_manager_enabled", True):
        tools.append(get_task_steps_manager_tool(TaskStepsManagerTool(helper.config)))
    tools.extend(
        [
            get_fetch_url_text_tool(helper.open),
            get_web_archive_search_tool(helper.open),
            get_wayback_fetch_tool(helper.open),
            get_gdelt_news_search_tool(helper.open),
        ]
    )
    if has_serpapi_keys(__import__("os").environ.get("SERPAPI_API_KEY", "")):
        tools.extend([get_google_web_search_tool(helper.web), get_bing_web_search_tool(helper.web)])
    add_research_artifact_tools(tools, helper.config)
    return tools


def _legal_tools(helper: OpenSpecialistResearchAgentTool) -> list[Any]:
    tools = _base_tools(helper)
    tools.extend(
        [
            get_boe_law_search_tool(helper.open),
            get_boe_law_metadata_tool(helper.open),
            get_boe_law_text_download_tool(helper.open),
            get_boe_aux_table_tool(helper.open),
            get_federal_register_search_tool(helper.open),
            get_wikidata_entity_search_tool(helper.open),
            get_wikidata_sparql_tool(helper.open),
        ]
    )
    if has_serpapi_keys(__import__("os").environ.get("SERPAPI_API_KEY", "")):
        tools.append(get_google_news_search_tool(helper.forum))
    return tools


def _data_statistics_tools(helper: OpenSpecialistResearchAgentTool) -> list[Any]:
    tools = _base_tools(helper)
    tools.append(get_controlled_shell_command_tool(ExecTool(helper.config)))
    tools.extend(
        [
            get_world_bank_indicator_tool(helper.open),
            get_wikidata_entity_search_tool(helper.open),
            get_wikidata_sparql_tool(helper.open),
        ]
    )
    return tools


def _news_media_tools(helper: OpenSpecialistResearchAgentTool) -> list[Any]:
    tools = _base_tools(helper)
    if has_serpapi_keys(__import__("os").environ.get("SERPAPI_API_KEY", "")):
        tools.extend(
            [
                get_google_news_search_tool(helper.forum),
                get_google_trends_search_tool(helper.forum),
                get_google_trends_trending_now_tool(helper.forum),
                get_google_videos_search_tool(helper.forum),
            ]
        )
    return tools


def _knowledge_graph_tools(helper: OpenSpecialistResearchAgentTool) -> list[Any]:
    tools = _base_tools(helper)
    tools.extend([get_wikidata_entity_search_tool(helper.open), get_wikidata_sparql_tool(helper.open)])
    return tools


def _religious_tools(helper: OpenSpecialistResearchAgentTool) -> list[Any]:
    tools = _base_tools(helper)
    tools.extend(
        [
            get_bible_passage_tool(helper.open),
            get_sefaria_search_tool(helper.open),
            get_sefaria_text_tool(helper.open),
            get_quran_search_tool(helper.open),
            get_quran_verse_tool(helper.open),
            get_quran_chapters_tool(helper.open),
            get_gita_chapters_tool(helper.open),
            get_gita_chapter_tool(helper.open),
            get_gita_verse_tool(helper.open),
            get_hadith_editions_tool(helper.open),
            get_hadith_search_tool(helper.open),
            get_hadith_collection_tool(helper.open),
            get_hadith_section_tool(helper.open),
            get_suttacentral_suttaplex_tool(helper.open),
            get_suttacentral_text_tool(helper.open),
            get_wikidata_entity_search_tool(helper.open),
            get_wikidata_sparql_tool(helper.open),
        ]
    )
    return tools


def _make_agent_tool(helper: OpenSpecialistResearchAgentTool, name: str, description: str):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    def research(prompt: str | list[str], save_artifacts: bool = False) -> str:
        tool_input = {"prompt": prompt, "save_artifacts": save_artifacts}
        try:
            return run_with_tool_logging(
                name,
                tool_input,
                lambda: _run_and_record_researcher_response(
                    name,
                    helper.run(prompt=prompt, save_artifacts=save_artifacts),
                ),
            )
        except Exception as exc:
            return f"ERROR: {name} failed ({exc})"

    full_description = f"""{description}

Use this tool to launch a dedicated researcher without consuming the parent agent's context.
Provide a detailed prompt with scope, entities, jurisdictions or platforms when relevant, date ranges, source preferences, evidence to collect, expected output, and caveats to check.

Args:
    prompt: Detailed research instructions as a string, or a list of up to 3 detailed prompts. Each prompt should be specific enough for an autonomous researcher to gather evidence, preserve useful content artifacts during the run, and return the configured JSON result.
    save_artifacts: If true, preserve the evidence folder after the run and return it in the JSON result. If false, artifacts are temporary and deleted after the run.

Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final specialized review, and artifact folder path only when artifacts are preserved.
"""
    research.__doc__ = full_description
    tool = function_tool(research, name_override=name, description_override=full_description)
    return enforce_prompt_str_or_list_schema(tool)


def _run_and_record_researcher_response(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output


def get_legal_research_tool(helper: OpenSpecialistResearchAgentTool):
    return _make_agent_tool(
        helper,
        "legal_research",
        "Run a dedicated legal/regulatory research sub-agent for statutes, regulations, notices, enforcement records, official legal text, jurisdictional provenance, and legal timelines.",
    )


def get_data_statistics_research_tool(helper: OpenSpecialistResearchAgentTool):
    return _make_agent_tool(
        helper,
        "data_statistics_research",
        "Run a dedicated data/statistics research sub-agent for datasets, indicators, time series, raw JSON/CSV-like evidence, units, methods, entity IDs, and reproducible checks.",
    )


def get_news_media_research_tool(helper: OpenSpecialistResearchAgentTool):
    return _make_agent_tool(
        helper,
        "news_media_research",
        "Run a dedicated news/media-intelligence research sub-agent for coverage timelines, source domains, story clusters, media/video evidence, trend signals, archived pages, and original URLs.",
    )


def get_knowledge_graph_research_tool(helper: OpenSpecialistResearchAgentTool):
    return _make_agent_tool(
        helper,
        "knowledge_graph_research",
        "Run a dedicated knowledge-graph/entity research sub-agent for entity resolution, identifiers, aliases, relationships, official URLs, registry IDs, graph claims, and provenance.",
    )


def get_religious_research_tool(helper: OpenSpecialistResearchAgentTool):
    return _make_agent_tool(
        helper,
        "religious_research",
        "Run a dedicated religious-text research sub-agent for scriptures, translations, commentaries, references, editions, provenance, languages, and comparative religious context.",
    )


def build_legal_agent(config: ToolsConfig, **kwargs: Any) -> OpenSpecialistResearchAgentTool:
    return OpenSpecialistResearchAgentTool(
        config,
        kind="legal",
        system_prompt=_LEGAL_PROMPT,
        build_tools=_legal_tools,
        max_tools_attr="legal_max_tools_used",
        **kwargs,
    )


def build_data_statistics_agent(config: ToolsConfig, **kwargs: Any) -> OpenSpecialistResearchAgentTool:
    return OpenSpecialistResearchAgentTool(
        config,
        kind="data_statistics",
        system_prompt=_DATA_STATS_PROMPT,
        build_tools=_data_statistics_tools,
        max_tools_attr="data_statistics_max_tools_used",
        **kwargs,
    )


def build_news_media_agent(config: ToolsConfig, **kwargs: Any) -> OpenSpecialistResearchAgentTool:
    return OpenSpecialistResearchAgentTool(
        config,
        kind="news_media",
        system_prompt=_NEWS_MEDIA_PROMPT,
        build_tools=_news_media_tools,
        max_tools_attr="news_media_max_tools_used",
        **kwargs,
    )


def build_knowledge_graph_agent(config: ToolsConfig, **kwargs: Any) -> OpenSpecialistResearchAgentTool:
    return OpenSpecialistResearchAgentTool(
        config,
        kind="knowledge_graph",
        system_prompt=_KG_ENTITY_PROMPT,
        build_tools=_knowledge_graph_tools,
        max_tools_attr="knowledge_graph_max_tools_used",
        **kwargs,
    )


def build_religious_agent(config: ToolsConfig, **kwargs: Any) -> OpenSpecialistResearchAgentTool:
    return OpenSpecialistResearchAgentTool(
        config,
        kind="religious",
        system_prompt=_RELIGIOUS_PROMPT,
        build_tools=_religious_tools,
        max_tools_attr="religious_max_tools_used",
        **kwargs,
    )
