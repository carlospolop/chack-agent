import os
import time
from typing import Any, Optional

from .config import ToolsConfig
from .pdf_text import PdfTextTool, get_pdf_text_tool
from .open_research_sources import (
    OpenResearchTool,
    get_biorxiv_download_tool,
    get_biorxiv_search_tool,
    get_clinicaltrial_get_tool,
    get_clinicaltrials_search_tool,
    get_crossref_doi_lookup_tool,
    get_crossref_search_tool,
    get_pubchem_search_tool,
    get_retraction_watch_tool,
)
from .scientific_search import (
    ScientificSearchTool,
    get_arxiv_search_tool,
    get_europe_pmc_search_tool,
    get_pmc_full_text_search_tool,
    get_pmc_full_text_download_tool,
    get_ncbi_bookshelf_search_tool,
    get_ncbi_bookshelf_download_tool,
    get_semantic_scholar_search_tool,
    get_openalex_search_tool,
    get_plos_search_tool,
    get_google_patents_search_tool,
    get_google_patents_details_tool,
    get_google_scholar_search_tool,
    get_google_scholar_cite_tool,
    get_youtube_video_search_tool,
    get_youtube_video_details_tool,
    get_youtube_transcript_tool,
    get_medrxiv_preprint_search_tool,
    get_medrxiv_full_text_download_tool,
)
from .task_steps_manager_tool import TaskStepsManagerTool, get_task_steps_manager_tool
from .exec_tool import ExecTool, get_controlled_shell_command_tool
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


_SCIENTIFIC_AGENT_SYSTEM_PROMPT = """### RULES
- Your only job is scientific research: papers, preprints, books, trials, DOI metadata, retractions/updates, chemicals/entities, patents/video when relevant, and concise evidence-backed findings.
- Use all relevant scientific tools repeatedly until coverage is strong. Prefer accessible full text; do not stop at titles, abstracts, snippets, or citation counts when full content can be fetched.
- Use Crossref for DOI/provenance/retraction signals, clinical-trial tools for studies, chemistry/entity tools when relevant, and patent/video tools only when they add evidence.
- Download available full text/PDFs and raw API JSON before analysis; use PDF text and exec/curl/wget/grep-style checks when needed to read or extract paper content.
- Mention sources, and mention artifact filenames only when artifacts are preserved, without naming internal tool names.
- IMPORTANT: NEVER UNDER ANY CONCEPT EXECUTE ANY POTENTIALLY DANGEROUS PROGRAM (MALWARE, VIRUS, C2, REV SHELL) UNDER ANY CIRCUNSTANCES
""" + OBJECTIVE_EVIDENCE_RULES


class ScientificResearchAgentTool:
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
        self.search = ScientificSearchTool(config)
        self.pdf = PdfTextTool(config)
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

        search = self.search
        pdf = self.pdf
        exec_helper = ExecTool(self.config)

        tools = []
        if getattr(self.config, "task_steps_manager_enabled", True):
            task_steps_manager_helper = TaskStepsManagerTool(self.config)
            tools.append(get_task_steps_manager_tool(task_steps_manager_helper))
        # Scientific sub-agent always has the full scientific toolset.
        tools.append(get_arxiv_search_tool(search))
        tools.append(get_europe_pmc_search_tool(search))
        tools.append(get_pmc_full_text_search_tool(search))
        tools.append(get_pmc_full_text_download_tool(search))
        tools.append(get_ncbi_bookshelf_search_tool(search))
        tools.append(get_ncbi_bookshelf_download_tool(search))
        tools.append(get_semantic_scholar_search_tool(search))
        tools.append(get_openalex_search_tool(search))
        tools.append(get_plos_search_tool(search))
        tools.append(get_google_patents_search_tool(search))
        tools.append(get_google_patents_details_tool(search))
        tools.append(get_google_scholar_search_tool(search))
        tools.append(get_google_scholar_cite_tool(search))
        tools.append(get_youtube_video_search_tool(search))
        tools.append(get_youtube_video_details_tool(search))
        tools.append(get_youtube_transcript_tool(search))
        tools.append(get_medrxiv_preprint_search_tool(search))
        tools.append(get_medrxiv_full_text_download_tool(search))
        tools.append(get_crossref_search_tool(self.open))
        tools.append(get_crossref_doi_lookup_tool(self.open))
        tools.append(get_clinicaltrials_search_tool(self.open))
        tools.append(get_clinicaltrial_get_tool(self.open))
        tools.append(get_biorxiv_search_tool(self.open))
        tools.append(get_biorxiv_download_tool(self.open))
        tools.append(get_retraction_watch_tool(self.open))
        tools.append(get_pubchem_search_tool(self.open))
        tools.append(get_pdf_text_tool(pdf))
        tools.append(get_controlled_shell_command_tool(exec_helper))
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
        evidence_dir = create_subagent_evidence_dir("scientific", parent_root_session_id)
        prompt = append_evidence_dir_instruction(
            prompt,
            evidence_dir,
            "Now start the research checking all the scientific research tools given!",
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
                "exec_enabled": True,
                "pdf_text_enabled": True,
                "scientific_enabled": True,
                "max_tools_used": self.config.scientific_max_tools_used,
                "scientific_arxiv_enabled": True,
                "scientific_europe_pmc_enabled": True,
                "scientific_pmc_full_text_enabled": True,
                "scientific_ncbi_bookshelf_enabled": True,
                "scientific_semantic_scholar_enabled": True,
                "scientific_openalex_enabled": True,
                "scientific_plos_enabled": True,
                "scientific_google_patents_enabled": True,
                "scientific_google_patents_details_enabled": True,
                "scientific_google_scholar_enabled": True,
                "scientific_google_scholar_cite_enabled": True,
                "scientific_youtube_search_enabled": True,
                "scientific_youtube_details_enabled": True,
                "scientific_youtube_transcript_enabled": True,
                "scientific_medrxiv_enabled": True,
                "scientific_crossref_enabled": True,
                "scientific_clinicaltrials_enabled": True,
                "scientific_biorxiv_enabled": True,
                "scientific_retraction_watch_enabled": True,
                "scientific_pubchem_enabled": True,
                "scientific_pdf_text_enabled": True,
                "scientific_exec_enabled": True,
                "brave_enabled": False,
                "serpapi_google_web_enabled": False,
                "serpapi_bing_web_enabled": False,
                "websearcher_enabled": False,
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
        overrides["agent"]["sub_action"] = "scientific"
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=_SCIENTIFIC_AGENT_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("scientific", parent_root_session_id)
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
                max_tools_used_override=self.config.scientific_max_tools_used,
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
                    max_tools_used_override=self.config.scientific_max_tools_used,
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


def get_scientific_research_tool(
    helper: ScientificResearchAgentTool,
):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="scientific_research")
    def scientific_research(prompt: str | list[str], save_artifacts: bool = False) -> str:
        """Run a dedicated scientific-research sub-agent.

        Use this tool to launch an autonomous scientific research agent to do specific complex researches for you.
        It's highly recommended to use this tool to do in depth reviews of scientific topics and get the results instead of checking them yourself to not compromise your agent's context.
        
        Be specific about topic, scope, constraints, and expected results and data inside the output.

        You can specify up to 3 prompts for the scientific resaerchers, and 1 agent in parallel will be launched per prompt given.

        Args:
            prompt: A detailed research request (string) or a list of up to 3 detailed requests. Each request must be at least 500 characters indicating all the details of the goals and objetives of the subagent, suggested process to obtain proper results, example expected output or relevant information to gather... the more detailed is each instruction to the sub agent, the better.
            save_artifacts: If true, preserve the evidence folder after the run and return it in the JSON result. If false, artifacts are temporary and deleted after the run.

        Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final scientific review, and artifact folder path only when artifacts are preserved.
        """
        tool_input = {"prompt": prompt, "save_artifacts": save_artifacts}
        try:
            return run_with_tool_logging(
                "scientific_research",
                tool_input,
                lambda: _run_and_record_researcher_response(
                    "scientific_research",
                    helper.run(prompt=prompt, save_artifacts=save_artifacts),
                ),
            )
        except Exception as exc:
            return f"ERROR: scientific_research failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(scientific_research)
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: Provide prompt as one detailed scientific request or up to 3 detailed requests; set save_artifacts true only when the evidence folder must be preserved.\n"
        "Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final scientific review, and artifact folder path only when artifacts are preserved."
    )
    return tool


def _run_and_record_researcher_response(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output
