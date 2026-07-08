import os
import time
from typing import Any, Optional

from .brave_search import BraveSearchTool, get_brave_search_tool
from .config import ToolsConfig
from .serpapi_web_search import (
    SerpApiWebSearchTool,
    get_google_web_search_tool,
)
from .exec_tool import ExecTool, get_controlled_shell_command_tool
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


_CLI_RESEARCH_SYSTEM_PROMPT = """### RULES
- Your only job is CLI-backed empirical research: code behavior, command-line checks, math assumptions, local system checks, docs/known-issue lookups, and execution-backed findings.
- Use all relevant CLI, execution, and web tools repeatedly until confidence is strong. Prefer running scripts/commands over reasoning from memory; use web search for docs/examples when failures need context.
- A successful CLI research run must use command execution for the core claim whenever the task can be checked locally or through public CLI/HTTP commands. If you cannot run commands, return research_worked=false and explain why; do not replace CLI evidence with only web-search evidence.
- For code, create temporary files if useful, run them, analyze output, and clean them up when appropriate. For math, write small scripts for nontrivial calculations.
- Synchronous CLI commands have hard timeouts. For long-running jobs, start them in the background with explicit log/output files, then monitor them with quick commands such as ps, tail, grep, cat, or checking exit/status files.
- Summarize actual execution results, failures, uncertainty, and useful artifact filenames only when artifacts are preserved.
- IMPORTANT: NEVER UNDER ANY CONCEPT EXECUTE ANY POTENTIALLY DANGEROUS PROGRAM (MALWARE, VIRUS, C2, REV SHELL) UNDER ANY CIRCUNSTANCES
""" + OBJECTIVE_EVIDENCE_RULES


class CliResearchAgentTool:
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
        self.exec = ExecTool(config)

    def _max_tools_used(self) -> int:
        return int(getattr(self.config, "cli_max_tools_used", 0) or 0)

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
        # CLI researcher always has execution and web-search capabilities.
        tools.append(get_controlled_shell_command_tool(self.exec))
        tools.append(get_brave_search_tool(self.brave))
        tools.append(get_google_web_search_tool(self.web))
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
        evidence_dir = create_subagent_evidence_dir("cli", parent_root_session_id)
        prompt = append_evidence_dir_instruction(
            prompt,
            evidence_dir,
            "Now start the CLI-backed research/verification.",
            save_artifacts=save_artifacts,
        )
        max_tools_used = self._max_tools_used()
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
                "max_tools_used": max_tools_used,
                "cli_enabled": True,
                "cli_exec_enabled": True,
                "cli_brave_enabled": True,
                "cli_google_web_enabled": True,
                "exec_enabled": True, # Sub-agent uses explicit tools_override, keep flags aligned.
                "brave_enabled": True,
                "serpapi_google_web_enabled": True,
                
                # Disable others
                "websearcher_enabled": False,
                "scientific_enabled": False,
                "social_network_enabled": False,
                "pdf_text_enabled": False,
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
        overrides["agent"]["sub_action"] = "cli"
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=_CLI_RESEARCH_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("cli", parent_root_session_id)
        
        # Avoid circular import at module level
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
                max_tools_used_override=max_tools_used,
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
                    max_tools_used_override=max_tools_used,
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


def get_cli_research_tool(
    helper: CliResearchAgentTool,
):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="cli_research")
    def cli_research(prompt: str | list[str], save_artifacts: bool = False) -> str:
        """Run a specialized CLI researcher to verify assumptions, run scripts, or check math.

        Use this researcher when you need to:
        1. Run local code to verify functionality.
        2. Create small scripts to test logic.
        3. Search the web for documentation to fix a script.
        4. Verify a complex math problem by running a python script.

        It's highly recommended to use this tool to do long tasks or actions getting the results directly instead of checking them yourself to not compromise your agent's context.
        
        Be specific about topic, scope, constraints, and expected results and data inside the output.

        You can specify up to 3 prompts for CLI researchers, and 1 agent in parallel will be launched per prompt given.

        Args:
            prompt: Detailed instructions (string) or a list of up to 3 detailed instructions. Each request must be at least 500 characters indicating all the details of the goals and objetives of the subagent, suggested process to obtain proper results, example expected output or relevant information to gather... the more detailed is each instruction to the sub agent, the better.
            save_artifacts: If true, preserve the evidence folder after the run and return it in the JSON result. If false, artifacts are temporary and deleted after the run.

        Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final CLI verification review, and artifact folder path only when artifacts are preserved.
        """
        try:
            return run_with_tool_logging(
                "cli_research",
                {"prompt": prompt, "save_artifacts": save_artifacts},
                lambda: _run_and_record_researcher_response(
                    "cli_research",
                    helper.run(prompt=prompt, save_artifacts=save_artifacts),
                ),
            )
        except Exception as exc:
            return f"ERROR: cli_research failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(cli_research)
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: Provide prompt as one detailed CLI verification request or up to 3 detailed requests; set save_artifacts true only when the evidence folder must be preserved.\n"
        "Output: Returns the researcher's JSON result with worked status, failure reason when relevant, final CLI verification review, and artifact folder path only when artifacts are preserved."
    )
    return tool


def _run_and_record_researcher_response(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output
