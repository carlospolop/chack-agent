import json
import os
import time
from typing import Any, Optional

from .config import ToolsConfig
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
from .research_artifacts import add_research_artifact_tools, cleanup_research_artifacts, reset_research_artifact_context, set_research_artifact_context
from .telemetry import current_log_context, run_with_tool_logging

try:
    from agents import function_tool
except ImportError:
    function_tool = None


_SUBCHACK_AGENT_SYSTEM_PROMPT = """### RULES
- You are a delegated autonomous sub-agent.
- Use all available relevant tools repeatedly to gather strong evidence and complete the task.
- You have the same tools as your parent agent, except you cannot call subchack_researcher recursively.
- Execute the best strategy with available tools and report only supported findings, gaps, uncertainty, and artifact filenames only when artifacts are preserved.
""" + OBJECTIVE_EVIDENCE_RULES


class SubChackResearchAgentTool:
    def __init__(
        self,
        config: ToolsConfig,
        model_name: str = "",
        fallback_model: str = "",
        model_provider: str = "",
        max_turns: int = 30,
        social_network_model: str = "",
        scientific_model: str = "",
        websearcher_model: str = "",
        business_model: str = "",
        product_model: str = "",
        legal_model: str = "",
        data_statistics_model: str = "",
        news_media_model: str = "",
        knowledge_graph_model: str = "",
        religious_model: str = "",
        cli_model: str = "",
        social_network_max_turns: int = 30,
        scientific_max_turns: int = 30,
        websearcher_max_turns: int = 30,
        business_max_turns: int = 30,
        product_max_turns: int = 30,
        legal_max_turns: int = 30,
        data_statistics_max_turns: int = 30,
        news_media_max_turns: int = 30,
        knowledge_graph_max_turns: int = 30,
        religious_max_turns: int = 30,
        cli_max_turns: int = 30,
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
        self.social_network_model = social_network_model
        self.scientific_model = scientific_model
        self.websearcher_model = websearcher_model
        self.business_model = business_model
        self.product_model = product_model
        self.legal_model = legal_model
        self.data_statistics_model = data_statistics_model
        self.news_media_model = news_media_model
        self.knowledge_graph_model = knowledge_graph_model
        self.religious_model = religious_model
        self.cli_model = cli_model
        self.social_network_max_turns = max(2, int(social_network_max_turns or 30))
        self.scientific_max_turns = max(2, int(scientific_max_turns or 30))
        self.websearcher_max_turns = max(2, int(websearcher_max_turns or 30))
        self.business_max_turns = max(2, int(business_max_turns or 30))
        self.product_max_turns = max(2, int(product_max_turns or 30))
        self.legal_max_turns = max(2, int(legal_max_turns or 30))
        self.data_statistics_max_turns = max(2, int(data_statistics_max_turns or 30))
        self.news_media_max_turns = max(2, int(news_media_max_turns or 30))
        self.knowledge_graph_max_turns = max(2, int(knowledge_graph_max_turns or 30))
        self.religious_max_turns = max(2, int(religious_max_turns or 30))
        self.cli_max_turns = max(2, int(cli_max_turns or 30))

    def _resolved_model(self) -> Optional[str]:
        configured = (self.model_name or "").strip()
        if configured:
            return configured
        fallback = (self.fallback_model or "").strip()
        return fallback or None

    @staticmethod
    def _name_of_tool(tool) -> str:
        return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()

    def _allowed_tool_names_from_context(self, ctx: Optional[dict[str, Any]] = None) -> Optional[set[str]]:
        allowed_tools_raw = str(os.environ.get("CHACK_ALLOWED_TOOLS_JSON", "") or "").strip()
        if allowed_tools_raw:
            try:
                parsed = json.loads(allowed_tools_raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                names = {str(item).strip().lower() for item in parsed if str(item).strip()}
                if names:
                    return names

        effective_ctx = ctx or current_log_context()
        raw = effective_ctx.get("available_tool_names") or effective_ctx.get("available_tools")
        if isinstance(raw, list):
            names = {str(item).strip().lower() for item in raw if str(item).strip()}
            if names:
                return names
        return None

    def _build_subagent_tools(self, ctx: Optional[dict[str, Any]] = None):
        if function_tool is None:
            raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")

        # Local import to avoid circular import from agents_toolset -> this module.
        from .agents_toolset import AgentsToolset

        toolset = AgentsToolset(
            self.config,
            model_provider=self.model_provider,
            default_model=self.fallback_model,
            social_network_model=self.social_network_model,
            scientific_model=self.scientific_model,
            websearcher_model=self.websearcher_model,
            business_model=self.business_model,
            product_model=self.product_model,
            legal_model=self.legal_model,
            data_statistics_model=self.data_statistics_model,
            news_media_model=self.news_media_model,
            knowledge_graph_model=self.knowledge_graph_model,
            religious_model=self.religious_model,
            cli_model=self.cli_model,
            subchack_model=self.model_name,
            social_network_max_turns=self.social_network_max_turns,
            scientific_max_turns=self.scientific_max_turns,
            websearcher_max_turns=self.websearcher_max_turns,
            business_max_turns=self.business_max_turns,
            product_max_turns=self.product_max_turns,
            legal_max_turns=self.legal_max_turns,
            data_statistics_max_turns=self.data_statistics_max_turns,
            news_media_max_turns=self.news_media_max_turns,
            knowledge_graph_max_turns=self.knowledge_graph_max_turns,
            religious_max_turns=self.religious_max_turns,
            cli_max_turns=self.cli_max_turns,
            subchack_max_turns=self.max_turns,
            self_critique_enabled=self.self_critique_enabled,
            self_critique_rounds=self.self_critique_rounds,
        )
        tools = list(getattr(toolset, "tools", []) or [])
        add_research_artifact_tools(tools, self.config)

        allowed_names = self._allowed_tool_names_from_context(ctx=ctx)
        if allowed_names is not None:
            tools = [
                tool for tool in tools
                if self._name_of_tool(tool).lower() in allowed_names
            ]

        # Prevent recursive self-invocation.
        tools = [tool for tool in tools if self._name_of_tool(tool) != "subchack_researcher"]
        return tools

    def _run_single(self, prompt: str, ctx: dict[str, Any], save_artifacts: bool = False) -> str:
        tools = self._build_subagent_tools(ctx=ctx)
        if not tools:
            return "ERROR: no tools available for subchack_researcher."

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
        evidence_dir = create_subagent_evidence_dir("subchack", parent_root_session_id)
        prompt = append_evidence_dir_instruction(
            prompt,
            evidence_dir,
            "Now start the delegated research using your available tools.",
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
                "long_term_memory_enabled": False
            },
            "tools": {
                "subchack_enabled": True,
                "max_tools_used": self.config.subchack_max_tools_used,
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
        overrides["agent"]["sub_action"] = "subchack"
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=_SUBCHACK_AGENT_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("subchack", parent_root_session_id)

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
                max_tools_used_override=self.config.subchack_max_tools_used,
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
                    max_tools_used_override=self.config.subchack_max_tools_used,
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


def get_subchack_research_tool(
    helper: SubChackResearchAgentTool,
):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="subchack_researcher")
    def subchack_researcher(prompt: str | list[str], save_artifacts: bool = False) -> str:
        """Run a delegated sub-agent with the parent's tool access.
        If you have access to this tool it means that you are a master agent.
        Use this tool to call subagents to perform specific complex operations or researches giving you back the responses so you don't lose the focus from the biggest task.

        You must use this tool to:
        - Run several steps in parallel with subagents with access to all the tools (but this one to avoid loops)
        - Don't lose the focus from the given big task asking independants agents to perform differnt steps
        - Be able to obtain all the information and check everything performed by subagent to have all under control ane be 1000% sure of the final result
        
        You can specify up to 3 prompts for the scientific resaerchers, and 1 agent in parallel will be launched per prompt given.

        Args:
            prompt: Detailed delegated task (string) or a list of up to 3 detailed tasks. Each request must be at least 500 characters indicating all the details of the goals and objetives of the subagent, suggested process to obtain proper results, example expected output or relevant information to gather... the more detailed is each instruction to the sub agent, the better.
            save_artifacts: If true, preserve the evidence folder after the run and return it in the JSON result. If false, artifacts are temporary and deleted after the run.

        Output: Returns the delegated agent's JSON/text result, including worked status and artifact path when the delegated tool preserves artifacts.
        """
        try:
            return run_with_tool_logging(
                "subchack_researcher",
                {"prompt": prompt, "save_artifacts": save_artifacts},
                lambda: _run_and_record_researcher_response(
                    "subchack_researcher",
                    helper.run(prompt=prompt, save_artifacts=save_artifacts),
                ),
            )
        except Exception as exc:
            return f"ERROR: subchack_researcher failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(subchack_researcher)
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: Provide prompt as one detailed delegated task or up to 3 detailed tasks; set save_artifacts true only when the delegated evidence folder must be preserved.\n"
        "Output: Returns the delegated agent's JSON/text result, including worked status and artifact path when the delegated tool preserves artifacts."
    )
    return tool


def _run_and_record_researcher_response(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output
