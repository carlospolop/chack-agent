import os
import time
from typing import Optional

from .brave_search import BraveSearchTool, get_brave_search_tool
from .config import ToolsConfig
from .serpapi_web_search import (
    SerpApiWebSearchTool,
    get_google_web_search_tool,
)
from .exec_tool import ExecTool, get_exec_tool
from .task_steps_manager_tool import TaskStepsManagerTool, get_task_steps_manager_tool
from .subagent_config import build_subagent_config
from .task_steps_manager_state import current_session_id
from .telemetry import current_log_context, run_with_tool_logging

try:
    from agents import function_tool
except ImportError:
    function_tool = None


_TESTER_AGENT_SYSTEM_PROMPT = """### RULES
- You are a specialized testing agent designed to verify code, math assumptions, and perform local system checks.
- Use the `exec` tool heavily to run scripts (python, bash, etc.) locally to verify behavior.
- Use web search (Brave/Google) to find documentation, known issues, or examples if a test fails or you need more context.
- Your primary goal is EMPIRICAL VERIFICATION. Do not assume; run it and see.
- If testing code:
    - Create temporary files if needed using `exec` (e.g. `echo "..." > test.py`).
    - Run them.
    - Analyze the output.
    - Clean up temporary files if appropriate (or leave them if useful for debugging).
- If checking math:
    - Write a small script to compute the result.
    - Don't just rely on your own internal training for complex math.
- Provide a summary of your findings based on the actual execution results.
- Do not ask the user questions, just proceed with the best testing strategy.
"""


class TesterAgentTool:
    def __init__(
        self,
        config: ToolsConfig,
        model_name: str = "",
        fallback_model: str = "",
        model_provider: str = "",
        max_turns: int = 30,
    ):
        self.config = config
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.max_turns = max(2, int(max_turns or 30))
        self.brave = BraveSearchTool(config)
        self.web = SerpApiWebSearchTool(config)
        self.exec = ExecTool(config)

    def _resolved_model(self) -> Optional[str]:
        configured = (self.model_name or "").strip()
        if configured:
            return configured
        fallback = (self.fallback_model or "").strip()
        return fallback or None

    def _build_subagent_tools(self):
        if function_tool is None:
            raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")
        
        task_helper = TaskStepsManagerTool(self.config)
        
        tools = [get_task_steps_manager_tool(task_helper)]
        # Tester sub-agent always has execution and web-search capabilities.
        tools.append(get_exec_tool(self.exec))
        tools.append(get_brave_search_tool(self.brave))
        tools.append(get_google_web_search_tool(self.web))

        return tools

    def run(self, prompt: str) -> str:
        if not prompt.strip():
            return "ERROR: prompt cannot be empty"

        prompt = f"{prompt.rstrip()}\n\nNow start the testing/verification."
        tools = self._build_subagent_tools()
        model_name = self._resolved_model() or ""
        overrides = {
            "agent": {"self_critique_enabled": False},
            "session": {
                "max_turns": self.max_turns,
                "memory_max_messages": 8,
                "memory_reset_to_messages": 8,
                "long_term_memory_enabled": False,
                "long_term_memory_max_chars": 0,
                "long_term_memory_dir": "",
            },
            "tools": {
                "max_tools_used": self.config.tester_max_tools_used,
                "tester_enabled": True,
                "tester_exec_enabled": True,
                "tester_brave_enabled": True,
                "tester_google_web_enabled": True,
                "exec_enabled": True, # Sub-agent uses explicit tools_override, keep flags aligned.
                "brave_enabled": True,
                "serpapi_google_web_enabled": True,
                
                # Disable others
                "websearcher_enabled": False,
                "scientific_enabled": False,
                "social_network_enabled": False,
                "pdf_text_enabled": False,
            },
        }
        ctx = current_log_context()
        main_action = str(ctx.get("main_action") or "").strip()
        if main_action:
            overrides["agent"]["main_action"] = main_action
        overrides["agent"]["sub_action"] = "tester"
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            model_provider=self.model_provider,
            max_turns=self.max_turns,
            system_prompt=_TESTER_AGENT_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        parent_root_session_id = str(ctx.get("session_id") or "").strip()
        subagent_session_id = parent_root_session_id or f"tester:{int(time.time() * 1000)}"
        
        # Avoid circular import at module level
        from chack_agent import Chack
        
        chack = Chack(config)
        result = chack.run(
            session_id=subagent_session_id,
            text=prompt,
            min_tools_used_override=0,
            max_tools_used_override=self.config.tester_max_tools_used,
            enable_self_critique=None,
            require_task_steps_manager_init_first=True,
            tools_override=tools,
            system_prompt_override=config.system_prompt,
            usage_session_id=parent_task_session_id,
        )
        return result.output.strip() if result.output else "ERROR: sub-agent returned an empty response."


def get_tester_agent_tool(
    helper: TesterAgentTool,
):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="tester_agent")
    def tester_agent(prompt: str) -> str:
        """Run a specialized testing agent to verify assumptions, run scripts, or check math.

        Use this agent when you need to:
        1. Run local code to verify functionality.
        2. Create small scripts to test logic.
        3. Search the web for documentation to fix a script.
        4. Verify a complex math problem by running a python script.

        Args:
            prompt: Detailed instructions for what to test or verify. Include any code snippets or specific command requirements if known.
        """
        try:
            return run_with_tool_logging(
                "tester_agent",
                {"prompt": prompt},
                lambda: helper.run(prompt=prompt),
            )
        except Exception as exc:
            return f"ERROR: tester_agent failed ({exc})"

    return tester_agent
