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
from .serpapi_keys import has_serpapi_keys
from .task_list_tool import TaskListTool, get_task_list_tool
from .subagent_config import build_subagent_config
from .task_list_state import current_session_id

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
        max_turns: int = 30,
    ):
        self.config = config
        self.model_name = model_name
        self.fallback_model = fallback_model
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
        
        task_helper = TaskListTool(self.config)
        
        tools = [get_task_list_tool(task_helper)]

        if self.config.tester_exec_enabled:
            tools.append(get_exec_tool(self.exec))

        if self.config.tester_brave_enabled:
            tools.append(get_brave_search_tool(self.brave))

        has_serpapi = has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if has_serpapi and self.config.tester_google_web_enabled:
            tools.append(get_google_web_search_tool(self.web))
            
        return tools

    def run(self, prompt: str) -> str:
        # Check if at least one critical tool is available
        exec_allowed = self.config.tester_exec_enabled
        brave_allowed = self.config.tester_brave_enabled
        google_allowed = self.config.tester_google_web_enabled and has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        
        # We need at least exec or search to be useful
        if not (exec_allowed or brave_allowed or google_allowed):
             return "ERROR: Tester agent requires at least Exec, Brave, or Google enabled."

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
                "tester_exec_enabled": self.config.tester_exec_enabled,
                "tester_brave_enabled": self.config.tester_brave_enabled,
                "tester_google_web_enabled": self.config.tester_google_web_enabled,
                "exec_enabled": self.config.tester_exec_enabled, # Subagent needs this flag true to use tool if passed, but we pass concrete tools override.
                "brave_enabled": self.config.brave_enabled and self.config.tester_brave_enabled,
                "serpapi_google_web_enabled": self.config.serpapi_google_web_enabled and self.config.tester_google_web_enabled,
                
                # Disable others
                "websearcher_enabled": False,
                "scientific_enabled": False,
                "social_network_enabled": False,
                "pdf_text_enabled": False,
            },
        }
        config = build_subagent_config(
            self.config,
            model_name=model_name,
            max_turns=self.max_turns,
            system_prompt=_TESTER_AGENT_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_session_id = current_session_id()
        
        # Avoid circular import at module level
        from chack_agent import Chack
        
        chack = Chack(config)
        result = chack.run(
            session_id=f"tester:{int(time.time() * 1000)}",
            text=prompt,
            min_tools_used_override=0,
            max_tools_used_override=self.config.tester_max_tools_used,
            enable_self_critique=None,
            require_task_list_init_first=True,
            tools_override=tools,
            system_prompt_override=config.system_prompt,
            usage_session_id=parent_session_id,
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
            return helper.run(prompt=prompt)
        except Exception as exc:
            return f"ERROR: tester_agent failed ({exc})"

    return tester_agent
