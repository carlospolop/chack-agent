"""Integration tests for budget warning injection across all backends.

These tests verify that the actual code paths used by subprocess backends
(Claude Code, Codex, Copilot) will correctly:
1. Include budget warnings in the prompt sent to the subprocess
2. Propagate budget env vars to MCP subprocesses
3. Register and serve the check_budget_status MCP tool
4. Inject warnings into MCP tool results
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from chack_agent.budget_warning_state import (
    BUDGET_ENV_KEYS,
    budget_prompt_warning,
    budget_status_from_env,
    export_budget_env,
    export_spent_usd_env,
    inject_budget_warning_from_env,
)


@pytest.fixture(autouse=True)
def _clean_budget_env():
    """Remove budget env vars after each test."""
    yield
    for key in BUDGET_ENV_KEYS:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# 1. Prompt injection: budget warnings appended to user prompt
# ---------------------------------------------------------------------------

class TestPromptInjection:
    """Verify budget_prompt_warning() produces text that would be appended
    to the user prompt before executor.invoke()."""

    def test_warning_at_70_percent_runtime(self):
        """At 70% runtime, warning text is generated."""
        w = budget_prompt_warning(
            start_epoch=time.time() - 42,  # 42s of 60s = 70%
            max_runtime_seconds=60,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        assert w, "Expected warning text, got empty string"
        assert "BUDGET" in w
        assert "Runtime budget" in w
        assert "running low" in w

    def test_critical_at_92_percent_runtime(self):
        """At 92% runtime, critical warning is generated."""
        w = budget_prompt_warning(
            start_epoch=time.time() - 55.2,  # 55.2s of 60s = 92%
            max_runtime_seconds=60,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        assert w, "Expected critical warning text"
        assert "nearly exhausted" in w

    def test_cost_warning_at_65_percent(self):
        """At 65% cost, warning text is generated."""
        w = budget_prompt_warning(
            start_epoch=time.time(),
            max_runtime_seconds=0,
            spent_usd=6.5,
            max_cost_usd=10.0,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        assert w, "Expected cost warning"
        assert "Cost budget" in w

    def test_no_warning_below_threshold(self):
        w = budget_prompt_warning(
            start_epoch=time.time() - 10,
            max_runtime_seconds=60,
            spent_usd=1.0,
            max_cost_usd=10.0,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        assert w == "", "Should return empty string below threshold"

    def test_prompt_warning_appended_to_user_text(self):
        """Simulate what agent.py _invoke() does: append warning to prompt."""
        user_prompt = "Analyze the security of this application"
        w = budget_prompt_warning(
            start_epoch=time.time() - 45,
            max_runtime_seconds=60,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        prompt_to_send = user_prompt + w if w else user_prompt
        assert prompt_to_send.startswith("Analyze the security")
        assert "BUDGET" in prompt_to_send
        assert "Runtime budget" in prompt_to_send

    def test_both_runtime_and_cost_in_prompt(self):
        """When both runtime and cost exceed thresholds, both appear."""
        w = budget_prompt_warning(
            start_epoch=time.time() - 42,
            max_runtime_seconds=60,
            spent_usd=7.0,
            max_cost_usd=10.0,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        assert "Runtime budget" in w
        assert "Cost budget" in w


# ---------------------------------------------------------------------------
# 2. MCP env var propagation
# ---------------------------------------------------------------------------

class TestEnvPropagation:
    """Verify env vars are set correctly for subprocess backends to read."""

    def test_export_sets_all_env_vars(self):
        export_budget_env(
            start_epoch=1000.0,
            max_runtime_seconds=300.0,
            max_cost_usd=5.0,
            warning_ratio=0.5,
            critical_ratio=0.85,
            injection_enabled=True,
        )
        assert os.environ["CHACK_BUDGET_START_EPOCH"] == "1000.0"
        assert os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] == "300.0"
        assert os.environ["CHACK_BUDGET_MAX_COST_USD"] == "5.0"
        assert os.environ["CHACK_BUDGET_SPENT_USD"] == "0"
        assert os.environ["CHACK_BUDGET_WARNING_RATIO"] == "0.5"
        assert os.environ["CHACK_BUDGET_CRITICAL_RATIO"] == "0.85"
        assert os.environ["CHACK_BUDGET_INJECTION_ENABLED"] == "1"

    def test_export_spent_updates_env(self):
        export_spent_usd_env(4.25)
        assert os.environ["CHACK_BUDGET_SPENT_USD"] == "4.25"
        export_spent_usd_env(7.50)
        assert os.environ["CHACK_BUDGET_SPENT_USD"] == "7.5"

    def test_env_keys_match_backend_lists(self):
        """All BUDGET_ENV_KEYS must include CHACK_BUDGET_SPENT_USD."""
        assert "CHACK_BUDGET_SPENT_USD" in BUDGET_ENV_KEYS

    def test_claude_backend_propagates_budget_env_keys(self):
        """Claude Code backend _mcp_env_map includes all budget keys."""
        from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor

        executor = _make_stub_claude_executor()
        env_map = executor._mcp_env_map()
        # _mcp_env_map uses os.environ, so set them first
        export_budget_env(
            start_epoch=time.time(),
            max_runtime_seconds=300,
            max_cost_usd=5.0,
        )
        export_spent_usd_env(2.0)
        env_map = executor._mcp_env_map()
        for key in BUDGET_ENV_KEYS:
            assert key in env_map, f"Claude backend missing env key: {key}"

    def test_codex_backend_includes_spent_usd_key(self):
        """Codex backend env_vars list includes CHACK_BUDGET_SPENT_USD."""
        from chack_agent.backends import codex_backend

        source = open(codex_backend.__file__).read()
        assert "CHACK_BUDGET_SPENT_USD" in source

    def test_copilot_backend_includes_spent_usd_key(self):
        """Copilot backend env_vars list includes CHACK_BUDGET_SPENT_USD."""
        from chack_agent.backends import copilot_cli_backend

        source = open(copilot_cli_backend.__file__).read()
        assert "CHACK_BUDGET_SPENT_USD" in source


# ---------------------------------------------------------------------------
# 3. MCP tool: check_budget_status
# ---------------------------------------------------------------------------

class TestCheckBudgetStatusTool:
    """Verify the check_budget_status MCP tool returns correct budget info."""

    def test_no_limits_configured(self):
        for key in BUDGET_ENV_KEYS:
            os.environ.pop(key, None)
        status = budget_status_from_env()
        assert "No limit configured" in status

    def test_runtime_ok(self):
        os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 10)
        os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
        os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
        os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
        status = budget_status_from_env()
        assert "STATUS: OK" in status
        assert "Runtime:" in status

    def test_runtime_warning(self):
        os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 42)
        os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
        os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
        os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
        status = budget_status_from_env()
        assert "STATUS: WARNING" in status

    def test_runtime_critical(self):
        os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 56)
        os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
        os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
        os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
        status = budget_status_from_env()
        assert "STATUS: CRITICAL" in status

    def test_cost_info_included(self):
        os.environ["CHACK_BUDGET_MAX_COST_USD"] = "10.0"
        os.environ["CHACK_BUDGET_SPENT_USD"] = "7.5"
        status = budget_status_from_env()
        assert "Cost:" in status
        assert "$7.5" in status
        assert "$10.0" in status

    def test_both_runtime_and_cost(self):
        os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 45)
        os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
        os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
        os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
        os.environ["CHACK_BUDGET_MAX_COST_USD"] = "5.0"
        os.environ["CHACK_BUDGET_SPENT_USD"] = "4.0"
        status = budget_status_from_env()
        assert "Runtime:" in status
        assert "Cost:" in status
        assert "STATUS: WARNING" in status


# ---------------------------------------------------------------------------
# 4. MCP tool output injection
# ---------------------------------------------------------------------------

class TestMCPToolOutputInjection:
    """Verify inject_budget_warning_from_env appends warnings to tool output."""

    def test_runtime_warning_injected(self):
        os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 40)
        os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
        os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
        os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
        os.environ["CHACK_BUDGET_INJECTION_ENABLED"] = "1"
        result = inject_budget_warning_from_env("Search results: found 3 items")
        assert result.startswith("Search results: found 3 items")
        assert "BUDGET" in result
        assert "Runtime budget" in result

    def test_disabled_no_injection(self):
        os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 55)
        os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
        os.environ["CHACK_BUDGET_INJECTION_ENABLED"] = "0"
        result = inject_budget_warning_from_env("tool output")
        assert result == "tool output"

    def test_below_threshold_no_injection(self):
        os.environ["CHACK_BUDGET_START_EPOCH"] = str(time.time() - 10)
        os.environ["CHACK_BUDGET_MAX_RUNTIME_SECONDS"] = "60"
        os.environ["CHACK_BUDGET_WARNING_RATIO"] = "0.6"
        os.environ["CHACK_BUDGET_CRITICAL_RATIO"] = "0.9"
        os.environ["CHACK_BUDGET_INJECTION_ENABLED"] = "1"
        result = inject_budget_warning_from_env("tool output")
        assert result == "tool output"


# ---------------------------------------------------------------------------
# 5. End-to-end compose_prompt simulation
# ---------------------------------------------------------------------------

class TestComposePromptEndToEnd:
    """Simulate what actually happens when agent.py calls
    executor.invoke({"input": prompt_with_budget_warning}).

    The user input already has budget warning appended by agent.py,
    then _compose_prompt() adds it to system prompt + policies.
    """

    def test_claude_compose_prompt_with_budget(self):
        from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor

        executor = _make_stub_claude_executor()
        # Simulate what agent.py does: append budget warning to user input
        user_input = "Find vulnerabilities in the codebase"
        budget_warning = budget_prompt_warning(
            start_epoch=time.time() - 50,
            max_runtime_seconds=60,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        prompt_to_send = user_input + budget_warning

        # This is what the executor sees
        composed = executor._compose_prompt(prompt_to_send)

        # Verify system prompt present
        assert "You are a helpful assistant" in composed
        # Verify user input present
        assert "Find vulnerabilities" in composed
        # Verify budget warning present (the model will see this!)
        assert "BUDGET" in composed
        assert "Runtime budget" in composed

    def test_codex_compose_prompt_with_budget(self):
        from chack_agent.backends.codex_backend import CodexExecutor

        executor = _make_stub_codex_executor()
        user_input = "Analyze the code"
        budget_warning = budget_prompt_warning(
            start_epoch=time.time() - 55,
            max_runtime_seconds=60,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        composed = executor._compose_prompt(user_input + budget_warning)
        assert "Analyze the code" in composed
        assert "BUDGET" in composed

    def test_copilot_compose_prompt_with_budget(self):
        from chack_agent.backends.copilot_cli_backend import CopilotCliExecutor

        executor = _make_stub_copilot_executor()
        user_input = "Check this code"
        budget_warning = budget_prompt_warning(
            start_epoch=time.time() - 50,
            max_runtime_seconds=60,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        composed = executor._compose_prompt(user_input + budget_warning)
        assert "Check this code" in composed
        assert "BUDGET" in composed


# ---------------------------------------------------------------------------
# 6. MCP server registers check_budget_status
# ---------------------------------------------------------------------------

class TestMCPServerRegistration:
    """Verify the MCP server would register check_budget_status tool."""

    def test_main_registers_budget_tool(self):
        """Verify that chack_tools_mcp_server.main() registers
        check_budget_status alongside dynamic tools."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "chack_agent", "backends", "chack_tools_mcp_server.py",
        )
        with open(source_path) as f:
            source = f.read()

        # Check the tool is registered
        assert "check_budget_status" in source
        assert "budget_status_from_env" in source
        assert "mcp.tool" in source or "@mcp.tool" in source


# ---------------------------------------------------------------------------
# 7. Full agent.py _invoke() simulation
# ---------------------------------------------------------------------------

class TestAgentInvokeSimulation:
    """Simulate the exact code path in agent.py _invoke() that prepends
    budget warnings to the prompt."""

    def test_invoke_appends_budget_warning_when_threshold_met(self):
        """Replicate agent.py _invoke() logic and verify warning is appended."""
        # Setup: simulate agent.py state
        run_started_at = time.time() - 50  # 50s elapsed
        max_runtime_seconds = 60.0
        max_cost_usd = 10.0
        estimated_cost_spent = 7.5
        budget_warning_ratio = 0.6
        budget_critical_ratio = 0.9
        budget_tool_injection_enabled = True

        current_prompt = "Original user request"

        # This is the exact code path from agent.py _invoke():
        export_spent_usd_env(estimated_cost_spent)
        prompt_to_send = current_prompt
        if budget_tool_injection_enabled:
            bw = budget_prompt_warning(
                start_epoch=run_started_at,
                max_runtime_seconds=max_runtime_seconds,
                elapsed_runtime_seconds=time.time() - run_started_at,
                spent_usd=estimated_cost_spent,
                max_cost_usd=max_cost_usd,
                warning_ratio=budget_warning_ratio,
                critical_ratio=budget_critical_ratio,
            )
            if bw:
                prompt_to_send = current_prompt + bw

        # Verify
        assert prompt_to_send.startswith("Original user request")
        assert "BUDGET" in prompt_to_send
        assert "Runtime budget" in prompt_to_send
        assert "Cost budget" in prompt_to_send
        # Verify env var was set for MCP tool
        assert os.environ["CHACK_BUDGET_SPENT_USD"] == "7.5"

    def test_invoke_no_warning_when_below_threshold(self):
        """No warning appended when under budget."""
        run_started_at = time.time() - 10
        current_prompt = "User request"

        prompt_to_send = current_prompt
        bw = budget_prompt_warning(
            start_epoch=run_started_at,
            max_runtime_seconds=60,
            elapsed_runtime_seconds=time.time() - run_started_at,
            spent_usd=1.0,
            max_cost_usd=10.0,
            warning_ratio=0.6,
            critical_ratio=0.9,
        )
        if bw:
            prompt_to_send = current_prompt + bw

        assert prompt_to_send == "User request"

    def test_invoke_no_warning_when_disabled(self):
        """No warning when budget_tool_injection_enabled=False."""
        run_started_at = time.time() - 55
        budget_tool_injection_enabled = False
        current_prompt = "User request"

        prompt_to_send = current_prompt
        if budget_tool_injection_enabled:
            bw = budget_prompt_warning(
                start_epoch=run_started_at,
                max_runtime_seconds=60,
                warning_ratio=0.6,
                critical_ratio=0.9,
            )
            if bw:
                prompt_to_send = current_prompt + bw

        assert prompt_to_send == "User request"


# ---------------------------------------------------------------------------
# Helpers: create stub executor instances
# ---------------------------------------------------------------------------

def _make_stub_claude_executor():
    from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor
    return ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=10,
        _memory_reset_to=5,
        _base_system_prompt="You are a helpful assistant.",
        _model_name="claude-sonnet-4-20250514",
        _max_turns=5,
        _claude_cli_path="claude",
        _tools_config_json="{}",
        _allowed_tools_json="[]",
        _serialized_tools_override_b64="",
        _serialized_tools_append_b64="",
        _model_provider="claude-code",
        _default_model="claude-sonnet-4-20250514",
        _social_network_model="",
        _scientific_model="",
        _websearcher_model="",
        _tester_model="",
        _subchack_model="",
        _social_network_max_turns=10,
        _scientific_max_turns=10,
        _websearcher_max_turns=10,
        _tester_max_turns=10,
        _subchack_max_turns=10,
        _min_tools_used=0,
        _max_tools_used=0,
        _require_task_steps_manager_init_first=False,
        _output_schema_json="",
    )


def _make_stub_codex_executor():
    from chack_agent.backends.codex_backend import CodexExecutor
    return CodexExecutor(
        _conversation=[],
        _memory_limit=10,
        _memory_reset_to=5,
        _base_system_prompt="You are a helpful assistant.",
        _model_name="codex",
        _max_turns=5,
        _codex_path="codex",
        _tools_config_json="{}",
        _allowed_tools_json="[]",
        _serialized_tools_override_b64="",
        _serialized_tools_append_b64="",
        _model_provider="codex",
        _default_model="codex",
        _social_network_model="",
        _scientific_model="",
        _websearcher_model="",
        _tester_model="",
        _subchack_model="",
        _social_network_max_turns=10,
        _scientific_max_turns=10,
        _websearcher_max_turns=10,
        _tester_max_turns=10,
        _subchack_max_turns=10,
        _min_tools_used=0,
        _max_tools_used=0,
        _require_task_steps_manager_init_first=False,
        _output_schema_json="",
        _openai_api_key="",
        _fallback_openai_api_key="",
        _codex_access_token="",
        _use_codex_access_token=False,
        _use_existing_codex_auth_file=False,
        _existing_codex_auth_file="",
    )


def test_codex_build_command_can_disable_native_shell_and_web_search():
    executor = _make_stub_codex_executor()
    executor._disable_native_shell = True
    executor._disable_native_web_search = True

    command = executor._build_command()

    assert command.count("--disable") == 4
    assert "-c" in command
    assert 'web_search="disabled"' in command
    assert "shell_tool" in command
    assert "unified_exec" in command
    assert "web_search_request" in command
    assert "web_search_cached" in command


def _make_stub_copilot_executor():
    from chack_agent.backends.copilot_cli_backend import CopilotCliExecutor
    return CopilotCliExecutor(
        conversation=[],
        memory_max_messages=10,
        memory_reset_to_messages=5,
        base_system_prompt="You are a helpful assistant.",
        model_name="gpt-4",
        max_turns=5,
        copilot_cli_path="copilot",
        copilot_github_token="",
        tools_config_json="{}",
        allowed_tools_json="[]",
        serialized_tools_override_b64="",
        serialized_tools_append_b64="",
        model_provider="copilot",
        default_model="gpt-4",
        social_network_model="",
        scientific_model="",
        websearcher_model="",
        tester_model="",
        subchack_model="",
        social_network_max_turns=10,
        scientific_max_turns=10,
        websearcher_max_turns=10,
        tester_max_turns=10,
        subchack_max_turns=10,
        min_tools_used=0,
        max_tools_used=0,
        require_task_steps_manager_init_first=False,
        output_schema_json="",
    )
