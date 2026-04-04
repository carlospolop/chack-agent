"""Test that the copilot backend correctly handles custom tool serialization."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from chack_agent.backends.tool_payloads import serialize_tools_payload, deserialize_tools_payload


class _FakeTool:
    """Minimal tool matching the agents-sdk FunctionTool interface."""
    def __init__(self, name: str, description: str, params: dict):
        self.name = name
        self.description = description
        self.params_json_schema = params

    async def on_invoke_tool(self, ctx, raw_args):
        args = json.loads(raw_args)
        return json.dumps({"status": "ok", "echo": args})


def _build_vuln_saver_like_tool():
    """Simulate a naxus-like vuln_saver tool."""
    return _FakeTool(
        name="vulnerability_saver",
        description="Save a discovered vulnerability with severity, title, and description.",
        params={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Vulnerability title"},
                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
                "description": {"type": "string", "description": "Detailed vulnerability description"},
            },
            "required": ["title", "severity", "description"],
        },
    )


def _build_vuln_checker_like_tool():
    """Simulate a naxus-like vuln_checker tool."""
    return _FakeTool(
        name="vulnerability_checker",
        description="Check if a vulnerability has already been reported.",
        params={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Vulnerability title to check"},
            },
            "required": ["title"],
        },
    )


class TestCopilotToolSerialization(unittest.TestCase):
    """Verify that naxus-like tools can be serialized and deserialized for CLI backends."""

    def test_serialize_and_deserialize_custom_tools(self):
        vuln_saver = _build_vuln_saver_like_tool()
        vuln_checker = _build_vuln_checker_like_tool()
        tools = [vuln_saver, vuln_checker]

        payload = serialize_tools_payload(tools)
        self.assertTrue(len(payload) > 0, "Serialized payload should not be empty")

        restored = deserialize_tools_payload(payload)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].name, "vulnerability_saver")
        self.assertEqual(restored[1].name, "vulnerability_checker")

    def test_serialize_none_returns_empty(self):
        self.assertEqual(serialize_tools_payload(None), "")

    def test_serialize_empty_list(self):
        payload = serialize_tools_payload([])
        self.assertTrue(len(payload) > 0)
        restored = deserialize_tools_payload(payload)
        self.assertEqual(len(restored), 0)

    def test_tool_schema_survives_round_trip(self):
        tool = _build_vuln_saver_like_tool()
        payload = serialize_tools_payload([tool])
        restored = deserialize_tools_payload(payload)
        self.assertEqual(
            restored[0].params_json_schema,
            tool.params_json_schema,
        )


class TestCopilotBackendToolEnvSetup(unittest.TestCase):
    """Verify that the copilot executor correctly passes tool env vars."""

    def test_build_env_includes_tool_envs(self):
        vuln_saver = _build_vuln_saver_like_tool()
        payload = serialize_tools_payload([vuln_saver])

        # Import the executor directly
        from chack_agent.backends.copilot_cli_backend import CopilotCliExecutor
        from chack_agent.backends.tool_payloads import (
            CHACK_TOOLS_OVERRIDE_B64_ENV,
            CHACK_TOOLS_APPEND_B64_ENV,
        )

        executor = CopilotCliExecutor(
            conversation=[],
            memory_max_messages=100,
            memory_reset_to_messages=10,
            base_system_prompt="test",
            model_name="gpt-5.4",
            max_turns=10,
            copilot_cli_path="/usr/bin/true",
            copilot_github_token="fake-token",
            tools_config_json="{}",
            allowed_tools_json="[]",
            serialized_tools_override_b64=payload,
            serialized_tools_append_b64="",
            model_provider="copilot",
            default_model="gpt-5.4",
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

        env = executor._build_env()
        self.assertEqual(env.get("COPILOT_GITHUB_TOKEN"), "fake-token")
        self.assertEqual(env.get(CHACK_TOOLS_OVERRIDE_B64_ENV), payload)
        self.assertIn("CHACK_MODEL_PROVIDER", env)

    def test_missing_token_not_set_in_env(self):
        from chack_agent.backends.copilot_cli_backend import CopilotCliExecutor

        executor = CopilotCliExecutor(
            conversation=[],
            memory_max_messages=100,
            memory_reset_to_messages=10,
            base_system_prompt="test",
            model_name="gpt-5.4",
            max_turns=10,
            copilot_cli_path="/usr/bin/true",
            copilot_github_token="",  # Empty token
            tools_config_json="{}",
            allowed_tools_json="[]",
            serialized_tools_override_b64="",
            serialized_tools_append_b64="",
            model_provider="copilot",
            default_model="gpt-5.4",
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

        env = executor._build_env()
        self.assertNotIn("COPILOT_GITHUB_TOKEN", env)


if __name__ == "__main__":
    unittest.main()
