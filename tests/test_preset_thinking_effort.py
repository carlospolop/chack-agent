import tempfile
import unittest
from unittest.mock import patch

import yaml

from chack_agent.config import load_config
from chack_agent.model_aliases import _load_remote_aliases, resolve_thinking_effort_alias


def _config(agent: dict, tools: dict | None = None):
    raw = {
        "system_prompt": "test system prompt",
        "agent": {"provider": "codex", "main_action": "test", "sub_action": "run", **agent},
        "tools": tools or {},
        "credentials": {"codex_access_token": "codex-access-token"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(raw, handle)
        path = handle.name
    return load_config(path)


class PresetThinkingEffortTests(unittest.TestCase):
    def setUp(self) -> None:
        patchers = [
            patch("chack_agent.model_aliases._refresh_remote_alias_caches"),
            patch("chack_agent.model_aliases._REMOTE_MODEL_CACHE", {
                "OPENAI_BEST_QUALITY": "gpt-6-sol",
                "OPENAI_CHEAP_BUT_QUALITY": "gpt-6-sol",
            }),
            patch("chack_agent.model_aliases._REMOTE_EFFORT_CACHE", {
                "OPENAI_BEST_QUALITY": "xhigh",
                "OPENAI_CHEAP_BUT_QUALITY": "medium",
                "CODEX_BEST_QUALITY": "max",
            }),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_alias_effort_comes_from_the_lambda(self) -> None:
        self.assertEqual(resolve_thinking_effort_alias("OPENAI_BEST_QUALITY", provider="codex"), "xhigh")
        # Generic tiers follow the provider's own alias, like the model does.
        self.assertEqual(resolve_thinking_effort_alias("BEST_QUALITY", provider="codex"), "max")
        self.assertEqual(resolve_thinking_effort_alias("CHEAP_BUT_QUALITY", provider="codex"), "medium")
        # Tiers the Lambda does not publish fall back to the local table.
        self.assertEqual(resolve_thinking_effort_alias("OPENAI_BEST_CHEAPEST", provider="codex"), "low")
        self.assertEqual(resolve_thinking_effort_alias("gpt-6-sol", provider="codex"), "")

    def test_unknown_published_level_is_ignored(self) -> None:
        with patch("chack_agent.model_aliases._REMOTE_EFFORT_CACHE", {"OPENAI_BEST_QUALITY": "turbo"}):
            self.assertEqual(resolve_thinking_effort_alias("OPENAI_BEST_QUALITY", provider="codex"), "high")

    def test_preset_effort_follows_each_agents_alias(self) -> None:
        config = _config({
            "primary": "OPENAI_BEST_QUALITY",
            "thinking_effort": "preset",
            "social_network": "OPENAI_CHEAP_BUT_QUALITY",
            "social_network_thinking_effort": "preset",
            "cli_thinking_effort": "low",
        })
        self.assertEqual(config.model.primary, "gpt-6-sol")
        self.assertEqual(config.agent.thinking_effort, "xhigh")
        self.assertEqual(config.model.social_network_thinking_effort, "medium")
        self.assertEqual(config.tools.social_network_agent["thinking_effort"], "medium")
        # Explicit levels are never replaced by the Lambda.
        self.assertEqual(config.model.cli_thinking_effort, "low")

    def test_tool_level_preset_uses_the_role_model(self) -> None:
        config = _config(
            {"primary": "OPENAI_BEST_QUALITY"},
            {"websearcher_agent": {"thinking_effort": "preset"}},
        )
        self.assertEqual(config.agent.thinking_effort, "high")
        # websearcher is unset, so it runs on CHEAP_BUT_QUALITY.
        self.assertEqual(config.tools.websearcher_agent["thinking_effort"], "medium")

    def test_preset_with_a_concrete_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent.thinking_effort='preset'"):
            _config({"primary": "gpt-6-sol", "thinking_effort": "preset"})

    def test_remote_payload_carries_thinking_efforts(self) -> None:
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"model_aliases": {"OPENAI_BEST_QUALITY": "gpt-6-sol"},
                        "thinking_efforts": {"OPENAI_BEST_QUALITY": "high"}}

        with patch("chack_agent.model_aliases._REMOTE_URL", "https://aliases.example"), patch(
            "chack_agent.model_aliases.requests.get", return_value=Response()
        ):
            remote = _load_remote_aliases()
        self.assertEqual(remote["thinking_efforts"], {"OPENAI_BEST_QUALITY": "high"})


if __name__ == "__main__":
    unittest.main()
