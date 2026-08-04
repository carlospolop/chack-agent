import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from chack_agent import Chack
from chack_agent.config import load_config, resolve_api_key_type
from chack_agent.model_aliases import (
    OPENROUTER_BEST_CHEAPEST,
    OPENROUTER_CHEAP_BUT_QUALITY,
    get_default_backend_aliases,
    get_default_model_aliases,
    resolve_backend_alias,
    resolve_model_alias,
)


class ModelAliasResolutionTests(unittest.TestCase):
    def test_load_config_supports_codex_access_token_for_explicit_codex_provider(self) -> None:
        config_yaml = textwrap.dedent(
            """
            system_prompt: test system prompt
            agent:
              primary: CHEAP_BUT_QUALITY
              provider: codex
              main_action: test
              sub_action: run
              self_critique_rounds: 2
            credentials:
              codex_access_token: codex-access-token
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_yaml)
            path = handle.name

        config = load_config(path)

        self.assertEqual(config.model.provider, "codex")
        self.assertEqual(config.model.primary, "gpt-5.4-mini")
        self.assertEqual(config.agent.self_critique_rounds, 2)
        self.assertEqual(config.credentials.codex_access_token, "codex-access-token")

    def test_resolve_backend_alias_prefers_codex_access_then_openai_then_anthropic_then_openrouter(self) -> None:
        self.assertEqual(
            resolve_backend_alias(
                "DEFAULT_BACKEND",
                codex_access_token="codex-access-token",
                openai_api_key="oa-test",
                anthropic_api_key="anth-test",
                openrouter_api_key="or-test",
            ),
            "codex",
        )
        self.assertEqual(
            resolve_backend_alias(
                "DEFAULT_BACKEND",
                openai_api_key="oa-test",
                anthropic_api_key="anth-test",
                openrouter_api_key="or-test",
            ),
            "codex",
        )
        self.assertEqual(
            resolve_backend_alias(
                "DEFAULT_BACKEND",
                anthropic_api_key="anth-test",
                openrouter_api_key="or-test",
            ),
            "claude",
        )
        self.assertEqual(
            resolve_backend_alias(
                "DEFAULT_BACKEND",
                openrouter_api_key="or-test",
            ),
            "openrouter",
        )

    def test_resolve_model_alias_uses_provider_specific_defaults(self) -> None:
        self.assertEqual(
            resolve_model_alias("CHEAP_BUT_QUALITY", provider="codex"),
            "gpt-5.4-mini",
        )
        self.assertEqual(
            resolve_model_alias("CHEAP_BUT_QUALITY", provider="claude"),
            "claude-sonnet-4-6",
        )
        self.assertEqual(
            resolve_model_alias("CHEAP_BUT_QUALITY", provider="openrouter"),
            OPENROUTER_CHEAP_BUT_QUALITY,
        )

    def test_resolve_model_alias_uses_api_key_priority_when_provider_is_unspecified(self) -> None:
        self.assertEqual(
            resolve_model_alias("BEST_QUALITY", openai_api_key="oa-test"),
            "gpt-5.4",
        )
        self.assertEqual(
            resolve_model_alias("CHEAP_BUT_QUALITY", anthropic_api_key="anth-test"),
            "claude-sonnet-4-6",
        )
        self.assertEqual(
            resolve_model_alias("BEST_CHEAPEST", openrouter_api_key="or-test"),
            OPENROUTER_BEST_CHEAPEST,
        )
        self.assertEqual(
            resolve_model_alias("CHEAP_BUT_QUALITY", openrouter_api_key="or-test"),
            OPENROUTER_CHEAP_BUT_QUALITY,
        )
        self.assertEqual(
            resolve_model_alias("BEST_QUALITY", openai_api_key="oa-test", anthropic_api_key="anth-test", openrouter_api_key="or-test"),
            "gpt-5.4",
        )

    def test_resolve_model_alias_does_not_treat_codex_access_token_as_generic_model_priority(self) -> None:
        with patch.dict("os.environ", {"CODEX_ACCESS_TOKEN": "codex-access-token"}, clear=False):
            with self.assertRaisesRegex(ValueError, "requires one of OPENAI_API_KEY"):
                resolve_model_alias("BEST_QUALITY")

    def test_resolve_model_alias_raises_when_generic_alias_has_no_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires one of OPENAI_API_KEY"):
            resolve_model_alias("BEST_QUALITY")

    def test_resolve_backend_alias_uses_codex_access_token_without_openai_api_key(self) -> None:
        self.assertEqual(
            resolve_backend_alias(
                "DEFAULT_BACKEND",
                codex_access_token="codex-access-token",
            ),
            "codex",
        )

    def test_resolve_backend_alias_uses_claude_access_token_without_anthropic_api_key(self) -> None:
        self.assertEqual(
            resolve_backend_alias(
                "DEFAULT_BACKEND",
                claude_access_token="claude-token",
            ),
            "claude",
        )

    def test_resolve_backend_alias_reads_canonical_claude_oauth_token_env(self) -> None:
        with patch.dict(
            os.environ,
            {"CLAUDE_CODE_OAUTH_TOKEN": "claude-token"},
            clear=True,
        ):
            self.assertEqual(resolve_backend_alias("DEFAULT_BACKEND"), "claude")

    def test_resolve_backend_alias_raises_when_default_backend_has_no_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "DEFAULT_BACKEND requires one of CODEX_ACCESS_TOKEN, OPENAI_API_KEY"):
            resolve_backend_alias("DEFAULT_BACKEND")

    def test_resolve_model_alias_logs_resolution_for_generic_alias(self) -> None:
        with patch("chack_agent.model_aliases._LOGGER.info") as info_mock:
            resolved = resolve_model_alias("CHEAP_BUT_QUALITY", openrouter_api_key="or-test")
        self.assertEqual(resolved, OPENROUTER_CHEAP_BUT_QUALITY)
        info_mock.assert_called()

    def test_resolve_backend_alias_logs_resolution_for_default_backend(self) -> None:
        with patch("chack_agent.model_aliases._LOGGER.info") as info_mock:
            resolved = resolve_backend_alias("DEFAULT_BACKEND", anthropic_api_key="anth-test")
        self.assertEqual(resolved, "claude")
        info_mock.assert_called()

    def test_resolve_model_alias_requires_full_openrouter_path(self) -> None:
        with patch("chack_agent.model_aliases._get_model_aliases", return_value={"OPENROUTER_BEST_QUALITY": "gpt-5.2-codex"}):
            with self.assertRaisesRegex(ValueError, "openrouter models must use a full 'openrouter/<vendor>/<model>' path"):
                resolve_model_alias("BEST_QUALITY", openrouter_api_key="or-test")

    def test_load_config_resolves_default_backend_and_provider_specific_model(self) -> None:
        config_yaml = textwrap.dedent(
            """
            system_prompt: test system prompt
            agent:
              primary: CHEAP_BUT_QUALITY
              provider: DEFAULT_BACKEND
              main_action: test
              sub_action: run
            credentials:
              anthropic_api_key: anthropic-test
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_yaml)
            path = handle.name

        config = load_config(path)

        self.assertEqual(config.model.provider, "claude")
        self.assertEqual(config.model.primary, "claude-sonnet-4-6")

    def test_chack_resolves_aliases_when_initialized_with_config_object(self) -> None:
        config_yaml = textwrap.dedent(
            """
            system_prompt: test system prompt
            agent:
              primary: OPENAI_CHEAP_BUT_QUALITY
              provider: openai
              main_action: test
              sub_action: run
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_yaml)
            path = handle.name

        config = load_config(path)
        config.model.primary = "OPENAI_CHEAP_BUT_QUALITY"

        with patch("chack_agent.agent.load_pricing", return_value={}), patch(
            "chack_agent.agent.resolve_pricing_path", return_value="pricing.yaml"
        ), patch("chack_agent.agent.export_env"):
            chack = Chack(config, config_path=path)

        self.assertEqual(chack.config.model.primary, "gpt-5.4-mini")

    def test_resolve_api_key_type_prefers_codex_token_for_codex_provider(self) -> None:
        config_yaml = textwrap.dedent(
            """
            system_prompt: test system prompt
            agent:
              primary: CHEAP_BUT_QUALITY
              provider: codex
              main_action: test
              sub_action: run
            credentials:
              codex_access_token: codex-access-token
              openai_api_key: oa-test
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_yaml)
            path = handle.name

        config = load_config(path)

        self.assertEqual(resolve_api_key_type(config), "codex_token")

    def test_resolve_api_key_type_honors_explicit_openai_selection(self) -> None:
        config_yaml = textwrap.dedent(
            """
            system_prompt: test system prompt
            agent:
              primary: gpt-5.4-mini
              provider: codex
              api_key_type: openai
              main_action: test
              sub_action: run
            credentials:
              codex_access_token: codex-access-token
              openai_api_key: oa-test
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_yaml)
            path = handle.name

        config = load_config(path)

        self.assertEqual(config.model.api_key_type, "openai")
        self.assertEqual(resolve_api_key_type(config), "openai")

    def test_chack_logs_instantiation_details(self) -> None:
        config_yaml = textwrap.dedent(
            """
            system_prompt: test system prompt
            agent:
              primary: BEST_QUALITY
              provider: openrouter
              main_action: test
              sub_action: run
            credentials:
              openrouter_api_key: or-test
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_yaml)
            path = handle.name

        with patch("chack_agent.agent.load_pricing", return_value={}), patch(
            "chack_agent.agent.resolve_pricing_path", return_value="pricing.yaml"
        ), patch("chack_agent.agent.export_env"), patch(
            "chack_agent.agent.logging.getLogger"
        ) as logger_mock:
            Chack(path)

        logger_mock.return_value.info.assert_any_call(
            "Agent instantiated: model=%s backend=%s api_key_type=%s",
            "openrouter/openai/gpt-5.4",
            "openrouter",
            "openrouter",
        )

    def test_default_model_aliases_do_not_publish_generic_best_aliases(self) -> None:
        aliases = get_default_model_aliases()
        self.assertNotIn("BEST_QUALITY", aliases)
        self.assertNotIn("CHEAP_BUT_QUALITY", aliases)
        self.assertNotIn("BEST_CHEAPEST", aliases)

    def test_default_backend_aliases_do_not_publish_generic_default_backend(self) -> None:
        aliases = get_default_backend_aliases()
        self.assertNotIn("DEFAULT_BACKEND", aliases)


if __name__ == "__main__":
    unittest.main()
