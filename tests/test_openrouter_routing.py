import os
import tempfile
import unittest

from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)
from chack_agent.openrouter_routing import (
    OPENROUTER_DEFAULT_BASE_URL,
    clone_config_for_openrouter,
    get_openrouter_route,
)
from chack_agent.backends.codex_backend import build_executor as build_codex_executor
from chack_agent.backends.claude_code_backend import build_executor as build_claude_executor
from chack_agent.backends.gemini_cli_backend import build_executor as build_gemini_executor
from chack_agent.backends.openai_compaction_backend import build_executor as build_openai_executor


def _make_config(provider: str, primary: str) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary=primary, provider=provider),
        agent=AgentConfig(main_action="test", sub_action="test"),
        session=SessionConfig(max_turns=2),
        tools=ToolsConfig(),
        credentials=CredentialsConfig(
            openrouter_api_key="or-test-key",
            openrouter_http_referer="https://example.test",
            openrouter_app_name="chack-test",
        ),
        logging=LoggingConfig(),
        system_prompt="test system prompt",
        env={},
    )


class OpenRouterRoutingTests(unittest.TestCase):
    def test_get_openrouter_route_strips_prefix_and_collects_headers(self) -> None:
        config = _make_config("codex", "openrouter/openai/gpt-4.1-mini")

        route = get_openrouter_route(config)

        assert route is not None
        self.assertEqual(route.model_name, "openai/gpt-4.1-mini")
        self.assertEqual(route.api_key, "or-test-key")
        self.assertEqual(route.base_url, OPENROUTER_DEFAULT_BASE_URL)
        self.assertEqual(route.headers["HTTP-Referer"], "https://example.test")
        self.assertEqual(route.headers["X-Title"], "chack-test")
        self.assertEqual(route.anthropic_base_url, "https://openrouter.ai/api")

    def test_clone_config_for_openrouter_rewrites_provider_and_prefixed_models(self) -> None:
        config = _make_config("gemini", "openrouter/google/gemini-2.5-flash")
        config.model.social_network = "openrouter/anthropic/claude-3.7-sonnet"

        cloned = clone_config_for_openrouter(config)

        self.assertEqual(cloned.model.provider, "openrouter")
        self.assertEqual(cloned.model.primary, "google/gemini-2.5-flash")
        self.assertEqual(cloned.model.social_network, "anthropic/claude-3.7-sonnet")

    def test_codex_executor_uses_openrouter_credentials_and_provider_config(self) -> None:
        config = _make_config("codex", "openrouter/openai/gpt-4.1-mini")
        with tempfile.TemporaryDirectory() as tmpdir:
            previous = os.environ.get("CHACK_CODEX_HOME_BASE")
            os.environ["CHACK_CODEX_HOME_BASE"] = tmpdir
            try:
                executor = build_codex_executor(
                    config,
                    system_prompt="system",
                    max_turns=2,
                    memory_max_messages=10,
                    memory_reset_to_messages=5,
                )
                self.assertTrue(executor._uses_openrouter_route)
                self.assertEqual(executor._model_name, "openai/gpt-4.1-mini")
                env = executor._build_env()
                self.assertEqual(env["OPENROUTER_API_KEY"], "or-test-key")
                self.assertEqual(env["OPENROUTER_BASE_URL"], OPENROUTER_DEFAULT_BASE_URL)
                executor._ensure_codex_home_and_config()
                with open(os.path.join(executor._codex_home, "config.toml"), "r", encoding="utf-8") as handle:
                    config_body = handle.read()
                self.assertIn('model_provider = "openrouter"', config_body)
                self.assertIn('[model_providers.openrouter]', config_body)
            finally:
                if previous is None:
                    os.environ.pop("CHACK_CODEX_HOME_BASE", None)
                else:
                    os.environ["CHACK_CODEX_HOME_BASE"] = previous

    def test_claude_backend_delegates_to_openrouter_backend_for_routed_models(self) -> None:
        config = _make_config("claude", "openrouter/anthropic/claude-3.7-sonnet")

        executor = build_claude_executor(
            config,
            system_prompt="system",
            max_turns=2,
            memory_max_messages=10,
            memory_reset_to_messages=5,
        )

        self.assertEqual(
            executor.__class__.__module__,
            "chack_agent.backends.openrouter_openai_backend",
        )

    def test_gemini_backend_delegates_to_openrouter_backend_for_routed_models(self) -> None:
        config = _make_config("gemini", "openrouter/google/gemini-2.5-flash")

        executor = build_gemini_executor(
            config,
            system_prompt="system",
            max_turns=2,
            memory_max_messages=10,
            memory_reset_to_messages=5,
        )

        self.assertEqual(
            executor.__class__.__module__,
            "chack_agent.backends.openrouter_openai_backend",
        )

    def test_openai_backend_delegates_to_openrouter_backend_for_routed_models(self) -> None:
        config = _make_config("openai", "openrouter/openai/gpt-4.1-mini")

        executor = build_openai_executor(
            config,
            system_prompt="system",
            max_turns=2,
            memory_max_messages=10,
            memory_reset_to_messages=5,
        )

        self.assertEqual(
            executor.__class__.__module__,
            "chack_agent.backends.openrouter_openai_backend",
        )


if __name__ == "__main__":
    unittest.main()
