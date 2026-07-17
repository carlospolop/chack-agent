import os
import tempfile
import unittest
import base64
import json
from unittest.mock import patch

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
from chack_agent.backends.tool_payloads import (
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_APPEND_B64_PATH_ENV,
)


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


def _fake_chatgpt_access_token(
    *,
    account_id: str = "workspace-123",
    plan_type: str = "plus",
) -> str:
    def _b64(data: dict) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    header = {"alg": "none", "typ": "JWT"}
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": plan_type,
        }
    }
    return f"{_b64(header)}.{_b64(payload)}.sig"


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

    def test_codex_executor_uses_direct_codex_access_token(self) -> None:
        config = _make_config("codex", "gpt-5-mini")
        config.credentials.openrouter_api_key = ""
        config.credentials.codex_access_token = _fake_chatgpt_access_token()
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
                self.assertTrue(executor._use_codex_access_token)
                executor._ensure_codex_home_and_config()
                env = executor._build_env()
                self.assertNotIn("OPENAI_API_KEY", env)
                self.assertNotIn("CODEX_API_KEY", env)
                with open(os.path.join(executor._codex_home, "auth.json"), "r", encoding="utf-8") as handle:
                    auth_payload = json.load(handle)
                self.assertEqual(auth_payload["auth_mode"], "chatgpt")
                self.assertEqual(auth_payload["tokens"]["access_token"], config.credentials.codex_access_token)
                self.assertEqual(auth_payload["tokens"]["account_id"], "workspace-123")
            finally:
                if previous is None:
                    os.environ.pop("CHACK_CODEX_HOME_BASE", None)
                else:
                    os.environ["CHACK_CODEX_HOME_BASE"] = previous

    def test_codex_executor_prefers_access_token_over_openai_api_key_when_both_are_present(self) -> None:
        config = _make_config("codex", "gpt-5-mini")
        config.credentials.openrouter_api_key = ""
        config.credentials.openai_api_key = "sk-openai-direct"
        config.credentials.codex_access_token = _fake_chatgpt_access_token()

        previous_openai = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        try:
            executor = build_codex_executor(
                config,
                system_prompt="system",
                max_turns=2,
                memory_max_messages=10,
                memory_reset_to_messages=5,
            )
            self.assertTrue(executor._use_codex_access_token)
            executor._ensure_codex_home_and_config()
            env = executor._build_env()
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("CODEX_API_KEY", env)
        finally:
            if previous_openai is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous_openai

    def test_codex_executor_falls_back_to_openai_api_key_when_access_token_fails(self) -> None:
        config = _make_config("codex", "gpt-5-mini")
        config.credentials.openrouter_api_key = ""
        config.credentials.openai_api_key = "sk-openai-direct"
        config.credentials.codex_access_token = _fake_chatgpt_access_token()

        executor = build_codex_executor(
            config,
            system_prompt="system",
            max_turns=2,
            memory_max_messages=10,
            memory_reset_to_messages=5,
        )

        self.assertTrue(executor._use_codex_access_token)
        with tempfile.TemporaryDirectory() as tmpdir:
            previous = os.environ.get("CHACK_CODEX_HOME_BASE")
            os.environ["CHACK_CODEX_HOME_BASE"] = tmpdir
            try:
                executor._ensure_codex_home_and_config()
                self.assertTrue(os.path.exists(os.path.join(executor._codex_home, "auth.json")))
            finally:
                if previous is None:
                    os.environ.pop("CHACK_CODEX_HOME_BASE", None)
                else:
                    os.environ["CHACK_CODEX_HOME_BASE"] = previous
        with patch.object(
            executor,
            "_run_codex_once",
            return_value=("fallback ok", [], None),
        ) as fallback_mock:
            result = executor._maybe_retry_with_api_key(
                "prompt",
                ("ERROR: 401 Unauthorized: Incorrect API key provided", [], None),
                allow_api_key_fallback=True,
                codex_exec_failed=True,
            )

        self.assertFalse(executor._use_codex_access_token)
        self.assertEqual(executor._openai_api_key, "sk-openai-direct")
        fallback_mock.assert_called_once_with("prompt", allow_api_key_fallback=False)
        self.assertEqual(result, ("fallback ok", [], None))
        self.assertFalse(os.path.exists(os.path.join(executor._codex_home, "auth.json")))

    def test_codex_access_token_does_not_fallback_on_successful_output_mentioning_401(self) -> None:
        config = _make_config("codex", "gpt-5.4")
        config.credentials.codex_access_token = _fake_chatgpt_access_token()
        config.openai_api_key = "sk-openai-direct"

        executor = build_codex_executor(
            config,
            system_prompt="system",
            max_turns=2,
            memory_max_messages=10,
            memory_reset_to_messages=5,
        )

        result = executor._maybe_retry_with_api_key(
            "prompt",
            ("Focus on HTTP 401 handling and authorization bypasses", [], None),
            allow_api_key_fallback=True,
            codex_exec_failed=False,
        )

        self.assertTrue(executor._use_codex_access_token)
        self.assertEqual(result, ("Focus on HTTP 401 handling and authorization bypasses", [], None))

    def test_codex_executor_spills_large_tool_payloads_to_file(self) -> None:
        config = _make_config("codex", "gpt-5.4")
        config.credentials.openrouter_api_key = ""
        config.credentials.codex_access_token = _fake_chatgpt_access_token()
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
                executor._ensure_codex_home_and_config()
                executor._serialized_tools_append_b64 = "x" * 30000

                env = executor._build_env()

                self.assertNotIn(CHACK_TOOLS_APPEND_B64_ENV, env)
                payload_path = env.get(CHACK_TOOLS_APPEND_B64_PATH_ENV, "")
                self.assertTrue(payload_path)
                with open(payload_path, "r", encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "x" * 30000)
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

    def test_claude_backend_prefers_oauth_token_when_anthropic_api_key_is_also_present(self) -> None:
        config = _make_config("claude", "claude-haiku-4-5")
        config.credentials.openrouter_api_key = ""
        config.credentials.anthropic_api_key = "sk-ant-api-fallback"

        with patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-primary",
                "ANTHROPIC_API_KEY": "sk-ant-api-fallback",
            },
            clear=False,
        ):
            executor = build_claude_executor(
                config,
                system_prompt="system",
                max_turns=2,
                memory_max_messages=10,
                memory_reset_to_messages=5,
            )
            env = executor._build_env()

        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-primary")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
        self.assertNotIn("CLAUDE_API_KEY", env)
        self.assertEqual(executor._anthropic_api_key, "sk-ant-api-fallback")

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
