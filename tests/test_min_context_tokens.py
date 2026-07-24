import json
import os
import tempfile
import unittest
from unittest.mock import patch

from chack_agent.config import (
    CLI_BACKEND_MIN_CONTEXT_TOKENS,
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
    resolve_config_aliases,
)
from chack_agent.backends.codex_backend import (
    CodexExecutor,
    build_executor as build_codex_executor,
)
from chack_agent.backends.claude_code_backend import (
    ClaudeCodeExecutor,
    _CLAUDE_1M_CONTEXT_BETA,
)


def _make_config(provider: str, max_context_tokens: int) -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(
            primary="gpt-5-mini",
            provider=provider,
            max_context_tokens=max_context_tokens,
        ),
        agent=AgentConfig(main_action="test", sub_action="run"),
        session=SessionConfig(max_turns=2),
        tools=ToolsConfig(),
        credentials=CredentialsConfig(openai_api_key="sk-test"),
        logging=LoggingConfig(),
        system_prompt="test system prompt",
        env={},
    )


class MinContextTokensConfigTests(unittest.TestCase):
    """The Codex / Claude Code backends must run with at least 250k context."""

    def setUp(self) -> None:
        # Avoid any network round-trip to the remote alias service.
        patcher_model = patch(
            "chack_agent.model_aliases._get_model_aliases", return_value={}
        )
        patcher_backend = patch(
            "chack_agent.model_aliases._get_backend_aliases", return_value={}
        )
        patcher_model.start()
        patcher_backend.start()
        self.addCleanup(patcher_model.stop)
        self.addCleanup(patcher_backend.stop)

    def test_floors_sub_minimum_for_codex(self) -> None:
        config = resolve_config_aliases(_make_config("codex", 100_000))
        self.assertEqual(
            config.model.max_context_tokens, CLI_BACKEND_MIN_CONTEXT_TOKENS
        )

    def test_floors_sub_minimum_for_claude_variants(self) -> None:
        for provider in ("claude", "claude-code", "claude_code", "anthropic"):
            with self.subTest(provider=provider):
                config = resolve_config_aliases(_make_config(provider, 50_000))
                self.assertEqual(
                    config.model.max_context_tokens,
                    CLI_BACKEND_MIN_CONTEXT_TOKENS,
                )

    def test_keeps_value_at_or_above_minimum(self) -> None:
        config = resolve_config_aliases(_make_config("codex", 1_000_000))
        self.assertEqual(config.model.max_context_tokens, 1_000_000)

    def test_leaves_unset_value_untouched(self) -> None:
        # 0 means "use the model's native window" -- do not force a floor that
        # could shrink a large native context window.
        config = resolve_config_aliases(_make_config("codex", 0))
        self.assertEqual(config.model.max_context_tokens, 0)

    def test_does_not_floor_other_providers(self) -> None:
        config = resolve_config_aliases(_make_config("openrouter", 100_000))
        self.assertEqual(config.model.max_context_tokens, 100_000)


class CodexContextWindowConfigTests(unittest.TestCase):
    """The floored value is written into the Codex config.toml."""

    def _build(self, max_context_tokens: int, tmpdir: str):
        config = _make_config("codex", max_context_tokens)
        config = resolve_config_aliases(config)
        executor = build_codex_executor(
            config,
            system_prompt="system",
            max_turns=2,
            memory_max_messages=10,
            memory_reset_to_messages=5,
        )
        executor._ensure_codex_home_and_config()
        with open(
            os.path.join(executor._codex_home, "config.toml"), "r", encoding="utf-8"
        ) as handle:
            return handle.read()

    def test_writes_hermes_style_auto_compact_limit_when_configured(self) -> None:
        with patch("chack_agent.model_aliases._get_model_aliases", return_value={}), patch(
            "chack_agent.model_aliases._get_backend_aliases", return_value={}
        ), tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"CHACK_CODEX_HOME_BASE": tmpdir}):
                body = self._build(100_000, tmpdir)
        # Sub-minimum capacity is floored to 250k, while the default 50% policy
        # triggers Codex's native summarizing compactor at 125k.
        self.assertIn(
            f"model_auto_compact_token_limit = {CLI_BACKEND_MIN_CONTEXT_TOKENS // 2}", body
        )
        self.assertNotIn("model_context_window", body)

    def test_honors_explicit_compaction_threshold_ratio(self) -> None:
        with patch("chack_agent.model_aliases._get_model_aliases", return_value={}), patch(
            "chack_agent.model_aliases._get_backend_aliases", return_value={}
        ), tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"CHACK_CODEX_HOME_BASE": tmpdir}):
                config = resolve_config_aliases(_make_config("codex", 250_000))
                config.agent.compaction_threshold_ratio = 0.40
                executor = build_codex_executor(
                    config,
                    system_prompt="system",
                    max_turns=2,
                    memory_max_messages=10,
                    memory_reset_to_messages=5,
                )
                executor._ensure_codex_home_and_config()
                assert executor._codex_home is not None
                with open(
                    os.path.join(executor._codex_home, "config.toml"),
                    encoding="utf-8",
                ) as handle:
                    body = handle.read()
        self.assertIn("model_auto_compact_token_limit = 100000", body)

    def test_seventy_five_percent_is_trigger_not_retained_context(self) -> None:
        with patch("chack_agent.model_aliases._get_model_aliases", return_value={}), patch(
            "chack_agent.model_aliases._get_backend_aliases", return_value={}
        ), tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"CHACK_CODEX_HOME_BASE": tmpdir}):
                config = resolve_config_aliases(_make_config("codex", 250_000))
                config.agent.compaction_threshold_ratio = 0.75
                executor = build_codex_executor(
                    config,
                    system_prompt="system",
                    max_turns=2,
                    memory_max_messages=250,
                    memory_reset_to_messages=12,
                )
                executor._ensure_codex_home_and_config()
                assert executor._codex_home is not None
                with open(
                    os.path.join(executor._codex_home, "config.toml"),
                    encoding="utf-8",
                ) as handle:
                    body = handle.read()

        # Codex receives the point at which its native summarizing compactor
        # starts. Nothing configures it to retain 75% of the transcript.
        self.assertIn("model_auto_compact_token_limit = 187500", body)
        self.assertNotIn("model_context_window = 187500", body)

    def test_omits_auto_compact_limit_when_unset(self) -> None:
        with patch("chack_agent.model_aliases._get_model_aliases", return_value={}), patch(
            "chack_agent.model_aliases._get_backend_aliases", return_value={}
        ), tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"CHACK_CODEX_HOME_BASE": tmpdir}):
                body = self._build(0, tmpdir)
        self.assertNotIn("model_auto_compact_token_limit", body)

    def test_non_strict_codex_schema_preserves_declared_optional_patch_fields(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "Title": {"type": "string"},
                "Description": {"type": "string"},
            },
            "required": ["Title"],
        }

        normalized = CodexExecutor._normalize_codex_output_schema(
            schema,
            force_all_required=False,
        )

        self.assertEqual(normalized["required"], ["Title"])
        self.assertNotIn("Description", normalized["required"])

    def test_strict_codex_schema_still_requires_every_declared_property(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "Title": {"type": "string"},
                "Description": {"type": "string"},
            },
            "required": ["Title"],
        }

        normalized = CodexExecutor._normalize_codex_output_schema(schema)

        self.assertEqual(normalized["required"], ["Title", "Description"])

    def test_codex_executor_writes_optional_fields_for_non_strict_patch_schema(
        self,
    ) -> None:
        with patch("chack_agent.model_aliases._get_model_aliases", return_value={}), patch(
            "chack_agent.model_aliases._get_backend_aliases", return_value={}
        ), tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"CHACK_CODEX_HOME_BASE": tmpdir}):
                config = resolve_config_aliases(_make_config("codex", 250_000))
                config.agent.output_schema_strict = False
                config.agent.output_schema_json = {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "Title": {"type": "string"},
                        "Description": {"type": "string"},
                    },
                    "required": ["Title"],
                }
                executor = build_codex_executor(
                    config,
                    system_prompt="system",
                    max_turns=2,
                    memory_max_messages=10,
                    memory_reset_to_messages=5,
                )
                executor._ensure_codex_home_and_config()
                assert executor._output_schema_path is not None
                with open(executor._output_schema_path, encoding="utf-8") as handle:
                    written = json.load(handle)

        self.assertEqual(written["required"], ["Title"])
        self.assertNotIn("Description", written["required"])


def _build_claude_executor(
    max_context_tokens: int = 0,
    uses_openrouter_route: bool = False,
    claude_access_token: str = "",
) -> ClaudeCodeExecutor:
    return ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=0,
        _memory_reset_to=0,
        _base_system_prompt="",
        _model_name="claude-sonnet-4-6",
        _max_turns=10,
        _claude_cli_path="claude",
        _tools_config_json="{}",
        _allowed_tools_json="[]",
        _serialized_tools_override_b64="",
        _serialized_tools_append_b64="",
        _model_provider="claude",
        _default_model="",
        _social_network_model="",
        _scientific_model="",
        _websearcher_model="",
        _business_model="",
        _product_model="",
        _legal_model="",
        _data_statistics_model="",
        _news_media_model="",
        _knowledge_graph_model="",
        _religious_model="",
        _cli_model="",
        _subchack_model="",
        _researcher_administrator_model="",
        _social_network_max_turns=0,
        _scientific_max_turns=0,
        _websearcher_max_turns=0,
        _business_max_turns=0,
        _product_max_turns=0,
        _legal_max_turns=0,
        _data_statistics_max_turns=0,
        _news_media_max_turns=0,
        _knowledge_graph_max_turns=0,
        _religious_max_turns=0,
        _cli_max_turns=0,
        _subchack_max_turns=0,
        _researcher_administrator_max_turns=0,
        _min_tools_used=0,
        _max_tools_used=0,
        _require_task_steps_manager_init_first=False,
        _output_schema_json="",
        _max_context_tokens=max_context_tokens,
        _uses_openrouter_route=uses_openrouter_route,
        _claude_access_token=claude_access_token,
    )


class ClaudeContextBetaTests(unittest.TestCase):
    """Claude Code opts into the 1M-context beta when the budget exceeds 200k."""

    def _betas_values(self, command: list) -> list:
        return [command[i + 1] for i, tok in enumerate(command) if tok == "--betas"]

    def test_enables_1m_beta_above_default_window(self) -> None:
        executor = _build_claude_executor(max_context_tokens=250_000)
        command = executor._build_command("prompt")
        self.assertIn("--betas", command)
        self.assertIn(_CLAUDE_1M_CONTEXT_BETA, self._betas_values(command))

    def test_no_beta_at_or_below_default_window(self) -> None:
        for value in (0, 200_000):
            with self.subTest(value=value):
                executor = _build_claude_executor(max_context_tokens=value)
                command = executor._build_command("prompt")
                self.assertNotIn("--betas", command)

    def test_no_beta_on_openrouter_route(self) -> None:
        executor = _build_claude_executor(
            max_context_tokens=250_000, uses_openrouter_route=True
        )
        command = executor._build_command("prompt")
        self.assertNotIn("--betas", command)

    def test_no_beta_for_claude_oauth_token(self) -> None:
        executor = _build_claude_executor(
            max_context_tokens=250_000,
            claude_access_token="oauth-token",
        )
        command = executor._build_command("prompt")
        self.assertNotIn("--betas", command)

    def test_sets_auto_compact_window_env_when_configured(self) -> None:
        executor = _build_claude_executor(max_context_tokens=250_000)
        env = executor._build_env()
        self.assertEqual(env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW"), "250000")

    def test_caps_auto_compact_window_for_claude_oauth_token(self) -> None:
        executor = _build_claude_executor(
            max_context_tokens=250_000,
            claude_access_token="oauth-token",
        )
        env = executor._build_env()
        self.assertEqual(env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW"), "200000")

    def test_no_auto_compact_window_env_when_unset(self) -> None:
        executor = _build_claude_executor(max_context_tokens=0)
        env = executor._build_env()
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", env)


if __name__ == "__main__":
    unittest.main()
