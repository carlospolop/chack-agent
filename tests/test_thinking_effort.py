from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
    load_config,
)
from chack_agent import thinking_effort
from chack_agent.thinking_effort import (
    claude_thinking_effort,
    codex_thinking_effort,
    copilot_thinking_effort,
    gemini_thinking_config,
    THINKING_EFFORT_LEVELS,
    _model_key,
    normalize_thinking_effort,
    openai_thinking_effort,
    published_thinking_efforts,
    supported_thinking_efforts,
    validate_thinking_effort,
)
from chack_tools.subagent_config import build_subagent_config


def _config(provider: str, effort: str = "high") -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary="test-model", provider=provider),
        agent=AgentConfig(
            thinking_effort=effort,
            main_action="test",
            sub_action="test",
        ),
        session=SessionConfig(max_turns=3),
        tools=ToolsConfig(task_steps_manager_enabled=False),
        credentials=CredentialsConfig(
            openai_api_key="sk-test",
            openrouter_api_key="or-test",
            gemini_api_key="gemini-test",
            anthropic_api_key="anthropic-test",
            copilot_github_token="github-test",
        ),
        logging=LoggingConfig(),
        system_prompt="test",
        env={},
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "high"),
        ("", "high"),
        ("LOW", "low"),
        ("extra high", "xhigh"),
        ("extra-high", "xhigh"),
        ("very_high", "xhigh"),
        ("max", "max"),
    ],
)
def test_normalizes_cross_backend_effort_names(value, expected) -> None:
    assert normalize_thinking_effort(value) == expected


def test_rejects_unknown_thinking_effort() -> None:
    with pytest.raises(ValueError, match="Unsupported thinking_effort"):
        normalize_thinking_effort("enormous")


def test_backend_effort_mappings_cover_backend_vocabulary() -> None:
    assert openai_thinking_effort("max") == "max"
    assert codex_thinking_effort("none") == "none"
    assert codex_thinking_effort("max") == "max"
    assert copilot_thinking_effort("none") == "none"
    assert claude_thinking_effort("minimal") == "low"
    assert claude_thinking_effort("extra high") == "max"
    assert claude_thinking_effort(
        "extra high", {"low", "medium", "high", "xhigh", "max"}
    ) == "xhigh"
    assert copilot_thinking_effort("minimal") == "minimal"
    assert gemini_thinking_config("extra high", "gemini-2.5-pro") == {
        "includeThoughts": True,
        "thinkingBudget": 32768,
    }
    assert gemini_thinking_config("extra high", "gemini-3.1-pro-preview") == {
        "includeThoughts": True,
        "thinkingLevel": "HIGH",
    }


@pytest.mark.parametrize(
    ("effort", "openai", "codex", "claude_legacy", "claude_current", "copilot"),
    [
        # Only Claude Code takes --effort from a fixed choice list, so it is
        # the only column that can sit below the configured level.
        ("none", "none", "none", "low", "low", "none"),
        ("minimal", "minimal", "minimal", "low", "low", "minimal"),
        ("low", "low", "low", "low", "low", "low"),
        ("medium", "medium", "medium", "medium", "medium", "medium"),
        ("high", "high", "high", "high", "high", "high"),
        ("xhigh", "xhigh", "xhigh", "max", "xhigh", "xhigh"),
        ("max", "max", "max", "max", "max", "max"),
    ],
)
def test_complete_backend_effort_mapping_matrix(
    effort, openai, codex, claude_legacy, claude_current, copilot
) -> None:
    assert openai_thinking_effort(effort) == openai
    assert codex_thinking_effort(effort) == codex
    assert claude_thinking_effort(
        effort, {"low", "medium", "high", "max"}
    ) == claude_legacy
    assert claude_thinking_effort(
        effort, {"low", "medium", "high", "xhigh", "max"}
    ) == claude_current
    assert copilot_thinking_effort(effort) == copilot


@pytest.mark.parametrize(
    ("effort", "model", "expected"),
    [
        ("none", "gemini-2.5-pro", {"includeThoughts": True, "thinkingBudget": 128}),
        ("none", "gemini-2.5-flash", {"includeThoughts": True, "thinkingBudget": 0}),
        ("max", "gemini-2.5-flash", {"includeThoughts": True, "thinkingBudget": 24576}),
        ("medium", "gemini-3-pro-preview", {"includeThoughts": True, "thinkingLevel": "HIGH"}),
        ("medium", "gemini-3.1-pro-preview", {"includeThoughts": True, "thinkingLevel": "MEDIUM"}),
        ("minimal", "gemini-3-flash-preview", {"includeThoughts": True, "thinkingLevel": "MINIMAL"}),
    ],
)
def test_gemini_mapping_respects_model_specific_contracts(effort, model, expected) -> None:
    assert gemini_thinking_config(effort, model) == expected
    assert not ({"thinkingLevel", "thinkingBudget"} <= expected.keys())


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Anthropic: xhigh is newer than max, so 4.6 has max but not xhigh.
        ("claude-opus-4-6", {"low", "medium", "high", "max"}),
        ("claude-sonnet-4-6", {"low", "medium", "high", "max"}),
        ("claude-opus-4-7", {"low", "medium", "high", "xhigh", "max"}),
        ("claude-opus-4-8", {"low", "medium", "high", "xhigh", "max"}),
        ("claude-opus-5", {"low", "medium", "high", "xhigh", "max"}),
        ("claude-sonnet-5", {"low", "medium", "high", "xhigh", "max"}),
        ("claude-fable-5", {"low", "medium", "high", "xhigh", "max"}),
        # OpenAI: 5.0 kept minimal, 5.1 replaced it with none, 5.2+ added
        # xhigh, 5.6 added max.
        ("gpt-5", {"minimal", "low", "medium", "high"}),
        ("gpt-5-mini", {"minimal", "low", "medium", "high"}),
        ("gpt-5.1", {"none", "low", "medium", "high"}),
        ("gpt-5.2", {"none", "low", "medium", "high", "xhigh"}),
        ("gpt-5.4", {"none", "low", "medium", "high", "xhigh"}),
        ("gpt-5.4-nano", {"none", "low", "medium", "high", "xhigh"}),
        ("gpt-5.5", {"none", "low", "medium", "high", "xhigh"}),
        ("gpt-5.6-sol", {"none", "low", "medium", "high", "xhigh", "max"}),
        ("gemini-3.1-pro-preview", {"low", "medium", "high"}),
        ("gemini-3-flash-preview", {"minimal", "low", "medium", "high"}),
    ],
)
def test_published_table_supplies_supported_levels(model, expected) -> None:
    assert supported_thinking_efforts(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Levels no family rule would guess: the Pro tiers drop the cheap end,
        # and non-first-party vendors are not modelled by the rules at all.
        ("gpt-5.4-pro", {"medium", "high", "xhigh"}),
        ("gpt-5.1-codex", {"low", "medium", "high"}),
        ("gpt-5-pro", {"high"}),
        ("grok-4.5", {"low", "medium", "high"}),
        ("gpt-oss-120b", {"low", "medium", "high"}),
    ],
)
def test_published_table_beats_the_family_rules(model, expected) -> None:
    assert supported_thinking_efforts(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Gemini 2.5 spends a token budget instead of an effort enum, so
        # OpenRouter publishes no levels for it; only Pro cannot reach zero.
        ("gemini-2.5-pro", {"minimal", "low", "medium", "high", "xhigh", "max"}),
        (
            "gemini-2.5-flash",
            {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        ),
        ("gemini-3-pro-preview", {"low", "high"}),
        ("claude-opus-4-5-20251101", {"low", "medium", "high"}),
        ("claude-mythos-preview", {"low", "medium", "high", "max"}),
        # Models with no effort parameter keep only the no-op default.
        ("claude-haiku-4-5", {"high"}),
        ("claude-3-5-sonnet-20241022", {"high"}),
        ("gpt-4o", {"high"}),
        ("o3-mini", {"low", "medium", "high"}),
    ],
)
def test_family_rules_cover_models_the_published_table_omits(model, expected) -> None:
    assert _model_key(model) not in published_thinking_efforts()
    assert supported_thinking_efforts(model) == expected


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-opus-4.6",
        "anthropic/claude-opus-4.6",
        "openrouter/anthropic/claude-opus-4.6",
        "us.anthropic.claude-opus-4-6-v1:0",
        "claude-opus-4-6-20260205",
    ],
)
def test_every_spelling_of_one_model_resolves_to_the_same_levels(model) -> None:
    assert supported_thinking_efforts(model) == {"low", "medium", "high", "max"}


def test_missing_published_table_falls_back_to_the_family_rules(monkeypatch) -> None:
    import chack_agent.thinking_effort as module

    monkeypatch.setattr(module, "_PUBLISHED_EFFORTS_FILE", "/nonexistent/effort.yaml")
    monkeypatch.setattr(module, "_published_efforts_cache", None)

    assert module.published_thinking_efforts() == {}
    assert module.supported_thinking_efforts("claude-opus-4-6") == {
        "low",
        "medium",
        "high",
        "max",
    }


@pytest.mark.parametrize(
    "model",
    ["", "test-model", "openrouter/xiaomi/mimo-v2-flash", "some-future-model-9"],
)
def test_unknown_models_skip_validation(model) -> None:
    assert supported_thinking_efforts(model) is None
    assert validate_thinking_effort("max", model=model) == "max"


@pytest.mark.parametrize(
    ("effort", "model"),
    [
        ("xhigh", "claude-sonnet-4-6"),
        ("minimal", "claude-opus-5"),
        ("none", "claude-opus-4-7"),
        ("max", "gpt-5.4"),
        ("minimal", "gpt-5.4"),
        ("none", "gpt-5"),
        ("medium", "gemini-3-pro-preview"),
        ("xhigh", "gemini-3-flash-preview"),
        ("none", "gemini-2.5-pro"),
    ],
)
def test_rejects_levels_the_selected_model_does_not_support(effort, model) -> None:
    with pytest.raises(ValueError, match="not supported by model"):
        validate_thinking_effort(effort, model=model)


def test_default_steps_down_on_a_model_that_lacks_it(caplog) -> None:
    """A level the user never chose must not block startup."""
    model = "nemotron-3-super-120b-a12b"
    assert supported_thinking_efforts(model) == {"low", "medium"}

    # The warning is deduped per model, so start from a clean slate.
    thinking_effort._stepped_down_models.discard(model)
    with caplog.at_level("WARNING", logger="chack.thinking_effort"):
        assert validate_thinking_effort("high", model=model) == "medium"

    assert "does not offer the default high" in caplog.text
    # Every role that shares the model would otherwise repeat it.
    caplog.clear()
    assert validate_thinking_effort("high", model=model) == "medium"
    assert not caplog.text
    # An explicit choice the model cannot honour still fails loudly.
    with pytest.raises(ValueError, match="not supported by model"):
        validate_thinking_effort("xhigh", model=model)


def test_stepped_down_default_is_written_back_to_the_config(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: openrouter/nvidia/nemotron-3-super-120b-a12b
  provider: openrouter
  main_action: test
  sub_action: test
""".strip()
    )

    config = load_config(str(path))

    assert config.agent.thinking_effort == "medium"


def test_models_without_effort_control_accept_only_the_default() -> None:
    assert validate_thinking_effort("high", model="claude-haiku-4-5") == "high"
    with pytest.raises(ValueError, match="no configurable thinking effort"):
        validate_thinking_effort("low", model="claude-haiku-4-5")


def test_error_lists_the_valid_values_in_provider_order() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_thinking_effort("xhigh", model="claude-opus-4-6")

    assert "Supported values for this model: low, medium, high, max" in str(
        excinfo.value
    )


def test_yaml_rejects_effort_the_primary_model_cannot_use(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: claude-sonnet-4-6
  provider: claude
  main_action: test
  sub_action: test
  thinking_effort: xhigh
""".strip()
    )

    with pytest.raises(ValueError, match="agent.thinking_effort='xhigh'"):
        load_config(str(path))


def test_yaml_validates_each_role_against_its_own_model(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: claude-opus-4-7
  provider: claude
  main_action: test
  sub_action: test
  thinking_effort: xhigh
  scientific: claude-sonnet-4-6
  scientific_thinking_effort: xhigh
""".strip()
    )

    with pytest.raises(ValueError, match="agent.scientific_thinking_effort='xhigh'"):
        load_config(str(path))


def test_yaml_validates_role_local_tools_override(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: gpt-5.4
  provider: openai
  main_action: test
  sub_action: test
tools:
  scientific_agent:
    thinking_effort: max
""".strip()
    )

    with pytest.raises(
        ValueError, match="tools.scientific_agent.thinking_effort='max'"
    ):
        load_config(str(path))


def test_yaml_accepts_a_level_the_selected_model_supports(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: claude-opus-4-7
  provider: claude
  main_action: test
  sub_action: test
  thinking_effort: xhigh
  scientific: claude-opus-4-6
  scientific_thinking_effort: max
""".strip()
    )

    config = load_config(str(path))

    assert config.agent.thinking_effort == "xhigh"
    assert config.tools.scientific_agent["thinking_effort"] == "max"


def test_subagent_config_validates_effort_against_its_model() -> None:
    with pytest.raises(ValueError, match="not supported by model"):
        build_subagent_config(
            ToolsConfig(scientific_agent={"thinking_effort": "xhigh"}),
            model_name="claude-sonnet-4-6",
            model_provider="claude",
            max_turns=3,
            system_prompt="test",
            overrides={"agent": {"sub_action": "scientific"}},
        )


def _openrouter_payload() -> list[dict]:
    return [
        {
            "id": "anthropic/claude-opus-4.6",
            "reasoning": {"supported_efforts": ["max", "high", "medium", "low"]},
        },
        {
            "id": "anthropic/claude-opus-4.6:batch",
            "reasoning": {"supported_efforts": ["max", "high", "medium", "low"]},
        },
        {
            "id": "openai/gpt-5.6-sol",
            "reasoning": {
                "supported_efforts": ["max", "xhigh", "high", "medium", "low", "none"]
            },
        },
        # No reasoning block, an empty one, and an unknown level: all skipped.
        {"id": "google/gemini-2.5-pro", "reasoning": {"mandatory": True}},
        {"id": "meta-llama/llama-4"},
        {"id": "vendor/experimental", "reasoning": {"supported_efforts": ["ultra"]}},
    ]


def test_generated_table_is_keyed_the_way_the_loader_looks_models_up() -> None:
    from scripts.update_openrouter_pricing import _build_effort_yaml

    parsed = yaml.safe_load(_build_effort_yaml(_openrouter_payload()))

    assert parsed["models"] == {
        "claude-opus-4-6": ["low", "medium", "high", "max"],
        "gpt-5-6-sol": ["none", "low", "medium", "high", "xhigh", "max"],
    }
    for model in parsed["models"]:
        assert _model_key(model) == model


def test_generator_writes_both_files(tmp_path, monkeypatch) -> None:
    import scripts.update_openrouter_pricing as updater

    monkeypatch.setattr(
        updater, "_load_models", lambda models_url, api_key: _openrouter_payload()
    )
    pricing_path = tmp_path / "pricing.yaml"
    effort_path = tmp_path / "thinking_effort.yaml"

    updater.update_pricing(
        output_path=pricing_path,
        models_url="https://example.invalid/models",
        api_key=None,
        effort_output_path=effort_path,
    )

    assert "anthropic/claude-opus-4.6:" in pricing_path.read_text()
    assert "claude-opus-4-6: [low, medium, high, max]" in effort_path.read_text()


def test_shipped_table_parses_and_only_uses_known_levels() -> None:
    table = published_thinking_efforts()

    assert len(table) > 50
    assert table["claude-opus-4-6"] == frozenset({"low", "medium", "high", "max"})
    for model, levels in table.items():
        assert levels <= THINKING_EFFORT_LEVELS
        assert _model_key(model) == model


def test_agent_default_is_high() -> None:
    assert AgentConfig().thinking_effort == "high"


@pytest.mark.parametrize(
    "provider",
    ["openai", "openrouter", "codex", "claude", "copilot", "gemini", "langgraph"],
)
def test_every_provider_and_backend_defaults_to_high(provider) -> None:
    assert _config(provider).agent.thinking_effort == "high"


def test_yaml_loads_and_normalizes_main_agent_effort(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: test-model
  provider: openai
  main_action: test
  sub_action: test
  thinking_effort: extra high
""".strip()
    )

    assert load_config(str(path)).agent.thinking_effort == "xhigh"


def test_flat_role_effort_is_normalized_and_routed_to_nested_agent(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: test-model
  provider: openai
  main_action: test
  sub_action: test
  scientific_thinking_effort: extra high
""".strip()
    )

    config = load_config(str(path))

    assert config.model.scientific_thinking_effort == "xhigh"
    assert config.tools.scientific_agent["thinking_effort"] == "xhigh"


def test_role_local_effort_takes_precedence_over_flat_role_effort(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
system_prompt: test
agent:
  primary: test-model
  provider: openai
  main_action: test
  sub_action: test
  scientific_thinking_effort: medium
tools:
  scientific_agent:
    thinking_effort: low
""".strip()
    )

    assert load_config(str(path)).tools.scientific_agent["thinking_effort"] == "low"


@pytest.mark.parametrize(
    ("sub_action", "field"),
    [
        ("social", "social_network_agent"),
        ("scientific", "scientific_agent"),
        ("webresearcher", "websearcher_agent"),
        ("business", "business_agent"),
        ("product", "product_agent"),
        ("legal", "legal_agent"),
        ("data_statistics", "data_statistics_agent"),
        ("news_media", "news_media_agent"),
        ("knowledge_graph", "knowledge_graph_agent"),
        ("religious", "religious_agent"),
        ("cli", "cli_agent"),
        ("subchack", "subchack_agent"),
        ("researcher_administrator", "researcher_administrator_agent"),
        ("researcher_queue_merge", "researcher_queue_agent"),
    ],
)
def test_every_nested_agent_type_can_override_effort(sub_action, field) -> None:
    tools = ToolsConfig()
    setattr(tools, field, {"thinking_effort": "low"})

    config = build_subagent_config(
        tools,
        model_name="test-model",
        model_provider="openai",
        max_turns=3,
        system_prompt="test",
        overrides={"agent": {"sub_action": sub_action}},
    )

    assert config.agent.thinking_effort == "low"


def test_nested_agent_default_and_explicit_override() -> None:
    default = build_subagent_config(
        ToolsConfig(),
        model_name="test-model",
        model_provider="openai",
        max_turns=3,
        system_prompt="test",
        overrides={"agent": {"sub_action": "scientific"}},
    )
    explicit = build_subagent_config(
        ToolsConfig(scientific_agent={"thinking_effort": "low"}),
        model_name="test-model",
        model_provider="openai",
        max_turns=3,
        system_prompt="test",
        overrides={
            "agent": {
                "sub_action": "scientific",
                "thinking_effort": "medium",
            }
        },
    )

    assert default.agent.thinking_effort == "high"
    assert explicit.agent.thinking_effort == "medium"


def test_mcp_researchers_are_eagerly_defaulted_to_high(monkeypatch) -> None:
    from chack_agent.backends.chack_tools_mcp_server import (
        _MCP_ROLE_AGENT_FIELDS,
        _mcp_tools_config,
    )

    monkeypatch.delenv("CHACK_THINKING_EFFORT", raising=False)
    for role in _MCP_ROLE_AGENT_FIELDS:
        monkeypatch.delenv(
            f"CHACK_{role.upper()}_THINKING_EFFORT",
            raising=False,
        )

    tools = _mcp_tools_config({})

    for field_name in _MCP_ROLE_AGENT_FIELDS.values():
        assert getattr(tools, field_name)["thinking_effort"] == "high"


def test_mcp_researcher_effort_preserves_config_and_supports_env_overrides(
    monkeypatch,
) -> None:
    from chack_agent.backends.chack_tools_mcp_server import _mcp_tools_config

    monkeypatch.setenv("CHACK_SCIENTIFIC_THINKING_EFFORT", "extra high")
    tools = _mcp_tools_config(
        {
            "scientific_agent": {"thinking_effort": "low"},
            "researcher_administrator_agent": {"thinking_effort": "medium"},
            "researcher_queue_agent": {"thinking_effort": "minimal"},
        }
    )

    assert tools.scientific_agent["thinking_effort"] == "xhigh"
    assert tools.researcher_administrator_agent["thinking_effort"] == "medium"
    assert tools.researcher_queue_agent["thinking_effort"] == "minimal"

    scientific = build_subagent_config(
        tools,
        model_name="test-model",
        model_provider="openai",
        max_turns=3,
        system_prompt="test",
        overrides={"agent": {"sub_action": "scientific"}},
    )
    administrator = build_subagent_config(
        tools,
        model_name="test-model",
        model_provider="openai",
        max_turns=3,
        system_prompt="test",
        overrides={"agent": {"sub_action": "researcher_administrator"}},
    )
    queue_merge = build_subagent_config(
        tools,
        model_name="test-model",
        model_provider="openai",
        max_turns=3,
        system_prompt="test",
        overrides={"agent": {"sub_action": "researcher_queue_merge"}},
    )

    assert scientific.agent.thinking_effort == "xhigh"
    assert administrator.agent.thinking_effort == "medium"
    assert queue_merge.agent.thinking_effort == "minimal"


def test_mcp_global_effort_configures_every_researcher_role(monkeypatch) -> None:
    from chack_agent.backends.chack_tools_mcp_server import (
        _MCP_ROLE_AGENT_FIELDS,
        _mcp_tools_config,
    )

    monkeypatch.setenv("CHACK_THINKING_EFFORT", "low")
    tools = _mcp_tools_config(
        {"scientific_agent": {"thinking_effort": "high"}}
    )

    for field_name in _MCP_ROLE_AGENT_FIELDS.values():
        assert getattr(tools, field_name)["thinking_effort"] == "low"


def test_mcp_loader_round_trips_serialized_researcher_effort(monkeypatch) -> None:
    import chack_agent.backends.chack_tools_mcp_server as server

    captured = {}

    class FakeToolset:
        def __init__(self, config, **kwargs):
            captured["config"] = config
            captured["kwargs"] = kwargs
            self.tools = []

    monkeypatch.setattr(server, "AgentsToolset", FakeToolset)
    monkeypatch.setenv("CHACK_MODEL_PROVIDER", "openai")
    monkeypatch.setenv(
        "CHACK_TOOLS_CONFIG_JSON",
        json.dumps(
            {
                "researcher_administrator_enabled": True,
                "scientific_agent": {"thinking_effort": "medium"},
                "researcher_administrator_agent": {"thinking_effort": "xhigh"},
                "researcher_queue_agent": {"thinking_effort": "low"},
            }
        ),
    )
    for name in (
        "CHACK_THINKING_EFFORT",
        "CHACK_SCIENTIFIC_THINKING_EFFORT",
        "CHACK_RESEARCHER_ADMINISTRATOR_THINKING_EFFORT",
        "CHACK_RESEARCHER_QUEUE_THINKING_EFFORT",
        "CHACK_ALLOWED_TOOLS_JSON",
        "CHACK_TOOLS_OVERRIDE_B64",
        "CHACK_TOOLS_APPEND_B64",
    ):
        monkeypatch.delenv(name, raising=False)

    assert server._load_toolset() == []
    tools = captured["config"]
    assert tools.scientific_agent["thinking_effort"] == "medium"
    assert tools.researcher_administrator_agent["thinking_effort"] == "xhigh"
    assert tools.researcher_queue_agent["thinking_effort"] == "low"
    assert tools.business_agent["thinking_effort"] == "high"


@pytest.mark.parametrize(
    ("provider", "backend_module"),
    [
        ("codex", "chack_agent.backends.codex_backend"),
        ("claude", "chack_agent.backends.claude_code_backend"),
        ("copilot", "chack_agent.backends.copilot_cli_backend"),
        ("gemini", "chack_agent.backends.gemini_cli_backend"),
    ],
)
def test_every_cli_mcp_transport_serializes_researcher_effort(
    provider,
    backend_module,
    monkeypatch,
) -> None:
    import importlib

    from chack_agent.backends.chack_tools_mcp_server import _mcp_tools_config

    monkeypatch.delenv("CHACK_THINKING_EFFORT", raising=False)
    monkeypatch.delenv("CHACK_SCIENTIFIC_THINKING_EFFORT", raising=False)
    monkeypatch.delenv(
        "CHACK_RESEARCHER_ADMINISTRATOR_THINKING_EFFORT",
        raising=False,
    )
    monkeypatch.delenv("CHACK_RESEARCHER_QUEUE_THINKING_EFFORT", raising=False)
    config = _config(provider)
    config.tools.scientific_agent = {"thinking_effort": "medium"}
    executor = importlib.import_module(backend_module).build_executor(
        config,
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    transported = _mcp_tools_config(json.loads(executor._tools_config_json))

    assert transported.scientific_agent["thinking_effort"] == "medium"
    assert transported.researcher_administrator_agent["thinking_effort"] == "high"
    assert transported.researcher_queue_agent["thinking_effort"] == "high"


def test_openai_backend_passes_effort_to_model_settings() -> None:
    from chack_agent.backends.openai_compaction_backend import build_executor

    executor = build_executor(
        _config("openai", "low"),
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )

    assert executor.agent.model_settings.reasoning.effort == "low"


def test_openrouter_backend_passes_effort_to_model_settings() -> None:
    from chack_agent.backends.openrouter_openai_backend import build_executor

    executor = build_executor(
        _config("openrouter", "medium"),
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )

    assert executor.agent.model_settings.reasoning.effort == "medium"
    assert executor._summary_agent.model_settings.reasoning.effort == "medium"


def test_codex_backend_passes_effort_on_new_and_resumed_commands() -> None:
    from chack_agent.backends.codex_backend import build_executor

    executor = build_executor(
        _config("codex", "extra high"),
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    expected = 'model_reasoning_effort="xhigh"'
    assert expected in executor._build_command()
    executor._thread_id = "thread-1"
    assert expected in executor._build_command()


def test_claude_backend_passes_native_effort_flag(monkeypatch) -> None:
    import chack_agent.backends.claude_code_backend as backend

    # Keep this test hermetic: the host may have a newer Claude CLI that
    # advertises xhigh. This case intentionally verifies the legacy CLI mapping.
    monkeypatch.setattr(
        backend,
        "_claude_supported_effort_levels",
        lambda _path: frozenset({"low", "medium", "high", "max"}),
    )

    executor = backend.build_executor(
        _config("claude", "extra high"),
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    command = executor._build_command("prompt")

    assert command[command.index("--effort") + 1] == "max"


def test_claude_backend_detects_cli_specific_effort_levels(monkeypatch) -> None:
    import chack_agent.backends.claude_code_backend as backend

    backend._claude_supported_effort_levels.cache_clear()
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="--effort <level>  choices: low, medium, high, xhigh, max"
        ),
    )

    supported = backend._claude_supported_effort_levels("claude-new")

    assert supported == {"low", "medium", "high", "xhigh", "max"}
    assert claude_thinking_effort("xhigh", supported) == "xhigh"
    backend._claude_supported_effort_levels.cache_clear()


def test_copilot_backend_passes_native_effort_flag() -> None:
    from chack_agent.backends.copilot_cli_backend import build_executor

    executor = build_executor(
        _config("copilot", "minimal"),
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    command = executor._build_command("prompt")

    assert command[command.index("--reasoning-effort") + 1] == "minimal"


def test_gemini_backend_writes_native_thinking_config(tmp_path, monkeypatch) -> None:
    from chack_agent.backends.gemini_cli_backend import build_executor

    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path))
    executor = build_executor(
        _config("gemini", "low"),
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    executor._ensure_gemini_home_and_config()
    settings = json.loads(
        (tmp_path / "default" / "settings.json").read_text()
    )
    thinking = settings["modelConfigs"]["overrides"][0]["modelConfig"][
        "generateContentConfig"
    ]["thinkingConfig"]

    assert thinking == {"includeThoughts": True, "thinkingLevel": "LOW"}


@pytest.mark.parametrize(
    ("model", "expected_key"),
    [
        ("gemini-2.5-flash", "thinkingBudget"),
        ("gemini-3-flash-preview", "thinkingLevel"),
    ],
)
def test_gemini_backend_writes_only_family_compatible_control(
    tmp_path, monkeypatch, model, expected_key
) -> None:
    from chack_agent.backends.gemini_cli_backend import build_executor

    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path))
    config = _config("gemini", "high")
    config.model.primary = model
    executor = build_executor(
        config,
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    executor._ensure_gemini_home_and_config()
    settings = json.loads((tmp_path / "default" / "settings.json").read_text())
    thinking = settings["modelConfigs"]["overrides"][0]["modelConfig"][
        "generateContentConfig"
    ]["thinkingConfig"]

    assert expected_key in thinking
    assert len({"thinkingLevel", "thinkingBudget"} & thinking.keys()) == 1


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "xhigh", "max"])
def test_configured_effort_reaches_openai_style_backends_unchanged(effort) -> None:
    """A non-default level must survive to the model, not snap to a neighbour."""
    from chack_agent.backends.openai_compaction_backend import (
        build_executor as openai_executor,
    )
    from chack_agent.backends.openrouter_openai_backend import (
        build_executor as openrouter_executor,
    )

    kwargs = dict(
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    openai = openai_executor(_config("openai", effort), **kwargs)
    openrouter = openrouter_executor(_config("openrouter", effort), **kwargs)

    assert openai.agent.model_settings.reasoning.effort == effort
    assert openrouter.agent.model_settings.reasoning.effort == effort
    assert openrouter._summary_agent.model_settings.reasoning.effort == effort


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "xhigh", "max"])
def test_configured_effort_reaches_cli_backends_unchanged(effort) -> None:
    from chack_agent.backends.codex_backend import build_executor as codex_executor
    from chack_agent.backends.copilot_cli_backend import (
        build_executor as copilot_executor,
    )

    kwargs = dict(
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )
    codex = codex_executor(_config("codex", effort), **kwargs)
    copilot = copilot_executor(_config("copilot", effort), **kwargs)
    copilot_command = copilot._build_command("prompt")

    assert f'model_reasoning_effort="{effort}"' in codex._build_command()
    assert copilot_command[copilot_command.index("--reasoning-effort") + 1] == effort


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gemini-2.5-pro", "minimal", {"thinkingBudget": 128}),
        ("gemini-2.5-pro", "low", {"thinkingBudget": 2048}),
        ("gemini-2.5-flash", "none", {"thinkingBudget": 0}),
        ("gemini-3-pro-preview", "low", {"thinkingLevel": "LOW"}),
        ("gemini-3-pro-preview", "high", {"thinkingLevel": "HIGH"}),
        ("gemini-3.1-pro-preview", "medium", {"thinkingLevel": "MEDIUM"}),
        ("gemini-3-flash-preview", "minimal", {"thinkingLevel": "MINIMAL"}),
        ("gemini-3-flash-preview", "medium", {"thinkingLevel": "MEDIUM"}),
    ],
)
def test_gemini_keeps_every_level_its_model_accepts(model, effort, expected) -> None:
    assert effort in supported_thinking_efforts(model)
    assert gemini_thinking_config(effort, model) == {
        "includeThoughts": True,
        **expected,
    }


def test_claude_cli_downgrade_is_reported(caplog) -> None:
    with caplog.at_level("WARNING", logger="chack.thinking_effort"):
        assert claude_thinking_effort("xhigh", {"low", "medium", "high", "max"}) == "max"

    assert "does not accept --effort xhigh" in caplog.text


def test_claude_cli_that_supports_the_level_stays_silent(caplog) -> None:
    with caplog.at_level("WARNING", logger="chack.thinking_effort"):
        assert claude_thinking_effort(
            "xhigh", {"low", "medium", "high", "xhigh", "max"}
        ) == "xhigh"

    assert not caplog.text


def test_langgraph_backend_passes_effort_to_openrouter_model(monkeypatch) -> None:
    import chack_agent.backends.langgraph_backend as backend

    seen: list[dict] = []

    class FakeModel:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def bind_tools(self, _tools):
            return self

    real_import = backend.importlib.import_module

    def fake_import(name):
        if name == "langchain_openai":
            return SimpleNamespace(ChatOpenAI=FakeModel)
        return real_import(name)

    monkeypatch.setattr(backend.importlib, "import_module", fake_import)
    monkeypatch.setattr(backend.LangGraphExecutor, "build_graph", lambda self: None)
    backend.build_executor(
        _config("langgraph", "max"),
        system_prompt="test",
        max_turns=3,
        memory_max_messages=3,
        memory_reset_to_messages=1,
        tools_override=[],
    )

    assert seen[0]["reasoning_effort"] == "max"
