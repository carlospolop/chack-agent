from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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
from chack_agent.thinking_effort import (
    claude_thinking_effort,
    codex_thinking_effort,
    copilot_thinking_effort,
    gemini_thinking_config,
    normalize_thinking_effort,
    openai_thinking_effort,
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
    assert openai_thinking_effort("max") == "xhigh"
    assert codex_thinking_effort("none") == "minimal"
    assert codex_thinking_effort("max") == "xhigh"
    assert claude_thinking_effort("minimal") == "low"
    assert claude_thinking_effort("extra high") == "max"
    assert claude_thinking_effort(
        "extra high", {"low", "medium", "high", "xhigh", "max"}
    ) == "xhigh"
    assert copilot_thinking_effort("minimal") == "low"
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
        ("none", "none", "minimal", "low", "low", "low"),
        ("minimal", "minimal", "minimal", "low", "low", "low"),
        ("low", "low", "low", "low", "low", "low"),
        ("medium", "medium", "medium", "medium", "medium", "medium"),
        ("high", "high", "high", "high", "high", "high"),
        ("xhigh", "xhigh", "xhigh", "max", "xhigh", "xhigh"),
        ("max", "xhigh", "xhigh", "max", "max", "max"),
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


def test_claude_backend_passes_native_effort_flag() -> None:
    from chack_agent.backends.claude_code_backend import build_executor

    executor = build_executor(
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

    assert command[command.index("--reasoning-effort") + 1] == "low"


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
