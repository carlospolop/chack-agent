from chack_tools.config import ToolsConfig
from chack_tools.cli_research_agent import _CLI_RESEARCH_SYSTEM_PROMPT
from chack_tools.open_research_agents import _DATA_STATS_PROMPT
from chack_tools.scientific_research_agent import _SCIENTIFIC_AGENT_SYSTEM_PROMPT
from chack_tools.subagent_config import (
    RESEARCHER_COMMON_SYSTEM_PROMPT,
    RESEARCHER_OUTPUT_SCHEMA,
    RESEARCHER_OUTPUT_SCHEMA_NO_ARTIFACTS,
    build_subagent_config,
    create_subagent_session_id,
    inherit_subagent_limits,
)


DANGEROUS_EXEC_WARNING = (
    "- IMPORTANT: NEVER UNDER ANY CONCEPT EXECUTE ANY POTENTIALLY DANGEROUS PROGRAM "
    "(MALWARE, VIRUS, C2, REV SHELL) UNDER ANY CIRCUNSTANCES"
)


def test_local_command_researcher_prompts_warn_against_dangerous_execution():
    assert DANGEROUS_EXEC_WARNING in _CLI_RESEARCH_SYSTEM_PROMPT
    assert DANGEROUS_EXEC_WARNING in _SCIENTIFIC_AGENT_SYSTEM_PROMPT
    assert DANGEROUS_EXEC_WARNING in _DATA_STATS_PROMPT


def test_build_subagent_config_prepends_common_researcher_prompt_once():
    config = build_subagent_config(
        ToolsConfig(),
        model_name="gpt-test",
        model_provider="openai",
        max_turns=3,
        system_prompt="### SPECIFIC\nResearch this domain carefully.",
    )

    assert config.system_prompt.startswith(RESEARCHER_COMMON_SYSTEM_PROMPT)
    assert config.system_prompt.count("### RESEARCHER SPECIALIZATION") == 1
    assert "### SPECIFIC" in config.system_prompt


def test_build_subagent_config_does_not_duplicate_common_researcher_prompt():
    prompt = f"{RESEARCHER_COMMON_SYSTEM_PROMPT}\n\n### SPECIFIC\nResearch this domain carefully."

    config = build_subagent_config(
        ToolsConfig(),
        model_name="gpt-test",
        model_provider="openai",
        max_turns=3,
        system_prompt=prompt,
    )

    assert config.system_prompt.count("### RESEARCHER SPECIALIZATION") == 1


def test_build_subagent_config_defaults_to_no_artifact_output_schema():
    config = build_subagent_config(
        ToolsConfig(),
        model_name="gpt-test",
        model_provider="openai",
        max_turns=3,
        system_prompt="### SPECIFIC\nResearch this domain carefully.",
    )

    assert config.agent.output_schema_name == "researcher_result"
    assert config.agent.output_schema_strict is True
    assert config.agent.output_schema_json == RESEARCHER_OUTPUT_SCHEMA_NO_ARTIFACTS
    assert set(config.agent.output_schema_json["required"]) == {
        "research_worked",
        "failure_reason",
        "final_research_review",
    }


def test_build_subagent_config_uses_artifact_schema_when_preserved():
    config = build_subagent_config(
        ToolsConfig(),
        model_name="gpt-test",
        model_provider="openai",
        max_turns=3,
        system_prompt="### SPECIFIC\nResearch this domain carefully.",
        overrides={"env": {"CHACK_RESEARCH_SAVE_ARTIFACTS": "1"}},
    )

    assert config.agent.output_schema_json == RESEARCHER_OUTPUT_SCHEMA
    assert set(config.agent.output_schema_json["required"]) == {
        "research_worked",
        "failure_reason",
        "final_research_review",
        "evidence_data_path",
        "key_artifacts",
    }
    artifact_schema = config.agent.output_schema_json["properties"]["key_artifacts"]["items"]
    assert artifact_schema["required"] == ["filename", "source_url", "description"]
    assert "path" not in artifact_schema["properties"]
    assert "filename" in artifact_schema["properties"]
    assert artifact_schema["properties"]["description"]["minLength"] == 100
    assert artifact_schema["properties"]["description"]["maxLength"] == 300


def test_create_subagent_session_id_isolates_sibling_researchers():
    first = create_subagent_session_id("scientific", "parent/session 1")
    second = create_subagent_session_id("product", "parent/session 1")
    third = create_subagent_session_id("scientific", "parent/session 1")

    assert first != second
    assert first != third
    assert first.startswith("parent_session_1:scientific:")
    assert second.startswith("parent_session_1:product:")


def test_inherit_subagent_limits_keeps_parent_time_for_synthesis():
    turns, runtime_minutes, cost_usd = inherit_subagent_limits(
        default_max_turns=30,
        parent_max_turns=24,
        parent_remaining_runtime_minutes=30,
        parent_remaining_cost_usd=9,
    )

    assert turns == 12
    assert runtime_minutes == 4
    assert cost_usd == 3
