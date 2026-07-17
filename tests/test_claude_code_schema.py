import json

from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor


def _build_executor(output_schema_json: str = "") -> ClaudeCodeExecutor:
    return ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=0,
        _memory_reset_to=0,
        _base_system_prompt="",
        _model_name="claude-opus-4-8",
        _max_turns=10,
        _claude_cli_path="claude",
        _tools_config_json='{"exec_enabled": true}',
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
        _output_schema_json=output_schema_json,
        _output_schema_name="attack_surface_entrypoints",
        _output_schema_strict=True,
    )


def test_claude_command_forces_configured_output_schema() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entrypoints"],
        "properties": {"entrypoints": {"type": "array", "items": {"type": "object"}}},
    }
    schema_json = json.dumps(schema, ensure_ascii=False)
    executor = _build_executor(schema_json)

    prompt = executor._compose_prompt("map inputs")
    command = executor._build_command(prompt)

    assert "--json-schema" in command
    assert command[command.index("--json-schema") + 1] == schema_json
    assert "--allow-dangerously-skip-permissions" in command
    assert "--dangerously-skip-permissions" in command
    assert command.index("--json-schema") < len(command) - 1
    assert command[-1] == prompt
    assert "Use schema name: attack_surface_entrypoints" in prompt
    assert "Your response must strictly match the JSON schema." in prompt


def test_claude_command_omits_json_schema_when_unconfigured() -> None:
    executor = _build_executor("")

    command = executor._build_command(executor._compose_prompt("map inputs"))

    assert "--json-schema" not in command


def test_claude_command_resumes_captured_session_for_followup_prompt() -> None:
    executor = _build_executor("")
    executor._claude_session_id = "11111111-2222-3333-4444-555555555555"
    prompt = executor._compose_prompt("try harder")

    command = executor._build_command(prompt)

    assert "--allow-dangerously-skip-permissions" in command
    assert "--dangerously-skip-permissions" in command
    assert "--resume" in command
    assert command[command.index("--resume") + 1] == "11111111-2222-3333-4444-555555555555"
    assert command[-1] == prompt


def test_claude_followup_prompt_only_suppresses_system_once() -> None:
    executor = _build_executor("")
    executor._base_system_prompt = "SYSTEM SHOULD NOT REPEAT"
    executor.suppress_system_prompt_for_next_invocation()

    prompt = executor._compose_prompt("original request\n\ntry harder")

    assert prompt == "original request\n\ntry harder"
    assert "SYSTEM SHOULD NOT REPEAT" not in prompt
    assert "SYSTEM SHOULD NOT REPEAT" in executor._compose_prompt("normal")


def test_claude_prompt_maps_custom_tools_to_exact_mcp_names() -> None:
    executor = _build_executor("")
    executor._allowed_tools_json = json.dumps(
        ["read_context_lines", "search_context", "mark_check_checked"]
    )

    prompt = executor._compose_prompt("audit checks")

    assert "`read_context_lines` -> `mcp__chack_tools__read_context_lines`" in prompt
    assert "`search_context` -> `mcp__chack_tools__search_context`" in prompt
    assert "`mark_check_checked` -> `mcp__chack_tools__mark_check_checked`" in prompt
    assert "do not call their unprefixed aliases" in prompt


def test_claude_task_manager_policy_uses_exact_mcp_name() -> None:
    executor = _build_executor("")
    executor._require_task_steps_manager_init_first = True
    executor._allowed_tools_json = json.dumps(["task_steps_manager"])

    prompt = executor._compose_prompt("audit checks")

    assert "call `mcp__chack_tools__task_steps_manager`" in prompt
