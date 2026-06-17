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
        _tester_model="",
        _subchack_model="",
        _social_network_max_turns=0,
        _scientific_max_turns=0,
        _websearcher_max_turns=0,
        _tester_max_turns=0,
        _subchack_max_turns=0,
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
    assert "Use schema name: attack_surface_entrypoints" in prompt
    assert "Your response must strictly match the JSON schema." in prompt


def test_claude_command_omits_json_schema_when_unconfigured() -> None:
    executor = _build_executor("")

    command = executor._build_command(executor._compose_prompt("map inputs"))

    assert "--json-schema" not in command
