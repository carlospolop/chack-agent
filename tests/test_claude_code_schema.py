import json
import pathlib
import tempfile

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


def test_claude_vulnerability_fallback_collects_only_new_files() -> None:
    executor = _build_executor("")
    payload = {
        "name": "Fallback finding",
        "description": "Validated fallback description",
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "steps": [
            {
                "file_path": "app.py",
                "code": "eval(data)",
                "description": "Attacker data reaches eval.",
            }
        ],
    }

    with tempfile.TemporaryDirectory(prefix="claude_fallback_") as root:
        executor._claude_home = str(pathlib.Path(root, "session-a"))
        vulns_dir = pathlib.Path(executor._bash_vuln_fallback_dir(root))
        vulns_dir.mkdir(parents=True)
        pathlib.Path(vulns_dir, "old.json").write_text(json.dumps(payload), encoding="utf-8")
        snapshot = executor._snapshot_bash_saved_vulns(str(vulns_dir))

        assert executor._collect_bash_saved_vulns(str(vulns_dir), snapshot) == []

        pathlib.Path(vulns_dir, "new.json").write_text(json.dumps(payload), encoding="utf-8")
        collected = executor._collect_bash_saved_vulns(str(vulns_dir), snapshot)

        assert len(collected) == 1
        action = collected[0][0]
        assert action.tool_input["tool_id"] == "bash_save_new.json"


def test_claude_save_policy_forbids_internal_store_writes() -> None:
    executor = _build_executor("")
    executor._allowed_tools_json = json.dumps(["save_discovered_vulnerability"])

    prompt = executor._compose_prompt("audit checks")

    assert "Never write directly to AISEC_LOCAL_VULN_STORE_PATH" in prompt
    assert "/tmp/aisec_local_vulnerabilities_*" in prompt


def test_claude_oauth_auth_failure_retries_once_with_api_key(monkeypatch) -> None:
    executor = _build_executor("")
    executor._claude_access_token = "oauth-primary"
    executor._anthropic_api_key = "api-fallback"
    results = iter(
        [
            ("ERROR: authentication_error: invalid OAuth token", [], None),
            ("fallback succeeded", [], None),
        ]
    )
    monkeypatch.setattr(executor, "_run_claude_once", lambda _prompt: next(results))

    output, _, _ = executor._run_claude("test")

    assert output == "fallback succeeded"
    assert executor._claude_access_token == ""


def test_claude_does_not_fallback_on_successful_output_mentioning_401(monkeypatch) -> None:
    executor = _build_executor("")
    executor._claude_access_token = "oauth-primary"
    executor._anthropic_api_key = "api-fallback"
    calls = []

    def _run_once(_prompt):
        calls.append(True)
        return ("The audited code handles HTTP 401 responses.", [], None)

    monkeypatch.setattr(executor, "_run_claude_once", _run_once)

    output, _, _ = executor._run_claude("test")

    assert output == "The audited code handles HTTP 401 responses."
    assert len(calls) == 1
    assert executor._claude_access_token == "oauth-primary"
