import json
import io
import os
import pathlib
import tempfile

import pytest
from datetime import datetime, timezone

from chack_agent.backends.claude_code_backend import (
    ClaudeCodeExecutor,
    _normalized_claude_usage,
    _seconds_until_claude_quota_reset,
)


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
    assert prompt not in command
    assert "Use schema name: attack_surface_entrypoints" in executor._cacheable_system_prompt
    assert "Your response must strictly match the JSON schema." in executor._cacheable_system_prompt


def test_claude_usage_preserves_cache_reads_and_writes_for_pricing() -> None:
    usage = _normalized_claude_usage(
        {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 200,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 200,
            },
        }
    )

    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 25
    assert usage["input_tokens_details"] == {
        "cached_tokens": 700,
        "cache_write_tokens": 200,
    }


def test_claude_command_removes_unsupported_json_schema_dialect_declarations() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "result": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "string",
            }
        },
        "required": ["result"],
    }
    executor = _build_executor(json.dumps(schema))

    command = executor._build_command(executor._compose_prompt("classify repository"))
    cli_schema = json.loads(command[command.index("--json-schema") + 1])

    assert "$schema" not in cli_schema
    assert "$schema" not in cli_schema["properties"]["result"]
    assert cli_schema["properties"]["result"]["type"] == "string"
    assert cli_schema["required"] == ["result"]


def test_claude_sends_large_prompt_via_stdin(monkeypatch, tmp_path) -> None:
    prompt = "large prompt\n" + ("x" * 500_000)
    captured: dict[str, object] = {}

    class _CaptureStdin(io.StringIO):
        def close(self) -> None:
            captured["stdin"] = self.getvalue()

    class _FakeProcess:
        pid = 999999

        def __init__(self, command):
            captured["command"] = command
            self.stdin = _CaptureStdin()
            self.stdout = io.StringIO(
                json.dumps({"type": "result", "subtype": "success", "result": "ok"}) + "\n"
            )

        def poll(self):
            return 0 if self.stdout.tell() >= len(self.stdout.getvalue()) else None

        def wait(self):
            return 0

    executor = _build_executor("")
    executor._claude_home = str(tmp_path)
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend.subprocess.Popen",
        lambda command, **kwargs: _FakeProcess(command),
    )

    output, _steps, _raw = executor._run_claude_once(prompt)

    assert output == "ok"
    assert prompt not in captured["command"]
    assert captured["stdin"] == prompt


def test_claude_command_omits_json_schema_when_unconfigured() -> None:
    executor = _build_executor("")

    command = executor._build_command(executor._compose_prompt("map inputs"))

    assert "--json-schema" not in command


def _builtin_tools(command: list[str]) -> set[str]:
    return set(command[command.index("--tools") + 1].split(","))


def test_claude_offers_only_the_builtin_tools_a_run_uses() -> None:
    executor = _build_executor("")
    executor._tools_config_json = '{"exec_enabled": true}'
    executor._allowed_tools_json = '["exec", "read_context"]'

    tools = _builtin_tools(executor._build_command("use tools"))

    assert {"Read", "Edit", "Write", "ToolSearch", "WebFetch", "WebSearch"} <= tools
    assert not tools & {"Agent", "Task", "Workflow", "Monitor", "CronCreate", "EnterWorktree", "SendMessage"}


@pytest.mark.parametrize(
    ("tools_config", "allowed_tools"),
    [
        ('{"exec_enabled": true}', '["exec", "read_context"]'),
        ('{"exec_enabled": true}', '["read_context"]'),
        ('{"exec_enabled": false}', '["read_context"]'),
    ],
)
def test_claude_never_offers_its_own_shell(tools_config, allowed_tools) -> None:
    executor = _build_executor("")
    executor._tools_config_json = tools_config
    executor._allowed_tools_json = allowed_tools

    assert "Bash" not in _builtin_tools(executor._build_command("use tools"))


@pytest.mark.parametrize("tools_config", ['{"exec_enabled": true}', '{"exec_enabled": false}'])
def test_claude_without_chack_exec_gets_no_host_file_tools(tools_config) -> None:
    executor = _build_executor("")
    executor._tools_config_json = tools_config
    executor._allowed_tools_json = '["read_context"]'

    tools = _builtin_tools(executor._build_command("use tools"))

    assert not tools & {"Bash", "Read", "Edit", "Write", "Glob", "Grep"}
    # Chack's own tools stay reachable.
    assert "ToolSearch" in tools


def test_claude_web_search_follows_the_config() -> None:
    executor = _build_executor("")
    executor._tools_config_json = '{"exec_enabled": true, "native_web_search_enabled": false}'
    executor._allowed_tools_json = '["read_context"]'

    assert not _builtin_tools(executor._build_command("use tools")) & {"WebSearch", "WebFetch"}


def test_claude_planning_tools_only_with_native_planning() -> None:
    executor = _build_executor("")
    executor._tools_config_json = '{"exec_enabled": true}'
    executor._allowed_tools_json = '["read_context"]'

    executor._native_task_planning_backend = ""
    assert "TaskCreate" not in _builtin_tools(executor._build_command("use tools"))
    executor._native_task_planning_backend = "claude"
    assert {"TaskCreate", "TaskUpdate", "TodoWrite"} <= _builtin_tools(executor._build_command("use tools"))


def test_claude_skips_all_tool_definitions_when_agent_has_no_tools() -> None:
    executor = _build_executor("")
    executor._allowed_tools_json = "[]"

    command = executor._build_command("analyze embedded context")

    assert command[command.index("--tools") + 1] == ""
    assert "--mcp-config" not in command


def test_claude_haiku_loads_mcp_tools_up_front(monkeypatch) -> None:
    executor = _build_executor("")
    executor._model_name = "claude-haiku-4-5"
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "true")

    env = executor._build_env()

    assert env["ENABLE_TOOL_SEARCH"] == "false"


def test_claude_non_haiku_preserves_explicit_tool_search_setting(monkeypatch) -> None:
    executor = _build_executor("")
    executor._model_name = "claude-sonnet-4-6"
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "auto")

    env = executor._build_env()

    assert env["ENABLE_TOOL_SEARCH"] == "auto"


def test_claude_large_custom_tool_payload_uses_file_backed_mcp_env(tmp_path) -> None:
    executor = _build_executor("")
    executor._claude_home = str(tmp_path)
    executor._serialized_tools_append_b64 = "x" * 24001

    env = executor._build_env()

    assert "CHACK_TOOLS_APPEND_B64" not in env
    payload_path = pathlib.Path(env["CHACK_TOOLS_APPEND_B64_PATH"])
    assert payload_path.parent == tmp_path
    assert payload_path.read_text(encoding="utf-8") == "x" * 24001


def test_claude_mcp_settings_block_until_core_tools_are_loaded(tmp_path) -> None:
    executor = _build_executor("")
    executor._claude_home = str(tmp_path)

    executor._write_claude_settings(str(tmp_path))

    settings = json.loads(pathlib.Path(tmp_path, "settings.json").read_text(encoding="utf-8"))
    assert settings["mcpServers"]["chack_tools"]["alwaysLoad"] is True
    assert settings["mcpServers"]["chack_tools"]["env"]["CHACK_MCP_STARTUP_STATUS_PATH"] == str(
        pathlib.Path(tmp_path, "mcp_startup_status.json")
    )


def test_claude_waits_for_required_mcp_tools_on_constrained_workers(monkeypatch) -> None:
    executor = _build_executor("")
    monkeypatch.setenv("MCP_CONNECTION_NONBLOCKING", "true")

    env = executor._build_env()

    assert env["MCP_CONNECTION_NONBLOCKING"] == "false"
    assert env["MCP_TIMEOUT"] == "60000"
    assert env["MCP_CONNECT_TIMEOUT_MS"] == "60000"


def test_claude_honors_explicit_chack_mcp_startup_timeout(monkeypatch) -> None:
    executor = _build_executor("")
    monkeypatch.setenv("CHACK_CLAUDE_MCP_STARTUP_TIMEOUT_MS", "90000")

    env = executor._build_env()

    assert env["MCP_TIMEOUT"] == "90000"
    assert env["MCP_CONNECT_TIMEOUT_MS"] == "90000"


def test_claude_mcp_receives_cli_tokens_and_api_fallbacks(monkeypatch, tmp_path) -> None:
    executor = _build_executor("")
    executor._claude_home = str(tmp_path)
    executor._claude_access_token = "claude-oauth-primary"
    executor._anthropic_api_key = "anthropic-api-fallback"
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-primary")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback")

    executor._write_claude_settings(str(tmp_path))

    settings = json.loads(pathlib.Path(tmp_path, "settings.json").read_text(encoding="utf-8"))
    mcp_env = settings["mcpServers"]["chack_tools"]["env"]
    assert mcp_env["CLAUDE_CODE_OAUTH_TOKEN"] == "claude-oauth-primary"
    assert "ANTHROPIC_API_KEY" not in mcp_env
    assert mcp_env["CODEX_ACCESS_TOKEN"] == "codex-primary"
    assert mcp_env["OPENAI_API_KEY"] == "openai-fallback"


def test_claude_mcp_pythonpath_preserves_application_import_root(monkeypatch, tmp_path) -> None:
    application_root = tmp_path / "application-root"
    application_root.mkdir()
    monkeypatch.syspath_prepend(str(application_root))
    executor = _build_executor("")

    env = executor._build_env()

    assert str(application_root) in env["PYTHONPATH"].split(os.pathsep)


def test_claude_nonzero_exit_with_result_event_is_still_failure(monkeypatch, tmp_path) -> None:
    events = "\n".join(
        [
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "session-id",
                    "mcp_servers": [{"name": "chack_tools", "status": "connected"}],
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "You've hit your session limit",
                }
            ),
        ]
    ) + "\n"

    class _FakeProcess:
        pid = 999999

        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(events)

        def poll(self):
            return 1 if self.stdout.tell() >= len(events) else None

        def wait(self):
            return 1

    executor = _build_executor("")
    executor._claude_home = str(tmp_path)
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    output, _steps, _raw = executor._run_claude_once("prompt")

    assert output.startswith("ERROR: Claude exec failed (exit=1).")
    assert "session limit" in output


def test_claude_success_result_marked_error_is_failure(monkeypatch, tmp_path) -> None:
    events = "\n".join(
        [
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "session-id",
                    "mcp_servers": [],
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": True,
                    "result": "You've hit your limit · resets 1:40pm",
                }
            ),
        ]
    ) + "\n"

    class _FakeProcess:
        pid = 999999

        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(events)

        def poll(self):
            return 0 if self.stdout.tell() >= len(events) else None

        def wait(self):
            return 0

    executor = _build_executor("")
    executor._claude_home = str(tmp_path)
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    output, _steps, _raw = executor._run_claude_once("prompt")

    assert output.startswith("ERROR: Claude returned an error in final result event.")
    assert "hit your limit" in output


def test_claude_rate_limit_falls_back_to_anthropic_api_key(monkeypatch) -> None:
    executor = _build_executor("")
    executor._claude_access_token = "oauth-primary"
    executor._anthropic_api_key = "api-key-fallback"
    responses = iter(
        [
            ("ERROR: Claude returned an error: You've hit your limit", [], None),
            ("{\"ok\":true}", [], None),
        ]
    )
    monkeypatch.setattr(executor, "_run_claude_once", lambda _prompt: next(responses))

    output, _steps, _raw = executor._run_claude("prompt")

    assert output == '{"ok":true}'
    assert executor._claude_access_token == ""


def test_claude_command_resumes_captured_session_for_followup_prompt() -> None:
    executor = _build_executor("")
    executor._claude_session_id = "11111111-2222-3333-4444-555555555555"
    prompt = executor._compose_prompt("try harder")

    command = executor._build_command(prompt)

    assert "--allow-dangerously-skip-permissions" in command
    assert "--dangerously-skip-permissions" in command
    assert "--resume" in command
    assert command[command.index("--resume") + 1] == "11111111-2222-3333-4444-555555555555"
    assert prompt not in command


def test_claude_cache_breakpoint_uses_system_prompt_without_tools() -> None:
    from chack_agent.backends.prompt_cache import PROMPT_CACHE_BREAKPOINT

    executor = _build_executor("")
    executor._base_system_prompt = "Threat-model only the supplied functionality."
    prompt = executor._compose_prompt(
        f"stable repository context\n{PROMPT_CACHE_BREAKPOINT}\nchanging checks"
    )
    command = executor._build_command(prompt)

    assert prompt == "\nchanging checks"
    assert PROMPT_CACHE_BREAKPOINT not in prompt
    assert "--system-prompt" in command
    cached_prefix = command[command.index("--system-prompt") + 1]
    assert "Threat-model only the supplied functionality." in cached_prefix
    assert "stable repository context" in cached_prefix
    assert "changing checks" not in cached_prefix


def test_claude_unmarked_prompt_still_caches_shared_system_layer() -> None:
    executor = _build_executor("")
    executor._base_system_prompt = "Stable Naxus instructions."

    prompt = executor._compose_prompt("changing task")
    command = executor._build_command(prompt)

    assert prompt == "changing task"
    assert executor._cacheable_system_prompt.startswith(
        "Stable Naxus instructions."
    )
    assert executor._prompt_cache_prefix_key.startswith("chack-")
    assert "--system-prompt" in command


def test_claude_followup_prompt_only_suppresses_system_once() -> None:
    executor = _build_executor("")
    executor._base_system_prompt = "SYSTEM SHOULD NOT REPEAT"
    executor.suppress_system_prompt_for_next_invocation()

    prompt = executor._compose_prompt("original request\n\ntry harder")

    assert prompt == "original request\n\ntry harder"
    assert "SYSTEM SHOULD NOT REPEAT" not in prompt
    assert executor._compose_prompt("normal") == "normal"
    assert "SYSTEM SHOULD NOT REPEAT" in executor._cacheable_system_prompt


def test_claude_prompt_maps_custom_tools_to_exact_mcp_names() -> None:
    executor = _build_executor("")
    executor._allowed_tools_json = json.dumps(
        ["read_context_lines", "search_context", "mark_check_checked"]
    )

    prompt = executor._compose_prompt("audit checks")

    assert "`read_context_lines` -> `mcp__chack_tools__read_context_lines`" in executor._cacheable_system_prompt
    assert "`search_context` -> `mcp__chack_tools__search_context`" in executor._cacheable_system_prompt
    assert "`mark_check_checked` -> `mcp__chack_tools__mark_check_checked`" in executor._cacheable_system_prompt
    assert "do not call their unprefixed aliases" in executor._cacheable_system_prompt


def test_claude_task_manager_policy_uses_exact_mcp_name() -> None:
    executor = _build_executor("")
    executor._require_task_steps_manager_init_first = True
    executor._allowed_tools_json = json.dumps(["task_steps_manager"])

    prompt = executor._compose_prompt("audit checks")

    assert "call `mcp__chack_tools__task_steps_manager`" in executor._cacheable_system_prompt


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

    assert "Never write directly to AISEC_LOCAL_VULN_STORE_PATH" in executor._cacheable_system_prompt
    assert "/tmp/aisec_local_vulnerabilities_*" in executor._cacheable_system_prompt


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


def test_claude_oauth_session_limit_falls_back_to_anthropic_api_key(monkeypatch) -> None:
    executor = _build_executor("")
    executor._claude_access_token = "oauth-primary"
    executor._anthropic_api_key = "api-fallback"
    results = iter(
        [
            ("ERROR: You've hit your session limit; resets later", [], None),
            ("fallback succeeded", [], None),
        ]
    )
    monkeypatch.setattr(executor, "_run_claude_once", lambda _prompt: next(results))

    output, _steps, _raw = executor._run_claude("prompt")

    assert output == "fallback succeeded"
    assert executor._claude_access_token == ""


def test_claude_oauth_waits_for_near_term_utc_reset_without_api_fallback(monkeypatch) -> None:
    executor = _build_executor("")
    executor._claude_access_token = "oauth-primary"
    executor._anthropic_api_key = ""
    results = iter(
        [
            ("ERROR: You've hit your session limit · resets 4:40pm (UTC)", [], None),
            ("claude token succeeded after reset", [], None),
        ]
    )
    sleeps = []
    monkeypatch.setenv("CHACK_CLAUDE_QUOTA_RESET_MAX_WAIT_SECONDS", "900")
    monkeypatch.setattr(executor, "_run_claude_once", lambda _prompt: next(results))
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend._seconds_until_claude_quota_reset",
        lambda _output: 241,
    )
    monkeypatch.setattr("chack_agent.backends.claude_code_backend.time.sleep", sleeps.append)

    output, _steps, _raw = executor._run_claude("prompt")

    assert output == "claude token succeeded after reset"
    assert executor._claude_access_token == "oauth-primary"
    assert sleeps == [241]


def test_claude_quota_reset_parser_uses_advertised_utc_time() -> None:
    seconds = _seconds_until_claude_quota_reset(
        "ERROR: Claude returned an error. You've hit your session limit · resets 4:40pm (UTC)",
        now=datetime(2026, 7, 17, 16, 36, tzinfo=timezone.utc),
    )

    assert seconds == 241


def test_claude_oauth_does_not_wait_past_configured_bound(monkeypatch) -> None:
    executor = _build_executor("")
    executor._claude_access_token = "oauth-primary"
    executor._anthropic_api_key = ""
    calls = []
    monkeypatch.setenv("CHACK_CLAUDE_QUOTA_RESET_MAX_WAIT_SECONDS", "60")
    monkeypatch.setattr(
        executor,
        "_run_claude_once",
        lambda _prompt: calls.append(True)
        or ("ERROR: You've hit your session limit · resets 4:40pm (UTC)", [], None),
    )
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend._seconds_until_claude_quota_reset",
        lambda _output: 241,
    )

    output, _steps, _raw = executor._run_claude("prompt")

    assert output.startswith("ERROR:")
    assert len(calls) == 1


def _run_claude_stream(monkeypatch, tmp_path, events: list[dict], output_schema_json: str = "") -> str:
    stream = "\n".join(json.dumps(event) for event in events) + "\n"

    class _FakeProcess:
        pid = 999999

        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(stream)

        def poll(self):
            return 0 if self.stdout.tell() >= len(stream) else None

        def wait(self):
            return 0

    executor = _build_executor(output_schema_json)
    executor._claude_home = str(tmp_path)
    monkeypatch.setattr(
        "chack_agent.backends.claude_code_backend.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )
    output, _steps, _raw = executor._run_claude_once("prompt")
    return output


_INIT_EVENT = {"type": "system", "subtype": "init", "session_id": "session-id", "mcp_servers": []}


def test_claude_final_answer_is_not_repeated_from_the_result_event(monkeypatch, tmp_path) -> None:
    output = _run_claude_stream(
        monkeypatch,
        tmp_path,
        [
            _INIT_EVENT,
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Running it now."}]}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "echo 42"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "42"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "42"}]}},
            {"type": "result", "subtype": "success", "result": "42"},
        ],
    )

    assert output == "42"


def test_claude_structured_output_is_the_whole_answer_when_the_result_text_also_carries_it(
    monkeypatch, tmp_path
) -> None:
    answer = {"verdict": "confirmed", "items": [1, 2]}
    for result_text in (json.dumps(answer), "Here is the report.\n" + json.dumps(answer)):
        output = _run_claude_stream(
            monkeypatch,
            tmp_path,
            [
                _INIT_EVENT,
                {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "s1", "name": "StructuredOutput", "input": answer}]}},
                {"type": "result", "subtype": "success", "result": result_text, "structured_output": answer},
            ],
        )

        assert json.loads(output) == answer


def test_claude_streamed_text_is_the_answer_when_the_result_event_is_empty(monkeypatch, tmp_path) -> None:
    output = _run_claude_stream(
        monkeypatch,
        tmp_path,
        [
            _INIT_EVENT,
            {"type": "user", "message": {"content": "replayed prompt"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "final answer"}]}},
            {"type": "result", "subtype": "success", "result": ""},
        ],
    )

    assert output == "final answer"


# A tool call whose name carries the model's leaked tool-call tokens, as Kimi
# K2.5 on Bedrock returns them through the gateway.
_LEAKED_TOOL_USE = {
    "type": "assistant",
    "message": {
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": 'functions_mcp__chack_tools__exec_2 <|tool_call_argument_begin|> {"command"',
                "input": {},
            }
        ]
    },
}
_NO_SUCH_TOOL = {
    "type": "user",
    "message": {
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "No such tool available"}
        ]
    },
}


def test_claude_names_malformed_provider_tool_calls_when_the_run_has_no_answer(monkeypatch, tmp_path) -> None:
    output = _run_claude_stream(
        monkeypatch,
        tmp_path,
        [
            _INIT_EVENT,
            _LEAKED_TOOL_USE,
            _NO_SUCH_TOOL,
            {"type": "result", "subtype": "success", "result": "", "stop_reason": "end_turn", "num_turns": 2},
        ],
    )

    assert output.startswith("ERROR: Claude provider returned malformed tool calls: 1 tool call(s)")
    assert "tool_call_argument_begin" in output
    assert "Claude finished without a final answer." in output
    assert '"stop_reason": "end_turn"' in output


def test_claude_keeps_the_answer_of_a_run_that_recovered_from_malformed_tool_calls(monkeypatch, tmp_path) -> None:
    output = _run_claude_stream(
        monkeypatch,
        tmp_path,
        [
            _INIT_EVENT,
            _LEAKED_TOOL_USE,
            _NO_SUCH_TOOL,
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "final answer"}]}},
            {"type": "result", "subtype": "success", "result": "final answer"},
        ],
    )

    assert output == "final answer"


def test_claude_stream_cut_without_result_event_is_an_error_not_the_raw_stream(monkeypatch, tmp_path) -> None:
    output = _run_claude_stream(monkeypatch, tmp_path, [_INIT_EVENT, _LEAKED_TOOL_USE])

    assert output.startswith("ERROR: Claude provider returned malformed tool calls")
    assert "The CLI sent no result event." in output


def test_claude_schema_run_ending_in_narration_is_an_error(monkeypatch, tmp_path) -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
            "additionalProperties": False,
        }
    )
    output = _run_claude_stream(
        monkeypatch,
        tmp_path,
        [
            _INIT_EVENT,
            {"type": "result", "subtype": "success", "result": "Let me check the handler first."},
        ],
        output_schema_json=schema,
    )

    assert output.startswith("ERROR: Claude finished without the structured output its schema requires.")
    assert "Let me check the handler first." in output


def test_claude_sends_cache_markers_unless_the_model_turns_them_off() -> None:
    executor = _build_executor("")
    assert "DISABLE_PROMPT_CACHING" not in executor._build_env()

    executor._prompt_cache_enabled = False
    assert executor._build_env()["DISABLE_PROMPT_CACHING"] == "1"


def test_claude_retries_once_without_cache_markers_when_the_provider_rejects_them(monkeypatch) -> None:
    monkeypatch.setattr("chack_agent.backends.claude_code_backend._PROMPT_CACHE_REJECTED", set())
    executor = _build_executor("")
    executor._claude_session_id = ""
    rejection = (
        "ERROR: Claude returned an error in final result event.\n"
        'API Error: 400 {"error":{"message":"litellm.BadRequestError: Model `gemini-4-argon` '
        'does not support explicit context caching."}}'
    )
    envs: list[dict[str, str]] = []

    def _run_once(_prompt):
        envs.append(executor._build_env())
        # The rejected attempt still opened a session that must not be resumed.
        executor._claude_session_id = "rejected-session"
        return (rejection, [], None) if len(envs) == 1 else ("answer", [], None)

    monkeypatch.setattr(executor, "_run_claude_once", _run_once)

    output, _, _ = executor._run_claude("prompt")

    assert output == "answer"
    assert "DISABLE_PROMPT_CACHING" not in envs[0]
    assert envs[1]["DISABLE_PROMPT_CACHING"] == "1"

    # Later runs of the same model skip the request the provider rejects.
    later = _build_executor("")
    assert later._build_env()["DISABLE_PROMPT_CACHING"] == "1"


def test_claude_does_not_retry_errors_unrelated_to_cache_markers(monkeypatch) -> None:
    monkeypatch.setattr("chack_agent.backends.claude_code_backend._PROMPT_CACHE_REJECTED", set())
    executor = _build_executor("")
    calls: list[str] = []

    def _run_once(prompt):
        calls.append(prompt)
        return ("ERROR: Claude exec failed (exit=1).\nboom", [], None)

    monkeypatch.setattr(executor, "_run_claude_once", _run_once)

    output, _, _ = executor._run_claude("prompt")

    assert output.startswith("ERROR: Claude exec failed")
    assert len(calls) == 1


def test_claude_save_retry_failure_keeps_the_answer_already_given(monkeypatch) -> None:
    executor = _build_executor("")
    outputs = iter(
        [
            ("found an injection", [], None),
            ("ERROR: Claude finished without a final answer.", [], None),
        ]
    )
    monkeypatch.setattr(executor, "_run_claude", lambda _prompt: (*next(outputs)[:2], type("R", (), {"raw_responses": []})()))
    monkeypatch.setattr(executor, "_has_save_vulnerability_tool", lambda: True)
    monkeypatch.setattr(executor, "_steps_contain_save", lambda _steps: False)
    monkeypatch.setattr(executor, "_steps_contain_analysis", lambda _steps: True)

    assert executor.invoke({"input": "review"})["output"] == "found an injection"


_ENHANCEMENT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "enhancements": {
                "type": "object",
                "properties": {"v001": {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}},
                "required": ["v001"],
            }
        },
        "required": ["enhancements"],
    }
)


def _gave_up_on_structured_output(*attempt_inputs: dict) -> list[dict]:
    attempts = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": f"s{index}", "name": "StructuredOutput", "input": attempt}]}}
        for index, attempt in enumerate(attempt_inputs)
    ]
    return [
        _INIT_EVENT,
        *attempts,
        {
            "type": "result",
            "subtype": "error_max_structured_output_retries",
            "is_error": True,
            "errors": ["Output does not match required schema: /enhancements: must be object"],
        },
    ]


def test_claude_uses_a_given_up_structured_answer_that_only_sent_an_object_as_a_string(monkeypatch, tmp_path) -> None:
    enhancements = {"v001": {"verdict": "true_positive"}}
    output = _run_claude_stream(
        monkeypatch,
        tmp_path,
        _gave_up_on_structured_output(
            {"enhancements": "not json"},
            {"enhancements": json.dumps(enhancements)},
        ),
        output_schema_json=_ENHANCEMENT_SCHEMA,
    )

    assert json.loads(output) == {"enhancements": enhancements}


@pytest.mark.parametrize(
    "last_attempt",
    [
        {"enhancements": json.dumps({"v001": {}})},
        {"enhancements": "not json"},
        {"report": "free text"},
    ],
)
def test_claude_still_fails_when_its_last_structured_attempt_does_not_match_the_schema(
    monkeypatch, tmp_path, last_attempt
) -> None:
    output = _run_claude_stream(
        monkeypatch,
        tmp_path,
        _gave_up_on_structured_output({"enhancements": {"v001": {"verdict": "ok"}}}, last_attempt),
        output_schema_json=_ENHANCEMENT_SCHEMA,
    )

    assert output.startswith("ERROR: Claude returned an error in final result event.")
    assert "error_max_structured_output_retries" in output


def test_claude_other_result_errors_are_not_replaced_by_a_structured_attempt(monkeypatch, tmp_path) -> None:
    events = _gave_up_on_structured_output({"enhancements": {"v001": {"verdict": "ok"}}})
    events[-1] = {"type": "result", "subtype": "error_max_turns", "is_error": True}
    output = _run_claude_stream(monkeypatch, tmp_path, events, output_schema_json=_ENHANCEMENT_SCHEMA)

    assert output.startswith("ERROR: Claude returned an error in final result event.")
