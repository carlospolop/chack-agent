"""opencode backend driven against a fake ``opencode`` executable.

The fake reads the prompt from stdin, records what it was given (arguments,
prompt, the config file and selected environment values) and replays a
scripted JSON event stream, so every test runs the real subprocess path.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

from chack_agent import llm_gateway
from chack_agent.agent import Chack
from chack_agent.backends import build_executor
from chack_agent.backends.opencode_cli_backend import OpencodeCliExecutor, OpencodeProvider
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
    resolve_backend_type,
)
from chack_agent.thinking_effort import (
    OPENCODE_SDK_ANTHROPIC,
    OPENCODE_SDK_OPENAI,
    OPENCODE_SDK_OPENROUTER,
    opencode_thinking_effort,
)
from chack_agent.backends.prompt_cache import PROMPT_CACHE_BREAKPOINT

GATEWAY_URL = "https://gateway.example.test/v1"
GATEWAY_KEY = "sk-gateway-test"

LITELLM_MODEL_INFO = {
    "data": [
        {
            "model_name": "corp/opus-5.5",
            "litellm_params": {"model": "vertex_ai/claude-opus-5-5"},
            "model_info": {"max_input_tokens": 1_000_000},
        },
        {
            "model_name": "corp/gpt-6-sol",
            "litellm_params": {"model": "bedrock_mantle/openai.gpt-6-sol"},
            "model_info": {
                "max_input_tokens": 922_000,
                "input_cost_per_token": 2e-06,
                "cache_read_input_token_cost": 2e-07,
                "output_cost_per_token": 1e-05,
            },
        },
        {
            "model_name": "corp/auto",
            "litellm_params": {"model": "auto_router/complexity_router"},
            "model_info": {},
        },
    ]
}

FAKE_OPENCODE = textwrap.dedent(
    """
    import json, os, sys, time
    record = os.environ["FAKE_OC_RECORD"]
    with open(os.environ["FAKE_OC_SCENARIO"]) as handle:
        scenario = json.load(handle)
    config_path = os.environ["OPENCODE_CONFIG"]
    with open(config_path) as handle:
        config_text = handle.read()
    config = json.loads(config_path and config_text)
    calls = []
    if os.path.exists(record):
        with open(record) as handle:
            calls = json.load(handle)
    calls.append({
        "argv": sys.argv[1:],
        "prompt": sys.stdin.read(),
        "config": config,
        "config_text": config_text,
        "env": {key: os.environ.get(key) for key in scenario.get("env_keys", [])},
    })
    with open(record, "w") as handle:
        json.dump(calls, handle)
    server = config.get("mcp", {}).get("chack_tools")
    if server and scenario.get("mcp_receipt", True):
        with open(server["environment"]["CHACK_MCP_STARTUP_STATUS_PATH"], "w") as handle:
            json.dump({"tool_names": ["task_steps_manager"], "error": scenario.get("mcp_error", "")}, handle)
    run = scenario["runs"][min(len(calls), len(scenario["runs"])) - 1]
    time.sleep(run.get("sleep", 0))
    for event in run.get("events", []):
        print(json.dumps(event), flush=True)
    for line in run.get("lines", []):
        print(line, flush=True)
    sys.exit(run.get("exit", 0))
    """
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("LLM_GATEWAY_", "OPENROUTER_", "CLAUDE_", "ANTHROPIC_", "CODEX_", "OPENAI_", "OPENCODE_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHACK_OPENCODE_HOME_BASE", str(tmp_path / "opencode-home"))
    llm_gateway._CATALOG_CACHE.clear()
    monkeypatch.setattr(
        llm_gateway, "_http_get", lambda url, headers=None, timeout=None: _FakeResponse(LITELLM_MODEL_INFO)
    )
    yield
    llm_gateway._CATALOG_CACHE.clear()


class FakeOpencode:
    def __init__(self, tmp_path, monkeypatch):
        self.dir = tmp_path
        self.path = tmp_path / "opencode"
        self.path.write_text(f"#!{sys.executable}\n{FAKE_OPENCODE}", encoding="utf-8")
        self.path.chmod(0o755)
        self.record = tmp_path / "calls.json"
        self.scenario = tmp_path / "scenario.json"
        monkeypatch.setenv("OPENCODE_CLI_PATH", str(self.path))
        monkeypatch.setenv("FAKE_OC_RECORD", str(self.record))
        monkeypatch.setenv("FAKE_OC_SCENARIO", str(self.scenario))
        self.script([{"events": [_text("ok"), _finish(10, 1)]}])

    def script(self, runs, **extra):
        self.scenario.write_text(json.dumps({"runs": runs, **extra}), encoding="utf-8")

    @property
    def calls(self):
        return json.loads(self.record.read_text(encoding="utf-8")) if self.record.exists() else []


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    return FakeOpencode(tmp_path, monkeypatch)


def _event(kind, part=None, session="ses_1"):
    return {"type": kind, "sessionID": session, "part": part or {}}


def _text(text, session="ses_1"):
    return _event("text", {"text": text}, session)


def _finish(input_tokens, output_tokens, *, reasoning=0, cache_read=0, cache_write=0, session="ses_1"):
    tokens = {
        "input": input_tokens,
        "output": output_tokens,
        "reasoning": reasoning,
        "cache": {"read": cache_read, "write": cache_write},
    }
    return _event("step_finish", {"tokens": tokens, "reason": "stop"}, session)


def _config(primary, *, api_key_type="", tools=None, agent=None, **credentials):
    return ChackConfig(
        model=ModelConfig(primary=primary, provider="opencode", api_key_type=api_key_type),
        agent=agent or AgentConfig(main_action="test", sub_action="opencode", thinking_effort="high"),
        session=SessionConfig(max_turns=7, long_term_memory_enabled=False),
        tools=tools or ToolsConfig(task_steps_manager_enabled=False),
        credentials=CredentialsConfig(**credentials),
        logging=LoggingConfig(level="WARNING"),
        system_prompt="You are the test agent.",
        env={},
    )


def _gateway_config(primary, **kwargs):
    return _config(
        primary,
        api_key_type="gateway",
        llm_gateway_type="litellm",
        llm_gateway_base_url=GATEWAY_URL,
        llm_gateway_api_key=GATEWAY_KEY,
        **kwargs,
    )


def _build(config):
    return build_executor(
        config, system_prompt=config.system_prompt, max_turns=7, memory_max_messages=10, memory_reset_to_messages=5
    )


def _provider_of(call):
    return call["config"]["provider"]["chack"]


# --- provider mapping --------------------------------------------------------


def test_gateway_claude_model_uses_the_messages_api_with_the_key_kept_off_disk(fake_opencode):
    fake_opencode.script([{"events": [_text("done"), _finish(5, 1)]}], env_keys=["CHACK_OPENCODE_API_KEY"])
    config = _gateway_config("claude-opus-5-5", llm_gateway_headers={"X-Team": "red"})
    executor = _build(config)

    result = executor.invoke({"input": "hello"})

    assert isinstance(executor, OpencodeCliExecutor)
    assert result["output"] == "done"
    call = fake_opencode.calls[0]
    provider = _provider_of(call)
    assert provider["npm"] == OPENCODE_SDK_ANTHROPIC
    assert provider["options"]["baseURL"] == "https://gateway.example.test/v1"
    assert list(provider["models"]) == ["corp/opus-5.5"]
    assert call["argv"][call["argv"].index("--model") + 1] == "chack/corp/opus-5.5"
    assert call["env"]["CHACK_OPENCODE_API_KEY"] == GATEWAY_KEY
    assert GATEWAY_KEY not in call["config_text"]
    assert "red" not in call["config_text"]
    assert provider["models"]["corp/opus-5.5"]["limit"]["context"] == 1_000_000
    assert provider["models"]["corp/opus-5.5"]["options"]["thinking"] == {"type": "adaptive"}


def test_gateway_header_values_reach_opencode_through_the_environment(fake_opencode):
    config = _gateway_config("gpt-6-sol", llm_gateway_headers={"X-Team": "red"})
    executor = _build(config)
    executor.invoke({"input": "hello"})
    headers = _provider_of(fake_opencode.calls[0])["options"]["headers"]
    reference = headers["X-Team"]
    env_name = reference[len("{env:"):-1]
    fake_opencode.script([{"events": [_text("ok"), _finish(1, 1)]}], env_keys=[env_name])

    executor.invoke({"input": "again"})

    assert fake_opencode.calls[1]["env"][env_name] == "red"


@pytest.mark.parametrize(
    ("primary", "sdk", "model_id"),
    [
        ("gpt-6-sol", OPENCODE_SDK_OPENAI, "corp/gpt-6-sol"),
        ("corp/auto", "@ai-sdk/openai-compatible", "corp/auto"),
    ],
)
def test_gateway_models_use_their_native_wire_format(fake_opencode, primary, sdk, model_id):
    executor = _build(_gateway_config(primary))

    executor.invoke({"input": "hi"})

    provider = _provider_of(fake_opencode.calls[0])
    assert provider["npm"] == sdk
    assert provider["options"]["baseURL"] == GATEWAY_URL
    assert list(provider["models"]) == [model_id]


def test_gateway_key_alone_selects_the_gateway(fake_opencode):
    config = _gateway_config("gpt-6-sol")
    config.model.api_key_type = ""

    _build(config).invoke({"input": "hi"})

    assert _provider_of(fake_opencode.calls[0])["options"]["baseURL"] == GATEWAY_URL


def test_openrouter_key_uses_the_openrouter_provider_and_ids(fake_opencode):
    fake_opencode.script([{"events": [_text("ok"), _finish(1, 1)]}], env_keys=["CHACK_OPENCODE_API_KEY"])
    config = _config(
        "claude-sonnet-5-5",
        api_key_type="openrouter",
        openrouter_api_key="or-key",
        openrouter_app_name="chack-test",
    )

    _build(config).invoke({"input": "hi"})

    call = fake_opencode.calls[0]
    provider = _provider_of(call)
    assert provider["npm"] == OPENCODE_SDK_OPENROUTER
    assert provider["options"]["baseURL"].startswith("https://openrouter.ai/api")
    assert list(provider["models"]) == ["anthropic/claude-sonnet-5.5"]
    assert call["env"]["CHACK_OPENCODE_API_KEY"] == "or-key"
    assert "or-key" not in call["config_text"]


def test_openai_key_uses_the_responses_api(fake_opencode, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    _build(_config("gpt-6-sol")).invoke({"input": "hi"})

    provider = _provider_of(fake_opencode.calls[0])
    assert provider["npm"] == OPENCODE_SDK_OPENAI
    assert provider["options"]["baseURL"] == "https://api.openai.com/v1"
    assert provider["models"]["gpt-6-sol"]["options"] == {"reasoningEffort": "high"}


def test_anthropic_key_uses_the_messages_api_with_its_own_auth(fake_opencode, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

    _build(_config("claude-sonnet-5-5", api_key_type="anthropic")).invoke({"input": "hi"})

    provider = _provider_of(fake_opencode.calls[0])
    assert provider["npm"] == OPENCODE_SDK_ANTHROPIC
    assert provider["options"]["baseURL"] == "https://api.anthropic.com/v1"
    assert "Authorization" not in provider["options"].get("headers", {})


@pytest.mark.parametrize("api_key_type", ["openai", "anthropic"])
def test_missing_vendor_key_fails_before_launch(api_key_type):
    with pytest.raises(ValueError, match="required for the opencode backend"):
        _build(_config("some-model", api_key_type=api_key_type))


@pytest.mark.parametrize("provider", ["opencode", "opencode-cli", "opencode_cli"])
def test_opencode_provider_names_select_the_opencode_backend(provider):
    config = _gateway_config("gpt-6-sol")
    config.model.provider = provider
    assert resolve_backend_type(config) == "opencode"


# --- thinking effort ---------------------------------------------------------


@pytest.mark.parametrize(
    ("effort", "sdk", "expected"),
    [
        ("none", OPENCODE_SDK_ANTHROPIC, {"thinking": {"type": "disabled"}}),
        ("minimal", OPENCODE_SDK_ANTHROPIC, {"thinking": {"type": "adaptive"}, "effort": "low"}),
        ("max", OPENCODE_SDK_ANTHROPIC, {"thinking": {"type": "adaptive"}, "effort": "max"}),
        ("max", OPENCODE_SDK_OPENROUTER, {"reasoning": {"effort": "xhigh"}}),
        ("medium", OPENCODE_SDK_OPENAI, {"reasoningEffort": "medium"}),
    ],
)
def test_thinking_effort_maps_to_each_sdk(effort, sdk, expected):
    assert opencode_thinking_effort(effort, sdk) == expected


def test_no_thinking_marks_the_model_as_non_reasoning(fake_opencode):
    agent = AgentConfig(main_action="test", sub_action="opencode", thinking_effort="none")

    _build(_gateway_config("claude-opus-5-5", agent=agent)).invoke({"input": "hi"})

    model = _provider_of(fake_opencode.calls[0])["models"]["corp/opus-5.5"]
    assert model["reasoning"] is False
    assert model["options"] == {"thinking": {"type": "disabled"}}


# --- run behaviour -----------------------------------------------------------


def test_prompt_goes_through_stdin_and_final_step_text_is_the_answer(fake_opencode):
    fake_opencode.script(
        [
            {
                "events": [
                    _event("step_start"),
                    _text("Let me plan first."),
                    _event(
                        "tool_use",
                        {
                            "tool": "chack_tools_task_steps_manager",
                            "callID": "c1",
                            "state": {"status": "completed", "input": {"action": "list"}, "output": "[]"},
                        },
                    ),
                    _finish(100, 20, reasoning=30, cache_read=400, cache_write=50),
                    _event("step_start"),
                    _text("final answer"),
                    _finish(10, 5, cache_read=600),
                ]
            }
        ]
    )
    executor = _build(_gateway_config("gpt-6-sol"))
    long_prompt = "x" * 200_000

    result = executor.invoke({"input": long_prompt})

    call = fake_opencode.calls[0]
    assert call["prompt"] == long_prompt
    assert long_prompt not in " ".join(call["argv"])
    assert result["output"] == "final answer"
    assert [action.tool for action, _ in result["intermediate_steps"]] == ["task_steps_manager"]
    usages = [item["usage"] for item in result["raw_result"].raw_responses]
    # Prompt tokens include cache reads and writes; completion includes reasoning.
    assert usages[0]["input_tokens"] == 550
    assert usages[0]["output_tokens"] == 50
    assert usages[0]["input_tokens_details"] == {"cached_tokens": 400, "cache_write_tokens": 50}
    assert usages[1]["input_tokens"] == 610


def test_system_prompt_and_stable_prefix_go_to_the_agent_and_the_task_to_stdin(fake_opencode):
    executor = _build(_gateway_config("gpt-6-sol"))

    executor.invoke({"input": f"shared catalog{PROMPT_CACHE_BREAKPOINT}task 1"})
    executor.invoke({"input": f"shared catalog{PROMPT_CACHE_BREAKPOINT}task 2"})

    first, second = fake_opencode.calls
    assert [first["prompt"], second["prompt"]] == ["task 1", "task 2"]
    agent = first["config"]["agent"]["chack"]
    assert agent["prompt"].startswith("You are the test agent.")
    assert agent["prompt"].endswith("shared catalog")
    assert agent["steps"] == 7
    # The agent prompt is identical across tasks so the provider cache is reused.
    assert second["config"]["agent"]["chack"]["prompt"] == agent["prompt"]


def test_later_invocations_resume_the_session(fake_opencode):
    fake_opencode.script(
        [
            {"events": [_text("one", session="ses_abc"), _finish(1, 1, session="ses_abc")]},
            {"events": [_text("two", session="ses_abc"), _finish(1, 1, session="ses_abc")]},
        ]
    )
    executor = _build(_gateway_config("gpt-6-sol"))

    executor.invoke({"input": "first"})
    executor.suppress_system_prompt_for_next_invocation()
    second = executor.invoke({"input": "second"})

    first_call, second_call = fake_opencode.calls
    assert "--session" not in first_call["argv"]
    assert second_call["argv"][second_call["argv"].index("--session") + 1] == "ses_abc"
    assert second["output"] == "two"
    assert second_call["prompt"] == "second"
    # The resumed session keeps the agent prompt it was started with.
    assert second_call["config"]["agent"]["chack"]["prompt"] == first_call["config"]["agent"]["chack"]["prompt"]


def test_provider_errors_are_reported_as_backend_errors(fake_opencode):
    fake_opencode.script(
        [
            {
                "events": [
                    {
                        "type": "error",
                        "sessionID": "ses_1",
                        "error": {"name": "APIError", "data": {"message": "invalid key", "statusCode": 401}},
                    }
                ],
                "exit": 1,
            }
        ]
    )

    result = _build(_gateway_config("gpt-6-sol")).invoke({"input": "hi"})

    assert result["output"] == "ERROR: opencode result error: APIError: invalid key (status 401)"


def test_a_lost_session_is_dropped_so_the_next_run_starts_fresh(fake_opencode):
    fake_opencode.script(
        [
            {"events": [_text("one"), _finish(1, 1)]},
            {"lines": ["Error: Session not found: ses_1"], "exit": 1},
            {"events": [_text("three", session="ses_2"), _finish(1, 1, session="ses_2")]},
        ]
    )
    executor = _build(_gateway_config("gpt-6-sol"))

    executor.invoke({"input": "a"})
    failed = executor.invoke({"input": "b"})
    executor.invoke({"input": "c"})

    assert failed["output"].startswith("ERROR: opencode exec failed (exit=1)")
    assert "--session" not in fake_opencode.calls[2]["argv"]


def test_tool_server_is_configured_and_its_startup_failure_is_an_error(fake_opencode):
    tools = ToolsConfig(task_steps_manager_enabled=True)
    executor = _build(_gateway_config("gpt-6-sol", tools=tools))

    ok = executor.invoke({"input": "hi"})
    fake_opencode.script([{"events": [_text("ok"), _finish(1, 1)]}], mcp_error="ImportError: boom")
    failed = executor.invoke({"input": "hi"})

    server = fake_opencode.calls[0]["config"]["mcp"]["chack_tools"]
    assert server["command"][1:] == ["-m", "chack_agent.backends.chack_tools_mcp_server"]
    assert ok["output"] == "ok"
    assert failed["output"].startswith("ERROR: Required opencode MCP server failed to start")


def test_the_native_shell_is_offered_only_when_exec_is_on_without_chack_exec(fake_opencode):
    _build(_gateway_config("gpt-6-sol")).invoke({"input": "hi"})
    with_exec = _build(
        _gateway_config("gpt-6-sol", tools=ToolsConfig(task_steps_manager_enabled=False, exec_enabled=True))
    )
    with_exec.invoke({"input": "hi"})

    without_exec, with_chack_exec = (call["config"]["permission"] for call in fake_opencode.calls)
    assert without_exec["bash"] == "deny"
    # chack's exec tool is already the run's shell.
    assert "exec" in with_exec._allowed_tool_names()
    assert with_chack_exec["bash"] == "deny"
    # Nobody can answer an interactive question in a Chack run.
    assert with_chack_exec["question"] == "deny"

    with_exec._allowed_tools_json = '["read_context"]'
    permission = with_exec.build_opencode_config()["permission"]
    assert permission.get("bash", permission["*"]) == "allow"


def test_structured_output_is_extracted_and_validated(fake_opencode):
    fake_opencode.script([{"events": [_text('Here it is: {"capital": "Canberra"}'), _finish(1, 1)]}])
    agent = AgentConfig(
        main_action="test",
        sub_action="opencode",
        output_schema_json={
            "type": "object",
            "properties": {"capital": {"type": "string"}},
            "required": ["capital"],
            "additionalProperties": False,
        },
        output_schema_name="answer",
    )

    result = _build(_gateway_config("gpt-6-sol", agent=agent)).invoke({"input": "capital of Australia?"})

    assert json.loads(result["output"]) == {"capital": "Canberra"}


def test_large_tool_payloads_are_passed_as_files(fake_opencode):
    big = json.dumps({"blob": "y" * 50_000})
    fake_opencode.script(
        [{"events": [_text("ok"), _finish(1, 1)]}],
        env_keys=["CHACK_TOOLS_CONFIG_JSON", "CHACK_TOOLS_CONFIG_JSON_PATH"],
    )
    executor = OpencodeCliExecutor(
        provider=OpencodeProvider(
            key_type="openai", sdk=OPENCODE_SDK_OPENAI, base_url="https://api.openai.com/v1", api_key="k", model_id="m"
        ),
        base_system_prompt="system",
        max_turns=3,
        opencode_cli_path=str(fake_opencode.path),
        tools_config_json=big,
        allowed_tools_json="[]",
    )

    executor.invoke({"input": "hi"})

    env = fake_opencode.calls[0]["env"]
    assert env["CHACK_TOOLS_CONFIG_JSON"] is None
    with open(env["CHACK_TOOLS_CONFIG_JSON_PATH"], encoding="utf-8") as handle:
        assert handle.read() == big


def test_a_run_past_its_timeout_is_stopped(fake_opencode, monkeypatch):
    monkeypatch.setenv("CHACK_OPENCODE_EXEC_TIMEOUT_SECONDS", "1")
    fake_opencode.script([{"sleep": 10, "events": [_text("late"), _finish(1, 1)]}])

    result = _build(_gateway_config("gpt-6-sol")).invoke({"input": "hi"})

    assert result["output"] == "ERROR: opencode execution timed out after 1s."


def test_missing_cli_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_CLI_PATH", str(tmp_path / "no-such-opencode"))

    result = _build(_gateway_config("gpt-6-sol")).invoke({"input": "hi"})

    assert result["output"].startswith("ERROR: opencode CLI executable was not found")


# --- through the agent loop --------------------------------------------------


def test_a_chack_run_reports_usage_and_gateway_priced_cost(fake_opencode):
    fake_opencode.script([{"events": [_text("391"), _finish(1000, 100, reasoning=100, cache_read=1000)]}])

    result = Chack(_gateway_config("gpt-6-sol")).run(
        "opencode-e2e", "What is 17 * 23?", enable_self_critique=False, require_task_steps_manager_init_first=False
    )

    assert result.output == "391"
    assert not result.error
    assert result.prompt_tokens == 2000
    assert result.cached_prompt_tokens == 1000
    assert result.completion_tokens == 200
    # 1000 uncached input at 2e-6, 1000 cached at 2e-7, 200 output at 1e-5.
    assert result.total_cost == pytest.approx(0.002 + 0.0002 + 0.002)


def test_a_failed_opencode_run_is_a_run_error(fake_opencode):
    fake_opencode.script([{"lines": ["boom"], "exit": 2}])

    result = Chack(_gateway_config("gpt-6-sol")).run(
        "opencode-e2e-fail", "hi", enable_self_critique=False, require_task_steps_manager_init_first=False
    )

    assert result.error


def test_opencode_and_its_tool_server_join_the_runs_trace(fake_opencode, monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    from chack_tools.telemetry.otel import end_tool_span, start_tool_span

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example.test")
    fake_opencode.script(
        [{"events": [_text("ok"), _finish(1, 1)]}],
        env_keys=["TRACEPARENT", "OTEL_EXPORTER_OTLP_ENDPOINT", "CHACK_DISABLE_STDOUT_EVENTS"],
    )
    executor = _build(_gateway_config("gpt-6-sol"))

    span = start_tool_span("outer", {})
    try:
        executor.invoke({"input": "hi"})
    finally:
        end_tool_span(span, result="")

    env = fake_opencode.calls[0]["env"]
    assert format(span.span.get_span_context().trace_id, "032x") in env["TRACEPARENT"]
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://collector.example.test"
    assert env["CHACK_DISABLE_STDOUT_EVENTS"] is None
