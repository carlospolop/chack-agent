"""A CLI run whose final message is not an answer is a provider failure."""
from __future__ import annotations

import json

import pytest

from chack_agent.agent import _backend_failure_reason, _looks_like_backend_failure_output
from chack_agent.backends.codex_backend import build_executor
from chack_agent.backends.final_answer import PLACEHOLDER, TOOL_CALL_AS_TEXT, final_answer_problem
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["true_positive", "false_positive"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
_EXEC = {"exec": {"properties": ["command", "cwd", "timeout"], "required": ["command"]}}


def _problem(output: str, schema: dict | None = _VERDICT_SCHEMA) -> str:
    kind, _detail = final_answer_problem(
        output, output_schema_json=json.dumps(schema) if schema else "", tools=_EXEC
    )
    return kind


@pytest.mark.parametrize(
    "output",
    [
        '{"command": "curl -s http://target/api?id=1", "cwd": "/work"}',
        '```json\n{"command": "ls"}\n```',
        '{"name": "exec", "arguments": {"command": "ls"}}',
        "<minimax:tool_call>\n<invoke name=\"exec\">\n<parameter name=\"command\">ls</parameter>",
        "I'll inspect it.\n<|tool_calls_section_begin|><|tool_call_begin|>functions.exec:0",
        "<tool_call>\n{\"name\": \"exec\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
    ],
)
def test_a_tool_call_written_as_text_is_not_an_answer(output: str) -> None:
    assert _problem(output) == TOOL_CALL_AS_TEXT
    assert _problem(output, schema=None) == TOOL_CALL_AS_TEXT


@pytest.mark.parametrize(
    "output",
    [
        '{"verdict": "false_positive", "reason": "pending"}',
        '{"verdict": "true_positive", "reason": "TBD."}',
    ],
)
def test_a_placeholder_answer_is_not_an_answer(output: str) -> None:
    assert _problem(output) == PLACEHOLDER


@pytest.mark.parametrize(
    "output",
    [
        '{"verdict": "true_positive", "reason": "id is concatenated into the SQL query"}',
        # An answer may quote tool-call markup or name a pending state.
        '{"verdict": "false_positive", "reason": "<tool_call> is only echoed back"}',
        '{"verdict": "false_positive", "reason": "the order stays pending until paid"}',
        # Text that only mentions a tool, or JSON that is not a tool's arguments.
        "Run exec with command ls to reproduce it.",
        '{"severity": "high"}',
    ],
)
def test_answers_are_kept(output: str) -> None:
    schema = None if not output.startswith('{"verdict"') else _VERDICT_SCHEMA
    assert _problem(output, schema=schema) == ""


def test_a_tool_call_inside_a_valid_schema_answer_is_kept() -> None:
    schema = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
    assert _problem('{"command": "ls"}', schema=schema) == ""


def _codex_executor():
    config = ChackConfig(
        model=ModelConfig(primary="minimax-m2.5", provider="codex"),
        agent=AgentConfig(main_action="test", sub_action="test", output_schema_json=_VERDICT_SCHEMA),
        session=SessionConfig(max_turns=3),
        tools=ToolsConfig(exec_enabled=True, task_steps_manager_enabled=False),
        credentials=CredentialsConfig(openai_api_key="sk-test"),
        logging=LoggingConfig(),
        system_prompt="test",
        env={},
    )
    return build_executor(
        config, system_prompt="test", max_turns=3, memory_max_messages=3, memory_reset_to_messages=1
    )


def test_codex_reports_a_tool_call_written_as_text_as_a_provider_failure(monkeypatch) -> None:
    executor = _codex_executor()
    monkeypatch.setattr(
        executor,
        "_run_codex",
        lambda _prompt: ('{"command": "curl -s http://target/", "cwd": "/work"}', [], None),
    )

    output = executor.invoke({"input": "triage the lead"})["output"]

    assert output.startswith("ERROR: Codex model wrote a tool call as text instead of calling the tool")
    assert "'exec'" in output
    assert _looks_like_backend_failure_output(output)
    assert _backend_failure_reason(output) == "backend_failure_malformed_tool_call"


def test_codex_reports_a_placeholder_as_a_provider_failure(monkeypatch) -> None:
    executor = _codex_executor()
    monkeypatch.setattr(
        executor, "_run_codex", lambda _prompt: ('{"verdict": "false_positive", "reason": "pending"}', [], None)
    )

    output = executor.invoke({"input": "triage the lead"})["output"]

    assert output.startswith("ERROR: Codex model returned a placeholder instead of an answer")
    assert _backend_failure_reason(output) == "backend_failure_no_answer"


def test_codex_schema_answer_is_the_last_message_not_all_of_them() -> None:
    executor = _codex_executor()
    interim = '{"verdict": "false_positive", "reason": "pending"}'
    final = '{"verdict": "true_positive", "reason": "id reaches the query unescaped"}'

    assert executor._final_output([interim, "", final]) == final

    executor._output_schema_json = ""
    assert executor._final_output(["first", "second"]) == "first\nsecond"


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("ERROR: Claude provider returned malformed tool calls: 3 tool call(s)", "backend_failure_malformed_tool_call"),
        ("ERROR: Claude finished without a final answer.", "backend_failure_no_answer"),
        ("ERROR: Claude finished without the structured output its schema requires.", "backend_failure_no_answer"),
        ("ERROR: Claude exec failed (exit=1).", "backend_failure"),
    ],
)
def test_claude_missing_answers_are_distinguishable_backend_failures(output: str, reason: str) -> None:
    assert _looks_like_backend_failure_output(output)
    assert _backend_failure_reason(output) == reason
