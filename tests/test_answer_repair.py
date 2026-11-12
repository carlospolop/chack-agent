from __future__ import annotations

import json
from dataclasses import replace

import pytest

from chack_agent import Chack
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


class _Executor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs: list[str] = []

    def invoke(self, payload, context=None):
        del context
        self.inputs.append(str(payload.get("input", "") or ""))
        output = self.outputs.pop(0) if self.outputs else self.inputs and "{}"
        return {"output": output, "intermediate_steps": [], "raw_result": None}


class _TestChack(Chack):
    def __init__(self, executor: _Executor, config: ChackConfig):
        self._test_executor = executor
        super().__init__(config)

    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return self._test_executor


def _agent(outputs, *, model="test-model", schema=None):
    config = ChackConfig(
        model=ModelConfig(primary=model, provider="openai"),
        agent=AgentConfig(
            self_critique_enabled=False,
            output_schema_json=schema,
            output_schema_name="answer",
        ),
        session=SessionConfig(long_term_memory_enabled=False),
        tools=replace(ToolsConfig(), min_tools_used=0, task_steps_manager_enabled=False),
        credentials=CredentialsConfig(),
        logging=LoggingConfig(level="ERROR"),
        system_prompt="test system",
        env={},
    )
    executor = _Executor(outputs)
    return _TestChack(executor, config), executor


def _file_must_exist(answer: str) -> str:
    path = json.loads(answer)["path"]
    return "" if path == "app.py" else f"path {path!r} is not a repository file"


def test_an_accepted_answer_is_returned_without_any_extra_turn():
    agent, executor = _agent(['{"path": "app.py"}'], schema=_SCHEMA)

    result = agent.run("s", "find it", output_validator=_file_must_exist)

    assert json.loads(result.output) == {"path": "app.py"}
    assert executor.inputs == ["find it"]


def test_a_caller_rejection_is_repaired_in_the_same_session_with_its_reason():
    agent, executor = _agent(['{"path": "app.py:12"}', '{"path": "app.py"}'], schema=_SCHEMA)

    result = agent.run("s", "find it", output_validator=_file_must_exist)

    assert json.loads(result.output) == {"path": "app.py"}
    assert len(executor.inputs) == 2
    assert "'app.py:12' is not a repository file" in executor.inputs[1]
    assert "find it" not in executor.inputs[1]


def test_a_validator_that_raises_rejects_the_answer():
    def strict(answer: str) -> str:
        raise ValueError("Enhancement v001 is invalid: code_path must be a file")

    agent, executor = _agent(['{"path": "x"}', '{"path": "y"}'], model="gpt-5.5", schema=_SCHEMA)

    agent.run("s", "find it", output_validator=strict)

    assert "code_path must be a file" in executor.inputs[1]


def test_an_answer_off_the_schema_is_repaired_without_a_caller_validator():
    agent, executor = _agent(['{"file": "app.py"}', '{"path": "app.py"}'], schema=_SCHEMA)

    result = agent.run("s", "find it")

    assert json.loads(result.output) == {"path": "app.py"}
    assert "path" in executor.inputs[1]


@pytest.mark.parametrize(
    "model,repairs",
    [
        ("gpt-5.5", 1),
        ("claude-opus-4-8", 1),
        ("us.anthropic.claude-sonnet-4-6", 1),
        ("bakkt/kimi-k3", 2),
        ("minimax-m3", 2),
    ],
)
def test_repair_turns_stop_at_the_model_familys_limit(model, repairs):
    agent, executor = _agent(['{"path": "bad"}'] * 5, model=model, schema=_SCHEMA)

    result = agent.run("s", "find it", output_validator=_file_must_exist)

    assert len(executor.inputs) == 1 + repairs
    assert json.loads(result.output) == {"path": "bad"}


def test_backend_errors_empty_answers_and_unstructured_agents_get_no_repair_turn():
    agent, executor = _agent(["ERROR: codex exec failed"], schema=_SCHEMA)
    agent.run("s", "find it", output_validator=_file_must_exist)
    assert len(executor.inputs) == 1

    agent, executor = _agent([""], schema=_SCHEMA)
    agent.run("s", "find it", output_validator=_file_must_exist)
    assert len(executor.inputs) == 1

    agent, executor = _agent(["plain prose answer"])
    agent.run("s", "find it")
    assert len(executor.inputs) == 1
