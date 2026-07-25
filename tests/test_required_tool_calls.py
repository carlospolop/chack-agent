from __future__ import annotations

from dataclasses import replace

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


class _Executor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.inputs = []

    def invoke(self, payload, context=None):
        del context
        self.inputs.append(str(payload.get("input", "") or ""))
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return {"output": "done", "intermediate_steps": [], "raw_result": None}


class _TestChack(Chack):
    def __init__(self, executor: _Executor, config: ChackConfig):
        self._test_executor = executor
        super().__init__(config)

    def _get_executor(self, *args, **kwargs):
        del args, kwargs
        return self._test_executor


def _config(**tool_overrides) -> ChackConfig:
    tool_payload = {"min_tools_used": 0, **tool_overrides}
    return ChackConfig(
        model=ModelConfig(primary="test-model", provider="openai"),
        agent=AgentConfig(
            self_critique_enabled=False,
            main_action="test",
            sub_action="required-tools",
        ),
        session=SessionConfig(long_term_memory_enabled=False),
        tools=replace(ToolsConfig(), **tool_payload),
        credentials=CredentialsConfig(),
        logging=LoggingConfig(level="ERROR"),
        system_prompt="test system",
        env={},
    )


def test_required_tool_call_retries_until_tool_is_seen():
    executor = _Executor([
        {"output": "I am done", "intermediate_steps": [], "raw_result": None},
        {
            "output": "saved",
            "intermediate_steps": [
                {
                    "tool": "chack_tools-update_vulnerability",
                    "tool_input": {"confirmation": "false_positive"},
                }
            ],
            "raw_result": None,
        },
    ])
    agent = _TestChack(executor, _config())

    result = agent.run(
        "required-tool-success",
        "finish only after updating",
        required_tool_names=["update_vulnerability"],
        required_tool_call_attempts=3,
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )

    assert executor.calls == 2
    assert result.output == "saved"
    assert result.tool_counts["chack_tools-update_vulnerability"] == 1


def test_required_tool_call_returns_error_after_retry_budget():
    executor = _Executor([
        {"output": "I am done", "intermediate_steps": [], "raw_result": None},
        {"output": "still done", "intermediate_steps": [], "raw_result": None},
    ])
    agent = _TestChack(executor, _config())

    result = agent.run(
        "required-tool-failure",
        "finish only after updating",
        required_tool_names="update_vulnerability",
        required_tool_call_attempts=1,
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )

    assert executor.calls == 2
    assert result.output == "ERROR: Agent finished without calling required tool(s): update_vulnerability"
    assert result.run1_output == result.output


def test_backend_failure_does_not_resume_to_satisfy_tool_requirements():
    failure = (
        "ERROR: Codex exec failed (exit=1).\n"
        "You've hit your usage limit. Try again next week."
    )
    executor = _Executor([
        {"output": failure, "intermediate_steps": [], "raw_result": None},
        {"output": "must not run", "intermediate_steps": [], "raw_result": None},
    ])
    agent = _TestChack(
        executor,
        _config(min_tools_used=2, missing_tools_reminders_max=3),
    )

    result = agent.run(
        "terminal-backend-failure",
        "inspect the repository",
        required_tool_names=["update_vulnerability"],
        required_tool_call_attempts=3,
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )

    assert executor.calls == 1
    assert result.output == failure
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_self_critique_reuses_same_executor_session_without_repeating_previous_answer():
    executor = _Executor([
        {"output": "first answer", "intermediate_steps": [], "raw_result": None},
        {"output": "improved answer", "intermediate_steps": [], "raw_result": None},
    ])
    config = _config()
    config.agent.self_critique_enabled = True
    agent = _TestChack(executor, config)

    result = agent.run(
        "self-critique-session-reuse",
        "original request",
        enable_self_critique=True,
        require_task_steps_manager_init_first=False,
    )

    assert executor.calls == 2
    assert result.run1_output == "first answer"
    assert result.run2_output == "improved answer"
    assert result.output == "improved answer"
    assert executor.inputs[0] == "original request"
    assert "original request" in executor.inputs[1]
    assert "Previous answer:" not in executor.inputs[1]
    assert "first answer" not in executor.inputs[1]
    assert "Reuse the conversation context" in executor.inputs[1]
    assert "targeted tool calls" in executor.inputs[1]


def test_self_critique_is_disabled_by_default():
    executor = _Executor([
        {"output": "first answer", "intermediate_steps": [], "raw_result": None},
        {"output": "should not be used", "intermediate_steps": [], "raw_result": None},
    ])
    config = _config()
    agent = _TestChack(executor, config)

    result = agent.run(
        "self-critique-default-off",
        "original request",
        require_task_steps_manager_init_first=False,
    )

    assert executor.calls == 1
    assert result.output == "first answer"
    assert result.run2_output == ""


def test_self_critique_rounds_override_runs_try_harder_multiple_times():
    executor = _Executor([
        {"output": "first answer", "intermediate_steps": [], "raw_result": None},
        {"output": "second answer", "intermediate_steps": [], "raw_result": None},
        {"output": "third answer", "intermediate_steps": [], "raw_result": None},
        {"output": "fourth answer", "intermediate_steps": [], "raw_result": None},
    ])
    config = _config()
    config.agent.self_critique_rounds = 3
    agent = _TestChack(executor, config)

    result = agent.run(
        "self-critique-multi-round",
        "original request",
        require_task_steps_manager_init_first=False,
    )

    assert executor.calls == 4
    assert result.run1_output == "first answer"
    assert result.run2_output == "fourth answer"
    assert result.output == "fourth answer"
    assert "Previous answer:" not in executor.inputs[1]
    assert "Previous answer:" not in executor.inputs[2]
    assert "Previous answer:" not in executor.inputs[3]
    assert "first answer" not in executor.inputs[1]
    assert "second answer" not in executor.inputs[2]
    assert "third answer" not in executor.inputs[3]
    assert "original request" in executor.inputs[1]
    assert "original request" in executor.inputs[2]
    assert "original request" in executor.inputs[3]
