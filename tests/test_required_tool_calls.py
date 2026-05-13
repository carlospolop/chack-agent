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

    def invoke(self, payload, context=None):
        del payload, context
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
