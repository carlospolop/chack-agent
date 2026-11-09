import json
from types import SimpleNamespace

import pytest
from agents import MaxTurnsExceeded
from langchain_core.messages import AIMessage, HumanMessage

from chack_agent.backends.langgraph_backend import LangGraphExecutor


class _ScriptedModel:
    """Chat model stand-in that answers each call from a script."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(list(messages))
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        return reply


def _tool_call(index=0):
    return AIMessage(content="", tool_calls=[{"name": "noop", "args": {}, "id": f"call_{index}"}])


def _executor(model, *, max_turns=50, output_schema=None):
    executor = LangGraphExecutor(
        _graph=None,
        _model_with_tools=model,
        _summary_model=SimpleNamespace(model_name="test-model"),
        _function_tools_by_name={},
        _mcp_tools_by_name={},
        _conversation=[],
        _memory_limit=0,
        _memory_reset_to=0,
        _base_system_prompt="system",
        _thread_id="thread-1",
        _max_non_task_tools=0,
        _require_task_steps_manager_init_first=False,
        _max_turns=max_turns,
        _summary_trigger_messages=1000,
        _summary_keep_messages=0,
        _summary_max_chars=0,
        _compaction_threshold_ratio=0.75,
        _max_context_tokens=0,
        _output_schema_json=json.dumps(output_schema) if output_schema else "",
        _output_schema_name="answer",
        _output_schema_strict=True,
    )
    executor.build_graph()
    return executor


def test_max_turns_counts_model_calls_not_graph_steps():
    # Five model calls with four tool rounds between them fit a five-turn budget.
    model = _ScriptedModel([_tool_call(1), _tool_call(2), _tool_call(3), _tool_call(4), AIMessage(content="done")])

    result = _executor(model, max_turns=5).invoke({"input": "go"})

    assert result["output"] == "done"
    assert len(model.prompts) == 5


def test_running_out_of_turns_ends_the_run_as_max_turns_exceeded():
    model = _ScriptedModel([_tool_call()])

    with pytest.raises(MaxTurnsExceeded, match=r"Max turns \(3\) exceeded"):
        _executor(model, max_turns=3).invoke({"input": "go"})

    assert len(model.prompts) == 3


_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def test_schema_invalid_answer_gets_one_repair_turn():
    model = _ScriptedModel([AIMessage(content="The finding reproduces."), AIMessage(content='{"verdict": "valid"}')])

    result = _executor(model, output_schema=_SCHEMA).invoke({"input": "review"})

    assert json.loads(result["output"]) == {"verdict": "valid"}
    assert len(model.prompts) == 2
    repair = model.prompts[1][-1]
    assert isinstance(repair, HumanMessage)
    assert "answer" in repair.content and "Invalid JSON output" in repair.content


def test_schema_valid_answer_is_returned_without_a_repair_turn():
    model = _ScriptedModel([AIMessage(content='Result:\n```json\n{"verdict": "valid"}\n```')])

    result = _executor(model, output_schema=_SCHEMA).invoke({"input": "review"})

    assert json.loads(result["output"]) == {"verdict": "valid"}
    assert len(model.prompts) == 1


def test_answer_still_invalid_after_the_repair_turn_is_returned_as_given():
    model = _ScriptedModel([AIMessage(content="first prose"), AIMessage(content="second prose")])

    result = _executor(model, output_schema=_SCHEMA).invoke({"input": "review"})

    assert result["output"] == "second prose"
    assert len(model.prompts) == 2


def test_content_filter_stop_ends_the_run_with_an_explicit_error():
    model = _ScriptedModel(
        [AIMessage(content="", response_metadata={"finish_reason": "content_filter"})]
    )

    with pytest.raises(RuntimeError, match="content_filter"):
        _executor(model, output_schema=_SCHEMA).invoke({"input": "review"})

    # A filtered response is a refusal, not an answer to repair.
    assert len(model.prompts) == 1
