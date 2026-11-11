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
        # Real model calls return distinct messages. LangGraph's message
        # reducer uses IDs to replace edits to an existing message.
        return reply.model_copy(deep=True, update={"id": None})


def _tool_call(index=0):
    return AIMessage(content="", tool_calls=[{"name": "noop", "args": {}, "id": f"call_{index}"}])


def _executor(model, *, max_turns=50, output_schema=None, summary_model=None,
              summary_trigger_messages=1000, summary_keep_messages=0,
              max_non_task_tools=0, function_tools=None):
    executor = LangGraphExecutor(
        _graph=None,
        _model_with_tools=model,
        _summary_model=summary_model or SimpleNamespace(model_name="test-model"),
        _function_tools_by_name=function_tools or {},
        _mcp_tools_by_name={},
        _conversation=[],
        _memory_limit=0,
        _memory_reset_to=0,
        _base_system_prompt="system",
        _thread_id="thread-1",
        _max_non_task_tools=max_non_task_tools,
        _require_task_steps_manager_init_first=False,
        _max_turns=max_turns,
        _summary_trigger_messages=summary_trigger_messages,
        _summary_keep_messages=summary_keep_messages,
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


def test_summary_prunes_checkpoint_messages_and_keeps_current_request():
    model = _ScriptedModel([AIMessage(content="one"), AIMessage(content="two"), AIMessage(content="three")])
    summary_model = _ScriptedModel([AIMessage(content="The first two turns are complete.")])
    executor = _executor(
        model, summary_model=summary_model,
        summary_trigger_messages=2, summary_keep_messages=0,
    )

    executor.invoke({"input": "First"})
    executor.invoke({"input": "Second"})
    result = executor.invoke({"input": "Third"})

    assert result["output"] == "three"
    state = executor._graph.get_state({"configurable": {"thread_id": executor._thread_id}}).values
    assert [message.content for message in state["messages"]] == ["Third", "three"]
    assert "The first two turns are complete." in model.prompts[-1][0].content
    assert model.prompts[-1][-1].content == "Third"


def test_summary_keeps_tool_call_with_its_result():
    model = _ScriptedModel([_tool_call(1), _tool_call(2), AIMessage(content="done")])
    summary_model = _ScriptedModel([AIMessage(content="First tool round completed.")])
    executor = _executor(
        model, summary_model=summary_model,
        summary_trigger_messages=1, summary_keep_messages=1,
    )

    assert executor.invoke({"input": "Run two tools"})["output"] == "done"
    state = executor._graph.get_state({"configurable": {"thread_id": executor._thread_id}}).values
    assert len(state["messages"]) == 3
    assert state["messages"][0].tool_calls[0]["id"] == "call_2"
    assert state["messages"][1].tool_call_id == "call_2"
    assert state["messages"][2].content == "done"


def test_tool_budget_remaining_is_shown_to_model_after_tool_call():
    async def noop(_ctx, _raw_args):
        return "ok"

    model = _ScriptedModel([_tool_call(1), AIMessage(content="done")])
    executor = _executor(
        model, max_non_task_tools=1,
        function_tools={"noop": SimpleNamespace(name="noop", on_invoke_tool=noop)},
    )
    assert executor.invoke({"input": "Use the tool once"})["output"] == "done"
    assert "1 non-task tool calls remain" in model.prompts[0][0].content
    assert "tool-call budget is exhausted" in model.prompts[1][0].content


def test_web_verification_guidance_only_when_web_tool_is_available():
    model = _ScriptedModel([AIMessage(content="done")])
    without_web = _executor(model)
    without_web.invoke({"input": "Check the docs"})
    assert "use an available web tool" not in model.prompts[-1][0].content

    async def web_fetch(_ctx, _raw_args):
        return "source"

    with_web = _executor(
        model, function_tools={"web_fetch": SimpleNamespace(name="web_fetch", on_invoke_tool=web_fetch)},
    )
    with_web.invoke({"input": "Check the docs"})
    assert "use an available web tool" in model.prompts[-1][0].content
