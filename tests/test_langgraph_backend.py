import json
from types import SimpleNamespace

import pytest
from agents import MaxTurnsExceeded
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chack_agent.backends.langgraph_backend import LangGraphExecutor, _with_reasoning_passthrough


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


def _tool_call(index=0, input_tokens=0):
    return AIMessage(
        content="", tool_calls=[{"name": "noop", "args": {}, "id": f"call_{index}"}],
        usage_metadata={"input_tokens": input_tokens, "output_tokens": 1, "total_tokens": input_tokens + 1},
    )


async def _noop(_ctx, _raw_args):
    return "ok"


_NOOP_TOOLS = {"noop": SimpleNamespace(name="noop", on_invoke_tool=_noop)}


def _executor(model, *, max_turns=50, output_schema=None, summary_model=None,
              summary_trigger_messages=1000, summary_keep_messages=0,
              max_non_task_tools=0, function_tools=None, max_context_tokens=0):
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
        _max_context_tokens=max_context_tokens,
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


def test_context_near_the_limit_is_summarized_but_keeps_the_task_and_tool_pairs():
    # The third call reports 900 input tokens against a 1000-token window
    # compacting at 75%, so the fourth prompt carries a summary instead.
    model = _ScriptedModel([_tool_call(1, 100), _tool_call(2, 200), _tool_call(3, 900), AIMessage(content="done")])
    summary_model = _ScriptedModel([AIMessage(content="Rounds one and two found nothing.")])
    executor = _executor(
        model, summary_model=summary_model, summary_keep_messages=1,
        function_tools=_NOOP_TOOLS, max_context_tokens=1000,
    )

    assert executor.invoke({"input": "Audit host A for CVE-1"})["output"] == "done"

    last_prompt = model.prompts[-1]
    assert "Rounds one and two found nothing." in last_prompt[0].content
    assert [type(m).__name__ for m in last_prompt[1:]] == ["HumanMessage", "AIMessage", "ToolMessage"]
    assert last_prompt[1].content == "Audit host A for CVE-1"
    assert last_prompt[2].tool_calls[0]["id"] == last_prompt[3].tool_call_id == "call_3"


def test_long_tool_loops_are_not_summarized_by_message_count():
    # Sixty tool rounds are 121 messages but one conversation turn: without
    # token pressure, the agent keeps its full history.
    model = _ScriptedModel([*(_tool_call(i, 50) for i in range(60)), AIMessage(content="done")])
    summary_model = _ScriptedModel([AIMessage(content="unused")])
    executor = _executor(
        model, summary_model=summary_model, summary_trigger_messages=50, summary_keep_messages=20,
        function_tools=_NOOP_TOOLS, max_context_tokens=100_000, max_turns=100,
    )

    assert executor.invoke({"input": "Research deeply"})["output"] == "done"
    assert summary_model.prompts == []
    assert len(model.prompts[-1]) == 1 + 1 + 2 * 60


def test_system_prompt_stays_identical_while_the_tool_budget_runs_out():
    model = _ScriptedModel([_tool_call(1), _tool_call(2), _tool_call(3), AIMessage(content="done")])
    executor = _executor(model, max_non_task_tools=2, function_tools=_NOOP_TOOLS)

    assert executor.invoke({"input": "Use the tool"})["output"] == "done"

    assert len({prompt[0].content for prompt in model.prompts}) == 1
    results = [m.content for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert "1 of 2 non-task tool calls remain" in results[0]
    assert "all 2 non-task tool calls are used" in results[1]
    assert results[2].startswith("ERROR: Tool budget reached (2)")


def test_tool_results_carry_no_budget_note_while_the_budget_is_ample():
    model = _ScriptedModel([_tool_call(1), AIMessage(content="done")])
    executor = _executor(model, max_non_task_tools=300, function_tools=_NOOP_TOOLS)

    executor.invoke({"input": "Use the tool"})

    assert [m.content for m in model.prompts[-1] if isinstance(m, ToolMessage)] == ["ok"]


@pytest.mark.parametrize("field, value", [
    ("reasoning_content", "Plan: probe the login page next."),
    ("reasoning_details", [{"type": "reasoning.text", "text": "Plan: probe next."}]),
])
def test_reasoning_returned_by_the_model_is_sent_back_on_its_tool_call_turn(field, value):
    from langchain_openai import ChatOpenAI

    model = _with_reasoning_passthrough(ChatOpenAI)(model="m", api_key="k", base_url="http://gateway.invalid/v1")
    reply = model._create_chat_result({
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", field: value,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "noop", "arguments": "{}"}}],
        }}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).generations[0].message

    payload = model._get_request_payload([
        HumanMessage(content="go"), reply, ToolMessage(content="ok", tool_call_id="c1"),
    ])

    assistant = payload["messages"][1]
    assert assistant["role"] == "assistant" and assistant[field] == value
    assert all(field not in m for i, m in enumerate(payload["messages"]) if i != 1)


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
