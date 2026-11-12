import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chack_agent.backends.langgraph_backend import LangGraphExecutor, _with_reasoning_passthrough


class _ScriptedModel:
    """Chat model stand-in that answers each call from a script."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []
        self.bound = []

    def bind(self, **kwargs):
        self.bound.append((len(self.prompts), kwargs))
        return self

    def invoke(self, messages):
        self.prompts.append(list(messages))
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(reply, Exception):
            raise reply
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


async def _long_output(_ctx, _raw_args):
    return "x" * 1000


_NOOP_TOOLS = {"noop": SimpleNamespace(name="noop", on_invoke_tool=_noop)}


def _executor(model, *, max_turns=50, output_schema=None, summary_model=None,
              summary_trigger_messages=1000, summary_keep_messages=0,
              max_non_task_tools=0, function_tools=None, max_context_tokens=0, plan_tool=False):
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
        _plan_tool=plan_tool,
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


def test_the_last_allowed_turn_answers_without_tools_from_the_evidence_gathered():
    # The model keeps calling tools; its third and last call may not, and
    # whatever it says then is the run's answer instead of a lost run.
    last = AIMessage(content="Partial: host A is vulnerable; host B unverified.",
                     tool_calls=[{"name": "noop", "args": {}, "id": "call_9"}])
    model = _ScriptedModel([_tool_call(1), _tool_call(2), last])

    result = _executor(model, max_turns=3, function_tools=_NOOP_TOOLS).invoke({"input": "go"})

    assert result["output"] == "Partial: host A is vulnerable; host B unverified."
    assert len(model.prompts) == 3
    assert model.bound == [(2, {"tool_choice": "none"})]
    assert "tools are no longer available" in model.prompts[-1][-1].content
    assert len(result["intermediate_steps"]) == 2


_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def test_schema_invalid_answer_gets_one_repair_turn():
    model = _ScriptedModel([
        AIMessage(content="The finding reproduces."), AIMessage(content="It reproduces on 2.1."),
        AIMessage(content='{"verdict": "valid"}'),
    ])

    result = _executor(model, output_schema=_SCHEMA).invoke({"input": "review"})

    assert json.loads(result["output"]) == {"verdict": "valid"}
    assert len(model.prompts) == 3
    assert "not the required final answer" in model.prompts[1][-1].content
    repair = model.prompts[2][-1]
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
    # One nudge to continue, then one repair turn; neither is repeated.
    assert len(model.prompts) == 3


def test_content_filter_stop_ends_the_run_with_an_explicit_error():
    model = _ScriptedModel(
        [AIMessage(content="", response_metadata={"finish_reason": "content_filter"})]
    )

    with pytest.raises(RuntimeError, match="content_filter"):
        _executor(model, output_schema=_SCHEMA).invoke({"input": "review"})

    # A filtered response is a refusal, not an answer to repair: the same
    # request is asked again a bounded number of times, then the run fails.
    assert len(model.prompts) == 3
    assert all(prompt == model.prompts[0] for prompt in model.prompts)


def test_a_response_filtered_once_is_asked_again_and_the_run_continues_with_its_work():
    model = _ScriptedModel([
        _tool_call(0),
        AIMessage(content="", response_metadata={"finish_reason": "content_filter"}),
        AIMessage(content='{"verdict": "valid"}'),
    ])

    result = _executor(model, output_schema=_SCHEMA).invoke({"input": "review"})

    assert json.loads(result["output"]) == {"verdict": "valid"}
    assert len(model.prompts) == 3
    # The retry carries the tool round already done, not a fresh start.
    assert model.prompts[2] == model.prompts[1]
    assert any(isinstance(m, ToolMessage) for m in model.prompts[2])


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
    assert "The first two turns are complete." in model.prompts[-1][1].content
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
    assert "Rounds one and two found nothing." in last_prompt[1].content
    assert [type(m).__name__ for m in last_prompt[2:]] == ["HumanMessage", "AIMessage", "ToolMessage"]
    assert last_prompt[2].content == "Audit host A for CVE-1"
    assert last_prompt[3].tool_calls[0]["id"] == last_prompt[4].tool_call_id == "call_3"
    # The checkpoint is a message, so the cached system prompt is unchanged.
    assert len({prompt[0].content for prompt in model.prompts}) == 1


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


def test_a_tool_call_with_unparseable_arguments_is_answered_and_the_run_continues():
    broken = AIMessage(content="", invalid_tool_calls=[
        {"name": "noop", "args": '{"path": "/etc/pas', "id": "call_1", "error": "bad json", "type": "invalid_tool_call"},
    ])
    model = _ScriptedModel([broken, AIMessage(content="done")])

    result = _executor(model, function_tools=_NOOP_TOOLS).invoke({"input": "go"})

    assert result["output"] == "done"
    error = model.prompts[1][-1]
    assert isinstance(error, ToolMessage) and error.tool_call_id == "call_1"
    assert "could not be parsed as JSON" in error.content and '{"path": "/etc/pas' in error.content


def test_an_empty_reply_is_dropped_and_the_model_asked_to_continue():
    model = _ScriptedModel([_tool_call(1), AIMessage(content=""), AIMessage(content="done")])

    result = _executor(model, function_tools=_NOOP_TOOLS).invoke({"input": "go"})

    assert result["output"] == "done"
    retry = model.prompts[2]
    assert "was empty" in retry[-1].content
    assert not any(isinstance(m, AIMessage) and not m.tool_calls for m in retry)


def test_a_reply_cut_at_the_token_limit_is_continued():
    cut = AIMessage(content="Findings: 1. CVE-1 on host", response_metadata={"finish_reason": "length"})
    model = _ScriptedModel([cut, AIMessage(content=" A; 2. none else.")])

    result = _executor(model).invoke({"input": "go"})

    assert result["output"] == "Findings: 1. CVE-1 on host A; 2. none else."
    assert model.prompts[1][-2].content == "Findings: 1. CVE-1 on host"
    assert "output token limit" in model.prompts[1][-1].content


def test_recovery_prompts_stop_after_two_and_the_run_ends():
    model = _ScriptedModel([AIMessage(content="")])

    result = _executor(model).invoke({"input": "go"})

    assert result["output"] == ""
    assert len(model.prompts) == 3


def test_a_call_repeated_a_third_time_is_flagged():
    model = _ScriptedModel([_tool_call(1), _tool_call(2), _tool_call(3), AIMessage(content="done")])

    _executor(model, function_tools=_NOOP_TOOLS).invoke({"input": "go"})

    results = [m.content for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert results[:2] == ["ok", "ok"]
    assert "made 3 times" in results[2]


def test_an_unknown_tool_error_names_the_tools_that_exist():
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "nop", "args": {}, "id": "call_1"}]), AIMessage(content="done"),
    ])

    _executor(model, function_tools=_NOOP_TOOLS).invoke({"input": "go"})

    assert model.prompts[1][-1].content == "ERROR: Unknown tool 'nop'. Available tools: noop."


def test_tool_call_markup_left_in_an_argument_by_the_gateway_is_not_executed():
    received = []

    async def _record(_ctx, raw_args):
        received.append(json.loads(raw_args))
        return "ok"

    leaked = {
        "command": "sleep 40; tail -3 /tmp/run.log</\uff5cDSML\uff5c parameter>\n<parameter name=",
        "env": ["A=1 <\uff5ctool\u2581call\u2581end\uff5c>"],
        "cwd": "/tmp/a|b",
    }
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "exec", "args": leaked, "id": "call_1"}]), AIMessage(content="done"),
    ])

    _executor(model, function_tools={"exec": SimpleNamespace(name="exec", on_invoke_tool=_record)}).invoke({"input": "go"})

    assert received == [{"command": "sleep 40; tail -3 /tmp/run.log", "env": ["A=1"], "cwd": "/tmp/a|b"}]


def _long_tool_call(index, input_tokens=0):
    return AIMessage(
        content="", tool_calls=[{"name": "dump", "args": {"page": index}, "id": f"call_{index}"}],
        usage_metadata={"input_tokens": input_tokens, "output_tokens": 1, "total_tokens": input_tokens + 1},
    )


_DUMP_TOOLS = {"dump": SimpleNamespace(name="dump", on_invoke_tool=_long_output)}


def test_near_the_limit_old_tool_outputs_are_cleared_before_anything_is_summarized():
    # Seven results, the last call reporting 900 of 1000 tokens: the two
    # oldest outputs are cleared, the newest five and every call stay.
    model = _ScriptedModel([*(_long_tool_call(i, 100) for i in range(6)), _long_tool_call(6, 900),
                            AIMessage(content="done")])
    summary_model = _ScriptedModel([AIMessage(content="Pages 0 and 1 list CVE-7.")])
    executor = _executor(model, summary_model=summary_model, function_tools=_DUMP_TOOLS, max_context_tokens=1000)

    assert executor.invoke({"input": "Read every page"})["output"] == "done"

    # Only the two cleared outputs are condensed, into the checkpoint.
    assert len(summary_model.prompts) == 1
    assert summary_model.prompts[0][0].content.count("[tool dump]") == 2
    last_prompt = model.prompts[-1]
    assert "Pages 0 and 1 list CVE-7." in last_prompt[1].content
    results = [m for m in last_prompt if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == [f"call_{i}" for i in range(7)]
    assert ["cleared" in m.content for m in results] == [True, True, False, False, False, False, False]
    assert sum(isinstance(m, AIMessage) for m in last_prompt) == 7


def test_cleared_outputs_reach_the_summary_whole_in_chunks_folded_into_one_checkpoint():
    # Two cleared 70k-character outputs, each with a fact in its middle:
    # they do not fit one chunk, so the second chunk is summarized on top of
    # the first checkpoint and neither fact is clipped away.
    async def _big_page(_ctx, raw_args):
        page = json.loads(raw_args)["page"]
        return f"{'x' * 35_000} GHSA-PAGE-{page} {'y' * 35_000}"

    model = _ScriptedModel([*(_long_tool_call(i, 100) for i in range(6)), _long_tool_call(6, 900),
                            AIMessage(content="done")])
    summary_model = _ScriptedModel([AIMessage(content="Checkpoint A."), AIMessage(content="Checkpoint B.")])
    executor = _executor(model, summary_model=summary_model, max_context_tokens=1000,
                         function_tools={"dump": SimpleNamespace(name="dump", on_invoke_tool=_big_page)})

    assert executor.invoke({"input": "Read every page"})["output"] == "done"

    first, second = (prompt[0].content for prompt in summary_model.prompts)
    assert "GHSA-PAGE-0" in first and "GHSA-PAGE-1" not in first
    assert "GHSA-PAGE-1" in second and "Earlier checkpoint:\nCheckpoint A." in second
    assert "Checkpoint B." in model.prompts[-1][1].content


def test_a_context_overflow_error_shrinks_the_context_and_retries_once():
    overflow = RuntimeError("Error code: 400 - This model's maximum context length is 1000 tokens")
    model = _ScriptedModel([*(_long_tool_call(i) for i in range(7)), overflow, AIMessage(content="done")])
    summary_model = _ScriptedModel([AIMessage(content="Pages 0 and 1 are blank.")])
    executor = _executor(model, summary_model=summary_model, function_tools=_DUMP_TOOLS, max_context_tokens=100_000)

    assert executor.invoke({"input": "Read every page"})["output"] == "done"

    retried = [m.content for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert sum("cleared" in content for content in retried) == 2


def _plan_call(index, *statuses):
    todos = [{"content": f"step {n}", "status": status} for n, status in enumerate(statuses, 1)]
    return AIMessage(content="", tool_calls=[{"name": "todo_write", "args": {"todos": todos}, "id": f"plan_{index}"}])


def test_an_answer_with_open_plan_items_is_sent_back_once():
    model = _ScriptedModel([
        _plan_call(1, "completed", "pending"), AIMessage(content="done early"), AIMessage(content="done anyway"),
    ])

    result = _executor(model, plan_tool=True).invoke({"input": "go"})

    assert result["output"] == "done anyway"
    assert len(model.prompts) == 3
    nudge = model.prompts[2][-1].content
    assert "[ ] step 2" in nudge and "step 1" not in nudge


def test_a_finished_plan_lets_the_answer_through():
    model = _ScriptedModel([_plan_call(1, "pending"), _plan_call(2, "completed"), AIMessage(content="done")])

    result = _executor(model, plan_tool=True).invoke({"input": "go"})

    assert result["output"] == "done"
    assert len(model.prompts) == 3


def test_open_plan_items_are_recalled_after_ten_calls_without_an_update():
    model = _ScriptedModel([_plan_call(0, "in_progress"), *(_tool_call(i) for i in range(1, 13)),
                            _plan_call(13, "completed"), AIMessage(content="done")])

    _executor(model, plan_tool=True, function_tools=_NOOP_TOOLS).invoke({"input": "go"})

    results = [m.content for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    reminded = [i for i, content in enumerate(results) if "Plan reminder" in content]
    assert reminded == [10]
    assert "[~] step 1" in results[10]


def test_plan_updates_do_not_use_the_tool_budget():
    model = _ScriptedModel([_plan_call(1, "pending"), _tool_call(2), _plan_call(3, "completed"), AIMessage(content="done")])

    _executor(model, plan_tool=True, max_non_task_tools=1, function_tools=_NOOP_TOOLS).invoke({"input": "go"})

    results = [m.content for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert results[1].startswith("ok") and not any(r.startswith("ERROR") for r in results)


def test_independent_tool_calls_in_one_turn_run_in_parallel_and_keep_order(monkeypatch):
    # Two slow tools issued in one turn should overlap (total wall time near one
    # tool's duration, not the sum), and their results must come back in the
    # order the model issued them.
    import time as _time

    def _make(name, delay, payload):
        async def _run(_ctx, _raw):
            _time.sleep(delay)
            return payload
        return SimpleNamespace(name=name, on_invoke_tool=_run)

    tools = {"slow_a": _make("slow_a", 0.4, "RESULT_A"), "slow_b": _make("slow_b", 0.4, "RESULT_B")}
    both = AIMessage(content="", tool_calls=[
        {"name": "slow_a", "args": {}, "id": "call_a"},
        {"name": "slow_b", "args": {}, "id": "call_b"},
    ])
    model = _ScriptedModel([both, AIMessage(content="done")])

    monkeypatch.setenv("CHACK_LANGGRAPH_PARALLEL_TOOLS", "4")
    start = _time.perf_counter()
    result = _executor(model, function_tools=tools).invoke({"input": "go"})
    elapsed = _time.perf_counter() - start

    steps = result["intermediate_steps"]
    # Results preserved and in issue order.
    assert [s[0].tool for s in steps] == ["slow_a", "slow_b"]
    # The tool messages are delivered back to the model on the next turn, in
    # the order the calls were issued.
    tool_msgs = [m for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_msgs] == ["slow_a", "slow_b"]
    assert [m.content for m in tool_msgs] == ["RESULT_A", "RESULT_B"]
    # Overlapped: well under the 0.8s a serial run would take.
    assert elapsed < 0.7


def test_serial_default_runs_tool_calls_one_at_a_time(monkeypatch):
    import time as _time

    def _make(name):
        async def _run(_ctx, _raw):
            _time.sleep(0.3)
            return name
        return SimpleNamespace(name=name, on_invoke_tool=_run)

    tools = {"a": _make("a"), "b": _make("b")}
    both = AIMessage(content="", tool_calls=[
        {"name": "a", "args": {}, "id": "ca"},
        {"name": "b", "args": {}, "id": "cb"},
    ])
    model = _ScriptedModel([both, AIMessage(content="done")])

    monkeypatch.delenv("CHACK_LANGGRAPH_PARALLEL_TOOLS", raising=False)
    start = _time.perf_counter()
    result = _executor(model, function_tools=tools).invoke({"input": "go"})
    elapsed = _time.perf_counter() - start

    assert [s[0].tool for s in result["intermediate_steps"]] == ["a", "b"]
    # Serial: both sleeps run end to end.
    assert elapsed >= 0.6
