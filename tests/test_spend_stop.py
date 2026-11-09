"""A run whose caller stops it (e.g. its spend cap is reached) makes no more model calls,
and its callers learn each call's usage while the run is still going."""

import json
import threading
from types import SimpleNamespace

import pytest
from agents import Agent, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from langchain_core.messages import AIMessage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from chack_agent.backends import gateway_responses_backend, openai_compaction_backend
from chack_agent.live_cost_state import reset_active_usage_sink, set_active_usage_sink
from chack_tools.cancellation import (
    RunCancelled,
    cancellation_requested,
    reset_cancellation_event,
    set_cancellation_event,
)
from chack_tools.subagent_config import run_parallel_subagent_prompts
from chack_agent.usage_observer import current_observer, observe

from tests.test_langgraph_backend import _executor, _ScriptedModel, _tool_call

MODEL = "gpt-5.6-sol"


@pytest.fixture
def cancel_event():
    event = threading.Event()
    token = set_cancellation_event(event)
    yield event
    reset_cancellation_event(token)


@pytest.fixture
def usage_reports():
    reports = []
    token = set_active_usage_sink(lambda model, prompt, completion, cached, write: reports.append(
        (model, prompt, completion)))
    yield reports
    reset_active_usage_sink(token)


# --------------------------------------------------------------------------
# LangGraph backend
# --------------------------------------------------------------------------

class _CancellingModel(_ScriptedModel):
    """Asks for a tool on every call; the run is cancelled during the first one."""

    def __init__(self, event):
        super().__init__([_tool_call(1), _tool_call(2), AIMessage(content="done")])
        self.event = event

    def invoke(self, messages):
        reply = super().invoke(messages)
        self.event.set()
        return reply


def test_langgraph_run_makes_no_model_call_after_it_is_cancelled(cancel_event):
    model = _CancellingModel(cancel_event)

    with pytest.raises(RunCancelled):
        _executor(model).invoke({"input": "go"})

    assert len(model.prompts) == 1


# --------------------------------------------------------------------------
# OpenAI Agents SDK backends
# --------------------------------------------------------------------------

def _message(text):
    return ResponseOutputMessage(id="msg", type="message", role="assistant", status="completed",
                                 content=[ResponseOutputText(type="output_text", text=text, annotations=[])])


class _ToolThenAnswerModel(Model):
    """Calls ``work`` on each turn but the last; every call bills 1000 input / 100 output tokens."""

    def __init__(self, turns, usage_reports):
        self.turns = turns
        self.calls = 0
        self.reports_seen_at_call = []
        self.usage_reports = usage_reports

    def with_client(self, _client):
        return self

    async def get_response(self, *args, **kwargs):
        self.reports_seen_at_call.append(len(self.usage_reports))
        self.calls += 1
        usage = Usage(requests=1, input_tokens=1000, output_tokens=100, total_tokens=1100)
        if self.calls < self.turns:
            output = [ResponseFunctionToolCall(type="function_call", call_id=f"c{self.calls}", name="work",
                                               arguments=json.dumps({}), id=f"fc{self.calls}", status="completed")]
        else:
            output = [_message("done")]
        return ModelResponse(output=output, usage=usage, response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


class _Client:
    async def close(self):
        return None


def _agent(model, on_tool=lambda: None):
    @function_tool
    def work() -> str:
        """Do one unit of work."""
        on_tool()
        return "ok"

    return Agent(name="worker", instructions="work", model=model, tools=[work])


def _config():
    return SimpleNamespace(model=SimpleNamespace(primary=MODEL, provider="openai"),
                           agent=SimpleNamespace())


def _compaction_executor(agent):
    return openai_compaction_backend.AgentsExecutor(
        _config=_config(), agent=agent, max_turns=10, _conversation=[], _memory_limit=0, _memory_reset_to=0,
        _base_system_prompt="work", _previous_response_id=None, _conversation_id=None,
        _compaction_threshold_ratio=0.75, _max_context_tokens=0, _compaction_model="",
        _client_factory=_Client,
    )


def _gateway_executor(agent):
    return gateway_responses_backend.AgentsExecutor(
        _config=_config(), agent=agent, _summary_agent=agent, max_turns=10, _conversation=[], _memory_limit=0,
        _memory_reset_to=0, _base_system_prompt="work", _previous_response_id=None, _conversation_id=None,
        _summary="", _summary_trigger_messages=1000, _summary_keep_messages=0, _summary_max_chars=0,
        _compaction_threshold_ratio=0.75, _max_context_tokens=0, _client_factory=_Client,
    )


@pytest.mark.parametrize("build", [_compaction_executor, _gateway_executor], ids=["openai", "gateway"])
def test_sdk_run_reports_each_call_while_it_runs_and_counts_it_once(build, usage_reports):
    model = _ToolThenAnswerModel(turns=3, usage_reports=usage_reports)

    result = build(_agent(model)).invoke({"input": "go"})

    assert result["output"] == "done"
    # Each call's usage was reported before the next call started...
    assert model.reports_seen_at_call == [0, 1, 2]
    # ...and not reported again when the run ended.
    assert usage_reports == [(MODEL, 1000, 100)] * 3


@pytest.mark.parametrize("build", [_compaction_executor, _gateway_executor], ids=["openai", "gateway"])
def test_sdk_run_makes_no_model_call_after_it_is_cancelled(build, cancel_event, usage_reports):
    model = _ToolThenAnswerModel(turns=5, usage_reports=usage_reports)

    with pytest.raises(RunCancelled):
        build(_agent(model, on_tool=cancel_event.set)).invoke({"input": "go"})

    assert model.calls == 1
    assert usage_reports == [(MODEL, 1000, 100)]


# --------------------------------------------------------------------------
# Parallel sub-agents
# --------------------------------------------------------------------------

def test_parallel_sub_agents_run_with_the_runs_observer_and_cancellation(cancel_event):
    seen = []

    def runner(prompt):
        seen.append((current_observer(), cancellation_requested()))
        return prompt

    def observer(_event):
        return None

    cancel_event.set()
    with observe(observer):
        run_parallel_subagent_prompts(["a", "b", "c"], runner)

    assert seen == [(observer, True)] * 3


# --------------------------------------------------------------------------
# Responses compaction
# --------------------------------------------------------------------------

class _CompactingClient:
    """``OpenAI`` whose ``responses.compact`` bills 5000 input / 500 output tokens."""

    calls = []

    def __init__(self, **_):
        self.responses = SimpleNamespace(compact=self.compact)

    def compact(self, **kwargs):
        _CompactingClient.calls.append(kwargs)
        return SimpleNamespace(id="resp-2", usage=SimpleNamespace(input_tokens=5000, output_tokens=500,
                                                                  input_tokens_details=None))


@pytest.fixture
def compacting(monkeypatch):
    _CompactingClient.calls = []
    monkeypatch.setattr(openai_compaction_backend, "OpenAI", _CompactingClient)
    executor = _compaction_executor(_agent(_ToolThenAnswerModel(turns=1, usage_reports=[])))
    executor._config.credentials = SimpleNamespace(openai_api_key="k", openai_org_id="")
    executor._compaction_model = MODEL
    executor._previous_response_id = "resp-1"
    return executor


def test_a_compaction_call_reports_its_usage(compacting, usage_reports):
    assert compacting.compact_for_resume().succeeded

    assert usage_reports == [(MODEL, 5000, 500)]


def test_a_cancelled_run_makes_no_compaction_call(compacting, cancel_event, usage_reports):
    cancel_event.set()

    with pytest.raises(RunCancelled):
        compacting.compact_for_resume()

    assert _CompactingClient.calls == []
    assert compacting._previous_response_id == "resp-1"


class _AnswerThenCancelledModel(_ToolThenAnswerModel):
    """Answers at once; the run is cancelled after that call (no budget left for another)."""

    def __init__(self, event, usage_reports):
        super().__init__(turns=1, usage_reports=usage_reports)
        self.event = event

    async def get_response(self, *args, **kwargs):
        response = await super().get_response(*args, **kwargs)
        self.event.set()
        return response


def test_a_run_cancelled_after_its_answer_keeps_it_and_makes_no_compaction_call(
        compacting, cancel_event, usage_reports):
    executor = _compaction_executor(_agent(_AnswerThenCancelledModel(cancel_event, usage_reports)))
    executor._config.credentials = SimpleNamespace(openai_api_key="k", openai_org_id="")
    executor._compaction_model = MODEL
    executor._previous_response_id = "resp-1"
    executor._memory_limit = executor._memory_reset_to = 1  # the turn's messages call for a compaction

    result = executor.invoke({"input": "go"})

    assert result["output"] == "done", "the finished answer was lost to the compaction after it"
    assert _CompactingClient.calls == []
    assert usage_reports == [(MODEL, 1000, 100)]


def test_a_gateway_run_cancelled_after_its_answer_keeps_it_and_makes_no_summary_call(cancel_event, usage_reports):
    model = _AnswerThenCancelledModel(cancel_event, usage_reports)
    executor = _gateway_executor(_agent(model))
    # One earlier message: the turn's two take the conversation past the summary trigger.
    executor._conversation = [{"role": "user", "content": "earlier"}]
    executor._summary_trigger_messages = 1

    result = executor.invoke({"input": "go"})

    assert result["output"] == "done", "the finished answer was lost to the summary after it"
    assert model.calls == 1, "a cancelled run made a summary call"
