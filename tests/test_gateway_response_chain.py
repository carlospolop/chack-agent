from types import SimpleNamespace

import httpx
from openai import BadRequestError

import pytest

from chack_agent.backends import gateway_responses_backend
from chack_agent.backends.gateway_responses_backend import AgentsExecutor


@pytest.fixture(autouse=True)
def _fresh_route_memory(monkeypatch):
    monkeypatch.setattr(gateway_responses_backend, "_UNCHAINED_ROUTES", set(), raising=False)


def _executor(route=("https://gateway.invalid/v1", "model")):
    executor = AgentsExecutor.__new__(AgentsExecutor)
    executor._chain_route = route
    executor.agent = SimpleNamespace(instructions="")
    executor.max_turns = 5
    executor._config = SimpleNamespace(model=SimpleNamespace(primary="openrouter/model", provider="openrouter"))
    executor._system_prompt = lambda: "test"
    executor._maybe_summarize_for_next_turn = lambda **_kwargs: None
    executor._conversation = []
    executor._memory_limit = 0
    executor._summary = ""
    executor._summary_keep_messages = 0
    executor._previous_response_id = None
    executor._conversation_id = None
    executor._chain_responses = True
    return executor


def _result(transcript, response_id):
    return SimpleNamespace(final_output="done", to_input_list=lambda: list(transcript),
                           last_response_id=response_id, new_items=[], raw_responses=[])


def test_gateway_that_rejects_response_chains_continues_from_the_local_transcript():
    executor = _executor()
    calls = []
    request = httpx.Request("POST", "https://gateway.invalid/v1/responses")
    rejected = BadRequestError(
        "Error code: 400 - previous_response_id is not supported on this proxy. Each response request is independent.",
        response=httpx.Response(400, request=request), body=None)

    def run_agent(_agent, input_items, **kwargs):
        calls.append((list(input_items), kwargs["previous_response_id"]))
        if kwargs["previous_response_id"]:
            raise rejected
        transcript = [*input_items, {"role": "assistant", "content": f"answer {len(calls)}"}]
        return _result(transcript, f"resp-{len(calls)}")

    executor._run_agent = run_agent

    executor.invoke({"input": "first turn"})
    executor.invoke({"input": "use a tool this time"})
    executor.invoke({"input": "third turn"})

    # The rejected chained call is retried once with the whole thread, and later
    # turns send the thread directly instead of another unsupported chain.
    assert [previous for _, previous in calls] == [None, "resp-1", None, None]
    retry_input, third_input = calls[2][0], calls[3][0]
    assert [item["content"] for item in retry_input] == ["first turn", "answer 1", "use a tool this time"]
    assert [item["content"] for item in third_input][:4] == ["first turn", "answer 1", "use a tool this time", "answer 3"]
    assert third_input[-1]["content"] == "third turn"


def test_later_roles_on_a_route_that_rejected_chains_start_unchained():
    calls = []
    request = httpx.Request("POST", "https://gateway.invalid/v1/responses")
    rejected = BadRequestError("Error code: 400 - previous_response_id is not supported on this proxy.",
                               response=httpx.Response(400, request=request), body=None)

    def make(route):
        executor = _executor(route)

        def run_agent(_agent, input_items, **kwargs):
            calls.append((route, kwargs["previous_response_id"]))
            if kwargs["previous_response_id"] and route[1] == "model":
                raise rejected
            return _result([*input_items, {"role": "assistant", "content": "answer"}], f"resp-{len(calls)}")

        executor._run_agent = run_agent
        return executor

    first = make(("https://gateway.invalid/v1", "model"))
    first.invoke({"input": "one"})
    first.invoke({"input": "two"})
    rejected_calls = len(calls)
    # A later role on the same route never sends the unsupported chain.
    later = make(("https://gateway.invalid/v1", "model"))
    later.invoke({"input": "one"})
    later.invoke({"input": "two"})
    assert [previous for _, previous in calls[rejected_calls:]] == [None, None]
    # Another model on the gateway still chains.
    other = make(("https://gateway.invalid/v1", "other-model"))
    other.invoke({"input": "one"})
    other.invoke({"input": "two"})
    assert calls[-1][1] is not None
