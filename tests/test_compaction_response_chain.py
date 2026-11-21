from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from chack_agent.backends.openai_compaction_backend import AgentsExecutor

_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")


def _executor(conversation_id=None):
    executor = AgentsExecutor.__new__(AgentsExecutor)
    executor._config = SimpleNamespace(model=SimpleNamespace(primary="openrouter/model"))
    executor.agent = SimpleNamespace(instructions="", model_settings=None)
    executor.max_turns = 5
    executor._conversation = []
    executor._memory_limit = 0
    executor._memory_reset_to = 0
    executor._base_system_prompt = "system"
    executor._previous_response_id = None
    executor._conversation_id = conversation_id
    executor._compaction_threshold_ratio = 0
    executor._max_context_tokens = 0
    executor._compaction_model = ""
    executor._client_factory = None
    executor._chain_responses = True
    executor._maybe_compact = lambda _tokens: None
    return executor


def _result(transcript, response_id):
    return SimpleNamespace(final_output="done", to_input_list=lambda: list(transcript),
                           last_response_id=response_id, new_items=[], raw_responses=[])


@pytest.mark.parametrize("message", [
    "Error code: 400 - previous_response_id is not supported by this provider.",
    "Error code: 400 - {'error': {'message': \"Previous response with id 'resp_1' not found.\"}}",
])
def test_a_provider_that_keeps_no_responses_continues_every_turn_from_the_local_transcript(message):
    executor = _executor()
    calls = []
    rejected = BadRequestError(message, response=httpx.Response(400, request=_REQUEST), body=None)

    def run_agent(_agent, input_items, **kwargs):
        calls.append((list(input_items), kwargs["previous_response_id"]))
        if kwargs["previous_response_id"]:
            raise rejected
        return _result([*input_items, {"role": "assistant", "content": f"answer {len(calls)}"}], f"resp-{len(calls)}")

    executor._run_agent = run_agent
    executor.invoke({"input": "assigned task"})
    executor.invoke({"input": "use a tool this time"})
    executor.invoke({"input": "fix your answer"})

    # The rejected chained turn is retried once with the whole thread, so the model
    # still sees its assigned task; later turns never send the rejected chain again.
    assert [previous for _, previous in calls] == [None, "resp-1", None, None]
    assert [item["content"] for item in calls[2][0]] == ["assigned task", "answer 1", "use a tool this time"]
    assert [item["content"] for item in calls[3][0]] == [
        "assigned task", "answer 1", "use a tool this time", "answer 3", "fix your answer"]


def test_a_server_side_conversation_still_carries_the_thread_after_a_rejected_chain():
    executor = _executor(conversation_id="conv_1")
    calls = []
    rejected = BadRequestError("Error code: 400 - Previous response with id 'resp_1' not found.",
                               response=httpx.Response(400, request=_REQUEST), body=None)

    def run_agent(_agent, input_items, **kwargs):
        calls.append((list(input_items), kwargs["previous_response_id"], kwargs["conversation_id"]))
        if kwargs["previous_response_id"]:
            raise rejected
        return _result([*input_items, {"role": "assistant", "content": "answer"}], f"resp-{len(calls)}")

    executor._run_agent = run_agent
    executor.invoke({"input": "assigned task"})
    executor.invoke({"input": "next turn"})
    executor.invoke({"input": "third turn"})

    # Every turn succeeds; the conversation holds the history, so only each new turn is sent,
    # and the rejected chain is not sent again.
    assert [previous for _, previous, _ in calls] == [None, "resp-1", None, None]
    assert [(len(items), conversation) for items, _, conversation in calls[2:]] == [(1, "conv_1"), (1, "conv_1")]
