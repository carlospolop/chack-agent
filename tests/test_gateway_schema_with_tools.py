import json
from types import SimpleNamespace

import httpx
import pytest
from agents import Agent, function_tool
from agents.exceptions import ModelBehaviorError
from openai import AsyncOpenAI

from chack_agent.backends.gateway_responses_backend import AgentsExecutor, _GatewayResponsesModel
from chack_agent.output_schema import JsonSchemaOutput

_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "maxLength": 200}},
           "required": ["answer"], "additionalProperties": False}


def _response(output, status="completed"):
    return {"id": "resp_x", "object": "response", "created_at": 0, "status": status, "model": "m",
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}


def _message(text, status="completed"):
    return {"type": "message", "id": "msg_x", "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _gateway(requests, under_schema, final=None):
    """A provider that decodes a JSON-schema format as a constraint on the whole reply."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if ((body.get("text") or {}).get("format") or {}).get("type") == "json_schema":
            return httpx.Response(200, json=under_schema())
        outputs = [item for item in body["input"] if item.get("type") == "function_call_output"]
        if not outputs:
            return httpx.Response(200, json=_response([{
                "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
                "arguments": "{}", "status": "completed"}]))
        text = final if final is not None else json.dumps({"answer": outputs[0]["output"]})
        return httpx.Response(200, json=_response([_message(text)]))

    return handle


def _run(under_schema, final=None):
    requests, looked_up = [], []

    @function_tool
    def lookup() -> str:
        """Look the answer up."""
        looked_up.append(True)
        return "found 42"

    def client():
        return AsyncOpenAI(api_key="test", base_url="https://openrouter.ai/api/v1", max_retries=0,
                           http_client=httpx.AsyncClient(
                               transport=httpx.MockTransport(_gateway(requests, under_schema, final))))

    executor = AgentsExecutor.__new__(AgentsExecutor)
    executor._config = SimpleNamespace(model=SimpleNamespace(primary="vendor/model"))
    executor._client_factory = client
    agent = Agent(name="Chack", instructions="Answer as JSON with an `answer` field.", tools=[lookup],
                  model=_GatewayResponsesModel(model="vendor/model", openai_client=client()),
                  output_type=JsonSchemaOutput(_SCHEMA, name="answer", strict=True))
    result = executor._run_agent(agent, [{"role": "user", "content": "find the answer"}], max_turns=5)
    return result.final_output, looked_up, requests


def test_a_gateway_run_with_an_output_schema_still_calls_its_tools():
    answer, looked_up, requests = _run(lambda: _response([_message('{"answer": "placeholder"}')]))

    assert looked_up == [True]
    assert answer == {"answer": "found 42"}
    assert all("format" not in (body.get("text") or {}) for body in requests)


def test_a_length_bounded_schema_no_longer_runs_the_reply_into_max_output_tokens():
    looping = _response([_message('{"answer": "' + "a" * 5000, status="incomplete")], status="incomplete")

    answer, looked_up, _ = _run(lambda: looping)

    assert looked_up == [True]
    assert answer == {"answer": "found 42"}


def test_a_gateway_answer_that_breaks_the_schema_is_still_rejected():
    with pytest.raises(ModelBehaviorError):
        _run(lambda: _response([_message('{"answer": "x"}')]), final='{"other": 1}')
