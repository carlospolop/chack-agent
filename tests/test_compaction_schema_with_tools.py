import json
from types import SimpleNamespace

import httpx
import pytest
from agents import Agent, function_tool
from openai import AsyncOpenAI

from chack_agent.backends.openai_compaction_backend import AgentsExecutor
from chack_agent.output_schema import JsonSchemaOutput

_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
           "additionalProperties": False}


def _response(output):
    return {"id": "resp_x", "object": "response", "created_at": 0, "status": "completed", "model": "m",
            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}


def _message(text):
    return {"type": "message", "id": "msg_x", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _gateway(requests):
    """A provider that decodes a JSON-schema format as a constraint on the whole reply:
    given one, the model can only write the answer, never call a tool."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if ((body.get("text") or {}).get("format") or {}).get("type") == "json_schema":
            return httpx.Response(200, json=_response([_message('{"answer": "placeholder"}')]))
        outputs = [item for item in body["input"] if item.get("type") == "function_call_output"]
        if not outputs:
            return httpx.Response(200, json=_response([{
                "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
                "arguments": "{}", "status": "completed"}]))
        return httpx.Response(200, json=_response([_message(json.dumps({"answer": outputs[0]["output"]}))]))

    return handle


def _run(base_url):
    requests, looked_up = [], []

    @function_tool
    def lookup() -> str:
        """Look the answer up."""
        looked_up.append(True)
        return "found 42"

    executor = AgentsExecutor.__new__(AgentsExecutor)
    executor._config = SimpleNamespace(model=SimpleNamespace(primary="model"))
    executor._client_factory = lambda: AsyncOpenAI(
        api_key="test", base_url=base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_gateway(requests))))
    agent = Agent(name="Chack", instructions="Answer as JSON with an `answer` field.", tools=[lookup],
                  model="model", output_type=JsonSchemaOutput(_SCHEMA, name="answer", strict=True))
    result = executor._run_agent(agent, [{"role": "user", "content": "find the answer"}], max_turns=5)
    return result.final_output, looked_up, requests


def test_a_gateway_run_with_an_output_schema_can_still_call_its_tools():
    answer, looked_up, requests = _run("https://openrouter.ai/api/v1")

    assert looked_up == [True]
    assert answer == {"answer": "found 42"}
    assert all("format" not in (body.get("text") or {}) for body in requests)


def test_a_gateway_answer_that_breaks_the_schema_is_still_rejected():
    from agents.exceptions import ModelBehaviorError

    executor = AgentsExecutor.__new__(AgentsExecutor)
    executor._config = SimpleNamespace(model=SimpleNamespace(primary="model"))
    executor._client_factory = lambda: AsyncOpenAI(
        api_key="test", base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_response([_message('{"other": 1}')])))))
    @function_tool
    def lookup() -> str:
        """Look the answer up."""
        return "found 42"

    agent = Agent(name="Chack", instructions="", tools=[lookup], model="model",
                  output_type=JsonSchemaOutput(_SCHEMA, name="answer", strict=True))

    # The format is not sent with tools, but the answer is still held to the schema.
    with pytest.raises(ModelBehaviorError):
        executor._run_agent(agent, [{"role": "user", "content": "answer"}], max_turns=2)


def test_openai_keeps_receiving_the_output_schema():
    _, _, requests = _run("https://api.openai.com/v1")

    assert requests[0]["text"]["format"]["type"] == "json_schema"
