import asyncio
import json
import logging

import httpx
import pytest
from openai import AsyncOpenAI

from agents import Agent, RunConfig, Runner
from agents.exceptions import ModelBehaviorError

from chack_agent.backends.gateway_responses_backend import _GatewayResponsesModel
from chack_agent.live_cost_state import reset_active_usage_sink, set_active_usage_sink


def _incomplete_response(text):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "vendor/model",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "incomplete",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1200, "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 4096, "output_tokens_details": {"reasoning_tokens": 96},
            "total_tokens": 5296,
        },
    }


def _run(text):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_incomplete_response(text)))
    client = AsyncOpenAI(api_key="x", base_url="https://gateway.invalid/v1",
                         http_client=httpx.AsyncClient(transport=transport), max_retries=0)
    agent = Agent(name="role", instructions="answer", model=_GatewayResponsesModel(model="vendor/model", openai_client=client))
    charged = []
    token = set_active_usage_sink(lambda *values: charged.append(values))
    try:
        with pytest.raises(ModelBehaviorError, match="max_output_tokens"):
            asyncio.run(Runner.run(agent, "go", run_config=RunConfig(tracing_disabled=True)))
    finally:
        reset_active_usage_sink(token)
    return charged


def test_a_response_cut_at_max_output_tokens_is_charged_and_its_partial_output_described(caplog):
    looping = '{"paths": [' + ', '.join(['"/api/v1/items"'] * 2000)

    with caplog.at_level(logging.WARNING, logger="chack.gateway_responses_backend"):
        charged = _run(looping)

    assert charged == [("vendor/model", 1200, 4096, 200, 0)]
    message = next(r.getMessage() for r in caplog.records if "ended incomplete" in r.getMessage())
    assert "(max_output_tokens)" in message and "reasoning=96" in message
    assert f"partial output {len(looping)} chars" in message
    ratio = float(message.split("compressed/raw=")[1].split(",")[0])
    assert ratio < 0.05


def test_a_legitimately_long_answer_reads_as_not_repetitive(caplog):
    varied = "\n".join(f"/api/v{i % 7}/resource-{i * 7919 % 10007}/{chr(97 + i % 26)}" for i in range(1500))

    with caplog.at_level(logging.WARNING, logger="chack.gateway_responses_backend"):
        _run(varied)

    message = next(r.getMessage() for r in caplog.records if "ended incomplete" in r.getMessage())
    assert float(message.split("compressed/raw=")[1].split(",")[0]) > 0.1
    assert float(message.split("top repeated line share=")[1]) < 0.01
