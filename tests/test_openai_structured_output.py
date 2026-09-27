import json
from types import SimpleNamespace

from chack_agent.backends.openai_compaction_backend import AgentsExecutor


def _executor_with_final_output(final_output):
    executor = AgentsExecutor.__new__(AgentsExecutor)
    executor.agent = SimpleNamespace(instructions="")
    executor._base_system_prompt = "test"
    executor._conversation = []
    executor._previous_response_id = None
    executor._conversation_id = None
    executor._maybe_compact = lambda _tokens: None
    executor._invoke_runner_with_recovery = lambda **_kwargs: SimpleNamespace(
        final_output=final_output,
        to_input_list=lambda: [],
        last_response_id=None,
        new_items=[],
        raw_responses=[],
    )
    return executor


def test_openai_schema_object_is_returned_as_json_text():
    output = {"tech_summary": "It's a browser app", "project_types": ["webapp"]}

    result = _executor_with_final_output(output).invoke({"input": "analyze"})

    assert isinstance(result["output"], str)
    assert json.loads(result["output"]) == output


def test_openai_plain_text_output_is_preserved():
    result = _executor_with_final_output("plain answer").invoke({"input": "answer"})

    assert result["output"] == "plain answer"
