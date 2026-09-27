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


def test_server_compaction_does_not_repeat_token_compaction_after_run():
    executor = AgentsExecutor.__new__(AgentsExecutor)
    executor.agent = SimpleNamespace(
        model_settings=SimpleNamespace(context_management=[{"type": "compaction", "compact_threshold": 262_500}])
    )
    executor._previous_response_id = "response-id"
    executor._max_context_tokens = 350_000
    executor._compaction_threshold_ratio = 0.75
    executor._memory_limit = 250
    executor._conversation = []
    executor._run_compaction = lambda *_args: (_ for _ in ()).throw(AssertionError("redundant compaction"))

    executor._maybe_compact(300_000)
