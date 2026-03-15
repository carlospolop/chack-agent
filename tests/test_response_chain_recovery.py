from chack_agent.backends.openai_compaction_backend import _is_sequence_recoverable_error
from chack_agent.backends.openrouter_openai_backend import OpenRouterResponsesModel


def test_openai_compaction_backend_recovers_missing_previous_response_id():
    exc = Exception(
        "Error code: 400 - {'error': {'message': \"Previous response with id 'resp_123' not found.\"}}"
    )

    assert _is_sequence_recoverable_error(exc) is True


def test_openrouter_backend_recovers_missing_previous_response_id():
    exc = Exception(
        "Error code: 400 - {'error': {'message': \"Previous response with id 'resp_123' not found.\"}}"
    )

    assert OpenRouterResponsesModel._is_sequence_recoverable_error(exc) is True
