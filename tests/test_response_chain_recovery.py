from chack_agent.backends.openai_compaction_backend import _is_sequence_recoverable_error
from chack_agent.backends.gateway_responses_backend import GatewayResponsesModel


def test_openai_compaction_backend_recovers_missing_previous_response_id():
    exc = Exception(
        "Error code: 400 - {'error': {'message': \"Previous response with id 'resp_123' not found.\"}}"
    )

    assert _is_sequence_recoverable_error(exc) is True


def test_openrouter_backend_recovers_missing_previous_response_id():
    exc = Exception(
        "Error code: 400 - {'error': {'message': \"Previous response with id 'resp_123' not found.\"}}"
    )

    assert GatewayResponsesModel._is_sequence_recoverable_error(exc) is True
