from chack_tools.telemetry.sqs_logger import _compact_tool_event_payload


def test_large_tool_result_is_compacted_without_mutating_runtime_payload(monkeypatch):
    monkeypatch.setenv("CHACK_TELEMETRY_TOOL_RESULT_MAX_BYTES", "512")
    payload = {
        "tool": "read_snippets",
        "tool_input": {
            "arguments": {"requests": [{"file_path": "src/example.py"}]},
            "status": "completed",
            "result": {"content": "x" * 5000},
        },
    }

    compact = _compact_tool_event_payload("tool_called", payload)

    assert payload["tool_input"]["result"] == {"content": "x" * 5000}
    assert compact["tool_input"]["arguments"] == payload["tool_input"]["arguments"]
    result = compact["tool_input"]["result"]
    assert result["_telemetry_truncated"] is True
    assert result["original_bytes"] > 5000
    assert len(result["preview"]) <= 512


def test_small_tool_result_remains_available_for_recovery(monkeypatch):
    monkeypatch.setenv("CHACK_TELEMETRY_TOOL_RESULT_MAX_BYTES", "8192")
    payload = {
        "tool": "save_discovered_vulnerability",
        "tool_input": {
            "arguments": {"name": "Example finding"},
            "status": "completed",
            "result": {"message": "Successfully saved vulnerability"},
        },
    }

    assert _compact_tool_event_payload("tool_called", payload) is payload


def test_non_tool_event_is_unchanged():
    payload = {"result": "x" * 10000}

    assert _compact_tool_event_payload("agent_end", payload) is payload
