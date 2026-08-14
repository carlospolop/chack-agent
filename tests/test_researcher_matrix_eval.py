from scripts.researcher_matrix_eval import parse_output


def test_parse_output_extracts_fenced_research_result_after_narrative() -> None:
    raw = """I finished the investigation.\n\n```json
{"research_worked": true, "findings": [{"claim": "A claim", "summary": "A substantive summary"}], "full_research_review": "review"}
```"""

    parsed = parse_output(raw)

    assert parsed["research_worked"] is True
    assert parsed["findings"][0]["claim"] == "A claim"
    assert parsed["full_research_review"] == "review"


def test_parse_output_prefers_research_result_over_unrelated_embedded_json() -> None:
    raw = (
        'Tool metadata: {"status": "success"}\n'
        'Final result: {"research_worked": true, "findings": [], '
        '"full_research_review": "complete review"}'
    )

    parsed = parse_output(raw)

    assert parsed["research_worked"] is True
    assert parsed["full_research_review"] == "complete review"
