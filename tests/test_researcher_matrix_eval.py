from argparse import Namespace

from scripts.researcher_matrix_eval import parse_output, resume_summary_matches


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


def test_resume_requires_same_provider_model_and_reasoning_effort() -> None:
    args = Namespace(provider="codex", model="gpt-5.4-mini", thinking_effort="high")
    matching = {
        "functional_pass": True,
        "provider": "codex",
        "model": "gpt-5.4-mini",
        "thinking_effort": "high",
    }

    assert resume_summary_matches(matching, args) is True
    assert resume_summary_matches({**matching, "provider": "claude"}, args) is False
    assert resume_summary_matches({**matching, "model": "gpt-5.5"}, args) is False
    assert resume_summary_matches({**matching, "thinking_effort": "max"}, args) is False
    assert resume_summary_matches({**matching, "functional_pass": False}, args) is False
