from __future__ import annotations

import json
from pathlib import Path

from scripts.researcher_administrator_eval import validate_case


def test_validate_case_prefers_successful_persisted_replacement_over_cancelled_attempt(tmp_path: Path):
    output_dir = tmp_path / "researcher_outputs"
    job_dir = tmp_path / "researcher_jobs"
    source_dir = tmp_path / "scientific"
    output_dir.mkdir()
    job_dir.mkdir()
    source_dir.mkdir()

    successful = {
        "research_worked": True,
        "failure_reason": "",
        "researcher_tool": "scientific_research",
        "findings": [
            {
                "claim": "The replacement collected primary scientific evidence.",
                "summary": "The completed worker returned a substantive review after the lifecycle probe was cancelled.",
            }
        ],
        "full_research_review": "A complete primary-source review with methods and caveats. " * 10,
    }
    cancelled = {
        "research_worked": False,
        "failure_reason": "cancelled; worker unwound",
        "researcher_tool": "scientific_research",
        "findings": [],
        "full_research_review": "",
        "status": "cancelled",
    }
    (output_dir / "001_scientific_research.json").write_text(json.dumps(successful), encoding="utf-8")
    # Sorts after the successful record, reproducing the real acceptance layout.
    (output_dir / "async_task-z_cancelled_scientific_research.json").write_text(
        json.dumps(cancelled), encoding="utf-8"
    )
    (output_dir / "async_task-a_success_scientific_research.raw.txt").write_text(
        json.dumps(successful), encoding="utf-8"
    )
    (source_dir / "primary-source.txt").write_text("inspectable evidence", encoding="utf-8")
    (job_dir / "cancelled.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task-cancelled",
                        "status": "cancelled",
                        "execution_active": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "replacement.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task-replacement",
                        "status": "done",
                        "execution_active": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    parsed = {
        "research_worked": True,
        "administrator_conclusions": (
            "The successful replacement supplied enough primary evidence for a supported synthesis."
        ),
        "researcher_responses": [successful],
        "researcher_call_counts": {"scientific_research": 2},
        "researcher_tool_call_counts": {"crossref_search": 3},
    }
    summary = {
        "enabled_researchers": ["scientific"],
        "save_artifacts": True,
        "evidence_data_path": str(tmp_path),
        "artifact_stats": {"file_count": 7},
    }

    result = validate_case(summary, parsed, json.dumps(parsed))

    assert result["acceptance_pass"] is True
    assert result["validation_failures"] == []
