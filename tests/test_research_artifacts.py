from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from chack_tools.config import ToolsConfig
from chack_tools import subagent_config as sc
from chack_tools.research_artifacts import (
    ARTIFACT_MANIFEST_FILENAME,
    ResearchArtifactsTool,
    cleanup_research_artifacts,
    record_research_artifact,
    reset_research_artifact_context,
    set_research_artifact_context,
)


def test_research_artifact_tools_list_read_and_grep(monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    nested = root / "pages"
    nested.mkdir(parents=True)
    artifact = nested / "source.txt"
    artifact.write_text("alpha\nneedle line\nomega\n", encoding="utf-8")
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(root))
    record_research_artifact(
        artifact,
        source_url="https://example.com/source.txt",
        provenance="test",
        tool="test_tool",
        kind="test",
        label="source",
    )

    tool = ResearchArtifactsTool(ToolsConfig())

    listed = tool.list_files()
    assert "pages/source.txt" in listed

    ranged = tool.read_file("pages/source.txt", start_line=2, end_line=2)
    assert "2: needle line" in ranged
    assert "1: alpha" not in ranged

    around = tool.read_file(str(artifact), around_text="needle", context_lines=1)
    assert "1: alpha" in around
    assert "2: needle line" in around
    assert "3: omega" in around

    grep = tool.grep("needle", glob="*.txt")
    assert "pages/source.txt:2: needle line" in grep

    registered = tool.register_file(
        "pages/source.txt",
        source_url="https://example.com/source.txt",
        provenance="exec curl https://example.com/source.txt",
        tool="exec",
        kind="raw-text",
        label="source text",
    )
    assert "Registered research artifact pages/source.txt" in registered

    deleted = tool.delete_file("pages/source.txt")
    assert "Deleted research artifact pages/source.txt" in deleted
    assert not artifact.exists()
    assert "pages/source.txt" not in (root / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8")


def test_record_research_artifact_deduplicates_identical_manifest_rows(monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    artifact = root / "source.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("alpha", encoding="utf-8")
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(root))

    for _ in range(2):
        record_research_artifact(
            artifact,
            source_url="https://example.com/source.txt",
            provenance="same provenance",
            tool="fetch_url_text",
            kind="web-pages",
            label="source",
        )

    lines = (root / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_concurrent_artifact_manifest_writes_remain_valid(monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(parents=True)
    artifacts = []
    for index in range(16):
        artifact = root / f"source-{index}.txt"
        artifact.write_text(f"source {index}", encoding="utf-8")
        artifacts.append(artifact)
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(root))

    def register(artifact):
        record_research_artifact(
            artifact,
            source_url=f"https://example.com/source-{artifact.stem}.txt",
            provenance="concurrent test",
            tool="fetch_url_text",
            kind="web",
            label=artifact.name,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(register, artifacts))

    lines = (root / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(artifacts)
    rows = [json.loads(line) for line in lines]
    assert {row["filename"] for row in rows} == {artifact.name for artifact in artifacts}


def test_research_artifact_reader_rejects_paths_outside_root(monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(root))

    tool = ResearchArtifactsTool(ToolsConfig())

    with pytest.raises(ValueError, match="inside CHACK_RESEARCH_DATA_DIR"):
        tool.read_file(str(outside))

    with pytest.raises(ValueError, match="inside CHACK_RESEARCH_DATA_DIR"):
        tool.delete_file(str(outside))


def test_cleanup_research_artifacts_deletes_only_when_not_preserved(tmp_path):
    root = Path("/tmp/chack-research-data/test-session/test-kind")
    root.mkdir(parents=True, exist_ok=True)
    (root / "artifact.txt").write_text("data", encoding="utf-8")

    cleanup_research_artifacts(str(root), save_artifacts=True)
    assert root.exists()

    cleanup_research_artifacts(str(root), save_artifacts=False)
    assert not root.exists()


def test_artifact_reconciliation_prompts_for_unlisted_preserved_files(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    listed = root / "listed.txt"
    missing = root / "missing.txt"
    listed.write_text("listed", encoding="utf-8")
    missing.write_text("missing", encoding="utf-8")
    output = (
        '{"research_worked":true,"failure_reason":"","final_research_review":"ok",'
        f'"evidence_data_path":"{root}",'
        '"key_artifacts":[{"filename":"listed.txt","source_url":"local",'
        '"description":"This listed artifact contains evidence already reviewed and cited in the research, so it is kept as supporting source material for the final conclusion."}]}'
    )

    prompt = sc.artifact_reconciliation_prompt(output=output, evidence_dir=str(root))

    assert "Artifact reconciliation required" in prompt
    assert "missing.txt" in prompt
    assert "delete_research_artifact" in prompt
    assert "listed.txt" not in prompt
    assert "`filename` relative to the evidence directory" in prompt
    assert "Do not include full file paths" in prompt
    assert "100-300 character description" in prompt
    assert "Do not repeat the full research review" in prompt
    assert "key_artifacts` for the missing/replacement artifact records" in prompt


def test_researcher_artifact_schema_requires_descriptive_artifact_summaries():
    schema = sc.researcher_output_schema(preserve_artifacts=True)
    description_schema = schema["properties"]["key_artifacts"]["items"]["properties"]["description"]

    assert description_schema["minLength"] == 100
    assert description_schema["maxLength"] == 300
    assert ToolsConfig().research_strict_artifact_manifest is True


def test_artifact_manifest_prefills_source_and_is_not_evidence(monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    artifact = root / "page.txt"
    artifact.write_text("relevant page text", encoding="utf-8")
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(root))

    record_research_artifact(
        artifact,
        source_url="https://example.com/source",
        provenance="fetch_url_text",
        tool="fetch_url_text",
        kind="web",
        label="example",
    )
    assert (root / ARTIFACT_MANIFEST_FILENAME).is_file()

    output = (
        '{"research_worked":true,"failure_reason":"","final_research_review":"ok",'
        f'"evidence_data_path":"{root}",'
        '"key_artifacts":[{"filename":"page.txt","source_url":"",'
        '"description":"This saved page text was used as direct evidence for the reviewed claim, grounding the conclusion in fetched source content rather than search snippets."}]}'
    )

    revised = sc.reconcile_research_artifacts(
        output,
        evidence_dir=str(root),
        save_artifacts=True,
        run_followup=lambda _prompt: "{}",
    )
    payload = json.loads(revised)

    assert payload["key_artifacts"][0]["source_url"] == "https://example.com/source"
    assert sc.artifact_reconciliation_prompt(output=revised, evidence_dir=str(root)) == ""


def test_fetch_url_text_records_source_url_and_tool(monkeypatch, tmp_path):
    from chack_tools.open_research_sources import OpenResearchTool

    root = tmp_path / "evidence"
    root.mkdir()
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(root))

    class FakeResponse:
        url = "https://example.com/final"
        headers = {"content-type": "text/html"}
        text = "<html><body><h1>Evidence</h1><p>" + ("useful source text " * 20) + "</p></body></html>"

        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, timeout=None, allow_redirects=None, stream=None):
        return FakeResponse()

    monkeypatch.setattr("chack_tools.open_research_sources.requests.get", fake_get)
    monkeypatch.setattr("chack_tools.open_research_sources._is_public_https_url", lambda url: url.startswith("https://example.com/"))

    result = OpenResearchTool(ToolsConfig()).fetch_url_text("https://example.com/start")
    assert result.startswith("SUCCESS:")

    rows = [
        json.loads(line)
        for line in (root / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert {row["source_url"] for row in rows} == {"https://example.com/final"}
    assert {row["tool"] for row in rows} == {"fetch_url_text"}


def test_fetch_url_text_rejects_private_targets_and_redirects(monkeypatch):
    from chack_tools.open_research_sources import OpenResearchTool

    called = []
    monkeypatch.setattr(
        "chack_tools.open_research_sources._is_public_https_url",
        lambda url: url == "https://public.example/start",
    )

    class RedirectResponse:
        status_code = 302
        headers = {"location": "http://169.254.169.254/latest/meta-data"}
        url = "https://public.example/start"

    monkeypatch.setattr(
        "chack_tools.open_research_sources.requests.get",
        lambda *args, **kwargs: called.append((args, kwargs)) or RedirectResponse(),
    )
    helper = OpenResearchTool(ToolsConfig())
    assert helper.fetch_url_text("http://127.0.0.1/") == "ERROR: url must be a public HTTPS address"
    assert called == []
    assert helper.fetch_url_text("https://public.example/start") == "ERROR: URL redirect was not a public HTTPS address"
    assert len(called) == 1 and called[0][1]["allow_redirects"] is False


def test_fetch_url_text_rejects_oversized_responses_before_reading(monkeypatch):
    from chack_tools.open_research_sources import OpenResearchTool, _MAX_FETCH_TEXT_BYTES

    class LargeResponse:
        status_code = 200
        headers = {"content-type": "text/plain", "content-length": str(_MAX_FETCH_TEXT_BYTES + 1)}
        url = "https://public.example/page"

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            raise AssertionError("oversized response body must not be read")

    monkeypatch.setattr("chack_tools.open_research_sources._is_public_https_url", lambda _url: True)
    monkeypatch.setattr("chack_tools.open_research_sources.requests.get", lambda *args, **kwargs: LargeResponse())
    result = OpenResearchTool(ToolsConfig()).fetch_url_text("https://public.example/page")
    assert result == "ERROR: fetched URL exceeded the response-size limit"


def test_artifact_writers_prefer_context_root_over_process_env(monkeypatch, tmp_path):
    from chack_tools.product_search import _write_json_artifact

    env_root = tmp_path / "env-root"
    context_root = tmp_path / "context-root"
    env_root.mkdir()
    context_root.mkdir()
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(env_root))

    tokens = set_research_artifact_context(str(context_root), "")
    try:
        artifact_path = Path(_write_json_artifact("open-food-facts", "context test", {"url": "https://example.com"}))
    finally:
        reset_research_artifact_context(tokens)

    assert artifact_path.is_file()
    assert artifact_path.is_relative_to(context_root)
    assert not any(env_root.rglob("*.json"))
    assert (context_root / ARTIFACT_MANIFEST_FILENAME).is_file()


def test_reconcile_skips_temporary_unpreserved_folders(monkeypatch, tmp_path):
    monkeypatch.delenv("CHACK_RESEARCH_MASTER_DIR", raising=False)
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "missing.txt").write_text("missing", encoding="utf-8")
    output = (
        '{"research_worked":true,"failure_reason":"","final_research_review":"ok",'
        '"evidence_data_path":"","key_artifacts":[]}'
    )
    called = False

    def followup(_prompt):
        nonlocal called
        called = True
        return "{}"

    assert sc.reconcile_research_artifacts(
        output,
        evidence_dir=str(root),
        save_artifacts=False,
        run_followup=followup,
    ) == output
    assert called is False


def test_reconcile_uses_artifact_patch_followup_for_preserved_missing_files(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    missing = root / "missing.txt"
    missing.write_text("missing", encoding="utf-8")
    output = (
        '{"research_worked":true,"failure_reason":"","final_research_review":"ok",'
        f'"evidence_data_path":"{root}","key_artifacts":[]}}'
    )
    prompts = []
    schemas = []

    def followup(prompt, output_schema_json=None):
        prompts.append(prompt)
        schemas.append(output_schema_json)
        return (
            '{"key_artifacts":[{"filename":"missing.txt","source_url":"local",'
            '"description":"This missing artifact contains the source evidence found during reconciliation and was used to support the existing final research review."}],'
            '"delete_artifacts":[]}'
        )

    revised = sc.reconcile_research_artifacts(
        output,
        evidence_dir=str(root),
        save_artifacts=True,
        run_followup=followup,
    )

    assert prompts and "missing.txt" in prompts[0]
    assert "final_research_review" in revised
    assert "ok revised" not in revised
    assert schemas and schemas[0]["required"] == ["key_artifacts", "delete_artifacts"]
    payload = json.loads(revised)
    assert payload["final_research_review"] == "ok"
    assert payload["key_artifacts"][0]["filename"] == "missing.txt"


def test_reconcile_patch_can_delete_useless_files(monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(root))
    useless = root / "useless.txt"
    useful = root / "useful.txt"
    useless.write_text("not used", encoding="utf-8")
    useful.write_text("used", encoding="utf-8")
    record_research_artifact(useless, source_url="local", provenance="test", tool="test", kind="test", label="useless")
    record_research_artifact(useful, source_url="local", provenance="test", tool="test", kind="test", label="useful")
    output = (
        '{"research_worked":true,"failure_reason":"","final_research_review":"ok",'
        f'"evidence_data_path":"{root}","key_artifacts":[]}}'
    )

    def followup(_prompt, output_schema_json=None):
        return (
            '{"key_artifacts":[{"filename":"useful.txt","source_url":"local",'
            '"description":"This useful artifact contains source evidence that was retained because it directly supports the final research conclusion and audit trail."}],'
            '"delete_artifacts":["useless.txt"]}'
        )

    revised = sc.reconcile_research_artifacts(
        output,
        evidence_dir=str(root),
        save_artifacts=True,
        run_followup=followup,
    )
    payload = json.loads(revised)

    assert not useless.exists()
    assert useful.exists()
    assert [item["filename"] for item in payload["key_artifacts"]] == ["useful.txt"]
    manifest = (root / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert "useless.txt" not in manifest
    assert "useful.txt" in manifest


def test_researcher_tool_usage_is_appended_in_code():
    output = (
        '{"research_worked":true,"failure_reason":"",'
        '"final_research_review":"review"}'
    )

    enriched = sc.append_research_tool_usage(
        output,
        Counter({"search_google_web": 2, "fetch_url_text": 1}),
    )
    payload = json.loads(enriched)

    assert payload["full_research_review"] == "review"
    assert payload["overall_summary"] == "review"
    assert payload["findings"]
    assert payload["open_topics"] == []
    assert "final_research_review" not in payload
    assert payload["tool_call_counts"] == {
        "fetch_url_text": 1,
        "search_google_web": 2,
    }
    assert payload["total_tool_calls"] == 3


def test_researcher_response_collector_parses_batched_results():
    batched = (
        'SUBAGENT_RESULT_1:\n{"research_worked":true,"failure_reason":"",'
        '"final_research_review":"one","tool_call_counts":{"a":1}}\n\n'
        'SUBAGENT_RESULT_2:\n{"research_worked":true,"failure_reason":"",'
        '"final_research_review":"two","tool_call_counts":{"b":2}}'
    )

    responses = sc.researcher_responses_from_output("websearcher_research", batched)

    assert [item["full_research_review"] for item in responses] == ["one", "two"]
    assert all(item["findings"] for item in responses)
    assert responses[0]["researcher_tool"] == "websearcher_research"
    assert responses[0]["batch_result_index"] == 1
    assert sc.aggregate_tool_call_counts(responses) == {"a": 1, "b": 2}


def test_open_topics_are_optional_bounded_followups_without_truncating_full_review():
    full_review = "Complete raw evidence " * 1000
    gap = "Primary evidence is unavailable for the requested historical period."
    payload = sc.normalize_researcher_response_payload(
        {
            "research_worked": True,
            "failure_reason": "",
            "overall_summary": "Summary",
            "findings": [],
            "gaps": [gap],
            "open_topics": [
                gap,
                "short",
                *(f"Follow-up topic {index}: " + "x" * 400 for index in range(6)),
            ],
            "full_research_review": full_review,
        }
    )

    assert len(payload["open_topics"]) == 5
    assert all(30 <= len(topic) <= 250 for topic in payload["open_topics"])
    assert gap not in payload["open_topics"]
    assert payload["full_research_review"] == full_review
    digest = sc.compact_researcher_digest(payload)
    assert digest["open_topics"] == payload["open_topics"]
    assert "full_research_review" not in digest
