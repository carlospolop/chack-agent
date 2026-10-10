import sqlite3
from pathlib import Path
from types import SimpleNamespace

from chack_tools.config import ToolsConfig
from chack_tools.knowledge_store import (
    KnowledgeStore,
    chunk_text,
    decode_knowledge_policy,
    delete_discarded_artifacts,
    documents_from_admin_decisions,
    encode_knowledge_policy,
    get_knowledge_read_tools,
    qualified_nominated_filenames_from_responses,
    reconcile_admin_knowledge_decisions,
    validate_admin_knowledge_decisions,
    normalize_admin_knowledge_decisions,
)
from chack_tools.researcher_administrator_agent import researcher_administrator_output_schema
from chack_tools.researcher_queue_agent import ResearcherQueue, ResearcherQueueAgentTool
from chack_tools.subagent_config import build_subagent_config


def test_knowledge_policy_round_trip_is_queue_safe():
    encoded = encode_knowledge_policy("Investigate durable evidence.", mode="read-write", knowledge_base="Lipedema")
    policy, prompt = decode_knowledge_policy(encoded)

    assert policy == {"mode": "read_write", "knowledge_base": "lipedema"}
    assert prompt == "Investigate durable evidence."


def test_chunk_text_is_bounded_and_overlapping():
    text = "A" * 900 + "\n\n" + "B" * 900
    chunks = chunk_text(text, size=1000, overlap=100)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert chunks[0][-50:] in chunks[1]


def test_xml_extraction_does_not_embed_markup(tmp_path):
    from chack_tools.knowledge_store import extract_document_text

    source = tmp_path / "article.xml"
    source.write_text("<article><title>Study</title><body>Clinical evidence.</body></article>", encoding="utf-8")

    assert extract_document_text(source) == "Study\nClinical evidence."


def test_manifest_groups_are_ordered_and_exclusions_are_recursive(tmp_path):
    (tmp_path / "researches" / "topic" / "live-round5").mkdir(parents=True)
    (tmp_path / "researches" / "topic" / "summary.md").write_text("summary", encoding="utf-8")
    (tmp_path / "researches" / "topic" / "notes.md").write_text("notes", encoding="utf-8")
    (tmp_path / "researches" / "topic" / "live-round5" / "draft.md").write_text("draft", encoding="utf-8")
    manifest = tmp_path / "knowledge.yaml"
    manifest.write_text(
        f"""version: 1
knowledge_base: lipedema
root: {tmp_path}
sources:
  - name: syntheses
    evidence_type: synthesis
    include: [researches/**/summary.md]
  - name: support
    evidence_type: support
    include: [researches/**/*.md]
exclude: [researches/**/live-round5/**]
""",
        encoding="utf-8",
    )
    config = ToolsConfig(
        knowledge_enabled=True,
        knowledge_mode="read_write",
        knowledge_base="lipedema",
        knowledge_allowed_bases=["lipedema"],
        knowledge_allowed_roots={"lipedema": str(tmp_path)},
        knowledge_seed_manifests={"lipedema": str(manifest)},
    )

    docs = KnowledgeStore(config).documents_from_seed_manifest("lipedema")

    assert [(doc.source_path, doc.source_group) for doc in docs] == [
        ("researches/topic/summary.md", "syntheses"),
        ("researches/topic/notes.md", "support"),
    ]


def test_admin_decisions_require_researcher_nomination_and_discard_safely(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    approved = child / "approved.md"
    approved.write_text("durable", encoding="utf-8")
    discarded = child / "discarded.txt"
    discarded.write_text("noise", encoding="utf-8")
    protected = tmp_path / "admin_output.json"
    protected.write_text("{}", encoding="utf-8")
    decisions = [
        {"filename": "child/approved.md", "disposition": "approved", "source_url": "https://example.test/a"},
        {"filename": "child/discarded.txt", "disposition": "discard"},
        {"filename": "admin_output.json", "disposition": "discard"},
    ]
    nominations = {"child/approved.md", "child/discarded.txt", "admin_output.json"}

    documents = documents_from_admin_decisions(
        str(tmp_path), decisions, nominated_filenames=nominations
    )
    cleanup = delete_discarded_artifacts(
        str(tmp_path), decisions, nominated_filenames=nominations
    )

    assert [doc.path for doc in documents] == [approved]
    assert documents[0].source_path.startswith("source-url/")
    assert documents[0].delete_after_ingest is False
    auto_delete_documents = documents_from_admin_decisions(
        str(tmp_path),
        decisions,
        nominated_filenames=nominations,
        delete_verified_text_artifacts=True,
    )
    assert auto_delete_documents[0].delete_after_ingest is True
    assert cleanup["deleted"] == ["child/discarded.txt"]
    assert not discarded.exists()
    assert protected.exists()


def test_admin_decisions_prefer_text_and_use_stable_url_identity(tmp_path):
    child = tmp_path / "scientific"
    child.mkdir()
    text_path = child / "study.txt"
    xml_path = child / "study.xml"
    alternate_path = child / "random-download.txt"
    text_path.write_text("normalized evidence", encoding="utf-8")
    xml_path.write_text("<study>normalized evidence</study>", encoding="utf-8")
    alternate_path.write_text("new extraction", encoding="utf-8")
    url = "https://example.test/study"
    decisions = [
        {"filename": "scientific/study.xml", "disposition": "approved", "source_url": url, "delete_after_ingest": True},
        {"filename": "scientific/study.txt", "disposition": "approved", "source_url": url, "delete_after_ingest": True},
    ]

    normalized = normalize_admin_knowledge_decisions(decisions)

    assert normalized[0]["disposition"] == "archive_only"
    assert normalized[0]["delete_after_ingest"] is False
    assert normalized[1]["disposition"] == "approved"
    documents = documents_from_admin_decisions(
        str(tmp_path),
        normalized,
        nominated_filenames={"scientific/study.xml", "scientific/study.txt"},
    )
    assert len(documents) == 1
    assert documents[0].path == text_path
    assert documents[0].delete_after_ingest is True

    # A later fetch can use a random local filename without creating a second
    # Qdrant source for the same canonical URL.
    later = documents_from_admin_decisions(
        str(tmp_path),
        [{"filename": "scientific/random-download.txt", "disposition": "approved", "source_url": url}],
        nominated_filenames={"scientific/random-download.txt"},
    )
    assert later[0].path == alternate_path
    assert later[0].source_path == documents[0].source_path


def test_markup_is_never_marked_for_post_ingest_deletion(tmp_path):
    markup = tmp_path / "source.xml"
    markup.write_text("<source>evidence</source>", encoding="utf-8")

    documents = documents_from_admin_decisions(
        str(tmp_path),
        [{"filename": "source.xml", "disposition": "approved", "delete_after_ingest": True}],
        nominated_filenames={"source.xml"},
        delete_verified_text_artifacts=True,
    )

    assert documents[0].delete_after_ingest is False


def test_markup_can_use_explicit_extracted_source_deletion_policy(tmp_path):
    markup = tmp_path / "source.xml"
    markup.write_text("<source>evidence</source>", encoding="utf-8")

    documents = documents_from_admin_decisions(
        str(tmp_path),
        [{"filename": "source.xml", "disposition": "approved"}],
        nominated_filenames={"source.xml"},
        delete_verified_extracted_artifacts=True,
    )

    assert documents[0].delete_after_ingest is False
    assert documents[0].delete_extracted_after_ingest is True


def test_admin_decision_validation_qualifies_repeated_researcher_runs(tmp_path):
    run_one = tmp_path / "prochatgpt_researcher" / "run-one"
    run_two = tmp_path / "prochatgpt_researcher" / "run-two"
    responses = [
        {
            "evidence_data_path": str(run_one),
            "knowledge_candidates": [{"filename": "response.md"}],
        },
        {
            "evidence_data_path": str(run_two),
            "knowledge_candidates": [{"filename": "response.md"}],
        },
    ]

    nominations = qualified_nominated_filenames_from_responses(
        responses,
        evidence_dir=str(tmp_path),
    )
    valid = validate_admin_knowledge_decisions(
        [
            {"filename": "prochatgpt_researcher/run-one/response.md"},
            {"filename": "prochatgpt_researcher/run-two/response.md"},
        ],
        nominated_filenames=nominations,
    )
    stale = validate_admin_knowledge_decisions(
        [
            {"filename": "prochatgpt_researcher/run-one/response.md"},
            {"filename": "prochatgpt_researcher/old-run/response.md"},
        ],
        nominated_filenames=nominations,
    )

    assert nominations == {
        "prochatgpt_researcher/run-one/response.md",
        "prochatgpt_researcher/run-two/response.md",
    }
    assert valid == {"missing": [], "unexpected": [], "duplicates": []}
    assert stale["missing"] == ["prochatgpt_researcher/run-two/response.md"]
    assert stale["unexpected"] == ["prochatgpt_researcher/old-run/response.md"]


def test_qualified_nominations_exclude_runtime_artifact_manifests(tmp_path):
    child_root = tmp_path / "websearcher"
    responses = [{
        "evidence_data_path": str(child_root),
        "knowledge_candidates": [
            {"filename": "source.txt"},
            {"filename": "_artifact_manifest.jsonl"},
        ],
    }]

    assert qualified_nominated_filenames_from_responses(
        responses,
        evidence_dir=str(tmp_path),
    ) == {"websearcher/source.txt"}


def test_admin_decision_reconciliation_canonicalizes_only_unambiguous_aliases():
    candidates = {
        "scientific/pmc/PMC123_deadbeef.txt": {
            "title": "Study",
            "source_url": "https://example.test/study",
            "evidence_type": "primary",
        },
        "scientific/other.txt": {
            "title": "Other",
            "source_url": "https://example.test/other",
            "evidence_type": "primary",
        },
    }
    decisions = [
        {
            "filename": "scientific/pmc/PMC123.txt",
            "disposition": "approved",
            "reason": "The source is durable, primary, and useful for future retrieval.",
            "title": "Study",
            "source_url": "https://example.test/study/",
            "evidence_type": "primary",
            "delete_after_ingest": False,
        },
        {
            "filename": "scientific/invented.txt",
            "disposition": "approved",
            "reason": "This row does not identify a runtime-owned candidate at all.",
            "title": "Invented",
            "source_url": "https://example.test/invented",
            "evidence_type": "primary",
            "delete_after_ingest": True,
        },
    ]

    reconciled, audit = reconcile_admin_knowledge_decisions(decisions, candidates=candidates)

    assert validate_admin_knowledge_decisions(
        reconciled,
        nominated_filenames=candidates,
    ) == {"missing": [], "unexpected": [], "duplicates": []}
    assert reconciled[0]["filename"] == "scientific/pmc/PMC123_deadbeef.txt"
    assert reconciled[0]["disposition"] == "approved"
    assert reconciled[1]["filename"] == "scientific/other.txt"
    assert reconciled[1]["disposition"] == "archive_only"
    assert audit["canonicalized_aliases"] == [{
        "from": "scientific/pmc/PMC123.txt",
        "to": "scientific/pmc/PMC123_deadbeef.txt",
    }]
    assert audit["ignored_unexpected"] == ["scientific/invented.txt"]
    assert audit["archive_only_missing"] == ["scientific/other.txt"]


def test_admin_decision_reconciliation_archives_conflicting_duplicates():
    candidate = {
        "title": "Evidence",
        "source_url": "https://example.test/evidence",
        "evidence_type": "primary",
    }
    decisions = [
        {
            "filename": "web/evidence.txt",
            "disposition": disposition,
            "reason": "A sufficiently detailed administrator reason for this decision.",
            "title": "Evidence",
            "source_url": "https://example.test/evidence",
            "evidence_type": "primary",
            "delete_after_ingest": disposition == "approved",
        }
        for disposition in ("approved", "discard")
    ]

    reconciled, audit = reconcile_admin_knowledge_decisions(
        decisions,
        candidates={"web/evidence.txt": candidate},
    )

    assert len(reconciled) == 1
    assert reconciled[0]["disposition"] == "archive_only"
    assert reconciled[0]["delete_after_ingest"] is False
    assert audit == {"archive_only_duplicates": ["web/evidence.txt"]}


def test_write_capable_runs_expand_agent_schemas_only_when_enabled():
    config = ToolsConfig(
        knowledge_enabled=True,
        knowledge_mode="read_write",
        knowledge_base="lipedema",
    )
    child = build_subagent_config(
        config,
        model_name="gpt-test",
        model_provider="openai",
        max_turns=3,
        system_prompt="### SPECIFIC\nResearch the supplied question.",
        overrides={"env": {"CHACK_RESEARCH_SAVE_ARTIFACTS": "1"}},
    )

    assert "knowledge_candidates" in child.agent.output_schema_json["required"]
    assert child.agent.output_schema_json["properties"]["knowledge_candidates"]["maxItems"] == 200
    assert "every retained" in child.system_prompt
    assert "knowledge_candidates" in child.system_prompt
    assert "`knowledge_search`" in child.system_prompt
    assert "`lipedema`" in child.system_prompt
    assert "knowledge_decisions" not in researcher_administrator_output_schema(
        preserve_artifacts=True
    )["required"]
    assert "knowledge_decisions" in researcher_administrator_output_schema(
        preserve_artifacts=True,
        include_knowledge=True,
    )["required"]


def test_missing_retained_artifact_classifications_default_to_archive_only():
    from chack_tools.subagent_config import normalize_researcher_response_payload

    payload = normalize_researcher_response_payload({
        "research_worked": True,
        "failure_reason": "",
        "overall_summary": "A sufficiently useful research summary.",
        "findings": [],
        "gaps": [],
        "open_topics": [],
        "full_research_review": "Evidence review",
        "key_artifacts": [
            {"filename": "selected.txt", "source_url": "https://example.test/selected", "description": "x" * 100},
            {"filename": "omitted.txt", "source_url": "https://example.test/omitted", "description": "y" * 100},
        ],
        "knowledge_candidates": [{
            "filename": "selected.txt",
            "disposition": "ingest_candidate",
            "reason": "This is durable primary evidence suitable for later retrieval.",
            "title": "Selected",
            "source_url": "https://example.test/selected",
            "evidence_type": "primary",
        }],
    })

    assert [(row["filename"], row["disposition"]) for row in payload["knowledge_candidates"]] == [
        ("selected.txt", "ingest_candidate"),
        ("omitted.txt", "archive_only"),
    ]

    read_only = build_subagent_config(
        ToolsConfig(
            knowledge_enabled=True,
            knowledge_mode="read",
            knowledge_base="lipedema",
        ),
        model_name="gpt-test",
        model_provider="openai",
        max_turns=3,
        system_prompt="### SPECIFIC\nResearch the supplied question.",
        overrides={"env": {"CHACK_RESEARCH_SAVE_ARTIFACTS": "1"}},
    )
    assert "`knowledge_search`" in read_only.system_prompt
    assert "knowledge_candidates" not in read_only.agent.output_schema_json["required"]


def test_request_bound_knowledge_tools_do_not_let_agents_choose_a_corpus():
    bound = get_knowledge_read_tools(
        ToolsConfig(
            knowledge_enabled=True,
            knowledge_mode="read_write",
            knowledge_base="lipedema",
            knowledge_allowed_bases=["lipedema", "another-program"],
            knowledge_bind_configured_base=True,
        )
    )
    schemas = {tool.name: tool.params_json_schema for tool in bound}

    assert "knowledge_base" not in schemas["knowledge_search"]["properties"]
    assert "knowledge_base" not in schemas["knowledge_status"]["properties"]

    unbound = get_knowledge_read_tools(
        ToolsConfig(
            knowledge_enabled=True,
            knowledge_mode="read",
            knowledge_base="lipedema",
            knowledge_allowed_bases=["lipedema", "another-program"],
        )
    )
    unbound_schema = {tool.name: tool.params_json_schema for tool in unbound}
    assert "knowledge_base" in unbound_schema["knowledge_search"]["properties"]


def test_queue_never_merges_requests_with_different_knowledge_policies(monkeypatch):
    helper = ResearcherQueueAgentTool(
        object(),
        config=ToolsConfig(),
        model_provider="openai",
        fallback_model="gpt-test",
        queue=ResearcherQueue(),
    )
    monkeypatch.setattr(helper, "_merge_prompts", lambda prompts: [("\n".join(prompts), list(range(len(prompts))), "merged")])
    prompts = [
        encode_knowledge_policy("persistent", mode="read_write", knowledge_base="lipedema"),
        encode_knowledge_policy("disposable", mode="off", knowledge_base=""),
    ]

    groups = helper._merge_prompts_by_knowledge_policy(prompts)

    assert len(groups) == 2
    assert sorted(decode_knowledge_policy(group[0])[0]["mode"] for group in groups) == ["off", "read_write"]


def test_search_enforces_operator_limits_and_never_fetches_full_source_payload(monkeypatch):
    config = ToolsConfig(
        knowledge_enabled=True,
        knowledge_mode="read",
        knowledge_base="lipedema",
        knowledge_allowed_bases=["lipedema"],
        knowledge_vector_results=2,
        knowledge_exact_results=1,
        knowledge_max_return_chars=1000,
    )
    store = KnowledgeStore(config)
    calls = []

    class Vector:
        indices = SimpleNamespace(tolist=lambda: [1])
        values = SimpleNamespace(tolist=lambda: [1.0])

        def tolist(self):
            return [0.1, 0.2]

    class Model:
        def query_embed(self, _query):
            yield Vector()

    class Client:
        def query_points(self, _collection, **kwargs):
            calls.append(kwargs)
            points = [
                SimpleNamespace(
                    id=f"{kwargs['using']}-{index}",
                    payload={
                        "source_path": f"doc-{index}.md",
                        "title": "Test",
                        "text": "x" * 700,
                        "source_text": "must never be fetched",
                    },
                )
                for index in range(kwargs["limit"])
            ]
            return SimpleNamespace(points=points)

    monkeypatch.setattr(store, "_ensure_collection", lambda: None)
    monkeypatch.setattr(store, "_models", lambda: (Model(), Model()))
    monkeypatch.setattr(store, "_qdrant", lambda: Client())

    result = store.search(
        "test query",
        "lipedema",
        vector_limit=30,
        exact_limit=30,
        max_chars=100_000,
    )

    assert result["vector_requested"] == 2
    assert result["exact_requested"] == 1
    assert result["max_returned_text_chars"] == 1000
    assert sum(len(row["text"]) for row in result["results"]) <= 1000
    assert [call["limit"] for call in calls] == [2, 1]
    assert all("source_text" not in call["with_payload"] for call in calls)


def test_status_migrates_and_reports_embedding_runtime_compatibility(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(catalog_path) as catalog:
        catalog.execute(
            """CREATE TABLE sources (
                knowledge_base TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                chunk_size INTEGER NOT NULL,
                chunk_overlap INTEGER NOT NULL,
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_deleted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (knowledge_base, source_id)
            )"""
        )
        catalog.execute(
            """INSERT INTO sources (
                knowledge_base,source_id,source_path,source_url,title,evidence_type,
                content_hash,byte_size,chunk_count,embedding_model,chunk_size,chunk_overlap
            ) VALUES ('lipedema','id','doc.md','','Doc','primary','hash',10,1,?,1800,250)""",
            ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",),
        )
    store = KnowledgeStore(ToolsConfig(
        knowledge_enabled=True,
        knowledge_mode="read",
        knowledge_base="lipedema",
        knowledge_allowed_bases=["lipedema"],
        knowledge_catalog_path=str(catalog_path),
    ))
    monkeypatch.setattr(store, "_ensure_collection", lambda: None)

    status = store.status("lipedema")

    assert status["embedding_runtime"].startswith("fastembed=")
    assert status["embedding_compatible_sources"] == 0
    assert status["embedding_incompatible_sources"] == 1
    with sqlite3.connect(catalog_path) as catalog:
        assert "embedding_runtime" in {
            row[1] for row in catalog.execute("PRAGMA table_info(sources)")
        }


def test_ingest_persists_embedding_runtime_signature(tmp_path, monkeypatch):
    from chack_tools.knowledge_store import KnowledgeDocument

    document_path = tmp_path / "durable.xml"
    document_path.write_text("<source>Durable sourced evidence about lipedema.</source>", encoding="utf-8")
    catalog_path = tmp_path / "catalog.sqlite3"
    store = KnowledgeStore(ToolsConfig(
        knowledge_enabled=True,
        knowledge_mode="read_write",
        knowledge_base="lipedema",
        knowledge_allowed_bases=["lipedema"],
        knowledge_catalog_path=str(catalog_path),
        knowledge_chunk_size_chars=300,
        knowledge_chunk_overlap_chars=0,
    ))

    class DenseVector:
        def tolist(self):
            return [0.1, 0.2]

    class SparseVector:
        indices = SimpleNamespace(tolist=lambda: [1])
        values = SimpleNamespace(tolist=lambda: [1.0])

    class DenseModel:
        def embed(self, chunks):
            return (DenseVector() for _ in chunks)

    class SparseModel:
        def embed(self, chunks):
            return (SparseVector() for _ in chunks)

    class Client:
        def __init__(self):
            self.points = []

        def upsert(self, _collection, points, wait):
            assert wait is True
            self.points.extend(points)

    client = Client()
    monkeypatch.setattr(store, "_ensure_collection", lambda: None)
    monkeypatch.setattr(store, "_models", lambda: (DenseModel(), SparseModel()))
    monkeypatch.setattr(store, "_qdrant", lambda: client)
    monkeypatch.setattr(store, "_verify_source_payload", lambda *_args: True)

    result = store.ingest_documents(
        "lipedema",
        [KnowledgeDocument(
            path=document_path,
            source_path="researches/topic/durable.xml",
            delete_extracted_after_ingest=True,
        )],
    )

    assert result["indexed"][0]["chunks"] == 1
    assert result["deleted"] == [str(document_path)]
    assert not document_path.exists()
    assert len(client.points) == 1
    with sqlite3.connect(catalog_path) as catalog:
        runtime, source_deleted = catalog.execute(
            "SELECT embedding_runtime, source_deleted FROM sources"
        ).fetchone()
    assert runtime.startswith("fastembed=")
    assert source_deleted == 1
