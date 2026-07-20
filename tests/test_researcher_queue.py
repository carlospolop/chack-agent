import json
import threading
from pathlib import Path
from types import SimpleNamespace

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.researcher_queue_agent import (
    ResearcherQueue,
    ResearcherQueueAgentTool,
    _QueueWaiter,
    get_researcher_queue_tool,
)


def _tool_names(tools):
    return {
        str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")
        for tool in tools
    }


def _make_helper(**kw):
    # A dummy administrator is fine: every test that reaches research stubs
    # _run_admin / _merge_prompts, so the real administrator is never invoked.
    return ResearcherQueueAgentTool(
        object(),
        config=ToolsConfig(),
        model_provider="openai",
        fallback_model="m",
        queue=ResearcherQueue(),
        **kw,
    )


# ── registration ──────────────────────────────────────────────────────────────
def test_queue_registered_only_when_enabled():
    off = AgentsToolset(ToolsConfig(scientific_enabled=True), model_provider="openai", default_model="m")
    assert "researcher_queue" not in _tool_names(off.tools)

    on = AgentsToolset(
        ToolsConfig(researcher_queue_enabled=True, scientific_enabled=True),
        model_provider="openai",
        default_model="m",
    )
    assert "researcher_queue" in _tool_names(on.tools)
    assert "researcher_queue_status" in _tool_names(on.tools)


def test_queue_default_wait_is_90_minutes():
    assert ToolsConfig().researcher_queue_max_wait_seconds == 5400
    assert ToolsConfig().researcher_queue_max_cost_usd == 5.0


def test_queue_create_returns_reusable_queue_folder():
    q = ResearcherQueue()

    created = json.loads(q.create_queue("shared research queue"))
    reused = json.loads(q.create_queue(created["queue_id"]))

    assert created["queue_id"] == "shared_research_queue"
    assert reused["queue_id"] == created["queue_id"]
    assert reused["queue_evidence_data_path"] == created["queue_evidence_data_path"]
    assert Path(created["queue_evidence_data_path"]).is_dir()


# ── batching + per-caller routing ──────────────────────────────────────────────
def test_queue_batches_concurrent_calls_and_returns_only_matching_results():
    q = ResearcherQueue()
    seen: dict[str, list[str]] = {}

    def processor(prompts, save_artifacts=False):
        assert save_artifacts is False
        seen["prompts"] = list(prompts)
        return json.dumps(
            {
                "researches": [
                    {"topic": p, "conclusions": "c", "members": [i]}
                    for i, p in enumerate(prompts)
                ],
                "count": len(prompts),
            }
        )

    results: dict[str, str] = {}

    def caller(name, prompts):
        results[name] = q.submit_and_wait(
            prompts,
            processor=processor,
            window_seconds=60,               # long window; the early-flush must win
            expected_participants=2,
            max_batch_requests=0,
            max_wait_seconds=30,
        )

    t1 = threading.Thread(target=caller, args=("a", ["req A"]))
    t2 = threading.Thread(target=caller, args=("b", ["req B"]))
    t1.start(); t2.start(); t1.join(); t2.join()

    # Both callers joined one batch and it flushed once both had submitted.
    assert sorted(seen["prompts"]) == ["req A", "req B"]
    # Each caller receives only the research covering its own request.
    assert results["a"] != results["b"]
    assert json.loads(results["a"])["count"] == 1
    assert json.loads(results["a"])["researches"][0]["topic"] == "req A"
    assert json.loads(results["b"])["count"] == 1
    assert json.loads(results["b"])["researches"][0]["topic"] == "req B"


def test_queue_preserved_artifacts_return_queue_and_request_folders():
    q = ResearcherQueue()
    created = json.loads(q.create_queue("artifact queue"))

    def processor(prompts, save_artifacts=False, queue_root="", batch_id=""):
        research_dir = Path(queue_root) / "researches" / "research-000"
        research_dir.mkdir(parents=True, exist_ok=True)
        (research_dir / "admin_output.json").write_text('{"ok":true}', encoding="utf-8")
        return json.dumps(
            {
                "queue_evidence_data_path": queue_root,
                "researches": [
                    {
                        "topic": prompts[0],
                        "conclusions": "c",
                        "members": [0],
                        "evidence_data_path": str(research_dir),
                    }
                ],
                "count": 1,
                "artifacts_preserved": True,
            }
        )

    out = q.submit_and_wait(
        ["request with artifacts"],
        processor=processor,
        window_seconds=0,
        expected_participants=0,
        max_batch_requests=0,
        max_wait_seconds=10,
        save_artifacts=True,
        queue_id=created["queue_id"],
    )
    payload = json.loads(out)

    assert payload["queue_evidence_data_path"] == created["queue_evidence_data_path"]
    assert Path(payload["request_evidence_data_path"]).is_dir()
    assert Path(payload["request_evidence_data_path"], "matched_researches.json").is_file()
    assert Path(payload["request_evidence_data_path"], "researches", "research-000").exists()
    assert payload["researches"][0]["evidence_data_path"].startswith(created["queue_evidence_data_path"])


def test_queue_timeout_returns_artifact_paths_when_preserved():
    q = ResearcherQueue()
    started = threading.Event()
    release = threading.Event()

    def processor(prompts, save_artifacts=False, queue_root="", batch_id=""):
        started.set()
        assert release.wait(10)
        return json.dumps({"researches": [{"topic": prompts[0], "members": [0]}], "count": 1})

    out = q.submit_and_wait(
        ["slow preserved request"],
        processor=processor,
        window_seconds=0,
        expected_participants=0,
        max_batch_requests=0,
        max_wait_seconds=1,
        save_artifacts=True,
    )
    payload = json.loads(out)
    release.set()

    assert started.is_set()
    assert payload["error"].startswith("researcher_queue timed out")
    assert payload["batch_id"]
    assert payload["queue_evidence_data_path"]
    assert payload["request_id"]
    assert Path(payload["request_evidence_data_path"]).is_dir()
    assert payload["artifacts_preserved"] is True


def test_queue_flushes_after_window_for_single_caller():
    q = ResearcherQueue()

    def processor(prompts, save_artifacts=False):
        return json.dumps(
            {
                "researches": [{"topic": prompts[0], "conclusions": "c", "members": [0]}],
                "count": len(prompts),
            }
        )

    out = q.submit_and_wait(
        ["only request"],
        processor=processor,
        window_seconds=0,                    # flush essentially immediately
        expected_participants=0,
        max_batch_requests=0,
        max_wait_seconds=10,
    )
    assert json.loads(out)["count"] == 1


def test_queue_flushes_when_max_batch_requests_reached():
    q = ResearcherQueue()

    def processor(prompts, save_artifacts=False):
        return json.dumps({"count": len(prompts)})

    out = q.submit_and_wait(
        ["p1", "p2", "p3"],
        processor=processor,
        window_seconds=60,
        expected_participants=0,
        max_batch_requests=3,                # 3 prompts trips the ceiling now
        max_wait_seconds=10,
    )
    assert json.loads(out)["count"] == 3


def test_queue_overflow_starts_new_batch_instead_of_exceeding_limit():
    q = ResearcherQueue()
    first_started = threading.Event()
    first_release = threading.Event()
    seen: dict[str, list[str]] = {}
    results: dict[str, str] = {}

    def first_processor(prompts, save_artifacts=False):
        seen["first"] = list(prompts)
        first_started.set()
        assert first_release.wait(10)
        return json.dumps({"researches": [{"topic": p, "members": [i]} for i, p in enumerate(prompts)], "count": len(prompts)})

    def second_processor(prompts, save_artifacts=False):
        seen["second"] = list(prompts)
        return json.dumps({"researches": [{"topic": p, "members": [i]} for i, p in enumerate(prompts)], "count": len(prompts)})

    def first_call():
        results["first"] = q.submit_and_wait(
            ["p0", "p1"],
            processor=first_processor,
            window_seconds=60,
            expected_participants=0,
            max_batch_requests=3,
            max_wait_seconds=10,
        )

    t1 = threading.Thread(target=first_call)
    t1.start()
    while True:
        status = json.loads(q.status())
        open_batch = status.get("open_batch") or {}
        if open_batch.get("prompt_count") == 2:
            break

    results["second"] = q.submit_and_wait(
        ["p2", "p3"],
        processor=second_processor,
        window_seconds=0,
        expected_participants=0,
        max_batch_requests=3,
        max_wait_seconds=10,
    )
    assert first_started.wait(10)
    first_release.set()
    t1.join(10)

    assert seen["first"] == ["p0", "p1"]
    assert seen["second"] == ["p2", "p3"]
    assert json.loads(results["first"])["count"] == 2
    assert json.loads(results["second"])["count"] == 2


def test_queue_closes_full_batch_before_slow_processing_finishes():
    q = ResearcherQueue()
    first_started = threading.Event()
    first_release = threading.Event()
    seen: dict[str, list[str]] = {}
    results: dict[str, str] = {}

    def first_processor(prompts, save_artifacts=False):
        seen["first"] = list(prompts)
        first_started.set()
        assert first_release.wait(10)
        return json.dumps({"researches": [{"topic": p, "conclusions": "first"} for p in prompts], "count": len(prompts)})

    def second_processor(prompts, save_artifacts=False):
        seen["second"] = list(prompts)
        return json.dumps({"researches": [{"topic": p, "conclusions": "second"} for p in prompts], "count": len(prompts)})

    t1 = threading.Thread(
        target=lambda: results.__setitem__(
            "first",
            q.submit_and_wait(
                ["first request"],
                processor=first_processor,
                window_seconds=60,
                expected_participants=0,
                max_batch_requests=1,
                max_wait_seconds=10,
            ),
        )
    )
    t1.start()
    assert first_started.wait(10)

    results["second"] = q.submit_and_wait(
        ["second request"],
        processor=second_processor,
        window_seconds=60,
        expected_participants=0,
        max_batch_requests=1,
        max_wait_seconds=10,
    )
    first_release.set()
    t1.join(10)

    assert seen["first"] == ["first request"]
    assert seen["second"] == ["second request"]
    assert json.loads(results["first"])["researches"][0]["conclusions"] == "first"
    assert json.loads(results["second"])["researches"][0]["conclusions"] == "second"


def test_queue_zero_window_closes_batch_immediately():
    q = ResearcherQueue()
    seen: list[list[str]] = []

    def processor(prompts, save_artifacts=False):
        seen.append(list(prompts))
        return json.dumps({"count": len(prompts)})

    first = q.submit_and_wait(
        ["first"],
        processor=processor,
        window_seconds=0,
        expected_participants=0,
        max_batch_requests=0,
        max_wait_seconds=10,
    )
    second = q.submit_and_wait(
        ["second"],
        processor=processor,
        window_seconds=0,
        expected_participants=0,
        max_batch_requests=0,
        max_wait_seconds=10,
    )

    assert json.loads(first)["count"] == 1
    assert json.loads(second)["count"] == 1
    assert seen == [["first"], ["second"]]


def test_queue_processor_error_never_hangs_caller():
    q = ResearcherQueue()

    def boom(prompts, save_artifacts=False):
        raise RuntimeError("kaboom")

    out = q.submit_and_wait(
        ["x"],
        processor=boom,
        window_seconds=0,
        expected_participants=0,
        max_batch_requests=0,
        max_wait_seconds=10,
    )
    payload = json.loads(out)
    assert payload["count"] == 0
    assert "kaboom" in payload["error"]


def test_queue_status_reports_open_and_processing_batches():
    q = ResearcherQueue()
    processing_started = threading.Event()
    release = threading.Event()

    def processor(prompts, save_artifacts=False):
        processing_started.set()
        assert release.wait(10)
        return json.dumps({"researches": [{"topic": prompts[0], "members": [0]}], "count": 1})

    def caller():
        q.submit_and_wait(
            ["queued"],
            processor=processor,
            window_seconds=60,
            expected_participants=0,
            max_batch_requests=0,
            max_wait_seconds=10,
            save_artifacts=True,
        )

    t = threading.Thread(target=caller)
    t.start()
    while True:
        status = json.loads(q.status())
        if status.get("open_batch"):
            break
    open_batch = status["open_batch"]
    assert open_batch["prompt_count"] == 1
    assert open_batch["caller_count"] == 1
    assert open_batch["save_artifacts"] is True
    assert open_batch["requests"][0]["request_id"]
    assert Path(open_batch["requests"][0]["request_evidence_data_path"]).is_dir()

    flush_thread = threading.Thread(target=lambda: q._flush(q._current))
    flush_thread.start()
    assert processing_started.wait(10)
    processing = json.loads(q.status())["processing_batches"]
    assert len(processing) == 1
    assert processing[0]["prompt_count"] == 1
    assert processing[0]["requests"][0]["request_id"] == open_batch["requests"][0]["request_id"]
    assert processing[0]["current_research_index"] == 0
    assert "latest_action" in processing[0]
    release.set()
    flush_thread.join(10)
    t.join(10)
    assert json.loads(q.status())["processing_count"] == 0


def test_queue_preserves_artifacts_when_any_caller_requests_it():
    q = ResearcherQueue()
    seen: dict[str, object] = {}
    results: dict[str, str] = {}

    def processor(prompts, save_artifacts=False):
        seen["prompts"] = list(prompts)
        seen["save_artifacts"] = save_artifacts
        return json.dumps({"researches": [], "count": len(prompts), "artifacts_preserved": save_artifacts})

    def caller(name, prompt, save_artifacts):
        results[name] = q.submit_and_wait(
            [prompt],
            processor=processor,
            window_seconds=60,
            expected_participants=2,
            max_batch_requests=0,
            max_wait_seconds=10,
            save_artifacts=save_artifacts,
        )

    t1 = threading.Thread(target=caller, args=("a", "req A", False))
    t2 = threading.Thread(target=caller, args=("b", "req B", True))
    t1.start(); t2.start(); t1.join(10); t2.join(10)

    assert sorted(seen["prompts"]) == ["req A", "req B"]
    assert seen["save_artifacts"] is True
    assert json.loads(results["a"])["artifacts_preserved"] is False
    assert json.loads(results["b"])["artifacts_preserved"] is True


# ── merge + dispatch ──────────────────────────────────────────────────────────
def test_process_batch_runs_admin_per_group_and_labels_topics(monkeypatch):
    helper = _make_helper()
    monkeypatch.setattr(
        helper,
        "_merge_prompts",
        lambda prompts: [("merged one covering a+b", [0, 1]), ("merged two", [2])],
    )
    monkeypatch.setattr(helper, "_run_admin", lambda prompt, ctx, save_artifacts=False: f"concl::{prompt}")

    out = helper._process_batch(["a", "b", "c"])
    payload = json.loads(out)

    assert payload["count"] == 2
    assert payload["researches"][0]["conclusions"] == "concl::merged one covering a+b"
    assert payload["researches"][0]["topic"].startswith("merged one covering a+b")
    assert payload["researches"][1]["conclusions"] == "concl::merged two"
    assert payload["researcher_usage"] == {
        "administrator_calls": 2,
        "researcher_call_counts": {},
        "total_researcher_calls": 0,
        "complete": False,
    }


def test_process_batch_aggregates_exact_private_researcher_usage(monkeypatch):
    helper = _make_helper()
    monkeypatch.setattr(
        helper,
        "_merge_prompts",
        lambda prompts: [("deep topic", [0]), ("focused topic", [1])],
    )

    def fake_admin(prompt, ctx, save_artifacts=False):
        if prompt == "deep topic":
            counts = {"deepchatgpt_researcher": 1, "websearcher_research": 2}
        else:
            counts = {"prochatgpt_researcher": 2, "websearcher_research": 1}
        return {
            "conclusions": f"done::{prompt}",
            "researcher_call_counts": counts,
            "total_researcher_calls": sum(counts.values()),
            "researcher_usage_complete": True,
        }

    monkeypatch.setattr(helper, "_run_admin", fake_admin)

    payload = json.loads(helper._process_batch(["a", "b"]))

    assert payload["researcher_usage"] == {
        "administrator_calls": 2,
        "researcher_call_counts": {
            "deepchatgpt_researcher": 1,
            "prochatgpt_researcher": 2,
            "websearcher_research": 3,
        },
        "total_researcher_calls": 6,
        "complete": True,
    }


def test_process_batch_includes_evidence_paths_when_artifacts_are_preserved(monkeypatch):
    helper = _make_helper()
    monkeypatch.setattr(helper, "_merge_prompts", lambda prompts: [("merged with artifacts", [0])])
    monkeypatch.setattr(
        helper,
        "_run_admin",
        lambda prompt, ctx, save_artifacts=False: {
            "conclusions": f"concl::{prompt}",
            "evidence_data_path": "/tmp/evidence/merged",
        },
    )

    out = helper._process_batch(["a"], save_artifacts=True)
    payload = json.loads(out)

    assert payload["artifacts_preserved"] is True
    assert payload["researches"][0]["conclusions"] == "concl::merged with artifacts"
    assert payload["researches"][0]["evidence_data_path"] == "/tmp/evidence/merged"


def test_process_batch_places_admin_runs_under_queue_root(monkeypatch, tmp_path):
    helper = _make_helper()
    queue_root = str(tmp_path / "queue-root")
    seen = {}
    monkeypatch.setattr(helper, "_merge_prompts", lambda prompts: [("merged prompt", [0], "single")])

    def fake_admin(prompt, ctx, save_artifacts=False):
        seen["research_master_dir"] = ctx["research_master_dir"]
        Path(ctx["research_master_dir"]).mkdir(parents=True, exist_ok=True)
        return {
            "conclusions": "admin conclusions",
            "evidence_data_path": ctx["research_master_dir"],
        }

    monkeypatch.setattr(helper, "_run_admin", fake_admin)

    out = helper._process_batch(["a"], save_artifacts=True, queue_root=queue_root, batch_id="batch-x")
    payload = json.loads(out)

    assert payload["queue_evidence_data_path"] == queue_root
    assert seen["research_master_dir"].startswith(str(Path(queue_root) / "researches"))
    assert payload["researches"][0]["evidence_data_path"] == seen["research_master_dir"]
    assert Path(seen["research_master_dir"], "merged_prompt.json").is_file()


def test_filter_removes_evidence_path_for_callers_that_did_not_request_artifacts():
    result = json.dumps(
        {
            "researches": [
                {
                    "topic": "merged",
                    "conclusions": "c",
                    "evidence_data_path": "/tmp/evidence",
                    "members": [0],
                }
            ],
            "count": 1,
            "artifacts_preserved": True,
        }
    )
    waiter = _QueueWaiter(0, 1, save_artifacts=False)

    filtered = ResearcherQueue._filter_result_for_waiter(
        result,
        waiter,
        batch_id="batch-test",
        artifacts_preserved=True,
    )
    payload = json.loads(filtered)

    assert payload["artifacts_preserved"] is False
    assert "evidence_data_path" not in payload["researches"][0]


def test_filter_recomputes_usage_for_each_batched_caller():
    result = json.dumps(
        {
            "researches": [
                {
                    "topic": "caller A",
                    "members": [0],
                    "researcher_call_counts": {"prochatgpt_researcher": 2},
                    "researcher_usage_complete": True,
                },
                {
                    "topic": "caller B",
                    "members": [1],
                    "researcher_call_counts": {"deepchatgpt_researcher": 1},
                    "researcher_usage_complete": True,
                },
            ],
            "count": 2,
        }
    )

    first = json.loads(
        ResearcherQueue._filter_result_for_waiter(
            result,
            _QueueWaiter(0, 1, save_artifacts=False),
            batch_id="batch-usage",
            artifacts_preserved=False,
        )
    )
    second = json.loads(
        ResearcherQueue._filter_result_for_waiter(
            result,
            _QueueWaiter(1, 2, save_artifacts=False),
            batch_id="batch-usage",
            artifacts_preserved=False,
        )
    )

    assert first["researcher_usage"] == {
        "administrator_calls": 1,
        "researcher_call_counts": {"prochatgpt_researcher": 2},
        "total_researcher_calls": 2,
        "complete": True,
    }
    assert second["researcher_usage"] == {
        "administrator_calls": 1,
        "researcher_call_counts": {"deepchatgpt_researcher": 1},
        "total_researcher_calls": 1,
        "complete": True,
    }


def test_helper_run_batches_merges_and_routes_shared_research_to_matching_callers(monkeypatch):
    helper = _make_helper(window_seconds=60, expected_participants=2, min_prompt_chars=20)
    admin_calls: list[str] = []

    def fake_merge(prompts):
        assert len(prompts) == 2
        return [("merged prompt covering both caller requests with all requested details", [0, 1])]

    def fake_admin(prompt, ctx, save_artifacts=False):
        admin_calls.append(prompt)
        return f"admin conclusions for {prompt}"

    monkeypatch.setattr(helper, "_merge_prompts", fake_merge)
    monkeypatch.setattr(helper, "_run_admin", fake_admin)

    results: dict[str, str] = {}

    t1 = threading.Thread(
        target=lambda: results.__setitem__(
            "a",
            helper.run("research the same product safety issue from angle A with primary sources"),
        )
    )
    t2 = threading.Thread(
        target=lambda: results.__setitem__(
            "b",
            helper.run("research the same product safety issue from angle B with primary sources"),
        )
    )
    t1.start(); t2.start(); t1.join(10); t2.join(10)

    assert len(admin_calls) == 1
    assert results["a"] == results["b"]
    payload = json.loads(results["a"])
    assert payload["count"] == 1
    assert payload["researches"][0]["topic"].startswith("merged prompt covering both caller")
    assert payload["researches"][0]["conclusions"].startswith("admin conclusions for merged prompt")


def test_helper_run_does_not_return_unrelated_batch_research_to_callers(monkeypatch):
    helper = _make_helper(window_seconds=60, expected_participants=2, min_prompt_chars=20)

    monkeypatch.setattr(
        helper,
        "_merge_prompts",
        lambda prompts: [
            ("research only caller A", [0], "separate topic for caller A"),
            ("research only caller B", [1], "separate topic for caller B"),
        ],
    )
    monkeypatch.setattr(
        helper,
        "_run_admin",
        lambda prompt, ctx, save_artifacts=False: {"conclusions": f"conclusions::{prompt}"},
    )

    results: dict[str, str] = {}
    t1 = threading.Thread(target=lambda: results.__setitem__("a", helper.run("research topic A with primary sources")))
    t2 = threading.Thread(target=lambda: results.__setitem__("b", helper.run("research topic B with primary sources")))
    t1.start(); t2.start(); t1.join(10); t2.join(10)

    a = json.loads(results["a"])
    b = json.loads(results["b"])
    assert a["count"] == 1
    assert b["count"] == 1
    assert a["researches"][0]["topic"] == "research only caller A"
    assert b["researches"][0]["topic"] == "research only caller B"
    assert "caller B" not in json.dumps(a)
    assert "caller A" not in json.dumps(b)


def test_run_admin_passes_save_artifacts_and_extracts_evidence_path():
    seen = {}

    class Admin:
        def _run_single(self, prompt, ctx, save_artifacts=False):
            seen["prompt"] = prompt
            seen["save_artifacts"] = save_artifacts
            return json.dumps(
                {
                    "research_worked": True,
                    "failure_reason": "",
                    "administrator_conclusions": "admin conclusion",
                    "evidence_data_path": "/tmp/evidence/admin",
                    "researcher_call_counts": {
                        "deepchatgpt_researcher": 1,
                        "prochatgpt_researcher": 2,
                    },
                }
            )

    helper = ResearcherQueueAgentTool(
        Admin(),
        config=ToolsConfig(),
        model_provider="openai",
        fallback_model="m",
        queue=ResearcherQueue(),
    )

    out = helper._run_admin("prompt text", {}, save_artifacts=True)

    assert seen == {"prompt": "prompt text", "save_artifacts": True}
    assert out == {
        "conclusions": "admin conclusion",
        "evidence_data_path": "/tmp/evidence/admin",
        "researcher_call_counts": {
            "deepchatgpt_researcher": 1,
            "prochatgpt_researcher": 2,
        },
        "total_researcher_calls": 3,
        "researcher_usage_complete": True,
    }


def test_queue_research_context_uses_fixed_queue_limits_not_requester_context():
    cfg = ToolsConfig(
        researcher_queue_max_runtime_minutes=42,
        researcher_queue_max_cost_usd=1.25,
    )
    helper = ResearcherQueueAgentTool(
        SimpleNamespace(max_turns=77),
        config=cfg,
        model_provider="openai",
        fallback_model="m",
        queue=ResearcherQueue(),
    )

    ctx = helper._queue_research_context()

    assert ctx["max_turns"] == 77
    assert ctx["max_runtime_minutes"] == 42
    assert ctx["remaining_runtime_minutes"] == 42.0
    assert ctx["max_cost_usd"] == 1.25
    assert ctx["remaining_cost_usd"] == 1.25
    assert ctx["main_action"] == "researcher_queue"


def test_parse_merge_groups_covers_every_request_and_ignores_bad_indices():
    prompts = ["p0", "p1", "p2"]
    output = json.dumps(
        {
            "groups": [
                {"prompt": "merged 0+1", "members": [0, 1, 99]},  # 99 is out of range
                {"prompt": "", "members": [2]},                    # empty prompt -> dropped
            ]
        }
    )
    groups = ResearcherQueueAgentTool._parse_merge_groups(output, prompts)
    # p2 was left uncovered (its group had an empty prompt) -> becomes its own research.
    assert groups[0] == ("merged 0+1", [0, 1], "")
    assert ("p2", [2], "merge agent omitted this request; dispatched separately") in groups
    covered = sorted(i for _p, members, _reason in groups for i in members)
    assert covered == [0, 1, 2]


def test_parse_merge_groups_returns_none_on_garbage():
    assert ResearcherQueueAgentTool._parse_merge_groups("not json", ["p0"]) is None
    assert ResearcherQueueAgentTool._parse_merge_groups(json.dumps({"nope": 1}), ["p0"]) is None


def test_merge_prompts_falls_back_to_one_to_one_when_agent_fails(monkeypatch):
    helper = _make_helper()
    monkeypatch.setattr(helper, "_run_merge_agent", lambda prompts: None)
    groups = helper._merge_prompts(["a", "b"])
    assert groups == [
        ("a", [0], "merge unavailable; dispatched separately"),
        ("b", [1], "merge unavailable; dispatched separately"),
    ]


def test_merge_prompts_single_prompt_skips_merge(monkeypatch):
    helper = _make_helper()
    called = {"n": 0}
    monkeypatch.setattr(helper, "_run_merge_agent", lambda prompts: called.__setitem__("n", called["n"] + 1))
    assert helper._merge_prompts(["solo"]) == [("solo", [0], "single request; no merge needed")]
    assert called["n"] == 0


# ── input validation ──────────────────────────────────────────────────────────
def test_run_rejects_more_than_max_requests_per_call():
    helper = _make_helper(max_requests_per_call=5)
    prompts = [f"detailed research request number {i} " * 12 for i in range(6)]
    out = helper.run(prompts)
    assert "at most 5 prompts" in out


def test_queue_tool_description_uses_configured_per_call_limit():
    helper = _make_helper(max_requests_per_call=3)
    tool = get_researcher_queue_tool(helper)
    assert "list of up to 3" in tool.description
    assert "save_artifacts true" in tool.description
