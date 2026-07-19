from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time

import pytest

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.chatgpt_research_agents import (
    CHATGPT_DEEP_OUTPUT_TIMEOUT_SECONDS,
    CHATGPT_PRO_OUTPUT_TIMEOUT_SECONDS,
    ChatGPTWebResearchAgentTool,
    ChatGPTWebResearchError,
    resolve_chatgpt_timeout_seconds,
)
from chack_tools.chatgpt_async_client import ChatGPTAsyncApiClient
from chack_tools.config import ToolsConfig
from chack_tools.researcher_administrator_agent import (
    ResearcherAdministratorAgentTool,
    normalize_researcher_name,
)


def _tool_names(tools):
    return {str(getattr(tool, "name", "")) for tool in tools}


def test_chatgpt_research_tools_register_only_when_enabled():
    off = AgentsToolset(ToolsConfig(), model_provider="openai", default_model="gpt-5-mini")
    assert "deepchatgpt_researcher" not in _tool_names(off.tools)
    assert "prochatgpt_researcher" not in _tool_names(off.tools)

    on = AgentsToolset(
        ToolsConfig(deepchatgpt_enabled=True, prochatgpt_enabled=True),
        model_provider="openai",
        default_model="gpt-5-mini",
    )
    assert {"deepchatgpt_researcher", "prochatgpt_researcher"} <= _tool_names(on.tools)


def test_chatgpt_modes_have_distinct_total_output_deadlines():
    config = ToolsConfig()
    assert resolve_chatgpt_timeout_seconds(config, "pro") == 30 * 60
    assert resolve_chatgpt_timeout_seconds(config, "deep") == 75 * 60
    assert CHATGPT_PRO_OUTPUT_TIMEOUT_SECONDS == 1800
    assert CHATGPT_DEEP_OUTPUT_TIMEOUT_SECONDS == 4500


def test_chatgpt_research_tool_accepts_and_runs_five_prompts_in_parallel(monkeypatch):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="pro")
    barrier = threading.Barrier(5)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def run_single(prompt, *, save_artifacts):
        nonlocal active, maximum_active
        assert not save_artifacts
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=5)
        time.sleep(0.02)
        with lock:
            active -= 1
        return f"answer:{prompt[-1]}"

    monkeypatch.setattr(helper, "_run_single", run_single)
    output = helper.run([("P" * 100) + str(index) for index in range(5)])
    assert maximum_active == 5
    assert output.count("SUBAGENT_RESULT_") == 5


def test_mode_specific_timeout_overrides_legacy_shared_timeout():
    config = ToolsConfig(
        chatgpt_pro_timeout_seconds=321,
        chatgpt_deep_timeout_seconds=654,
        chatgpt_research_timeout_seconds=999,
    )
    assert resolve_chatgpt_timeout_seconds(config, "pro") == 321
    assert resolve_chatgpt_timeout_seconds(config, "deep") == 654


def test_legacy_shared_timeout_remains_a_compatibility_fallback():
    config = ToolsConfig(chatgpt_research_timeout_seconds=777)
    assert resolve_chatgpt_timeout_seconds(config, "pro") == 777
    assert resolve_chatgpt_timeout_seconds(config, "deep") == 777


def test_chatgpt_aliases_are_accepted_by_administrator():
    assert normalize_researcher_name("chatgpt-deep") == "deepchatgpt"
    assert normalize_researcher_name("prochatgpt_researcher") == "prochatgpt"

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(deepchatgpt_enabled=True, prochatgpt_enabled=True),
        model_provider="openai",
        fallback_model="gpt-5-mini",
        researchers=["chatgpt_deep", "chatgpt_pro"],
    )
    assert helper._enabled_researchers() == ["deepchatgpt", "prochatgpt"]


def test_successful_chatgpt_run_uses_researcher_contract(monkeypatch, tmp_path):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="pro")
    evidence = tmp_path / "evidence"
    monkeypatch.setattr(
        "chack_tools.chatgpt_research_agents.create_subagent_evidence_dir",
        lambda *_args, **_kwargs: str(evidence),
    )
    monkeypatch.setattr(
        helper,
        "_browser_research",
        lambda _prompt, **_kwargs: (
            "A" * 2500,
            "https://chatgpt.com/c/test-conversation",
            {"mode": "pro", "terminal_state": "extracted", "answer_chars": 2500},
        ),
    )

    payload = json.loads(helper._run_single("P" * 500, save_artifacts=True))
    assert payload["research_worked"] is True
    run_dir = Path(payload["evidence_data_path"])
    assert run_dir.parent == evidence
    assert run_dir.name.startswith("run-")
    assert payload["final_research_review"] == "A" * 2500
    assert {row["filename"] for row in payload["key_artifacts"]} == {
        "chatgpt-pro-response.md",
        "chatgpt-request.md",
        "chatgpt-run.json",
    }
    assert (run_dir / "chatgpt-pro-response.md").read_text() == "A" * 2500


def test_five_same_mode_runs_use_distinct_artifact_directories(monkeypatch, tmp_path):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="pro")
    evidence = tmp_path / "evidence"
    monkeypatch.setattr(
        "chack_tools.chatgpt_research_agents.create_subagent_evidence_dir",
        lambda *_args, **_kwargs: str(evidence),
    )
    barrier = threading.Barrier(5)

    def browser(_prompt, *, run_state_path, partial_path):
        assert run_state_path.parent == partial_path.parent
        barrier.wait(timeout=5)
        return "A" * 500, "", {"terminal_state": "extracted"}

    monkeypatch.setattr(helper, "_browser_research", browser)
    prompts = [f"Prompt {index} " * 60 for index in range(5)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        payloads = [json.loads(future.result()) for future in [pool.submit(helper._run_single, prompt, save_artifacts=True) for prompt in prompts]]

    run_dirs = [Path(payload["evidence_data_path"]) for payload in payloads]
    assert len(set(run_dirs)) == 5
    assert all(path.parent == evidence for path in run_dirs)
    assert {path.joinpath("chatgpt-request.md").read_text() for path in run_dirs} == set(prompts)


def test_auto_backend_uses_remote_only_when_url_and_secret_are_present(monkeypatch):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="pro")
    monkeypatch.delenv("CHACK_CHATGPT_ASYNC_API_URL", raising=False)
    monkeypatch.delenv("CHACK_CHATGPT_ASYNC_API_SECRET", raising=False)
    assert helper._execution_backend() == "local"

    monkeypatch.setenv("CHACK_CHATGPT_ASYNC_API_URL", "https://broker.example")
    monkeypatch.setenv("CHACK_CHATGPT_ASYNC_API_SECRET", "test-secret")
    assert helper._execution_backend() == "remote"

    local = ChatGPTWebResearchAgentTool(ToolsConfig(chatgpt_execution_backend="local"), mode="deep")
    assert local._execution_backend() == "local"


def test_remote_backend_submits_polls_and_preserves_result(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self):
            self.polls = 0

        def submit(self, **kwargs):
            assert kwargs["mode"] == "deep"
            assert kwargs["prompt"] == "P" * 500
            assert kwargs["idempotency_key"]
            return {"job_id": "job_00000000-0000-0000-0000-000000000001", "status": "QUEUED"}

        def status(self, _job_id):
            self.polls += 1
            if self.polls == 1:
                return {"status": "RUNNING", "stage": "browser_running", "answer_chars": 100}
            return {"status": "SUCCEEDED", "stage": "extracted", "answer_chars": 2500}

        def result(self, _job_id):
            return {
                "status": "SUCCEEDED",
                "result": "R" * 2500,
                "partial_result": "",
                "metadata": {"conversation_url": "https://chatgpt.com/c/remote-test"},
            }

    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(
            chatgpt_execution_backend="remote",
            chatgpt_async_api_url="https://broker.example",
            chatgpt_async_api_secret="test-secret",
            chatgpt_async_poll_seconds=2,
        ),
        mode="deep",
    )
    fake = FakeClient()
    monkeypatch.setattr(helper, "_async_client", lambda: fake)
    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.sleep", lambda *_args: None)
    run_state = tmp_path / "run.json"
    partial = tmp_path / "partial.md"

    answer, url, metadata = helper._remote_research(
        "P" * 500,
        run_state_path=run_state,
        partial_path=partial,
    )
    assert answer == "R" * 2500
    assert url == ""
    assert "conversation_url" not in metadata
    assert metadata["execution_backend"] == "remote"
    assert metadata["remote_status"] == "SUCCEEDED"
    assert json.loads(run_state.read_text())["remote_job_id"].startswith("job_")


def test_async_client_sends_bearer_secret_only_in_header():
    class Response:
        status_code = 202
        content = b"{}"

        @staticmethod
        def json():
            return {"job_id": "job_test"}

    class Session:
        def __init__(self):
            self.call = None

        def request(self, method, url, **kwargs):
            self.call = (method, url, kwargs)
            return Response()

    session = Session()
    client = ChatGPTAsyncApiClient("https://broker.example", "known-secret", session=session)  # type: ignore[arg-type]
    client.submit(mode="pro", prompt="P" * 100, idempotency_key="idem")
    assert session.call is not None
    method, url, kwargs = session.call
    assert method == "POST"
    assert "known-secret" not in url
    assert kwargs["headers"]["authorization"] == "Bearer known-secret"
    assert "known-secret" not in json.dumps(kwargs["json"])


def test_deep_research_counter_noise_is_removed_without_touching_normal_numbered_answers():
    noisy = (
        "Research completed in 8m ·\n"
        + "\n".join(str(i % 10) for i in range(30))
        + "\n citations · \nsearches\n10\n11\n2008\nExecutive summary\nSubstantive evidence.\n1\n\n"
        + "LIPEDEMA_DEEP_MCP_OK"
    )
    cleaned = ChatGPTWebResearchAgentTool._clean_extracted_text(noisy)
    assert "Executive summary" in cleaned
    assert "Substantive evidence." in cleaned
    assert "2008" in cleaned
    assert "LIPEDEMA_DEEP_MCP_OK" in cleaned
    assert "citations ·" not in cleaned
    assert "searches" not in cleaned
    assert not re.search(r"(?m)^\d{1,2}$", cleaned)

    normal = "1\nFirst finding\n2\nSecond finding"
    assert ChatGPTWebResearchAgentTool._clean_extracted_text(normal) == normal


def test_deep_connector_wait_path_applies_counter_cleanup(monkeypatch):
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(chatgpt_research_timeout_seconds=60, chatgpt_research_poll_seconds=1),
        mode="deep",
    )
    noisy = (
        "Research completed in 8m ·\n"
        + "\n".join(str(i % 10) for i in range(30))
        + "\nExecutive summary\n"
        + ("Substantive controlled-study evidence with limitations and direct interpretation. " * 30)
        + "\nFINAL_DEEP_BROWSER_EXTRACT_OK"
    )
    state = {
        "text": noisy,
        "links": [{"label": "Trial", "url": "https://example.org/trial"}],
        "completed": True,
        "hasStop": False,
    }
    monkeypatch.setattr(helper, "_deep_connector_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.sleep", lambda *_args: None)

    answer = helper._wait_and_extract_deep({"webSocketDebuggerUrl": "ws://test"})
    assert "FINAL_DEEP_BROWSER_EXTRACT_OK" in answer
    assert "https://example.org/trial" in answer
    assert not re.search(r"(?m)^\d{1,2}$", answer)


def test_source_links_are_preserved_deduplicated_and_tracking_is_removed():
    answer = ChatGPTWebResearchAgentTool._append_source_links(
        "Clinical synthesis.\n\nLIPEDEMA_PRO_MCP_OK",
        [
            {
                "label": "Wright et al., 2023",
                "url": "https://pubmed.ncbi.nlm.nih.gov/36519532/?utm_source=chatgpt.com",
            },
            {
                "label": "Duplicate",
                "url": "https://pubmed.ncbi.nlm.nih.gov/36519532/?utm_source=chatgpt.com",
            },
            {"label": "Relative UI asset", "url": "/cdn/citation"},
        ],
    )

    assert "Source links:" in answer
    assert answer.count("https://pubmed.ncbi.nlm.nih.gov/36519532/") == 1
    assert "utm_source" not in answer
    assert "/cdn/citation" not in answer
    assert answer.endswith("LIPEDEMA_PRO_MCP_OK")


def test_source_links_are_not_repeated_when_already_rendered_in_text():
    url = "https://example.org/study"
    answer = ChatGPTWebResearchAgentTool._append_source_links(
        f"Evidence: {url}",
        [{"label": "Study", "url": url}],
    )
    assert answer == f"Evidence: {url}"


def test_running_state_accepts_stop_answering_label():
    class Locator:
        def __init__(self, count):
            self._count = count

        def count(self):
            return self._count

    class Page:
        def get_by_role(self, _role, name):
            return Locator(1 if name.search("Stop answering") else 0)

    assert ChatGPTWebResearchAgentTool._is_running(Page()) is True


def test_running_state_accepts_answer_now_label():
    class Locator:
        def count(self):
            return 1

    class Page:
        def get_by_role(self, _role, name):
            return Locator() if name.search("Answer now") else type("Empty", (), {"count": lambda self: 0})()

    assert ChatGPTWebResearchAgentTool._is_running(Page()) is True


def test_pro_timeout_forces_answer_early_and_extracts_within_total_deadline(monkeypatch, tmp_path):
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(
            chatgpt_pro_timeout_seconds=120,
            chatgpt_research_poll_seconds=2,
            chatgpt_force_answer_grace_seconds=60,
        ),
        mode="pro",
    )
    clock = {"now": 0.0}
    clicked = []

    class Page:
        url = "https://chatgpt.com/c/forced-answer"

        def wait_for_timeout(self, milliseconds):
            clock["now"] += milliseconds / 1000.0

    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(helper, "_click_answer_now_if_present", lambda _page: clicked.append(clock["now"]) or True)
    monkeypatch.setattr(helper, "_longest_answer", lambda _page: "R" * 300 if clicked else "")
    monkeypatch.setattr(helper, "_is_running", lambda _page: not bool(clicked))
    run_state = tmp_path / "chatgpt-run.json"
    partial = tmp_path / "partial.md"

    answer = helper._wait_and_extract(Page(), partial_path=partial, run_state_path=run_state)

    assert answer == "R" * 300
    assert clicked == [60.0]
    assert clock["now"] == 64.0
    assert clock["now"] < 120.0
    assert partial.read_text() == "R" * 300
    state = json.loads(run_state.read_text())
    assert state["forced_answer"] is True
    assert state["output_timeout_seconds"] == 120


def test_pro_answer_now_window_never_extends_total_output_deadline(monkeypatch):
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(
            chatgpt_pro_timeout_seconds=120,
            chatgpt_research_poll_seconds=10,
            chatgpt_force_answer_grace_seconds=60,
        ),
        mode="pro",
    )
    clock = {"now": 0.0}
    clicked = []

    class Page:
        url = "https://chatgpt.com/c/broken"

        def wait_for_timeout(self, milliseconds):
            clock["now"] += milliseconds / 1000.0

    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(helper, "_click_answer_now_if_present", lambda _page: clicked.append(clock["now"]) or True)
    monkeypatch.setattr(helper, "_longest_answer", lambda _page: "")
    monkeypatch.setattr(helper, "_is_running", lambda _page: True)

    with pytest.raises(ChatGPTWebResearchError, match="120-second total output deadline"):
        helper._wait_and_extract(Page())

    assert clicked == [60.0]
    assert clock["now"] == 120.0


def test_deep_research_stops_at_its_total_output_deadline(monkeypatch):
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(chatgpt_deep_timeout_seconds=120, chatgpt_research_poll_seconds=45),
        mode="deep",
    )
    clock = {"now": 0.0}
    state = {"text": "", "links": [], "completed": False, "hasStop": True}
    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))
    monkeypatch.setattr(helper, "_deep_connector_state", lambda *_args, **_kwargs: state)

    with pytest.raises(ChatGPTWebResearchError, match="120-second total output deadline"):
        helper._wait_and_extract_deep({"webSocketDebuggerUrl": "ws://test"})

    assert clock["now"] == 120.0


def test_failed_run_preserves_partial_response_and_conversation_url(monkeypatch, tmp_path):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="pro")
    evidence = tmp_path / "evidence"
    monkeypatch.setattr(
        "chack_tools.chatgpt_research_agents.create_subagent_evidence_dir",
        lambda *_args, **_kwargs: str(evidence),
    )

    def fail_after_partial(_prompt, *, run_state_path, partial_path):
        helper._write_json(
            run_state_path,
            {
                "mode": "pro",
                "terminal_state": "timeout",
                "conversation_url": "https://chatgpt.com/c/recoverable",
            },
        )
        partial_path.write_text("Recovered partial evidence", encoding="utf-8")
        raise RuntimeError("browser deadline")

    monkeypatch.setattr(helper, "_browser_research", fail_after_partial)
    payload = json.loads(helper._run_single("P" * 500, save_artifacts=True))

    assert payload["research_worked"] is False
    assert payload["partial_result"] is True
    assert payload["final_research_review"] == "Recovered partial evidence"
    assert {row["filename"] for row in payload["key_artifacts"]} == {
        "chatgpt-run.json",
        "chatgpt-request.md",
        "chatgpt-pro-partial.md",
    }
    assert all(row["source_url"] == "https://chatgpt.com/c/recoverable" for row in payload["key_artifacts"])
