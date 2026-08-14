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
    CHATGPT_XHIGH_OUTPUT_TIMEOUT_SECONDS,
    _XHIGH_COMPAT_PROMPT_PREFIX,
    ChatGPTWebResearchAgentTool,
    ChatGPTWebResearchError,
    resolve_chatgpt_timeout_seconds,
)
from chack_tools.chatgpt_async_client import ChatGPTAsyncApiClient, ChatGPTAsyncApiError
from chack_tools.cancellation import reset_cancellation_event, set_cancellation_event
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
    assert "chatgptxhigh" not in _tool_names(off.tools)

    on = AgentsToolset(
        ToolsConfig(
            deepchatgpt_enabled=True,
            prochatgpt_enabled=True,
            chatgptxhigh_enabled=True,
        ),
        model_provider="openai",
        default_model="gpt-5-mini",
    )
    assert {
        "deepchatgpt_researcher",
        "prochatgpt_researcher",
        "chatgptxhigh",
    } <= _tool_names(on.tools)


def test_chatgpt_modes_have_distinct_total_output_deadlines():
    config = ToolsConfig()
    assert resolve_chatgpt_timeout_seconds(config, "pro") == 120 * 60
    assert resolve_chatgpt_timeout_seconds(config, "xhigh") == 30 * 60
    assert resolve_chatgpt_timeout_seconds(config, "deep") == 75 * 60
    assert CHATGPT_PRO_OUTPUT_TIMEOUT_SECONDS == 7200
    assert CHATGPT_XHIGH_OUTPUT_TIMEOUT_SECONDS == 1800
    assert CHATGPT_DEEP_OUTPUT_TIMEOUT_SECONDS == 4500


def test_browser_progress_can_report_source_url_count(monkeypatch):
    events = []
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="deep")
    monkeypatch.setattr(
        "chack_tools.chatgpt_research_agents.current_log_context",
        lambda: {"_chack_tool_progress_callback": lambda event, payload: events.append((event, payload))},
    )

    helper._emit_progress(
        "answer_extracted",
        answer_chars=1234,
        source_url_count=7,
        running=False,
    )

    assert events[0][0] == "research_progress"
    assert events[0][1]["answer_chars"] == 1234
    assert events[0][1]["source_url_count"] == 7
    assert events[0][1]["running"] is False


class _ModeLocator:
    def __init__(self, page, role, pattern, label=""):
        self.page = page
        self.role = role
        self.pattern = pattern
        self.label = label

    def count(self):
        if self.role == "button":
            return 1 if self.pattern.search(self.page.selected) else 0
        return 0

    def nth(self, _index):
        return self

    def is_visible(self):
        return True

    def inner_text(self, timeout=0):
        return self.page.selected

    def get_attribute(self, name):
        if name == "aria-label":
            return self.page.selected
        return None

    def click(self, timeout=0):
        assert timeout == 5000
        self.page.menu_open = True
        self.page.clicks.append(f"open:{self.page.selected}")


class _TextLocator:
    def __init__(self, page):
        self.page = page

    def count(self):
        return 1 if self.page.menu_open else 0

    def nth(self, _index):
        return self

    def is_visible(self):
        return True

    def inner_text(self, timeout=0):
        return f"{self.page.selected}, {self.page.power + 1} of 5. Use Left and Right arrow keys to adjust power."


class _SliderLocator:
    def __init__(self, page):
        self.page = page

    def count(self):
        return 1 if self.page.menu_open else 0

    def nth(self, _index):
        return self

    def is_visible(self):
        return True

    def get_attribute(self, name):
        return {
            "aria-valuemin": "0",
            "aria-valuemax": "4",
            "aria-valuenow": str(self.page.power),
        }.get(name)

    def press(self, key, timeout=0):
        assert timeout == 5000
        if key == "ArrowRight":
            self.page.power = min(4, self.page.power + 1)
        elif key == "ArrowLeft":
            self.page.power = max(0, self.page.power - 1)
        self.page.selected = self.page.options[self.page.power]
        self.page.clicks.append(f"power:{key}")


class _EmptyLocator:
    def count(self):
        return 0


class _ModePage:
    options = ("Instant", "Medium", "High", "Extra High", "Pro")

    def __init__(self, selected):
        self.power = self.options.index(selected)
        self.selected = selected
        self.menu_open = False
        self.clicks = []

    def get_by_role(self, role, name=None):
        if role == "button":
            return _ModeLocator(self, role, name)
        if role == "menu":
            return _TextLocator(self)
        return _EmptyLocator()

    def get_by_text(self, _name):
        return _EmptyLocator()

    def locator(self, selector):
        if "role='slider'" in selector:
            return _SliderLocator(self)
        if "composer-model-picker-slider-simple-view" in selector or "[role='menu']" in selector:
            return _TextLocator(self)
        return _EmptyLocator()

    def wait_for_timeout(self, milliseconds):
        assert milliseconds in {250, 500}


@pytest.mark.parametrize(
    ("mode", "starting", "expected", "expected_power"),
    [
        ("pro", "Extra High", "Pro", "5/5"),
        ("pro", "Pro", "Pro", "5/5"),
        ("xhigh", "Pro", "Extra High", "4/5"),
        ("xhigh", "High", "Extra High", "4/5"),
    ],
)
def test_current_power_picker_selects_distinct_pro_and_xhigh_levels(mode, starting, expected, expected_power):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode=mode)
    page = _ModePage(starting)
    selected = helper._select_reasoning_mode(page)
    assert page.selected == expected
    assert selected["selected_effort"] == expected
    assert selected["selected_power"] == expected_power
    assert page.clicks[0] == f"open:{starting}"


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
        chatgpt_xhigh_timeout_seconds=987,
        chatgpt_deep_timeout_seconds=654,
        chatgpt_research_timeout_seconds=999,
    )
    assert resolve_chatgpt_timeout_seconds(config, "pro") == 321
    assert resolve_chatgpt_timeout_seconds(config, "xhigh") == 987
    assert resolve_chatgpt_timeout_seconds(config, "deep") == 654


def test_default_xhigh_timeout_and_async_wait_are_bounded():
    config = ToolsConfig()
    helper = ChatGPTWebResearchAgentTool(config, mode="xhigh")
    assert config.chatgpt_xhigh_timeout_seconds is None
    assert resolve_chatgpt_timeout_seconds(config, "xhigh") == 1800
    assert helper._async_max_wait_seconds() == 2100
    assert config.researcher_administrator_child_timeout_seconds >= helper._async_max_wait_seconds()


def test_xhigh_async_wait_caps_stale_legacy_values_at_timeout_plus_grace():
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(
            chatgpt_xhigh_timeout_seconds=1800,
            chatgpt_async_max_wait_seconds=10800,
            chatgpt_force_answer_grace_seconds=300,
        ),
        mode="xhigh",
    )
    assert helper._async_max_wait_seconds() == 2100


def test_legacy_shared_timeout_remains_a_compatibility_fallback():
    config = ToolsConfig(chatgpt_research_timeout_seconds=777)
    assert resolve_chatgpt_timeout_seconds(config, "pro") == 777
    assert resolve_chatgpt_timeout_seconds(config, "xhigh") == 777
    assert resolve_chatgpt_timeout_seconds(config, "deep") == 777


def test_chatgpt_aliases_are_accepted_by_administrator():
    assert normalize_researcher_name("chatgpt-deep") == "deepchatgpt"
    assert normalize_researcher_name("prochatgpt_researcher") == "prochatgpt"
    assert normalize_researcher_name("chatgpt_xhigh") == "chatgptxhigh"
    assert normalize_researcher_name("chatgptxhigh") == "chatgptxhigh"

    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(
            deepchatgpt_enabled=True,
            prochatgpt_enabled=True,
            chatgptxhigh_enabled=True,
        ),
        model_provider="openai",
        fallback_model="gpt-5-mini",
        researchers=["chatgpt_deep", "chatgpt_pro", "chatgpt_xhigh"],
    )
    assert helper._enabled_researchers() == [
        "deepchatgpt",
        "prochatgpt",
        "chatgptxhigh",
    ]


def test_successful_chatgpt_run_uses_researcher_contract(monkeypatch, tmp_path):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(chatgpt_execution_backend="local"), mode="pro")
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
    assert payload["full_research_review"] == "A" * 2500
    assert len(payload["overall_summary"]) == 1000
    assert 1 <= len(payload["findings"]) <= 8
    assert "final_research_review" not in payload
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


def test_auto_backend_fails_closed_when_any_remote_setting_is_present(monkeypatch):
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="pro")
    monkeypatch.delenv("CHACK_CHATGPT_ASYNC_API_URL", raising=False)
    monkeypatch.delenv("CHACK_CHATGPT_ASYNC_API_SECRET", raising=False)
    assert helper._execution_backend() == "local"

    monkeypatch.setenv("CHACK_CHATGPT_ASYNC_API_URL", "https://broker.example")
    assert helper._execution_backend() == "remote"
    with pytest.raises(ChatGPTWebResearchError, match="requires CHACK_CHATGPT_ASYNC_API_URL"):
        helper._async_client()

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
            assert kwargs["output_timeout_seconds"] == 4500
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


def test_xhigh_remote_backend_submits_exact_mode_and_preserves_result(monkeypatch, tmp_path):
    class FakeClient:
        def submit(self, **kwargs):
            assert kwargs["mode"] == "xhigh"
            assert kwargs["output_timeout_seconds"] == 1800
            return {
                "job_id": "job_00000000-0000-0000-0000-000000000002",
                "status": "QUEUED",
            }

        def status(self, _job_id):
            return {"status": "SUCCEEDED", "stage": "extracted", "answer_chars": 300}

        def result(self, _job_id):
            return {
                "status": "SUCCEEDED",
                "result": "XHIGH_OK " + ("R" * 300),
                "partial_result": "",
                "metadata": {"mode": "xhigh", "terminal_state": "extracted"},
            }

    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(chatgpt_execution_backend="remote"),
        mode="xhigh",
    )
    monkeypatch.setattr(helper, "_async_client", lambda: FakeClient())
    answer, url, metadata = helper._remote_research(
        "P" * 500,
        run_state_path=tmp_path / "run.json",
        partial_path=tmp_path / "partial.md",
    )
    assert answer.startswith("XHIGH_OK")
    assert url == ""
    assert metadata["mode"] == "xhigh"
    assert helper.tool_name == "chatgptxhigh"


def test_remote_backend_honors_async_cancellation(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self):
            self.cancelled = []

        def submit(self, **_kwargs):
            return {"job_id": "job_cancelled"}

        def cancel(self, job_id):
            self.cancelled.append(job_id)
            return {"status": "CANCELLED"}

    client = FakeClient()
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(
            chatgpt_execution_backend="remote",
            chatgpt_async_api_url="https://broker.example",
            chatgpt_async_api_secret="test-secret",
        ),
        mode="xhigh",
    )
    monkeypatch.setattr(helper, "_async_client", lambda: client)
    event = threading.Event()
    event.set()
    token = set_cancellation_event(event)
    try:
        with pytest.raises(ChatGPTWebResearchError, match="cancelled"):
            helper._remote_research(
                "P" * 500,
                run_state_path=tmp_path / "run.json",
                partial_path=tmp_path / "partial.md",
            )
    finally:
        reset_cancellation_event(token)
    assert client.cancelled == ["job_cancelled"]


def test_xhigh_remote_backend_rolls_through_a_stale_broker_without_losing_real_mode(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self):
            self.submissions = []

        def submit(self, **kwargs):
            self.submissions.append(kwargs)
            if len(self.submissions) == 1:
                raise ChatGPTAsyncApiError(
                    "old broker",
                    status_code=400,
                    error_code="invalid_mode",
                )
            return {"job_id": "job_00000000-0000-0000-0000-000000000003"}

        def status(self, _job_id):
            return {"status": "SUCCEEDED", "stage": "extracted", "answer_chars": 300}

        def result(self, _job_id):
            return {
                "status": "SUCCEEDED",
                "result": "XHIGH_COMPAT_OK " + ("R" * 300),
                "metadata": {"mode": "xhigh", "terminal_state": "extracted"},
            }

    client = FakeClient()
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(chatgpt_execution_backend="remote"),
        mode="xhigh",
    )
    monkeypatch.setattr(helper, "_async_client", lambda: client)
    answer, _, metadata = helper._remote_research(
        "P" * 500,
        run_state_path=tmp_path / "run.json",
        partial_path=tmp_path / "partial.md",
    )
    assert answer.startswith("XHIGH_COMPAT_OK")
    assert metadata["mode"] == "xhigh"
    assert [call["mode"] for call in client.submissions] == ["xhigh", "pro"]
    assert client.submissions[1]["prompt"] == _XHIGH_COMPAT_PROMPT_PREFIX + ("P" * 500)
    assert client.submissions[0]["idempotency_key"] == client.submissions[1]["idempotency_key"]


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
    client.submit(mode="pro", prompt="P" * 100, idempotency_key="idem", output_timeout_seconds=5400)
    assert session.call is not None
    method, url, kwargs = session.call
    assert method == "POST"
    assert "known-secret" not in url
    assert kwargs["headers"]["authorization"] == "Bearer known-secret"
    assert "known-secret" not in json.dumps(kwargs["json"])
    assert kwargs["json"]["output_timeout_seconds"] == 5400
    assert kwargs["allow_redirects"] is False


def test_async_client_rejects_non_origin_or_credential_bearing_urls():
    for url in (
        "http://broker.example",
        "https://user:pass@broker.example",
        "https://broker.example/path",
        "https://broker.example?next=elsewhere",
    ):
        with pytest.raises(ValueError, match="clean HTTPS origin"):
            ChatGPTAsyncApiClient(url, "known-secret")


def test_deep_connector_discovery_accepts_current_hyphenated_target_url(monkeypatch):
    target = {
        "type": "iframe",
        "id": "connector-target",
        "parentId": "parent-target",
        "url": "https://connector-openai-deep-research.web-sandbox.oaiusercontent.com/",
        "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/connector-target",
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps([target]).encode()

    helper = ChatGPTWebResearchAgentTool(ToolsConfig(), mode="deep")
    monkeypatch.setattr(
        "chack_tools.chatgpt_research_agents.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    assert helper._deep_connector_target("parent-target", timeout_seconds=1) == target


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


def test_running_state_accepts_searching_the_web_label():
    class Locator:
        def count(self):
            return 1

    class Page:
        def get_by_role(self, _role, name):
            return Locator() if name.search("Searching the web") else type("Empty", (), {"count": lambda self: 0})()

    assert ChatGPTWebResearchAgentTool._is_running(Page()) is True


def test_pro_retries_one_provider_generation_error_within_original_deadline(monkeypatch, tmp_path):
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(chatgpt_pro_timeout_seconds=120, chatgpt_research_poll_seconds=2),
        mode="pro",
    )
    clock = {"now": 0.0}
    retries = []

    class Page:
        url = "https://chatgpt.com/c/provider-retry"

        def wait_for_timeout(self, milliseconds):
            clock["now"] += milliseconds / 1000.0

    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        helper,
        "_click_provider_retry_if_present",
        lambda _page: retries.append(clock["now"]) is None if not retries else False,
    )
    monkeypatch.setattr(helper, "_click_answer_now_if_present", lambda _page: False)
    monkeypatch.setattr(helper, "_longest_answer", lambda _page: "R" * 500 if retries else "")
    monkeypatch.setattr(helper, "_is_running", lambda _page: False)
    run_state = tmp_path / "chatgpt-run.json"

    answer = helper._wait_and_extract(Page(), run_state_path=run_state)

    assert answer == "R" * 500
    assert retries == [0.0]
    assert clock["now"] == 6.0
    state = json.loads(run_state.read_text())
    assert state["provider_retry_count"] == 1


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
    helper = ChatGPTWebResearchAgentTool(ToolsConfig(chatgpt_execution_backend="local"), mode="pro")
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
    assert payload["full_research_review"] == "Recovered partial evidence"
    assert payload["overall_summary"] == "Recovered partial evidence"
    assert {row["filename"] for row in payload["key_artifacts"]} == {
        "chatgpt-run.json",
        "chatgpt-request.md",
        "chatgpt-pro-partial.md",
    }
    assert all(row["source_url"] == "https://chatgpt.com/c/recoverable" for row in payload["key_artifacts"])
