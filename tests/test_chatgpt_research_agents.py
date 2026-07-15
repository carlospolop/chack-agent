import json
import re
from pathlib import Path

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.chatgpt_research_agents import ChatGPTWebResearchAgentTool
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
    assert payload["evidence_data_path"] == str(evidence)
    assert payload["final_research_review"] == "A" * 2500
    assert {row["filename"] for row in payload["key_artifacts"]} == {
        "chatgpt-pro-response.md",
        "chatgpt-request.md",
        "chatgpt-run.json",
    }
    assert (evidence / "chatgpt-pro-response.md").read_text() == "A" * 2500


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


def test_pro_timeout_forces_answer_now_and_extracts_stable_result(monkeypatch, tmp_path):
    helper = ChatGPTWebResearchAgentTool(
        ToolsConfig(
            chatgpt_research_timeout_seconds=60,
            chatgpt_research_poll_seconds=2,
            chatgpt_force_answer_grace_seconds=60,
        ),
        mode="pro",
    )
    ticks = iter([0.0, 61.0, 61.0, 62.0, 62.0, 63.0, 63.0])
    clicked = []

    class Page:
        url = "https://chatgpt.com/c/forced-answer"

        def wait_for_timeout(self, _milliseconds):
            return None

    monkeypatch.setattr("chack_tools.chatgpt_research_agents.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr(helper, "_click_answer_now_if_present", lambda _page: clicked.append(True) or True)
    monkeypatch.setattr(helper, "_longest_answer", lambda _page: "R" * 300)
    monkeypatch.setattr(helper, "_is_running", lambda _page: False)
    run_state = tmp_path / "chatgpt-run.json"
    partial = tmp_path / "partial.md"

    answer = helper._wait_and_extract(Page(), partial_path=partial, run_state_path=run_state)

    assert answer == "R" * 300
    assert clicked == [True]
    assert partial.read_text() == "R" * 300
    assert json.loads(run_state.read_text())["forced_answer"] is True


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
