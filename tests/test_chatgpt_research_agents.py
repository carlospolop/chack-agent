import json
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
        lambda _prompt: (
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
