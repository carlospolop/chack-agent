from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.subagent_config import normalize_subagent_prompts
from chack_tools.websearcher_agent import WebSearcherAgentTool


def test_webresearcher_enabled_alias_exposes_websearcher_tool():
    toolset = AgentsToolset(
        ToolsConfig(webresearcher_enabled=True),
        model_provider="codex",
        default_model="gpt-5.4-mini",
    )

    names = {tool.name for tool in toolset.tools}

    assert "websearcher_research" in names


def test_subagent_prompt_validation_rejection_is_not_logged_as_error():
    prompts, error = normalize_subagent_prompts("too short", min_chars=300)

    assert prompts == []
    assert error.startswith("INPUT_REJECTED:")
    assert "ERROR" not in error


def test_websearcher_subagent_includes_own_serpapi_tools(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    helper = WebSearcherAgentTool(
        ToolsConfig(task_steps_manager_enabled=False),
        model_provider="openai",
        fallback_model="gpt-test",
    )

    names = {getattr(tool, "name", "") for tool in helper._build_subagent_tools()}

    assert "search_google_web" in names
    assert "search_bing_web" in names
    assert "search_google_ai_mode" in names
    assert "search_bing_copilot" in names
