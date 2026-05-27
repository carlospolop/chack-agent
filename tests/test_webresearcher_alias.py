from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig


def test_webresearcher_enabled_alias_exposes_websearcher_tool():
    toolset = AgentsToolset(
        ToolsConfig(webresearcher_enabled=True),
        model_provider="codex",
        default_model="gpt-5.4-mini",
    )

    names = {tool.name for tool in toolset.tools}

    assert "websearcher_research" in names
