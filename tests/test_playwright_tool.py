from types import SimpleNamespace

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
import chack_tools.agents_toolset as agents_toolset_module


def test_agents_toolset_registers_playwright_when_enabled_and_available(monkeypatch):
    monkeypatch.setattr(
        agents_toolset_module,
        "TaskStepsManagerTool",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        agents_toolset_module,
        "get_task_steps_manager_tool",
        lambda helper: SimpleNamespace(name="task_steps_manager"),
    )
    monkeypatch.setattr(
        agents_toolset_module,
        "PlaywrightFetchTool",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        agents_toolset_module,
        "get_playwright_fetch_tool",
        lambda helper: SimpleNamespace(name="playwright_fetch"),
    )
    monkeypatch.setattr(
        agents_toolset_module,
        "is_playwright_available",
        lambda: True,
    )

    toolset = AgentsToolset(
        ToolsConfig(playwright_enabled=True),
        model_provider="openai",
    )

    assert [tool.name for tool in toolset.tools] == [
        "task_steps_manager",
        "playwright_fetch",
    ]


def test_agents_toolset_skips_playwright_when_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(
        agents_toolset_module,
        "TaskStepsManagerTool",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        agents_toolset_module,
        "get_task_steps_manager_tool",
        lambda helper: SimpleNamespace(name="task_steps_manager"),
    )
    monkeypatch.setattr(
        agents_toolset_module,
        "is_playwright_available",
        lambda: False,
    )

    toolset = AgentsToolset(
        ToolsConfig(playwright_enabled=True),
        model_provider="openai",
    )

    assert [tool.name for tool in toolset.tools] == ["task_steps_manager"]
