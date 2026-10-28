from __future__ import annotations

from types import SimpleNamespace

from chack_agent.backends import codex_backend
from chack_agent.backends.codex_backend import CodexExecutor
from chack_agent.backends.prompt_cache import PROMPT_CACHE_BREAKPOINT


def _executor(base: str = "system rules") -> SimpleNamespace:
    return SimpleNamespace(
        _prompt_only_next_invocation=False,
        _base_system_prompt=base,
        _native_task_planning_backend="",
        _require_native_plan_first=False,
        _require_task_steps_manager_init_first=False,
        _min_tools_used=0,
        _max_tools_used=0,
    )


def test_small_stable_prefix_stays_in_the_developer_argument() -> None:
    executor = _executor()
    prompt = CodexExecutor._compose_prompt(executor, f"catalog{PROMPT_CACHE_BREAKPOINT}round 1")
    assert prompt == "round 1"
    assert executor._cacheable_developer_prompt == "system rules\ncatalog"


def test_stable_prefix_above_the_linux_argument_limit_goes_on_stdin() -> None:
    # Escaped newlines double in json.dumps, so this fits as text but not as the argument.
    catalog = "route\n" * 20000
    assert len(catalog.encode()) < codex_backend._MAX_ARGUMENT_BYTES
    executor = _executor()
    prompt = CodexExecutor._compose_prompt(executor, f"{catalog}{PROMPT_CACHE_BREAKPOINT}round 1")
    assert prompt == f"{catalog}round 1"
    assert executor._cacheable_developer_prompt == "system rules"
    assert codex_backend._fits_in_argument(executor._cacheable_developer_prompt)


def test_system_prompt_above_the_linux_argument_limit_goes_on_stdin() -> None:
    executor = _executor(base="rule " * 30000)
    prompt = CodexExecutor._compose_prompt(executor, "do work")
    assert executor._cacheable_developer_prompt == ""
    assert prompt.startswith("rule rule") and prompt.endswith("\n\ndo work")
