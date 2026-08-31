import io
import json
from collections import Counter
from types import SimpleNamespace

import chack_agent.backends.codex_backend as codex_backend_module
from chack_agent.agent import Chack, _build_initial_system_prompt
from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor
from chack_agent.backends.codex_backend import CodexExecutor
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.native_planning import (
    sync_claude_native_task,
    sync_native_plan_snapshot,
)
from chack_tools.task_steps_manager_state import (
    STORE,
    reset_active_context,
    set_active_context,
)


def _tool_names(toolset: AgentsToolset) -> set[str]:
    return {
        str(getattr(tool, "name", "") or getattr(tool, "__name__", ""))
        for tool in toolset.tools
    }


def test_native_backends_do_not_receive_chack_task_manager() -> None:
    config = ToolsConfig(task_steps_manager_enabled=True)

    assert "task_steps_manager" in _tool_names(
        AgentsToolset(config, model_provider="openai", default_model="test")
    )
    assert "task_steps_manager" not in _tool_names(
        AgentsToolset(config, model_provider="codex", default_model="test")
    )
    assert "task_steps_manager" not in _tool_names(
        AgentsToolset(config, model_provider="claude", default_model="test")
    )
    assert "task_steps_manager" not in _tool_names(
        AgentsToolset(config, model_provider="claude-code", default_model="test")
    )


def test_forced_planning_prompt_uses_backend_native_tool_names() -> None:
    codex = _build_initial_system_prompt(
        task_steps_manager_enabled=True,
        require_task_steps_manager_init_first=True,
        native_task_planning_backend="codex",
    )
    claude = _build_initial_system_prompt(
        task_steps_manager_enabled=True,
        require_task_steps_manager_init_first=True,
        native_task_planning_backend="claude_code",
    )

    assert "built-in `update_plan`" in codex
    assert "task_steps_manager action=init" not in codex
    assert "`TodoWrite` or `TaskCreate`/`TaskUpdate`" in claude
    assert "task_steps_manager action=init" not in claude


def test_an_agent_that_does_not_require_the_tasklist_is_not_told_to_keep_one() -> None:
    # Otherwise every such agent spends tool rounds maintaining a list nobody asked
    # for. The strong wording belongs only to runs that actually require it.
    forced = _build_initial_system_prompt(
        task_steps_manager_enabled=True,
        require_task_steps_manager_init_first=True,
    )
    optional = _build_initial_system_prompt(
        task_steps_manager_enabled=True,
        require_task_steps_manager_init_first=False,
    )
    disabled = _build_initial_system_prompt(
        task_steps_manager_enabled=False,
        require_task_steps_manager_init_first=False,
    )

    assert "keep updating this task list" in forced
    for prompt in (optional, disabled):
        assert "keep updating this task list" not in prompt
        assert "think through the steps the task will require" in prompt


def test_codex_native_snapshot_notifies_common_plan_listener() -> None:
    session_id = "native-codex-snapshot"
    STORE.create_session(session_id, title="Agent Plan")
    updates: list[str] = []
    STORE.register_listener(session_id, updates.append)
    tokens = set_active_context(session_id, "Run 1")
    try:
        assert sync_native_plan_snapshot(
            [
                {"text": "Inspect repository", "completed": True},
                {"text": "Implement fix", "completed": False},
                {"text": "Run tests", "completed": False},
            ],
            source="codex:update_plan",
            infer_current=True,
        )
        first_count = len(updates)
        # Repeated item.started/item.updated payloads must not spam chat edits.
        sync_native_plan_snapshot(
            [
                {"text": "Inspect repository", "completed": True},
                {"text": "Implement fix", "completed": False},
                {"text": "Run tests", "completed": False},
            ],
            source="codex:update_plan",
            infer_current=True,
        )
    finally:
        reset_active_context(tokens)
        STORE.unregister_listener(session_id, updates.append)

    snapshot = STORE.snapshot(session_id)
    tasks = snapshot["runs"][0]["tasks"]
    assert [task["status"] for task in tasks] == ["done", "doing", "todo"]
    assert len(updates) == first_count == 1
    assert "Agent Plan" in updates[-1]
    assert "Implement fix" in updates[-1]


def test_claude_todowrite_and_task_deltas_notify_common_plan_listener() -> None:
    session_id = "native-claude-plan"
    STORE.create_session(session_id, title="Agent Plan")
    updates: list[str] = []
    STORE.register_listener(session_id, updates.append)
    tokens = set_active_context(session_id, "Run 1")
    try:
        assert sync_claude_native_task(
            "TodoWrite",
            {
                "todos": [
                    {"content": "Inspect inputs", "status": "completed"},
                    {"content": "Apply changes", "status": "in_progress"},
                ]
            },
            status="success",
        )
        assert sync_claude_native_task(
            "TaskCreate",
            {"subject": "Verify deployment", "description": "Run live checks"},
            status="success",
            result="Task #3 created successfully",
        )
        assert sync_claude_native_task(
            "TaskUpdate",
            {"taskId": "3", "status": "completed"},
            status="success",
        )
    finally:
        reset_active_context(tokens)
        STORE.unregister_listener(session_id, updates.append)

    tasks = STORE.snapshot(session_id)["runs"][0]["tasks"]
    assert [task["text"] for task in tasks] == [
        "Inspect inputs",
        "Apply changes",
        "Verify deployment",
    ]
    assert [task["status"] for task in tasks] == ["done", "doing", "done"]
    assert len(updates) == 3
    assert "Verify deployment" in updates[-1]


def test_claude_executor_mirrors_native_tool_result() -> None:
    executor = ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=0,
        _memory_reset_to=0,
        _base_system_prompt="",
        _model_name="test",
        _max_turns=1,
        _claude_cli_path="claude",
        _tools_config_json="{}",
        _allowed_tools_json="[]",
        _serialized_tools_override_b64="",
        _serialized_tools_append_b64="",
        _model_provider="claude",
        _default_model="",
        _social_network_model="",
        _scientific_model="",
        _websearcher_model="",
        _business_model="",
        _product_model="",
        _legal_model="",
        _data_statistics_model="",
        _news_media_model="",
        _knowledge_graph_model="",
        _religious_model="",
        _cli_model="",
        _subchack_model="",
        _researcher_administrator_model="",
        _social_network_max_turns=0,
        _scientific_max_turns=0,
        _websearcher_max_turns=0,
        _business_max_turns=0,
        _product_max_turns=0,
        _legal_max_turns=0,
        _data_statistics_max_turns=0,
        _news_media_max_turns=0,
        _knowledge_graph_max_turns=0,
        _religious_max_turns=0,
        _cli_max_turns=0,
        _subchack_max_turns=0,
        _researcher_administrator_max_turns=0,
        _min_tools_used=0,
        _max_tools_used=0,
        _require_task_steps_manager_init_first=False,
        _output_schema_json="",
        _native_task_planning_backend="claude",
    )
    session_id = "claude-executor-plan"
    STORE.create_session(session_id, title="Agent Plan")
    tokens = set_active_context(session_id, "Run 1")
    calls: dict[str, tuple[str, object]] = {}
    steps: list[tuple[object, object]] = []
    try:
        executor._record_tool_use(
            {
                "id": "tool-1",
                "name": "TodoWrite",
                "input": {"todos": [{"content": "Finish work", "status": "in_progress"}]},
            },
            calls,
        )
        executor._record_tool_result(
            {"tool_use_id": "tool-1", "content": "Todos updated"},
            calls,
            steps,
        )
    finally:
        reset_active_context(tokens)

    assert STORE.snapshot(session_id)["current_task"] == "Finish work"
    assert steps[0][0].tool == "TodoWrite"


def test_backend_compose_prompts_use_native_planning_policy() -> None:
    codex = SimpleNamespace(
        _prompt_only_next_invocation=False,
        _base_system_prompt="",
        _native_task_planning_backend="codex",
        _require_native_plan_first=True,
        _require_task_steps_manager_init_first=False,
        _min_tools_used=0,
        _max_tools_used=0,
    )
    claude = SimpleNamespace(
        _prompt_only_next_invocation=False,
        _base_system_prompt="",
        _native_task_planning_backend="claude",
        _require_native_plan_first=True,
        _require_task_steps_manager_init_first=False,
        _allowed_tools_json="[]",
        _min_tools_used=0,
        _max_tools_used=0,
        _output_schema_json="",
        _output_schema_name="",
        _output_schema_strict=True,
        _has_save_vulnerability_tool=lambda: False,
    )

    codex_prompt = CodexExecutor._compose_prompt(codex, "do work")
    claude_prompt = ClaudeCodeExecutor._compose_prompt(claude, "do work")
    assert codex_prompt == "do work"
    assert claude_prompt == "do work"
    assert "built-in `update_plan`" in codex._cacheable_developer_prompt
    assert "`TodoWrite` or `TaskCreate`/`TaskUpdate`" in claude._cacheable_system_prompt
    assert "mcp__chack_tools__task_steps_manager" not in claude._cacheable_system_prompt


def test_native_planning_calls_do_not_count_as_non_task_tools() -> None:
    counts = Counter(
        {
            "TodoWrite": 2,
            "TaskCreate": 1,
            "TaskUpdate": 3,
            "mcp__chack_tools__task_steps_manager": 1,
            "update_plan": 2,
            "EnterPlanMode": 1,
            "ExitPlanMode": 1,
            "mcp__chack_tools__check_budget_status": 3,
            "Read": 4,
        }
    )

    assert Chack._non_task_tool_count_from_counter(counts) == 4


def test_non_tool_steps_and_empty_counter_entries_do_not_consume_tool_budget() -> None:
    agent = object.__new__(Chack)
    steps = [
        {"output": "assistant text"},
        {"tool": ""},
        "plain model response",
        SimpleNamespace(output="another model response"),
        {"tool": "web_search", "output": "result"},
    ]
    counts = Counter(
        {
            "": 20,
            "task_steps_manager": 3,
            "update_plan": 2,
            "web_search": 1,
        }
    )

    assert agent._non_task_tool_count(steps) == 1
    assert Chack._non_task_tool_count_from_counter(counts) == 1


def test_codex_stream_item_updates_reach_live_plan_callback(monkeypatch) -> None:
    events = [
        {
            "type": "item.started",
            "item": {
                "type": "todo_list",
                "items": [
                    {"text": "Inspect", "completed": False},
                    {"text": "Implement", "completed": False},
                ],
            },
        },
        {
            "type": "item.updated",
            "item": {
                "type": "todo_list",
                "items": [
                    {"text": "Inspect", "completed": True},
                    {"text": "Implement", "completed": False},
                ],
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "todo_list",
                "items": [
                    {"text": "Inspect", "completed": True},
                    {"text": "Implement", "completed": True},
                ],
            },
        },
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 1}},
    ]

    class _Process:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
            self.pid = 999_999

        def poll(self):
            assert self.stdout is not None
            return 0 if self.stdout.tell() >= len(self.stdout.getvalue()) else None

        @staticmethod
        def wait() -> int:
            return 0

    process = _Process()
    monkeypatch.setattr(codex_backend_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        codex_backend_module,
        "_readline_when_ready",
        lambda stream, timeout: stream.readline(),
    )
    monkeypatch.setattr(codex_backend_module, "register_process", lambda *args: object())
    monkeypatch.setattr(codex_backend_module, "unregister_process", lambda *args: None)
    monkeypatch.setattr(codex_backend_module, "cancellation_requested", lambda: False)
    monkeypatch.setattr(codex_backend_module, "report_live_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(codex_backend_module, "_resolve_codex_exec_timeout", lambda *args: 60)
    monkeypatch.setattr(codex_backend_module, "_resolve_codex_exec_cwd", lambda *args: "")

    executor = object.__new__(CodexExecutor)
    executor._build_command = lambda: ["codex"]
    executor._build_env = lambda: {}
    executor._runtime_env = lambda: {}
    executor._sub_action = "test"
    executor._model_name = "test"
    executor._model_provider = "codex"
    executor._thread_id = ""
    executor._native_task_planning_backend = "codex"
    executor._use_codex_access_token = False
    executor._fallback_openai_api_key = ""
    executor._log_tool_called = lambda *args, **kwargs: None
    executor._maybe_retry_with_api_key = lambda prompt, result, *args, **kwargs: result

    session_id = "codex-stream-native-plan"
    STORE.create_session(session_id, title="Agent Plan")
    updates: list[str] = []
    STORE.register_listener(session_id, updates.append)
    tokens = set_active_context(session_id, "Run 1")
    try:
        output, steps, _raw = executor._run_codex_once(
            "test prompt",
            allow_api_key_fallback=False,
        )
    finally:
        reset_active_context(tokens)

    assert output == "done"
    assert [task["status"] for task in STORE.snapshot(session_id)["runs"][0]["tasks"]] == [
        "done",
        "done",
    ]
    assert len(updates) == 3
    assert len(steps) == 1
    assert steps[0][0].tool == "task_steps_manager"
