from __future__ import annotations

import inspect
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI
from agents import Agent, ModelSettings, Runner, ToolGuardrailFunctionOutput, tool_input_guardrail
from agents.items import ToolCallItem

from ..config import ChackConfig
from chack.conversation_memory import build_memory_summary, format_messages
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_list_state import current_run_label, current_session_id
from chack_tools.tool_usage_state import (
    STORE as TOOL_USAGE_STORE,
    current_max_tools_used,
    current_usage_session_id,
    non_task_tool_count,
)


_FIRST_TOOL_LOCK = threading.Lock()
_FIRST_TOOL_INIT_DONE: dict[str, bool] = {}
_FIRST_TOOL_STATE_MAX = 5000
_LOGGER = logging.getLogger("chack.openai_agents_backend")


def _run_scope_key() -> str:
    session_id = current_session_id() or "no-session"
    run_label = current_run_label() or "Run 1"
    return f"{session_id}:{run_label}"


def _item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type", "") or "")
    return str(getattr(item, "type", "") or "")


def _item_call_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("call_id", "") or "")
    return str(getattr(item, "call_id", "") or "")


def _sanitize_input_items(items: list[Any]) -> list[Any]:
    # Keep function call/output pairs consistent to avoid Responses API 400s when
    # history truncation drops one side of the pair.
    call_ids = set()
    for item in items:
        item_type = _item_type(item)
        if item_type in {"function_call", "tool_call"}:
            call_id = _item_call_id(item)
            if call_id:
                call_ids.add(call_id)

    sanitized: list[Any] = []
    for item in items:
        item_type = _item_type(item)
        if item_type == "function_call_output":
            call_id = _item_call_id(item)
            if call_id and call_id not in call_ids:
                continue
        sanitized.append(item)
    return sanitized


def _is_message_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") == "message":
        return True
    role = item.get("role")
    if role in {"user", "assistant", "system", "developer"} and "content" in item:
        return True
    return False


def _filter_message_items(items: list[Any]) -> list[Any]:
    return [item for item in items if _is_message_item(item)]


def _is_first_tool_gate_open() -> bool:
    key = _run_scope_key()
    with _FIRST_TOOL_LOCK:
        return bool(_FIRST_TOOL_INIT_DONE.get(key))


def _open_first_tool_gate() -> None:
    key = _run_scope_key()
    with _FIRST_TOOL_LOCK:
        _FIRST_TOOL_INIT_DONE[key] = True
        # Keep memory bounded; keys are per-run and naturally high-churn.
        if len(_FIRST_TOOL_INIT_DONE) > _FIRST_TOOL_STATE_MAX:
            _FIRST_TOOL_INIT_DONE.clear()


@tool_input_guardrail(name="require_task_list_init_first")
def _require_task_list_init_first(data) -> ToolGuardrailFunctionOutput:
    run_label = (current_run_label() or "").strip().lower()
    if "self-critique" in run_label or "self critique" in run_label:
        return ToolGuardrailFunctionOutput.allow()

    if _is_first_tool_gate_open():
        return ToolGuardrailFunctionOutput.allow()

    reminder = (
        "First tool call of this run must be task_list with action=init. "
        "Call task_list init first before any other tool indicating the initial task plan for this run, "
        "so that you can keep track of your progress and next steps effectively. "
        "Note that if in the future you need to modify/update the task list based on new knowledge, you can "
        "do so by calling the task_list tool with the appropriate action and providing any relevant notes about the update. "
    )
    tool_name = str(getattr(data.context, "tool_name", "") or "").strip().lower()
    if tool_name != "task_list":
        return ToolGuardrailFunctionOutput.reject_content(reminder)

    raw_args = getattr(data.context, "tool_arguments", "") or ""
    try:
        payload = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
    except Exception:
        payload = {}
    action = ""
    if isinstance(payload, dict):
        action = str(payload.get("action", "")).strip().lower()
    if action != "init":
        return ToolGuardrailFunctionOutput.reject_content(reminder)

    _open_first_tool_gate()
    return ToolGuardrailFunctionOutput.allow()


@tool_input_guardrail(name="respect_max_tools_used")
def _respect_max_tools_used(data) -> ToolGuardrailFunctionOutput:
    tool_name = str(getattr(data.context, "tool_name", "") or "").strip().lower()
    if tool_name.startswith("task_list"):
        return ToolGuardrailFunctionOutput.allow()

    max_tools_used = current_max_tools_used()
    if max_tools_used <= 0:
        return ToolGuardrailFunctionOutput.allow()

    session_id = current_usage_session_id()
    if not session_id:
        return ToolGuardrailFunctionOutput.allow()

    used = non_task_tool_count(TOOL_USAGE_STORE.snapshot(session_id))
    if used >= max_tools_used:
        print(f"Tool usage limit reached: used {used} tools, max is {max_tools_used}. Rejecting tool call to {tool_name}.")
        return ToolGuardrailFunctionOutput.reject_content(
            "Tool budget reached. Please use the information already gathered and finish the execution."
        )
    return ToolGuardrailFunctionOutput.allow()


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class AgentsExecutor:
    _config: ChackConfig
    agent: Agent
    max_turns: int
    _conversation: list[dict[str, Any]]
    _memory_limit: int
    _memory_reset_to: int
    _summary_text: str
    _summary_max_chars: int
    _base_system_prompt: str
    _previous_response_id: Optional[str]
    _compaction_threshold_ratio: float
    _max_context_tokens: int
    _compaction_model: str

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_input = payload.get("input", "")
        if self._summary_text:
            self.agent.instructions = (
                f"{self._base_system_prompt}\n\n### MEMORY SUMMARY\n{self._summary_text}"
            )
        else:
            self.agent.instructions = self._base_system_prompt
        input_items: list[dict[str, Any]] = []
        if self._previous_response_id:
            if user_input:
                input_items.append({"role": "user", "content": user_input})
        else:
            input_items = list(self._conversation)
            if user_input:
                input_items.append({"role": "user", "content": user_input})
        input_items = _sanitize_input_items(input_items)
        result = Runner.run_sync(
            self.agent,
            input_items,
            max_turns=self.max_turns,
            previous_response_id=self._previous_response_id,
        )
        output = result.final_output or ""
        updated_transcript = result.to_input_list()
        if isinstance(updated_transcript, list) and updated_transcript:
            message_items = _filter_message_items(updated_transcript)
            if message_items:
                self._conversation = message_items
        else:
            if user_input:
                self._conversation.append({"role": "user", "content": user_input})
            if output:
                self._conversation.append({"role": "assistant", "content": output})
        if self._memory_limit:
            if len(self._conversation) > self._memory_limit:
                reset_to = self._memory_reset_to or self._memory_limit
                if reset_to > self._memory_limit:
                    reset_to = self._memory_limit
                if reset_to < 1:
                    reset_to = 1
                removed = self._conversation[:-reset_to]
                self._conversation = self._conversation[-reset_to:]
                if removed:
                    conversation = format_messages(removed)
                    self._summary_text = build_memory_summary(
                        self._config,
                        conversation,
                        self._summary_text,
                        self._summary_max_chars,
                    )
        if result.last_response_id:
            self._previous_response_id = result.last_response_id
        self._maybe_compact(result)
        steps = _extract_tool_steps(result.new_items)
        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": result,
        }

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)

    def _maybe_compact(self, result: Any) -> None:
        if not self._previous_response_id:
            return
        if self._max_context_tokens <= 0:
            return
        if self._compaction_threshold_ratio <= 0:
            return

        input_tokens = 0
        raw_responses = getattr(result, "raw_responses", None)
        if raw_responses:
            last_response = raw_responses[-1]
            usage = getattr(last_response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)

        if not input_tokens:
            return
        if input_tokens < int(self._compaction_threshold_ratio * self._max_context_tokens):
            return

        new_response_id = self._run_compaction(self._previous_response_id)
        if new_response_id:
            self._previous_response_id = new_response_id

    def _run_compaction(self, response_id: str) -> Optional[str]:
        try:
            client = OpenAI()
            compacted = client.responses.compact(
                model=self._compaction_model,
                previous_response_id=response_id,
            )
            response_id = getattr(compacted, "id", None) or getattr(
                compacted, "response_id", None
            )
            return response_id
        except Exception:
            _LOGGER.exception("Responses compaction failed.")
            return None


def _extract_tool_steps(items: list[Any]) -> list[tuple[ToolAction, Any]]:
    steps: list[tuple[ToolAction, Any]] = []
    for item in items:
        if not isinstance(item, ToolCallItem):
            continue
        raw = item.raw_item
        tool_name = _get_tool_name(raw) or "tool"
        tool_input = _get_tool_input(raw)
        steps.append((ToolAction(tool=tool_name, tool_input=tool_input), None))
    return steps


def _get_tool_name(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if hasattr(raw, "name"):
        return getattr(raw, "name", None)
    if hasattr(raw, "function"):
        func = getattr(raw, "function", None)
        if func and hasattr(func, "name"):
            return getattr(func, "name", None)
    if isinstance(raw, dict):
        name = raw.get("name")
        if name:
            return name
        func = raw.get("function", {})
        if isinstance(func, dict):
            return func.get("name")
    return None


def _get_tool_input(raw: Any) -> Any:
    if raw is None:
        return None
    if hasattr(raw, "arguments"):
        return getattr(raw, "arguments", None)
    if hasattr(raw, "input"):
        return getattr(raw, "input", None)
    if hasattr(raw, "function"):
        func = getattr(raw, "function", None)
        if func and hasattr(func, "arguments"):
            return getattr(func, "arguments", None)
    if isinstance(raw, dict):
        if "arguments" in raw:
            return raw.get("arguments")
        if "input" in raw:
            return raw.get("input")
        func = raw.get("function", {})
        if isinstance(func, dict):
            return func.get("arguments") or func.get("input")
    return None


def _apply_guardrails(tools: list[Any]) -> list[Any]:
    for tool in tools:
        guards = getattr(tool, "tool_input_guardrails", None)
        if guards is None:
            setattr(tool, "tool_input_guardrails", [_require_task_list_init_first])
        elif _require_task_list_init_first not in guards:
            guards.append(_require_task_list_init_first)
    return tools


def build_executor(
    config: ChackConfig,
    *,
    system_prompt: str,
    max_turns: int,
    memory_max_messages: int,
    memory_reset_to_messages: int,
    memory_summary_prompt: str,
    summary_max_chars: int,
    tool_profile: str = "all",
    tools_override: Optional[list[Any]] = None,
    tools_append: Optional[list[Any]] = None,
) -> AgentsExecutor:
    model_name = config.model.primary

    if tools_override is None:
        init_params = inspect.signature(AgentsToolset.__init__).parameters
        toolset_kwargs = {
            "tool_profile": tool_profile,
            "default_model": config.model.primary,
            "social_network_model": config.model.social_network,
            "scientific_model": config.model.scientific,
            "social_network_max_turns": config.model.social_network_max_turns,
            "scientific_max_turns": config.model.scientific_max_turns,
        }
        if "websearcher_model" in init_params:
            toolset_kwargs["websearcher_model"] = config.model.websearcher
        if "websearcher_max_turns" in init_params:
            toolset_kwargs["websearcher_max_turns"] = config.model.websearcher_max_turns
        if "tester_model" in init_params:
            toolset_kwargs["tester_model"] = config.model.tester
        if "tester_max_turns" in init_params:
            toolset_kwargs["tester_max_turns"] = config.model.tester_max_turns
        toolset = AgentsToolset(config.tools, **toolset_kwargs)
        tools = toolset.tools
        if tools_append:
            tools = list(tools) + list(tools_append)
    else:
        tools = list(tools_override)

    tools = _apply_guardrails(tools)
    agent = Agent(
        name="Chack",
        instructions=system_prompt,
        tools=tools,
        model=model_name,
        model_settings=ModelSettings(),
    )

    max_messages = memory_max_messages
    if max_messages < 1:
        max_messages = 1
    reset_to = memory_reset_to_messages
    if reset_to < 1 or reset_to > max_messages:
        reset_to = max_messages
    return AgentsExecutor(
        _config=config,
        agent=agent,
        max_turns=max_turns,
        _conversation=[],
        _memory_limit=max_messages,
        _memory_reset_to=reset_to,
        _summary_text="",
        _summary_max_chars=summary_max_chars,
        _base_system_prompt=system_prompt,
        _previous_response_id=None,
        _compaction_threshold_ratio=float(
            config.agent.compaction_threshold_ratio or 0.75
        ),
        _max_context_tokens=int(config.model.max_context_tokens or 0),
        _compaction_model=(
            str(config.agent.compaction_model).strip() or model_name
        ),
    )
