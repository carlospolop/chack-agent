from __future__ import annotations

import inspect
import asyncio
import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Callable, Optional

from openai import AsyncOpenAI, BadRequestError, RateLimitError
from agents import (
    Agent,
    ModelSettings,
    Runner,
    ToolGuardrailFunctionOutput,
    tool_input_guardrail,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.models.openai_responses import OpenAIResponsesModel
from agents.items import ToolCallItem
from agents.exceptions import ModelBehaviorError

from ..config import ChackConfig
from ..output_schema import JsonSchemaOutput
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_steps_manager_state import current_run_label, current_session_id
from chack_tools.telemetry import log_event, log_tool_started, log_tool_executed, log_tool_error
from chack_tools.tool_usage_state import (
    STORE as TOOL_USAGE_STORE,
    current_max_tools_used,
    current_usage_session_id,
    non_task_tool_count,
)


_FIRST_TOOL_LOCK = threading.Lock()
_FIRST_TOOL_INIT_DONE: dict[str, bool] = {}
_FIRST_TOOL_STATE_MAX = 5000
_LOGGER = logging.getLogger("chack.openrouter_openai_backend")

_OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class _OpenRouterResponsesModel(OpenAIResponsesModel):
    @staticmethod
    def _prepare_input_items(
        items: Any,
        previous_response_id: str | None,
    ) -> Any:
        if not isinstance(items, list):
            return items
        prepared = _sanitize_input_items(list(items))
        # Keep structured tool context even when previous_response_id is missing.
        # `_sanitize_input_items` already removes orphan function_call_output items.
        # Dropping to message-only here can make the model lose tool state and
        # repeatedly restart planning/tool calls in the same run.
        if prepared:
            return prepared
        return [
            {
                "role": "user",
                "content": "Continue and provide the best possible answer based on current context.",
            }
        ]

    async def _get_response_with_retries(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema,
        handoffs,
        tracing,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt,
    ):
        max_attempts = 3
        delay_seconds = 3
        for attempt in range(1, max_attempts + 1):
            try:
                return await super().get_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
            except RateLimitError:
                if attempt >= max_attempts:
                    raise
                _LOGGER.warning(
                    "OpenRouter rate limited (attempt %s/%s). Retrying in %ss.",
                    attempt,
                    max_attempts,
                    delay_seconds,
                )
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 20)

    def _normalize_tool_name(self, name: str, tool_names: set[str]) -> str:
        if not name or not tool_names:
            return name
        if name in tool_names:
            return name
        candidate = name
        # OpenRouter/Gemini can prepend `tool_` multiple times.
        while candidate.startswith("tool_"):
            candidate = candidate[len("tool_") :]
            if candidate in tool_names:
                return candidate
        # OpenRouter (Gemini) may append a unique suffix to tool names.
        sorted_tools = sorted(tool_names, key=len, reverse=True)
        for tool_name in sorted_tools:
            if candidate.startswith(f"{tool_name}_"):
                return tool_name
            if name.startswith(f"{tool_name}_"):
                return tool_name
            if candidate.startswith(f"tool_{tool_name}_"):
                return tool_name
            if name.startswith(f"tool_{tool_name}_"):
                return tool_name
        return name

    @staticmethod
    def _is_sequence_recoverable_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return (
            "function response turn comes immediately after a function call turn" in err
            or ("invalid_argument" in err and "function call" in err and "function response" in err)
            or "not found in agent chack" in err
            or ("tool " in err and " not found in agent " in err)
        )

    def _normalize_output_tools(self, output_items: list[Any], tool_names: set[str]) -> None:
        for raw in output_items:
            current = None
            if hasattr(raw, "name"):
                current = getattr(raw, "name", None)
            elif hasattr(raw, "function"):
                func = getattr(raw, "function", None)
                if func is not None and hasattr(func, "name"):
                    current = getattr(func, "name", None)
            elif isinstance(raw, dict):
                current = raw.get("name")
                if current is None:
                    func = raw.get("function")
                    if isinstance(func, dict):
                        current = func.get("name")

            normalized = self._normalize_tool_name(str(current or ""), tool_names) if current else None
            if not normalized or normalized == current:
                continue

            if hasattr(raw, "name"):
                setattr(raw, "name", normalized)
                continue
            if hasattr(raw, "function"):
                func = getattr(raw, "function", None)
                if func is not None and hasattr(func, "name"):
                    setattr(func, "name", normalized)
                    continue
            if isinstance(raw, dict):
                if "name" in raw:
                    raw["name"] = normalized
                    continue
                func = raw.get("function")
                if isinstance(func, dict) and "name" in func:
                    func["name"] = normalized

    async def get_response(  # type: ignore[override]
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema,
        handoffs,
        tracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ):
        try:
            prepared_input = self._prepare_input_items(input, previous_response_id)
            response = await self._get_response_with_retries(
                system_instructions=system_instructions,
                input=prepared_input,
                model_settings=model_settings,
                tools=tools,
                output_schema=output_schema,
                handoffs=handoffs,
                tracing=tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )
        except BadRequestError as exc:
            should_retry = (
                previous_response_id is not None
                and self._is_sequence_recoverable_error(exc)
            )
            if not should_retry:
                raise
            _LOGGER.warning(
                "OpenRouter rejected response chain with previous_response_id; retrying without it."
            )
            fallback_input = self._prepare_input_items(input, None)
            response = await self._get_response_with_retries(
                system_instructions=system_instructions,
                input=fallback_input,
                model_settings=model_settings,
                tools=tools,
                output_schema=output_schema,
                handoffs=handoffs,
                tracing=tracing,
                previous_response_id=None,
                conversation_id=conversation_id,
                prompt=prompt,
            )
        tool_names = {getattr(tool, "name", "") for tool in tools if getattr(tool, "name", "")}
        if tool_names:
            self._normalize_output_tools(response.output, tool_names)
        return response

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema,
        handoffs,
        tracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ):
        tool_names = {getattr(tool, "name", "") for tool in tools if getattr(tool, "name", "")}
        async for event in super().stream_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        ):
            if tool_names and getattr(event, "type", "") == "response.completed":
                response = getattr(event, "response", None)
                if response is not None and getattr(response, "output", None):
                    self._normalize_output_tools(response.output, tool_names)
            yield event


def _select_provider(config: ChackConfig) -> str:
    provider = str(getattr(config.model, "provider", "") or "openai").strip().lower()
    if provider != "openrouter":
        raise ValueError(
            f"openrouter_openai_backend requires model.provider='openrouter' (got {provider!r})"
        )
    return provider


def _configure_openai_client(config: ChackConfig) -> tuple[str, Optional[AsyncOpenAI]]:
    provider = _select_provider(config)
    api_key = (
        str(config.credentials.openrouter_api_key or "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required when model.provider=openrouter")
    base_url = (
        str(config.credentials.openrouter_base_url or "").strip()
        or os.environ.get("OPENROUTER_BASE_URL", "").strip()
        or _OPENROUTER_DEFAULT_BASE_URL
    )
    headers: dict[str, str] = {}
    referer = (
        str(config.credentials.openrouter_http_referer or "").strip()
        or os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    )
    title = (
        str(config.credentials.openrouter_app_name or "").strip()
        or os.environ.get("OPENROUTER_APP_NAME", "").strip()
    )
    if not title:
        main_action = str(config.agent.main_action or "").strip()
        sub_action = str(config.agent.sub_action or "").strip()
        if main_action and sub_action:
            title = f"{main_action}-{sub_action}"
        elif main_action:
            title = main_action
        elif sub_action:
            title = sub_action
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=headers or None,
    )
    set_default_openai_client(client, use_for_tracing=False)
    # OpenRouter doesn't support OpenAI tracing endpoints.
    set_tracing_disabled(True)
    return provider, client


def _run_scope_key() -> str:
    session_id = current_session_id() or "no-session"
    run_label = current_run_label() or "Run 1"
    return f"{session_id}:{run_label}"


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_int(raw: str, default: int) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception:
        return default


def _item_type(item: Any) -> str:
    if isinstance(item, dict):
        # 'function' logic from some older/wrapper libs
        if item.get("role") == "tool":
            return "function_call_output"
        return str(item.get("type", "") or "")
    if getattr(item, "role", None) == "tool":
        return "function_call_output"
    return str(getattr(item, "type", "") or "")


def _item_call_id(item: Any) -> str:
    val = None
    if isinstance(item, dict):
        val = item.get("call_id") or item.get("id") or item.get("tool_call_id")
    else:
        val = (
            getattr(item, "call_id", None)
            or getattr(item, "id", None)
            or getattr(item, "tool_call_id", None)
        )
    return str(val or "")


def _sanitize_input_items(items: list[Any]) -> list[Any]:
    # Keep function call/output pairs consistent to avoid Responses API 400s when
    # history truncation drops one side of the pair.
    call_ids = set()
    has_idless_calls = False
    
    # Expanded set of known tool call types to ensure we track ALL calls.
    # This prevents dropping valid success outputs just because we missed the call type.
    _call_types = {
        "function",
        "function_call", 
        "tool_call",
        "computer_call",
        "computer_20241022", # Anthropic specific
        "file_search",
        "code_interpreter",
        "mcp_call",
        "local_shell_call",
    }

    for item in items:
        item_type = _item_type(item)
        if item_type in _call_types or item_type.endswith("_call"):
            call_id = _item_call_id(item)
            if call_id:
                call_ids.add(call_id)
            else:
                has_idless_calls = True

    sanitized: list[Any] = []
    for item in items:
        item_type = _item_type(item)
        
        # Check if this looks like a tool output
        is_output_type = (
            item_type == "function_call_output" 
            or item_type.endswith("_call_output")
            or item_type == "tool_output"
        )
        
        if is_output_type:
            call_id = _item_call_id(item)
            # If output has a specific ID, it generally must match a known call ID.
            # However, if we saw calls without IDs (e.g. from OpenRouter/Gemini),
            # we rely on order/wildcard matching and keep the output to be safe.
            if call_id and call_id not in call_ids and not has_idless_calls:
                continue
        sanitized.append(item)
    return sanitized


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


@tool_input_guardrail(name="require_task_steps_manager_init_first")
def _require_task_steps_manager_init_first(data) -> ToolGuardrailFunctionOutput:
    run_label = (current_run_label() or "").strip().lower()
    if "self-critique" in run_label or "self critique" in run_label:
        return ToolGuardrailFunctionOutput.allow()

    if _is_first_tool_gate_open():
        return ToolGuardrailFunctionOutput.allow()

    reminder = (
        "First tool call of this run must be task_steps_manager with action=init. "
        "Call task_steps_manager init first before any other tool indicating the initial task plan for this run, "
        "so that you can keep track of your progress and next steps effectively. "
        "Note that if in the future you need to modify/update the task list based on new knowledge, you can "
        "do so by calling the task_steps_manager tool with the appropriate action and providing any relevant notes about the update. "
    )
    tool_name = str(getattr(data.context, "tool_name", "") or "").strip().lower()
    raw_args = getattr(data.context, "tool_arguments", "") or ""
    if tool_name != "task_steps_manager":
        try:
            log_event(
                "tool_disallowed",
                payload={
                    "tool": tool_name or "unknown",
                    "tool_input": raw_args,
                    "reason": "first_tool_must_be_task_steps_manager_init",
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass
        return ToolGuardrailFunctionOutput.reject_content(reminder)

    try:
        payload = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
    except Exception:
        payload = {}
    action = ""
    if isinstance(payload, dict):
        action = str(payload.get("action", "")).strip().lower()
    if action != "init":
        try:
            log_event(
                "tool_disallowed",
                payload={
                    "tool": tool_name or "task_steps_manager",
                    "tool_input": raw_args,
                    "reason": "task_steps_manager_init_required",
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass
        return ToolGuardrailFunctionOutput.reject_content(reminder)

    _open_first_tool_gate()
    return ToolGuardrailFunctionOutput.allow()


@tool_input_guardrail(name="respect_max_tools_used")
def _respect_max_tools_used(data) -> ToolGuardrailFunctionOutput:
    tool_name = str(getattr(data.context, "tool_name", "") or "").strip().lower()
    if tool_name.startswith("task_steps_manager"):
        return ToolGuardrailFunctionOutput.allow()

    max_tools_used = current_max_tools_used()
    if max_tools_used <= 0:
        return ToolGuardrailFunctionOutput.allow()

    session_id = current_usage_session_id()
    if not session_id:
        return ToolGuardrailFunctionOutput.allow()

    used = non_task_tool_count(TOOL_USAGE_STORE.snapshot(session_id))
    if used >= max_tools_used:
        _LOGGER.warning(f"Tool usage limit reached: used {used} tools, max is {max_tools_used}. Rejecting tool call to {tool_name}.")
        try:
            log_event(
                "tool_disallowed",
                payload={
                    "tool": tool_name or "unknown",
                    "tool_input": getattr(data.context, "tool_arguments", "") or "",
                    "reason": "max_tools_used_reached",
                    "max_tools_used": max_tools_used,
                    "used": used,
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass
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
    agent: Agent
    _summary_agent: Agent
    max_turns: int
    _conversation: list[dict[str, Any]]
    _memory_limit: int
    _memory_reset_to: int
    _base_system_prompt: str
    _previous_response_id: Optional[str]
    _conversation_id: Optional[str]
    _summary: str
    _summary_trigger_messages: int
    _summary_keep_messages: int
    _summary_max_chars: int
    _compaction_threshold_ratio: float
    _max_context_tokens: int

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        user_input = payload.get("input", "")
        self._maybe_summarize_for_next_turn(input_tokens=0)
        self.agent.instructions = self._system_prompt()
        result = self._invoke_runner_with_recovery(user_input=user_input, context=context)
        output = result.final_output or ""
        updated_transcript = result.to_input_list()
        if isinstance(updated_transcript, list) and updated_transcript:
            # Keep the full transcript (tool calls + outputs + messages) so a
            # recovered run without previous_response_id can continue from the
            # same thread context instead of restarting from message-only state.
            transcript_items = _sanitize_input_items(updated_transcript)
            if transcript_items:
                self._conversation = transcript_items
        else:
            if user_input:
                self._conversation.append({"role": "user", "content": user_input})
            if output:
                self._conversation.append({"role": "assistant", "content": output})
        if self._memory_limit and len(self._conversation) > self._memory_limit:
            before_len = len(self._conversation)
            reset_to = self._memory_reset_to or self._memory_limit
            if reset_to > self._memory_limit:
                reset_to = self._memory_limit
            if reset_to < 1:
                reset_to = 1
            self._conversation = self._conversation[-reset_to:]
            after_len = len(self._conversation)
            trimmed = max(0, before_len - after_len)
            _LOGGER.info(
                "OpenRouter memory trim applied: before=%s after=%s trimmed=%s memory_limit=%s memory_reset_to=%s ts=%s",
                before_len,
                after_len,
                trimmed,
                self._memory_limit,
                self._memory_reset_to,
                _log_timestamp(),
            )
            log_event(
                "agent_memory_trimmed",
                payload={
                    "backend": "openrouter_openai",
                    "provider": "openrouter",
                    "before_messages": int(before_len),
                    "after_messages": int(after_len),
                    "trimmed_messages": int(trimmed),
                    "memory_limit": int(self._memory_limit or 0),
                    "memory_reset_to": int(self._memory_reset_to or 0),
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        if result.last_response_id:
            self._previous_response_id = result.last_response_id
        conversation_id = getattr(result, "_conversation_id", None)
        if isinstance(conversation_id, str) and conversation_id.strip():
            self._conversation_id = conversation_id.strip()
        self._maybe_summarize_for_next_turn(input_tokens=self._extract_input_tokens(result))
        steps = _extract_tool_steps(result.new_items)
        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": result,
        }

    def _extract_input_tokens(self, result: Any) -> int:
        try:
            raw_responses = getattr(result, "raw_responses", None)
            if not raw_responses:
                return 0
            last_response = raw_responses[-1]
            usage = getattr(last_response, "usage", None)
            return int(getattr(usage, "input_tokens", 0) or 0)
        except Exception:
            return 0

    def _system_prompt(self) -> str:
        base = str(self._base_system_prompt or "").strip()
        if not self._summary:
            return base
        summary_block = f"\n\n### MEMORY SUMMARY\n{self._summary}"
        return f"{base}{summary_block}".strip()

    def _conversation_to_summary_lines(self, items: list[Any]) -> str:
        rendered: list[str] = []
        for item in items:
            role = "assistant"
            content = ""
            if isinstance(item, dict):
                item_type = str(item.get("type", "") or "")
                role = str(item.get("role", "") or "").strip().lower() or "assistant"
                if role == "tool" or item_type.endswith("_call") or item_type.endswith("_call_output"):
                    role = "tool"
                if isinstance(item.get("content"), str):
                    content = str(item.get("content", "") or "")
                elif "arguments" in item:
                    content = json.dumps(item.get("arguments"), ensure_ascii=False)
                elif "output" in item:
                    content = str(item.get("output", "") or "")
                else:
                    content = json.dumps(item, ensure_ascii=False)
            else:
                content = str(item)
            rendered.append(f"[{role}] {content}")
        return "\n".join(rendered)

    def _summarize_items(
        self,
        previous_summary: str,
        items: list[Any],
        *,
        trigger_input_tokens: int = 0,
        threshold_tokens: int = 0,
        messages_before: int = 0,
        messages_after_keep: int = 0,
        triggered_by_messages: bool = False,
        triggered_by_tokens: bool = False,
    ) -> str:
        if not items:
            return previous_summary

        chunk = self._conversation_to_summary_lines(items)
        previous_summary_chars = len(previous_summary or "")
        source_messages_count = len(items)
        source_chars = len(chunk)
        did_truncate = False

        if previous_summary:
            prompt = (
                "You maintain compact conversation memory.\n"
                "Existing summary:\n"
                f"{previous_summary}\n\n"
                "New messages:\n"
                f"{chunk}\n\n"
                "Return an updated concise summary preserving key facts, decisions, and constraints."
            )
        else:
            prompt = (
                "Summarize the following conversation history concisely, preserving key facts, decisions, and constraints:\n"
                f"{chunk}"
            )

        result = Runner.run_sync(
            self._summary_agent,
            [{"role": "user", "content": prompt}],
            max_turns=1,
            previous_response_id=None,
            conversation_id=None,
            context=None,
        )
        summary = str(getattr(result, "final_output", "") or "").strip() or previous_summary
        if self._summary_max_chars > 0 and len(summary) > self._summary_max_chars:
            summary = summary[-self._summary_max_chars :]
            did_truncate = True

        _LOGGER.info(
            "OpenRouter summary compaction performed: messages_before=%s messages_summarized=%s messages_after_keep=%s trigger_input_tokens=%s threshold_tokens=%s source_chars=%s prev_summary_chars=%s new_summary_chars=%s truncated=%s keep_messages=%s trigger_messages=%s by_messages=%s by_tokens=%s ts=%s",
            messages_before,
            source_messages_count,
            messages_after_keep,
            trigger_input_tokens,
            threshold_tokens,
            source_chars,
            previous_summary_chars,
            len(summary),
            did_truncate,
            self._summary_keep_messages,
            self._summary_trigger_messages,
            triggered_by_messages,
            triggered_by_tokens,
            _log_timestamp(),
        )
        log_event(
            "agent_summary_compaction",
            payload={
                "backend": "openrouter_openai",
                "provider": "openrouter",
                "messages_summarized": int(source_messages_count),
                "source_chars": int(source_chars),
                "previous_summary_chars": int(previous_summary_chars),
                "new_summary_chars": int(len(summary)),
                "summary_max_chars": int(self._summary_max_chars),
                "summary_truncated": bool(did_truncate),
                "summary_keep_messages": int(self._summary_keep_messages),
                "summary_trigger_messages": int(self._summary_trigger_messages),
                "messages_before": int(messages_before),
                "messages_after_keep": int(messages_after_keep),
                "trigger_input_tokens": int(trigger_input_tokens),
                "threshold_tokens": int(threshold_tokens),
                "max_context_tokens": int(self._max_context_tokens),
                "threshold_ratio": float(self._compaction_threshold_ratio),
                "triggered_by_messages": bool(triggered_by_messages),
                "triggered_by_tokens": bool(triggered_by_tokens),
            },
            task_session_id=current_session_id() or "",
            run_label=current_run_label() or "",
        )
        return summary

    def _maybe_summarize_for_next_turn(self, input_tokens: int) -> None:
        if not self._conversation:
            return

        threshold_tokens = 0
        if self._max_context_tokens > 0 and self._compaction_threshold_ratio > 0:
            threshold_tokens = int(self._compaction_threshold_ratio * self._max_context_tokens)

        should_summarize_by_messages = (
            len(self._conversation) > max(self._summary_trigger_messages, self._summary_keep_messages + 2)
        )
        should_summarize_by_tokens = bool(
            threshold_tokens > 0 and input_tokens >= threshold_tokens
        )
        if not should_summarize_by_messages and not should_summarize_by_tokens:
            return

        before_len = len(self._conversation)
        keep = self._summary_keep_messages if self._summary_keep_messages > 0 else 0
        older_items = self._conversation[:-keep] if keep else list(self._conversation)
        if not older_items:
            return

        self._summary = self._summarize_items(
            self._summary,
            older_items,
            trigger_input_tokens=input_tokens,
            threshold_tokens=threshold_tokens,
            messages_before=before_len,
            messages_after_keep=min(before_len, keep),
            triggered_by_messages=should_summarize_by_messages,
            triggered_by_tokens=should_summarize_by_tokens,
        )
        if keep:
            self._conversation = self._conversation[-keep:]
        else:
            self._conversation = []

    def _build_runner_input(self, user_input: str, include_history: bool) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        if include_history:
            keep = self._summary_keep_messages if self._summary and self._summary_keep_messages > 0 else 0
            if keep:
                input_items = list(self._conversation[-keep:])
            else:
                input_items = list(self._conversation)
        if user_input:
            input_items.append({"role": "user", "content": user_input})
        return _sanitize_input_items(input_items)

    def _invoke_runner_with_recovery(self, user_input: str, context: Any) -> Any:
        include_history = not bool(self._previous_response_id or self._conversation_id)
        input_items = self._build_runner_input(user_input, include_history=include_history)
        try:
            return Runner.run_sync(
                self.agent,
                input_items,
                max_turns=self.max_turns,
                previous_response_id=self._previous_response_id,
                conversation_id=self._conversation_id,
                context=context,
            )
        except (BadRequestError, ModelBehaviorError) as exc:
            err = str(exc).lower()
            recoverable = (
                self._previous_response_id is not None
                and (
                    "function response turn comes immediately after a function call turn" in err
                    or ("invalid_argument" in err and "function call" in err and "function response" in err)
                    or "not found in agent chack" in err
                    or ("tool " in err and " not found in agent " in err)
                )
            )
            if not recoverable:
                raise
            _LOGGER.warning(
                "Runner rejected response chain with previous_response_id; retrying with fresh chain."
            )
            self._previous_response_id = None
            self._conversation_id = None
            retry_previous_response_id = None
            retry_conversation_id = None
            retry_input = self._build_runner_input(user_input, include_history=True)
            return Runner.run_sync(
                self.agent,
                retry_input,
                max_turns=self.max_turns,
                previous_response_id=retry_previous_response_id,
                conversation_id=retry_conversation_id,
                context=context,
            )

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)


def _extract_tool_steps(items: list[Any]) -> list[tuple[ToolAction, Any]]:
    steps: list[tuple[ToolAction, Any]] = []
    for item in items:
        if not isinstance(item, ToolCallItem):
            continue
        raw = item.raw_item
        tool_name = _get_tool_name(raw) or "tool"
        tool_input = _get_tool_input(raw)
        _LOGGER.info(
            "Tool call: tool=%s session=%s run=%s ts=%s",
            tool_name,
            current_session_id() or "no-session",
            current_run_label() or "Run 1",
            _log_timestamp(),
        )
        log_event(
            "tool_called",
            payload={
                "tool": tool_name,
                "tool_input": tool_input,
            },
            task_session_id=current_session_id() or "",
            run_label=current_run_label() or "",
        )
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
            setattr(
                tool,
                "tool_input_guardrails",
                [_require_task_steps_manager_init_first, _respect_max_tools_used],
            )
            continue
        if _require_task_steps_manager_init_first not in guards:
            guards.append(_require_task_steps_manager_init_first)
        if _respect_max_tools_used not in guards:
            guards.append(_respect_max_tools_used)
    return _wrap_tools_with_logging(tools)


def _get_tool_callable(tool: Any) -> tuple[Optional[Callable[..., Any]], Optional[str]]:
    for attr in ("callable", "func", "_callable", "_function", "_func"):
        target = getattr(tool, attr, None)
        if callable(target):
            return target, attr
    if callable(tool):
        return tool, None
    return None, None


def _extract_tool_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if kwargs:
        return kwargs
    if len(args) == 1 and isinstance(args[0], dict):
        return args[0]
    if len(args) == 1:
        return args[0]
    return {"args": list(args)}


def _wrap_tools_with_logging(tools: list[Any]) -> list[Any]:
    wrapped: list[Any] = []
    for tool in tools:
        if getattr(tool, "_chack_tool_wrapped", False):
            wrapped.append(tool)
            continue
        call_target, attr = _get_tool_callable(tool)
        if call_target is None:
            wrapped.append(tool)
            continue
        module_name = getattr(call_target, "__module__", "") or ""
        if module_name.startswith("chack_tools.") or module_name.startswith("chack_agent."):
            wrapped.append(tool)
            continue
        tool_name = (
            getattr(tool, "name", None)
            or getattr(tool, "name_override", None)
            or getattr(call_target, "__name__", None)
            or "tool"
        )

        def _wrapped(*args, **kwargs):
            tool_input = _extract_tool_input(args, kwargs)
            start_ts = log_tool_started(tool_name, tool_input)
            start_time = time.time()
            error = None
            try:
                return call_target(*args, **kwargs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                try:
                    log_tool_error(
                        tool_name,
                        tool_input,
                        error=error,
                        trace=traceback.format_exc(),
                    )
                except Exception:
                    pass
                raise
            finally:
                end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                duration_ms = int((time.time() - start_time) * 1000)
                log_tool_executed(
                    tool_name,
                    tool_input,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    duration_ms=duration_ms,
                    error=error,
                )

        _wrapped.__name__ = getattr(call_target, "__name__", "wrapped_tool")
        _wrapped.__doc__ = getattr(call_target, "__doc__", None)
        _wrapped.__module__ = module_name

        if attr:
            setattr(tool, attr, _wrapped)
            setattr(tool, "_chack_tool_wrapped", True)
            wrapped.append(tool)
        else:
            setattr(_wrapped, "_chack_tool_wrapped", True)
            wrapped.append(_wrapped)
    return wrapped


def build_executor(
    config: ChackConfig,
    *,
    system_prompt: str,
    max_turns: int,
    memory_max_messages: int,
    memory_reset_to_messages: int,
    tools_override: Optional[list[Any]] = None,
    tools_append: Optional[list[Any]] = None,
) -> AgentsExecutor:
    _, client = _configure_openai_client(config)
    model_name = config.model.primary

    if tools_override is None:
        init_params = inspect.signature(AgentsToolset.__init__).parameters
        toolset_kwargs = {
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
        if "model_provider" in init_params:
            toolset_kwargs["model_provider"] = str(config.model.provider or "")
        toolset = AgentsToolset(config.tools, **toolset_kwargs)
        tools = toolset.tools
        if tools_append:
            tools = list(tools) + list(tools_append)
    else:
        tools = list(tools_override)

    tools = _apply_guardrails(tools)
    model: Any = _OpenRouterResponsesModel(model=model_name, openai_client=client)

    output_schema = None
    schema_json = getattr(config.agent, "output_schema_json", None)
    if schema_json:
        output_schema = JsonSchemaOutput(
            schema_json,
            name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
            strict=bool(getattr(config.agent, "output_schema_strict", True)),
        )

    agent = Agent(
        name="Chack",
        instructions=system_prompt,
        tools=tools,
        model=model,
        model_settings=ModelSettings(),
        output_type=output_schema,
    )

    summary_agent = Agent(
        name="ChackSummary",
        instructions=(
            "You maintain compact conversation memory. "
            "Always return a concise factual summary preserving key facts, decisions, and constraints."
        ),
        tools=[],
        model=model,
        model_settings=ModelSettings(),
        output_type=None,
    )

    return AgentsExecutor(
        agent=agent,
        _summary_agent=summary_agent,
        max_turns=max_turns,
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _previous_response_id=None,
        _conversation_id=None,
        _summary="",
        _summary_trigger_messages=memory_max_messages,
        _summary_keep_messages=memory_reset_to_messages,
        _summary_max_chars=max(
            0,
            _to_int(os.environ.get("CHACK_OPENROUTER_SUMMARY_MAX_CHARS", "6000"), 6000),
        ),
        _compaction_threshold_ratio=float(getattr(config.agent, "compaction_threshold_ratio", 0.75) or 0.75),
        _max_context_tokens=int(config.model.max_context_tokens or 0),
    )
