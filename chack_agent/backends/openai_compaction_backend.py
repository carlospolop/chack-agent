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

from openai import OpenAI, AsyncOpenAI, BadRequestError
from agents import (
    Agent,
    ModelSettings,
    Runner,
    ToolGuardrailFunctionOutput,
    tool_input_guardrail,
    set_default_openai_client,
)
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
_LOGGER = logging.getLogger("chack.openai_compaction_backend")


def _select_provider(config: ChackConfig) -> str:
    provider = str(getattr(config.model, "provider", "") or "openai").strip().lower()
    if provider != "openai":
        raise ValueError(
            f"openai_compaction_backend requires model.provider='openai' (got {provider!r})"
        )
    return provider


def _configure_openai_client(config: ChackConfig) -> tuple[str, Optional[AsyncOpenAI]]:
    provider = _select_provider(config)
    # OpenAI
    api_key = (
        str(config.credentials.openai_api_key or "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    organization = str(config.credentials.openai_org_id or "").strip() or None
    client: Optional[AsyncOpenAI] = None
    if api_key or base_url or organization:
        client = AsyncOpenAI(
            api_key=api_key or None,
            base_url=base_url,
            organization=organization,
        )
        set_default_openai_client(client)
    return provider, client


def _run_scope_key() -> str:
    session_id = current_session_id() or "no-session"
    run_label = current_run_label() or "Run 1"
    return f"{session_id}:{run_label}"


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    _config: ChackConfig
    agent: Agent
    max_turns: int
    _conversation: list[dict[str, Any]]
    _memory_limit: int
    _memory_reset_to: int
    _base_system_prompt: str
    _previous_response_id: Optional[str]
    _conversation_id: Optional[str]
    _compaction_threshold_ratio: float
    _max_context_tokens: int
    _compaction_model: str

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        user_input = payload.get("input", "")
        self.agent.instructions = self._base_system_prompt
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
            reset_to = self._memory_reset_to or self._memory_limit
            if reset_to > self._memory_limit:
                reset_to = self._memory_limit
            if reset_to < 1:
                reset_to = 1
            self._conversation = self._conversation[-reset_to:]
        if result.last_response_id:
            self._previous_response_id = result.last_response_id
        conversation_id = getattr(result, "_conversation_id", None)
        if isinstance(conversation_id, str) and conversation_id.strip():
            self._conversation_id = conversation_id.strip()
        self._maybe_compact(result)
        steps = _extract_tool_steps(result.new_items)
        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": result,
        }

    def _build_runner_input(self, user_input: str, include_history: bool) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        if include_history:
            input_items = list(self._conversation)
        if user_input:
            input_items.append({"role": "user", "content": user_input})
        return _sanitize_input_items(input_items)

    def _invoke_runner_with_recovery(self, user_input: str, context: Any) -> Any:
        # For OpenAI, rely on server-side conversation state only.
        include_history = False
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
            # Keep OpenAI-managed conversation when available; only drop previous_response_id.
            self._previous_response_id = None
            retry_previous_response_id = None
            retry_conversation_id = self._conversation_id
            retry_input = self._build_runner_input(user_input, include_history=False)
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

        threshold_tokens = int(self._compaction_threshold_ratio * self._max_context_tokens)
        conversation_messages_before = len(self._conversation or [])

        _LOGGER.info(
            "Triggering response compaction: input_tokens=%s threshold_tokens=%s max_context=%s threshold_ratio=%s conversation_messages_before=%s ts=%s.",
            input_tokens,
            threshold_tokens,
            self._max_context_tokens,
            self._compaction_threshold_ratio,
            conversation_messages_before,
            _log_timestamp(),
        )
        log_event(
            "agent_compaction_triggered",
            payload={
                "backend": "openai_compaction",
                "provider": str(getattr(self._config.model, "provider", "") or "openai"),
                "model": str(getattr(self._config.model, "primary", "") or ""),
                "compaction_model": str(self._compaction_model or ""),
                "input_tokens": int(input_tokens),
                "max_context_tokens": int(self._max_context_tokens),
                "threshold_ratio": float(self._compaction_threshold_ratio),
                "threshold_tokens": int(threshold_tokens),
                "conversation_messages_before": int(conversation_messages_before),
            },
            task_session_id=current_session_id() or "",
            run_label=current_run_label() or "",
        )
        new_response_id = self._run_compaction(self._previous_response_id)
        if new_response_id:
            _LOGGER.info(
                "Responses compaction complete. New response id: %s ts=%s.",
                new_response_id,
                _log_timestamp(),
            )
            log_event(
                "agent_compaction_completed",
                payload={
                    "backend": "openai_compaction",
                    "provider": str(getattr(self._config.model, "provider", "") or "openai"),
                    "model": str(getattr(self._config.model, "primary", "") or ""),
                    "compaction_model": str(self._compaction_model or ""),
                    "input_tokens": int(input_tokens),
                    "threshold_tokens": int(threshold_tokens),
                    "conversation_messages_before": int(conversation_messages_before),
                    "previous_response_id": str(self._previous_response_id or ""),
                    "new_response_id": str(new_response_id or ""),
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
            self._previous_response_id = new_response_id

    def _run_compaction(self, response_id: str) -> Optional[str]:
        try:
            api_key = (
                str(self._config.credentials.openai_api_key or "").strip()
                or os.environ.get("OPENAI_API_KEY", "").strip()
            )
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
            organization = str(self._config.credentials.openai_org_id or "").strip() or None
            client = OpenAI(
                api_key=api_key or None,
                base_url=base_url,
                organization=organization,
            )
            compacted = client.responses.compact(
                model=self._compaction_model,
                previous_response_id=response_id,
            )
            response_id = getattr(compacted, "id", None) or getattr(
                compacted, "response_id", None
            )
            return response_id
        except Exception as exc:
            _LOGGER.exception("Responses compaction failed.")
            log_event(
                "agent_compaction_failed",
                payload={
                    "backend": "openai_compaction",
                    "provider": str(getattr(self._config.model, "provider", "") or "openai"),
                    "model": str(getattr(self._config.model, "primary", "") or ""),
                    "compaction_model": str(self._compaction_model or ""),
                    "previous_response_id": str(response_id or ""),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
            return None


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
    memory_summary_max_chars: int = 0,
    tools_override: Optional[list[Any]] = None,
    tools_append: Optional[list[Any]] = None,
) -> AgentsExecutor:
    try:
        _LOGGER.debug(
            "openai_compaction build_executor: memory_summary_max_chars=%s (not used in this backend)",
            int(memory_summary_max_chars),
        )
    except Exception:
        _LOGGER.debug(
            "openai_compaction build_executor: memory_summary_max_chars provided (unable to coerce to int in debug log)"
        )
    _configure_openai_client(config)
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
    model: Any = model_name
    # For OpenAI compaction backend use native OpenAI Responses model.

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

    return AgentsExecutor(
        _config=config,
        agent=agent,
        max_turns=max_turns,
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _previous_response_id=None,
        _conversation_id=None,
        _compaction_threshold_ratio=float(
            config.agent.compaction_threshold_ratio or 0.75
        ),
        _max_context_tokens=int(config.model.max_context_tokens or 0),
        _compaction_model=(
            str(config.agent.compaction_model).strip() or model_name
        ),
    )
