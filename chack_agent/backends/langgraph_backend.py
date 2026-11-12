from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from operator import add
import threading
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from agents import MaxTurnsExceeded
from agents.tool_context import ToolContext
from agents.usage import Usage
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph.message import add_messages

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_steps_manager_state import current_run_label, current_session_id
from chack_tools.telemetry import log_event
from chack_tools.tool_call_policy import ToolCallPolicy, has_exec_tool
from chack_tools.tool_usage_state import current_max_tools_used, is_budget_exempt_tool_name

from ..config import ChackConfig
from ..llm_gateway import get_gateway_route
from ..thinking_effort import normalize_thinking_effort
from ..budget_warning_state import inject_budget_warning
from ..limit_event_state import emit_limit_reached
from chack_tools.cancellation import raise_if_cancelled

from ..live_cost_state import report_live_usage
from ..output_schema import schema_repair_prompt, validate_schema_output
from ..resume_compaction import ResumeCompactionResult
from .content_filter import ProviderContentFilterError
from .hosted_web_search import WebToolPlan, plan_web_tools
from .playwright_mcp import (
    playwright_mcp_call_tool,
    playwright_mcp_is_available,
    playwright_mcp_list_tools,
    playwright_mcp_result_to_text,
)


_LOGGER = logging.getLogger("chack.langgraph_backend")


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _report_message_usage(model: Any, response: Any) -> None:
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details", {}) or {}
    report_live_usage(
        str(getattr(model, "model_name", "") or ""),
        prompt_tokens=int(usage.get("input_tokens", 0) or 0),
        completion_tokens=int(usage.get("output_tokens", 0) or 0),
        cached_prompt_tokens=int(details.get("cache_read", 0) or 0),
        cache_write_tokens=int(details.get("cache_creation", 0) or 0),
    )


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]


class _GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    tool_events: Annotated[list[dict[str, Any]], add]
    usage_events: Annotated[list[dict[str, int]], add]
    has_task_steps_manager_init: bool
    non_task_tool_calls: int
    # Per invoke: model calls made, recovery nudges sent, recent call keys.
    model_calls: int
    nudges: int
    recent_calls: list[str]


# Interleaved-thinking models (Kimi, MiniMax, GLM, DeepSeek; OpenRouter's
# reasoning_details) expect their reasoning back on each assistant tool-call
# turn, as opencode sends it. Without it they forget their plan every step.
_REASONING_FIELDS = ("reasoning_content", "reasoning_details")

# Window assumed when the model's context size is not configured.
_DEFAULT_CONTEXT_TOKENS = 200_000


def _with_reasoning_passthrough(chat_cls: type) -> type:
    """Subclass a langchain ChatOpenAI so reasoning survives the round trip."""

    class _ReasoningChatOpenAI(chat_cls):  # type: ignore[misc, valid-type]
        def _create_chat_result(self, response: Any, generation_info: Optional[dict] = None) -> Any:
            result = super()._create_chat_result(response, generation_info)
            response_dict = response if isinstance(response, dict) else response.model_dump()
            for generation, choice in zip(result.generations, response_dict.get("choices") or []):
                message = (choice or {}).get("message") or {}
                for field in _REASONING_FIELDS:
                    if message.get(field):
                        generation.message.additional_kwargs[field] = message[field]
            return result

        def _get_request_payload(self, input_: Any, *, stop: Optional[list[str]] = None, **kwargs: Any) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            if "messages" in payload:
                messages = self._convert_input(input_).to_messages()
                for message, entry in zip(messages, payload["messages"]):
                    if isinstance(message, AIMessage) and entry.get("role") == "assistant":
                        for field in _REASONING_FIELDS:
                            if message.additional_kwargs.get(field):
                                entry[field] = message.additional_kwargs[field]
            return payload

    return _ReasoningChatOpenAI


def _tool_budget_note(remaining: int, limit: int) -> str:
    """A note for a tool result once the budget runs low, else nothing."""
    if remaining <= 0:
        return f"\n\n[Tool budget: all {limit} non-task tool calls are used. Answer from the evidence already gathered.]"
    if remaining <= max(3, limit // 10):
        return f"\n\n[Tool budget: {remaining} of {limit} non-task tool calls remain. Spend them on the checks that still matter.]"
    return ""


def _to_int(raw: str, default: int) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception:
        return default


def _recursion_limit_for(max_turns: int) -> int:
    # A turn is one model call, as in the Agents SDK backends. In the graph a
    # call takes two steps (llm_call, then tool_node or a nudge); the graph
    # itself ends the run on the last turn, so this is only a backstop.
    return 2 * max(1, int(max_turns or 0)) + 2


# Recovery turns: a reply that is empty, cut off, or neither a tool call nor
# the required answer is not taken as final, as the CLIs retry or continue it.
_MAX_NUDGES = 2
_EMPTY_REPLY_NUDGE = (
    "Your last reply was empty. Continue the task: call a tool, or give your final answer if the work is done."
)
_LENGTH_NUDGE = (
    "Your last reply hit the output token limit. Continue from where it stopped, without repeating "
    "yourself, and break large outputs into smaller pieces."
)
_UNFINISHED_NUDGE = (
    "Your last reply had no tool call, so it would end the run, but it is not the required final answer. "
    "If work remains, continue it with tool calls now; otherwise reply with only the final answer in the "
    "required format."
)
_FINAL_TURN_PROMPT = (
    "This is the last turn of this run: tools are no longer available. Give your final answer now from the "
    "evidence gathered so far, in the required format, and state what remains unverified."
)
_MASKED_OUTPUT = "[Earlier output of this tool call was cleared to save context; call the tool again if you need it.]"
# Tool results kept in full when older ones are cleared.
_KEEP_RECENT_TOOL_RESULTS = 5
_REPEAT_WINDOW = 30
_SUMMARY_SOURCE_CLIP = 2000


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters omitted ...]\n{text[-half:]}"


def _call_key(name: str, args: Any) -> str:
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    except Exception:
        return f"{name}:{args!r}"


def _is_context_overflow(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in (
        "context_length_exceeded", "maximum context length", "context window", "prompt is too long",
        "input is too long", "too many tokens", "context length",
    ))


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    box: dict[str, Any] = {"result": None, "error": None}

    def _target() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover
            box["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if box["error"] is not None:
        raise box["error"]
    return box["result"]


@dataclass
class LangGraphExecutor:
    _graph: Any
    _model_with_tools: Any
    _summary_model: Any
    _function_tools_by_name: dict[str, Any]
    _mcp_tools_by_name: dict[str, Any]
    _conversation: list[dict[str, Any]]
    _memory_limit: int
    _memory_reset_to: int
    _base_system_prompt: str
    _thread_id: str
    _max_non_task_tools: int
    _require_task_steps_manager_init_first: bool
    _summary_trigger_messages: int
    _summary_keep_messages: int
    _summary_max_chars: int
    _compaction_threshold_ratio: float
    _max_context_tokens: int
    _output_schema_json: str
    _output_schema_name: str
    _output_schema_strict: bool
    _max_turns: int = 0
    _pending_resume_summary: str = ""
    # Set when a run ended near the context limit: the next run compacts first.
    _compact_before_next_run: bool = False
    _compact_before_next_run_tokens: int = 0

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        if not user_input.strip():
            return {"output": "", "intermediate_steps": [], "raw_result": _RawResult(raw_responses=[])}

        if self._compact_before_next_run:
            # Deferred from the previous run, so a run that is never
            # continued does not pay for a summary.
            self._compact_before_next_run = False
            self._compact_for_threshold(latest_input_tokens=self._compact_before_next_run_tokens)

        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": _recursion_limit_for(self._max_turns),
        }

        prev_tool_events = 0
        prev_usage_events = 0
        try:
            snapshot = self._graph.get_state(config)
            values = snapshot.values if snapshot is not None else {}
            if isinstance(values, dict):
                prev_tool_events = len(values.get("tool_events", []) or [])
                prev_usage_events = len(values.get("usage_events", []) or [])
        except Exception:
            pass

        fresh_run = {"model_calls": 0, "nudges": 0, "recent_calls": []}
        graph_input: dict[str, Any] = {
            "messages": [HumanMessage(content=user_input)],
            **fresh_run,
        }
        if self._pending_resume_summary:
            graph_input["summary"] = self._pending_resume_summary
            self._pending_resume_summary = ""
        result_state = self._invoke_graph(graph_input, config)
        state_values = dict(result_state or {})

        messages = state_values.get("messages", []) or []
        output = self._extract_output(messages)
        if self._output_schema_json:
            output, schema_error = self._validate_schema_output(output)
            if schema_error:
                # The schema is only a prompt contract here, so the model can
                # answer in prose. Give it one turn to fix the answer.
                _LOGGER.warning("LangGraph answer did not match the output schema; asking for a repair: %s", schema_error)
                result_state = self._invoke_graph(
                    {
                        "messages": [HumanMessage(content=self._schema_repair_prompt(schema_error))],
                        # The repair prompt already says what is wrong.
                        **fresh_run, "nudges": _MAX_NUDGES,
                    },
                    config,
                )
                state_values = dict(result_state or {})
                messages = state_values.get("messages", []) or []
                output, schema_error = self._validate_schema_output(self._extract_output(messages))
                if schema_error:
                    _LOGGER.warning("LangGraph answer still did not match the output schema after a repair turn: %s", schema_error)

        tool_events = state_values.get("tool_events", []) or []
        usage_events = state_values.get("usage_events", []) or []
        new_tool_events = tool_events[prev_tool_events:]
        new_usage_events = usage_events[prev_usage_events:]

        steps: list[tuple[ToolAction, Any]] = []
        for event in new_tool_events:
            if not isinstance(event, dict):
                continue
            tool_name = str(event.get("tool", "") or "")
            tool_input = event.get("tool_input")
            if not tool_name:
                continue
            action = ToolAction(tool=tool_name, tool_input=tool_input)
            steps.append((action, None))
            self._log_tool_called(tool_name, tool_input)

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        cache_write_tokens = 0
        for usage in new_usage_events:
            if not isinstance(usage, dict):
                continue
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
            cached_tokens += int(usage.get("cached_input_tokens", 0) or 0)
            cache_write_tokens += int(usage.get("cache_write_input_tokens", 0) or 0)

        raw_responses: list[Any] = []
        if input_tokens or output_tokens or cached_tokens or cache_write_tokens:
            raw_responses.append(
                {
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "input_tokens_details": {
                            "cached_tokens": cached_tokens,
                            "cache_write_tokens": cache_write_tokens,
                        },
                    }
                }
            )

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

        threshold_tokens = int(self._max_context_tokens * self._compaction_threshold_ratio)
        latest_input_tokens = max(
            (
                int(usage.get("input_tokens", 0) or 0)
                for usage in new_usage_events
                if isinstance(usage, dict)
            ),
            default=0,
        )
        if threshold_tokens > 0 and latest_input_tokens >= threshold_tokens:
            self._compact_before_next_run = True
            self._compact_before_next_run_tokens = latest_input_tokens

        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": _RawResult(raw_responses=raw_responses),
        }

    def _compact_for_threshold(self, *, latest_input_tokens: int) -> None:
        threshold_tokens = int(self._max_context_tokens * self._compaction_threshold_ratio)
        log_event(
            "agent_compaction_triggered",
            payload={
                "backend": "langgraph",
                "input_tokens": latest_input_tokens,
                "threshold_tokens": threshold_tokens,
                "max_context_tokens": self._max_context_tokens,
            },
            task_session_id=current_session_id() or "",
            run_label=current_run_label() or "",
        )
        compacted = self.compact_for_resume()
        log_event(
            "agent_compaction_completed" if compacted.succeeded else "agent_compaction_failed",
            payload={
                "backend": "langgraph",
                "method": compacted.method,
                "input_tokens": latest_input_tokens,
                "threshold_tokens": threshold_tokens,
                "error": compacted.error[:500],
            },
            task_session_id=current_session_id() or "",
            run_label=current_run_label() or "",
        )

    def _invoke_graph(self, graph_input: dict[str, Any], config: dict[str, Any]) -> Any:
        graph_errors = importlib.import_module("langgraph.errors")
        try:
            return self._graph.invoke(graph_input, config=config)
        except graph_errors.GraphRecursionError as exc:
            raise MaxTurnsExceeded(f"Max turns ({self._max_turns}) exceeded") from exc

    def _validate_schema_output(self, output: str) -> tuple[str, str]:
        """Return the answer as schema-shaped JSON, or as given with the reason it fails."""
        return validate_schema_output(
            output,
            self._output_schema_json,
            name=str(self._output_schema_name or "output_schema"),
            strict=bool(self._output_schema_strict),
        )

    def _schema_repair_prompt(self, schema_error: str) -> str:
        return schema_repair_prompt(
            schema_error,
            self._output_schema_json,
            name=str(self._output_schema_name or "output_schema"),
        )

    async def aget_memory_messages(self) -> list[Any]:
        config = {"configurable": {"thread_id": self._thread_id}}
        try:
            snapshot = self._graph.get_state(config)
            values = snapshot.values if snapshot is not None else {}
            if isinstance(values, dict) and isinstance(values.get("messages"), list):
                return list(values.get("messages") or [])
        except Exception:
            pass
        return list(self._conversation)

    def compact_for_resume(
        self, focus_instructions: str = ""
    ) -> ResumeCompactionResult:
        result = ResumeCompactionResult(
            backend="langgraph",
            method="summary_thread_rotation",
        )
        config = {"configurable": {"thread_id": self._thread_id}}
        try:
            snapshot = self._graph.get_state(config)
            values = snapshot.values if snapshot is not None else {}
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        messages = (
            list(values.get("messages") or [])
            if isinstance(values, dict)
            else []
        )
        previous_summary = (
            str(values.get("summary") or "")
            if isinstance(values, dict)
            else ""
        )
        if not messages:
            return result
        result.attempted = True
        started_at = time.monotonic()
        try:
            summary = self._summarize_messages(
                previous_summary,
                messages,
                messages_before=len(messages),
                messages_after_keep=0,
                focus_instructions=focus_instructions,
            )
            if not summary:
                raise RuntimeError("summary model returned an empty summary")
            # LangGraph's additive message reducer cannot delete the old
            # checkpoint safely. Start a fresh thread seeded with the summary.
            self._thread_id = f"langgraph-{uuid.uuid4()}"
            self._pending_resume_summary = summary
            self._conversation = []
            result.succeeded = True
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_seconds = max(0.0, time.monotonic() - started_at)
        return result

    def _extract_output(self, messages: list[AnyMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                if getattr(message, "tool_calls", None):
                    continue
                content = message.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "\n".join(str(part) for part in content)
                return str(content or "")
        return ""

    def _invoke_function_tool(self, tool: Any, args: dict[str, Any]) -> str:
        raw_args = json.dumps(args, ensure_ascii=False)
        ctx = ToolContext(
            context=None,
            usage=Usage(),
            tool_name=str(getattr(tool, "name", "tool") or "tool"),
            tool_call_id=f"langgraph-{uuid.uuid4()}",
            tool_arguments=raw_args,
            tool_input=args,
        )
        result = _run_coro_sync(tool.on_invoke_tool(ctx, raw_args))
        if isinstance(result, str):
            return result
        if isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")
        if isinstance(result, (dict, list, tuple)):
            return json.dumps(result, ensure_ascii=False)
        try:
            return json.dumps(result.model_dump(), ensure_ascii=False)
        except Exception:
            pass
        return str(result)

    def _summarize_messages(
        self,
        previous_summary: str,
        messages: list[AnyMessage],
        *,
        trigger_input_tokens: int = 0,
        threshold_tokens: int = 0,
        messages_before: int = 0,
        messages_after_keep: int = 0,
        triggered_by_messages: bool = False,
        triggered_by_tokens: bool = False,
        focus_instructions: str = "",
    ) -> str:
        if not messages:
            return previous_summary

        rendered: list[str] = []
        for message in messages:
            role = "assistant"
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, ToolMessage):
                role = "tool"
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            calls = "".join(
                f" -> {call.get('name')}({json.dumps(call.get('args'), ensure_ascii=False, default=str)})"
                for call in (getattr(message, "tool_calls", None) or [])
            )
            # Raw tool output can be huge; its head and tail carry what matters.
            rendered.append(f"[{role}] {_clip_middle(content, _SUMMARY_SOURCE_CLIP)}{calls}")
        chunk = "\n".join(rendered)
        previous_summary_chars = len(previous_summary or "")
        source_messages_count = len(messages)
        source_chars = len(chunk)
        did_truncate = False

        focus_block = (
            f"\n\nCompaction focus:\n{focus_instructions.strip()}"
            if str(focus_instructions or "").strip()
            else ""
        )
        # A handoff for the same agent to keep working, as Codex compacts:
        # what is done and found, with the exact identifiers, and what is left.
        earlier = f"Earlier checkpoint:\n{previous_summary}\n\n" if previous_summary else ""
        prompt = (
            "Write a checkpoint so you can continue this task after the messages below are removed. "
            "Keep: the progress so far and key decisions; every finding and fact with its exact identifiers, "
            "versions, URLs, paths and commands; leads ruled out and why; and the concrete next steps."
            f"{f' Stay within {self._summary_max_chars} characters.' if self._summary_max_chars > 0 else ''}\n\n"
            f"{earlier}Messages:\n{chunk}{focus_block}"
        )

        raise_if_cancelled()
        response = self._summary_model.invoke([HumanMessage(content=prompt)])
        _report_message_usage(self._summary_model, response)
        content = getattr(response, "content", "")
        if isinstance(content, str):
            summary = content.strip() or previous_summary
        elif isinstance(content, list):
            summary = "\n".join(str(part) for part in content).strip() or previous_summary
        else:
            summary = str(content or previous_summary).strip() or previous_summary
        if self._summary_max_chars > 0 and len(summary) > self._summary_max_chars:
            summary = summary[: self._summary_max_chars]
            did_truncate = True

        _LOGGER.info(
            "LangGraph summary compaction performed: messages_before=%s messages_summarized=%s messages_after_keep=%s trigger_input_tokens=%s threshold_tokens=%s source_chars=%s prev_summary_chars=%s new_summary_chars=%s truncated=%s keep_messages=%s trigger_messages=%s by_messages=%s by_tokens=%s ts=%s",
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
                "backend": "langgraph",
                "provider": "langgraph",
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

    def _system_prompt(self) -> str:
        # Stays byte-identical for the whole run so the provider can cache it:
        # the tool budget is reported on tool results and the checkpoint
        # summary is sent as a message.
        base = str(self._base_system_prompt or "").strip()
        policy_lines: list[str] = []
        if self._require_task_steps_manager_init_first:
            policy_lines.append(
                "- First, call task_steps_manager with action=init before any other tool call."
            )
        if self._max_non_task_tools > 0:
            policy_lines.append(
                f"- Do not exceed {self._max_non_task_tools} non-task tool calls."
            )
        # The CLIs' persistence rules; many models do not know a reply
        # without a tool call ends their run.
        policy_lines.append(
            "- Keep working until the task is completely done: when a tool call fails or a lead runs dry, "
            "try another approach instead of stopping, and do not end at a plan or a partial result."
        )
        policy_lines.append(
            "- A reply without a tool call ends the run and is taken as your final answer; "
            "send one only when the work is done."
        )
        available_tools = (*self._function_tools_by_name, *self._mcp_tools_by_name)
        if any("web" in name.lower() or "fetch" in name.lower() for name in available_tools):
            policy_lines.append(
                "- When the task asks you to check a current external source, use an available web tool before citing it."
            )

        policy_block = ""
        if policy_lines:
            policy_block = "\n\n### TOOL USAGE POLICY\n" + "\n".join(policy_lines)

        output_schema_block = ""
        if self._output_schema_json:
            schema_name = str(self._output_schema_name or "output_schema").strip() or "output_schema"
            strict_text = "true" if self._output_schema_strict else "false"
            output_schema_block = (
                "\n\n### OUTPUT FORMAT (REQUIRED)\n"
                "Your final response must be valid JSON matching exactly this JSON Schema.\n"
                "Return ONLY the JSON object, with no markdown and no extra text.\n"
                f"Schema name: {schema_name}\n"
                f"Strict: {strict_text}\n"
                f"Schema:\n{self._output_schema_json}"
            )

        return f"{base}{policy_block}{output_schema_block}".strip()

    def build_graph(self) -> None:
        checkpoint_mod = importlib.import_module("langgraph.checkpoint.memory")
        graph_mod = importlib.import_module("langgraph.graph")
        InMemorySaver = getattr(checkpoint_mod, "InMemorySaver")
        StateGraph = getattr(graph_mod, "StateGraph")
        START = getattr(graph_mod, "START")
        END = getattr(graph_mod, "END")

        tool_map = self._function_tools_by_name

        def compact(
            messages: list[AnyMessage], summary: str, *, latest_input_tokens: int, threshold_tokens: int,
            by_messages: bool, by_tokens: bool,
        ) -> tuple[list[AnyMessage], list[AnyMessage], list[AnyMessage], str]:
            """Return (removed, replaced, kept, summary) for a context that must shrink."""
            if by_tokens:
                # Clearing old tool outputs first, as Claude Code and opencode
                # do, keeps every call and reasoning step the agent made; a
                # summary rewrites them all.
                results = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
                old = results[:-_KEEP_RECENT_TOOL_RESULTS] if _KEEP_RECENT_TOOL_RESULTS else results
                masked = [
                    ToolMessage(content=_MASKED_OUTPUT, id=messages[i].id, tool_call_id=messages[i].tool_call_id,
                                name=messages[i].name)
                    for i in old
                    if isinstance(messages[i].content, str) and len(messages[i].content) > 4 * len(_MASKED_OUTPUT)
                ]
                if masked:
                    by_id = {m.id: m for m in masked}
                    _LOGGER.info("LangGraph cleared %s old tool outputs: trigger_input_tokens=%s threshold_tokens=%s",
                                 len(masked), latest_input_tokens, threshold_tokens)
                    return [], masked, [by_id.get(m.id, m) for m in messages], summary
            # Keep the current request even when the configured tail size
            # is zero. Never start the retained tail with a tool result:
            # its matching assistant tool call must remain beside it.
            keep = max(1, self._summary_keep_messages)
            start = max(0, len(messages) - keep)
            while start > 0 and isinstance(messages[start], ToolMessage):
                start -= 1
            # The request that started this run stays verbatim: a summary
            # of it loses the details the agent must still act on.
            request = next((m for m in reversed(messages[:start]) if isinstance(m, HumanMessage)), None)
            if request is not None and any(isinstance(m, HumanMessage) for m in messages[start:]):
                request = None
            older = [m for m in messages[:start] if m is not request]
            if not older:
                return [], [], messages, summary
            new_summary = self._summarize_messages(
                summary,
                older,
                trigger_input_tokens=latest_input_tokens,
                threshold_tokens=threshold_tokens,
                messages_before=len(messages),
                messages_after_keep=len(messages) - start,
                triggered_by_messages=by_messages,
                triggered_by_tokens=by_tokens,
            )
            return older, [], ([request] if request is not None else []) + messages[start:], new_summary

        def call_model(prompt_messages: list[AnyMessage], final_turn: bool) -> Any:
            raise_if_cancelled()
            if not final_turn:
                return self._model_with_tools.invoke(prompt_messages)
            # The last allowed call answers without tools, as opencode ends
            # at its step limit, instead of the run failing with no answer.
            prompt_messages = [*prompt_messages, HumanMessage(content=_FINAL_TURN_PROMPT)]
            try:
                response = self._model_with_tools.bind(tool_choice="none").invoke(prompt_messages)
            except Exception as exc:
                if _is_context_overflow(exc):
                    raise
                response = self._summary_model.invoke(prompt_messages)
            kwargs = {k: v for k, v in (response.additional_kwargs or {}).items() if k != "tool_calls"}
            return response.model_copy(update={"tool_calls": [], "invalid_tool_calls": [], "additional_kwargs": kwargs})

        def llm_call(state: _GraphState) -> dict[str, Any]:
            messages = list(state.get("messages", []) or [])
            summary = str(state.get("summary", "") or "")
            model_calls = int(state.get("model_calls", 0) or 0)
            usage_events = list(state.get("usage_events", []) or [])
            latest_input_tokens = 0
            if usage_events:
                last_usage = usage_events[-1] if isinstance(usage_events[-1], dict) else {}
                latest_input_tokens = int(last_usage.get("input_tokens", 0) or 0)

            threshold_tokens = 0
            if self._max_context_tokens > 0 and self._compaction_threshold_ratio > 0:
                threshold_tokens = int(self._compaction_threshold_ratio * self._max_context_tokens)

            # The message limit counts conversation turns, as on the other
            # backends; tool rounds inside one run only compact on tokens,
            # near the context limit, like Codex, Claude Code and opencode.
            turns = sum(
                1 for m in messages
                if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
            )
            should_summarize_by_messages = (
                turns > max(self._summary_trigger_messages, self._summary_keep_messages + 2)
            )
            should_summarize_by_tokens = bool(
                threshold_tokens > 0 and latest_input_tokens >= threshold_tokens
            )

            removed: list[AnyMessage] = []
            replaced: list[AnyMessage] = []
            new_summary = summary
            if should_summarize_by_messages or should_summarize_by_tokens:
                removed, replaced, messages, new_summary = compact(
                    messages, summary, latest_input_tokens=latest_input_tokens, threshold_tokens=threshold_tokens,
                    by_messages=should_summarize_by_messages, by_tokens=should_summarize_by_tokens,
                )

            final_turn = model_calls + 1 >= max(1, int(self._max_turns or 0))

            def prompt_for(conversation: list[AnyMessage], checkpoint: str) -> list[AnyMessage]:
                # The checkpoint is a message, not part of the system prompt,
                # so the system prompt stays cached across compactions.
                head: list[AnyMessage] = [SystemMessage(content=self._system_prompt())]
                if checkpoint:
                    head.append(HumanMessage(
                        content=f"Checkpoint of your earlier work in this conversation, whose messages were removed:\n{checkpoint}"
                    ))
                return head + conversation

            try:
                response = call_model(prompt_for(messages, new_summary), final_turn)
            except Exception as exc:
                if not _is_context_overflow(exc):
                    raise
                # The window was smaller than configured or one result was
                # huge: shrink now and retry once, as the CLIs do.
                _LOGGER.warning("LangGraph context overflow; compacting and retrying: %s", str(exc)[:300])
                more_removed, more_replaced, messages, new_summary = compact(
                    messages, new_summary, latest_input_tokens=latest_input_tokens, threshold_tokens=threshold_tokens,
                    by_messages=False, by_tokens=True,
                )
                if not more_removed and not more_replaced:
                    raise
                removed += more_removed
                replaced = list({m.id: m for m in [*replaced, *more_replaced]}.values())
                response = call_model(prompt_for(messages, new_summary), final_turn)

            usage = getattr(response, "usage_metadata", None) or {}
            usage_event = {
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "cached_input_tokens": int(usage.get("input_token_details", {}).get("cache_read", 0) or 0),
                "cache_write_input_tokens": int(usage.get("input_token_details", {}).get("cache_creation", 0) or 0),
            }
            report_live_usage(
                str(getattr(self._summary_model, "model_name", "") or ""),
                prompt_tokens=usage_event["input_tokens"],
                completion_tokens=usage_event["output_tokens"],
                cached_prompt_tokens=usage_event["cached_input_tokens"],
                cache_write_tokens=usage_event["cache_write_input_tokens"],
            )
            finish_reason = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
            if finish_reason == "content_filter":
                raise ProviderContentFilterError()
            payload: dict[str, Any] = {
                "messages": [RemoveMessage(id=m.id) for m in removed] + replaced + [response],
                "usage_events": [usage_event],
                "model_calls": model_calls + 1,
            }
            if new_summary != summary:
                payload["summary"] = new_summary
            return payload

        def tool_node(state: _GraphState) -> dict[str, Any]:
            messages = list(state.get("messages", []) or [])
            if not messages:
                return {}
            last_message = messages[-1]
            tool_calls = getattr(last_message, "tool_calls", None) or []
            has_init = bool(state.get("has_task_steps_manager_init", False))
            non_task_count = int(state.get("non_task_tool_calls", 0) or 0)
            configured_max_non_task_tools = int(current_max_tools_used() or self._max_non_task_tools or 0)

            recent_calls = list(state.get("recent_calls", []) or [])

            outputs: list[ToolMessage] = []
            events: list[dict[str, Any]] = []

            # Calls whose arguments were not valid JSON would otherwise end
            # the run as if they were a final answer; tell the model instead.
            for bad_call in getattr(last_message, "invalid_tool_calls", None) or []:
                call_name = str(bad_call.get("name", "") or "tool")
                received = str(bad_call.get("args", "") or "")[:500]
                outputs.append(
                    ToolMessage(
                        content=(
                            f"ERROR: The arguments for {call_name} could not be parsed as JSON, so the call was not "
                            f"executed. Reissue the call with valid JSON arguments. Received: {received}"
                        ),
                        tool_call_id=str(bad_call.get("id", "") or ""),
                        name=call_name,
                    )
                )
                events.append({"tool": call_name, "tool_input": received, "status": "error", "reason": "invalid_arguments"})

            for tool_call in tool_calls:
                call_name = str(tool_call.get("name", "") or "").strip()
                call_id = str(tool_call.get("id", "") or "")
                args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}

                if not call_name:
                    continue

                if self._require_task_steps_manager_init_first and not has_init:
                    is_init = call_name == "task_steps_manager" and str(args.get("action", "")).strip().lower() == "init"
                    if not is_init:
                        outputs.append(
                            ToolMessage(
                                content="ERROR: You must call task_steps_manager with action=init before any other tool.",
                                tool_call_id=call_id,
                                name=call_name,
                            )
                        )
                        events.append(
                            {
                                "tool": call_name,
                                "tool_input": args,
                                "status": "blocked",
                                "reason": "task_steps_manager_init_required",
                            }
                        )
                        continue

                is_non_task = not is_budget_exempt_tool_name(call_name)
                if (
                    is_non_task
                    and configured_max_non_task_tools > 0
                    and non_task_count >= configured_max_non_task_tools
                ):
                    emit_limit_reached(
                        "tools",
                        {
                            "max_tools_used": configured_max_non_task_tools,
                            "used": non_task_count,
                            "tool": call_name,
                        },
                    )
                    outputs.append(
                        ToolMessage(
                            content=(
                                f"ERROR: Tool budget reached ({configured_max_non_task_tools}). "
                                "Finish using already gathered context."
                            ),
                            tool_call_id=call_id,
                            name=call_name,
                        )
                    )
                    events.append(
                        {
                            "tool": call_name,
                            "tool_input": args,
                            "status": "blocked",
                            "reason": "max_non_task_tools_reached",
                        }
                    )
                    continue

                tool = tool_map.get(call_name)
                mcp_tool = self._mcp_tools_by_name.get(call_name)
                if tool is None and mcp_tool is None:
                    outputs.append(
                        ToolMessage(
                            content=(
                                f"ERROR: Unknown tool '{call_name}'. Available tools: "
                                f"{', '.join(sorted({*tool_map, *self._mcp_tools_by_name}))}."
                            ),
                            tool_call_id=call_id,
                            name=call_name,
                        )
                    )
                    events.append(
                        {
                            "tool": call_name,
                            "tool_input": args,
                            "status": "error",
                            "reason": "unknown_tool",
                        }
                    )
                    continue

                try:
                    if tool is not None:
                        result = self._invoke_function_tool(tool, args)
                    else:
                        result = playwright_mcp_result_to_text(playwright_mcp_call_tool(call_name, args))
                    result = inject_budget_warning(result)
                    status = "ok"
                except Exception as exc:  # pragma: no cover
                    result = f"ERROR: {type(exc).__name__}: {exc}"
                    status = "error"

                if call_name == "task_steps_manager" and str(args.get("action", "")).strip().lower() == "init" and str(result).startswith("SUCCESS:"):
                    has_init = True
                if is_non_task:
                    non_task_count += 1
                    if configured_max_non_task_tools > 0:
                        result = f"{result}{_tool_budget_note(configured_max_non_task_tools - non_task_count, configured_max_non_task_tools)}"
                # Models stuck in a loop repeat the same call; Gemini CLI
                # detects this. Point it out so the agent changes approach.
                key = _call_key(call_name, args)
                repeats = recent_calls.count(key) + 1
                recent_calls = [*recent_calls, key][-_REPEAT_WINDOW:]
                if repeats >= 3:
                    result = (
                        f"{result}\n\n[Repeated call: this exact call was made {repeats} times in this run. "
                        "Reuse its earlier result or change approach instead of repeating it.]"
                    )

                outputs.append(ToolMessage(content=result, tool_call_id=call_id, name=call_name))
                events.append(
                    {
                        "tool": call_name,
                        "tool_input": args,
                        "status": status,
                    }
                )

            payload: dict[str, Any] = {
                "messages": outputs,
                "tool_events": events,
                "has_task_steps_manager_init": has_init,
                "non_task_tool_calls": non_task_count,
                "recent_calls": recent_calls,
            }
            return payload

        def nudge_for(message: Any) -> str:
            """The recovery prompt for a reply that must not end the run, or ''."""
            if (getattr(message, "response_metadata", None) or {}).get("finish_reason") == "length":
                return _LENGTH_NUDGE
            content = message.content
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
            if not text.strip() or text.strip() in ("[]", '""'):
                return _EMPTY_REPLY_NUDGE
            if self._output_schema_json and "{" not in text:
                return _UNFINISHED_NUDGE
            return ""

        def should_continue(state: _GraphState):
            messages = list(state.get("messages", []) or [])
            if not messages:
                return END
            last = messages[-1]
            if not isinstance(last, AIMessage):
                return END
            if getattr(last, "tool_calls", None) or getattr(last, "invalid_tool_calls", None):
                return "tool_node"
            if int(state.get("model_calls", 0) or 0) >= max(1, int(self._max_turns or 0)):
                return END
            nudges = int(state.get("nudges", 0) or 0)
            prompt = nudge_for(last)
            # A prose answer is questioned once; after that it stands.
            if prompt and nudges < (1 if prompt == _UNFINISHED_NUDGE else _MAX_NUDGES):
                return "nudge"
            return END

        def nudge(state: _GraphState) -> dict[str, Any]:
            last = list(state.get("messages", []) or [])[-1]
            prompt = nudge_for(last)
            # An empty assistant turn is dropped: some providers reject it
            # when it is sent back.
            dropped = [RemoveMessage(id=last.id)] if prompt == _EMPTY_REPLY_NUDGE and last.id else []
            _LOGGER.info("LangGraph reply would end the run unfinished; nudging: %s", prompt[:60])
            return {
                "messages": [*dropped, HumanMessage(content=prompt)],
                "nudges": int(state.get("nudges", 0) or 0) + 1,
            }

        builder = StateGraph(_GraphState)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tool_node", tool_node)
        builder.add_node("nudge", nudge)
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "nudge", END])
        builder.add_edge("tool_node", "llm_call")
        builder.add_edge("nudge", "llm_call")

        self._graph = builder.compile(checkpointer=InMemorySaver())

    @staticmethod
    def _log_tool_called(tool_name: str, tool_input: Any) -> None:
        try:
            log_event(
                "tool_called",
                payload={
                    "tool": tool_name,
                    "tool_input": tool_input,
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass


def _tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "params_json_schema", None)
    if not isinstance(schema, dict):
        schema = getattr(tool, "inputSchema", None)
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    description = str(getattr(tool, "description", "") or "")
    if not description:
        description = str(getattr(tool, "title", "") or "")
    return {
        "type": "function",
        "function": {
            "name": str(getattr(tool, "name", "") or ""),
            "description": description,
            "parameters": schema,
        },
    }


def _load_playwright_mcp_tools(config: ChackConfig) -> list[Any]:
    if not bool(getattr(config.tools, "playwright_enabled", False)):
        return []
    if not playwright_mcp_is_available():
        return []
    return list(playwright_mcp_list_tools())


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
) -> LangGraphExecutor:
    try:
        ChatOpenAI = _with_reasoning_passthrough(getattr(importlib.import_module("langchain_openai"), "ChatOpenAI"))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "langgraph backend requires langchain-openai and langgraph packages installed"
        ) from exc

    # LangGraph talks to an OpenAI-compatible Chat Completions endpoint:
    # the configured gateway when one is selected, otherwise OpenRouter.
    route = get_gateway_route(config) or get_gateway_route(
        replace(config, model=replace(config.model, provider="openrouter"))
    )
    if route is None:
        raise ValueError("model.provider=langgraph requires OPENROUTER_API_KEY or an LLM gateway")

    web_plan = (
        plan_web_tools(config, route=route, responses_api=False) if tools_override is None else WebToolPlan()
    )
    if tools_override is None:
        toolset = AgentsToolset(
            config.tools,
            model_provider=str(config.model.provider or ""),
            default_model=config.model.primary,
            social_network_model=config.model.social_network,
            scientific_model=config.model.scientific,
            websearcher_model=config.model.websearcher,
            business_model=config.model.business,
            product_model=config.model.product,
            travel_model=config.model.travel,
            cli_model=config.model.cli,
            subchack_model=config.model.subchack,
            researcher_administrator_model=config.model.researcher_administrator,
            social_network_max_turns=config.model.social_network_max_turns,
            scientific_max_turns=config.model.scientific_max_turns,
            websearcher_max_turns=config.model.websearcher_max_turns,
            business_max_turns=config.model.business_max_turns,
            product_max_turns=config.model.product_max_turns,
            travel_max_turns=config.model.travel_max_turns,
            cli_max_turns=config.model.cli_max_turns,
            subchack_max_turns=config.model.subchack_max_turns,
            researcher_administrator_max_turns=config.model.researcher_administrator_max_turns,
            self_critique_enabled=bool(getattr(config.agent, "self_critique_enabled", False)),
            keyless_web_tools=web_plan.keyless_tools,
            self_critique_rounds=int(getattr(config.agent, "self_critique_rounds", 0) or 0),
        )
        tools = list(toolset.tools)
        if tools_append:
            tools.extend(list(tools_append))
    else:
        tools = list(tools_override)

    tools = ToolCallPolicy(config.tools, outputs_readable=has_exec_tool(tools)).wrap(tools)

    has_task_steps_manager_tool = any(
        str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()
        == "task_steps_manager"
        for tool in tools
    )

    function_tools: dict[str, Any] = {}
    mcp_tools: dict[str, Any] = {}
    tool_schemas: list[dict[str, Any]] = []
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").strip()
        on_invoke = getattr(tool, "on_invoke_tool", None)
        if not name or on_invoke is None:
            continue
        if name in function_tools:
            continue
        function_tools[name] = tool
        tool_schemas.append(_tool_to_openai_schema(tool))

    for tool in _load_playwright_mcp_tools(config):
        name = str(getattr(tool, "name", "") or "").strip()
        if not name or name in function_tools or name in mcp_tools:
            continue
        mcp_tools[name] = tool
        tool_schemas.append(_tool_to_openai_schema(tool))

    timeout = _to_int(os.environ.get("CHACK_LANGGRAPH_MODEL_TIMEOUT_SECONDS", "120"), 120)
    model_name = str(route.model_name or "gpt-5")

    model = ChatOpenAI(
        model=model_name,
        api_key=route.api_key,
        base_url=route.base_url,
        timeout=timeout,
        default_headers=dict(route.headers) or None,
        reasoning_effort=normalize_thinking_effort(config.agent.thinking_effort),
        # Server tools would otherwise switch langchain to the Responses API,
        # whose replies this executor does not read; OpenRouter accepts them
        # on Chat Completions too.
        use_responses_api=False,
    )
    if web_plan.server_tools:
        # Tools the provider runs itself (e.g. OpenRouter's web search and
        # fetch) are sent as given; ``bind_tools`` only accepts functions.
        model_with_tools = model.bind(
            tools=[*(convert_to_openai_tool(schema) for schema in tool_schemas), *web_plan.server_tools]
        )
    else:
        model_with_tools = model.bind_tools(tool_schemas) if tool_schemas else model

    thread_id = str(current_session_id() or f"langgraph-{uuid.uuid4()}")

    executor = LangGraphExecutor(
        _graph=None,
        _model_with_tools=model_with_tools,
        _summary_model=model,
        _function_tools_by_name=function_tools,
        _mcp_tools_by_name=mcp_tools,
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _thread_id=thread_id,
        _max_non_task_tools=max(0, int(config.tools.max_tools_used or 0)),
        _require_task_steps_manager_init_first=bool(
            getattr(config.agent, "require_task_steps_manager_init_first", True)
            and has_task_steps_manager_tool
        ),
        _max_turns=max_turns,
        _summary_trigger_messages=memory_max_messages,
        _summary_keep_messages=memory_reset_to_messages,
        _summary_max_chars=max(
            0,
            int(memory_summary_max_chars)
            if int(memory_summary_max_chars) > 0
            else _to_int(os.environ.get("CHACK_LANGGRAPH_SUMMARY_MAX_CHARS", "6000"), 6000),
        ),
        _compaction_threshold_ratio=float(getattr(config.agent, "compaction_threshold_ratio", 0.75) or 0.75),
        _max_context_tokens=int(config.model.max_context_tokens or 0)
        or _to_int(os.environ.get("CHACK_LANGGRAPH_CONTEXT_TOKENS", ""), _DEFAULT_CONTEXT_TOKENS),
        _output_schema_json=(
            json.dumps(getattr(config.agent, "output_schema_json", None), ensure_ascii=False, indent=2)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        _output_schema_name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
        _output_schema_strict=bool(getattr(config.agent, "output_schema_strict", True)),
    )
    try:
        executor.build_graph()
    except Exception as exc:
        raise RuntimeError(
            "langgraph backend requires langgraph packages installed and importable"
        ) from exc
    return executor
