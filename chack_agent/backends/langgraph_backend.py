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
from agents.exceptions import ModelBehaviorError
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
from ..output_schema import JsonSchemaOutput
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


def _to_int(raw: str, default: int) -> int:
    try:
        return int(str(raw or "").strip())
    except Exception:
        return default


def _recursion_limit_for(max_turns: int) -> int:
    # A turn is one model call, as in the Agents SDK backends. In the graph a
    # call that uses tools takes two steps (llm_call, then tool_node), so the
    # step limit is twice the turn budget.
    return 2 * max(1, int(max_turns or 0))


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

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        if not user_input.strip():
            return {"output": "", "intermediate_steps": [], "raw_result": _RawResult(raw_responses=[])}

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

        graph_input: dict[str, Any] = {
            "messages": [HumanMessage(content=user_input)]
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
                    {"messages": [HumanMessage(content=self._schema_repair_prompt(schema_error))]},
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
            raw_responses.extend(compacted.raw_responses)
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

        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": _RawResult(raw_responses=raw_responses),
        }

    def _invoke_graph(self, graph_input: dict[str, Any], config: dict[str, Any]) -> Any:
        graph_errors = importlib.import_module("langgraph.errors")
        try:
            return self._graph.invoke(graph_input, config=config)
        except graph_errors.GraphRecursionError as exc:
            raise MaxTurnsExceeded(f"Max turns ({self._max_turns}) exceeded") from exc

    def _validate_schema_output(self, output: str) -> tuple[str, str]:
        """Return the answer as schema-shaped JSON, or as given with the reason it fails."""
        validator = JsonSchemaOutput(
            json.loads(self._output_schema_json),
            name=str(self._output_schema_name or "output_schema"),
            strict=bool(self._output_schema_strict),
        )
        try:
            return json.dumps(validator.validate_json(output or ""), ensure_ascii=False), ""
        except ModelBehaviorError as exc:
            return output, str(exc)

    def _schema_repair_prompt(self, schema_error: str) -> str:
        schema_name = str(self._output_schema_name or "output_schema").strip() or "output_schema"
        return (
            f"Your final answer does not match the required output schema `{schema_name}`: {schema_error}\n"
            "Reply again with only the final answer as one JSON object that matches the schema, "
            "with no text before or after it."
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
            rendered.append(f"[{role}] {content}")
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
        if previous_summary:
            prompt = (
                "You maintain compact conversation memory.\n"
                "Existing summary:\n"
                f"{previous_summary}\n\n"
                "New messages:\n"
                f"{chunk}\n\n"
                "Return an updated concise summary preserving key facts, decisions, and constraints."
                f"{focus_block}"
            )
        else:
            prompt = (
                "Summarize the following conversation history concisely, preserving key facts, decisions, and constraints:\n"
                f"{chunk}{focus_block}"
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
            summary = summary[-self._summary_max_chars :]
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

    def _system_prompt(self, summary: str, *, remaining_tools: int | None = None) -> str:
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
        if remaining_tools is not None:
            if remaining_tools == 0:
                policy_lines.append("- The tool-call budget is exhausted. Answer from the evidence already gathered; do not call another tool.")
            else:
                policy_lines.append(
                    f"- {remaining_tools} non-task tool calls remain. Combine related checks and do not repeat a successful lookup."
                )
        available_tools = (*self._function_tools_by_name, *self._mcp_tools_by_name)
        if any("web" in name.lower() or "fetch" in name.lower() for name in available_tools):
            policy_lines.append(
                "- When the task asks you to check a current external source, use an available web tool before citing it."
            )

        policy_block = ""
        if policy_lines:
            policy_block = "\n\n### TOOL USAGE POLICY\n" + "\n".join(policy_lines)

        summary_block = ""
        if summary:
            summary_block = f"\n\n### MEMORY SUMMARY\n{summary}"
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

        return f"{base}{policy_block}{summary_block}{output_schema_block}".strip()

    def build_graph(self) -> None:
        checkpoint_mod = importlib.import_module("langgraph.checkpoint.memory")
        graph_mod = importlib.import_module("langgraph.graph")
        InMemorySaver = getattr(checkpoint_mod, "InMemorySaver")
        StateGraph = getattr(graph_mod, "StateGraph")
        START = getattr(graph_mod, "START")
        END = getattr(graph_mod, "END")

        tool_map = self._function_tools_by_name

        def llm_call(state: _GraphState) -> dict[str, Any]:
            messages = list(state.get("messages", []) or [])
            summary = str(state.get("summary", "") or "")
            maybe_new_summary = summary
            usage_events = list(state.get("usage_events", []) or [])
            latest_input_tokens = 0
            if usage_events:
                last_usage = usage_events[-1] if isinstance(usage_events[-1], dict) else {}
                latest_input_tokens = int(last_usage.get("input_tokens", 0) or 0)

            threshold_tokens = 0
            if self._max_context_tokens > 0 and self._compaction_threshold_ratio > 0:
                threshold_tokens = int(self._compaction_threshold_ratio * self._max_context_tokens)

            should_summarize_by_messages = (
                len(messages) > max(self._summary_trigger_messages, self._summary_keep_messages + 2)
            )
            should_summarize_by_tokens = bool(
                threshold_tokens > 0 and latest_input_tokens >= threshold_tokens
            )

            older: list[AnyMessage] = []
            start = 0
            if should_summarize_by_messages or should_summarize_by_tokens:
                # Keep the current request even when the configured tail size
                # is zero. Never start the retained tail with a tool result:
                # its matching assistant tool call must remain beside it.
                keep = max(1, self._summary_keep_messages)
                start = max(0, len(messages) - keep)
                while start > 0 and isinstance(messages[start], ToolMessage):
                    start -= 1
                older = messages[:start]
                maybe_new_summary = self._summarize_messages(
                    summary,
                    older,
                    trigger_input_tokens=latest_input_tokens,
                    threshold_tokens=threshold_tokens,
                    messages_before=len(messages),
                    messages_after_keep=len(messages) - start,
                    triggered_by_messages=should_summarize_by_messages,
                    triggered_by_tokens=should_summarize_by_tokens,
                )

            tool_limit = int(current_max_tools_used() or self._max_non_task_tools or 0)
            remaining = max(0, tool_limit - int(state.get("non_task_tool_calls", 0) or 0)) if tool_limit > 0 else None
            prompt_messages: list[AnyMessage] = [
                SystemMessage(content=self._system_prompt(maybe_new_summary, remaining_tools=remaining))
            ]
            # Until a summary exists, every message is still part of the live
            # conversation. Silently keeping only the tail here discards facts
            # before the configured compaction threshold is reached.
            if maybe_new_summary:
                prompt_messages.extend(messages[start:] if older else messages)
            else:
                prompt_messages.extend(messages)

            raise_if_cancelled()
            response = self._model_with_tools.invoke(prompt_messages)
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
                "messages": [RemoveMessage(id=m.id) for m in older] + [response] if older else [response],
                "usage_events": [usage_event],
            }
            if maybe_new_summary != summary:
                payload["summary"] = maybe_new_summary
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

            outputs: list[ToolMessage] = []
            events: list[dict[str, Any]] = []

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
                            content=f"ERROR: Unknown tool '{call_name}'.",
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
            }
            return payload

        def should_continue(state: _GraphState):
            messages = list(state.get("messages", []) or [])
            if not messages:
                return END
            last = messages[-1]
            if isinstance(last, AIMessage) and (getattr(last, "tool_calls", None) or []):
                return "tool_node"
            return END

        builder = StateGraph(_GraphState)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tool_node", tool_node)
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        builder.add_edge("tool_node", "llm_call")

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
        ChatOpenAI = getattr(importlib.import_module("langchain_openai"), "ChatOpenAI")
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
        _max_context_tokens=int(config.model.max_context_tokens or 0),
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
