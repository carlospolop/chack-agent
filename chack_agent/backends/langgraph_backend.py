from __future__ import annotations

import asyncio
import importlib
import json
import os
from operator import add
import threading
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from agents.tool_context import ToolContext
from agents.usage import Usage
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_steps_manager_state import current_run_label, current_session_id
from chack_tools.telemetry import log_event
from chack_tools.tool_usage_state import current_max_tools_used

from ..config import ChackConfig


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]


class _GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add]
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
    _conversation: list[dict[str, Any]]
    _memory_limit: int
    _memory_reset_to: int
    _base_system_prompt: str
    _thread_id: str
    _max_non_task_tools: int
    _require_task_steps_manager_init_first: bool
    _recursion_limit: int
    _summary_trigger_messages: int
    _summary_keep_messages: int
    _summary_max_chars: int

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        if not user_input.strip():
            return {"output": "", "intermediate_steps": [], "raw_result": _RawResult(raw_responses=[])}

        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self._recursion_limit,
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

        result_state = self._graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        state_values = dict(result_state or {})

        messages = state_values.get("messages", []) or []
        output = self._extract_output(messages)

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

        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": _RawResult(raw_responses=raw_responses),
        }

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

    def _summarize_messages(self, previous_summary: str, messages: list[AnyMessage]) -> str:
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

        response = self._summary_model.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", "")
        if isinstance(content, str):
            summary = content.strip() or previous_summary
        elif isinstance(content, list):
            summary = "\n".join(str(part) for part in content).strip() or previous_summary
        else:
            summary = str(content or previous_summary).strip() or previous_summary
        if self._summary_max_chars > 0 and len(summary) > self._summary_max_chars:
            summary = summary[-self._summary_max_chars :]
        return summary

    def _system_prompt(self, summary: str) -> str:
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

        policy_block = ""
        if policy_lines:
            policy_block = "\n\n### TOOL USAGE POLICY\n" + "\n".join(policy_lines)

        summary_block = ""
        if summary:
            summary_block = f"\n\n### MEMORY SUMMARY\n{summary}"

        return f"{base}{policy_block}{summary_block}".strip()

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

            if len(messages) > max(self._summary_trigger_messages, self._summary_keep_messages + 2):
                older = messages[:-self._summary_keep_messages]
                maybe_new_summary = self._summarize_messages(summary, older)

            prompt_messages: list[AnyMessage] = [
                SystemMessage(content=self._system_prompt(maybe_new_summary))
            ]
            prompt_messages.extend(messages[-self._summary_keep_messages:])

            response = self._model_with_tools.invoke(prompt_messages)
            usage = getattr(response, "usage_metadata", None) or {}
            usage_event = {
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "cached_input_tokens": int(usage.get("input_token_details", {}).get("cache_read", 0) or 0),
                "cache_write_input_tokens": int(usage.get("input_token_details", {}).get("cache_creation", 0) or 0),
            }
            payload: dict[str, Any] = {
                "messages": [response],
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

                is_non_task = call_name != "task_steps_manager"
                if (
                    is_non_task
                    and configured_max_non_task_tools > 0
                    and non_task_count >= configured_max_non_task_tools
                ):
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
                if tool is None:
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
                    result = self._invoke_function_tool(tool, args)
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
        schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": str(getattr(tool, "name", "") or ""),
            "description": str(getattr(tool, "description", "") or ""),
            "parameters": schema,
        },
    }


def build_executor(
    config: ChackConfig,
    *,
    system_prompt: str,
    max_turns: int,
    memory_max_messages: int,
    memory_reset_to_messages: int,
    tool_profile: str = "all",
    tools_override: Optional[list[Any]] = None,
    tools_append: Optional[list[Any]] = None,
) -> LangGraphExecutor:
    try:
        ChatOpenAI = getattr(importlib.import_module("langchain_openai"), "ChatOpenAI")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "langgraph backend requires langchain-openai and langgraph packages installed"
        ) from exc

    openrouter_api_key = (
        str(config.credentials.openrouter_api_key or "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required when model.provider=langgraph")

    if tools_override is None:
        toolset = AgentsToolset(
            config.tools,
            tool_profile=tool_profile,
            model_provider=str(config.model.provider or ""),
            default_model=config.model.primary,
            social_network_model=config.model.social_network,
            scientific_model=config.model.scientific,
            websearcher_model=config.model.websearcher,
            tester_model=config.model.tester,
            social_network_max_turns=config.model.social_network_max_turns,
            scientific_max_turns=config.model.scientific_max_turns,
            websearcher_max_turns=config.model.websearcher_max_turns,
            tester_max_turns=config.model.tester_max_turns,
        )
        tools = list(toolset.tools)
        if tools_append:
            tools.extend(list(tools_append))
    else:
        tools = list(tools_override)

    function_tools: dict[str, Any] = {}
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

    base_url = (
        str(config.credentials.openrouter_base_url or "").strip()
        or os.environ.get("OPENROUTER_BASE_URL", "").strip()
        or "https://openrouter.ai/api/v1"
    )
    http_referer = (
        str(config.credentials.openrouter_http_referer or "").strip()
        or os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    )
    app_name = (
        str(config.credentials.openrouter_app_name or "").strip()
        or os.environ.get("OPENROUTER_APP_NAME", "").strip()
    )
    default_headers: dict[str, str] = {}
    if http_referer:
        default_headers["HTTP-Referer"] = http_referer
    if app_name:
        default_headers["X-Title"] = app_name
    timeout = _to_int(os.environ.get("CHACK_LANGGRAPH_MODEL_TIMEOUT_SECONDS", "120"), 120)
    model_name = str(config.model.primary or "gpt-5")

    model = ChatOpenAI(
        model=model_name,
        api_key=openrouter_api_key,
        base_url=base_url,
        timeout=timeout,
        default_headers=default_headers or None,
    )
    model_with_tools = model.bind_tools(tool_schemas) if tool_schemas else model

    max_messages = max(1, int(memory_max_messages or 1))
    reset_to = int(memory_reset_to_messages or max_messages)
    if reset_to < 1 or reset_to > max_messages:
        reset_to = max_messages

    thread_id = str(current_session_id() or f"langgraph-{uuid.uuid4()}")
    recursion_limit = max(8, _to_int(os.environ.get("CHACK_LANGGRAPH_RECURSION_LIMIT", "0"), 0))
    if recursion_limit <= 8:
        recursion_limit = max(16, int(max_turns or 8) * 4)

    summary_trigger_messages = max(8, int(memory_max_messages or 0))
    summary_keep_messages = max(4, int(memory_reset_to_messages or 0))
    if summary_keep_messages > summary_trigger_messages:
        summary_keep_messages = summary_trigger_messages

    executor = LangGraphExecutor(
        _graph=None,
        _model_with_tools=model_with_tools,
        _summary_model=model,
        _function_tools_by_name=function_tools,
        _conversation=[],
        _memory_limit=max_messages,
        _memory_reset_to=reset_to,
        _base_system_prompt=system_prompt,
        _thread_id=thread_id,
        _max_non_task_tools=max(0, int(config.tools.max_tools_used or 0)),
        _require_task_steps_manager_init_first=bool(
            getattr(config.agent, "require_task_steps_manager_init_first", True)
        ),
        _recursion_limit=recursion_limit,
        _summary_trigger_messages=summary_trigger_messages,
        _summary_keep_messages=summary_keep_messages,
        _summary_max_chars=max(
            0,
            _to_int(os.environ.get("CHACK_LANGGRAPH_SUMMARY_MAX_CHARS", "6000"), 6000),
        ),
    )
    try:
        executor.build_graph()
    except Exception as exc:
        raise RuntimeError(
            "langgraph backend requires langgraph packages installed and importable"
        ) from exc
    return executor
