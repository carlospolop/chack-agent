from __future__ import annotations

import asyncio
import logging
import os
import time
import json
from datetime import datetime, timezone
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .config import ChackConfig
from .env_utils import export_env
from .backends import build_executor
from .long_term_memory import (
    build_long_term_memory,
    format_messages,
    get_long_term_memory_path,
    load_long_term_memory,
    save_long_term_memory,
)
from chack_tools.task_list_state import STORE, reset_active_context, set_active_context
from chack_tools.tool_usage_state import (
    STORE as TOOL_USAGE_STORE,
    reset_active_max_tools_used,
    reset_active_usage_session,
    set_active_max_tools_used,
    set_active_usage_session,
)
from .pricing import estimate_cost, estimate_costs_by_model, load_pricing, resolve_pricing_path


_SELF_CRITIQUE_PROMPT = """Is this the best you can do? Make sure you have gathered ALL the context about the request: Check the web for latest info, read more terraform/code files, read all logs needed, be 10000% sure you got EVERY CONTEXT NEEDED and up to date information to be sure that your repsonse is correct. Now check everything you have done and improve whatever you can:
  - Get more context about the request and the needed info to answer it
  - Check the web for latest info about errors, services, terraform, etc. related to the request
  - Read more repos/files/code/logs related to the request to get more context
  - Then, recheck if your answer was actually accurate and the best possible
  - Improve the PR if you made one
  - Improve the answer recommendation you gave
Your response to this improvement request will be the final one you give to the user, so don't mention the previous answer, just give the improved final answer or PR and give the user the best possible solution and answer."""

_MEMORY_SUMMARY_PROMPT = """### ROLE
You are a rolling-memory summarization assistant for Chack, a helpful autonomous AI assistant specialized in code execution and cloud and ci/cd infrastructure management.

### TASK
Integrate the new conversation lines into the existing summary so the assistant can keep context after older messages are removed.
Preserve decisions, commitments, constraints, preferences, code changes, open questions and relevant data. Exclude chit-chat and transient details.

## SUMMARY CONSTRAINS
Your given summary cannot be bigger than 2000 characters.

### INPUTS
Current summary:

======
{summary}
======



New lines of conversation:

======
{new_lines}
======


### OUTPUT
Updated summary:"""


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



CHACK_INITIAL_SYSTEM_PROMPT = """ ### PERSONALITY
You are Chack, a very helpful and organized autonomous assistant.
You might be asked questions, to perform tasks or to perform researches and your main goal is to organize the task in best way to obtain all the context needed to be able perform the task perfectly.
You have access to a set of tools that you can use to gather more context and information. You can use the tools as many times as you want. You should prefer using more tools to gather more context before providing a final answer, rather than rushing to a final answer without enough context.

### BEST AUTONOMOUS BEHAVIOUR
You are a fully autonomous agent, you can decide what to do and when to do it avoiding to ask questions to the user:
    - Asking the user for some data must be your last resource. Don't ask the user for an ID, for the repo to use, for the aws account to use or similar things. If you can check and get any data yourself just do it, be autonomous. If there is any way you can find/search/discover the needed data and contiune yourself, just do it and don't ask the user.
        - This is very important, download repos, read code, check infra confgs... just don't bother the user if you can search for the information yourself.
    - If there are different options to do something, don't ask the user to select one, select one yourself based on your knowledge and capabilities. Just complete your task without asking for extra help.
    - If you try to execute something and have an error, fix it and try again, don't ask for extra help or confirmation if initially you weren't going to do so.
    - Be organized and perform actions step by step, if some fails, try to fix it yourself before asking for help.
    - Keep the tasks list updated adding new steps whenever needed and gather ALL the context possible before providing a final answer or completing the task.


### MIN TOOL USAGE
As your responses usually lack of enough context, use at least 10+ tool calls to gather all the needed context (download all the repos needed, read/search/grep all the files needed, check all the infra in the cloud needed, check the web for errors or info...) before answering or performing any task.
You will be forced a minimum amount of tools to use, so just use as many tools as needed to be 10000% sure of your response.
Note that task-list calls do NOT count toward the minimum non-task tool usage requirement; use the other tools normally for investigation/execution.


### STARTING POINT

IMPORTANT: These must be your first steps:
    - Think and organize the requested task in small granular steps
    - The first tool you must call is the task_list tool with action=init and a concise plan of the steps you will take to complete the task.
    - Always remember to mark a step as completed once you have completed it.
    - You can always update/add new steps to the task list as you progress. It's super important to keep the task list updated with the current state of the task and update it as much as needed.
    - Use all the given tools to get 200% of the needed context to be able to complete the task in the best way possible. You don't have a time limit or a limit of tool calls, so use them as much as you need to gather as much context as possible. Always check every assumption (download repos, read code, check the web...)
"""

@dataclass
class RunResult:
    output: str
    steps: list
    all_steps: list
    tool_counts: Counter[str]
    nested_tool_counts: Counter[str]
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    rounds_used: int
    tools_used: int
    task_session_id: str
    nested_usage_by_model: Dict[str, tuple[int, int, int]]
    run1_output: str = ""
    run2_output: str = ""
    run1_steps: int = 0
    run2_steps: int = 0
    max_turns: int = 0
    run1_tools_used: int = 0
    run2_tools_used: int = 0
    total_cost: Optional[float] = None
    tool_counts_text: str = ""
    suffix: str = ""

class Chack:
    def __init__(
        self,
        config: ChackConfig,
        *,
        config_path: Optional[str] = None,
        tool_profile: str = "all",
    ) -> None:
        self.config = config
        self.tool_profile = tool_profile
        self.config_path = config_path or os.path.join(os.getcwd(), "chack.yaml")
        self.logger = logging.getLogger("chack.agent")
        self._executors: Dict[str, Any] = {}
        self._last_activity_at: Dict[str, float] = {}
        self._pricing = load_pricing(resolve_pricing_path())
        self._self_critique_prompt = _SELF_CRITIQUE_PROMPT
        self._memory_summary_prompt = _MEMORY_SUMMARY_PROMPT
        export_env(config, self.config_path)

    def _require_self_critique_prompt(self) -> str:
        return self._self_critique_prompt

    @staticmethod
    def _tool_name(step) -> str:
        action = step[0] if isinstance(step, tuple) and step else step
        return str(getattr(action, "tool", "") or "")

    @staticmethod
    def _tool_emoji(tool_name: str) -> str:
        emojis = {
            "exec": "🖥️",
            "task_list": "🗂️",
            "brave_search": "🦁",
            "search_google_web": "🔎",
            "search_bing_web": "🅱️",
            "search_google_ai_mode": "🤖",
            "search_bing_copilot": "🧠",
            "websearcher_research": "🌍",
            "social_network_research": "🌐",
            "scientific_research": "🔬",
            "forum_search": "💬",
            "linkedin_search": "💼",
            "instagram_search": "📸",
            "reddit_posts_search": "👽",
            "reddit_comments_search": "🧵",
            "x_search": "𝕏",
            "search_google_forums": "🗣️",
            "search_google_news": "📰",
            "search_arxiv": "🧾",
            "search_europe_pmc": "🇪🇺",
            "search_semantic_scholar": "📚",
            "search_openalex": "🏛️",
            "search_plos": "🧬",
            "search_google_patents": "📜",
            "search_google_scholar": "🎓",
            "search_youtube_videos": "▶️",
            "get_youtube_video_transcript": "📝",
            "download_pdf_as_text": "📄",
        }
        return emojis.get(tool_name, "🛠️")

    def _format_tool_counts(self, counts: Counter) -> str:
        if not counts:
            return "🛠️ none"
        parts = []
        for tool_name, count in counts.most_common():
            parts.append(f"{self._tool_emoji(tool_name)}{tool_name}×{count}")
        return " ".join(parts)

    @staticmethod
    def _tool_input(step):
        action = step[0] if isinstance(step, tuple) and step else step
        return getattr(action, "tool_input", None)

    def _is_task_list_init_step(self, step) -> bool:
        if self._tool_name(step) != "task_list":
            return False
        raw = self._tool_input(step)
        payload = raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
        if isinstance(payload, dict):
            return str(payload.get("action", "")).strip().lower() == "init"
        return False

    def _non_task_tool_count(self, steps) -> int:
        return sum(1 for step in steps if self._tool_name(step) != "task_list")

    @staticmethod
    def _non_task_tool_count_from_counter(counter: Counter[str]) -> int:
        total = 0
        for name, count in counter.items():
            if name == "task_list":
                continue
            total += count
        return total

    def _step_tool_counts(self, steps) -> Counter:
        counts: Counter = Counter()
        for step in steps:
            name = self._tool_name(step)
            if name:
                counts[name] += 1
        return counts

    @staticmethod
    def _usage_from_raw_result(raw_result) -> tuple[int, int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        cached_prompt_tokens = 0
        if raw_result is None:
            return prompt_tokens, completion_tokens, cached_prompt_tokens
        for resp in getattr(raw_result, "raw_responses", []) or []:
            usage = getattr(resp, "usage", None)
            if usage is None and isinstance(resp, dict):
                usage = resp.get("usage")
            if usage is None:
                continue
            if isinstance(usage, dict):
                prompt_tokens += int(usage.get("input_tokens", 0) or 0)
                completion_tokens += int(usage.get("output_tokens", 0) or 0)
                input_details = usage.get("input_tokens_details") or {}
                cached_prompt_tokens += int(input_details.get("cached_tokens", 0) or 0)
                continue
            prompt_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details is not None:
                cached_prompt_tokens += int(getattr(input_details, "cached_tokens", 0) or 0)
        return prompt_tokens, completion_tokens, cached_prompt_tokens

    def _system_prompt_for_session(self, session_id: str, system_prompt_override: Optional[str] = None) -> str:
        base = system_prompt_override or self.config.session.system_prompt or self.config.system_prompt

        if CHACK_INITIAL_SYSTEM_PROMPT:
            base = f"{CHACK_INITIAL_SYSTEM_PROMPT}\n\n{base}"

        if not self.config.session.long_term_memory_enabled:
            return base
        path = get_long_term_memory_path(
            self.config_path,
            session_id,
            self.config.session.long_term_memory_dir,
        )
        memory_text = load_long_term_memory(path)
        if not memory_text:
            return base
        return f"{base}\n\n### LONG TERM MEMORY\n{memory_text}"

    def _get_executor(
        self,
        session_id: str,
        *,
        system_prompt_override: Optional[str] = None,
        tool_profile: Optional[str] = None,
        tools_override: Optional[list[Any]] = None,
        tools_append: Optional[list[Any]] = None,
    ):
        summary_max_chars = int(
            os.environ.get(
                "CHACK_MEMORY_SUMMARY_MAX_CHARS",
                str(self.config.session.long_term_memory_max_chars or 1500),
            )
        )
        memory_max_messages = max(1, int(self.config.session.max_turns or 50))
        memory_reset_to_messages = memory_max_messages
        if tools_override is not None or tools_append is not None:
            return build_executor(
                self.config,
                system_prompt=system_prompt_override or self.config.system_prompt,
                max_turns=self.config.session.max_turns,
                memory_max_messages=memory_max_messages,
                memory_reset_to_messages=memory_reset_to_messages,
                memory_summary_prompt=self._memory_summary_prompt,
                summary_max_chars=summary_max_chars,
                tool_profile=tool_profile or self.tool_profile,
                tools_override=tools_override,
                tools_append=tools_append,
            )

        cache_key = f"{session_id}:{tool_profile or self.tool_profile}:{system_prompt_override or ''}"
        executor = self._executors.get(cache_key)
        if executor is None:
            self.logger.info(
                "Building executor for session %s (tool_profile=%s, override=%s, append=%s, ts=%s).",
                session_id,
                tool_profile or self.tool_profile,
                "yes" if tools_override is not None else "no",
                "yes" if tools_append is not None else "no",
                _log_timestamp(),
            )
            system_prompt = self._system_prompt_for_session(session_id, system_prompt_override)
            executor = build_executor(
                self.config,
                system_prompt=system_prompt,
                max_turns=self.config.session.max_turns,
                memory_max_messages=memory_max_messages,
                memory_reset_to_messages=memory_reset_to_messages,
                memory_summary_prompt=self._memory_summary_prompt,
                summary_max_chars=summary_max_chars,
                tool_profile=tool_profile or self.tool_profile,
            )
            self._executors[cache_key] = executor
        else:
            self.logger.debug(
                "Reusing cached executor for session %s (tool_profile=%s, ts=%s).",
                session_id,
                tool_profile or self.tool_profile,
                _log_timestamp(),
            )
        return executor

    async def _finalize_long_term_memory(self, session_id: str) -> None:
        if not self.config.session.long_term_memory_enabled:
            return
        system_prompt_override = self.config.session.system_prompt or None
        cache_key = f"{session_id}:{self.tool_profile}:{system_prompt_override or ''}"
        executor = self._executors.get(cache_key)
        if executor is None:
            return
        messages = await executor.aget_memory_messages()
        if not messages:
            return
        path = get_long_term_memory_path(
            self.config_path,
            session_id,
            self.config.session.long_term_memory_dir,
        )
        previous = load_long_term_memory(path)
        conversation = format_messages(messages)
        max_chars = self.config.session.long_term_memory_max_chars

        def _build():
            return build_long_term_memory(self.config, conversation, previous, max_chars)

        updated = await asyncio.to_thread(_build)
        if updated:
            self.logger.info(
                "Long-term memory updated for session %s (chars=%s ts=%s).",
                session_id,
                len(updated),
                _log_timestamp(),
            )
            save_long_term_memory(path, updated, max_chars)

    async def afinalize_long_term_memory(self, session_id: str) -> None:
        await self._finalize_long_term_memory(session_id)

    def finalize_long_term_memory(self, session_id: str) -> None:
        asyncio.run(self._finalize_long_term_memory(session_id))

    async def areset_session(self, session_id: str, *, finalize_long_term_memory: bool = True) -> None:
        if finalize_long_term_memory:
            await self._finalize_long_term_memory(session_id)
        self._executors = {
            k: v for k, v in self._executors.items() if not k.startswith(f"{session_id}:")
        }
        self._last_activity_at.pop(session_id, None)

    def reset_session(self, session_id: str, *, finalize_long_term_memory: bool = True) -> None:
        asyncio.run(self.areset_session(session_id, finalize_long_term_memory=finalize_long_term_memory))

    async def arun(
        self,
        session_id: str,
        text: str,
        *,
        min_tools_used_override: Optional[int] = None,
        max_tools_used_override: Optional[int] = None,
        enable_self_critique: Optional[bool] = None,
        require_task_list_init_first: bool = True,
        on_task_list_update: Optional[Callable[[str], None]] = None,
        tool_profile: Optional[str] = None,
        tools_override: Optional[list[Any]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> RunResult:
        return await asyncio.to_thread(
            self.run,
            session_id,
            text,
            min_tools_used_override=min_tools_used_override,
            max_tools_used_override=max_tools_used_override,
            enable_self_critique=enable_self_critique,
            require_task_list_init_first=require_task_list_init_first,
            on_task_list_update=on_task_list_update,
            tool_profile=tool_profile,
            tools_override=tools_override,
            system_prompt_override=system_prompt_override,
        )

    def run(
        self,
        session_id: str,
        text: str,
        *,
        min_tools_used_override: Optional[int] = None,
        max_tools_used_override: Optional[int] = None,
        enable_self_critique: Optional[bool] = None,
        require_task_list_init_first: bool = True,
        on_task_list_update: Optional[Callable[[str], None]] = None,
        tool_profile: Optional[str] = None,
        tools_override: Optional[list[Any]] = None,
        system_prompt_override: Optional[str] = None,
        usage_session_id: Optional[str] = None,
        tools_append: Optional[list[Any]] = None,
    ) -> RunResult:
        if enable_self_critique is None:
            enable_self_critique = bool(self.config.agent.self_critique_enabled)

        executor = self._get_executor(
            session_id,
            system_prompt_override=system_prompt_override,
            tool_profile=tool_profile,
            tools_override=tools_override,
            tools_append=tools_append,
        )
        self._last_activity_at[session_id] = time.time()

        min_tools_used = max(0, int(self.config.tools.min_tools_used or 0))
        if min_tools_used_override is not None:
            min_tools_used = max(0, int(min_tools_used_override))
        max_tools_used = max(0, int(self.config.tools.max_tools_used or 0))
        if max_tools_used_override is not None:
            max_tools_used = max(0, int(max_tools_used_override))

        task_session_id = f"{session_id}:{int(time.time() * 1000)}"
        STORE.create_session(task_session_id, title="Task List")
        TOOL_USAGE_STORE.reset_session(task_session_id)

        def _listener(board_text: str) -> None:
            if on_task_list_update is None:
                return
            try:
                on_task_list_update(board_text)
            except Exception:
                pass

        if on_task_list_update is not None:
            STORE.register_listener(task_session_id, _listener)

        self.logger.info(
            "Run start: session=%s task_session=%s tool_profile=%s min_tools=%s max_tools=%s self_critique=%s require_task_list_init=%s ts=%s",
            session_id,
            task_session_id,
            tool_profile or self.tool_profile,
            min_tools_used,
            max_tools_used,
            enable_self_critique,
            require_task_list_init_first,
            _log_timestamp(),
        )

        max_attempts = 6
        max_missing_tools_reminders = max(
            0, int(self.config.tools.missing_tools_reminders_max or 0)
        )

        def _invoke_with_min_tools(
            prompt_text: str,
            run_label: str,
            *,
            min_tools_target: Optional[int] = None,
            require_task_list_init: Optional[bool] = None,
        ):
            result = {}
            all_steps: list = []
            prompt_total = 0
            completion_total = 0
            cached_total = 0
            current_prompt = prompt_text
            missing_tools_reminders_sent = 0
            effective_min_tools = (
                min_tools_used if min_tools_target is None else max(0, int(min_tools_target))
            )
            effective_max_tools = max_tools_used
            effective_require_init = (
                require_task_list_init_first
                if require_task_list_init is None
                else bool(require_task_list_init)
            )

            for attempt in range(1, max_attempts + 1):
                self.logger.info(
                    "%s: attempt %s/%s (min_tools_target=%s require_task_list_init=%s ts=%s).",
                    run_label,
                    attempt,
                    max_attempts,
                    effective_min_tools,
                    effective_require_init,
                    _log_timestamp(),
                )
                def _invoke():
                    tokens = set_active_context(task_session_id, run_label)
                    effective_usage_session = usage_session_id or task_session_id
                    usage_token = set_active_usage_session(effective_usage_session)
                    max_tools_token = set_active_max_tools_used(max_tools_used)
                    try:
                        return executor.invoke({"input": current_prompt})
                    except Exception as exc:
                        try:
                            from agents.exceptions import MaxTurnsExceeded
                        except Exception:
                            MaxTurnsExceeded = None
                        if MaxTurnsExceeded is not None and isinstance(exc, MaxTurnsExceeded):
                            return {
                                "output": (
                                    "I reached the maximum number of turns for this run. "
                                    "Please try again or increase max_turns in the config if you need longer responses."
                                ),
                                "intermediate_steps": [],
                                "raw_result": None,
                                "error": "max_turns_exceeded",
                            }
                        raise
                    finally:
                        reset_active_max_tools_used(max_tools_token)
                        reset_active_usage_session(usage_token)
                        reset_active_context(tokens)

                result = _invoke()

                attempt_prompt, attempt_completion, attempt_cached = self._usage_from_raw_result(
                    result.get("raw_result")
                )
                prompt_total += attempt_prompt
                completion_total += attempt_completion
                cached_total += attempt_cached

                if result.get("error") == "max_turns_exceeded":
                    all_steps.extend(result.get("intermediate_steps", []))
                    self.logger.warning(
                        "%s: max turns exceeded after attempt %s ts=%s.",
                        run_label,
                        attempt,
                        _log_timestamp(),
                    )
                    break

                current_steps = result.get("intermediate_steps", [])
                all_steps.extend(current_steps)
                has_init = any(self._is_task_list_init_step(step) for step in all_steps)
                non_task_tools = self._non_task_tool_count(all_steps)
                missing_init = effective_require_init and not has_init
                missing_tools = effective_min_tools > 0 and non_task_tools < effective_min_tools
                max_tools_reached = effective_max_tools > 0 and non_task_tools >= effective_max_tools
                self.logger.info(
                    "%s: steps=%s non_task_tools=%s has_init=%s missing_tools=%s max_tools_reached=%s ts=%s.",
                    run_label,
                    len(all_steps),
                    non_task_tools,
                    has_init,
                    missing_tools,
                    max_tools_reached,
                    _log_timestamp(),
                )
                if not missing_init and not missing_tools:
                    break
                if max_tools_reached:
                    break
                if (
                    missing_tools
                    and not missing_init
                    and missing_tools_reminders_sent >= max_missing_tools_reminders
                ):
                    break

                reminders = []
                if missing_init:
                    reminders.append(
                        "Before continuing, call task_list with action=init and a concise plan."
                    )
                if missing_tools:
                    remaining = max(0, effective_min_tools - non_task_tools)
                    reminders.append(
                        f"Use at least {remaining} more non-task tool calls to gather context before finalizing. "
                        "Use these extra tool calls to get more context to be able to answer the question more "
                        "accurately and confidently, rather than rushing to a final answer."
                    )
                    missing_tools_reminders_sent += 1
                current_prompt = (
                    "Continue the same run from your current context. "
                    "Do not provide your final answer yet.\n"
                    + " ".join(reminders)
                    + f"\n\nOriginal request:\n{prompt_text}"
                )

            return result, all_steps, prompt_total, completion_total, cached_total

        (
            result,
            run1_all_steps,
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens,
        ) = _invoke_with_min_tools(text, "Run 1")
        output = result.get("output", "")
        run1_output = output
        rounds_used = len(run1_all_steps) + (1 if run1_output else 0)
        tools_used = self._non_task_tool_count(run1_all_steps)
        self.logger.info(
            "Run 1 complete: output_chars=%s steps=%s non_task_tools=%s ts=%s.",
            len(run1_output or ""),
            len(run1_all_steps),
            tools_used,
            _log_timestamp(),
        )

        nested_counts_run1 = TOOL_USAGE_STORE.snapshot(task_session_id)

        run2_all_steps: list = []
        run2_output = ""
        if enable_self_critique:
            self.logger.info("Run 2 (self-critique) starting. ts=%s", _log_timestamp())
            critique_prompt = self._require_self_critique_prompt()
            critique_input = (
                f"{text}\n\nPrevious answer:\n{output}\n\n{critique_prompt}"
            )
            (
                critique_result,
                run2_all_steps,
                run2_prompt_tokens,
                run2_completion_tokens,
                run2_cached_prompt_tokens,
            ) = _invoke_with_min_tools(
                critique_input,
                "Run 2 (self-critique)",
                min_tools_target=0,
                require_task_list_init=False,
            )
            prompt_tokens += run2_prompt_tokens
            completion_tokens += run2_completion_tokens
            cached_prompt_tokens += run2_cached_prompt_tokens

            critique_output = critique_result.get("output", "")
            run2_output = critique_output
            output = critique_output or output
            rounds_used += len(run2_all_steps) + (1 if run2_output else 0)
            tools_used = self._non_task_tool_count(run1_all_steps + run2_all_steps)
            self.logger.info(
                "Run 2 complete: output_chars=%s steps=%s non_task_tools=%s ts=%s.",
                len(run2_output or ""),
                len(run2_all_steps),
                tools_used,
                _log_timestamp(),
            )

        nested_counts_total = TOOL_USAGE_STORE.snapshot(task_session_id)
        nested_counts_run2 = Counter(nested_counts_total)
        nested_counts_run2.subtract(nested_counts_run1)
        nested_counts_run2 = Counter({k: v for k, v in nested_counts_run2.items() if v > 0})

        run1_tool_counts = self._step_tool_counts(run1_all_steps)
        run2_tool_counts = self._step_tool_counts(run2_all_steps)
        run1_tool_counts.update(nested_counts_run1)
        run2_tool_counts.update(nested_counts_run2)

        tool_counts = Counter(run1_tool_counts)
        tool_counts.update(run2_tool_counts)
        nested_usage_by_model = TOOL_USAGE_STORE.tokens_snapshot(task_session_id)

        run1_tools_used = (
            self._non_task_tool_count(run1_all_steps)
            + self._non_task_tool_count_from_counter(nested_counts_run1)
        )
        run2_tools_used = (
            self._non_task_tool_count(run2_all_steps)
            + self._non_task_tool_count_from_counter(nested_counts_run2)
        )

        model_name = self.config.model.primary
        main_cost = estimate_cost(
            self._pricing,
            model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        nested_cost, _missing_nested_models = estimate_costs_by_model(
            self._pricing,
            nested_usage_by_model,
        )
        if main_cost is None and nested_cost == 0:
            total_cost = None
        else:
            total_cost = (main_cost or 0.0) + nested_cost
        if total_cost is None:
            cost_text = "unknown"
        else:
            cost_text = f"${total_cost:.4f}"

        run1_steps = len(run1_all_steps)
        run2_steps = len(run2_all_steps)
        max_turns = int(self.config.session.max_turns or 0)
        tool_counts_text = self._format_tool_counts(tool_counts)
        suffix = (
            f"\n\n🔁 {run1_steps}/{run2_steps}/{max_turns} | 🧰 {run1_tools_used}/{run2_tools_used} | 💲 {cost_text}\n"
            f"{tool_counts_text}"
        )

        if on_task_list_update is not None:
            STORE.unregister_listener(task_session_id, _listener)
        TOOL_USAGE_STORE.clear(task_session_id)

        if self.config.session.long_term_memory_enabled:
            asyncio.run(self._finalize_long_term_memory(session_id))

        self.logger.info(
            "Run finished: session=%s rounds=%s tools_used=%s cost=%s ts=%s.",
            session_id,
            rounds_used,
            tools_used,
            cost_text,
            _log_timestamp(),
        )
        return RunResult(
            output=output,
            steps=result.get("intermediate_steps", []),
            all_steps=run1_all_steps + run2_all_steps,
            tool_counts=tool_counts,
            nested_tool_counts=nested_counts_total,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            rounds_used=rounds_used,
            tools_used=tools_used,
            task_session_id=task_session_id,
            nested_usage_by_model=nested_usage_by_model,
            run1_output=run1_output,
            run2_output=run2_output,
            run1_steps=run1_steps,
            run2_steps=run2_steps,
            max_turns=max_turns,
            run1_tools_used=run1_tools_used,
            run2_tools_used=run2_tools_used,
            total_cost=total_cost,
            tool_counts_text=tool_counts_text,
            suffix=suffix,
        )
