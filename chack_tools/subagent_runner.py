import asyncio
from typing import Optional
from collections import Counter
import threading
import os

try:
    from agents import Agent, Runner
except ImportError:  # pragma: no cover
    Agent = None
    Runner = None

from .tool_usage_state import STORE


class SubAgentRunner:
    def __init__(self, model_name: str = "", env_var_name: str = "", max_turns: int = 30):
        self.model_name = model_name
        self.env_var_name = env_var_name
        self.max_turns = max_turns

    def _resolved_model(self) -> Optional[str]:
        configured = (self.model_name or "").strip()
        if configured:
            return configured
        if not self.env_var_name:
            return None

        return (os.environ.get(self.env_var_name, "") or "").strip() or None

    def run(
        self,
        *,
        prompt: str,
        agent_name: str,
        system_prompt: str,
        tools: list,
    ) -> str:
        if not prompt.strip():
            return "ERROR: prompt cannot be empty"
        if Agent is None or Runner is None:
            return "ERROR: OpenAI Agents SDK is not available."

        agent_kwargs = {
            "name": agent_name,
            "instructions": system_prompt,
            "tools": tools,
        }
        model_name = self._resolved_model()
        if model_name:
            agent_kwargs["model"] = model_name
        agent = Agent(**agent_kwargs)

        run_input = prompt.strip()
        result = None
        nested_counts: Counter[str] = Counter()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_prompt_tokens = 0
        for attempt in range(2):
            attempt_input = run_input
            if attempt:
                attempt_input = (
                    f"{run_input}\n\nMANDATORY: Use your available tools to gather evidence "
                    "before answering. Do not answer from memory only."
                )
            result = self._runner_run_sync(
                agent,
                [{"role": "user", "content": attempt_input}],
                self.max_turns,
            )
            nested_counts = self._collect_nested_tool_usage(
                getattr(result, "new_items", []) or [],
                getattr(result, "raw_responses", []) or [],
            )
            attempt_prompt, attempt_completion, attempt_cached = self._usage_from_raw_result(
                getattr(result, "raw_responses", []) or [],
            )
            total_prompt_tokens += attempt_prompt
            total_completion_tokens += attempt_completion
            total_cached_prompt_tokens += attempt_cached
            if sum(nested_counts.values()) > 0:
                break

        for tool_name, count in nested_counts.items():
            STORE.add(tool_name, count=count)
        if model_name:
            STORE.add_tokens(
                model_name,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_prompt_tokens=total_cached_prompt_tokens,
            )

        if result is None:
            return "ERROR: sub-agent run failed."
        output = (result.final_output or "").strip()
        if not output:
            return "ERROR: sub-agent returned an empty response."
        return output

    @staticmethod
    def _runner_run_sync(agent, input_items, max_turns):
        # Nested tools can execute while the parent agent loop is active.
        # Runner.run_sync cannot run on an active event loop, so in that case
        # we execute it in a dedicated thread.
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False

        if not loop_running:
            return Runner.run_sync(agent, input_items, max_turns=max_turns)

        box = {"result": None, "error": None}

        def _target():
            try:
                box["result"] = Runner.run_sync(agent, input_items, max_turns=max_turns)
            except Exception as exc:
                box["error"] = exc

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join()
        if box["error"] is not None:
            raise box["error"]
        return box["result"]

    @staticmethod
    def _extract_tool_name(raw) -> str:
        if raw is None:
            return ""
        if hasattr(raw, "name"):
            return str(getattr(raw, "name", "") or "")
        if hasattr(raw, "function"):
            func = getattr(raw, "function", None)
            if func and hasattr(func, "name"):
                return str(getattr(func, "name", "") or "")
        if isinstance(raw, dict):
            name = raw.get("name")
            if name:
                return str(name)
            func = raw.get("function", {})
            if isinstance(func, dict):
                return str(func.get("name", "") or "")
        return ""

    def _collect_from_items(self, items: list) -> Counter[str]:
        counts: Counter[str] = Counter()
        for item in items:
            # Be permissive here: different SDK versions can expose different
            # run item classes, but most include either `raw_item` or a dict-like payload.
            raw = getattr(item, "raw_item", item)
            name = self._extract_tool_name(raw)
            if name:
                counts[name] += 1
        return counts

    def _collect_from_raw_responses(self, raw_responses: list) -> Counter[str]:
        counts: Counter[str] = Counter()
        for response in raw_responses:
            output_items = getattr(response, "output", None)
            if output_items is None and isinstance(response, dict):
                output_items = response.get("output", [])
            if not output_items:
                continue
            for out in output_items:
                raw = out
                item_type = ""
                if isinstance(out, dict):
                    item_type = str(out.get("type", "") or "")
                else:
                    item_type = str(getattr(out, "type", "") or "")
                # Track function/hosted tool calls from raw model output.
                if "call" in item_type or "tool" in item_type:
                    name = self._extract_tool_name(raw)
                    if not name:
                        # Fallback for some payloads where tool name is under `tool_name`.
                        if isinstance(raw, dict):
                            name = str(raw.get("tool_name", "") or "")
                        else:
                            name = str(getattr(raw, "tool_name", "") or "")
                    if name:
                        counts[name] += 1
        return counts

    def _collect_nested_tool_usage(self, items: list, raw_responses: list) -> Counter[str]:
        counts = self._collect_from_items(items)
        counts.update(self._collect_from_raw_responses(raw_responses))
        return counts

    @staticmethod
    def _usage_from_raw_result(raw_responses: list) -> tuple[int, int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        cached_prompt_tokens = 0
        for resp in raw_responses:
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
