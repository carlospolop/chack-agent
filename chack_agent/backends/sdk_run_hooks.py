"""Run hooks for the OpenAI Agents SDK backends.

The SDK reports usage only in the run's result, so a long run spent its whole
budget before its caller learnt of it, and a cancelled run kept calling the
model. These hooks stop a cancelled run before its next model call and report
each call's usage as soon as it ends.
"""

from __future__ import annotations

from typing import Any, Optional

from agents import RunHooks

from chack_tools.cancellation import raise_if_cancelled

from ..live_cost_state import report_live_usage

REPORTED_RESPONSES_ATTR = "_chack_reported_responses"


def response_usage(response: Any) -> Optional[tuple[int, int, int, int]]:
    """(input, output, cached input, cache write) tokens of one model response, or None."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None
    if isinstance(usage, dict):
        details = usage.get("input_tokens_details") or {}
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            int(details.get("cached_tokens", 0) or 0),
            int(details.get("cache_write_tokens", 0) or 0),
        )
    details = getattr(usage, "input_tokens_details", None)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        int(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        int(getattr(details, "cache_write_tokens", 0) or 0) if details else 0,
    )


def report_response(model_name: str, response: Any) -> Optional[tuple[int, int, int, int]]:
    tokens = response_usage(response)
    if tokens is not None:
        report_live_usage(
            model_name,
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
            cached_prompt_tokens=tokens[2],
            cache_write_tokens=tokens[3],
        )
    return tokens


class LiveUsageHooks(RunHooks):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.reported: set[int] = set()

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        raise_if_cancelled()

    async def on_llm_end(self, context, agent, response) -> None:
        # Marked first: a live budget callback may raise to stop the run.
        self.reported.add(id(response))
        report_response(self.model_name, response)


def mark_reported(result: Any, hooks: LiveUsageHooks) -> None:
    try:
        setattr(result, REPORTED_RESPONSES_ATTR, hooks.reported)
    except Exception:
        pass


def unreported_responses(result: Any) -> list[Any]:
    """The result's model responses whose usage the hooks did not report."""
    reported = getattr(result, REPORTED_RESPONSES_ATTR, None) or set()
    return [response for response in (getattr(result, "raw_responses", None) or []) if id(response) not in reported]
