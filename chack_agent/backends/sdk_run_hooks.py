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


TERMINAL_FAILURE_STATUSES = frozenset({"failed", "incomplete"})


def _output_text(response: Any) -> str:
    """The text the model produced in a response: messages, reasoning, tool arguments."""
    parts: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"text", "arguments"} and isinstance(item, str):
                    parts.append(item)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for item in getattr(response, "output", None) or []:
        dump = getattr(item, "model_dump", None)
        collect(dump() if callable(dump) else item)
    return "\n".join(parts)


def repetition_measures(text: str) -> tuple[float, float]:
    """(compressed/raw size, share of non-blank lines that repeat the most common line).

    A degenerate loop compresses far below ordinary prose (roughly 0.3) and
    is dominated by one repeated line.
    """
    import zlib
    from collections import Counter

    raw = text.encode("utf-8", "replace")
    ratio = len(zlib.compress(raw, 6)) / len(raw) if raw else 1.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    top_share = Counter(lines).most_common(1)[0][1] / len(lines) if lines else 0.0
    return round(ratio, 3), round(top_share, 3)


def observe_terminal_failure(model_name: str, response: Any, logger: Any) -> None:
    """Log and charge a model response that ended failed or incomplete.

    The SDK raises on such a response, so no run hook sees it: without this
    its tokens go unreported and its cost unknown.
    """
    tokens = report_response(model_name, response)
    usage = getattr(response, "usage", None)
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
    text = _output_text(response)
    ratio, top_share = repetition_measures(text)
    logger.warning(
        "Model %s response ended %s (%s): usage input=%s output=%s reasoning=%s cached=%s; "
        "partial output %s chars, compressed/raw=%s, top repeated line share=%s",
        model_name,
        getattr(response, "status", None),
        getattr(getattr(response, "incomplete_details", None), "reason", None)
        or getattr(response, "error", None),
        tokens[0] if tokens else None,
        tokens[1] if tokens else None,
        reasoning,
        tokens[2] if tokens else None,
        len(text),
        ratio,
        top_share,
    )
