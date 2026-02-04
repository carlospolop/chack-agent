import textwrap
from typing import List


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[the output was truncated, exceeded limit of {} chars. You probably want to rerun the command with a grep/jq or similar pipe to extract the data you are looking for]".format(limit)


def redact_sensitive(text: str) -> str:
    if not text:
        return text
    redactions = [
        "api_key",
        "token",
        "secret",
        "password",
    ]
    lowered = text.lower()
    for key in redactions:
        if key in lowered:
            return "[redacted]"
    return textwrap.shorten(text, width=300, placeholder="...")


def format_tool_steps(
    steps,
    max_chars: int = 320,
    max_turns: int = 50,
    notify_every: int = 10,
) -> str:
    if not steps:
        return ""
    lines: List[str] = []
    for idx, (action, observation) in enumerate(steps, start=1):
        tool_name = getattr(action, "tool", "tool")
        tool_input = getattr(action, "tool_input", "")
        tool_input_text = redact_sensitive(str(tool_input))
        tool_input_text = _truncate(tool_input_text, max_chars)
        lines.append(f"- {tool_name}: {tool_input_text}")
        if notify_every and max_turns and idx % notify_every == 0:
            remaining = max(max_turns - idx, 0)
            lines.append(f"- turns-remaining: {remaining} (used {idx}/{max_turns})")
    return "\n".join(lines)
