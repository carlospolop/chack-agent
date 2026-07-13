from __future__ import annotations

from collections.abc import Collection
from typing import Any


DEFAULT_THINKING_EFFORT = "high"
THINKING_EFFORT_LEVELS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

_ALIASES = {
    "off": "none",
    "disabled": "none",
    "extra_high": "xhigh",
    "extra-high": "xhigh",
    "extra high": "xhigh",
    "very_high": "xhigh",
    "very-high": "xhigh",
    "very high": "xhigh",
}


def normalize_thinking_effort(value: Any) -> str:
    """Return the canonical cross-backend effort name or raise ValueError."""
    raw = str(value or "").strip().lower()
    normalized = _ALIASES.get(raw, raw) or DEFAULT_THINKING_EFFORT
    if normalized not in THINKING_EFFORT_LEVELS:
        allowed = ", ".join(sorted(THINKING_EFFORT_LEVELS))
        raise ValueError(
            f"Unsupported thinking_effort value {value!r}. Expected one of: {allowed}"
        )
    return normalized


def openai_thinking_effort(value: Any) -> str:
    """Map the common vocabulary to the OpenAI Responses vocabulary."""
    effort = normalize_thinking_effort(value)
    return "xhigh" if effort == "max" else effort


def codex_thinking_effort(value: Any) -> str:
    """Map to Codex's documented minimal/low/medium/high/xhigh vocabulary."""
    effort = normalize_thinking_effort(value)
    if effort == "none":
        return "minimal"
    if effort == "max":
        return "xhigh"
    return effort


def claude_thinking_effort(
    value: Any,
    supported_levels: Collection[str] | None = None,
) -> str:
    """Map to the levels supported by the installed Claude Code CLI.

    Claude Code releases do not all expose the same top tier: older releases
    accept ``max`` while newer releases may also accept ``xhigh``.  The caller
    can pass the levels discovered from ``claude --help`` so both generations
    work without rejecting the command line.
    """
    effort = normalize_thinking_effort(value)
    supported = set(supported_levels or {"low", "medium", "high", "max"})
    if effort in {"none", "minimal"}:
        return "low"
    if effort == "xhigh":
        if "xhigh" in supported:
            return "xhigh"
        return "max" if "max" in supported else "high"
    if effort == "max" and "max" not in supported:
        return "xhigh" if "xhigh" in supported else "high"
    return effort


def copilot_thinking_effort(value: Any) -> str:
    """Map to Copilot CLI's documented low/medium/high/xhigh/max vocabulary."""
    effort = normalize_thinking_effort(value)
    return "low" if effort in {"none", "minimal"} else effort


def _gemini_model_name(model_name: Any) -> str:
    return str(model_name or "").strip().lower().rsplit("/", 1)[-1]


def gemini_thinking_config(value: Any, model_name: Any = "") -> dict[str, Any]:
    """Build a model-family-safe Gemini CLI ``thinkingConfig``.

    Gemini 3 uses ``thinkingLevel`` and Gemini 2.5 uses ``thinkingBudget``.
    Sending both is not portable, so select exactly one based on the configured
    model. Unknown/future models use the level-based API used by current Gemini.
    """
    effort = normalize_thinking_effort(value)
    model = _gemini_model_name(model_name)

    if not model.startswith("gemini-2.5"):
        # Gemini 3 Pro only supports LOW/HIGH, while Flash and newer Pro
        # variants also expose MINIMAL and/or MEDIUM. Clamp unsupported common
        # levels instead of letting the provider reject the request.
        if "gemini-3-pro" in model and "gemini-3.1-pro" not in model:
            level = "LOW" if effort in {"none", "minimal", "low"} else "HIGH"
        elif "flash-lite-image" in model:
            level = "MINIMAL" if effort in {"none", "minimal", "low"} else "HIGH"
        else:
            level = {
                "none": "MINIMAL",
                "minimal": "MINIMAL",
                "low": "LOW",
                "medium": "MEDIUM",
                "high": "HIGH",
                "xhigh": "HIGH",
                "max": "HIGH",
            }[effort]
        return {"includeThoughts": True, "thinkingLevel": level}

    is_pro = "-pro" in model
    budget = {
        # Gemini 2.5 Pro cannot disable thinking and has a minimum of 128.
        "none": 128 if is_pro else 0,
        "minimal": 128 if is_pro else 512,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
        # Flash/Flash-Lite cap at 24576; Pro permits 32768.
        "xhigh": 32768 if is_pro else 24576,
        "max": 32768 if is_pro else 24576,
    }[effort]
    return {"includeThoughts": True, "thinkingBudget": budget}
