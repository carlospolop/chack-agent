from __future__ import annotations

SUPPORTED_API_KEY_TYPES = {"openai", "anthropic", "openrouter", "gateway"}
SUPPORTED_API_KEY_TYPE_ORDER = ("openai", "anthropic", "openrouter", "gateway")

BACKENDS_BY_API_KEY_TYPE = {
    "openai": {"codex", "openai", "langgraph"},
    "anthropic": {"claude", "langgraph"},
    "openrouter": {"openai", "codex", "claude", "openrouter", "langgraph", "gemini"},
    "gateway": {"openai", "codex", "claude", "gateway", "langgraph", "gemini", "copilot"},
}


def get_supported_backends() -> set[str]:
    backends: set[str] = set()
    for values in BACKENDS_BY_API_KEY_TYPE.values():
        backends.update(values)
    return backends
