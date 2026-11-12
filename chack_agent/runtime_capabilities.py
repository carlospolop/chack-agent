from __future__ import annotations

SUPPORTED_API_KEY_TYPES = {"openai", "anthropic", "openrouter", "gateway", "kimi", "qwen"}
SUPPORTED_API_KEY_TYPE_ORDER = ("openai", "anthropic", "openrouter", "gateway", "kimi", "qwen")

BACKENDS_BY_API_KEY_TYPE = {
    "openai": {"codex", "openai", "langgraph", "opencode"},
    "anthropic": {"claude", "langgraph", "opencode"},
    "openrouter": {"openai", "codex", "claude", "openrouter", "langgraph", "gemini", "opencode"},
    "gateway": {"openai", "codex", "claude", "gateway", "langgraph", "gemini", "copilot", "opencode", "kimi", "qwen"},
    "kimi": {"kimi"},
    "qwen": {"qwen"},
}


def get_supported_backends() -> set[str]:
    backends: set[str] = set()
    for values in BACKENDS_BY_API_KEY_TYPE.values():
        backends.update(values)
    return backends
