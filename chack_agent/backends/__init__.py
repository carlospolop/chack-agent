from __future__ import annotations

import inspect
from typing import Any

from ..config import ChackConfig, resolve_backend_type
from .codex_backend import build_executor as build_codex_executor
from .langgraph_backend import build_executor as build_langgraph_executor
from .openai_compaction_backend import build_executor as build_openai_compaction_executor
from .openrouter_openai_backend import build_executor as build_openrouter_executor
from .gemini_cli_backend import build_executor as build_gemini_executor
from .claude_code_backend import build_executor as build_claude_executor
from ..openrouter_routing import get_openrouter_route


def _call_executor_with_supported_kwargs(builder: Any, config: ChackConfig, **kwargs: Any):
    """Call a backend builder while tolerating mixed backend signature versions."""
    try:
        signature = inspect.signature(builder)
    except Exception:
        return builder(config, **kwargs)

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return builder(config, **kwargs)

    allowed_names = set(signature.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_names}
    return builder(config, **filtered_kwargs)


def build_executor(config: ChackConfig, **kwargs: Any):
    backend_type = resolve_backend_type(config)
    if backend_type == "codex":
        if get_openrouter_route(config) is not None:
            return _call_executor_with_supported_kwargs(
                build_openai_compaction_executor, config, **kwargs
            )
        return _call_executor_with_supported_kwargs(build_codex_executor, config, **kwargs)
    if backend_type == "langgraph":
        return _call_executor_with_supported_kwargs(build_langgraph_executor, config, **kwargs)
    if backend_type == "openrouter":
        return _call_executor_with_supported_kwargs(build_openrouter_executor, config, **kwargs)
    if backend_type == "openai_compaction":
        return _call_executor_with_supported_kwargs(
            build_openai_compaction_executor, config, **kwargs
        )
    if backend_type == "gemini":
        return _call_executor_with_supported_kwargs(build_gemini_executor, config, **kwargs)
    if backend_type == "claude":
        return _call_executor_with_supported_kwargs(build_claude_executor, config, **kwargs)
    raise ValueError(f"Unsupported backend resolved from model.provider: {backend_type}")

__all__ = ["build_executor"]
