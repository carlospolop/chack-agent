from __future__ import annotations

from typing import Any

from ..config import ChackConfig, resolve_backend_type
from .codex_backend import build_executor as build_codex_executor
from .openai_compaction_backend import build_executor as build_openai_compaction_executor
from .openrouter_openai_backend import build_executor as build_openrouter_executor


def build_executor(config: ChackConfig, **kwargs: Any):
    backend_type = resolve_backend_type(config)
    if backend_type == "codex":
        return build_codex_executor(config, **kwargs)
    if backend_type == "openrouter":
        return build_openrouter_executor(config, **kwargs)
    if backend_type == "openai_compaction":
        return build_openai_compaction_executor(config, **kwargs)
    raise ValueError(f"Unsupported backend resolved from model.provider: {backend_type}")

__all__ = ["build_executor"]
