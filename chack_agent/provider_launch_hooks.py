"""Optional host callback for refreshing provider credentials before CLI launches."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any


_LOGGER = logging.getLogger("chack.provider_launch_hooks")
_HOOK_LOCK = threading.RLock()
_provider_pre_launch_hook: Callable[[str], Mapping[str, Any] | None] | None = None


def set_provider_pre_launch_hook(
    hook: Callable[[str], Mapping[str, Any] | None] | None,
) -> None:
    """Register one process-wide hook invoked with ``codex`` or ``claude``."""
    global _provider_pre_launch_hook
    with _HOOK_LOCK:
        _provider_pre_launch_hook = hook


def run_provider_pre_launch_hook(provider: str) -> dict[str, str]:
    """Run the hook without allowing monitoring failures to block a model call."""
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider not in {"codex", "claude"}:
        return {}
    with _HOOK_LOCK:
        hook = _provider_pre_launch_hook
    if hook is None:
        return {}
    try:
        values = hook(normalized_provider)
    except Exception as exc:
        _LOGGER.warning(
            "Provider pre-launch hook failed for %s: %s",
            normalized_provider,
            exc,
        )
        return {}
    if not isinstance(values, Mapping):
        return {}
    return {
        str(key): str(value)
        for key, value in values.items()
        if value is not None and str(value).strip()
    }


__all__ = ["run_provider_pre_launch_hook", "set_provider_pre_launch_hook"]
