"""Process-wide fallback rotation for provider credentials supplied in env pools."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


_LOCK = threading.RLock()
_POOL_SIGNATURES: dict[str, tuple[tuple[str, ...], ...]] = {}
_ACTIVE_INDEXES: dict[str, int] = {"codex": 0, "claude": 0}
_FAILED_AT: dict[str, dict[str, float]] = {"codex": {}, "claude": {}}
_DEFAULT_RETRY_SECONDS = 15 * 60.0


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _load_entries(provider: str) -> list[dict[str, str]]:
    env_name = (
        "CODEX_TOKEN_POOL_JSON" if provider == "codex" else "CLAUDE_TOKEN_POOL_JSON"
    )
    try:
        raw_entries = json.loads(_clean(os.environ.get(env_name)) or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(raw_entries, list):
        return []

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        if provider == "codex":
            token = _clean(raw_entry.get("access_token"))
            if not token or token in seen:
                continue
            entry = {
                "access_token": token,
                "id_token": _clean(raw_entry.get("id_token")),
                "refresh_token": _clean(raw_entry.get("refresh_token")),
                "account_id": _clean(raw_entry.get("account_id")),
                "last_refresh": _clean(raw_entry.get("last_refresh")),
            }
        else:
            token = _clean(raw_entry.get("token"))
            if not token or token in seen:
                continue
            entry = {"access_token": token}
        entries.append(entry)
        seen.add(token)
    return entries


def _entry_signature(entry: dict[str, str]) -> tuple[str, ...]:
    return tuple(entry.get(name, "") for name in sorted(entry))


def _current_env_token(provider: str) -> str:
    if provider == "codex":
        return _clean(os.environ.get("CODEX_ACCESS_TOKEN"))
    return _clean(
        os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        or os.environ.get("CLAUDE_ACCESS_TOKEN")
    )


def _synchronize(provider: str, entries: list[dict[str, str]]) -> int:
    signature = tuple(_entry_signature(entry) for entry in entries)
    if _POOL_SIGNATURES.get(provider) != signature:
        _POOL_SIGNATURES[provider] = signature
        _FAILED_AT[provider] = {}
        current = _current_env_token(provider)
        _ACTIVE_INDEXES[provider] = next(
            (
                index
                for index, entry in enumerate(entries)
                if entry["access_token"] == current
            ),
            0,
        )
    return min(_ACTIVE_INDEXES.get(provider, 0), max(0, len(entries) - 1))


def _activate(provider: str, entry: dict[str, str]) -> dict[str, str]:
    token = entry["access_token"]
    if provider == "codex":
        values = {
            "CODEX_ACCESS_TOKEN": token,
            "CHACK_CODEX_ACCESS_TOKEN": token,
            "CODEX_ID_TOKEN": entry.get("id_token", ""),
            "CODEX_REFRESH_TOKEN": entry.get("refresh_token", ""),
            "CODEX_ACCOUNT_ID": entry.get("account_id", ""),
            "CODEX_LAST_REFRESH": entry.get("last_refresh", ""),
        }
    else:
        values = {
            "CLAUDE_CODE_OAUTH_TOKEN": token,
            "CLAUDE_ACCESS_TOKEN": token,
        }
    for name, value in values.items():
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)
    return {"access_token": token}


def active_environment_credentials(provider: str) -> dict[str, str]:
    """Return the pool's current credential when no host hook is registered."""
    normalized = _clean(provider).lower()
    if normalized not in {"codex", "claude"}:
        return {}
    with _LOCK:
        entries = _load_entries(normalized)
        if not entries:
            return {}
        index = _synchronize(normalized, entries)
        retry_seconds = _retry_seconds()
        now = time.monotonic()
        failed_at = _FAILED_AT[normalized]
        current_failed_at = failed_at.get(entries[index]["access_token"])
        if current_failed_at is not None and now - current_failed_at < retry_seconds:
            replacement = next(
                (
                    candidate_index
                    for candidate_index, entry in enumerate(entries)
                    if now - failed_at.get(entry["access_token"], float("-inf"))
                    >= retry_seconds
                ),
                None,
            )
            if replacement is not None:
                index = replacement
                _ACTIVE_INDEXES[normalized] = index
        return _activate(normalized, entries[index])


def rotate_environment_credentials(provider: str, failed_token: str) -> dict[str, str]:
    """Advance after an auth/quota failure, without double-advancing concurrent calls."""
    normalized = _clean(provider).lower()
    if normalized not in {"codex", "claude"}:
        return {}
    with _LOCK:
        entries = _load_entries(normalized)
        if len(entries) < 2:
            return {}
        index = _synchronize(normalized, entries)
        if entries[index]["access_token"] != _clean(failed_token):
            return _activate(normalized, entries[index])
        now = time.monotonic()
        _FAILED_AT[normalized][_clean(failed_token)] = now
        retry_seconds = _retry_seconds()
        next_index = None
        for offset in range(1, len(entries)):
            candidate_index = (index + offset) % len(entries)
            candidate_token = entries[candidate_index]["access_token"]
            if (
                now
                - _FAILED_AT[normalized].get(candidate_token, float("-inf"))
                >= retry_seconds
            ):
                next_index = candidate_index
                break
        if next_index is None:
            return {}
        _ACTIVE_INDEXES[normalized] = next_index
        return _activate(normalized, entries[next_index])


def _retry_seconds() -> float:
    try:
        return max(
            0.0,
            float(
                os.environ.get("CHACK_PROVIDER_POOL_RETRY_SECONDS", "")
                or _DEFAULT_RETRY_SECONDS
            ),
        )
    except (TypeError, ValueError):
        return _DEFAULT_RETRY_SECONDS


def reset_environment_credential_pool_for_tests() -> None:
    with _LOCK:
        _POOL_SIGNATURES.clear()
        _ACTIVE_INDEXES.update({"codex": 0, "claude": 0})
        _FAILED_AT.update({"codex": {}, "claude": {}})


__all__ = [
    "active_environment_credentials",
    "reset_environment_credential_pool_for_tests",
    "rotate_environment_credentials",
]
