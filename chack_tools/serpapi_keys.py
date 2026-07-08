from __future__ import annotations

import os
import random
import threading
import time
from typing import Any, Optional

try:
    import requests
except ImportError:  # pragma: no cover - requests is a runtime dep of the search tools
    requests = None


def parse_serpapi_keys(raw: Any) -> list[str]:
    if raw is None:
        return []
    parts: list[str] = []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, str):
                parts.extend([p.strip() for p in item.split(",")])
            elif item is not None:
                text = str(item).strip()
                if text:
                    parts.extend([p.strip() for p in text.split(",")])
    else:
        text = str(raw).strip()
        if text:
            parts = [p.strip() for p in text.split(",")]

    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def has_serpapi_keys(raw: Any) -> bool:
    return bool(parse_serpapi_keys(raw))


def shuffled_serpapi_keys(raw: Any) -> list[str]:
    keys = parse_serpapi_keys(raw)
    if len(keys) <= 1:
        return keys
    random.shuffle(keys)
    return keys


def is_serpapi_rate_limited(status_code: int, error_text: str = "") -> bool:
    text = (error_text or "").lower()
    if status_code == 429:
        return True
    if "rate limit" in text:
        return True
    if "too many requests" in text:
        return True
    if "searches per month" in text:
        return True
    if "insufficient searches" in text:
        return True
    if "quota" in text and "exceed" in text:
        return True
    return False


def is_serpapi_quota_exhausted(status_code: int, error_text: str = "") -> bool:
    """True only for signals that the key's *plan quota* is used up (not a
    transient hourly rate limit). A plain 429 is deliberately NOT treated as
    exhaustion so a good key is never disabled for a whole day by a spike."""
    text = (error_text or "").lower()
    if "searches per month" in text:
        return True
    if "insufficient searches" in text:
        return True
    if "run out of searches" in text or "ran out of searches" in text:
        return True
    if "no searches left" in text or "out of searches" in text:
        return True
    if "quota" in text and ("exceed" in text or "exhaust" in text):
        return True
    return False


# ── Exhaustion cache ────────────────────────────────────────────────────────────
# Before spending a search we check each key's remaining quota against the SerpApi
# account endpoint and cache the verdict so we neither waste searches on dead keys
# nor re-check on every call. Usable keys are re-checked every 15 min; exhausted
# keys are parked for a day. Everything is in-memory and per-process (matching the
# rest of the toolset); several threads share one cache via a lock.
_SERPAPI_ACCOUNT_URL = "https://serpapi.com/account.json"
_DEFAULT_EXHAUSTED_TTL_SECONDS = 24 * 60 * 60   # 1 day
_DEFAULT_OK_TTL_SECONDS = 15 * 60               # 15 minutes
_DEFAULT_ACCOUNT_TIMEOUT_SECONDS = 10

_exhaustion_lock = threading.Lock()
# api_key -> (exhausted: bool, expiry_monotonic: float)
_exhaustion_cache: dict[str, tuple[bool, float]] = {}


def _now() -> float:
    return time.monotonic()


def _int_env(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _exhausted_ttl_seconds() -> int:
    return _int_env("SERPAPI_EXHAUSTED_CACHE_SECONDS", _DEFAULT_EXHAUSTED_TTL_SECONDS)


def _ok_ttl_seconds() -> int:
    return _int_env("SERPAPI_OK_CACHE_SECONDS", _DEFAULT_OK_TTL_SECONDS)


def _exhaustion_check_enabled() -> bool:
    raw = str(os.environ.get("SERPAPI_EXHAUSTION_CHECK_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def reset_serpapi_exhaustion_cache() -> None:
    with _exhaustion_lock:
        _exhaustion_cache.clear()


def _cache_get(api_key: str) -> Optional[bool]:
    with _exhaustion_lock:
        entry = _exhaustion_cache.get(api_key)
        if entry is None:
            return None
        exhausted, expiry = entry
        if _now() >= expiry:
            _exhaustion_cache.pop(api_key, None)
            return None
        return exhausted


def _cache_set(api_key: str, exhausted: bool) -> None:
    ttl = _exhausted_ttl_seconds() if exhausted else _ok_ttl_seconds()
    with _exhaustion_lock:
        _exhaustion_cache[api_key] = (exhausted, _now() + ttl)


def mark_serpapi_key_exhausted(api_key: str) -> None:
    """Record that a key is exhausted (parked for the exhausted TTL)."""
    if api_key:
        _cache_set(api_key, True)


def note_serpapi_response_error(api_key: str, status_code: int, error_text: str = "") -> None:
    """Park a key as exhausted when a live response proves its quota is used up."""
    if api_key and is_serpapi_quota_exhausted(status_code, error_text):
        mark_serpapi_key_exhausted(api_key)


def _query_account_exhausted(api_key: str) -> Optional[bool]:
    """Ask the SerpApi account endpoint whether a key has searches left.

    Returns True (exhausted), False (has searches), or None when the answer is
    inconclusive (network/HTTP/parse issue) so the caller can fail open.
    """
    if requests is None or not api_key:
        return None
    timeout = _int_env("SERPAPI_ACCOUNT_TIMEOUT_SECONDS", _DEFAULT_ACCOUNT_TIMEOUT_SECONDS)
    try:
        response = requests.get(_SERPAPI_ACCOUNT_URL, params={"api_key": api_key}, timeout=timeout)
    except Exception:
        return None
    if response.status_code in (401, 403):
        return True  # invalid / revoked key -> never usable
    if response.status_code >= 400:
        return None  # transient; unknown
    try:
        data = response.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    remaining = data.get("total_searches_left")
    if remaining is None:
        remaining = data.get("plan_searches_left")
    if remaining is None:
        return None
    try:
        return int(remaining) <= 0
    except (TypeError, ValueError):
        return None


def is_serpapi_key_exhausted(api_key: str) -> bool:
    """True when a key should be skipped. Uses the cache, querying the account
    endpoint on a miss. Fails open (usable) when the check is inconclusive."""
    if not api_key:
        return True
    cached = _cache_get(api_key)
    if cached is not None:
        return cached
    exhausted = _query_account_exhausted(api_key)
    if exhausted is None:
        # Unknown: assume usable but cache briefly so we don't re-check every call.
        _cache_set(api_key, False)
        return False
    _cache_set(api_key, exhausted)
    return exhausted


def usable_serpapi_keys(raw: Any) -> list[str]:
    """Shuffled keys with the ones known to be exhausted removed.

    Returns an empty list when every configured key is exhausted (callers should
    surface that instead of wasting a search). Set SERPAPI_EXHAUSTION_CHECK_ENABLED=0
    to bypass the pre-check entirely and fall back to plain shuffled keys.
    """
    keys = shuffled_serpapi_keys(raw)
    if not keys or not _exhaustion_check_enabled():
        return keys
    return [key for key in keys if not is_serpapi_key_exhausted(key)]
