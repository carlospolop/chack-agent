from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import requests

BEST_QUALITY = "google/gemini-3-pro-preview"
CHEAP_BUT_QUALITY = "google/gemini-3-flash-preview"
BEST_CHEAPEST = "xiaomi/mimo-v2-flash"
OPENAI_BEST_QUALITY = "gpt-5.2-codex"
OPENAI_CHEAP_BUT_QUALITY = "gpt-5-mini"
OPENAI_BEST_CHEAPEST = "gpt-5-nano"
_OPENAI_ALIAS_EQUIVALENTS: Dict[str, str] = {
    "BEST_QUALITY": "OPENAI_BEST_QUALITY",
    "CHEAP_BUT_QUALITY": "OPENAI_CHEAP_BUT_QUALITY",
    "BEST_CHEAPEST": "OPENAI_BEST_CHEAPEST",
}


MODEL_ALIASES: Dict[str, str] = {
    "BEST_QUALITY": BEST_QUALITY,
    "CHEAP_BUT_QUALITY": CHEAP_BUT_QUALITY,
    "BEST_CHEAPEST": BEST_CHEAPEST,
    "OPENAI_BEST_QUALITY": OPENAI_BEST_QUALITY,
    "OPENAI_CHEAP_BUT_QUALITY": OPENAI_CHEAP_BUT_QUALITY,
    "OPENAI_BEST_CHEAPEST": OPENAI_BEST_CHEAPEST,
}

_LOGGER = logging.getLogger("chack.model_aliases")
_REMOTE_CACHE_SECONDS = int(os.environ.get("CHACK_MODEL_ALIASES_CACHE_SECONDS", "300") or "300")
_REMOTE_URL = os.environ.get("CHACK_MODEL_ALIASES_LAMBDA_URL", "").strip()
_DEFAULT_REMOTE_URL = "https://6lj6nwv3krblocoano5k33zzna0uqebx.lambda-url.us-east-1.on.aws/"
if not _REMOTE_URL:
    _REMOTE_URL = _DEFAULT_REMOTE_URL
_REMOTE_CACHE: Optional[Dict[str, str]] = None
_REMOTE_CACHE_LOADED_AT: float = 0.0


def _load_remote_aliases() -> Optional[Dict[str, str]]:
    if not _REMOTE_URL:
        return None
    try:
        response = requests.get(_REMOTE_URL, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        _LOGGER.warning("Failed to fetch model aliases from %s: %s", _REMOTE_URL, exc)
        return None
    except ValueError as exc:
        _LOGGER.warning("Invalid JSON from %s: %s", _REMOTE_URL, exc)
        return None

    if isinstance(payload, dict) and "model_aliases" in payload:
        payload = payload.get("model_aliases")
    if not isinstance(payload, dict):
        _LOGGER.warning("Unexpected aliases payload from %s: %r", _REMOTE_URL, payload)
        return None

    aliases: Dict[str, str] = {}
    for key, value in payload.items():
        if not key or not isinstance(key, str):
            continue
        if not value or not isinstance(value, str):
            continue
        aliases[key.strip()] = value.strip()

    return aliases or None


def _get_model_aliases() -> Dict[str, str]:
    global _REMOTE_CACHE, _REMOTE_CACHE_LOADED_AT
    if not _REMOTE_URL:
        return MODEL_ALIASES
    now = time.time()
    if _REMOTE_CACHE is None or (now - _REMOTE_CACHE_LOADED_AT) > _REMOTE_CACHE_SECONDS:
        remote = _load_remote_aliases()
        if remote:
            _REMOTE_CACHE = remote
            _REMOTE_CACHE_LOADED_AT = now
    merged = dict(MODEL_ALIASES)
    if _REMOTE_CACHE:
        merged.update(_REMOTE_CACHE)
    return merged


def _adapt_alias_for_runtime(name: str, *, provider: str = "") -> str:
    key = str(name or "").strip()
    if not key:
        return name
    model_provider = str(provider or "").strip().lower()
    if model_provider in {"openai", "codex"}:
        return _OPENAI_ALIAS_EQUIVALENTS.get(key, key)
    return key


def resolve_model_alias(
    name: str,
    *,
    provider: str = "",
) -> str:
    if not name:
        return name
    key = _adapt_alias_for_runtime(name, provider=provider)
    if not key:
        return name
    aliases = _get_model_aliases()
    if key in aliases:
        return aliases[key]
    if key.startswith("OPENAI_"):
        fallback = key[len("OPENAI_") :]
        if fallback in aliases:
            return aliases[fallback]
    else:
        fallback = f"OPENAI_{key}"
        if fallback in aliases:
            return aliases[fallback]
    return name


if _REMOTE_URL:
    _get_model_aliases()
