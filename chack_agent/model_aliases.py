from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import requests

BEST_QUALITY = "google/gemini-3-pro-preview"
CHEAP_BUT_QUALITY = "google/gemini-3-flash-preview"
BEST_CHEAPEST = "xiaomi/mimo-v2-flash"


MODEL_ALIASES: Dict[str, str] = {
    "BEST_QUALITY": BEST_QUALITY,
    "CHEAP_BUT_QUALITY": CHEAP_BUT_QUALITY,
    "BEST_CHEAPEST": BEST_CHEAPEST,
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
    return _REMOTE_CACHE or MODEL_ALIASES


def resolve_model_alias(name: str) -> str:
    if not name:
        return name
    key = str(name).strip()
    if not key:
        return name
    return _get_model_aliases().get(key, name)


if _REMOTE_URL:
    _get_model_aliases()
