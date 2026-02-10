from __future__ import annotations

from typing import Dict

BEST_QUALITY = "google/gemini-3-pro-preview"
CHEAP_BUT_QUALITY = "google/gemini-3-flash-preview"
BEST_CHEAPEST = "xiaomi/mimo-v2-flash"


MODEL_ALIASES: Dict[str, str] = {
    "BEST_QUALITY": BEST_QUALITY,
    "CHEAP_BUT_QUALITY": CHEAP_BUT_QUALITY,
    "BEST_CHEAPEST": BEST_CHEAPEST,
}


def resolve_model_alias(name: str) -> str:
    if not name:
        return name
    key = str(name).strip()
    if not key:
        return name
    return MODEL_ALIASES.get(key, name)
