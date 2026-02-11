import json
import os
from typing import Dict


DEFAULT_ALIASES: Dict[str, str] = {
    "BEST_QUALITY": "google/gemini-3-pro-preview",
    "CHEAP_BUT_QUALITY": "google/gemini-3-flash-preview",
    "BEST_CHEAPEST": "xiaomi/mimo-v2-flash",
}


def _env_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for key, default_value in DEFAULT_ALIASES.items():
        value = os.environ.get(f"MODEL_ALIAS_{key}", default_value).strip()
        if value:
            aliases[key] = value
    return aliases


def handler(event, context):
    body = {"model_aliases": _env_aliases()}
    return {
        "statusCode": 200,
        "headers": {
            "content-type": "application/json",
            "cache-control": "public, max-age=60",
        },
        "body": json.dumps(body),
    }
