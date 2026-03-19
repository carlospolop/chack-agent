import json
import os
from typing import Dict

from chack_agent.model_aliases import get_public_backend_aliases, get_public_model_aliases

DEFAULT_MODEL_ALIASES: Dict[str, str] = get_public_model_aliases()
DEFAULT_BACKEND_ALIASES: Dict[str, str] = get_public_backend_aliases()

def _env_aliases(prefix: str, defaults: Dict[str, str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for key, default_value in defaults.items():
        value = os.environ.get(f"{prefix}{key}", default_value).strip()
        if value:
            aliases[key] = value
    return aliases


def handler(event, context):
    body = {
        "model_aliases": _env_aliases("MODEL_ALIAS_", DEFAULT_MODEL_ALIASES),
        "backend_aliases": _env_aliases("BACKEND_ALIAS_", DEFAULT_BACKEND_ALIASES),
    }
    return {
        "statusCode": 200,
        "headers": {
            "content-type": "application/json",
            "cache-control": "public, max-age=60",
        },
        "body": json.dumps(body),
    }
