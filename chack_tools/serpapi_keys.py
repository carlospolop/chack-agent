from __future__ import annotations

import random
from typing import Any


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
