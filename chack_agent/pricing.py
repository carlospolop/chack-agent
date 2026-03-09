from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


DEFAULT_REMOTE_PRICING_URL = (
    "https://raw.githubusercontent.com/"
    "carlospolop/chack-agent/master/chack_agent/config/pricing.yaml"
)
DEFAULT_REMOTE_TIMEOUT_SECONDS = 1.0


@dataclass
class ModelPricing:
    input: float
    cached_input: float
    output: float
    cache_write: float = 0.0


@dataclass
class PricingTable:
    models: Dict[str, ModelPricing]


def _package_pricing_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "config", "pricing.yaml")


def _cache_root_dir() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "chack-agent"
    return Path.home() / ".cache" / "chack-agent"


def _cached_pricing_path() -> Path:
    return _cache_root_dir() / "pricing" / "pricing.yaml"


def _cached_pricing_metadata_path() -> Path:
    return _cache_root_dir() / "pricing" / "pricing-meta.json"


def _load_pricing_metadata() -> dict[str, str]:
    metadata_path = _cached_pricing_metadata_path()
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _write_pricing_metadata(metadata: dict[str, str]) -> None:
    metadata_path = _cached_pricing_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True)


def _write_cached_pricing(body: bytes, metadata: dict[str, str]) -> None:
    pricing_path = _cached_pricing_path()
    pricing_path.parent.mkdir(parents=True, exist_ok=True)
    pricing_path.write_bytes(body)
    _write_pricing_metadata(metadata)


def _auto_refresh_enabled() -> bool:
    value = os.environ.get("CHACK_PRICING_AUTO_REFRESH", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _remote_timeout_seconds() -> float:
    raw = os.environ.get("CHACK_PRICING_REFRESH_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_REMOTE_TIMEOUT_SECONDS
    try:
        return max(float(raw), 0.1)
    except ValueError:
        return DEFAULT_REMOTE_TIMEOUT_SECONDS


def refresh_pricing_from_github_if_newer() -> str:
    override = os.environ.get("CHACK_PRICING", "").strip()
    if override:
        return override
    if not _auto_refresh_enabled():
        return resolve_pricing_path(refresh=False)

    remote_url = (
        os.environ.get("CHACK_PRICING_REMOTE_URL", "").strip()
        or DEFAULT_REMOTE_PRICING_URL
    )
    metadata = _load_pricing_metadata()
    headers = {
        "Accept": "application/x-yaml, text/yaml, text/plain",
        "User-Agent": "chack-agent-pricing/1",
    }
    etag = metadata.get("etag", "").strip()
    last_modified = metadata.get("last_modified", "").strip()
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    request = Request(remote_url, headers=headers)
    try:
        with urlopen(request, timeout=_remote_timeout_seconds()) as response:
            body = response.read()
            if body:
                updated_metadata = {}
                response_etag = response.headers.get("ETag", "").strip()
                response_last_modified = response.headers.get("Last-Modified", "").strip()
                if response_etag:
                    updated_metadata["etag"] = response_etag
                if response_last_modified:
                    updated_metadata["last_modified"] = response_last_modified
                _write_cached_pricing(body, updated_metadata)
    except HTTPError as exc:
        if exc.code != 304:
            return resolve_pricing_path(refresh=False)
    except (OSError, URLError, TimeoutError):
        return resolve_pricing_path(refresh=False)

    return resolve_pricing_path(refresh=False)


def _strip_provider_prefix(model_name: str) -> str:
    raw = str(model_name or "").strip()
    return raw.split("/", 1)[1] if "/" in raw else raw


def _resolve_model_lookup(pricing: PricingTable, model: str) -> Optional[str]:
    lookup = str(model or "").strip()
    if not lookup:
        return None
    if lookup in pricing.models:
        return lookup

    bare_lookup = _strip_provider_prefix(lookup)
    stripped_matches = [
        key for key in pricing.models.keys() if _strip_provider_prefix(key) == bare_lookup
    ]
    if len(stripped_matches) == 1:
        return stripped_matches[0]
    return None


def _build_pricing_table(raw_models: Dict[str, Dict[str, float]]) -> PricingTable:
    models: Dict[str, ModelPricing] = {}
    for name, values in raw_models.items():
        if not isinstance(values, dict):
            continue
        try:
            cache_read = values.get("cached_input")
            if cache_read is None:
                cache_read = values.get("cache_read")
            if cache_read is None:
                cache_read = values.get("input_cache_read")
            cache_write = values.get("cache_write")
            if cache_write is None:
                cache_write = values.get("input_cache_write")
            models[name] = ModelPricing(
                input=float(values.get("input", 0.0)),
                cached_input=float(cache_read or 0.0),
                output=float(values.get("output", 0.0)),
                cache_write=float(cache_write or 0.0),
            )
        except (TypeError, ValueError):
            continue
    return PricingTable(models=models)


def load_pricing(path: str) -> PricingTable:
    raw_models: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        raw_models = raw.get("models", {}) or {}
    except OSError:
        raw_models = {}

    return _build_pricing_table(raw_models)


def resolve_pricing_path(refresh: bool = True) -> str:
    override = os.environ.get("CHACK_PRICING")
    if override:
        return override
    if refresh:
        refreshed = refresh_pricing_from_github_if_newer()
        if refreshed:
            return refreshed
    cached_path = _cached_pricing_path()
    if cached_path.exists():
        return str(cached_path)
    return _package_pricing_path()


def estimate_cost(
    pricing: PricingTable,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Optional[float]:
    lookup = _resolve_model_lookup(pricing, model)
    if not lookup:
        return None

    rates = pricing.models[lookup]
    billable_prompt = max(prompt_tokens - cached_prompt_tokens, 0)
    total = (
        billable_prompt * rates.input
        + cached_prompt_tokens * rates.cached_input
        + cache_write_tokens * rates.cache_write
        + completion_tokens * rates.output
    )
    return total / 1_000_000.0


def estimate_costs_by_model(
    pricing: PricingTable,
    usage_by_model: Dict[str, Tuple[int, int, int, int]],
) -> tuple[float, List[str]]:
    total = 0.0
    missing_models: List[str] = []
    for model_name, usage in usage_by_model.items():
        cache_write_tokens = 0
        if len(usage) == 4:
            prompt_tokens, completion_tokens, cached_prompt_tokens, cache_write_tokens = usage
        else:
            prompt_tokens, completion_tokens, cached_prompt_tokens = usage
        model_cost = estimate_cost(
            pricing,
            model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        if model_cost is None:
            missing_models.append(model_name)
            continue
        total += model_cost
    return total, missing_models
