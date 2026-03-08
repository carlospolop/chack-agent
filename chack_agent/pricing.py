from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import yaml

@dataclass
class ModelPricing:
    input: float
    cached_input: float
    output: float
    cache_write: float = 0.0


@dataclass
class PricingTable:
    models: Dict[str, ModelPricing]


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


def resolve_pricing_path() -> str:
    override = os.environ.get("CHACK_PRICING")
    if override:
        return override
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "config", "pricing.yaml")


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
