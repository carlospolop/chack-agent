from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import yaml


DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5.2-chat-latest": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.2-codex": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-codex-max": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
}


@dataclass
class ModelPricing:
    input: float
    cached_input: float
    output: float
    cache_write: float = 0.0


@dataclass
class PricingTable:
    models: Dict[str, ModelPricing]


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


def _merge_with_defaults(table: PricingTable) -> PricingTable:
    merged = dict(table.models)
    for name, defaults in DEFAULT_PRICING.items():
        current = merged.get(name)
        if current is None:
            merged[name] = ModelPricing(**defaults)
            continue
        if (
            current.input == 0.0
            and current.cached_input == 0.0
            and current.output == 0.0
        ):
            merged[name] = ModelPricing(**defaults)
    return PricingTable(models=merged)


def load_pricing(path: str) -> PricingTable:
    raw_models: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        raw_models = raw.get("models", {}) or {}
    except OSError:
        raw_models = {}

    table = _build_pricing_table(raw_models)
    if not table.models:
        table = _build_pricing_table(DEFAULT_PRICING)
        return table
    return _merge_with_defaults(table)


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
    if model not in pricing.models:
        return None
    rates = pricing.models[model]
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


def estimate_cost_with_defaults(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Optional[float]:
    if model not in DEFAULT_PRICING:
        return None
    rates = DEFAULT_PRICING[model]
    billable_prompt = max(prompt_tokens - cached_prompt_tokens, 0)
    total = (
        billable_prompt * rates["input"]
        + cached_prompt_tokens * rates["cached_input"]
        + cache_write_tokens * rates.get("cache_write", 0.0)
        + completion_tokens * rates["output"]
    )
    return total / 1_000_000.0
