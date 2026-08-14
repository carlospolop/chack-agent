from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OUTPUT_PATH = Path("chack_agent/config/pricing.yaml")
DEFAULT_EFFORT_OUTPUT_PATH = Path("chack_agent/config/thinking_effort.yaml")
MICRO_TOKENS = Decimal("1000000")

# Mirrors chack_agent.thinking_effort: the shared vocabulary, and the key shape
# that makes every spelling of one model (OpenRouter's ``anthropic/claude-opus-
# 4.6``, the API's ``claude-opus-4-6``, Copilot's ``claude-sonnet-4.6``) agree.
THINKING_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _decimal_million(value: Any) -> Decimal:
    raw = str(value or "0").strip() or "0"
    try:
        return Decimal(raw) * MICRO_TOKENS
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _format_decimal(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _load_models(models_url: str, api_key: str | None) -> list[dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "chack-agent-pricing-updater",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(models_url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

    if isinstance(payload, dict):
        models = payload.get("data", payload.get("models", []))
    else:
        models = payload
    if not isinstance(models, list):
        raise RuntimeError("Unexpected OpenRouter response format for models payload")
    return [item for item in models if isinstance(item, dict)]


def _build_yaml(models: list[dict[str, Any]]) -> str:
    lines = ["models:"]
    ordered = sorted(models, key=lambda item: str(item.get("id", "")))
    for item in ordered:
        model_id = str(item.get("id", "")).strip()
        if not model_id:
            continue
        pricing = item.get("pricing") or {}
        if not isinstance(pricing, dict):
            pricing = {}

        input_price = _decimal_million(pricing.get("prompt"))
        raw_cache_read = pricing.get("input_cache_read")
        cache_read_price = (
            _decimal_million(raw_cache_read)
            if raw_cache_read is not None
            else input_price
        )
        cache_write_price = _decimal_million(pricing.get("input_cache_write"))
        output_price = _decimal_million(pricing.get("completion"))

        lines.append(f"  {model_id}:")
        lines.append(f"    input: {_format_decimal(input_price)}")
        lines.append(f"    cache_read: {_format_decimal(cache_read_price)}")
        if cache_write_price != 0:
            lines.append(f"    cache_write: {_format_decimal(cache_write_price)}")
        lines.append(f"    output: {_format_decimal(output_price)}")
    return "\n".join(lines) + "\n"


def _effort_key(model_id: str) -> str:
    key = str(model_id or "").strip().lower().split(":", 1)[0]
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    return key.replace(".", "-")


def _build_effort_yaml(models: list[dict[str, Any]]) -> str:
    """Record the reasoning levels OpenRouter publishes for each model.

    Models with no ``supported_efforts`` are skipped rather than written as
    "no levels": OpenRouter only reports the ones it exposes an ``effort``
    parameter for, and Gemini 2.5 (a token budget instead of an effort enum)
    would otherwise look like it had no thinking control at all.
    """
    collected: dict[str, set[str]] = {}
    for item in models:
        key = _effort_key(str(item.get("id", "")))
        if not key:
            continue
        reasoning = item.get("reasoning")
        if not isinstance(reasoning, dict):
            continue
        efforts = reasoning.get("supported_efforts")
        if not isinstance(efforts, list):
            continue
        known = {
            str(effort).strip().lower()
            for effort in efforts
            if str(effort).strip().lower() in THINKING_EFFORT_ORDER
        }
        if known:
            collected.setdefault(key, set()).update(known)

    lines = [
        "# Reasoning effort levels each model accepts.",
        "# Generated from OpenRouter's reasoning.supported_efforts by",
        "# scripts/update_openrouter_pricing.py -- do not edit by hand.",
        "models:",
    ]
    for key in sorted(collected):
        ordered = [level for level in THINKING_EFFORT_ORDER if level in collected[key]]
        lines.append(f"  {key}: [{', '.join(ordered)}]")
    return "\n".join(lines) + "\n"


def update_pricing(
    output_path: Path,
    models_url: str,
    api_key: str | None,
    effort_output_path: Path | None = None,
) -> None:
    models = _load_models(models_url=models_url, api_key=api_key)
    output_path.write_text(_build_yaml(models), encoding="utf-8")
    if effort_output_path is not None:
        effort_output_path.write_text(_build_effort_yaml(models), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the OpenRouter model list and write the chack pricing and "
            "thinking effort YAML files."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output YAML path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--effort-output",
        default=str(DEFAULT_EFFORT_OUTPUT_PATH),
        help=(
            "Supported thinking effort YAML path. Defaults to "
            f"{DEFAULT_EFFORT_OUTPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--models-url",
        default=os.environ.get("OPENROUTER_MODELS_URL", DEFAULT_MODELS_URL),
        help=f"OpenRouter models API URL. Defaults to {DEFAULT_MODELS_URL}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or None
    update_pricing(
        output_path=Path(args.output),
        models_url=args.models_url,
        api_key=api_key,
        effort_output_path=Path(args.effort_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
