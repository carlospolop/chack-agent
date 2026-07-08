from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from .config import ChackConfig, ModelConfig


OPENROUTER_MODEL_PREFIX = "openrouter/"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class OpenRouterRoute:
    model_name: str
    api_key: str
    base_url: str
    headers: dict[str, str]
    anthropic_base_url: str


def strip_openrouter_prefix(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if raw.lower().startswith(OPENROUTER_MODEL_PREFIX):
        return raw[len(OPENROUTER_MODEL_PREFIX) :]
    return raw


def uses_openrouter_route(*, provider: str, model_name: str) -> bool:
    normalized_provider = str(provider or "").strip().lower()
    raw_model = str(model_name or "").strip()
    if normalized_provider == "openrouter":
        return True
    return raw_model.lower().startswith(OPENROUTER_MODEL_PREFIX)


def _openrouter_base_url_from_config(config: ChackConfig) -> str:
    return (
        str(getattr(config.credentials, "openrouter_base_url", "") or "").strip()
        or os.environ.get("OPENROUTER_BASE_URL", "").strip()
        or OPENROUTER_DEFAULT_BASE_URL
    )


def _openrouter_headers_from_config(config: ChackConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = (
        str(getattr(config.credentials, "openrouter_http_referer", "") or "").strip()
        or os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    )
    title = (
        str(getattr(config.credentials, "openrouter_app_name", "") or "").strip()
        or os.environ.get("OPENROUTER_APP_NAME", "").strip()
    )
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _anthropic_base_url_from_openrouter(base_url: str) -> str:
    trimmed = str(base_url or "").rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[: -len("/v1")]
    return trimmed


def get_openrouter_route(config: ChackConfig, *, model_name: str | None = None) -> OpenRouterRoute | None:
    raw_model = str(model_name or config.model.primary or "").strip()
    provider = str(getattr(config.model, "provider", "") or "").strip()
    if not uses_openrouter_route(provider=provider, model_name=raw_model):
        return None

    api_key = (
        str(getattr(config.credentials, "openrouter_api_key", "") or "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required when selecting an OpenRouter-routed model"
        )
    base_url = _openrouter_base_url_from_config(config)
    headers = _openrouter_headers_from_config(config)
    return OpenRouterRoute(
        model_name=strip_openrouter_prefix(raw_model),
        api_key=api_key,
        base_url=base_url,
        headers=headers,
        anthropic_base_url=_anthropic_base_url_from_openrouter(base_url),
    )


def _strip_prefixed_model(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    return strip_openrouter_prefix(raw)


def clone_config_for_openrouter(config: ChackConfig) -> ChackConfig:
    model = config.model
    routed_model = ModelConfig(
        primary=_strip_prefixed_model(model.primary),
        provider="openrouter",
        max_context_tokens=model.max_context_tokens,
        social_network=_strip_prefixed_model(model.social_network),
        scientific=_strip_prefixed_model(model.scientific),
        websearcher=_strip_prefixed_model(model.websearcher),
        business=_strip_prefixed_model(model.business),
        product=_strip_prefixed_model(model.product),
        legal=_strip_prefixed_model(model.legal),
        data_statistics=_strip_prefixed_model(model.data_statistics),
        news_media=_strip_prefixed_model(model.news_media),
        knowledge_graph=_strip_prefixed_model(model.knowledge_graph),
        religious=_strip_prefixed_model(model.religious),
        cli=_strip_prefixed_model(model.cli),
        subchack=_strip_prefixed_model(model.subchack),
        social_network_max_turns=model.social_network_max_turns,
        scientific_max_turns=model.scientific_max_turns,
        websearcher_max_turns=model.websearcher_max_turns,
        business_max_turns=model.business_max_turns,
        product_max_turns=model.product_max_turns,
        legal_max_turns=model.legal_max_turns,
        data_statistics_max_turns=model.data_statistics_max_turns,
        news_media_max_turns=model.news_media_max_turns,
        knowledge_graph_max_turns=model.knowledge_graph_max_turns,
        religious_max_turns=model.religious_max_turns,
        cli_max_turns=model.cli_max_turns,
        subchack_max_turns=model.subchack_max_turns,
    )
    return replace(config, model=routed_model)
