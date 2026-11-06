"""OpenRouter-only view of :mod:`chack_agent.llm_gateway`, kept for existing imports."""

from __future__ import annotations

from .config import ChackConfig
from .llm_gateway import (
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_MODEL_PREFIX,
    GatewayRoute,
    clone_config_for_gateway,
    get_gateway_route,
    strip_openrouter_prefix,
    uses_openrouter_route,
)

OpenRouterRoute = GatewayRoute


def get_openrouter_route(config: ChackConfig, *, model_name: str | None = None) -> GatewayRoute | None:
    route = get_gateway_route(config, model_name=model_name)
    return route if route is not None and route.is_openrouter else None


def clone_config_for_openrouter(config: ChackConfig) -> ChackConfig:
    route = get_openrouter_route(config)
    if route is None:
        raise ValueError("clone_config_for_openrouter requires an OpenRouter-routed model")
    return clone_config_for_gateway(config, route)


__all__ = [
    "OPENROUTER_DEFAULT_BASE_URL",
    "OPENROUTER_MODEL_PREFIX",
    "OpenRouterRoute",
    "clone_config_for_openrouter",
    "get_openrouter_route",
    "strip_openrouter_prefix",
    "uses_openrouter_route",
]
