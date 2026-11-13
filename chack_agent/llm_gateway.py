"""Routes that send model traffic through an LLM gateway instead of a vendor API.

A gateway is any endpoint that speaks the vendor wire formats with its own key
and model names: OpenRouter, a LiteLLM proxy, Portkey, a company AI gateway...
Two kinds are configured here:

* ``openrouter`` -- selected by ``model.provider: openrouter``,
  ``model.api_key_type: openrouter`` or an ``openrouter/`` model prefix;
  credentials in ``OPENROUTER_*``. Vendor model IDs such as ``gpt-6-sol`` or
  ``claude-sonnet-5-5`` are sent as OpenRouter IDs (``openai/gpt-6-sol``,
  ``anthropic/claude-sonnet-5.5``).
* ``gateway`` -- any other OpenAI/Anthropic-compatible gateway, selected by
  ``model.provider: gateway``, ``model.api_key_type: gateway`` or a
  ``gateway/`` model prefix; credentials in ``LLM_GATEWAY_*`` (or the matching
  ``credentials.llm_gateway_*`` fields). ``LLM_GATEWAY_TYPE=litellm`` adds the
  LiteLLM extras: the ``/model/info`` catalog maps vendor model IDs (for
  example ``claude-opus-5-5``) to the gateway's own names and supplies prices.

Routing only decides where requests go. The Codex, Claude Code and opencode
backends keep running their CLIs against the gateway (for OpenRouter, Codex and
Claude Code do so when it is selected with ``api_key_type: openrouter``). The
Gemini CLI keeps running when the gateway serves Google's Gemini API: LiteLLM
does at its root for every model it serves, other gateways where
``LLM_GATEWAY_GEMINI_BASE_URL`` points. The other backends switch to the
Responses API backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Mapping

import requests

from .config import ChackConfig, CredentialsConfig
from .model_aliases import resolve_model_alias
from .pricing import ModelPricing


_LOGGER = logging.getLogger("chack.llm_gateway")

OPENROUTER_MODEL_PREFIX = "openrouter/"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
GATEWAY_MODEL_PREFIX = "gateway/"

GATEWAY_KIND_OPENROUTER = "openrouter"
GATEWAY_KIND_CUSTOM = "custom"
GATEWAY_KIND_LITELLM = "litellm"
_GATEWAY_TYPES = {GATEWAY_KIND_CUSTOM, GATEWAY_KIND_LITELLM}
_GATEWAY_API_KEY_TYPES = {"gateway", "llm_gateway", "litellm"}
_OPENROUTER_API_KEY_TYPES = {"openrouter", "openrouter_api"}
# Vendor prefix OpenRouter uses for a model family that the pricing table
# does not list yet.
_OPENROUTER_FAMILY_VENDORS = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("codex", "openai"),
    ("gemini", "google"),
    ("gemma", "google"),
    ("grok", "x-ai"),
    ("mistral", "mistralai"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
)

# Environment names the CLI subprocesses (and the Chack MCP tool server they
# start) read the active gateway from.
LLM_GATEWAY_ENV_TYPE = "LLM_GATEWAY_TYPE"
LLM_GATEWAY_ENV_BASE_URL = "LLM_GATEWAY_BASE_URL"
LLM_GATEWAY_ENV_ANTHROPIC_BASE_URL = "LLM_GATEWAY_ANTHROPIC_BASE_URL"
LLM_GATEWAY_ENV_GEMINI_BASE_URL = "LLM_GATEWAY_GEMINI_BASE_URL"
LLM_GATEWAY_ENV_API_KEY = "LLM_GATEWAY_API_KEY"
LLM_GATEWAY_ENV_HEADERS = "LLM_GATEWAY_HEADERS"
LLM_GATEWAY_ENV_MODEL_MAP = "LLM_GATEWAY_MODEL_MAP"

LLM_GATEWAY_ENV_NAMES = (
    LLM_GATEWAY_ENV_TYPE,
    LLM_GATEWAY_ENV_BASE_URL,
    LLM_GATEWAY_ENV_ANTHROPIC_BASE_URL,
    LLM_GATEWAY_ENV_GEMINI_BASE_URL,
    LLM_GATEWAY_ENV_API_KEY,
    LLM_GATEWAY_ENV_HEADERS,
    LLM_GATEWAY_ENV_MODEL_MAP,
)

_CATALOG_TTL_SECONDS = 3600.0
_CATALOG_FAILURE_TTL_SECONDS = 60.0

# Model-name fields of ModelConfig that a route rewrites (prefix removal and
# gateway name mapping). Every other field is copied unchanged.
_MODEL_NAME_FIELDS = (
    "primary",
    "social_network",
    "scientific",
    "websearcher",
    "business",
    "product",
    "travel",
    "legal",
    "data_statistics",
    "news_media",
    "knowledge_graph",
    "religious",
    "cli",
    "subchack",
    "researcher_administrator",
)


@dataclass(frozen=True)
class GatewayRoute:
    kind: str
    model_name: str
    api_key: str
    # OpenAI-compatible root (Responses/Chat Completions), usually ending in /v1.
    base_url: str
    # Anthropic Messages root, without /v1 (Claude Code appends /v1/messages).
    anthropic_base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    # Context window the gateway reports for the model; 0 when unknown.
    context_window: int = 0
    # Upstream model the gateway forwards to (e.g. "bedrock/..."); "" when unknown.
    upstream: str = ""
    # Google Gemini API root (the Gemini CLI appends /v1beta/models/...); ""
    # when the gateway does not serve that API.
    gemini_base_url: str = ""

    @property
    def is_openrouter(self) -> bool:
        return self.kind == GATEWAY_KIND_OPENROUTER

    @property
    def display_name(self) -> str:
        if self.kind == GATEWAY_KIND_OPENROUTER:
            return "OpenRouter"
        if self.kind == GATEWAY_KIND_LITELLM:
            return "LiteLLM gateway"
        return "LLM gateway"

    @property
    def api_key_env(self) -> str:
        return "OPENROUTER_API_KEY" if self.is_openrouter else LLM_GATEWAY_ENV_API_KEY


@dataclass(frozen=True)
class GatewayModel:
    name: str
    upstream: str = ""
    context_window: int = 0
    # USD per 1M tokens; None when the gateway does not report the rate.
    input: float | None = None
    cached_input: float | None = None
    cache_write: float | None = None
    output: float | None = None


# ---------------------------------------------------------------------------
# Selection


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _strip_prefix(model_name: str, prefix: str) -> str:
    raw = _clean(model_name)
    if raw.lower().startswith(prefix):
        return raw[len(prefix) :]
    return raw


def strip_openrouter_prefix(model_name: str) -> str:
    return _strip_prefix(model_name, OPENROUTER_MODEL_PREFIX)


def strip_gateway_prefix(model_name: str) -> str:
    return _strip_prefix(model_name, GATEWAY_MODEL_PREFIX)


def uses_openrouter_route(*, provider: str, model_name: str, api_key_type: str = "") -> bool:
    if _clean(provider).lower() == "openrouter":
        return True
    if _clean(api_key_type).lower() in _OPENROUTER_API_KEY_TYPES:
        return True
    return _clean(model_name).lower().startswith(OPENROUTER_MODEL_PREFIX)


def openrouter_selected_by_api_key_type(config: ChackConfig) -> bool:
    """True when OpenRouter is the key of a CLI backend, not a model prefix.

    ``api_key_type: openrouter`` keeps the Codex and Claude Code CLIs and points
    them at OpenRouter; the ``openrouter/`` model prefix keeps selecting the
    Responses API backend.
    """

    api_key_type = _clean(getattr(config.model, "api_key_type", "")).lower()
    return api_key_type in _OPENROUTER_API_KEY_TYPES


def _openrouter_pricing_ids() -> list[str]:
    try:
        from .pricing import load_pricing, resolve_pricing_path

        table = load_pricing(resolve_pricing_path(refresh=False))
    except Exception as exc:  # pragma: no cover - packaged table is always present
        _LOGGER.debug("Pricing table unavailable for OpenRouter model IDs: %s", exc)
        return []
    return [name for name in table.models if "/" in name and ":" not in name]


def openrouter_model_id(model_name: str) -> str:
    """OpenRouter ID for a model: ``claude-sonnet-5-5`` -> ``anthropic/claude-sonnet-5.5``.

    Names that already carry a vendor (``openai/gpt-6-sol``) are kept. Bare
    vendor IDs are looked up in the pricing table, which lists OpenRouter's
    catalog; unknown ones get the vendor prefix of their family.
    """

    name = strip_openrouter_prefix(model_name)
    if not name or "/" in name:
        return name
    wanted = normalize_model_id(name)
    matches = sorted(
        (candidate for candidate in _openrouter_pricing_ids() if normalize_model_id(candidate) == wanted),
        key=lambda candidate: (len(candidate), candidate),
    )
    if matches:
        return matches[0]
    lowered = name.lower()
    for family, vendor in _OPENROUTER_FAMILY_VENDORS:
        if lowered.startswith(family):
            if vendor == "anthropic":
                # Anthropic IDs use a dotted version on OpenRouter.
                lowered = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", lowered)
            return f"{vendor}/{lowered}"
    return name


def uses_custom_gateway_route(*, provider: str, model_name: str, api_key_type: str = "") -> bool:
    if _clean(provider).lower() == "gateway":
        return True
    if _clean(api_key_type).lower() in _GATEWAY_API_KEY_TYPES:
        return True
    return _clean(model_name).lower().startswith(GATEWAY_MODEL_PREFIX)


def _credential(config: ChackConfig, attribute: str, env_name: str) -> str:
    return _clean(getattr(config.credentials, attribute, "")) or _clean(os.environ.get(env_name, ""))


def _mapping(config: ChackConfig, attribute: str, env_name: str) -> dict[str, str]:
    value: Any = getattr(config.credentials, attribute, None)
    if not value:
        value = os.environ.get(env_name, "")
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{env_name} must be a JSON object of strings") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{env_name} must be a JSON object of strings")
    return {_clean(k): _clean(v) for k, v in value.items() if _clean(k) and _clean(v)}


def _anthropic_root(base_url: str) -> str:
    trimmed = _clean(base_url).rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[: -len("/v1")]
    return trimmed


# ---------------------------------------------------------------------------
# OpenRouter


def _openrouter_headers(config: ChackConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = _credential(config, "openrouter_http_referer", "OPENROUTER_HTTP_REFERER")
    title = _credential(config, "openrouter_app_name", "OPENROUTER_APP_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _openrouter_route(config: ChackConfig, raw_model: str) -> GatewayRoute:
    api_key = _credential(config, "openrouter_api_key", "OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required when selecting an OpenRouter-routed model")
    base_url = (
        _credential(config, "openrouter_base_url", "OPENROUTER_BASE_URL")
        or OPENROUTER_DEFAULT_BASE_URL
    )
    return GatewayRoute(
        kind=GATEWAY_KIND_OPENROUTER,
        model_name=openrouter_model_id(raw_model),
        api_key=api_key,
        base_url=base_url,
        anthropic_base_url=_anthropic_root(base_url),
        headers=_openrouter_headers(config),
    )


# ---------------------------------------------------------------------------
# LiteLLM catalog


_CATALOG_LOCK = threading.Lock()
_CATALOG_CACHE: dict[str, tuple[float, dict[str, GatewayModel]]] = {}

# Region and vendor qualifiers that clouds put in front of a vendor model ID,
# e.g. "global.openai.gpt-5.6-sol" or "us.anthropic.claude-opus-5-5-v1:0".
_UPSTREAM_QUALIFIER_RE = re.compile(
    r"^(?:(?:global|us|eu|apac|jp|au|ca)\.)?"
    r"(?:openai|anthropic|google|meta|mistral|cohere|amazon|deepseek|qwen|xai|"
    r"moonshotai|zai|minimax|nvidia)\."
)


def normalize_model_id(model_name: str) -> str:
    """Comparable vendor model ID: no routing prefix, region, version suffix or dots."""

    bare = _clean(model_name).lower().rsplit("/", 1)[-1]
    bare = bare.split("@", 1)[0]
    bare = _UPSTREAM_QUALIFIER_RE.sub("", bare)
    bare = re.sub(r"-v\d+(?::\d+)?$", "", bare)
    bare = re.sub(r"-maas$", "", bare)
    return bare.replace(".", "-")


def _per_million(value: Any) -> float | None:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    return rate * 1_000_000.0


def _parse_litellm_catalog(payload: Any) -> dict[str, GatewayModel]:
    entries = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("LiteLLM /model/info returned no 'data' list")
    catalog: dict[str, GatewayModel] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = _clean(entry.get("model_name"))
        if not name or name in catalog:
            continue
        params = entry.get("litellm_params") if isinstance(entry.get("litellm_params"), Mapping) else {}
        info = entry.get("model_info") if isinstance(entry.get("model_info"), Mapping) else {}
        try:
            context_window = int(info.get("max_input_tokens") or 0)
        except (TypeError, ValueError):
            context_window = 0
        catalog[name] = GatewayModel(
            name=name,
            upstream=_clean(params.get("model")),
            context_window=max(0, context_window),
            input=_per_million(info.get("input_cost_per_token")),
            cached_input=_per_million(info.get("cache_read_input_token_cost")),
            cache_write=_per_million(info.get("cache_creation_input_token_cost")),
            output=_per_million(info.get("output_cost_per_token")),
        )
    return catalog


def _parse_openai_models(payload: Any) -> dict[str, GatewayModel]:
    """Parse the OpenAI-compatible ``/models`` list most gateways implement."""

    entries = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("/models returned no 'data' list")
    catalog: dict[str, GatewayModel] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = _clean(entry.get("id"))
        if not name or name in catalog:
            continue
        try:
            context_window = int(entry.get("context_length") or entry.get("context_window") or 0)
        except (TypeError, ValueError):
            context_window = 0
        # The ID is usually the vendor ID itself, possibly behind a vendor prefix.
        catalog[name] = GatewayModel(name=name, upstream=name, context_window=max(0, context_window))
    return catalog


def _http_get(url: str, *, headers: Mapping[str, str], timeout: float) -> Any:
    return requests.get(url, headers=dict(headers), timeout=timeout)


def _catalog_timeout_seconds() -> float:
    try:
        return max(0.5, float(os.environ.get("LLM_GATEWAY_CATALOG_TIMEOUT_SECONDS", "15")))
    except ValueError:
        return 15.0


def gateway_catalog(
    kind: str,
    base_url: str,
    api_key: str,
    headers: Mapping[str, str] | None = None,
) -> dict[str, GatewayModel]:
    """Models the gateway serves to this key, cached per gateway and key.

    LiteLLM's ``/model/info`` adds upstream IDs, context windows and prices;
    other gateways are read through the OpenAI-compatible ``/models`` list.
    A failed refresh keeps the last catalog read from the gateway (its prices
    and context windows are still right); with none, it returns an empty
    catalog (model names are then used as given and cannot be checked).
    Either way the read is retried after a short back-off.
    """

    root = _clean(base_url).rstrip("/")
    litellm = kind == GATEWAY_KIND_LITELLM
    cache_key = hashlib.sha256(f"{kind}\0{root}\0{api_key}".encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _CATALOG_LOCK:
        cached = _CATALOG_CACHE.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1]
    last_read = cached[1] if cached is not None else {}
    request_headers = dict(headers or {})
    request_headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = _http_get(
            f"{root}/model/info" if litellm else f"{root}/models",
            headers=request_headers,
            timeout=_catalog_timeout_seconds(),
        )
        response.raise_for_status()
        parse = _parse_litellm_catalog if litellm else _parse_openai_models
        catalog = parse(response.json())
        expires_at = now + _CATALOG_TTL_SECONDS
    except Exception as exc:
        _LOGGER.warning(
            "Could not read the model list of gateway %s%s: %s",
            root,
            "; keeping the last one read" if last_read else "",
            exc,
        )
        catalog = last_read
        expires_at = now + _CATALOG_FAILURE_TTL_SECONDS
    with _CATALOG_LOCK:
        _CATALOG_CACHE[cache_key] = (expires_at, catalog)
    return catalog


def resolve_catalog_model(requested: str, catalog: Mapping[str, GatewayModel]) -> str:
    """Gateway name for a requested model; unknown names are returned unchanged."""

    name = _clean(requested)
    if not name or name in catalog:
        return name
    wanted = normalize_model_id(name)
    candidates = sorted(
        (
            model.name
            for model in catalog.values()
            if model.upstream and normalize_model_id(model.upstream) == wanted
        ),
        key=lambda candidate: (len(candidate), candidate),
    )
    return candidates[0] if candidates else name


# ---------------------------------------------------------------------------
# Custom gateway


@dataclass(frozen=True)
class _GatewaySettings:
    kind: str
    base_url: str
    anthropic_base_url: str
    gemini_base_url: str
    api_key: str
    headers: dict[str, str]
    model_map: dict[str, str]


def _gateway_settings(config: ChackConfig) -> _GatewaySettings:
    kind = _credential(config, "llm_gateway_type", LLM_GATEWAY_ENV_TYPE).lower() or GATEWAY_KIND_CUSTOM
    if kind not in _GATEWAY_TYPES:
        raise ValueError(
            f"{LLM_GATEWAY_ENV_TYPE} must be one of {sorted(_GATEWAY_TYPES)} (got {kind!r})"
        )
    base_url = _credential(config, "llm_gateway_base_url", LLM_GATEWAY_ENV_BASE_URL).rstrip("/")
    api_key = _credential(config, "llm_gateway_api_key", LLM_GATEWAY_ENV_API_KEY)
    if not base_url or not api_key:
        raise ValueError(
            f"{LLM_GATEWAY_ENV_BASE_URL} and {LLM_GATEWAY_ENV_API_KEY} are required when "
            "selecting a gateway-routed model"
        )
    anthropic_base_url = (
        _credential(config, "llm_gateway_anthropic_base_url", LLM_GATEWAY_ENV_ANTHROPIC_BASE_URL)
        or _anthropic_root(base_url)
    ).rstrip("/")
    # LiteLLM serves the Gemini API at its root and translates it for every
    # model it serves, not only Gemini ones; other gateways must name theirs.
    gemini_base_url = (
        _credential(config, "llm_gateway_gemini_base_url", LLM_GATEWAY_ENV_GEMINI_BASE_URL)
        or (_anthropic_root(base_url) if kind == GATEWAY_KIND_LITELLM else "")
    ).rstrip("/")
    return _GatewaySettings(
        kind=kind,
        base_url=base_url,
        anthropic_base_url=anthropic_base_url,
        gemini_base_url=gemini_base_url,
        api_key=api_key,
        headers=_mapping(config, "llm_gateway_headers", LLM_GATEWAY_ENV_HEADERS),
        model_map=_mapping(config, "llm_gateway_model_map", LLM_GATEWAY_ENV_MODEL_MAP),
    )


def _settings_catalog(settings: _GatewaySettings) -> dict[str, GatewayModel]:
    return gateway_catalog(settings.kind, settings.base_url, settings.api_key, settings.headers)


_UNAVAILABLE_MODEL_LIST_LIMIT = 40


def _unavailable_model_error(requested: str, resolved: str, catalog: Mapping[str, GatewayModel]) -> ValueError:
    names = sorted(catalog)
    listed = ", ".join(names[:_UNAVAILABLE_MODEL_LIST_LIMIT])
    if len(names) > _UNAVAILABLE_MODEL_LIST_LIMIT:
        listed += f", ... ({len(names) - _UNAVAILABLE_MODEL_LIST_LIMIT} more)"
    shown = requested if requested == resolved else f"{requested} (as {resolved})"
    return ValueError(f"Model {shown} is not available to this gateway API key. Available models: {listed}")


def _gateway_model_name(
    raw_model: str,
    settings: _GatewaySettings,
    catalog: Mapping[str, GatewayModel],
) -> str:
    name = strip_gateway_prefix(raw_model)
    if not name:
        return name
    resolved = settings.model_map.get(name) or resolve_catalog_model(name, catalog)
    if catalog and resolved not in catalog:
        raise _unavailable_model_error(name, resolved, catalog)
    return resolved


def _custom_gateway_route(config: ChackConfig, raw_model: str) -> GatewayRoute:
    settings = _gateway_settings(config)
    catalog = _settings_catalog(settings)
    model_name = _gateway_model_name(raw_model, settings, catalog)
    entry = catalog.get(model_name)
    return GatewayRoute(
        kind=settings.kind,
        model_name=model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        anthropic_base_url=settings.anthropic_base_url,
        headers=settings.headers,
        context_window=entry.context_window if entry is not None else 0,
        upstream=entry.upstream if entry is not None else "",
        gemini_base_url=settings.gemini_base_url,
    )


# ---------------------------------------------------------------------------
# Public API


def get_gateway_route(config: ChackConfig, *, model_name: str | None = None) -> GatewayRoute | None:
    """Gateway route for the configured (or given) model, or None for vendor APIs."""

    raw_model = _clean(model_name or config.model.primary)
    provider = _clean(getattr(config.model, "provider", ""))
    api_key_type = _clean(getattr(config.model, "api_key_type", ""))
    if uses_openrouter_route(provider=provider, model_name=raw_model, api_key_type=api_key_type):
        return _openrouter_route(config, raw_model)
    if uses_custom_gateway_route(
        provider=provider,
        model_name=raw_model,
        api_key_type=api_key_type,
    ):
        return _custom_gateway_route(config, raw_model)
    return None


def clone_config_for_gateway(config: ChackConfig, route: GatewayRoute) -> ChackConfig:
    """Config for the Responses API backend: gateway provider and gateway model names."""

    model = config.model
    if route.is_openrouter:
        rewritten = {name: openrouter_model_id(getattr(model, name)) for name in _MODEL_NAME_FIELDS}
        return replace(config, model=replace(model, provider="openrouter", **rewritten))
    settings = _gateway_settings(config)
    catalog = _settings_catalog(settings)
    primary = _gateway_model_name(model.primary, settings, catalog)
    rewritten = {"primary": primary}
    for name in _MODEL_NAME_FIELDS[1:]:
        requested = _clean(getattr(model, name))
        resolved = _clean(resolve_model_alias(requested, provider=_clean(model.provider)))
        try:
            rewritten[name] = _gateway_model_name(resolved, settings, catalog)
        except ValueError:
            if requested == resolved:
                raise
            # Built-in aliases name vendor models this key may not be allowed to use.
            _LOGGER.warning(
                "Gateway key cannot use %s (alias %s) for model.%s; using %s instead",
                resolved,
                requested,
                name,
                primary,
            )
            rewritten[name] = primary
    return replace(config, model=replace(model, provider="gateway", **rewritten))


def route_model_name(config: ChackConfig, route: GatewayRoute, model_name: str) -> str:
    """Name ``route`` uses for vendor model ``model_name``, or "" if it is not served."""

    if route.is_openrouter:
        return openrouter_model_id(model_name)
    return gateway_model_name(config, model_name)


def gateway_model_name(config: ChackConfig, model_name: str) -> str:
    """Gateway name for a vendor model ID if the configured gateway serves it, else ""."""

    settings = _gateway_settings(config)
    name = strip_gateway_prefix(model_name)
    if name in settings.model_map:
        return settings.model_map[name]
    catalog = _settings_catalog(settings)
    resolved = resolve_catalog_model(name, catalog)
    return resolved if resolved in catalog else ""


def list_gateway_models(config: ChackConfig | CredentialsConfig | None = None) -> list[GatewayModel]:
    """Models the configured gateway API key may use, sorted by name.

    Takes a full config or just its credentials; unset fields fall back to the
    ``LLM_GATEWAY_*`` environment. Empty when the gateway does not publish a
    model list or cannot be reached.
    """

    if not isinstance(config, ChackConfig):
        config = SimpleNamespace(credentials=config or CredentialsConfig())
    catalog = _settings_catalog(_gateway_settings(config))
    return [catalog[name] for name in sorted(catalog)]


def gateway_subprocess_env(config: ChackConfig, route: GatewayRoute) -> dict[str, str]:
    """Variables that let CLI subprocesses and the Chack MCP server reach the same gateway."""

    if route.is_openrouter:
        env = {"OPENROUTER_API_KEY": route.api_key, "OPENROUTER_BASE_URL": route.base_url}
        if route.headers.get("HTTP-Referer"):
            env["OPENROUTER_HTTP_REFERER"] = route.headers["HTTP-Referer"]
        if route.headers.get("X-Title"):
            env["OPENROUTER_APP_NAME"] = route.headers["X-Title"]
        return env
    settings = _gateway_settings(config)
    env = {
        LLM_GATEWAY_ENV_TYPE: settings.kind,
        LLM_GATEWAY_ENV_BASE_URL: settings.base_url,
        LLM_GATEWAY_ENV_ANTHROPIC_BASE_URL: settings.anthropic_base_url,
        LLM_GATEWAY_ENV_API_KEY: settings.api_key,
    }
    if settings.gemini_base_url:
        env[LLM_GATEWAY_ENV_GEMINI_BASE_URL] = settings.gemini_base_url
    if settings.headers:
        env[LLM_GATEWAY_ENV_HEADERS] = json.dumps(settings.headers, sort_keys=True)
    if settings.model_map:
        env[LLM_GATEWAY_ENV_MODEL_MAP] = json.dumps(settings.model_map, sort_keys=True)
    return env


def gateway_model_pricing(config: ChackConfig) -> dict[str, ModelPricing]:
    """USD per 1M tokens, keyed by gateway model name.

    Only LiteLLM gateways report prices. Models without both an input and an
    output rate are left out so the shared pricing table still applies to them.
    """

    route = get_gateway_route(config)
    if route is None or route.kind != GATEWAY_KIND_LITELLM:
        return {}
    settings = _gateway_settings(config)
    prices: dict[str, ModelPricing] = {}
    for model in _settings_catalog(settings).values():
        if model.input is None or model.output is None:
            continue
        prices[model.name] = ModelPricing(
            input=model.input,
            cached_input=model.cached_input if model.cached_input is not None else model.input,
            output=model.output,
            cache_write=model.cache_write if model.cache_write is not None else model.input,
        )
    # Usage is also reported under the configured names (e.g. "gpt-6-sol"),
    # which the gateway bills at the rate of the model they resolve to.
    for field_name in _MODEL_NAME_FIELDS:
        requested = _clean(getattr(config.model, field_name, ""))
        resolved = gateway_model_name(config, requested) if requested else ""
        if resolved in prices:
            prices.setdefault(requested, prices[resolved])
    return prices
