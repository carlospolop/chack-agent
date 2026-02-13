from __future__ import annotations

from typing import Any, Mapping

from .config import ToolsConfig as BaseToolsConfig


def _build_tools_config(base: BaseToolsConfig, overrides: Mapping[str, Any] | None) -> BaseToolsConfig:
    allowed = set(getattr(BaseToolsConfig, "__dataclass_fields__", {}).keys())
    data = {k: v for k, v in dict(base.__dict__).items() if k in allowed}
    for key, value in (overrides or {}).items():
        if key in data:
            data[key] = value
    return BaseToolsConfig(**data)


def build_subagent_config(
    base_tools: BaseToolsConfig,
    *,
    model_name: str,
    model_provider: str,
    max_turns: int,
    system_prompt: str,
    overrides: Mapping[str, Any] | None = None,
) -> ChackConfig:
    from chack_agent import (
        AgentConfig,
        ChackConfig,
        CredentialsConfig,
        LoggingConfig,
        ModelConfig,
        SessionConfig,
        ToolsConfig as AgentToolsConfig,
    )

    def _resolve_alias(name: str, *, provider: str, fallback: str = "") -> str:
        raw = str(name or "").strip() or fallback
        if not raw:
            return ""
        try:
            from chack_agent.model_aliases import resolve_model_alias

            return resolve_model_alias(raw, provider=provider)
        except Exception:
            return raw

    overrides = dict(overrides or {})
    prompt = str(overrides.get("system_prompt") or system_prompt).strip() or system_prompt

    model_overrides = overrides.get("model") or {}
    provider = str(model_overrides.get("provider") or model_provider or "").strip()
    if not provider:
        raise ValueError("model_provider must be defined for sub-agent config")
    model_primary = _resolve_alias(
        str(model_overrides.get("primary") or model_name or "").strip(),
        provider=provider,
    )
    model = ModelConfig(
        primary=model_primary,
        provider=provider,
        max_context_tokens=int(model_overrides.get("max_context_tokens") or 0),
        social_network=_resolve_alias(
            str(model_overrides.get("social_network") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        scientific=_resolve_alias(
            str(model_overrides.get("scientific") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        websearcher=_resolve_alias(
            str(model_overrides.get("websearcher") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        tester=_resolve_alias(
            str(model_overrides.get("tester") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        social_network_max_turns=int(model_overrides.get("social_network_max_turns") or 30),
        scientific_max_turns=int(model_overrides.get("scientific_max_turns") or 30),
        websearcher_max_turns=int(model_overrides.get("websearcher_max_turns") or 30),
        tester_max_turns=int(model_overrides.get("tester_max_turns") or 30),
    )

    agent_overrides = overrides.get("agent") or {}
    agent = AgentConfig(
        self_critique_enabled=bool(agent_overrides.get("self_critique_enabled", False)),
        compaction_threshold_ratio=float(agent_overrides.get("compaction_threshold_ratio") or 0.75),
        compaction_model=str(agent_overrides.get("compaction_model") or ""),
    )

    session_overrides = overrides.get("session") or {}
    session = SessionConfig(
        max_turns=int(session_overrides.get("max_turns") or max_turns),
        long_term_memory_enabled=bool(
            session_overrides.get("long_term_memory_enabled", False)
        ),
        long_term_memory_max_chars=int(session_overrides.get("long_term_memory_max_chars") or 0),
        long_term_memory_dir=str(session_overrides.get("long_term_memory_dir") or ""),
        system_prompt="",
    )

    tools = _build_tools_config(base_tools, overrides.get("tools") or {})
    logging_overrides = overrides.get("logging") or {}
    logging = LoggingConfig(level=str(logging_overrides.get("level") or "INFO"))
    env = overrides.get("env") or {}

    return ChackConfig(
        model=model,
        agent=agent,
        session=session,
        tools=tools,
        credentials=CredentialsConfig(),
        logging=logging,
        system_prompt=prompt,
        env=env,
    )
