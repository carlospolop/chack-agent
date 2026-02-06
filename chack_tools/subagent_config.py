from __future__ import annotations

from typing import Any, Mapping

from .config import ToolsConfig as BaseToolsConfig


def _build_tools_config(base: BaseToolsConfig, overrides: Mapping[str, Any] | None) -> AgentToolsConfig:
    data = dict(base.__dict__)
    for key, value in (overrides or {}).items():
        if key in data:
            data[key] = value
    return AgentToolsConfig(**data)


def build_subagent_config(
    base_tools: BaseToolsConfig,
    *,
    model_name: str,
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
    overrides = dict(overrides or {})
    prompt = str(overrides.get("system_prompt") or system_prompt).strip() or system_prompt

    model_overrides = overrides.get("model") or {}
    model_primary = str(model_overrides.get("primary") or model_name or "").strip()
    model = ModelConfig(
        primary=model_primary,
        max_context_tokens=int(model_overrides.get("max_context_tokens") or 0),
        social_network=str(model_overrides.get("social_network") or ""),
        scientific=str(model_overrides.get("scientific") or ""),
        websearcher=str(model_overrides.get("websearcher") or ""),
        social_network_max_turns=int(model_overrides.get("social_network_max_turns") or 30),
        scientific_max_turns=int(model_overrides.get("scientific_max_turns") or 30),
        websearcher_max_turns=int(model_overrides.get("websearcher_max_turns") or 30),
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
