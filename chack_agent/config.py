import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .model_aliases import resolve_model_alias

from chack_tools.config import ToolsConfig as BaseToolsConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _interpolate_env(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var = match.group(1)
            return os.environ.get(var, "")
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    if isinstance(value, dict):
        return {
            k: (v if k == "exec_timeout_seconds" else _interpolate_env(v))
            for k, v in value.items()
        }
    return value


@dataclass
class ModelConfig:
    primary: str
    provider: str = ""
    max_context_tokens: int = 0
    social_network: str = "CHEAP_BUT_QUALITY"
    scientific: str = "CHEAP_BUT_QUALITY"
    websearcher: str = "CHEAP_BUT_QUALITY"
    tester: str = "CHEAP_BUT_QUALITY"
    social_network_max_turns: int = 30
    scientific_max_turns: int = 30
    websearcher_max_turns: int = 30
    tester_max_turns: int = 30


@dataclass
class AgentConfig:
    self_critique_enabled: bool = True
    require_task_steps_manager_init_first: bool = True
    compaction_threshold_ratio: float = 0.75
    compaction_model: str = ""
    main_action: str = ""
    sub_action: str = ""
    output_schema_json: Optional[Dict[str, Any]] = None
    output_schema_file: str = ""
    output_schema_name: str = ""
    output_schema_strict: bool = True


@dataclass
class SessionConfig:
    max_turns: int = 50
    memory_max_messages: int = 50
    memory_reset_to_messages: int = 20
    long_term_memory_enabled: bool = True
    long_term_memory_max_chars: int = 3000
    long_term_memory_dir: str = "longterm"
    system_prompt: str = ""  # Optional override for this session


@dataclass
class ToolsConfig(BaseToolsConfig):
    missing_tools_reminders_max: int = 3


@dataclass
class CredentialsConfig:
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = ""
    aws_profiles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    stripe_api_key: str = ""
    gcp_credentials_path: str = ""
    gcp_quota_project: str = ""
    azure_app_id: str = ""
    azure_sa_name: str = ""
    azure_sa_secret_value: str = ""
    azure_tenant_id: str = ""
    gh_token: str = ""
    openai_api_key: str = ""
    openai_admin_key: str = ""
    openai_org_id: str = ""
    openai_org_ids: List[str] = field(default_factory=list)
    openrouter_api_key: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_name: str = ""
    openrouter_base_url: str = ""
    aws_profile: str = ""
    aws_credentials_file: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class ChackConfig:
    model: ModelConfig
    agent: AgentConfig
    session: SessionConfig
    tools: ToolsConfig
    credentials: CredentialsConfig
    logging: LoggingConfig
    system_prompt: str
    env: Dict[str, str]
    user_prompt: str = ""
    user_prompt_variables: Dict[str, Any] = field(default_factory=dict)


def resolve_backend_type(config: ChackConfig) -> str:
    provider = str(getattr(config.model, "provider", "") or "").strip().lower()
    if not provider:
        raise ValueError("model.provider must be defined in config")
    if provider == "langgraph":
        return "langgraph"
    if provider == "openrouter":
        return "openrouter"
    if provider == "codex":
        return "codex"
    if provider == "openai":
        return "openai_compaction"
    raise ValueError(f"Unsupported model.provider value: {provider!r}")


def _load_section(data: Dict[str, Any], key: str, cls):
    section = data.get(key, {})
    if section is None or not isinstance(section, dict):
        return cls()
    allowed = set(getattr(cls, "__dataclass_fields__", {}).keys())
    filtered = {k: v for k, v in section.items() if k in allowed}
    return cls(**filtered)


def _extract_session_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("session", "runtime", "telegram", "discord"):
        section = raw.get(key)
        if isinstance(section, dict):
            return section
    return {}


def load_config(path: str) -> ChackConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _interpolate_env(raw)

    if "system_prompt" not in raw or not str(raw.get("system_prompt", "")).strip():
        raise ValueError("system_prompt is required in config")
    if "model" not in raw or not isinstance(raw.get("model"), dict):
        raise ValueError("model.primary is required in config")
    if not str(raw.get("model", {}).get("primary", "")).strip():
        raise ValueError("model.primary is required in config")
    if not str(raw.get("model", {}).get("provider", "")).strip():
        raise ValueError("model.provider is required in config")

    base_dir = os.path.dirname(os.path.abspath(path))
    if "tools_prompt_file" in raw:
        tools_prompt_file = str(raw.get("tools_prompt_file") or "TOOLS.md").strip()
    elif isinstance(raw.get("telegram"), dict):
        tools_prompt_file = "TOOLS_TELEGRAM.md"
    elif isinstance(raw.get("discord"), dict):
        tools_prompt_file = "TOOLS_DISCORD.md"
    else:
        tools_prompt_file = "TOOLS.md"

    def _get_tools_text(filename: str) -> str:
        tools_path = filename
        if not os.path.isabs(tools_path):
            tools_path = os.path.join(base_dir, tools_path)
        if not os.path.exists(tools_path):
            raise ValueError(
                f"{filename} is required when using $$TOOLS$$ in prompts (missing at {tools_path})"
            )
        with open(tools_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()

    def _inject_tools(prompt_text: str) -> str:
        if "$$TOOLS$$" not in prompt_text:
            return prompt_text
        return prompt_text.replace("$$TOOLS$$", _get_tools_text(tools_prompt_file))

    system_prompt_template = str(raw.get("system_prompt")).strip()
    system_prompt = _inject_tools(system_prompt_template)
    user_prompt_template = str(raw.get("user_prompt", "") or "").strip()
    user_prompt = _inject_tools(user_prompt_template) if user_prompt_template else ""

    credentials = _load_section(raw, "credentials", CredentialsConfig)
    if isinstance(credentials.aws_profiles, str) and credentials.aws_profiles.strip():
        try:
            parsed_profiles = yaml.safe_load(credentials.aws_profiles) or {}
            if isinstance(parsed_profiles, dict):
                credentials.aws_profiles = parsed_profiles
        except yaml.YAMLError:
            credentials.aws_profiles = {}
    if isinstance(credentials.openai_org_ids, str):
        credentials.openai_org_ids = [
            item.strip() for item in credentials.openai_org_ids.split(",") if item.strip()
        ]

    session_raw = _extract_session_section(raw)
    session = _load_section({"session": session_raw}, "session", SessionConfig)
    if session.system_prompt:
        session.system_prompt = _inject_tools(session.system_prompt)

    agent = _load_section(raw, "agent", AgentConfig)
    # self_critique_prompt is hardcoded in chack_agent.agent
    if not str(agent.main_action or "").strip():
        raise ValueError("agent.main_action is required in config")
    if not str(agent.sub_action or "").strip():
        raise ValueError("agent.sub_action is required in config")
    if agent.output_schema_file and not agent.output_schema_json:
        schema_path = agent.output_schema_file
        if not os.path.isabs(schema_path):
            schema_path = os.path.join(base_dir, schema_path)
        if not os.path.exists(schema_path):
            raise ValueError(f"agent.output_schema_file not found: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as handle:
            agent.output_schema_json = yaml.safe_load(handle) or {}

    model_cfg = _load_section(raw, "model", ModelConfig)
    provider = str(model_cfg.provider or "").strip().lower()
    model_cfg.primary = resolve_model_alias(
        model_cfg.primary,
        provider=provider,
    )
    if not str(model_cfg.social_network or "").strip():
        model_cfg.social_network = "CHEAP_BUT_QUALITY"
    if not str(model_cfg.scientific or "").strip():
        model_cfg.scientific = "CHEAP_BUT_QUALITY"
    if not str(model_cfg.websearcher or "").strip():
        model_cfg.websearcher = "CHEAP_BUT_QUALITY"
    if not str(model_cfg.tester or "").strip():
        model_cfg.tester = "CHEAP_BUT_QUALITY"
    model_cfg.social_network = resolve_model_alias(
        model_cfg.social_network,
        provider=provider,
    )
    model_cfg.scientific = resolve_model_alias(
        model_cfg.scientific,
        provider=provider,
    )
    model_cfg.websearcher = resolve_model_alias(
        model_cfg.websearcher,
        provider=provider,
    )
    model_cfg.tester = resolve_model_alias(
        model_cfg.tester,
        provider=provider,
    )

    config = ChackConfig(
        model=model_cfg,
        agent=agent,
        session=session,
        tools=_load_section(raw, "tools", ToolsConfig),
        credentials=credentials,
        logging=_load_section(raw, "logging", LoggingConfig),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        user_prompt_variables=raw.get("user_prompt_variables", {}) or {},
        env=raw.get("env", {}) or {},
    )

    return config
