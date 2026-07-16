import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .model_aliases import resolve_backend_alias, resolve_model_alias
from .thinking_effort import normalize_thinking_effort

from chack_tools.config import ToolsConfig as BaseToolsConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Codex and Claude Code run as CLI backends that manage their own context
# window. When they are used we guarantee a minimum context budget of 250k
# tokens: if the yaml configures a smaller (non-zero) window, bump it up.
CLI_BACKEND_MIN_CONTEXT_TOKENS = 250_000
_MIN_CONTEXT_TOKEN_PROVIDERS = frozenset(
    {"codex", "claude", "claude-code", "claude_code", "anthropic"}
)


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
    business: str = "CHEAP_BUT_QUALITY"
    product: str = "CHEAP_BUT_QUALITY"
    legal: str = "CHEAP_BUT_QUALITY"
    data_statistics: str = "CHEAP_BUT_QUALITY"
    news_media: str = "CHEAP_BUT_QUALITY"
    knowledge_graph: str = "CHEAP_BUT_QUALITY"
    religious: str = "CHEAP_BUT_QUALITY"
    cli: str = "CHEAP_BUT_QUALITY"
    subchack: str = ""
    researcher_administrator: str = ""
    social_network_thinking_effort: str = "high"
    scientific_thinking_effort: str = "high"
    websearcher_thinking_effort: str = "high"
    business_thinking_effort: str = "high"
    product_thinking_effort: str = "high"
    legal_thinking_effort: str = "high"
    data_statistics_thinking_effort: str = "high"
    news_media_thinking_effort: str = "high"
    knowledge_graph_thinking_effort: str = "high"
    religious_thinking_effort: str = "high"
    cli_thinking_effort: str = "high"
    subchack_thinking_effort: str = "high"
    researcher_administrator_thinking_effort: str = "high"
    researcher_queue_thinking_effort: str = "high"
    social_network_max_turns: int = 50
    scientific_max_turns: int = 50
    websearcher_max_turns: int = 50
    business_max_turns: int = 50
    product_max_turns: int = 50
    legal_max_turns: int = 50
    data_statistics_max_turns: int = 50
    news_media_max_turns: int = 50
    knowledge_graph_max_turns: int = 50
    religious_max_turns: int = 50
    cli_max_turns: int = 50
    subchack_max_turns: int = 100
    researcher_administrator_max_turns: int = 100


@dataclass
class AgentConfig:
    thinking_effort: str = "high"
    self_critique_enabled: bool = False
    self_critique_rounds: int = 0
    max_runtime_minutes: int = 0
    max_cost_usd: float = 0.0
    require_task_steps_manager_init_first: bool = True
    # Match Hermes's balanced context policy by default: keep full capacity,
    # but compact once the active context reaches 50%.
    compaction_threshold_ratio: float = 0.50
    compaction_target_ratio: float = 0.20
    compaction_model: str = ""
    main_action: str = ""
    sub_action: str = ""
    output_schema_json: Optional[Dict[str, Any]] = None
    output_schema_file: str = ""
    output_schema_name: str = ""
    output_schema_strict: bool = True
    budget_warning_ratio: float = 0.6
    budget_critical_ratio: float = 0.9
    budget_tool_injection_enabled: bool = True


@dataclass
class SessionConfig:
    max_turns: int = 50
    memory_max_messages: int = 50
    memory_reset_to_messages: int = 20
    long_term_memory_enabled: bool = True
    memory_summary_max_chars: int = 0
    long_term_memory_max_chars: int = 3000
    long_term_memory_dir: str = "longterm"
    # Zero disables a boundary. Idle time is measured from the previous run;
    # max age is measured from the first run in the native conversation.
    idle_reset_minutes: int = 0
    max_age_minutes: int = 0
    # Applications may summarize only on reset/rotation to avoid an extra model
    # call after every user message.
    long_term_memory_update_every_run: bool = True
    system_prompt: str = ""  # Optional override for this session


@dataclass
class ToolsConfig(BaseToolsConfig):
    missing_tools_reminders_max: int = 3
    required_tool_names: List[str] = field(default_factory=list)
    required_tool_call_attempts: int = 3


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
    codex_access_token: str = ""
    openai_admin_key: str = ""
    openai_org_id: str = ""
    openai_org_ids: List[str] = field(default_factory=list)
    anthropic_api_key: str = ""
    claude_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_name: str = ""
    openrouter_base_url: str = ""
    gemini_api_key: str = ""
    aws_profile: str = ""
    aws_credentials_file: str = ""
    copilot_github_token: str = ""


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


def resolve_api_key_type(config: ChackConfig) -> str:
    provider = str(getattr(config.model, "provider", "") or "").strip().lower()
    credentials = getattr(config, "credentials", CredentialsConfig())

    codex_access_token = (
        str(getattr(credentials, "codex_access_token", "") or "").strip()
        or os.environ.get("CODEX_ACCESS_TOKEN", "").strip()
    )
    openai_api_key = (
        str(getattr(credentials, "openai_api_key", "") or "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    anthropic_api_key = (
        str(getattr(credentials, "anthropic_api_key", "") or "").strip()
        or str(getattr(credentials, "claude_api_key", "") or "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("CLAUDE_API_KEY", "").strip()
    )
    claude_access_token = os.environ.get("CLAUDE_ACCESS_TOKEN", "").strip()
    openrouter_api_key = (
        str(getattr(credentials, "openrouter_api_key", "") or "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )

    if provider == "codex":
        if codex_access_token:
            return "codex_token"
        if openai_api_key:
            return "openai"
    if provider == "openai":
        if openai_api_key:
            return "openai"
    if provider in {"claude", "claude-code", "claude_code", "anthropic"}:
        if claude_access_token or anthropic_api_key:
            return "anthropic"
    if provider in {"copilot", "copilot-cli", "copilot_cli", "gh-copilot", "gh_copilot"}:
        copilot_token = (
            str(getattr(credentials, "copilot_github_token", "") or "").strip()
            or os.environ.get("COPILOT_GITHUB_TOKEN", "").strip()
            or os.environ.get("GH_TOKEN", "").strip()
            or os.environ.get("GITHUB_TOKEN", "").strip()
        )
        if copilot_token:
            return "copilot"
    if provider in {"openrouter", "langgraph"}:
        if openrouter_api_key:
            return "openrouter"

    if codex_access_token:
        return "codex_token"
    if openai_api_key:
        return "openai"
    if claude_access_token:
        return "anthropic"
    if anthropic_api_key:
        return "anthropic"
    if openrouter_api_key:
        return "openrouter"
    return "openai"


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
    if provider == "gemini":
        return "gemini"
    if provider in {"claude", "claude-code", "claude_code", "anthropic"}:
        return "claude"
    if provider in {"copilot", "copilot-cli", "copilot_cli", "gh-copilot", "gh_copilot"}:
        return "copilot"
    raise ValueError(f"Unsupported model.provider value: {provider!r}")


def _load_section(data: Dict[str, Any], key: str, cls):
    section = data.get(key, {})
    if section is None or not isinstance(section, dict):
        return cls()
    allowed = set(getattr(cls, "__dataclass_fields__", {}).keys())
    filtered = {k: v for k, v in section.items() if k in allowed}
    return cls(**filtered)


def resolve_config_aliases(config: ChackConfig) -> ChackConfig:
    credentials = getattr(config, "credentials", CredentialsConfig())
    model_cfg = config.model
    config.agent.thinking_effort = normalize_thinking_effort(
        getattr(config.agent, "thinking_effort", "high")
    )
    role_effort_fields = {
        "social_network": "social_network_agent",
        "scientific": "scientific_agent",
        "websearcher": "websearcher_agent",
        "business": "business_agent",
        "product": "product_agent",
        "legal": "legal_agent",
        "data_statistics": "data_statistics_agent",
        "news_media": "news_media_agent",
        "knowledge_graph": "knowledge_graph_agent",
        "religious": "religious_agent",
        "cli": "cli_agent",
        "subchack": "subchack_agent",
        "researcher_administrator": "researcher_administrator_agent",
        "researcher_queue": "researcher_queue_agent",
    }
    for role, tools_field in role_effort_fields.items():
        effort_field = f"{role}_thinking_effort"
        effort = normalize_thinking_effort(getattr(model_cfg, effort_field, "high"))
        setattr(model_cfg, effort_field, effort)
        role_settings = dict(getattr(config.tools, tools_field, {}) or {})
        role_settings.setdefault("thinking_effort", effort)
        setattr(config.tools, tools_field, role_settings)

    provider = resolve_backend_alias(
        model_cfg.provider,
        credentials=credentials,
    ).strip().lower()
    if not provider:
        raise ValueError("agent.provider could not be resolved from config")
    model_cfg.provider = provider
    # Enforce the minimum context window for the Codex / Claude Code backends.
    # Only bump an explicitly configured (non-zero) value that sits below the
    # floor; leaving 0 unchanged preserves "use the model's native window".
    if provider in _MIN_CONTEXT_TOKEN_PROVIDERS:
        configured_context = int(getattr(model_cfg, "max_context_tokens", 0) or 0)
        if 0 < configured_context < CLI_BACKEND_MIN_CONTEXT_TOKENS:
            model_cfg.max_context_tokens = CLI_BACKEND_MIN_CONTEXT_TOKENS
    model_cfg.primary = resolve_model_alias(
        model_cfg.primary,
        provider=provider,
        credentials=credentials,
    )
    if not str(model_cfg.social_network or "").strip():
        model_cfg.social_network = "CHEAP_BUT_QUALITY"
    if not str(model_cfg.scientific or "").strip():
        model_cfg.scientific = "CHEAP_BUT_QUALITY"
    if not str(model_cfg.websearcher or "").strip():
        model_cfg.websearcher = "CHEAP_BUT_QUALITY"
    if not str(model_cfg.cli or "").strip():
        model_cfg.cli = "CHEAP_BUT_QUALITY"
    model_cfg.social_network = resolve_model_alias(
        model_cfg.social_network,
        provider=provider,
        credentials=credentials,
    )
    model_cfg.scientific = resolve_model_alias(
        model_cfg.scientific,
        provider=provider,
        credentials=credentials,
    )
    model_cfg.websearcher = resolve_model_alias(
        model_cfg.websearcher,
        provider=provider,
        credentials=credentials,
    )
    model_cfg.cli = resolve_model_alias(
        model_cfg.cli,
        provider=provider,
        credentials=credentials,
    )
    if str(model_cfg.subchack or "").strip():
        model_cfg.subchack = resolve_model_alias(
            model_cfg.subchack,
            provider=provider,
            credentials=credentials,
        )
    if str(model_cfg.researcher_administrator or "").strip():
        model_cfg.researcher_administrator = resolve_model_alias(
            model_cfg.researcher_administrator,
            provider=provider,
            credentials=credentials,
        )
    return config


def load_config(path: str) -> ChackConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _interpolate_env(raw)

    if "system_prompt" not in raw or not str(raw.get("system_prompt", "")).strip():
        raise ValueError("system_prompt is required in config")
    if "agent" not in raw or not isinstance(raw.get("agent"), dict):
        raise ValueError("agent is required in config")
    agent_raw = raw.get("agent")

    if "model" in raw or "session" in raw:
        raise ValueError(
            "Legacy top-level config sections are no longer supported. "
            "Move model/session settings into the top-level agent: section."
        )
    if "model" in agent_raw or "session" in agent_raw:
        raise ValueError(
            "Legacy agent sub-sections are no longer supported. "
            "Use one flat agent section (model/session keys inside agent)."
        )
    if not str(agent_raw.get("primary", "")).strip():
        raise ValueError("agent.primary is required in config")
    if not str(agent_raw.get("provider", "")).strip():
        raise ValueError("agent.provider is required in config")

    model_fields = set(getattr(ModelConfig, "__dataclass_fields__", {}).keys())
    session_fields = set(getattr(SessionConfig, "__dataclass_fields__", {}).keys())
    agent_fields = set(getattr(AgentConfig, "__dataclass_fields__", {}).keys())

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

    model_payload = {k: v for k, v in agent_raw.items() if k in model_fields}
    session_payload = {k: v for k, v in agent_raw.items() if k in session_fields}
    agent_payload = {k: v for k, v in agent_raw.items() if k in agent_fields}

    model_cfg = _load_section({"model": model_payload}, "model", ModelConfig)
    session = _load_section({"session": session_payload}, "session", SessionConfig)
    if session.system_prompt:
        session.system_prompt = _inject_tools(session.system_prompt)

    agent = _load_section({"agent": agent_payload}, "agent", AgentConfig)
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

    return resolve_config_aliases(config)
