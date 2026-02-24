import json
import os
import re
import time
from pathlib import Path

from chack_agent import (
    AgentConfig,
    Chack,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)
from chack_agent.model_aliases import resolve_model_alias


def _load_json(name: str) -> dict:
    raw = os.environ.get(name, "{}").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {name}: {exc}")


def _resolve_prompt() -> str:
    prompt_file = os.environ.get("INPUT_PROMPT_FILE", "").strip()
    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = Path(os.getcwd()) / prompt_path
        if not prompt_path.exists():
            raise SystemExit(f"prompt_file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    user_prompt = os.environ.get("INPUT_USER_PROMPT", "").strip()
    if not user_prompt:
        raise SystemExit("user_prompt is required when prompt_file is not provided")
    return user_prompt


def _load_output_schema() -> tuple[dict, str, bool] | tuple[None, str, bool]:
    schema_file = os.environ.get("INPUT_OUTPUT_SCHEMA_FILE", "").strip()
    schema_json_raw = os.environ.get("INPUT_OUTPUT_SCHEMA_JSON", "").strip()
    schema_name = os.environ.get("INPUT_OUTPUT_SCHEMA_NAME", "").strip() or "output_schema"
    strict_raw = os.environ.get("INPUT_OUTPUT_SCHEMA_STRICT", "").strip().lower()
    strict = strict_raw not in {"0", "false", "no"}

    schema = None
    if schema_file:
        path = Path(schema_file)
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
        if not path.exists():
            raise SystemExit(f"output_schema_file not found: {path}")
        schema = json.loads(path.read_text(encoding="utf-8"))
    elif schema_json_raw:
        try:
            schema = json.loads(schema_json_raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in output_schema_json: {exc}")

    return schema, schema_name, strict


def _write_github_output(message: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write("final-message<<EOF\n")
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")
        handle.write("EOF\n")


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._:-]+", "-", str(value or "").strip())
    token = token.strip("-")
    return token or "na"


def _resolve_session_id(main_action: str, sub_action: str) -> str:
    explicit = os.environ.get("INPUT_SESSION_ID", "").strip()
    if explicit:
        return explicit

    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    if run_id:
        run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "").strip() or "1"
        job = _safe_token(os.environ.get("GITHUB_JOB", "job"))
        repo = _safe_token(os.environ.get("GITHUB_REPOSITORY", "repo"))
        return (
            f"gha:{repo}:{_safe_token(run_id)}:{_safe_token(run_attempt)}:"
            f"{job}:{_safe_token(main_action)}:{_safe_token(sub_action)}"
        )

    return f"github-action:{int(time.time() * 1000)}"


def main() -> None:
    provider = os.environ.get("INPUT_PROVIDER", "openai").strip() or "openai"
    if provider not in {"openai", "openrouter", "codex", "langgraph", "gemini", "claude", "claude-code", "claude_code"}:
        raise SystemExit(
            "provider must be 'openai', 'openrouter', 'codex', 'gemini', 'langgraph', or 'claude'"
        )

    openai_api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
        "INPUT_OPENAI_API_KEY", ""
    )
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get(
        "INPUT_OPENROUTER_API_KEY", ""
    )
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get(
        "INPUT_GEMINI_API_KEY", ""
    )
    google_api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get(
        "INPUT_GOOGLE_API_KEY", ""
    )
    anthropic_api_key = (
        os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("CLAUDE_API_KEY", "")
        or os.environ.get("INPUT_ANTHROPIC_API_KEY", "")
        or os.environ.get("INPUT_CLAUDE_API_KEY", "")
    )
    if provider == "openai" and not openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for provider=openai")
    if provider == "codex" and not openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for provider=codex")
    if provider == "langgraph" and not openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for provider=langgraph")
    if provider == "openrouter" and not openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for provider=openrouter")
    if provider == "gemini" and not gemini_api_key and not google_api_key:
        raise SystemExit(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required for provider=gemini"
        )
    if provider in {"claude", "claude-code", "claude_code"} and not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("CLAUDE_API_KEY"):
        if os.environ.get("INPUT_ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = os.environ.get("INPUT_ANTHROPIC_API_KEY", "")
        elif os.environ.get("INPUT_CLAUDE_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = os.environ.get("INPUT_CLAUDE_API_KEY", "")

    if provider in {"claude", "claude-code", "claude_code"} and not anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY or CLAUDE_API_KEY is required for provider=claude"
        )

    tools_overrides = _load_json("INPUT_TOOLS_CONFIG_JSON")
    session_overrides = _load_json("INPUT_SESSION_CONFIG_JSON")
    agent_overrides = _load_json("INPUT_AGENT_CONFIG_JSON")
    if (
        "require_task_list_init_first" in agent_overrides
        and "require_task_steps_manager_init_first" not in agent_overrides
    ):
        agent_overrides["require_task_steps_manager_init_first"] = agent_overrides.pop(
            "require_task_list_init_first"
        )
    agent_runtime_raw = os.environ.get("INPUT_AGENT_MAX_RUNTIME_MINUTES", "").strip()
    if agent_runtime_raw:
        try:
            agent_overrides["max_runtime_minutes"] = int(agent_runtime_raw)
        except ValueError:
            raise SystemExit("agent_max_runtime_minutes must be an integer")

    output_schema, output_schema_name, output_schema_strict = _load_output_schema()
    if output_schema is not None:
        agent_overrides.setdefault("output_schema_json", output_schema)
        agent_overrides.setdefault("output_schema_name", output_schema_name)
        agent_overrides.setdefault("output_schema_strict", output_schema_strict)

    max_turns_raw = os.environ.get("INPUT_MAX_TURNS", "0").strip()
    try:
        max_turns = int(max_turns_raw) if max_turns_raw else 0
    except ValueError:
        raise SystemExit("max_turns must be an integer")

    model = ModelConfig(
        primary=resolve_model_alias(
            os.environ.get("INPUT_MODEL_PRIMARY", "gpt-4o"),
            provider=provider,
        ),
        provider=provider,
        social_network=resolve_model_alias(
            os.environ.get("INPUT_MODEL_SOCIAL", "CHEAP_BUT_QUALITY"),
            provider=provider,
        ),
        scientific=resolve_model_alias(
            os.environ.get("INPUT_MODEL_SCIENTIFIC", "CHEAP_BUT_QUALITY"),
            provider=provider,
        ),
        websearcher=resolve_model_alias(
            os.environ.get("INPUT_MODEL_WEBSEARCHER", "CHEAP_BUT_QUALITY"),
            provider=provider,
        ),
        tester=resolve_model_alias(
            os.environ.get("INPUT_MODEL_TESTER", "CHEAP_BUT_QUALITY"),
            provider=provider,
        ),
    )

    agent_cfg = AgentConfig(
        main_action=os.environ.get("INPUT_MAIN_ACTION", "github_action"),
        sub_action=os.environ.get("INPUT_SUB_ACTION", "run"),
        **agent_overrides,
    )

    session_cfg = SessionConfig(
        **session_overrides,
    )
    if max_turns > 0:
        session_cfg.max_turns = max_turns

    tools_cfg = ToolsConfig(
        **tools_overrides,
    )

    config = ChackConfig(
        model=model,
        agent=agent_cfg,
        session=session_cfg,
        tools=tools_cfg,
        credentials=CredentialsConfig(
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            openrouter_http_referer=os.environ.get("OPENROUTER_HTTP_REFERER", "")
            or os.environ.get("INPUT_OPENROUTER_HTTP_REFERER", ""),
            openrouter_app_name=os.environ.get("OPENROUTER_APP_NAME", "")
            or os.environ.get("INPUT_OPENROUTER_APP_NAME", ""),
            openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "")
            or os.environ.get("INPUT_OPENROUTER_BASE_URL", ""),
            gemini_api_key=gemini_api_key or google_api_key,
        ),
        logging=LoggingConfig(level=os.environ.get("INPUT_LOGGING_LEVEL", "INFO")),
        system_prompt=os.environ.get(
            "INPUT_SYSTEM_PROMPT",
            "You are an advanced research agent.",
        ),
        env={},
    )

    prompt = _resolve_prompt()
    agent = Chack(config)
    session_id = _resolve_session_id(agent_cfg.main_action, agent_cfg.sub_action)
    result = agent.run(
        session_id=session_id,
        text=prompt,
        require_task_steps_manager_init_first=bool(agent_cfg.require_task_steps_manager_init_first),
    )
    output = result.output or ""

    print(output)
    _write_github_output(output)


if __name__ == "__main__":
    main()
