import json
import os
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


def main() -> None:
    provider = os.environ.get("INPUT_PROVIDER", "openai").strip() or "openai"
    if provider not in {"openai", "openrouter", "codex", "langgraph"}:
        raise SystemExit("provider must be 'openai', 'openrouter', 'codex' or 'langgraph'")

    openai_api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
        "INPUT_OPENAI_API_KEY", ""
    )
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get(
        "INPUT_OPENROUTER_API_KEY", ""
    )
    if provider == "openai" and not openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for provider=openai")
    if provider == "codex" and not openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for provider=codex")
    if provider == "langgraph" and not openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for provider=langgraph")
    if provider == "openrouter" and not openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for provider=openrouter")

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
    result = agent.run(
        session_id="github-action",
        text=prompt,
        require_task_steps_manager_init_first=bool(agent_cfg.require_task_steps_manager_init_first),
    )
    output = result.output or ""

    print(output)
    _write_github_output(output)


if __name__ == "__main__":
    main()
