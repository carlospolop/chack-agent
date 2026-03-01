from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Mapping

from .config import ToolsConfig as BaseToolsConfig


def _build_tools_config(base: BaseToolsConfig, overrides: Mapping[str, Any] | None) -> BaseToolsConfig:
    allowed = set(getattr(BaseToolsConfig, "__dataclass_fields__", {}).keys())
    data = {k: v for k, v in dict(base.__dict__).items() if k in allowed}
    for key, value in (overrides or {}).items():
        if key in data:
            data[key] = value
    return BaseToolsConfig(**data)


def _scaled_limit_int(value: float, ratio: float, minimum: int) -> int:
    raw = max(0.0, float(value or 0.0))
    if raw <= 0.0:
        return 0
    return max(minimum, int(raw * ratio))


def _scaled_limit_float(value: float, ratio: float) -> float:
    raw = max(0.0, float(value or 0.0))
    if raw <= 0.0:
        return 0.0
    return raw * ratio


def inherit_subagent_limits(
    *,
    default_max_turns: int,
    parent_max_turns: int,
    parent_remaining_runtime_minutes: float,
    parent_remaining_cost_usd: float,
) -> tuple[int, int, float]:
    # Child turns cap: 1/2 of parent max turns.
    parent_turns_cap = _scaled_limit_int(parent_max_turns, 0.5, minimum=2)
    effective_max_turns = max(2, int(default_max_turns or 30))
    if parent_turns_cap > 0:
        effective_max_turns = min(effective_max_turns, parent_turns_cap)

    # Child runtime/cost cap: 2/3 of parent's remaining runtime/cost.
    effective_runtime_minutes = _scaled_limit_int(
        parent_remaining_runtime_minutes,
        2.0 / 3.0,
        minimum=1,
    )
    effective_cost_usd = _scaled_limit_float(parent_remaining_cost_usd, 2.0 / 3.0)
    return effective_max_turns, effective_runtime_minutes, effective_cost_usd


def subagent_launch_block_reason(
    *,
    parent_original_runtime_minutes: int,
    parent_remaining_runtime_minutes: float,
    parent_original_cost_usd: float,
    parent_remaining_cost_usd: float,
) -> str | None:
    runtime_limited = max(0, int(parent_original_runtime_minutes or 0)) > 0
    cost_limited = max(0.0, float(parent_original_cost_usd or 0.0)) > 0.0

    if runtime_limited:
        original_runtime = max(0, int(parent_original_runtime_minutes or 0))
        remaining_runtime = max(0.0, float(parent_remaining_runtime_minutes or 0.0))
        runtime_floor = max(10.0, float(original_runtime) / 3.0)
        if remaining_runtime < runtime_floor:
            return (
                "ERROR: cannot launch delegated agent. "
                f"Parent remaining runtime is too low ({remaining_runtime:.2f} min, "
                f"requires at least {runtime_floor:.2f} min) to launch tools that run autonomous agents."
            )

    if cost_limited:
        original_cost = max(0.0, float(parent_original_cost_usd or 0.0))
        remaining_cost = max(0.0, float(parent_remaining_cost_usd or 0.0))
        cost_floor = max(1.0, original_cost / 3.0)
        if remaining_cost < cost_floor:
            return (
                "ERROR: cannot launch delegated agent. "
                f"Parent remaining budget is too low (${remaining_cost:.4f}, "
                f"requires at least ${cost_floor:.4f}) to launch tools that run autonomous agents."
            )
    return None


def validate_subagent_instruction_length(prompt: str, *, min_chars: int = 500) -> str | None:
    text = str(prompt or "").strip()
    if not text:
        return "ERROR: prompt cannot be empty"
    if len(text) < int(min_chars):
        return (
            "ERROR: delegated sub-agent launch blocked. "
            f"Provide at least {int(min_chars)} characters of detailed instructions "
            f"(received {len(text)})."
            f"Use the extra chars to indicate more details on the goals of the agents, expected example responses/information, or any other relevant data. The more specific you are, the better."
        )
    return None


def normalize_subagent_prompts(
    prompt_input: Any,
    *,
    min_chars: int = 500,
    max_prompts: int = 3,
) -> tuple[List[str], str | None]:
    prompts: List[str]
    if isinstance(prompt_input, list):
        prompts = [str(item or "").strip() for item in prompt_input]
    else:
        prompts = [str(prompt_input or "").strip()]

    prompts = [item for item in prompts if item]
    if not prompts:
        return [], "ERROR: prompt cannot be empty"
    if len(prompts) > int(max_prompts):
        return [], (
            "ERROR: delegated sub-agent launch blocked. "
            f"You can provide at most {int(max_prompts)} prompts."
        )

    for idx, text in enumerate(prompts, start=1):
        guard = validate_subagent_instruction_length(text, min_chars=min_chars)
        if guard:
            return [], f"{guard} (prompt #{idx})"
    return prompts, None


def run_parallel_subagent_prompts(
    prompts: List[str],
    runner: Callable[[str], str],
) -> str:
    if len(prompts) == 1:
        return runner(prompts[0])

    results: dict[int, str] = {}
    max_workers = min(3, len(prompts))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(runner, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = str(future.result() or "").strip() or "ERROR: empty sub-agent output."
            except Exception as exc:
                results[idx] = f"ERROR: sub-agent batch worker failed ({exc})"

    chunks: List[str] = []
    for idx in range(len(prompts)):
        output = results.get(idx, "ERROR: missing sub-agent output.")
        chunks.append(f"SUBAGENT_RESULT_{idx + 1}:\n{output}")
    return "\n\n".join(chunks)


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
        subchack=_resolve_alias(
            str(model_overrides.get("subchack") or ""),
            provider=provider,
            fallback="",
        ),
        social_network_max_turns=int(model_overrides.get("social_network_max_turns") or 30),
        scientific_max_turns=int(model_overrides.get("scientific_max_turns") or 30),
        websearcher_max_turns=int(model_overrides.get("websearcher_max_turns") or 30),
        tester_max_turns=int(model_overrides.get("tester_max_turns") or 30),
        subchack_max_turns=int(model_overrides.get("subchack_max_turns") or 30),
    )

    agent_overrides = overrides.get("agent") or {}
    agent = AgentConfig(
        self_critique_enabled=bool(agent_overrides.get("self_critique_enabled", False)),
        max_runtime_minutes=int(agent_overrides.get("max_runtime_minutes") or 0),
        max_cost_usd=float(agent_overrides.get("max_cost_usd") or 0.0),
        compaction_threshold_ratio=float(agent_overrides.get("compaction_threshold_ratio") or 0.75),
        compaction_model=str(agent_overrides.get("compaction_model") or ""),
        main_action=str(agent_overrides.get("main_action") or ""),
        sub_action=str(agent_overrides.get("sub_action") or ""),
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
