from __future__ import annotations

import json
import logging
import os
import inspect
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from chack_tools.telemetry import OTEL_CHILD_ENV_NAMES, child_process_env
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_steps_manager_state import (
    STORE as TASK_STEPS_STORE,
    current_run_label,
    current_session_id,
)
from chack_tools.native_planning import native_planning_prompt, sync_claude_native_task
from chack_tools.telemetry import log_event
from chack_tools.cancellation import cancellation_requested, register_process, unregister_process
from chack_tools.tool_usage_state import effective_max_tools_used

from ..config import ChackConfig
from ..live_cost_state import report_live_usage
from ..resume_compaction import ResumeCompactionResult
from ..llm_gateway import (
    GatewayRoute,
    clone_config_for_gateway,
    gateway_subprocess_env,
    get_gateway_route,
    openrouter_selected_by_api_key_type,
    route_model_name,
)
from ..model_aliases import ANTHROPIC_BEST_CHEAPEST
from ..output_schema import JsonSchemaOutput
from ..thinking_effort import claude_thinking_effort, normalize_thinking_effort
from ..environment_credential_pool import rotate_environment_credentials
from ..provider_launch_hooks import run_provider_pre_launch_hook
from .cli_process import cli_exec_timeout_seconds
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .web_search_fallback import (
    FALLBACK_WEB_MCP_SERVER,
    fallback_web_mcp_url,
    keyless_web_search_enabled,
    web_search_enabled,
)
from .tool_payloads import (
    configured_mcp_env_names,
    CHACK_ALLOWED_TOOLS_JSON_PATH_ENV,
    CHACK_INLINE_ENV_VALUE_MAX_CHARS,
    CHACK_TOOLS_APPEND_B64_PATH_ENV,
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_CONFIG_JSON_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_B64_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV,
    CHACK_TOOLS_APPEND_NAMES_JSON_ENV,
    augment_subprocess_pythonpath,
    serialize_tools_payload,
    write_payload_to_file,
)
from .prompt_cache import (
    prompt_cache_key,
    provider_rejects_prompt_cache,
    split_prompt_cache_breakpoint,
)


_LOGGER = logging.getLogger("chack.claude_code_backend")

CLAUDE_EXEC_TIMEOUT_ENV = "CHACK_CLAUDE_EXEC_TIMEOUT_SECONDS"
_MCP_STARTUP_STATUS_PATH_ENV = "CHACK_MCP_STARTUP_STATUS_PATH"


def _claude_cli_json_schema(schema_json: str) -> str:
    """Remove dialect declarations that Claude CLI cannot resolve.

    Claude Code validates ``--json-schema`` with its bundled validator. Some
    releases do not register the draft 2020-12 metaschema and reject an
    otherwise supported schema solely because it contains ``$schema``. The
    declaration does not define an output constraint, so remove it while
    preserving every validation keyword. Keep invalid/non-object input intact
    so the CLI still reports the original configuration error.
    """

    raw_schema = str(schema_json or "")
    try:
        schema = json.loads(raw_schema)
    except (TypeError, json.JSONDecodeError):
        return raw_schema
    if not isinstance(schema, dict):
        return raw_schema

    changed = False

    def strip_dialect(value: Any) -> Any:
        nonlocal changed
        if isinstance(value, dict):
            normalized = {}
            for key, item in value.items():
                if key == "$schema":
                    changed = True
                    continue
                normalized[key] = strip_dialect(item)
            return normalized
        if isinstance(value, list):
            return [strip_dialect(item) for item in value]
        return value

    normalized = strip_dialect(schema)
    if not changed:
        return raw_schema
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _normalized_claude_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Map Claude usage into Chack's total/read/write token convention.

    Anthropic reports ordinary input, cache creation, and cache reads as
    disjoint counters. Chack's ``prompt_tokens`` is the total effective input,
    with read/write subsets carried in ``input_tokens_details`` for pricing.
    Claude Code versions have emitted a few aliases, so accept all observed
    wire names without double-counting the nested cache-creation breakdown.
    """

    ordinary = int(usage.get("input_tokens", 0) or 0)
    output = int(usage.get("output_tokens", 0) or 0)
    cache_read = int(
        usage.get(
            "cache_read_input_tokens",
            usage.get("cached_input_tokens", usage.get("cached_tokens", 0)),
        )
        or 0
    )
    raw_cache_write = usage.get(
        "cache_creation_input_tokens",
        usage.get("cache_write_input_tokens"),
    )
    if raw_cache_write is None:
        creation = usage.get("cache_creation")
        if isinstance(creation, dict):
            raw_cache_write = sum(
                int(value or 0)
                for key, value in creation.items()
                if str(key).endswith("_input_tokens")
            )
    cache_write = int(raw_cache_write or 0)
    return {
        "input_tokens": ordinary + cache_read + cache_write,
        "output_tokens": output,
        "input_tokens_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
    }


def _seconds_until_claude_quota_reset(
    output: str,
    *,
    now: datetime | None = None,
) -> int | None:
    """Return the wait until a near-term UTC reset advertised by Claude CLI."""
    text = str(output or "").strip()
    lowered = text.lower()
    if not text.startswith("ERROR:") or not any(
        marker in lowered
        for marker in ("session limit", "usage limit", "hit your limit", "quota exceeded")
    ):
        return None

    match = re.search(
        r"\bresets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*(?:\((?:utc|gmt)\)|(?:utc|gmt))",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    hour = int(match.group(1)) % 12
    if match.group(3).lower() == "pm":
        hour += 12
    minute = int(match.group(2) or "0")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    reset = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= current:
        reset += timedelta(days=1)
    return max(1, int((reset - current).total_seconds()) + 1)


def _descendant_pids(pid: int) -> list[int]:
    found: list[int] = []
    stack = [int(pid)]
    while stack:
        current = stack.pop()
        try:
            proc = subprocess.run(
                ["pgrep", "-P", str(current)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            continue
        children = []
        for raw in str(proc.stdout or "").splitlines():
            try:
                child = int(raw.strip())
            except Exception:
                continue
            children.append(child)
        found.extend(children)
        stack.extend(children)
    return found


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(process.pid)), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        return
    pids = list(reversed(_descendant_pids(int(process.pid)))) + [int(process.pid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
            except Exception:
                try:
                    if pid == process.pid:
                        process.kill()
                except Exception:
                    pass
        time.sleep(1 if sig == signal.SIGTERM else 0)
        if process.poll() is not None:
            break


# Claude models cap at a 200k-token context window by default. Opus 4.6 and
# Sonnet 4.6 can extend to 1M tokens, but only when the request carries the
# `context-1m-2025-08-07` beta header, which the CLI exposes via `--betas`.
# So whenever a larger context budget is configured we opt into that beta.
_CLAUDE_DEFAULT_CONTEXT_WINDOW = 200_000
_CLAUDE_1M_CONTEXT_BETA = "context-1m-2025-08-07"
# Built-in tools a Chack run keeps; the shell, web and planning tools are added
# when the run's configuration allows them. ToolSearch loads deferred MCP tool
# schemas, so Chack's own tools stay reachable when there are many.
_CLAUDE_BUILTIN_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep", "ToolSearch")
_CLAUDE_PLANNING_TOOLS = ("TodoWrite", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet")
# Prompt parts a headless run never uses: the auto-memory instructions, the git
# status snapshot and the commit/PR attribution reminder.
_CLAUDE_HEADLESS_SETTINGS = json.dumps(
    {
        "autoMemoryEnabled": False,
        "includeGitInstructions": False,
        "attribution": {"commit": "", "pr": ""},
    },
    separators=(",", ":"),
)
# Model-family defaults Claude Code falls back to for subagents and background
# work. On a gateway they must name models the gateway serves.
_CLAUDE_GATEWAY_MODEL_ENV = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)
# The model Claude Code itself retries on when a model's safeguards refuse a
# request. It sends this vendor ID as is, which a gateway does not serve, so on
# a gateway its fallback is off and the run is retried here under the
# gateway's name for the same model.
_CLAUDE_REFUSAL_FALLBACK_MODEL = "claude-opus-4-8"


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


def _malformed_tool_use_names(message: dict[str, Any]) -> list[str]:
    """Names of tool calls in an assistant message that are not tool names.

    A provider that fails to parse a model's tool-call tokens can leak them
    into the call, e.g. ``functions_mcp__chack_tools__exec_2
    <|tool_call_argument_begin|> {"command"`` with empty input (Kimi K2.5 on
    Bedrock behind LiteLLM).
    """
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        str(item.get("name") or "")
        for item in content
        if isinstance(item, dict)
        and str(item.get("type") or "") == "tool_use"
        and not _TOOL_NAME_RE.match(str(item.get("name") or ""))
    ]


# (Anthropic base URL, model) pairs whose provider rejected cache markers in
# this process, so later runs of the same model skip the failing request.
_PROMPT_CACHE_REJECTED: set[tuple[str, str]] = set()


def _claude_gateway_env(config: ChackConfig, route: GatewayRoute) -> dict[str, str]:
    """Claude Code environment for a gateway serving the Anthropic Messages API."""

    env = gateway_subprocess_env(config, route)
    env.update(
        {
            "ANTHROPIC_BASE_URL": route.anthropic_base_url,
            # A bearer token, not an Anthropic key; the key variable stays
            # empty so Claude Code does not prefer it over the gateway token.
            "ANTHROPIC_AUTH_TOKEN": route.api_key,
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            # On a refusal or a model it cannot reach, Claude Code retries on a
            # hardcoded Anthropic model id that bypasses the model variables
            # below, and the gateway rejects it as an unknown model. Refusals
            # are retried by the executor on the gateway's own name instead.
            "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
            "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
        }
    )
    # Betas such as deferred tool loading save tens of thousands of tokens per
    # call, so they stay on unless the model is served by Bedrock, which rejects
    # Anthropic beta headers it does not know.
    if route.upstream.lower().startswith("bedrock"):
        env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
    if route.headers:
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(
            f"{name}: {value}" for name, value in sorted(route.headers.items())
        )
    # Every model family resolves to the selected model unless the gateway also
    # serves a cheaper model for background work.
    background_model = route_model_name(config, route, ANTHROPIC_BEST_CHEAPEST) or route.model_name
    for name in _CLAUDE_GATEWAY_MODEL_ENV:
        env[name] = route.model_name
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = background_model
    env["ANTHROPIC_SMALL_FAST_MODEL"] = background_model
    return env


def _is_model_refusal(output: str) -> bool:
    """Whether a run ended because the model's safeguards refused it."""

    return str(output or "").startswith("ERROR:") and '"stop_reason": "refusal"' in str(output)


def _gateway_refusal_fallback(config: ChackConfig, route: GatewayRoute) -> tuple[str, str]:
    """Gateway name and Claude Code environment for the refusal fallback model, if served."""

    try:
        fallback = get_gateway_route(config, model_name=_CLAUDE_REFUSAL_FALLBACK_MODEL)
    except ValueError:
        return "", ""
    if fallback is None or not fallback.model_name or fallback.model_name == route.model_name:
        return "", ""
    return fallback.model_name, json.dumps(_claude_gateway_env(config, fallback), sort_keys=True)


@lru_cache(maxsize=8)
def _claude_supported_effort_levels(cli_path: str) -> frozenset[str]:
    """Discover effort levels exposed by this Claude Code installation."""
    fallback = frozenset({"low", "medium", "high", "max"})
    try:
        result = subprocess.run(
            [cli_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return fallback

    output = str(result.stdout or "")
    lines = output.splitlines()
    effort_index = next(
        (index for index, line in enumerate(lines) if "--effort" in line),
        -1,
    )
    effort_line = (
        " ".join(lines[effort_index : effort_index + 3])
        if effort_index >= 0
        else ""
    )
    levels = frozenset(
        level
        for level in ("low", "medium", "high", "xhigh", "max")
        if re.search(rf"\b{level}\b", effort_line, flags=re.IGNORECASE)
    )
    return levels or fallback


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]
    time_to_first_token_seconds: float | None = None
    time_to_first_token_source: str = "unavailable"


@dataclass
class ClaudeCodeExecutor:
    _conversation: list[dict[str, Any]]
    _memory_limit: int
    _memory_reset_to: int
    _base_system_prompt: str
    _model_name: str
    _max_turns: int
    _claude_cli_path: str
    _tools_config_json: str
    _allowed_tools_json: str
    _serialized_tools_override_b64: str
    _serialized_tools_append_b64: str
    _model_provider: str
    _default_model: str
    _social_network_model: str
    _scientific_model: str
    _websearcher_model: str
    _business_model: str
    _product_model: str
    _legal_model: str
    _data_statistics_model: str
    _news_media_model: str
    _knowledge_graph_model: str
    _religious_model: str
    _cli_model: str
    _subchack_model: str
    _researcher_administrator_model: str
    _social_network_max_turns: int
    _scientific_max_turns: int
    _websearcher_max_turns: int
    _business_max_turns: int
    _product_max_turns: int
    _legal_max_turns: int
    _data_statistics_max_turns: int
    _news_media_max_turns: int
    _knowledge_graph_max_turns: int
    _religious_max_turns: int
    _cli_max_turns: int
    _subchack_max_turns: int
    _researcher_administrator_max_turns: int
    _min_tools_used: int
    _max_tools_used: int
    _require_task_steps_manager_init_first: bool
    _output_schema_json: str
    _native_task_planning_backend: str = ""
    _require_native_plan_first: bool = False
    _output_schema_name: str = "output_schema"
    _output_schema_strict: bool = True
    _max_context_tokens: int = 0
    _compaction_threshold_ratio: float = 0.50
    _uses_gateway_route: bool = False
    # Whether Claude Code's own WebSearch tool works on this route.
    _native_web_search: bool = True
    # Exa's hosted MCP server, for routes where WebSearch cannot run.
    _fallback_web_mcp_url: str = ""
    _anthropic_api_key: str = ""
    _claude_access_token: str = ""
    # Environment that points the Claude Code CLI (and the Chack MCP server it
    # starts) at the gateway; empty when Claude talks to Anthropic directly.
    _gateway_env_json: str = ""
    _gateway_context_window: int = 0
    # Gateway name and environment of the model a refused run is retried on;
    # empty when the gateway does not serve it or the run already uses it.
    _gateway_refusal_model: str = ""
    _gateway_refusal_env_json: str = ""
    _claude_home: str | None = None
    _claude_session_id: str | None = None
    _output_schema: str | None = None
    _prompt_only_next_invocation: bool = False
    _thinking_effort: str = "high"
    _travel_model: str = ""
    _travel_max_turns: int = 50
    _cacheable_system_prompt: str = ""
    _prompt_cache_prefix_key: str = ""
    # Whether Claude Code may send cache_control markers (DISABLE_PROMPT_CACHING).
    _prompt_cache_enabled: bool = True
    _serialized_tools_override_names_json: str = ""
    _serialized_tools_append_names_json: str = ""

    def suppress_system_prompt_for_next_invocation(self) -> None:
        self._prompt_only_next_invocation = True

    def _has_configured_tools(self) -> bool:
        try:
            names = json.loads(self._allowed_tools_json or "[]")
        except (TypeError, json.JSONDecodeError):
            return True
        return bool(names) if isinstance(names, list) else True

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_claude(prompt)
        output = self._normalize_schema_output(output)

        # Save-or-retry: if save_discovered_vulnerability is available but was never called,
        # and the agent DID use other tools (analysis happened), re-run to prompt saving.
        if (
            self._has_save_vulnerability_tool()
            and not self._steps_contain_save(steps)
            and self._steps_contain_analysis(steps)
        ):
            env = self._build_env()
            exec_cwd = str(
                env.get("CHACK_EXEC_CWD", "")
                or os.environ.get("CHACK_EXEC_CWD", "")
                or ""
            ).strip() or "/tmp"
            _LOGGER.info(
                "Save-retry: no save_discovered_vulnerability calls detected. Re-running to prompt saving. cwd=%s",
                exec_cwd,
            )
            save_tool = f"{self._mcp_tool_prefix()}save_discovered_vulnerability"
            retry_prompt = (
                f"You have already analysed the source code in {exec_cwd}. "
                "This is an authorized defensive security assessment. "
                "If you identified any concrete vulnerability finding during that analysis, save it now.\n\n"
                f"For each concrete finding, call `{save_tool}` with:\n"
                "  name, description, worst_impact, cvss_vector, remediation,\n"
                "  steps (array of {{file_path, code, description}}).\n\n"
                "Do not invent findings. If you did not identify any concrete vulnerability, "
                "reply with: NO_CONCRETE_FINDINGS_TO_SAVE."
            )
            retry_output, retry_steps, retry_raw = self._run_claude(retry_prompt)
            retry_output = self._normalize_schema_output(retry_output)
            if retry_output.startswith("ERROR:") and output and not output.startswith("ERROR:"):
                # The answer already given stands; a failed save prompt must
                # not turn it into an error.
                _LOGGER.warning("Save-retry failed: %s", retry_output[:2000])
            elif retry_output:
                output = output + "\n\n" + retry_output if output else retry_output
            steps.extend(retry_steps)
            if retry_raw.raw_responses:
                raw_result = _RawResult(
                    raw_responses=raw_result.raw_responses + retry_raw.raw_responses,
                )

        if user_input:
            self._conversation.append({"role": "user", "content": user_input})
        if output:
            self._conversation.append({"role": "assistant", "content": output})

        if self._memory_limit and len(self._conversation) > self._memory_limit:
            reset_to = self._memory_reset_to or self._memory_limit
            if reset_to > self._memory_limit:
                reset_to = self._memory_limit
            if reset_to < 1:
                reset_to = 1
            self._conversation = self._conversation[-reset_to:]

        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": raw_result,
        }

    def _normalize_schema_output(self, output: str) -> str:
        """Validate Claude CLI output against the configured schema when present.

        The schema is given to Claude as a plain prompt contract. Normalize the final
        payload through the same JsonSchemaOutput helper used by the OpenAI/OpenRouter
        backends so downstream agents consistently receive schema-shaped JSON.
        """
        if not self._output_schema_json:
            return output
        if (output or "").startswith("ERROR:"):
            # A backend error is not an answer: repairing it against the schema
            # turns its diagnostics into an empty object and hides the error.
            _LOGGER.warning("Claude backend returned an error instead of schema output: %s", output[:2000])
            return output
        validated = self._validated_schema_output(output)
        if validated is None:
            _LOGGER.warning("Claude backend output failed local schema normalization")
            return output
        return validated

    def _validated_schema_output(self, output: str) -> str | None:
        """Schema-shaped JSON for ``output``, or None when it does not match.

        Without a usable schema every output is accepted unchanged.
        """
        raw_schema = str(self._output_schema_json or "").strip()
        if not raw_schema:
            return output

        try:
            schema_obj = json.loads(raw_schema)
        except json.JSONDecodeError:
            _LOGGER.warning("Claude backend received invalid output schema JSON; skipping local validation")
            return output

        if not isinstance(schema_obj, dict):
            return output

        validator = JsonSchemaOutput(
            schema_obj,
            name=str(self._output_schema_name or "output_schema"),
            strict=bool(self._output_schema_strict),
        )
        try:
            return json.dumps(validator.validate_json(output or ""), ensure_ascii=False)
        except Exception:
            extracted = self._extract_embedded_json(output or "")
            if extracted is None:
                return None
            try:
                validated = validator.validate_json(json.dumps(extracted, ensure_ascii=False))
            except Exception:
                return None
            return json.dumps(validated, ensure_ascii=False)

    @staticmethod
    def _missing_answer_error(
        reason: str,
        malformed_tool_names: list[str],
        result_event: dict[str, Any],
        raw_lines: list[str],
    ) -> str:
        """The error for a run that ended without a usable answer.

        A provider that broke the model's tool calls is named as the cause, so
        callers can tell it from a model that simply stopped.
        """
        if malformed_tool_names:
            message = (
                "ERROR: Claude provider returned malformed tool calls: "
                f"{len(malformed_tool_names)} tool call(s) were named with leaked "
                f"tool-call tokens, e.g. {malformed_tool_names[-1][:200]!r}. {reason}"
            )
        else:
            message = f"ERROR: {reason}"
        if result_event:
            diagnostic = {
                key: result_event.get(key)
                for key in ("subtype", "stop_reason", "num_turns", "session_id")
                if result_event.get(key) not in (None, "")
            }
            message += " Event diagnostics: " + json.dumps(diagnostic, ensure_ascii=False, default=str)
        elif raw_lines:
            message += " The CLI sent no result event. Output tail: " + "\n".join(raw_lines)[-2000:]
        return message

    @staticmethod
    def _extract_embedded_json(output: str) -> Any | None:
        text = str(output or "")
        if not text.strip():
            return None
        decoder = json.JSONDecoder()
        idx = 0
        candidates: list[Any] = []
        while idx < len(text):
            starts = [pos for pos in (text.find("{", idx), text.find("[", idx)) if pos >= 0]
            if not starts:
                break
            start = min(starts)
            try:
                parsed, end = decoder.raw_decode(text[start:])
                candidates.append(parsed)
                idx = start + max(end, 1)
            except Exception:
                idx = start + 1
        return candidates[-1] if candidates else None

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)

    def compact_for_resume(
        self, focus_instructions: str = ""
    ) -> ResumeCompactionResult:
        result = ResumeCompactionResult(
            backend="claude",
            method="/compact",
        )
        if not self._claude_session_id:
            return result
        result.attempted = True
        started_at = time.monotonic()
        command = "/compact"
        if str(focus_instructions or "").strip():
            command += f" {focus_instructions.strip()}"
        try:
            output, _steps, raw_result = self._run_claude_once(
                command,
                resume_compaction=True,
            )
            result.raw_responses = list(raw_result.raw_responses or [])
            normalized = str(output or "").strip().lower()
            if normalized.startswith("error:") or "unknown slash command" in normalized:
                result.error = str(output or "Claude /compact failed.")
            else:
                result.succeeded = True
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_seconds = max(0.0, time.monotonic() - started_at)
        return result

    def _has_save_vulnerability_tool(self) -> bool:
        """Check if save_discovered_vulnerability is in the allowed tools."""
        try:
            if self._allowed_tools_json:
                return "save_discovered_vulnerability" in self._allowed_tools_json
        except Exception:
            pass
        try:
            if self._serialized_tools_append_b64:
                import base64
                decoded = base64.b64decode(self._serialized_tools_append_b64).decode("utf-8", errors="replace")
                return "save_discovered_vulnerability" in decoded
        except Exception:
            pass
        return False

    def _mcp_tool_prefix(self) -> str:
        # Claude Code names MCP tools as mcp__<server>__<tool>
        return "mcp__chack_tools__"

    @staticmethod
    def _steps_contain_save(steps: list[tuple[ToolAction, Any]]) -> bool:
        for step, _ in steps:
            if "save_discovered_vulnerability" in str(getattr(step, "tool", "") or ""):
                return True
        return False

    @staticmethod
    def _steps_contain_analysis(steps: list[tuple[ToolAction, Any]]) -> bool:
        analysis_tools = {"bash", "exec", "read", "grep", "glob"}
        for step, _ in steps:
            if str(getattr(step, "tool", "") or "").lower() in analysis_tools:
                return True
        return False

    def _compose_prompt(self, user_input: str) -> str:
        if self._prompt_only_next_invocation:
            self._prompt_only_next_invocation = False
            self._cacheable_system_prompt = ""
            self._prompt_cache_prefix_key = ""
            return str(user_input or "")
        base = str(self._base_system_prompt or "").strip()

        policy_lines: list[str] = []
        native_plan_line = native_planning_prompt(
            self._native_task_planning_backend,
            required_first=self._require_native_plan_first,
        )
        if native_plan_line:
            policy_lines.append(native_plan_line)
        try:
            parsed_allowed_tools = json.loads(self._allowed_tools_json or "[]")
        except (TypeError, json.JSONDecodeError):
            parsed_allowed_tools = []
        mcp_tool_names: list[str] = []
        if isinstance(parsed_allowed_tools, list):
            seen_mcp_tool_names: set[str] = set()
            for raw_name in parsed_allowed_tools:
                name = str(raw_name or "").strip()
                if not name or name in seen_mcp_tool_names:
                    continue
                seen_mcp_tool_names.add(name)
                mcp_tool_names.append(name)

        if mcp_tool_names:
            mappings = ", ".join(
                f"`{name}` -> `mcp__chack_tools__{name}`" for name in mcp_tool_names
            )
            policy_lines.append(
                "- Claude Code exposes the configured custom tools through the "
                "`chack_tools` MCP server. Call the exact prefixed MCP names below; "
                "do not call their unprefixed aliases: "
                f"{mappings}."
            )
        if self._require_task_steps_manager_init_first:
            policy_lines.append(
                "- First, call `mcp__chack_tools__task_steps_manager` with "
                "action=init before any other tool call."
            )
        if self._min_tools_used > 0:
            policy_lines.append(
                f"- Use at least {self._min_tools_used} non-task tool calls before finalizing."
            )
        if self._max_tools_used > 0:
            policy_lines.append(
                f"- Do not exceed {self._max_tools_used} non-task tool calls in total."
            )

        # Inject save requirement when save_discovered_vulnerability is available
        if self._has_save_vulnerability_tool():
            mcp_prefix = self._mcp_tool_prefix()
            save_tool_name = f"{mcp_prefix}save_discovered_vulnerability"
            save_block = (
                "- CRITICAL: For EACH vulnerability you find, you MUST save it.\n"
                f"  Call `{save_tool_name}` with parameters:\n"
                "    name, description, worst_impact, cvss_vector, remediation,\n"
                "    steps (array of objects with file_path, code, description).\n"
                "    Each step MUST include the actual file_path in the repo and the\n"
                "    verbatim source code snippet from that file.\n"
                "  Never write directly to AISEC_LOCAL_VULN_STORE_PATH or any\n"
                "  /tmp/aisec_local_vulnerabilities_* directory. Use only the MCP\n"
                "  save tool or save_vuln.sh in the current repository.\n"
                "  If the save tool is unavailable, run\n"
                "    bash save_vuln.sh '<JSON with the same fields>'\n"
                "  If you do NOT save findings they are LOST."
            )
            policy_lines.append(save_block)

        schema_lines: list[str] = []
        if self._output_schema_json:
            schema_name = str(self._output_schema_name or "output_schema").strip() or "output_schema"
            schema_lines.append(
                "\n### OUTPUT CONTRACT\nReturn JSON only, exactly one JSON object."
            )
            schema_lines.append(f"Use schema name: {schema_name}")
            schema_lines.append("Schema:")
            schema_lines.append(self._output_schema_json)
            if self._output_schema_strict:
                schema_lines.append("Your response must strictly match the JSON schema.")
            else:
                schema_lines.append("Match the schema as closely as possible.")

        policy_block = ""
        schema_block = ""
        if policy_lines:
            policy_block = "\n\n### TOOL USAGE POLICY\n" + "\n".join(policy_lines)
        if schema_lines:
            schema_block = "\n" + "\n".join(schema_lines)

        cache_parts = split_prompt_cache_breakpoint(user_input)
        if cache_parts.has_breakpoint:
            system_parts = [
                part
                for part in (
                    base,
                    policy_block,
                    schema_block,
                    cache_parts.stable_prefix,
                )
                if part.strip()
            ]
            self._cacheable_system_prompt = "\n".join(system_parts)
            self._prompt_cache_prefix_key = prompt_cache_key(
                self._cacheable_system_prompt
            )
            _LOGGER.info(
                "Using cache-stable Claude system prefix: chars=%d key=%s",
                len(self._cacheable_system_prompt),
                self._prompt_cache_prefix_key,
            )
            return cache_parts.dynamic_suffix

        # Claude caches prompt prefixes in tools -> system -> messages order.
        # Always keep Chack's immutable runtime/system, policy, and output
        # contract in Claude Code's system layer, even when the YAML has no
        # task-specific breakpoint. The changing task stays in the user
        # message. Explicit markers can still extend this stable layer with
        # large per-task context.
        system_parts = [
            part
            for part in (base, policy_block, schema_block)
            if part.strip()
        ]
        self._cacheable_system_prompt = "\n".join(system_parts)
        self._prompt_cache_prefix_key = (
            prompt_cache_key(self._cacheable_system_prompt)
            if self._cacheable_system_prompt
            else ""
        )
        if self._cacheable_system_prompt:
            _LOGGER.info(
                "Using cache-stable Claude system prefix: chars=%d key=%s",
                len(self._cacheable_system_prompt),
                self._prompt_cache_prefix_key,
            )
        return str(user_input or "")

    def _refresh_provider_credentials(self) -> None:
        if getattr(self, "_uses_gateway_route", False):
            # Gateway and OpenRouter runs authenticate with the gateway key;
            # a Claude subscription token from the hook must not replace it.
            return
        credentials = run_provider_pre_launch_hook("claude")
        if credentials.get("clear_access_token", "").lower() == "true":
            self._claude_access_token = ""
            self._claude_session_id = None
            return
        access_token = str(credentials.get("access_token", "") or "").strip()
        if access_token and access_token != self._claude_access_token:
            self._claude_access_token = access_token
            # Claude resume IDs are account-scoped; start a fresh session when
            # the ordered credential pool advances.
            self._claude_session_id = None

    def _prompt_cache_route(self) -> tuple[str, str]:
        base_url = str(json.loads(self._gateway_env_json or "{}").get("ANTHROPIC_BASE_URL", "") or "")
        return base_url, str(self._model_name or "")

    def _sends_prompt_cache_markers(self) -> bool:
        return (
            self._prompt_cache_enabled
            and self._prompt_cache_route() not in _PROMPT_CACHE_REJECTED
        )

    def _run_claude(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._refresh_provider_credentials()
        session_before = self._claude_session_id
        result = self._run_claude_once(prompt)
        if (
            result[0].startswith("ERROR:")
            and provider_rejects_prompt_cache(result[0])
            and self._sends_prompt_cache_markers()
        ):
            # The provider refused the request before the model ran, so it is
            # sent again, once, without cache markers.
            _LOGGER.warning(
                "Model %s rejects prompt-cache markers; retrying without them. "
                "Set prompt_cache_enabled: false in the agent config for this model to skip the failing request.",
                self._model_name,
            )
            _PROMPT_CACHE_REJECTED.add(self._prompt_cache_route())
            self._claude_session_id = session_before
            result = self._run_claude_once(prompt)
        if self._gateway_refusal_model and _is_model_refusal(result[0]):
            _LOGGER.warning(
                "Model %s refused the request; retrying on %s.",
                self._model_name,
                self._gateway_refusal_model,
            )
            model_name, env_json = self._model_name, self._gateway_env_json
            self._model_name, self._gateway_env_json = self._gateway_refusal_model, self._gateway_refusal_env_json
            self._claude_session_id = session_before
            try:
                result = self._run_claude_once(prompt)
            finally:
                self._model_name, self._gateway_env_json = model_name, env_json
        while self._looks_like_credential_failure(result[0]):
            rotated = rotate_environment_credentials("claude", self._claude_access_token)
            rotated_token = str(rotated.get("access_token", "") or "").strip()
            if not rotated_token or rotated_token == self._claude_access_token:
                break
            _LOGGER.warning(
                "Claude access token hit an auth or quota limit. Retrying with "
                "the next credential from CLAUDE_TOKEN_POOL_JSON."
            )
            self._claude_access_token = rotated_token
            self._claude_session_id = None
            result = self._run_claude_once(prompt)
        if self._should_retry_with_anthropic_api_key(result[0]):
            _LOGGER.warning(
                "Claude Code OAuth token is unavailable for this request. "
                "Falling back to ANTHROPIC_API_KEY."
            )
            self._claude_access_token = ""
            self._claude_session_id = None
            return self._run_claude_once(prompt)

        wait_seconds = _seconds_until_claude_quota_reset(result[0])
        max_wait_seconds = max(
            0,
            int(os.environ.get("CHACK_CLAUDE_QUOTA_RESET_MAX_WAIT_SECONDS", "900") or "900"),
        )
        if wait_seconds is None or wait_seconds > max_wait_seconds:
            return result

        _LOGGER.warning(
            "Claude Code OAuth session quota resets in %ss; waiting once and retrying "
            "with the configured Claude token.",
            wait_seconds,
        )
        time.sleep(wait_seconds)
        self._claude_session_id = None
        return self._run_claude_once(prompt)

    def _looks_like_credential_failure(self, output: str) -> bool:
        if self._uses_gateway_route or not self._claude_access_token:
            return False
        text = str(output or "").strip().lower()
        if not text.startswith("error:"):
            return False
        return any(
            marker in text
            for marker in (
                "authentication",
                "unauthorized",
                "invalid token",
                "invalid oauth",
                "oauth token",
                "status code 401",
                "http 401",
                "authentication_error",
                "session limit",
                "usage limit",
                "rate limit",
                "hit your limit",
                "quota exceeded",
                "credit balance",
            )
        )

    def _should_retry_with_anthropic_api_key(self, output: str) -> bool:
        if self._uses_gateway_route:
            return False
        if not self._claude_access_token or not self._anthropic_api_key:
            return False
        text = str(output or "").strip().lower()
        if not text.startswith("error:"):
            return False
        return any(
            marker in text
            for marker in (
                "authentication",
                "unauthorized",
                "invalid token",
                "invalid oauth",
                "oauth token",
                "status code 401",
                "http 401",
                "authentication_error",
                "session limit",
                "usage limit",
                "rate limit",
                "hit your limit",
                "quota exceeded",
                "credit balance",
            )
        )

    def _run_claude_once(
        self,
        prompt: str,
        *,
        resume_compaction: bool = False,
    ) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_claude_home_and_settings()

        startup_status_path = self._mcp_startup_status_path()
        try:
            os.remove(startup_status_path)
        except FileNotFoundError:
            pass

        command = self._build_command(
            prompt,
            resume_compaction=resume_compaction,
        )
        env = self._build_env()
        exec_cwd = str(env.get("CHACK_EXEC_CWD", "") or os.environ.get("CHACK_EXEC_CWD", "") or "").strip() or None
        bash_vulns_dir = self._bash_vuln_fallback_dir(exec_cwd)
        if bash_vulns_dir:
            os.makedirs(bash_vulns_dir, exist_ok=True)
            env["CHACK_VULN_FALLBACK_DIR"] = bash_vulns_dir
        self._write_save_vuln_script(exec_cwd)
        bash_vuln_snapshot = self._snapshot_bash_saved_vulns(bash_vulns_dir)
        timeout_seconds = cli_exec_timeout_seconds(CLAUDE_EXEC_TIMEOUT_ENV)

        _LOGGER.info(
            "Starting Claude Code process: model=%s timeout_seconds=%s session_id=%s cwd=%s",
            self._model_name,
            timeout_seconds,
            self._claude_session_id or "",
            exec_cwd or "(inherited)",
        )

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=exec_cwd,
                start_new_session=True,
            )
        except FileNotFoundError:
            return (
                "ERROR: Claude CLI executable was not found. "
                f"Configured path: {self._claude_cli_path!r}. "
                "Install Claude Code (e.g. curl -fsSL https://claude.ai/install.sh | bash) "
                "or set CLAUDE_CLI_PATH to the absolute executable path.",
                [],
                _RawResult(raw_responses=[]),
            )
        except Exception as exc:
            return (
                f"ERROR: Failed to launch Claude CLI: {type(exc).__name__}: {exc}",
                [],
                _RawResult(raw_responses=[]),
            )

        cancel_registration = register_process(process, _terminate_process_tree)
        output_parts: list[str] = []
        # The closing ``result`` event repeats the final assistant text; when it
        # is present it is the answer, and the streamed text is only a fallback.
        result_parts: list[str] = []
        steps: list[tuple[ToolAction, Any]] = []
        raw_lines: list[str] = []
        raw_responses: list[Any] = []
        tool_calls: dict[str, tuple[str, Any]] = {}
        # Tool calls whose name is not a tool name at all: the provider failed
        # to parse the model's tool-call tokens and leaked them into the name.
        malformed_tool_names: list[str] = []
        result_event: dict[str, Any] = {}
        failed_mcp_servers: dict[str, str] = {}
        started_at = time.monotonic()
        time_to_first_token_seconds: float | None = None
        return_seen = False

        try:
            # Claude Code accepts print-mode prompts on stdin. Keeping the prompt
            # out of argv avoids the operating-system ARG_MAX limit when a
            # multi-round agent carries a large set of prior findings forward.
            if process.stdin is not None:
                try:
                    process.stdin.write(prompt)
                    process.stdin.close()
                except Exception as exc:
                    _terminate_process_tree(process)
                    return (
                        f"ERROR: Failed to send prompt to Claude CLI: {type(exc).__name__}: {exc}",
                        steps,
                        _RawResult(raw_responses=raw_responses),
                    )

            while True:
                if cancellation_requested():
                    _terminate_process_tree(process)
                    return (
                        "ERROR: Claude execution cancelled.",
                        steps,
                        _RawResult(raw_responses=raw_responses),
                    )
                if timeout_seconds > 0 and (time.monotonic() - started_at) > timeout_seconds:
                    _terminate_process_tree(process)
                    return (
                        f"ERROR: Claude execution timed out after {timeout_seconds}s.",
                        steps,
                        _RawResult(raw_responses=raw_responses),
                    )

                if process.stdout is None:
                    break

                line = process.stdout.readline()
                if line == "" and process.poll() is not None:
                    break
                if not line:
                    time.sleep(0.05)
                    continue

                raw_line = str(line).rstrip("\n")
                if raw_line:
                    raw_lines.append(raw_line)

                event = self._parse_event_line(raw_line)
                if not event:
                    continue

                event_type = str(event.get("type") or event.get("event") or "").strip().lower()
                subtype = str(event.get("subtype") or "").strip().lower()
                if (
                    time_to_first_token_seconds is None
                    and event_type != "system"
                ):
                    time_to_first_token_seconds = max(
                        0.0,
                        time.monotonic() - started_at,
                    )

                if event_type == "system" and subtype == "init":
                    session_id = str(event.get("session_id") or "").strip()
                    if session_id:
                        self._claude_session_id = session_id
                    for server in event.get("mcp_servers") or []:
                        if not isinstance(server, dict):
                            continue
                        name = str(server.get("name") or "").strip()
                        status = str(server.get("status") or "").strip().lower()
                        if name and status not in {"connected", "ready"}:
                            failed_mcp_servers[name] = status or "unknown"
                    continue

                if event_type == "assistant":
                    msg = event.get("message")
                    if isinstance(msg, dict):
                        self._extract_message_content(msg, output_parts, tool_calls, steps)
                        malformed_tool_names.extend(_malformed_tool_use_names(msg))
                    continue

                if event_type == "user":
                    msg = event.get("message")
                    if isinstance(msg, dict):
                        # User turns carry tool results and the prompt, never answer text.
                        self._extract_message_content(msg, [], tool_calls, steps)
                    continue

                if event_type == "result":
                    result_event = event
                    result_text = str(event.get("result") or "").strip()
                    structured_output = event.get("structured_output")
                    if structured_output is not None:
                        # The schema-validated object is the answer; the result
                        # text may repeat it or add prose, and joining the two
                        # yields text that is no longer one JSON document.
                        self._append_structured_output(result_parts, structured_output)
                    elif result_text:
                        result_parts.append(result_text)
                    return_seen = True

                    usage: dict[str, Any] = {}
                    if "usage" in event and isinstance(event.get("usage"), dict):
                        usage_dict = event["usage"]
                        if isinstance(usage_dict, dict):
                            usage = _normalized_claude_usage(usage_dict)
                    elif "input_tokens" in event or "output_tokens" in event:
                        usage = _normalized_claude_usage(event)
                    if usage:
                        report_live_usage(
                            self._model_name,
                            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                            completion_tokens=int(usage.get("output_tokens", 0) or 0),
                            cached_prompt_tokens=int(
                                usage.get("input_tokens_details", {}).get("cached_tokens", 0) or 0
                            ),
                            cache_write_tokens=int(
                                usage.get("input_tokens_details", {}).get("cache_write_tokens", 0) or 0
                            ),
                        )
                        raw_responses.append({"usage": usage})

                    if (
                        str(event.get("subtype") or "").strip().lower() == "error"
                        or bool(event.get("is_error"))
                    ):
                        diagnostic: dict[str, Any] = {}
                        for key in (
                            "subtype",
                            "is_error",
                            "stop_reason",
                            "stop_sequence",
                            "num_turns",
                            "duration_ms",
                            "duration_api_ms",
                            "session_id",
                            "error",
                            "errors",
                            "permission_denials",
                        ):
                            value = event.get(key)
                            if value not in (None, "", [], {}):
                                diagnostic[key] = value
                        diagnostic_text = ""
                        if diagnostic:
                            try:
                                diagnostic_text = (
                                    " Event diagnostics: "
                                    + json.dumps(diagnostic, ensure_ascii=False, default=str)[:2000]
                                )
                            except Exception:
                                diagnostic_text = ""
                        return (
                            "ERROR: Claude returned an error in final result event. "
                            + (result_text or "No error text was returned.")
                            + diagnostic_text,
                            steps,
                            _RawResult(raw_responses=raw_responses),
                        )
                    break

                if event_type == "tool_use":
                    self._record_tool_use(event, tool_calls)
                    if str(event.get("name") or "").strip() == "StructuredOutput":
                        self._append_structured_output(output_parts, event.get("input"))
                    continue

                if event_type == "tool_result":
                    self._record_tool_result(event, tool_calls, steps)
                    continue
        finally:
            unregister_process(cancel_registration)
            self._log_mcp_startup_status(startup_status_path)
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass

        return_code = process.wait()

        _LOGGER.info(
            "Claude process exit: return_code=%s raw_lines=%d output_parts=%d return_seen=%s",
            return_code,
            len(raw_lines),
            len(output_parts),
            return_seen,
        )
        if raw_lines:
            _LOGGER.info("Claude raw_lines[0]: %s", raw_lines[0][:300] if raw_lines else "")
        if return_code != 0 and raw_lines:
            _LOGGER.warning("Claude stderr/stdout dump (first 1000 chars): %s", "\n".join(raw_lines)[:1000])
            if self._raw_lines_have_missing_resume_session(raw_lines):
                _LOGGER.warning(
                    "Claude resume session %s no longer exists; clearing it for the next launch.",
                    self._claude_session_id or "",
                )
                self._claude_session_id = None

        if return_code != 0:
            details = "\n".join(raw_lines).strip() or "No output captured."
            return (
                f"ERROR: Claude exec failed (exit={return_code}). {details}",
                steps,
                _RawResult(raw_responses=raw_responses),
            )

        if failed_mcp_servers:
            failure_summary = ", ".join(
                f"{name}={status}" for name, status in sorted(failed_mcp_servers.items())
            )
            return (
                "ERROR: Required Claude MCP server failed to start: " + failure_summary,
                steps,
                _RawResult(raw_responses=raw_responses),
            )

        # Pick up any JSON-fallback vulns written via save_vuln.sh
        bash_vuln_steps = self._collect_bash_saved_vulns(bash_vulns_dir, bash_vuln_snapshot)
        if bash_vuln_steps:
            steps.extend(bash_vuln_steps)

        if malformed_tool_names:
            _LOGGER.warning(
                "Claude provider returned %d malformed tool call(s) for model %s, e.g. %r.",
                len(malformed_tool_names),
                self._model_name,
                malformed_tool_names[0][:200],
            )
        response = "".join(result_parts or output_parts).strip()
        missing_answer = ""
        if not response:
            missing_answer = "Claude finished without a final answer."
        elif (
            self._output_schema_json
            and result_event.get("structured_output") is None
            and self._validated_schema_output(response) is None
        ):
            missing_answer = (
                "Claude finished without the structured output its schema requires. "
                f"Final text: {response[:500]!r}."
            )
        if missing_answer:
            return (
                self._missing_answer_error(
                    missing_answer,
                    malformed_tool_names,
                    result_event,
                    raw_lines,
                ),
                steps,
                _RawResult(raw_responses=raw_responses),
            )

        return response, steps, _RawResult(
            raw_responses=raw_responses,
            time_to_first_token_seconds=time_to_first_token_seconds,
            time_to_first_token_source="claude_first_response_event",
        )

    def _mcp_startup_status_path(self) -> str:
        return os.path.join(str(self._claude_home or ""), "mcp_startup_status.json")

    def _log_mcp_startup_status(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            _LOGGER.warning("Claude MCP startup receipt was not written: %s", path)
            return
        except Exception as exc:
            _LOGGER.warning("Claude MCP startup receipt could not be read: %s", exc)
            return
        tool_names = payload.get("tool_names") if isinstance(payload, dict) else []
        error = str(payload.get("error", "") if isinstance(payload, dict) else "")
        log_method = _LOGGER.warning if error else _LOGGER.info
        log_method(
            "Claude MCP startup receipt: tools=%s error=%s",
            ",".join(str(name) for name in (tool_names or [])),
            error[:500] or "none",
        )

    @staticmethod
    def _raw_lines_have_missing_resume_session(raw_lines: list[str]) -> bool:
        text = "\n".join(str(line or "") for line in raw_lines).lower()
        return "no conversation found with session id" in text

    @staticmethod
    def _append_structured_output(output_parts: list[str], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (dict, list)):
            output_parts.append(json.dumps(value, ensure_ascii=False))
            return
        text = str(value).strip()
        if text:
            output_parts.append(text)

    def _extract_message_content(
        self,
        message: dict[str, Any],
        output_parts: list[str],
        tool_calls: dict[str, tuple[str, Any]],
        steps: list[tuple[ToolAction, Any]],
    ) -> None:
        content = message.get("content")
        if content is None:
            return

        if isinstance(content, str):
            if content.strip():
                output_parts.append(content)
            return

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type == "text":
                    text = str(item.get("text") or item.get("content") or "")
                    if text:
                        output_parts.append(text)
                    continue

                if item_type == "tool_use":
                    tool_use_id = str(item.get("id") or item.get("tool_use_id") or "").strip()
                    tool_name = str(item.get("name") or "").strip()
                    tool_input = item.get("input")
                    if not isinstance(tool_input, dict):
                        if isinstance(tool_input, list):
                            tool_input = {"input": tool_input}
                        elif tool_input is None:
                            tool_input = {}
                        else:
                            try:
                                tool_input = {
                                    "input": json.loads(str(tool_input))
                                    if str(tool_input).strip().startswith("{")
                                    else str(tool_input)
                                }
                            except Exception:
                                tool_input = {"input": str(tool_input)}
                    if tool_use_id:
                        tool_calls[tool_use_id] = (tool_name, tool_input)
                    if tool_name == "StructuredOutput" and tool_input:
                        self._append_structured_output(output_parts, tool_input)
                    continue

                if item_type == "tool_result":
                    synthetic_event = {
                        "tool_use_id": str(item.get("tool_use_id") or ""),
                        "is_error": bool(item.get("is_error") or False),
                        "content": item.get("content"),
                        "result": item.get("result"),
                    }
                    self._record_tool_result(synthetic_event, tool_calls, steps)
                    continue

    def _record_tool_use(self, event: dict[str, Any], tool_calls: dict[str, tuple[str, Any]]) -> None:
        tool_use_id = str(event.get("tool_use_id") or event.get("id") or "").strip()
        tool_name = str(event.get("name") or "").strip()
        tool_input = event.get("input")
        if not isinstance(tool_input, dict):
            if isinstance(tool_input, list):
                tool_input = {"input": tool_input}
            elif tool_input is None:
                tool_input = {}
            else:
                try:
                    tool_input = {
                        "input": json.loads(str(tool_input))
                        if str(tool_input).strip().startswith("{")
                        else str(tool_input)
                    }
                except Exception:
                    tool_input = {"input": str(tool_input)}
        if tool_use_id:
            tool_calls[tool_use_id] = (tool_name, tool_input)

    def _record_tool_result(
        self,
        event: dict[str, Any],
        tool_calls: dict[str, tuple[str, Any]],
        steps: list[tuple[ToolAction, Any]],
    ) -> None:
        tool_use_id = str(event.get("tool_use_id") or "").strip()
        status = "error" if bool(event.get("is_error") or str(event.get("status") or "").lower() == "error") else "success"

        tool_name = ""
        tool_input: Any = {}
        if tool_use_id and tool_use_id in tool_calls:
            tool_name, tool_input = tool_calls.pop(tool_use_id)
        if not tool_name:
            tool_name = str(event.get("name") or "").strip()
        result_payload = event.get("content")
        if result_payload is None:
            result_payload = event.get("result")
        if isinstance(result_payload, list) and result_payload:
            flattened = "".join(
                str(item.get("text") or item.get("content") or "")
                if isinstance(item, dict)
                else str(item)
                for item in result_payload
            )
            result_payload = flattened

        step_input = {
            "tool_id": tool_use_id,
            "status": status,
            "tool_input": tool_input,
        }
        if result_payload is not None:
            step_input["result"] = result_payload
        step = ToolAction(tool=tool_name or "tool", tool_input=step_input)
        steps.append((step, None))
        self._log_tool_called(step.tool, step.tool_input)

        if self._native_task_planning_backend:
            sync_claude_native_task(
                step.tool,
                tool_input,
                status=status,
                result=result_payload,
            )
        if str(step.tool).strip() == "task_steps_manager":
            self._sync_task_steps_manager(tool_input, status)

    def _write_save_vuln_script(self, exec_cwd: str | None) -> str | None:
        """Write save_vuln.sh + the .vulns/ directory in exec_cwd.

        The script is the JSON fallback for saving a vulnerability when the
        MCP save tool cannot be called; the prompt tells the model how to use
        it. Only written when save_discovered_vulnerability is available.
        Instruction files (AGENTS.md, .github/copilot-instructions.md) are not
        written: Claude Code runs with project settings disabled, so it would
        not read them, and the working directory is usually a repository
        under audit that must not be modified. Returns the script path or None.
        """
        if not exec_cwd or not self._has_save_vulnerability_tool():
            return None
        try:
            # Create .vulns/ directory for bash JSON fallback
            vulns_dir = os.path.join(exec_cwd, ".vulns")
            os.makedirs(vulns_dir, exist_ok=True)

            # Write save_vuln.sh
            save_vuln_path = os.path.join(exec_cwd, "save_vuln.sh")
            save_vuln_sh = '''#!/usr/bin/env bash
# save_vuln.sh — Save a vulnerability as validated JSON to .vulns/
# Usage: bash save_vuln.sh '<JSON>'
set -e
JSON="$1"
if [ -z "$JSON" ]; then
    echo "Usage: bash save_vuln.sh '<JSON>'" >&2
    exit 1
fi
VULNS_DIR="${CHACK_VULN_FALLBACK_DIR:-.vulns}"
mkdir -p "$VULNS_DIR"
python3 - "$JSON" "$VULNS_DIR" <<'PYEOF'
import json, os, sys, time
data = json.loads(sys.argv[1])
vulns_dir = sys.argv[2]
errors = []
if not data.get("name"):
    errors.append("missing: name")
if not data.get("description"):
    errors.append("missing: description")
if errors:
    print("ERROR: Vulnerability NOT saved. Fix these issues:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
fname = os.path.join(vulns_dir, f"vuln_{int(time.time()*1000)}.json")
with open(fname, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved to {fname}")
PYEOF
'''
            with open(save_vuln_path, "w", encoding="utf-8") as f:
                f.write(save_vuln_sh)
            os.chmod(save_vuln_path, 0o755)
            return save_vuln_path
        except Exception as exc:
            _LOGGER.warning("Failed to write save_vuln.sh: %s", exc)
            return None

    def _bash_vuln_fallback_dir(self, exec_cwd: str | None) -> str:
        if not exec_cwd:
            return ""
        session_dir = os.path.basename(str(self._claude_home or "default")) or "default"
        return os.path.join(exec_cwd, ".vulns", session_dir)

    @staticmethod
    def _snapshot_bash_saved_vulns(vulns_dir: str | None) -> dict[str, tuple[int, int]]:
        if not vulns_dir:
            return {}
        if not os.path.isdir(vulns_dir):
            return {}
        snapshot: dict[str, tuple[int, int]] = {}
        for fname in os.listdir(vulns_dir):
            if not fname.endswith(".json"):
                continue
            try:
                stat = os.stat(os.path.join(vulns_dir, fname))
                snapshot[fname] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
        return snapshot

    @staticmethod
    def _collect_bash_saved_vulns(
        vulns_dir: str | None,
        previous_snapshot: dict[str, tuple[int, int]] | None = None,
    ) -> list[tuple[ToolAction, Any]]:
        """Read this session's new save_vuln.sh JSON files and convert them to steps."""
        if not vulns_dir:
            return []
        if not os.path.isdir(vulns_dir):
            return []
        steps: list[tuple[ToolAction, Any]] = []
        for fname in os.listdir(vulns_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(vulns_dir, fname)
            try:
                stat = os.stat(fpath)
                if (previous_snapshot or {}).get(fname) == (stat.st_mtime_ns, stat.st_size):
                    continue
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.loads(fh.read())
                if not isinstance(data, dict) or not data.get("name"):
                    continue
                tool_input: dict[str, Any] = {
                    "name": str(data.get("name", "")),
                    "description": str(data.get("description", "")),
                    "worst_impact": str(data.get("worst_impact", "")),
                    "cvss_vector": str(data.get("cvss_vector", "")),
                    "remediation": str(data.get("remediation", "")),
                }
                raw_steps = data.get("steps")
                if isinstance(raw_steps, list) and raw_steps:
                    tool_input["steps"] = raw_steps
                step = ToolAction(
                    tool="chack_tools-save_discovered_vulnerability",
                    tool_input={
                        "tool_id": f"bash_save_{fname}",
                        "status": "success",
                        "tool_input": tool_input,
                        "result": "Vulnerability saved via JSON fallback",
                    },
                )
                steps.append((step, None))
            except Exception:
                continue
        if steps:
            _LOGGER.info("Collected %d JSON-fallback vulnerabilities from %s", len(steps), vulns_dir)
        return steps

    def _build_command(
        self,
        prompt: str,
        *,
        resume_compaction: bool = False,
    ) -> list[str]:
        try:
            tools_cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            tools_cfg = {}

        args: list[str] = [
            self._claude_cli_path,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--allow-dangerously-skip-permissions",
            "--dangerously-skip-permissions",
            # Load only user settings (Chack's isolated configuration): the
            # working directory is often a repository under audit, and its
            # CLAUDE.md, .claude/ settings, hooks, skills and agents are
            # untrusted input that would also cost tokens on every request.
            "--setting-sources",
            "user",
        ]
        has_cli_tools = self._has_configured_tools() or self._playwright_mcp_enabled()
        if resume_compaction or not has_cli_tools:
            args.extend(["--tools", ""])
        else:
            args.extend(
                [
                    "--mcp-config",
                    os.path.join(
                        self._claude_home or os.getcwd(),
                        "settings.json",
                    ),
                    "--strict-mcp-config",
                ]
            )
        if not resume_compaction and has_cli_tools:
            # Claude Code otherwise sends every built-in tool (subagents,
            # workflows, cron, worktrees, monitors, ...) on each request: ~60k
            # chars a headless run never uses. ``--tools`` limits only the
            # built-in set; MCP tools stay available. Names a CLI version does
            # not have are ignored.
            # Never Bash: Chack's exec tool is the run's only shell, and a run
            # without it gets no host file access either.
            builtin_tools = (
                list(_CLAUDE_BUILTIN_TOOLS) if "exec" in self._offered_tool_names() else ["ToolSearch"]
            )
            if tools_cfg.get("native_web_search_enabled", True):
                builtin_tools.append("WebFetch")
                if self._native_web_search:
                    # WebSearch runs on Anthropic's API (OpenRouter serves it
                    # too); elsewhere the run searches through the fallback
                    # MCP server.
                    builtin_tools.append("WebSearch")
            if self._native_task_planning_backend:
                builtin_tools.extend(_CLAUDE_PLANNING_TOOLS)
            args.extend(["--tools", ",".join(builtin_tools)])
        if not resume_compaction:
            args.extend(["--settings", _CLAUDE_HEADLESS_SETTINGS])

        if resume_compaction:
            args.extend(["--max-turns", "1"])
        elif self._max_turns > 0:
            args.extend(["--max-turns", str(self._max_turns)])
        args.extend(
            [
                "--effort",
                claude_thinking_effort(
                    self._thinking_effort,
                    _claude_supported_effort_levels(self._claude_cli_path),
                ),
            ]
        )
        # Opt into the 1M-token context beta when the configured budget exceeds
        # the default 200k window. This only raises the model's window so the
        # auto-compaction trigger (CLAUDE_CODE_AUTO_COMPACT_WINDOW, set in the
        # env) can be placed above 200k; the compaction threshold itself keeps
        # the session alive rather than killing it. Skipped on gateway routes:
        # the Anthropic beta header does not apply and betas are disabled.
        if (
            self._max_context_tokens > _CLAUDE_DEFAULT_CONTEXT_WINDOW
            and not self._uses_gateway_route
            and not self._claude_access_token
        ):
            args.extend(["--betas", _CLAUDE_1M_CONTEXT_BETA])
        if self._model_name:
            args.extend(["--model", self._model_name])
        if self._cacheable_system_prompt and not resume_compaction:
            # Claude Code owns the provider cache_control blocks. Appended
            # system content is part of its stable system cache layer, while
            # the changing suffix continues to arrive as the user message.
            system_prompt_flag = (
                "--append-system-prompt"
                if has_cli_tools
                else "--system-prompt"
            )
            args.extend(
                [system_prompt_flag, self._cacheable_system_prompt]
            )
        if self._claude_session_id:
            args.extend(["--resume", self._claude_session_id])
        if self._output_schema_json and not resume_compaction:
            args.extend(
                [
                    "--json-schema",
                    _claude_cli_json_schema(self._output_schema_json),
                ]
            )

        return args

    def _offered_tool_names(self) -> list[str]:
        try:
            names = json.loads(self._allowed_tools_json or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
        return [str(name) for name in names] if isinstance(names, list) else []

    def _build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if v is not None}
        env["PYTHONUNBUFFERED"] = "1"
        augment_subprocess_pythonpath(env)
        # A run's instructions come from Chack only, never from CLAUDE.md files
        # on the host or in the working directory.
        env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"

        # Claude Code defers MCP schemas through ToolSearch by default, but
        # Haiku does not support the tool_reference blocks that ToolSearch
        # requires. Load MCP tools up front for Haiku so exact custom tools do
        # not intermittently fail as "No such tool available" and fall back to
        # similarly named native tools.
        if "haiku" in str(self._model_name or "").strip().lower():
            env["ENABLE_TOOL_SEARCH"] = "false"

        self._set_env_or_file(
            env,
            "CHACK_TOOLS_CONFIG_JSON",
            self._tools_config_json,
            path_env_key=CHACK_TOOLS_CONFIG_JSON_PATH_ENV,
            prefix="chack_tools_config_",
        )
        self._set_env_or_file(
            env,
            "CHACK_ALLOWED_TOOLS_JSON",
            self._allowed_tools_json,
            path_env_key=CHACK_ALLOWED_TOOLS_JSON_PATH_ENV,
            prefix="chack_allowed_tools_",
        )
        self._set_env_or_file(
            env,
            CHACK_TOOLS_OVERRIDE_B64_ENV,
            self._serialized_tools_override_b64,
            path_env_key=CHACK_TOOLS_OVERRIDE_B64_PATH_ENV,
            prefix="chack_tools_override_",
        )
        self._set_env_or_file(
            env,
            CHACK_TOOLS_APPEND_B64_ENV,
            self._serialized_tools_append_b64,
            path_env_key=CHACK_TOOLS_APPEND_B64_PATH_ENV,
            prefix="chack_tools_append_",
        )
        if self._serialized_tools_override_names_json:
            env[CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV] = self._serialized_tools_override_names_json
        else:
            env.pop(CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV, None)
        if self._serialized_tools_append_names_json:
            env[CHACK_TOOLS_APPEND_NAMES_JSON_ENV] = self._serialized_tools_append_names_json
        else:
            env.pop(CHACK_TOOLS_APPEND_NAMES_JSON_ENV, None)
        env["CHACK_MODEL_PROVIDER"] = self._model_provider
        env["CHACK_DEFAULT_MODEL"] = self._default_model
        env["CHACK_SOCIAL_NETWORK_MODEL"] = self._social_network_model
        env["CHACK_SCIENTIFIC_MODEL"] = self._scientific_model
        env["CHACK_WEBSEARCHER_MODEL"] = self._websearcher_model
        env["CHACK_BUSINESS_MODEL"] = self._business_model
        env["CHACK_PRODUCT_MODEL"] = self._product_model
        env["CHACK_TRAVEL_MODEL"] = self._travel_model
        env["CHACK_LEGAL_MODEL"] = self._legal_model
        env["CHACK_DATA_STATISTICS_MODEL"] = self._data_statistics_model
        env["CHACK_NEWS_MEDIA_MODEL"] = self._news_media_model
        env["CHACK_KNOWLEDGE_GRAPH_MODEL"] = self._knowledge_graph_model
        env["CHACK_RELIGIOUS_MODEL"] = self._religious_model
        env["CHACK_CLI_MODEL"] = self._cli_model
        env["CHACK_SUBCHACK_MODEL"] = self._subchack_model
        env["CHACK_RESEARCHER_ADMINISTRATOR_MODEL"] = self._researcher_administrator_model
        env["CHACK_SOCIAL_NETWORK_MAX_TURNS"] = str(self._social_network_max_turns)
        env["CHACK_SCIENTIFIC_MAX_TURNS"] = str(self._scientific_max_turns)
        env["CHACK_WEBSEARCHER_MAX_TURNS"] = str(self._websearcher_max_turns)
        env["CHACK_BUSINESS_MAX_TURNS"] = str(self._business_max_turns)
        env["CHACK_PRODUCT_MAX_TURNS"] = str(self._product_max_turns)
        env["CHACK_TRAVEL_MAX_TURNS"] = str(self._travel_max_turns)
        env["CHACK_LEGAL_MAX_TURNS"] = str(self._legal_max_turns)
        env["CHACK_DATA_STATISTICS_MAX_TURNS"] = str(self._data_statistics_max_turns)
        env["CHACK_NEWS_MEDIA_MAX_TURNS"] = str(self._news_media_max_turns)
        env["CHACK_KNOWLEDGE_GRAPH_MAX_TURNS"] = str(self._knowledge_graph_max_turns)
        env["CHACK_RELIGIOUS_MAX_TURNS"] = str(self._religious_max_turns)
        env["CHACK_CLI_MAX_TURNS"] = str(self._cli_max_turns)
        env["CHACK_SUBCHACK_MAX_TURNS"] = str(self._subchack_max_turns)
        env["CHACK_RESEARCHER_ADMINISTRATOR_MAX_TURNS"] = str(self._researcher_administrator_max_turns)
        env["CHACK_MIN_TOOLS_USED"] = str(self._min_tools_used)
        env["CHACK_MAX_TOOLS_USED"] = str(effective_max_tools_used(self._max_tools_used))
        env["CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST"] = (
            "1" if self._require_task_steps_manager_init_first else "0"
        )
        env["CHACK_TASK_SESSION_ID"] = str(current_session_id() or "")
        env["CHACK_RUN_LABEL"] = str(current_run_label() or "Run 1")
        # Child processes export into the same place and join this run's trace.
        env.update(child_process_env())

        if self._uses_gateway_route:
            for name in (
                "CLAUDE_ACCESS_TOKEN",
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CLAUDE_API_KEY",
                *_CLAUDE_GATEWAY_MODEL_ENV,
            ):
                env.pop(name, None)
            env.update(json.loads(self._gateway_env_json or "{}"))
        else:
            env.pop("CLAUDE_ACCESS_TOKEN", None)
            if self._claude_access_token:
                # Claude Code subscription tokens created by `claude setup-token`
                # are consumed through CLAUDE_CODE_OAUTH_TOKEN. Keep accepting the
                # legacy application variable at the configuration boundary, but
                # always invoke the CLI with its canonical environment variable.
                env["CLAUDE_CODE_OAUTH_TOKEN"] = self._claude_access_token
                env.pop("ANTHROPIC_API_KEY", None)
                env.pop("CLAUDE_API_KEY", None)
                env.pop("ANTHROPIC_AUTH_TOKEN", None)
            else:
                env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
                _api_key = str(
                    self._anthropic_api_key
                    or os.environ.get("ANTHROPIC_API_KEY", "")
                    or os.environ.get("CLAUDE_API_KEY", "")
                )
                if _api_key:
                    env["ANTHROPIC_API_KEY"] = _api_key

        # Drive auto-compaction at the configured token budget. This caps the
        # effective context window used for the compaction calculation (it only
        # ever lowers it), so Claude Code summarizes/compacts the conversation
        # when the budget is approached instead of ending the session. Paired
        # with the 1M-context beta above when the budget exceeds 200k.
        if self._max_context_tokens > 0:
            compact_window = int(self._max_context_tokens)
            if self._claude_access_token and not self._uses_gateway_route:
                compact_window = min(compact_window, _CLAUDE_DEFAULT_CONTEXT_WINDOW)
            if self._uses_gateway_route and self._gateway_context_window > 0:
                compact_window = min(compact_window, self._gateway_context_window)
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(compact_window)
            # The window alone uses Claude Code's later default trigger. Honor
            # Chack's configured fraction inside the running CLI conversation.
            ratio = min(0.95, max(0.01, self._compaction_threshold_ratio))
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(round(ratio * 100))

        # Claude Code marks the tools, system prompt and last message with
        # cache_control. A model that cannot cache rejects the whole request.
        if not self._sends_prompt_cache_markers():
            env["DISABLE_PROMPT_CACHING"] = "1"

        # Allow --dangerously-skip-permissions when running as root inside Docker/CI.
        env.setdefault("IS_SANDBOX", "1")

        env.setdefault("AWS_SHARED_CREDENTIALS_FILE", os.environ.get("AWS_SHARED_CREDENTIALS_FILE", ""))
        env.setdefault("AWS_CONFIG_FILE", os.environ.get("AWS_CONFIG_FILE", ""))

        # Claude Code internal tool timeouts -----------------------------------
        # Required local MCP servers can take more than Claude Code's default
        # 5-second connection snapshot on CPU-constrained CI workers, especially
        # when several agents start together and deserialize custom toolsets.
        # Keep startup blocking and use one bounded 60-second window so the
        # first prompt never sees an incomplete tool registry.
        mcp_startup_timeout_ms = (
            os.environ.get("CHACK_CLAUDE_MCP_STARTUP_TIMEOUT_MS", "").strip()
            or os.environ.get("MCP_CONNECT_TIMEOUT_MS", "").strip()
            or os.environ.get("MCP_TIMEOUT", "").strip()
            or "60000"
        )
        env["MCP_CONNECTION_NONBLOCKING"] = "false"
        env["MCP_TIMEOUT"] = mcp_startup_timeout_ms
        env["MCP_CONNECT_TIMEOUT_MS"] = mcp_startup_timeout_ms
        # BASH_DEFAULT_TIMEOUT_MS: default timeout for the Bash tool (default 120s)
        env.setdefault(
            "BASH_DEFAULT_TIMEOUT_MS",
            os.environ.get("CHACK_CLAUDE_BASH_DEFAULT_TIMEOUT_MS", "120000"),
        )
        # BASH_MAX_TIMEOUT_MS: hard cap the model can request for Bash (default 600s)
        env.setdefault(
            "BASH_MAX_TIMEOUT_MS",
            os.environ.get("CHACK_CLAUDE_BASH_MAX_TIMEOUT_MS", "600000"),
        )
        # MCP_TOOL_TIMEOUT: per-tool timeout for MCP calls.
        # Claude Code defaults this to 1e8 ms (~27 hours) which is effectively
        # infinite and can cause the agent to hang. We cap it at 5 minutes.
        env.setdefault(
            "MCP_TOOL_TIMEOUT",
            os.environ.get("CHACK_CLAUDE_MCP_TOOL_TIMEOUT_MS", "300000"),
        )

        return env

    def _set_env_or_file(
        self,
        env: dict[str, str],
        env_key: str,
        value: str,
        *,
        path_env_key: str,
        prefix: str,
    ) -> None:
        raw = str(value or "")
        env.pop(env_key, None)
        env.pop(path_env_key, None)
        if not raw:
            return
        if len(raw) <= CHACK_INLINE_ENV_VALUE_MAX_CHARS:
            env[env_key] = raw
            return
        payload_dir = self._claude_home or os.environ.get("CHACK_CLAUDE_HOME_BASE", "").strip() or ""
        path = write_payload_to_file(raw, prefix=prefix, directory=payload_dir)
        env[path_env_key] = path

    def _ensure_claude_home_and_settings(self) -> None:
        if self._claude_home:
            return

        safe_session = re.sub(
            r"[^A-Za-z0-9._-]",
            "_",
            str(current_session_id() or "default"),
        )
        base = os.path.join(
            os.path.expanduser(
                os.environ.get("CHACK_CLAUDE_HOME_BASE", os.path.expanduser("~/.claude/chack"))
            ),
            safe_session,
        )
        os.makedirs(base, exist_ok=True)
        self._claude_home = base
        self._write_claude_settings(base)

    def _write_claude_settings(self, claude_home: str) -> None:
        settings_path = os.path.join(claude_home, "settings.json")
        mcp_env = self._mcp_env_map()
        mcp_env[_MCP_STARTUP_STATUS_PATH_ENV] = self._mcp_startup_status_path()
        settings_payload = {
            "mcpServers": {
                "chack_tools": {
                    "command": sys.executable,
                    "args": ["-m", "chack_agent.backends.chack_tools_mcp_server"],
                    "env": mcp_env,
                    "alwaysLoad": True,
                }
            }
        }
        if self._playwright_mcp_enabled():
            settings_payload["mcpServers"]["playwright"] = playwright_mcp_server_config()
        if self._fallback_web_mcp_url:
            settings_payload["mcpServers"][FALLBACK_WEB_MCP_SERVER] = {
                "type": "http",
                "url": self._fallback_web_mcp_url,
            }

        self._output_schema = None
        if self._output_schema_json:
            try:
                self._output_schema = json.loads(self._output_schema_json)
            except Exception:
                self._output_schema = None
        # Keep schema available in memory for later reference if needed.

        with open(settings_path, "w", encoding="utf-8") as handle:
            json.dump(settings_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _playwright_mcp_enabled(self) -> bool:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        return bool(cfg.get("playwright_enabled")) and playwright_mcp_is_available()

    def _mcp_env_map(self) -> dict[str, str]:
        env_keys = [
            "CHACK_TOOLS_CONFIG_JSON",
            "CHACK_TOOLS_CONFIG_JSON_PATH",
            "CHACK_ALLOWED_TOOLS_JSON",
            "CHACK_ALLOWED_TOOLS_JSON_PATH",
            "CHACK_TOOLS_OVERRIDE_B64",
            "CHACK_TOOLS_OVERRIDE_B64_PATH",
            "CHACK_TOOLS_APPEND_B64",
            "CHACK_TOOLS_APPEND_B64_PATH",
            "CHACK_TOOLS_OVERRIDE_NAMES_JSON",
            "CHACK_TOOLS_APPEND_NAMES_JSON",
            "CHACK_CHATGPT_ASYNC_API_URL",
            "CHACK_CHATGPT_ASYNC_API_SECRET",
            "PATH",
            "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "NO_PROXY", "no_proxy", "NAXUS_DISCOVERY_PROXY",
            "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
            "NODE_EXTRA_CA_CERTS", "PROXY_STRATEGY", "PROXY_PROTOCOL",
            "PROXY_HOST", "PROXY_PORT",
            "VIRTUAL_ENV",
            "PIP_REQUIRE_VIRTUALENV",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PYTHONNOUSERSITE",
            "DYNAMIC_POC_VIRTUAL_ENV",
            "ANDROID_HOME",
            "ANDROID_SDK_ROOT",
            "ADB_SERVER_SOCKET",
            "ANDROID_SERIAL",
            "DYNAMIC_ANDROID_SERIALS",
            "CHACK_EXEC_DENY_REGEX",
            "CHACK_MODEL_PROVIDER",
            "CHACK_DEFAULT_MODEL",
            "CHACK_SOCIAL_NETWORK_MODEL",
            "CHACK_SCIENTIFIC_MODEL",
            "CHACK_WEBSEARCHER_MODEL",
            "CHACK_BUSINESS_MODEL",
            "CHACK_PRODUCT_MODEL",
            "CHACK_LEGAL_MODEL",
            "CHACK_DATA_STATISTICS_MODEL",
            "CHACK_NEWS_MEDIA_MODEL",
            "CHACK_KNOWLEDGE_GRAPH_MODEL",
            "CHACK_RELIGIOUS_MODEL",
            "CHACK_CLI_MODEL",
            "CHACK_SUBCHACK_MODEL",
            "CHACK_RESEARCHER_ADMINISTRATOR_MODEL",
            "CHACK_SOCIAL_NETWORK_MAX_TURNS",
            "CHACK_SCIENTIFIC_MAX_TURNS",
            "CHACK_WEBSEARCHER_MAX_TURNS",
            "CHACK_BUSINESS_MAX_TURNS",
            "CHACK_PRODUCT_MAX_TURNS",
            "CHACK_LEGAL_MAX_TURNS",
            "CHACK_DATA_STATISTICS_MAX_TURNS",
            "CHACK_NEWS_MEDIA_MAX_TURNS",
            "CHACK_KNOWLEDGE_GRAPH_MAX_TURNS",
            "CHACK_RELIGIOUS_MAX_TURNS",
            "CHACK_CLI_MAX_TURNS",
            "CHACK_SUBCHACK_MAX_TURNS",
            "CHACK_RESEARCHER_ADMINISTRATOR_MAX_TURNS",
            "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST",
            "CHACK_TASK_SESSION_ID",
            "CHACK_RUN_LABEL",
            "CHACK_RESEARCH_MASTER_DIR",
            "CHACK_RESEARCH_DATA_DIR",
            "CHACK_RESEARCH_SAVE_ARTIFACTS",
            "CHACK_MIN_TOOLS_USED",
            "CHACK_MAX_TOOLS_USED",
            "CHACK_EXEC_CWD",
            "AISEC_LOCAL_VULN_STORE_PATH",
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "CODEX_ID_TOKEN",
            "CODEX_REFRESH_TOKEN",
            "CODEX_ACCOUNT_ID",
            "CODEX_LAST_REFRESH",
            "CODEX_HOME",
            "BRAVE_API_KEY",
            "SERPAPI_API_KEY",
            "FORUMSCOUT_API_KEY",
            "FORUMSCOUT_BASE_URL",
            "GH_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_PROFILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
            "AWS_ROLE_SESSION_NAME",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_CPP_USER_PROJECT",
            "AZURE_APP_ID",
            "AZURE_SA_NAME",
            "AZURE_SA_SECRET_VALUE",
            "AZURE_TENANT_ID",
            "STRIPE_API_KEY",
            "CHACK_EXEC_TIMEOUT",
            "CHACK_EXEC_MAX_OUTPUT",
            "CHACK_MCP_TOOL_MAX_TOKENS",
            "CHACK_BUDGET_START_EPOCH",
            "CHACK_BUDGET_MAX_RUNTIME_SECONDS",
            "CHACK_BUDGET_MAX_COST_USD",
            "CHACK_BUDGET_SPENT_USD",
            "CHACK_BUDGET_WARNING_RATIO",
            "CHACK_BUDGET_CRITICAL_RATIO",
            "CHACK_BUDGET_INJECTION_ENABLED",
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_ACCESS_TOKEN",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_CUSTOM_HEADERS",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_NAME",
            "LLM_GATEWAY_TYPE",
            "LLM_GATEWAY_BASE_URL",
            "LLM_GATEWAY_ANTHROPIC_BASE_URL",
            "LLM_GATEWAY_GEMINI_BASE_URL",
            "LLM_GATEWAY_API_KEY",
            "LLM_GATEWAY_HEADERS",
            "LLM_GATEWAY_MODEL_MAP",
            "PYTHONPATH",
            "CHACK_CALLER_PYTHONPATH_JSON",
        ]
        env_keys.extend(OTEL_CHILD_ENV_NAMES)
        env_keys.extend(configured_mcp_env_names(self._tools_config_json))

        src_env = self._build_env()
        env_payload: dict[str, str] = {}
        for key in env_keys:
            value = src_env.get(key)
            if value is None:
                continue
            env_payload[key] = str(value)

        env_payload["CHACK_MCP_PARENT_PID"] = str(os.getpid())
        return env_payload

    @staticmethod
    def _parse_event_line(line: str) -> dict[str, Any] | None:
        text = str(line or "").strip()
        if not text or not text.startswith("{"):
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _log_tool_called(tool_name: str, tool_input: Any) -> None:
        try:
            log_event(
                "tool_called",
                payload={"tool": tool_name, "tool_input": tool_input},
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass

    @staticmethod
    def _sync_task_steps_manager(arguments: dict[str, Any], status: str) -> None:
        try:
            if status.lower() != "success":
                return
            if not isinstance(arguments, dict):
                return
            session_id = str(current_session_id() or "").strip()
            if not session_id:
                return
            run_label = str(current_run_label() or "Run 1")

            raw_task_id = arguments.get("task_id")
            task_id: int | None = None
            if raw_task_id is not None and str(raw_task_id).strip() != "":
                try:
                    task_id = int(raw_task_id)
                except Exception:
                    task_id = None

            TASK_STEPS_STORE.apply(
                session_id=session_id,
                run_label=run_label,
                action=str(arguments.get("action") or "").strip(),
                task_id=task_id,
                text=str(arguments.get("text") or "").strip(),
                status=str(arguments.get("status") or "").strip(),
                tasks_text=str(arguments.get("tasks") or "").strip(),
                notes=str(arguments.get("notes") or "").strip(),
            )
        except Exception:
            pass


def build_executor(
    config: ChackConfig,
    *,
    system_prompt: str,
    max_turns: int,
    memory_max_messages: int,
    memory_reset_to_messages: int,
    memory_summary_max_chars: int = 0,
    tools_override: list[Any] | None = None,
    tools_append: list[Any] | None = None,
) -> ClaudeCodeExecutor:
    gateway_route = get_gateway_route(config)
    if (
        gateway_route is not None
        and gateway_route.is_openrouter
        and not openrouter_selected_by_api_key_type(config)
    ):
        # ``openrouter/`` model names select the Responses API backend; the
        # ``openrouter`` API key type keeps Claude Code on OpenRouter's
        # Anthropic-compatible endpoint.
        from .gateway_responses_backend import build_executor as build_gateway_responses_executor

        return build_gateway_responses_executor(
            clone_config_for_gateway(config, gateway_route),
            system_prompt=system_prompt,
            max_turns=max_turns,
            memory_max_messages=memory_max_messages,
            memory_reset_to_messages=memory_reset_to_messages,
            memory_summary_max_chars=memory_summary_max_chars,
            tools_override=tools_override,
            tools_append=tools_append,
        )
    del max_turns
    try:
        _LOGGER.debug(
            "claude build_executor: memory_summary_max_chars=%s (not used in this backend)",
            int(memory_summary_max_chars),
        )
    except Exception:
        _LOGGER.debug(
            "claude build_executor: memory_summary_max_chars provided (unable to coerce to int in debug log)"
        )

    def _extract_tool_names(items: list[Any] | None) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for tool in items or []:
            name = str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    model_provider = str(config.model.provider or "").strip()
    if not model_provider:
        raise ValueError("model.provider must be defined in config")
    if model_provider not in {"claude", "claude-code", "claude_code"}:
        raise ValueError(
            f"claude backend requires model.provider='claude', 'claude-code', or 'claude_code' (got {model_provider!r})"
        )

    def _build_toolset_kwargs() -> dict[str, Any]:
        init_params = inspect.signature(AgentsToolset.__init__).parameters
        toolset_kwargs: dict[str, Any] = {
            "default_model": config.model.primary,
            "social_network_model": config.model.social_network,
            "scientific_model": config.model.scientific,
            "social_network_max_turns": config.model.social_network_max_turns,
            "scientific_max_turns": config.model.scientific_max_turns,
        }
        if "websearcher_model" in init_params:
            toolset_kwargs["websearcher_model"] = config.model.websearcher
        if "websearcher_max_turns" in init_params:
            toolset_kwargs["websearcher_max_turns"] = config.model.websearcher_max_turns
        if "business_model" in init_params:
            toolset_kwargs["business_model"] = config.model.business
        if "business_max_turns" in init_params:
            toolset_kwargs["business_max_turns"] = config.model.business_max_turns
        if "product_model" in init_params:
            toolset_kwargs["product_model"] = config.model.product
        if "product_max_turns" in init_params:
            toolset_kwargs["product_max_turns"] = config.model.product_max_turns
        if "travel_model" in init_params:
            toolset_kwargs["travel_model"] = config.model.travel
        if "travel_max_turns" in init_params:
            toolset_kwargs["travel_max_turns"] = config.model.travel_max_turns
        if "legal_model" in init_params:
            toolset_kwargs["legal_model"] = config.model.legal
        if "legal_max_turns" in init_params:
            toolset_kwargs["legal_max_turns"] = config.model.legal_max_turns
        if "data_statistics_model" in init_params:
            toolset_kwargs["data_statistics_model"] = config.model.data_statistics
        if "data_statistics_max_turns" in init_params:
            toolset_kwargs["data_statistics_max_turns"] = config.model.data_statistics_max_turns
        if "news_media_model" in init_params:
            toolset_kwargs["news_media_model"] = config.model.news_media
        if "news_media_max_turns" in init_params:
            toolset_kwargs["news_media_max_turns"] = config.model.news_media_max_turns
        if "knowledge_graph_model" in init_params:
            toolset_kwargs["knowledge_graph_model"] = config.model.knowledge_graph
        if "knowledge_graph_max_turns" in init_params:
            toolset_kwargs["knowledge_graph_max_turns"] = config.model.knowledge_graph_max_turns
        if "religious_model" in init_params:
            toolset_kwargs["religious_model"] = config.model.religious
        if "religious_max_turns" in init_params:
            toolset_kwargs["religious_max_turns"] = config.model.religious_max_turns
        if "cli_model" in init_params:
            toolset_kwargs["cli_model"] = config.model.cli
        if "cli_max_turns" in init_params:
            toolset_kwargs["cli_max_turns"] = config.model.cli_max_turns
        if "subchack_model" in init_params:
            toolset_kwargs["subchack_model"] = config.model.subchack
        if "subchack_max_turns" in init_params:
            toolset_kwargs["subchack_max_turns"] = config.model.subchack_max_turns
        if "researcher_administrator_model" in init_params:
            toolset_kwargs["researcher_administrator_model"] = config.model.researcher_administrator
        if "researcher_administrator_max_turns" in init_params:
            toolset_kwargs["researcher_administrator_max_turns"] = config.model.researcher_administrator_max_turns
        if "model_provider" in init_params:
            toolset_kwargs["model_provider"] = model_provider
        if "self_critique_enabled" in init_params:
            toolset_kwargs["self_critique_enabled"] = bool(
                getattr(config.agent, "self_critique_enabled", False)
            )
        if "self_critique_rounds" in init_params:
            toolset_kwargs["self_critique_rounds"] = int(
                getattr(config.agent, "self_critique_rounds", 0) or 0
            )
        return toolset_kwargs

    if tools_override is not None:
        allowed_tool_names = _extract_tool_names(list(tools_override))
    elif tools_append:
        base_toolset = AgentsToolset(config.tools, **_build_toolset_kwargs())
        allowed_tool_names = _extract_tool_names(list(base_toolset.tools) + list(tools_append))
    else:
        base_toolset = AgentsToolset(config.tools, **_build_toolset_kwargs())
        allowed_tool_names = _extract_tool_names(list(base_toolset.tools))

    native_task_planning_backend = (
        "claude"
        if bool(getattr(config.tools, "task_steps_manager_enabled", True))
        else ""
    )
    # Claude Code's TodoWrite/Task* tools produce better native plans. Explicit
    # Chack tool overrides are translated into native-plan prompt guidance too.
    allowed_tool_names = [
        name for name in allowed_tool_names if name != "task_steps_manager"
    ]

    require_task_steps_manager_init_first = False

    configured_claude_path = os.environ.get("CLAUDE_CLI_PATH", "").strip() or "claude"
    claude_cli_path = shutil.which(configured_claude_path) or configured_claude_path
    serialized_tools_override_b64 = ""
    serialized_tools_append_b64 = ""
    serialized_tools_override_names_json = ""
    serialized_tools_append_names_json = ""
    try:
        serialized_tools_override_b64 = serialize_tools_payload(tools_override)
    except Exception as exc:
        override_names = _extract_tool_names(tools_override)
        management_names = {
            "run_researchers_batch", "start_researchers_async", "list_researcher_jobs",
            "get_researcher_task", "poll_researchers_async", "get_researcher_result",
            "cancel_researcher_task", "retry_researcher_task", "cancel_researchers_async",
        }
        if not (set(override_names) & management_names):
            raise
        serialized_tools_override_names_json = json.dumps(override_names, ensure_ascii=False)
        _LOGGER.info("Using MCP name-based reconstruction for researcher administrator tools: %s", type(exc).__name__)
    serialized_tools_append_b64 = serialize_tools_payload(tools_append)

    route = gateway_route
    refusal_model, refusal_env_json = _gateway_refusal_fallback(config, route) if route is not None else ("", "")
    native_web_search = web_search_enabled(config) and (route is None or route.is_openrouter)

    return ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _model_name=str(route.model_name if route is not None else config.model.primary),
        _max_turns=int(config.session.max_turns or 100),
        _claude_cli_path=claude_cli_path,
        _tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False),
        _allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False),
        _serialized_tools_override_b64=serialized_tools_override_b64,
        _serialized_tools_append_b64=serialized_tools_append_b64,
        _serialized_tools_override_names_json=serialized_tools_override_names_json,
        _serialized_tools_append_names_json=serialized_tools_append_names_json,
        _model_provider=model_provider,
        _default_model=str(config.model.primary or ""),
        _social_network_model=str(config.model.social_network or ""),
        _scientific_model=str(config.model.scientific or ""),
        _websearcher_model=str(config.model.websearcher or ""),
        _business_model=str(config.model.business or ""),
        _product_model=str(config.model.product or ""),
        _travel_model=str(config.model.travel or ""),
        _legal_model=str(config.model.legal or ""),
        _data_statistics_model=str(config.model.data_statistics or ""),
        _news_media_model=str(config.model.news_media or ""),
        _knowledge_graph_model=str(config.model.knowledge_graph or ""),
        _religious_model=str(config.model.religious or ""),
        _cli_model=str(config.model.cli or ""),
        _subchack_model=str(config.model.subchack or ""),
        _researcher_administrator_model=str(config.model.researcher_administrator or ""),
        _social_network_max_turns=int(config.model.social_network_max_turns or 30),
        _scientific_max_turns=int(config.model.scientific_max_turns or 30),
        _websearcher_max_turns=int(config.model.websearcher_max_turns or 30),
        _business_max_turns=int(config.model.business_max_turns or 30),
        _product_max_turns=int(config.model.product_max_turns or 30),
        _travel_max_turns=int(config.model.travel_max_turns or 40),
        _legal_max_turns=int(config.model.legal_max_turns or 30),
        _data_statistics_max_turns=int(config.model.data_statistics_max_turns or 30),
        _news_media_max_turns=int(config.model.news_media_max_turns or 30),
        _knowledge_graph_max_turns=int(config.model.knowledge_graph_max_turns or 30),
        _religious_max_turns=int(config.model.religious_max_turns or 30),
        _cli_max_turns=int(config.model.cli_max_turns or 30),
        _subchack_max_turns=int(config.model.subchack_max_turns or 30),
        _researcher_administrator_max_turns=int(config.model.researcher_administrator_max_turns or 100),
        _min_tools_used=max(0, int(config.tools.min_tools_used or 0)),
        _max_tools_used=max(0, int(config.tools.max_tools_used or 0)),
        _require_task_steps_manager_init_first=require_task_steps_manager_init_first,
        _output_schema_json=(
            json.dumps(config.agent.output_schema_json, ensure_ascii=False)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        _native_task_planning_backend=native_task_planning_backend,
        _require_native_plan_first=bool(
            native_task_planning_backend
            and getattr(config.agent, "require_task_steps_manager_init_first", True)
        ),
        _output_schema_name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
        _output_schema_strict=bool(getattr(config.agent, "output_schema_strict", True)),
        _max_context_tokens=int(getattr(config.model, "max_context_tokens", 0) or 0),
        _prompt_cache_enabled=bool(getattr(config.model, "prompt_cache_enabled", True)),
        _compaction_threshold_ratio=float(
            getattr(config.agent, "compaction_threshold_ratio", 0.50) or 0.50
        ),
        _thinking_effort=normalize_thinking_effort(config.agent.thinking_effort),
        _uses_gateway_route=route is not None,
        _native_web_search=native_web_search,
        _fallback_web_mcp_url=(
            fallback_web_mcp_url()
            if keyless_web_search_enabled(config) and not native_web_search
            else ""
        ),
        _anthropic_api_key=str(
            ""
            if route is not None
            else (
                os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("CLAUDE_API_KEY", "")
            )
        ),
        _claude_access_token=str(
            ""
            if route is not None
            else (
                os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
                or os.environ.get("CLAUDE_ACCESS_TOKEN", "")
            )
        ),
        _gateway_env_json=(
            json.dumps(_claude_gateway_env(config, route), sort_keys=True)
            if route is not None
            else ""
        ),
        _gateway_context_window=int(route.context_window if route is not None else 0),
        _gateway_refusal_model=refusal_model,
        _gateway_refusal_env_json=refusal_env_json,
    )
