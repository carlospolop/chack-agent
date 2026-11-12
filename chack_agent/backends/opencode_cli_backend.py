"""Run Chack agents through the opencode CLI (``opencode run``).

opencode (https://opencode.ai, npm package ``opencode-ai``) is a provider-neutral
coding agent. Chack writes a private opencode config per session that defines:

* one provider, ``chack``, whose AI SDK package and endpoint follow the API key
  type: OpenAI (Responses API), Anthropic (Messages API), OpenRouter, or a
  custom/LiteLLM gateway (Responses, Messages or Chat Completions depending on
  the model family);
* one primary agent, ``chack``, carrying the system prompt, the tool policy,
  the output contract and ``steps`` (= max turns);
* the Chack tool MCP server, ``chack_tools``.

Each invocation runs ``opencode run --format json`` with the prompt on stdin
and reads the JSON event stream for the answer, tool calls and token usage.
Later invocations resume the same opencode session.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.cancellation import cancellation_requested, register_process, unregister_process
from chack_tools.task_steps_manager_state import (
    STORE as TASK_STEPS_STORE,
    current_run_label,
    current_session_id,
)
from chack_tools.telemetry import child_process_env, log_event
from chack_tools.tool_usage_state import effective_max_tools_used

from ..config import ChackConfig, resolve_api_key_type
from ..live_cost_state import report_live_usage
from ..llm_gateway import GatewayRoute, gateway_subprocess_env, get_gateway_route
from ..output_schema import repair_schema_output
from ..resume_compaction import ResumeCompactionResult
from ..thinking_effort import (
    OPENCODE_SDK_ANTHROPIC,
    OPENCODE_SDK_OPENAI,
    OPENCODE_SDK_OPENAI_COMPATIBLE,
    OPENCODE_SDK_OPENROUTER,
    normalize_thinking_effort,
    opencode_thinking_effort,
)
from .claude_code_backend import _terminate_process_tree
from .cli_process import cli_exec_timeout_seconds
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .prompt_cache import split_prompt_cache_breakpoint
from .tool_payloads import (
    CHACK_ALLOWED_TOOLS_JSON_PATH_ENV,
    CHACK_INLINE_ENV_VALUE_MAX_CHARS,
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_APPEND_B64_PATH_ENV,
    CHACK_TOOLS_APPEND_NAMES_JSON_ENV,
    CHACK_TOOLS_CONFIG_JSON_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    CHACK_TOOLS_OVERRIDE_B64_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV,
    augment_subprocess_pythonpath,
    serialize_tools_payload,
    write_payload_to_file,
)

_LOGGER = logging.getLogger("chack.opencode_cli_backend")

OPENCODE_EXEC_TIMEOUT_ENV = "CHACK_OPENCODE_EXEC_TIMEOUT_SECONDS"

OPENCODE_PROVIDER_ID = "chack"
OPENCODE_AGENT_NAME = "chack"
OPENCODE_MCP_SERVER = "chack_tools"
# opencode names MCP tools ``<server>_<tool>``.
OPENCODE_MCP_TOOL_PREFIX = f"{OPENCODE_MCP_SERVER}_"
OPENCODE_API_KEY_ENV = "CHACK_OPENCODE_API_KEY"
_MCP_STARTUP_STATUS_PATH_ENV = "CHACK_MCP_STARTUP_STATUS_PATH"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_DEFAULT_CONTEXT_WINDOW = 200_000
# opencode caps a request's output at 32k tokens unless
# OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX raises it.
_DEFAULT_OUTPUT_LIMIT = 32_000
_XDG_DIRS = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")
# Inherited settings that would replace the config Chack writes.
_INHERITED_OPENCODE_OVERRIDES = (
    "OPENCODE_CONFIG_CONTENT",
    "OPENCODE_PERMISSION",
    "OPENCODE_CONFIG_DIR",
    "OPENCODE_AUTH_CONTENT",
)
# Native tools that make no sense in a non-interactive Chack run: nobody can
# answer ``question``; ``task`` subagents and ``skill`` run outside Chack's
# accounting; Chack plans with task_steps_manager instead of ``todowrite``.
# Chack's ``exec`` tool is the run's only shell, so the native ``bash`` is
# always off: exec-disabled runs get no shell at all, and the rest keep the
# exec limits (timeout, output cap, tool-call budget) the prompts rely on.
_DENIED_NATIVE_TOOLS = ("question", "task", "skill", "todowrite", "todoread", "bash")
# Host file access, only for runs that also have Chack's ``exec``.
_NATIVE_FILE_TOOLS = ("read", "edit", "write", "patch", "apply_patch", "glob", "grep", "list", "lsp", "codesearch")
_OPENAI_FAMILY_RE = re.compile(r"^(?:gpt|o\d|codex|chatgpt)")
# opencode 2 runs ``opencode run`` on a background service shared by every run
# on one fixed port, so concurrent runs fight over it; ``--standalone`` gives
# each run a private server. opencode 1 always used one and has no such flag.
_STANDALONE_FLAG = "--standalone"
_HELP_TIMEOUT_SECONDS = 60

_ROLE_MODEL_FIELDS = (
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
_ROLE_DEFAULT_MAX_TURNS = {"travel": 40, "researcher_administrator": 100}


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]
    time_to_first_token_seconds: float | None = None
    time_to_first_token_source: str = "unavailable"


@dataclass(frozen=True)
class OpencodeProvider:
    """Where opencode sends model requests, and how it authenticates."""

    key_type: str
    sdk: str
    base_url: str
    api_key: str
    model_id: str
    headers: dict[str, str] = field(default_factory=dict)
    context_window: int = 0
    # Extra environment so the Chack MCP server (and its sub-agents) reach the
    # same gateway.
    subprocess_env: dict[str, str] = field(default_factory=dict)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _model_family_sdk(model_name: str, upstream: str = "") -> str:
    """AI SDK package that speaks a gateway model's native wire format."""

    for candidate in (upstream, model_name):
        bare = _clean(candidate).lower().rsplit("/", 1)[-1]
        bare = re.sub(r"^(?:[a-z]{2,6}\.)?(?:anthropic|openai)\.", "", bare)
        if "claude" in bare:
            return OPENCODE_SDK_ANTHROPIC
        if _OPENAI_FAMILY_RE.match(bare):
            return OPENCODE_SDK_OPENAI
    return OPENCODE_SDK_OPENAI_COMPATIBLE


def _anthropic_api_base(root_url: str) -> str:
    root = _clean(root_url).rstrip("/")
    return root if root.endswith("/v1") else f"{root}/v1"


def resolve_opencode_provider(config: ChackConfig) -> OpencodeProvider:
    """Provider settings for the configured API key type and model."""

    route: GatewayRoute | None = get_gateway_route(config)
    if route is None and not _clean(config.model.api_key_type):
        # Without an explicit key type, opencode uses whichever key is present;
        # an OpenRouter or gateway key alone selects that route.
        inferred = resolve_api_key_type(config)
        if inferred in {"openrouter", "gateway"}:
            route = get_gateway_route(replace(config, model=replace(config.model, api_key_type=inferred)))
    if route is not None:
        extra_env = gateway_subprocess_env(config, route)
        if route.is_openrouter:
            return OpencodeProvider(
                key_type="openrouter",
                sdk=OPENCODE_SDK_OPENROUTER,
                base_url=route.base_url,
                api_key=route.api_key,
                model_id=route.model_name,
                headers=dict(route.headers),
                context_window=route.context_window,
                subprocess_env=extra_env,
            )
        sdk = _model_family_sdk(route.model_name, route.upstream)
        return OpencodeProvider(
            key_type="gateway",
            sdk=sdk,
            base_url=(
                _anthropic_api_base(route.anthropic_base_url)
                if sdk == OPENCODE_SDK_ANTHROPIC
                else route.base_url
            ),
            api_key=route.api_key,
            model_id=route.model_name,
            headers=dict(route.headers),
            context_window=route.context_window,
            subprocess_env=extra_env,
        )

    key_type = resolve_api_key_type(config)
    model_id = _clean(config.model.primary)
    credentials = config.credentials
    if key_type == "anthropic":
        api_key = (
            _clean(getattr(credentials, "anthropic_api_key", ""))
            or _clean(getattr(credentials, "claude_api_key", ""))
            or _clean(os.environ.get("ANTHROPIC_API_KEY"))
            or _clean(os.environ.get("CLAUDE_API_KEY"))
        )
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for the opencode backend with api_key_type=anthropic "
                "(opencode cannot use a Claude Code subscription token)"
            )
        base_url = _clean(os.environ.get("ANTHROPIC_BASE_URL"))
        return OpencodeProvider(
            key_type="anthropic",
            sdk=OPENCODE_SDK_ANTHROPIC,
            base_url=_anthropic_api_base(base_url) if base_url else _ANTHROPIC_DEFAULT_BASE_URL,
            api_key=api_key,
            model_id=model_id,
        )
    if key_type in {"openai", "codex_token"}:
        api_key = _clean(getattr(credentials, "openai_api_key", "")) or _clean(
            os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for the opencode backend with api_key_type=openai "
                "(opencode cannot use a Codex/ChatGPT access token)"
            )
        return OpencodeProvider(
            key_type="openai",
            sdk=OPENCODE_SDK_OPENAI,
            base_url=_clean(os.environ.get("OPENAI_BASE_URL")).rstrip("/") or _OPENAI_DEFAULT_BASE_URL,
            api_key=api_key,
            model_id=model_id,
        )
    raise ValueError(
        f"The opencode backend supports api_key_type openai, anthropic, openrouter or gateway (got {key_type!r})"
    )


@functools.lru_cache(maxsize=None)
def _run_help_offers_standalone(cli_path: str, modified_ns: int) -> bool:
    del modified_ns  # part of the cache key: a reinstalled CLI is checked again
    try:
        done = subprocess.run(
            [cli_path, "run", "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_HELP_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _LOGGER.warning("Could not read `%s run --help` to check for %s: %s", cli_path, _STANDALONE_FLAG, exc)
        return False
    return _STANDALONE_FLAG in f"{done.stdout}\n{done.stderr}"


def opencode_supports_standalone(cli_path: str) -> bool:
    """True when ``opencode run`` takes ``--standalone`` (opencode 2 and later)."""

    try:
        modified_ns = os.stat(cli_path).st_mtime_ns
    except OSError:
        # A missing CLI is reported when the run launches it.
        return False
    return _run_help_offers_standalone(cli_path, modified_ns)


def _sanitize_session_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", _clean(session_id)) or "default"


def _usage_from_tokens(tokens: Any) -> dict[str, Any] | None:
    """Chack usage for one opencode step.

    opencode reports ``input`` without cache reads and writes, and ``output``
    without reasoning; Chack's usage counts the whole prompt as input and
    reasoning as output, as every provider bills it.
    """

    if not isinstance(tokens, dict):
        return None

    def _int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    cache_read = _int(cache.get("read"))
    cache_write = _int(cache.get("write"))
    prompt = _int(tokens.get("input")) + cache_read + cache_write
    completion = _int(tokens.get("output")) + _int(tokens.get("reasoning"))
    if not prompt and not completion:
        return None
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "input_tokens_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
    }


class OpencodeCliExecutor:
    def __init__(
        self,
        *,
        provider: OpencodeProvider,
        base_system_prompt: str,
        max_turns: int,
        opencode_cli_path: str,
        tools_config_json: str,
        allowed_tools_json: str,
        serialized_tools_override_b64: str = "",
        serialized_tools_append_b64: str = "",
        serialized_tools_override_names_json: str = "",
        serialized_tools_append_names_json: str = "",
        model_provider: str = "opencode",
        role_env: dict[str, str] | None = None,
        min_tools_used: int = 0,
        max_tools_used: int = 0,
        require_task_steps_manager_init_first: bool = False,
        output_schema_json: str = "",
        output_schema_name: str = "output_schema",
        output_schema_strict: bool = True,
        thinking_effort: str = "high",
        max_context_tokens: int = 0,
        memory_max_messages: int = 0,
        memory_reset_to_messages: int = 0,
    ) -> None:
        self._provider = provider
        self._model_name = provider.model_id
        self._base_system_prompt = str(base_system_prompt or "")
        self._max_turns = max(0, int(max_turns or 0))
        self._opencode_cli_path = opencode_cli_path
        self._tools_config_json = tools_config_json
        self._allowed_tools_json = allowed_tools_json
        self._serialized_tools_override_b64 = str(serialized_tools_override_b64 or "")
        self._serialized_tools_append_b64 = str(serialized_tools_append_b64 or "")
        self._serialized_tools_override_names_json = str(serialized_tools_override_names_json or "")
        self._serialized_tools_append_names_json = str(serialized_tools_append_names_json or "")
        self._model_provider = model_provider
        self._role_env = dict(role_env or {})
        self._min_tools_used = max(0, int(min_tools_used or 0))
        self._max_tools_used = max(0, int(max_tools_used or 0))
        self._require_task_steps_manager_init_first = bool(require_task_steps_manager_init_first)
        self._output_schema_json = str(output_schema_json or "")
        self._output_schema_name = str(output_schema_name or "output_schema")
        self._output_schema_strict = bool(output_schema_strict)
        self._thinking_effort = normalize_thinking_effort(thinking_effort)
        self._max_context_tokens = max(0, int(max_context_tokens or 0))
        self._memory_limit = max(0, int(memory_max_messages or 0))
        self._memory_reset_to = max(0, int(memory_reset_to_messages or 0))

        self._conversation: list[dict[str, Any]] = []
        self._home: str | None = None
        self._session_id: str | None = None
        self._agent_prompt = ""
        self._prompt_only_next_invocation = False

    # ------------------------------------------------------------------
    # Executor interface

    def suppress_system_prompt_for_next_invocation(self) -> None:
        self._prompt_only_next_invocation = True

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_opencode(prompt)
        # The schema is only a prompt contract here. A drifting answer gets one
        # repair turn in the same session; the agent prompt stays unchanged,
        # so the provider's prompt cache still applies.
        output, repair_steps, repair_raw = repair_schema_output(
            output,
            schema=self._output_schema_json,
            name=self._output_schema_name,
            strict=self._output_schema_strict,
            resumable=bool(self._session_id),
            run_turn=self._run_opencode,
            label="opencode",
        )
        steps.extend(repair_steps)
        raw_result.raw_responses.extend(repair_raw)

        if user_input:
            self._conversation.append({"role": "user", "content": user_input})
        if output:
            self._conversation.append({"role": "assistant", "content": output})
        if self._memory_limit and len(self._conversation) > self._memory_limit:
            reset_to = min(self._memory_reset_to or self._memory_limit, self._memory_limit)
            self._conversation = self._conversation[-max(1, reset_to):]

        return {"output": output, "intermediate_steps": steps, "raw_result": raw_result}

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)

    def compact_for_resume(self, focus_instructions: str = "") -> ResumeCompactionResult:
        # opencode compacts a session on its own when it nears the model's
        # context limit and has no non-interactive compaction command, so a
        # resumed session is continued as is.
        del focus_instructions
        return ResumeCompactionResult(backend="opencode", method="automatic")

    # ------------------------------------------------------------------
    # Prompt

    def _allowed_tool_names(self) -> list[str]:
        try:
            parsed = json.loads(self._allowed_tools_json or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
        names: list[str] = []
        for raw in parsed if isinstance(parsed, list) else []:
            name = _clean(raw)
            if name and name not in names:
                names.append(name)
        return names

    def _compose_prompt(self, user_input: str) -> str:
        if self._prompt_only_next_invocation:
            # The resumed session keeps the agent prompt it already has.
            self._prompt_only_next_invocation = False
            return str(user_input or "")

        policy_lines: list[str] = []
        tool_names = self._allowed_tool_names()
        if tool_names:
            mappings = ", ".join(f"`{name}` -> `{OPENCODE_MCP_TOOL_PREFIX}{name}`" for name in tool_names)
            policy_lines.append(
                "- The configured custom tools come from the `chack_tools` MCP server. "
                f"Call them by their prefixed names: {mappings}."
            )
        if self._require_task_steps_manager_init_first:
            policy_lines.append(
                f"- First, call `{OPENCODE_MCP_TOOL_PREFIX}task_steps_manager` with action=init "
                "before any other tool call."
            )
        if self._min_tools_used > 0:
            policy_lines.append(f"- Use at least {self._min_tools_used} non-task tool calls before finalizing.")
        if self._max_tools_used > 0:
            policy_lines.append(f"- Do not exceed {self._max_tools_used} non-task tool calls in total.")

        schema_lines: list[str] = []
        if self._output_schema_json:
            schema_lines.extend(
                [
                    "### OUTPUT CONTRACT\nReturn JSON only, exactly one JSON object.",
                    f"Use schema name: {self._output_schema_name}",
                    "Schema:",
                    self._output_schema_json,
                    "Your response must strictly match the JSON schema."
                    if self._output_schema_strict
                    else "Match the schema as closely as possible.",
                ]
            )

        policy_block = "### TOOL USAGE POLICY\n" + "\n".join(policy_lines) if policy_lines else ""
        schema_block = "\n".join(schema_lines)

        # opencode sends the agent prompt as the system message of every
        # request; keeping the stable part there and the changing task in the
        # user message keeps the provider's prompt cache warm.
        cache_parts = split_prompt_cache_breakpoint(user_input)
        stable_prefix = cache_parts.stable_prefix if cache_parts.has_breakpoint else ""
        user_message = cache_parts.dynamic_suffix if cache_parts.has_breakpoint else str(user_input or "")
        self._agent_prompt = "\n\n".join(
            part.strip()
            for part in (self._base_system_prompt, policy_block, schema_block, stable_prefix)
            if part and part.strip()
        )
        return user_message

    # ------------------------------------------------------------------
    # Files and environment

    def _ensure_home(self) -> str:
        if self._home:
            return self._home
        base = os.path.expanduser(
            os.environ.get("CHACK_OPENCODE_HOME_BASE", "").strip() or "~/.local/share/chack/opencode"
        )
        home = os.path.join(base, _sanitize_session_id(current_session_id() or "default"))
        for sub in ("config", "data", "cache", "state"):
            os.makedirs(os.path.join(home, sub), exist_ok=True)
        self._home = home
        return home

    def _config_path(self) -> str:
        return os.path.join(self._ensure_home(), "opencode.json")

    def _mcp_startup_status_path(self) -> str:
        return os.path.join(self._ensure_home(), "mcp_startup_status.json")

    def _web_search_enabled(self) -> bool:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        return bool(cfg.get("native_web_search_enabled", True)) if isinstance(cfg, dict) else True

    def _exa_search_enabled(self) -> bool:
        # opencode's websearch runs on Exa's keyless hosted search.
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        keyless = bool(cfg.get("keyless_web_search_enabled", True)) if isinstance(cfg, dict) else True
        return self._web_search_enabled() and keyless

    def _playwright_mcp_enabled(self) -> bool:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        return bool(isinstance(cfg, dict) and cfg.get("playwright_enabled")) and playwright_mcp_is_available()

    def _header_env(self) -> dict[str, str]:
        """Header name -> variable holding its value, so no value is written to disk."""

        return {
            name: f"CHACK_OPENCODE_HEADER_{index}"
            for index, name in enumerate(sorted(self._provider.headers))
        }

    def _context_limit(self) -> int:
        window = self._provider.context_window or _DEFAULT_CONTEXT_WINDOW
        if self._max_context_tokens > 0:
            # opencode compacts when a session nears the model's context limit,
            # so the configured budget becomes the limit it compacts against.
            window = min(window, self._max_context_tokens) if self._provider.context_window else self._max_context_tokens
        return window

    def build_opencode_config(self) -> dict[str, Any]:
        provider = self._provider
        options: dict[str, Any] = {
            "baseURL": provider.base_url,
            "apiKey": "{env:" + OPENCODE_API_KEY_ENV + "}",
        }
        headers = {name: "{env:" + env + "}" for name, env in self._header_env().items()}
        if provider.sdk == OPENCODE_SDK_ANTHROPIC and provider.key_type == "gateway":
            # Gateways authenticate the Messages API with a bearer token, as
            # Claude Code's ANTHROPIC_AUTH_TOKEN does.
            headers.setdefault("Authorization", "Bearer {env:" + OPENCODE_API_KEY_ENV + "}")
        if headers:
            options["headers"] = headers
        model_entry: dict[str, Any] = {
            "id": provider.model_id,
            "name": provider.model_id,
            "tool_call": True,
            "reasoning": self._thinking_effort != "none",
            "limit": {"context": self._context_limit(), "output": _DEFAULT_OUTPUT_LIMIT},
            "options": opencode_thinking_effort(self._thinking_effort, provider.sdk),
        }
        permission: dict[str, Any] = {"*": "allow"}
        for tool in _DENIED_NATIVE_TOOLS:
            permission[tool] = "deny"
        if "exec" not in self._allowed_tool_names():
            for tool in _NATIVE_FILE_TOOLS:
                permission[tool] = "deny"
        if not self._web_search_enabled():
            permission["webfetch"] = "deny"
        if not self._exa_search_enabled():
            permission["websearch"] = "deny"
        agent: dict[str, Any] = {
            "mode": "primary",
            "description": "Chack agent",
            "prompt": self._agent_prompt,
            "permission": permission,
        }
        if self._max_turns > 0:
            agent["steps"] = self._max_turns
        model_ref = f"{OPENCODE_PROVIDER_ID}/{provider.model_id}"
        mcp: dict[str, Any] = {}
        if self._allowed_tool_names():
            mcp[OPENCODE_MCP_SERVER] = {
                "type": "local",
                "command": [sys.executable, "-m", "chack_agent.backends.chack_tools_mcp_server"],
                "environment": self._mcp_environment(),
                "enabled": True,
                "timeout": int(os.environ.get("CHACK_OPENCODE_MCP_STARTUP_TIMEOUT_MS", "") or "60000"),
            }
        if self._playwright_mcp_enabled():
            pw = playwright_mcp_server_config()
            mcp["playwright"] = {
                "type": "local",
                "command": [pw["command"], *pw.get("args", [])],
                "enabled": True,
                "timeout": int(os.environ.get("CHACK_OPENCODE_MCP_STARTUP_TIMEOUT_MS", "") or "60000"),
            }
        config: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "snapshot": False,
            "enabled_providers": [OPENCODE_PROVIDER_ID],
            "provider": {
                OPENCODE_PROVIDER_ID: {
                    "npm": provider.sdk,
                    "name": "Chack",
                    "options": options,
                    "models": {provider.model_id: model_entry},
                }
            },
            "model": model_ref,
            "small_model": model_ref,
            "default_agent": OPENCODE_AGENT_NAME,
            "agent": {OPENCODE_AGENT_NAME: agent},
            "permission": permission,
        }
        if mcp:
            config["mcp"] = mcp
        return config

    def _write_config(self) -> str:
        path = self._config_path()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.build_opencode_config(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def _mcp_environment(self) -> dict[str, str]:
        """Values the Chack MCP server needs besides what it inherits from opencode.

        opencode starts local MCP servers with its own environment, which
        already carries the Chack tool payloads and credentials; only the
        values that differ for the MCP server are set here, so no secret is
        written to the config file.
        """

        env = {
            "CHACK_MCP_PARENT_PID": str(os.getpid()),
            _MCP_STARTUP_STATUS_PATH_ENV: self._mcp_startup_status_path(),
        }
        # opencode runs with private XDG directories; tools started by the MCP
        # server see the caller's.
        for name in _XDG_DIRS:
            env[name] = os.environ.get(name, "")
        return env

    def _set_env_or_file(self, env: dict[str, str], key: str, value: str, *, path_key: str, prefix: str) -> None:
        raw = str(value or "")
        env.pop(key, None)
        env.pop(path_key, None)
        if not raw:
            return
        if len(raw) <= CHACK_INLINE_ENV_VALUE_MAX_CHARS:
            env[key] = raw
            return
        # Large payloads go to files: a single environment string is limited
        # to 128 KiB on Linux.
        env[path_key] = write_payload_to_file(raw, prefix=prefix, directory=self._ensure_home())

    def _build_env(self) -> dict[str, str]:
        home = self._ensure_home()
        env = {k: v for k, v in os.environ.items() if v is not None}
        augment_subprocess_pythonpath(env)
        env["PYTHONUNBUFFERED"] = "1"
        for name in _INHERITED_OPENCODE_OVERRIDES:
            env.pop(name, None)

        env["OPENCODE_CONFIG"] = self._config_path()
        env["XDG_CONFIG_HOME"] = os.path.join(home, "config")
        env["XDG_DATA_HOME"] = os.path.join(home, "data")
        env["XDG_CACHE_HOME"] = os.path.join(home, "cache")
        env["XDG_STATE_HOME"] = os.path.join(home, "state")
        for flag in (
            "OPENCODE_DISABLE_CLAUDE_CODE",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS",
            "OPENCODE_DISABLE_PROJECT_CONFIG",
            "OPENCODE_DISABLE_AUTOUPDATE",
            "OPENCODE_DISABLE_MODELS_FETCH",
            "OPENCODE_DISABLE_LSP_DOWNLOAD",
        ):
            env[flag] = "1"
        if self._exa_search_enabled():
            # opencode's websearch uses Exa's hosted search, which needs no
            # account, so it works with every provider and key.
            env["OPENCODE_ENABLE_EXA"] = "1"
        env.setdefault(
            "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS",
            os.environ.get("CHACK_OPENCODE_BASH_DEFAULT_TIMEOUT_MS", "120000"),
        )

        env[OPENCODE_API_KEY_ENV] = self._provider.api_key
        for name, env_name in self._header_env().items():
            env[env_name] = self._provider.headers[name]
        env.update(self._provider.subprocess_env)

        self._set_env_or_file(
            env, "CHACK_TOOLS_CONFIG_JSON", self._tools_config_json,
            path_key=CHACK_TOOLS_CONFIG_JSON_PATH_ENV, prefix="chack_tools_config_",
        )
        self._set_env_or_file(
            env, "CHACK_ALLOWED_TOOLS_JSON", self._allowed_tools_json,
            path_key=CHACK_ALLOWED_TOOLS_JSON_PATH_ENV, prefix="chack_allowed_tools_",
        )
        self._set_env_or_file(
            env, CHACK_TOOLS_OVERRIDE_B64_ENV, self._serialized_tools_override_b64,
            path_key=CHACK_TOOLS_OVERRIDE_B64_PATH_ENV, prefix="chack_tools_override_",
        )
        self._set_env_or_file(
            env, CHACK_TOOLS_APPEND_B64_ENV, self._serialized_tools_append_b64,
            path_key=CHACK_TOOLS_APPEND_B64_PATH_ENV, prefix="chack_tools_append_",
        )
        for key, value in (
            (CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV, self._serialized_tools_override_names_json),
            (CHACK_TOOLS_APPEND_NAMES_JSON_ENV, self._serialized_tools_append_names_json),
        ):
            if value:
                env[key] = value
            else:
                env.pop(key, None)

        env.update(self._role_env)
        env["CHACK_MODEL_PROVIDER"] = self._model_provider
        env["CHACK_MIN_TOOLS_USED"] = str(self._min_tools_used)
        env["CHACK_MAX_TOOLS_USED"] = str(effective_max_tools_used(self._max_tools_used))
        env["CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST"] = (
            "1" if self._require_task_steps_manager_init_first else "0"
        )
        env["CHACK_TASK_SESSION_ID"] = str(current_session_id() or "")
        env["CHACK_RUN_LABEL"] = str(current_run_label() or "Run 1")
        # Export settings and the run's trace context, so the Chack MCP
        # server's tool spans join this run's trace.
        env.update(child_process_env())
        return env

    def _build_command(self) -> list[str]:
        command = [
            self._opencode_cli_path,
            "run",
            *([_STANDALONE_FLAG] if opencode_supports_standalone(self._opencode_cli_path) else []),
            "--format",
            "json",
            "--model",
            f"{OPENCODE_PROVIDER_ID}/{self._provider.model_id}",
            "--agent",
            OPENCODE_AGENT_NAME,
            # A fixed title skips opencode's extra title-generation request.
            "--title",
            "chack",
        ]
        if self._session_id:
            command.extend(["--session", self._session_id])
        return command

    # ------------------------------------------------------------------
    # Run

    def _run_opencode(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_home()
        self._write_config()
        status_path = self._mcp_startup_status_path()
        try:
            os.remove(status_path)
        except FileNotFoundError:
            pass
        command = self._build_command()
        env = self._build_env()
        exec_cwd = _clean(env.get("CHACK_EXEC_CWD")) or None
        timeout_seconds = cli_exec_timeout_seconds(OPENCODE_EXEC_TIMEOUT_ENV)
        _LOGGER.info(
            "Starting opencode process: model=%s timeout_seconds=%s session_id=%s cwd=%s",
            self._model_name,
            timeout_seconds,
            self._session_id or "",
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
                "ERROR: opencode CLI executable was not found. "
                f"Configured path: {self._opencode_cli_path!r}. "
                "Install it with `npm i -g opencode-ai` or set OPENCODE_CLI_PATH to the executable.",
                [],
                _RawResult(raw_responses=[]),
            )
        except Exception as exc:
            return (
                f"ERROR: Failed to launch opencode CLI: {type(exc).__name__}: {exc}",
                [],
                _RawResult(raw_responses=[]),
            )

        cancel_registration = register_process(process, _terminate_process_tree)
        steps: list[tuple[ToolAction, Any]] = []
        raw_lines: list[str] = []
        raw_responses: list[Any] = []
        step_texts: list[str] = []
        answer_texts: list[str] = []
        error_text = ""
        started_at = time.monotonic()
        time_to_first_token: float | None = None
        try:
            # The prompt goes through stdin: argv strings are limited to 128 KiB.
            if process.stdin is not None:
                try:
                    process.stdin.write(prompt)
                    process.stdin.close()
                except Exception as exc:
                    _terminate_process_tree(process)
                    return (
                        f"ERROR: Failed to send prompt to opencode CLI: {type(exc).__name__}: {exc}",
                        steps,
                        _RawResult(raw_responses=raw_responses),
                    )
            while True:
                if cancellation_requested():
                    _terminate_process_tree(process)
                    return "ERROR: opencode execution cancelled.", steps, _RawResult(raw_responses=raw_responses)
                if timeout_seconds > 0 and time.monotonic() - started_at > timeout_seconds:
                    _terminate_process_tree(process)
                    return (
                        f"ERROR: opencode execution timed out after {timeout_seconds}s.",
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
                raw_line = line.rstrip("\n")
                if raw_line:
                    raw_lines.append(raw_line)
                event = self._parse_event_line(raw_line)
                if not event:
                    continue
                session_id = _clean(event.get("sessionID"))
                if session_id:
                    self._session_id = session_id
                event_type = _clean(event.get("type")).lower()
                part = event.get("part") if isinstance(event.get("part"), dict) else {}
                if time_to_first_token is None and event_type not in {"step_start", ""}:
                    time_to_first_token = max(0.0, time.monotonic() - started_at)

                if event_type == "step_start":
                    step_texts = []
                elif event_type == "text":
                    text = str(part.get("text") or "")
                    if text:
                        step_texts.append(text)
                elif event_type == "tool_use":
                    self._record_tool_use(part, steps)
                elif event_type == "step_finish":
                    if step_texts:
                        answer_texts = list(step_texts)
                    usage = _usage_from_tokens(part.get("tokens"))
                    if usage:
                        raw_responses.append({"usage": usage})
                        details = usage["input_tokens_details"]
                        report_live_usage(
                            self._model_name,
                            prompt_tokens=usage["input_tokens"],
                            completion_tokens=usage["output_tokens"],
                            cached_prompt_tokens=details["cached_tokens"],
                            cache_write_tokens=details["cache_write_tokens"],
                        )
                elif event_type == "error":
                    error_text = self._error_text(event.get("error"))
        finally:
            unregister_process(cancel_registration)
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass
        return_code = process.wait()
        if step_texts and not answer_texts:
            answer_texts = step_texts

        mcp_error = self._mcp_startup_error(status_path)
        if error_text:
            if self._session_id and "session" in error_text.lower() and "not found" in error_text.lower():
                self._session_id = None
            return (
                f"ERROR: opencode result error: {error_text}",
                steps,
                _RawResult(raw_responses=raw_responses),
            )
        if return_code != 0:
            details = "\n".join(raw_lines).strip()[-4000:] or "No output captured."
            if self._session_id and "not found" in details.lower():
                self._session_id = None
            return (
                f"ERROR: opencode exec failed (exit={return_code}). {details}",
                steps,
                _RawResult(raw_responses=raw_responses),
            )
        if mcp_error:
            return (
                f"ERROR: Required opencode MCP server failed to start: {mcp_error}",
                steps,
                _RawResult(raw_responses=raw_responses),
            )
        response = "".join(answer_texts).strip()
        if not response and raw_lines:
            response = "\n".join(raw_lines).strip()[-4000:]
        return response, steps, _RawResult(
            raw_responses=raw_responses,
            time_to_first_token_seconds=time_to_first_token,
            time_to_first_token_source="opencode_first_response_event",
        )

    def _mcp_startup_error(self, path: str) -> str:
        if not self._allowed_tool_names():
            return ""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return f"{OPENCODE_MCP_SERVER}=not started"
        except Exception as exc:
            _LOGGER.warning("opencode MCP startup receipt could not be read: %s", exc)
            return ""
        error = _clean(payload.get("error") if isinstance(payload, dict) else "")
        if error:
            return f"{OPENCODE_MCP_SERVER}={error[:500]}"
        return ""

    @staticmethod
    def _error_text(error: Any) -> str:
        if not isinstance(error, dict):
            return _clean(error) or "unknown error"
        data = error.get("data") if isinstance(error.get("data"), dict) else {}
        name = _clean(error.get("name"))
        message = _clean(data.get("message")) or _clean(error.get("message"))
        status = data.get("statusCode")
        text = ": ".join(item for item in (name, message) if item) or "unknown error"
        if status:
            text += f" (status {status})"
        return text

    @staticmethod
    def _parse_event_line(line: str) -> dict[str, Any] | None:
        text = _clean(line)
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _record_tool_use(self, part: dict[str, Any], steps: list[tuple[ToolAction, Any]]) -> None:
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        raw_name = _clean(part.get("tool"))
        name = raw_name[len(OPENCODE_MCP_TOOL_PREFIX):] if raw_name.startswith(OPENCODE_MCP_TOOL_PREFIX) else raw_name
        status = "error" if _clean(state.get("status")).lower() == "error" else "success"
        tool_input = state.get("input") if isinstance(state.get("input"), dict) else {}
        step_input: dict[str, Any] = {
            "tool_id": _clean(part.get("callID")),
            "status": status,
            "tool_input": tool_input,
        }
        result = state.get("output") if status == "success" else state.get("error")
        if result is not None:
            step_input["result"] = result
        action = ToolAction(tool=name or "tool", tool_input=step_input)
        steps.append((action, None))
        try:
            log_event(
                "tool_called",
                payload={"tool": action.tool, "tool_input": step_input},
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass
        if name == "task_steps_manager" and status == "success":
            self._sync_task_steps_manager(tool_input)

    @staticmethod
    def _sync_task_steps_manager(arguments: dict[str, Any]) -> None:
        """Mirror the MCP server's plan into this process, where Chack reads it."""

        session_id = _clean(current_session_id())
        if not session_id or not isinstance(arguments, dict):
            return
        task_id: int | None = None
        try:
            if _clean(arguments.get("task_id")):
                task_id = int(arguments["task_id"])
        except (TypeError, ValueError):
            task_id = None
        try:
            TASK_STEPS_STORE.apply(
                session_id=session_id,
                run_label=str(current_run_label() or "Run 1"),
                action=_clean(arguments.get("action")),
                task_id=task_id,
                text=_clean(arguments.get("text")),
                status=_clean(arguments.get("status")),
                tasks_text=_clean(arguments.get("tasks")),
                notes=_clean(arguments.get("notes")),
            )
        except Exception:
            pass


def _toolset_kwargs(config: ChackConfig, model_provider: str) -> dict[str, Any]:
    init_params = inspect.signature(AgentsToolset.__init__).parameters
    kwargs: dict[str, Any] = {"default_model": config.model.primary}
    for role in _ROLE_MODEL_FIELDS:
        for name, value in (
            (f"{role}_model", getattr(config.model, role, "")),
            (f"{role}_max_turns", getattr(config.model, f"{role}_max_turns", 0)),
        ):
            if name in init_params:
                kwargs[name] = value
    if "model_provider" in init_params:
        kwargs["model_provider"] = model_provider
    if "self_critique_enabled" in init_params:
        kwargs["self_critique_enabled"] = bool(getattr(config.agent, "self_critique_enabled", False))
    if "self_critique_rounds" in init_params:
        kwargs["self_critique_rounds"] = int(getattr(config.agent, "self_critique_rounds", 0) or 0)
    return kwargs


def _role_env(config: ChackConfig) -> dict[str, str]:
    env = {"CHACK_DEFAULT_MODEL": _clean(config.model.primary)}
    for role in _ROLE_MODEL_FIELDS:
        upper = role.upper()
        env[f"CHACK_{upper}_MODEL"] = _clean(getattr(config.model, role, ""))
        max_turns = getattr(config.model, f"{role}_max_turns", 0) or _ROLE_DEFAULT_MAX_TURNS.get(role, 30)
        env[f"CHACK_{upper}_MAX_TURNS"] = str(int(max_turns))
    return env


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
) -> OpencodeCliExecutor:
    del max_turns, memory_summary_max_chars
    from ..config import OPENCODE_PROVIDERS

    model_provider = _clean(config.model.provider).lower()
    if model_provider not in OPENCODE_PROVIDERS:
        raise ValueError(f"opencode backend requires model.provider='opencode' (got {model_provider!r})")

    provider = resolve_opencode_provider(config)

    def _tool_names(items: list[Any] | None) -> list[str]:
        names: list[str] = []
        for tool in items or []:
            name = _clean(getattr(tool, "name", "") or getattr(tool, "__name__", ""))
            if name and name not in names:
                names.append(name)
        return names

    if tools_override is not None:
        allowed_tool_names = _tool_names(list(tools_override))
    else:
        base_tools = list(AgentsToolset(config.tools, **_toolset_kwargs(config, model_provider)).tools)
        allowed_tool_names = _tool_names(base_tools + list(tools_append or []))

    serialized_override = ""
    override_names_json = ""
    try:
        serialized_override = serialize_tools_payload(tools_override)
    except Exception as exc:
        override_names = _tool_names(tools_override)
        management_names = {
            "run_researchers_batch", "start_researchers_async", "list_researcher_jobs",
            "get_researcher_task", "poll_researchers_async", "get_researcher_result",
            "cancel_researcher_task", "retry_researcher_task", "cancel_researchers_async",
        }
        if not (set(override_names) & management_names):
            raise
        override_names_json = json.dumps(override_names, ensure_ascii=False)
        _LOGGER.info("Using MCP name-based reconstruction for researcher administrator tools: %s", type(exc).__name__)

    configured_path = os.environ.get("OPENCODE_CLI_PATH", "").strip() or "opencode"
    return OpencodeCliExecutor(
        provider=provider,
        base_system_prompt=system_prompt,
        max_turns=int(config.session.max_turns or 100),
        opencode_cli_path=shutil.which(configured_path) or configured_path,
        tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False, default=str),
        allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False),
        serialized_tools_override_b64=serialized_override,
        serialized_tools_append_b64=serialize_tools_payload(tools_append),
        serialized_tools_override_names_json=override_names_json,
        model_provider=model_provider,
        role_env=_role_env(config),
        min_tools_used=max(0, int(config.tools.min_tools_used or 0)),
        max_tools_used=max(0, int(config.tools.max_tools_used or 0)),
        require_task_steps_manager_init_first=bool(
            getattr(config.agent, "require_task_steps_manager_init_first", True)
            and "task_steps_manager" in allowed_tool_names
        ),
        output_schema_json=(
            json.dumps(config.agent.output_schema_json, ensure_ascii=False)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        output_schema_name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
        output_schema_strict=bool(getattr(config.agent, "output_schema_strict", True)),
        thinking_effort=normalize_thinking_effort(config.agent.thinking_effort),
        max_context_tokens=int(getattr(config.model, "max_context_tokens", 0) or 0),
        memory_max_messages=memory_max_messages,
        memory_reset_to_messages=memory_reset_to_messages,
    )
