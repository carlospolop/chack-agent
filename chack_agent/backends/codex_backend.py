from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import base64
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Optional

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_steps_manager_state import (
    STORE as TASK_STEPS_STORE,
    current_run_label,
    current_session_id,
)
from chack_tools.telemetry import log_event

from ..config import ChackConfig
from ..live_cost_state import report_live_usage
from ..openrouter_routing import get_openrouter_route
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .tool_payloads import (
    CHACK_ALLOWED_TOOLS_JSON_PATH_ENV,
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_APPEND_B64_PATH_ENV,
    CHACK_TOOLS_CONFIG_JSON_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    CHACK_TOOLS_OVERRIDE_B64_PATH_ENV,
    CHACK_INLINE_ENV_VALUE_MAX_CHARS,
    serialize_tools_payload,
    write_payload_to_file,
)


_LOGGER = logging.getLogger("chack.codex_backend")


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _preview_text(value: Any, *, max_chars: int = 2000) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _resolve_codex_exec_cwd() -> str:
    candidate = str(os.environ.get("CHACK_EXEC_CWD", "") or "").strip()
    if candidate:
        return candidate
    return os.getcwd()


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]


@dataclass
class CodexExecutor:
    _conversation: list[dict[str, Any]]
    _memory_limit: int
    _memory_reset_to: int
    _base_system_prompt: str
    _model_name: str
    _max_turns: int
    _codex_path: str
    _openai_api_key: str
    _fallback_openai_api_key: str
    _codex_access_token: str
    _use_codex_access_token: bool
    _tools_config_json: str
    _allowed_tools_json: str
    _serialized_tools_override_b64: str
    _serialized_tools_append_b64: str
    _model_provider: str
    _default_model: str
    _social_network_model: str
    _scientific_model: str
    _websearcher_model: str
    _tester_model: str
    _subchack_model: str
    _social_network_max_turns: int
    _scientific_max_turns: int
    _websearcher_max_turns: int
    _tester_max_turns: int
    _subchack_max_turns: int
    _min_tools_used: int
    _max_tools_used: int
    _require_task_steps_manager_init_first: bool
    _output_schema_json: str
    _uses_openrouter_route: bool = False
    _openrouter_base_url: str = ""
    _openrouter_http_referer: str = ""
    _openrouter_app_name: str = ""
    _output_schema_path: Optional[str] = None
    _thread_id: Optional[str] = None
    _codex_home: Optional[str] = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_codex(prompt)

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

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)

    def _compose_prompt(self, user_input: str) -> str:
        base = str(self._base_system_prompt or "").strip()
        policy_lines: list[str] = []
        if self._require_task_steps_manager_init_first:
            policy_lines.append(
                "- First, call task_steps_manager with action=init before any other tool call."
            )
        if self._min_tools_used > 0:
            policy_lines.append(
                f"- Use at least {self._min_tools_used} non-task tool calls before finalizing to ensure you gather enough information and context for your task."
            )
        if self._max_tools_used > 0:
            policy_lines.append(
                f"- To complete your task you must not exceed {self._max_tools_used} non-task tool calls."
            )
        policy_block = ""
        if policy_lines:
            policy_block = "\n\n### TOOL USAGE POLICY\n" + "\n".join(policy_lines)

        if not base:
            return f"{user_input}{policy_block}" if policy_block else user_input
        if not user_input:
            return f"{base}{policy_block}" if policy_block else base
        return f"{base}{policy_block}\n\n### USER REQUEST\n{user_input}"

    def _run_codex(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_codex_home_and_config()
        return self._run_codex_once(prompt, allow_api_key_fallback=True)

    def _run_codex_once(
        self,
        prompt: str,
        *,
        allow_api_key_fallback: bool,
    ) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        command = self._build_command()
        env = self._build_env()
        timeout_seconds = int(os.environ.get("CHACK_CODEX_EXEC_TIMEOUT_SECONDS", "900") or "900")
        exec_cwd = _resolve_codex_exec_cwd()
        _LOGGER.info(
            "Starting Codex CLI process: model=%s timeout_seconds=%s thread_id=%s cwd=%s ts=%s",
            self._model_name,
            timeout_seconds,
            self._thread_id or "",
            exec_cwd,
            _log_timestamp(),
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
                cwd=exec_cwd or None,
            )
        except FileNotFoundError:
            self._log_codex_failure(
                "codex_cli_missing",
                command=command,
                cwd=exec_cwd,
                details=(
                    "ERROR: Codex CLI executable was not found.\n"
                    f"Configured path: {self._codex_path!r}.\n"
                    "Install Codex CLI (e.g. `npm i -g @openai/codex`) or set CODEX_PATH "
                    "to the absolute executable path."
                ),
            )
            return (
                (
                    "ERROR: Codex CLI executable was not found.\n"
                    f"Configured path: {self._codex_path!r}.\n"
                    "Install Codex CLI (e.g. `npm i -g @openai/codex`) or set CODEX_PATH "
                    "to the absolute executable path."
                ),
                [],
                _RawResult(raw_responses=[]),
            )
        except Exception as exc:
            self._log_codex_failure(
                "codex_cli_launch_failed",
                command=command,
                cwd=exec_cwd,
                details=f"{type(exc).__name__}: {exc}",
            )
            return (
                f"ERROR: Failed to launch Codex CLI: {type(exc).__name__}: {exc}",
                [],
                _RawResult(raw_responses=[]),
            )

        steps: list[tuple[ToolAction, Any]] = []
        output_parts: list[str] = []
        usage_payload: dict[str, Any] | None = None
        combined_output_lines: list[str] = []
        error_messages: list[str] = []
        started_at = time.monotonic()

        if process.stdin is not None:
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except Exception:
                pass

        while True:
            if (time.monotonic() - started_at) > timeout_seconds:
                process.kill()
                self._log_codex_failure(
                    "codex_cli_timeout",
                    command=command,
                    cwd=exec_cwd,
                    details="\n".join(combined_output_lines).strip()
                    or f"Timed out after {timeout_seconds}s with no captured output.",
                )
                return (
                    f"ERROR: Codex execution timed out after {timeout_seconds}s.",
                    steps,
                    _RawResult(raw_responses=[]),
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
                combined_output_lines.append(raw_line)
            event = self._parse_event_line(raw_line)
            if not event:
                continue

            event_type = str(event.get("type", "") or "")
            if event_type == "thread.started":
                thread_id = str(event.get("thread_id", "") or "").strip()
                if thread_id:
                    self._thread_id = thread_id
                continue

            if event_type == "error":
                message = str(event.get("message", "") or "").strip()
                if message:
                    error_messages.append(message)
                continue

            if event_type == "item.completed":
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                item_type = str(item.get("type", "") or "")

                if item_type == "error":
                    message = str(item.get("message", "") or "").strip()
                    if message:
                        error_messages.append(message)
                    continue

                if item_type == "reasoning":
                    reasoning_text = str(item.get("text", "") or "")
                    _LOGGER.info(
                        "Codex reasoning summary observed: chars=%s thread_id=%s ts=%s",
                        len(reasoning_text),
                        self._thread_id or "",
                        _log_timestamp(),
                    )
                    log_event(
                        "agent_reasoning_summary",
                        payload={
                            "backend": "codex",
                            "provider": str(self._model_provider or "codex"),
                            "model": str(self._model_name or ""),
                            "thread_id": str(self._thread_id or ""),
                            "summary_chars": int(len(reasoning_text)),
                            "summary_preview": reasoning_text[:500],
                        },
                        task_session_id=current_session_id() or "",
                        run_label=current_run_label() or "",
                    )
                    continue

                if item_type == "agent_message":
                    message_text = self._extract_message_text(item)
                    if message_text:
                        output_parts.append(message_text)
                    continue

                if item_type in {"assistant_message", "message"}:
                    message_text = self._extract_message_text(item)
                    if message_text:
                        output_parts.append(message_text)
                    continue

                step = self._item_to_step(item)
                if step is not None:
                    steps.append((step, None))
                    self._log_tool_called(step.tool, step.tool_input)
                    self._sync_task_steps_manager(item)
                continue

            if event_type in {"message", "agent_message"}:
                message_text = self._extract_message_text(event)
                if message_text:
                    output_parts.append(message_text)
                continue

            if event_type == "turn.completed":
                usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
                usage_payload = {
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    "input_tokens_details": {
                        "cached_tokens": int(usage.get("cached_input_tokens", 0) or 0),
                        "cache_write_tokens": 0,
                    },
                }
                report_live_usage(
                    self._model_name,
                    prompt_tokens=usage_payload["input_tokens"],
                    completion_tokens=usage_payload["output_tokens"],
                    cached_prompt_tokens=usage_payload["input_tokens_details"]["cached_tokens"],
                    cache_write_tokens=0,
                )

        output = "\n".join(part for part in output_parts if part).strip()
        return_code = process.wait()
        if return_code != 0:
            details = "\n".join(combined_output_lines).strip() or "No error output captured."
            self._log_codex_failure(
                "codex_exec_failed",
                command=command,
                cwd=exec_cwd,
                details=details,
                return_code=return_code,
            )
            result = (
                f"ERROR: Codex exec failed (exit={return_code}).\n{details}",
                steps,
                _RawResult(raw_responses=[]),
            )
            return self._maybe_retry_with_api_key(
                prompt,
                result,
                allow_api_key_fallback,
                codex_exec_failed=True,
            )

        raw_responses: list[Any] = []
        if usage_payload is not None:
            raw_responses.append({"usage": usage_payload})
        if not output and usage_payload is None and not steps:
            details = "\n".join(error_messages or combined_output_lines).strip() or (
                "Codex CLI exited successfully but produced no response events."
            )
            self._log_codex_failure(
                "codex_no_usable_response",
                command=command,
                cwd=exec_cwd,
                details=details,
                return_code=return_code,
            )
            result = (
                f"ERROR: Codex exec produced no usable response.\n{details}",
                steps,
                _RawResult(raw_responses=[]),
            )
            return self._maybe_retry_with_api_key(
                prompt,
                result,
                allow_api_key_fallback,
                codex_exec_failed=True,
            )
        result = (output, steps, _RawResult(raw_responses=raw_responses))
        return self._maybe_retry_with_api_key(
            prompt,
            result,
            allow_api_key_fallback,
            codex_exec_failed=False,
        )

    def _maybe_retry_with_api_key(
        self,
        prompt: str,
        result: tuple[str, list[tuple[ToolAction, Any]], _RawResult],
        allow_api_key_fallback: bool,
        *,
        codex_exec_failed: bool,
    ) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        if not allow_api_key_fallback:
            return result
        if not self._use_codex_access_token:
            return result
        if not self._fallback_openai_api_key:
            return result
        if not codex_exec_failed:
            return result
        if not self._looks_like_auth_failure(result[0]):
            return result

        _LOGGER.warning(
            "Codex access token failed. Falling back to OPENAI_API_KEY for provider=codex."
        )
        self._use_codex_access_token = False
        self._openai_api_key = self._fallback_openai_api_key
        self._remove_codex_auth_file()
        return self._run_codex_once(prompt, allow_api_key_fallback=False)

    def _log_codex_failure(
        self,
        failure_type: str,
        *,
        command: list[str],
        cwd: str,
        details: str,
        return_code: int | None = None,
    ) -> None:
        preview = _preview_text(details)
        _LOGGER.error(
            "Codex CLI failure: type=%s provider=%s model=%s return_code=%s thread_id=%s cwd=%s command=%s details=%s ts=%s",
            failure_type,
            self._model_provider,
            self._model_name,
            return_code,
            self._thread_id or "",
            cwd,
            command,
            preview,
            _log_timestamp(),
        )
        try:
            log_event(
                "codex_cli_failure",
                payload={
                    "failure_type": failure_type,
                    "provider": str(self._model_provider or ""),
                    "model": str(self._model_name or ""),
                    "return_code": return_code,
                    "thread_id": str(self._thread_id or ""),
                    "cwd": str(cwd or ""),
                    "command": [str(part) for part in command],
                    "details_preview": preview,
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass

    @staticmethod
    def _looks_like_auth_failure(output: str) -> bool:
        normalized = str(output or "").lower()
        indicators = (
            "not signed in",
            "error checking login status",
            "401 unauthorized",
            "status code: 401",
            "unauthorized",
            "missing bearer or basic authentication",
            "incorrect api key provided",
        )
        return any(indicator in normalized for indicator in indicators)

    def _build_command(self) -> list[str]:
        exec_cwd = _resolve_codex_exec_cwd()
        if self._thread_id:
            args = [
                self._codex_path,
                "exec",
                "resume",
                "--json",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
            ]
            args.extend(
                [
                "--model",
                self._model_name,
                self._thread_id,
                "-",
                ]
            )
            return args
        output_schema_args: list[str] = []
        if self._output_schema_path:
            output_schema_args = ["--output-schema", self._output_schema_path]
        args = [
            self._codex_path,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            exec_cwd,
        ]
        args.extend(
            [
                "--model",
                self._model_name,
            ]
        )
        if output_schema_args:
            args.extend(output_schema_args)
        args.append("-")
        return args

    def _build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if v is not None}
        # Ensure PYTHONPATH includes the script directory so MCP server
        # subprocesses can import application modules via cloudpickle.
        if "PYTHONPATH" not in env:
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else os.getcwd()
            env["PYTHONPATH"] = script_dir
        if self._uses_openrouter_route:
            env["OPENROUTER_API_KEY"] = self._openai_api_key
            env["OPENROUTER_BASE_URL"] = self._openrouter_base_url
            if self._openrouter_http_referer:
                env["OPENROUTER_HTTP_REFERER"] = self._openrouter_http_referer
            if self._openrouter_app_name:
                env["OPENROUTER_APP_NAME"] = self._openrouter_app_name
        elif self._use_codex_access_token:
            env.pop("OPENAI_API_KEY", None)
            env.pop("CODEX_API_KEY", None)
        else:
            env.setdefault("OPENAI_API_KEY", self._openai_api_key)
            env.setdefault("CODEX_API_KEY", self._openai_api_key)
        if self._codex_home:
            env["CODEX_HOME"] = self._codex_home
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
        env["CHACK_MODEL_PROVIDER"] = self._model_provider
        env["CHACK_DEFAULT_MODEL"] = self._default_model
        env["CHACK_SOCIAL_NETWORK_MODEL"] = self._social_network_model
        env["CHACK_SCIENTIFIC_MODEL"] = self._scientific_model
        env["CHACK_WEBSEARCHER_MODEL"] = self._websearcher_model
        env["CHACK_TESTER_MODEL"] = self._tester_model
        env["CHACK_SUBCHACK_MODEL"] = self._subchack_model
        env["CHACK_SOCIAL_NETWORK_MAX_TURNS"] = str(self._social_network_max_turns)
        env["CHACK_SCIENTIFIC_MAX_TURNS"] = str(self._scientific_max_turns)
        env["CHACK_WEBSEARCHER_MAX_TURNS"] = str(self._websearcher_max_turns)
        env["CHACK_TESTER_MAX_TURNS"] = str(self._tester_max_turns)
        env["CHACK_SUBCHACK_MAX_TURNS"] = str(self._subchack_max_turns)
        env["CHACK_MIN_TOOLS_USED"] = str(self._min_tools_used)
        env["CHACK_MAX_TOOLS_USED"] = str(self._max_tools_used)
        env["CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST"] = (
            "1" if self._require_task_steps_manager_init_first else "0"
        )
        env["CHACK_TASK_SESSION_ID"] = str(current_session_id() or "")
        env["CHACK_RUN_LABEL"] = str(current_run_label() or "Run 1")
        env["CHACK_DISABLE_STDOUT_EVENTS"] = "1"
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
        payload_dir = self._codex_home or os.environ.get("CHACK_CODEX_HOME_BASE", "").strip() or ""
        path = write_payload_to_file(raw, prefix=prefix, directory=payload_dir)
        env[path_env_key] = path

    def _ensure_codex_home_and_config(self) -> None:
        if self._codex_home:
            return
        safe_session = re.sub(r"[^A-Za-z0-9._-]", "_", str(current_session_id() or "default"))
        home_base = str(
            os.environ.get("CHACK_CODEX_HOME_BASE", os.path.expanduser("~/.codex/chack")) or ""
        ).strip() or os.path.expanduser("~/.codex/chack")
        base = os.path.join(home_base, safe_session)
        os.makedirs(base, exist_ok=True)
        self._codex_home = base
        self._write_codex_config(base)
        self._write_codex_auth(base)
        self._write_output_schema_file(base)

    def _write_codex_config(self, codex_home: str) -> None:
        os.makedirs(codex_home, exist_ok=True)
        config_path = os.path.join(codex_home, "config.toml")
        python_cmd = sys.executable or "python3"
        env_vars = [
            "CHACK_TOOLS_CONFIG_JSON",
            "CHACK_TOOLS_CONFIG_JSON_PATH",
            "CHACK_ALLOWED_TOOLS_JSON",
            "CHACK_ALLOWED_TOOLS_JSON_PATH",
            "CHACK_TOOLS_OVERRIDE_B64",
            "CHACK_TOOLS_OVERRIDE_B64_PATH",
            "CHACK_TOOLS_APPEND_B64",
            "CHACK_TOOLS_APPEND_B64_PATH",
            "PYTHONPATH",
            "CHACK_MODEL_PROVIDER",
            "CHACK_DEFAULT_MODEL",
            "CHACK_SOCIAL_NETWORK_MODEL",
            "CHACK_SCIENTIFIC_MODEL",
            "CHACK_WEBSEARCHER_MODEL",
            "CHACK_TESTER_MODEL",
            "CHACK_SUBCHACK_MODEL",
            "CHACK_SOCIAL_NETWORK_MAX_TURNS",
            "CHACK_SCIENTIFIC_MAX_TURNS",
            "CHACK_WEBSEARCHER_MAX_TURNS",
            "CHACK_TESTER_MAX_TURNS",
            "CHACK_SUBCHACK_MAX_TURNS",
            "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST",
            "CHACK_TASK_SESSION_ID",
            "CHACK_RUN_LABEL",
            "CHACK_DISABLE_STDOUT_EVENTS",
            "AISEC_LOCAL_VULN_STORE_PATH",
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
            "GEMINI_API_KEY",
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
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_NAME",
        ]

        def _toml_string(value: str) -> str:
            return json.dumps(str(value))

        env_vars_toml = "[" + ", ".join(_toml_string(v) for v in env_vars) + "]"
        args_toml = "[" + ", ".join(
            _toml_string(v)
            for v in ["-m", "chack_agent.backends.chack_tools_mcp_server"]
        ) + "]"

        config_lines = [f"model = {_toml_string(self._model_name)}"]
        if self._uses_openrouter_route:
            config_lines.extend(
                [
                    'model_provider = "openrouter"',
                    "",
                    "[model_providers.openrouter]",
                    'name = "OpenRouter"',
                    f'base_url = {_toml_string(self._openrouter_base_url)}',
                    'env_key = "OPENROUTER_API_KEY"',
                    'wire_api = "responses"',
                    "requires_openai_auth = false",
                    "supports_websockets = false",
                ]
            )
            header_entries: list[str] = []
            if self._openrouter_http_referer:
                header_entries.append('"HTTP-Referer" = "OPENROUTER_HTTP_REFERER"')
            if self._openrouter_app_name:
                header_entries.append('"X-Title" = "OPENROUTER_APP_NAME"')
            if header_entries:
                config_lines.extend(["[model_providers.openrouter.env_http_headers]"])
                config_lines.extend(header_entries)
        config_lines.extend(
            [
                "",
                "[mcp_servers.chack_tools]",
                f"command = {_toml_string(python_cmd)}",
                f"args = {args_toml}",
                f"env_vars = {env_vars_toml}",
                "required = true",
                "startup_timeout_sec = 30",
                f"tool_timeout_sec = {int(os.environ.get('CHACK_MCP_TOOL_TIMEOUT_SECONDS', '3600') or '3600')}",
            ]
        )
        if self._playwright_mcp_enabled():
            playwright_server = playwright_mcp_server_config()
            playwright_args_toml = "[" + ", ".join(
                _toml_string(v) for v in list(playwright_server.get("args") or [])
            ) + "]"
            config_lines.extend(
                [
                    "",
                    "[mcp_servers.playwright]",
                    f"command = {_toml_string(str(playwright_server['command']))}",
                    f"args = {playwright_args_toml}",
                    "startup_timeout_sec = 30",
                    "tool_timeout_sec = 180",
                ]
            )
        config_body = "\n".join(config_lines)
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(config_body + "\n")

    def _write_codex_auth(self, codex_home: str) -> None:
        if self._uses_openrouter_route or not self._use_codex_access_token:
            self._remove_codex_auth_file()
            return
        auth_path = os.path.join(codex_home, "auth.json")
        auth_payload = self._build_codex_chatgpt_auth_payload(self._codex_access_token)
        with open(auth_path, "w", encoding="utf-8") as handle:
            json.dump(auth_payload, handle, ensure_ascii=False)
            handle.write("\n")

    def _remove_codex_auth_file(self) -> None:
        if not self._codex_home:
            return
        auth_path = os.path.join(self._codex_home, "auth.json")
        try:
            os.remove(auth_path)
        except FileNotFoundError:
            return

    @staticmethod
    def _build_codex_chatgpt_auth_payload(access_token: str) -> dict[str, Any]:
        claims = CodexExecutor._decode_jwt_claims(access_token)
        auth_claims = claims.get("https://api.openai.com/auth")
        if not isinstance(auth_claims, dict):
            auth_claims = {}
        account_id = str(auth_claims.get("chatgpt_account_id", "") or "").strip()
        if not account_id:
            raise ValueError(
                "CODEX_ACCESS_TOKEN is missing chatgpt_account_id in JWT claims and cannot be used for Codex ChatGPT auth."
            )
        return {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": access_token,
                "access_token": access_token,
                "refresh_token": "",
                "account_id": account_id,
            },
            "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @staticmethod
    def _decode_jwt_claims(token: str) -> dict[str, Any]:
        raw = str(token or "").strip()
        parts = raw.split(".")
        if len(parts) != 3 or not parts[1]:
            raise ValueError("CODEX_ACCESS_TOKEN is not a valid JWT.")
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
            parsed = json.loads(decoded.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Failed to decode CODEX_ACCESS_TOKEN JWT claims.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("CODEX_ACCESS_TOKEN JWT payload is not a JSON object.")
        return parsed

    def _playwright_mcp_enabled(self) -> bool:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        return bool(cfg.get("playwright_enabled")) and playwright_mcp_is_available()

    def _write_output_schema_file(self, codex_home: str) -> None:
        self._output_schema_path = None
        raw = str(self._output_schema_json or "").strip()
        if not raw:
            return
        try:
            schema_obj = json.loads(raw)
        except Exception:
            return
        if not isinstance(schema_obj, dict):
            return
        path = os.path.join(codex_home, "output_schema.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(schema_obj, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            self._output_schema_path = path
        except Exception:
            self._output_schema_path = None

    @staticmethod
    def _parse_event_line(line: str) -> Optional[dict[str, Any]]:
        raw = str(line or "").strip()
        if not raw or not raw.startswith("{"):
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @classmethod
    def _extract_message_text(cls, payload: Any) -> str:
        text = cls._extract_text_candidate(payload)
        return text.strip()

    @classmethod
    def _extract_text_candidate(cls, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            parts = [cls._extract_text_candidate(item) for item in payload]
            return "\n".join(part for part in parts if part).strip()
        if not isinstance(payload, dict):
            return ""

        direct_fields = (
            "text",
            "message",
            "content",
            "output_text",
            "final_text",
        )
        for field in direct_fields:
            value = payload.get(field)
            text = cls._extract_text_candidate(value)
            if text:
                return text

        for field in ("parts", "chunks", "content_parts"):
            value = payload.get(field)
            text = cls._extract_text_candidate(value)
            if text:
                return text

        if payload.get("type") in {"text", "output_text"}:
            value = payload.get("text") or payload.get("value")
            text = cls._extract_text_candidate(value)
            if text:
                return text

        if isinstance(payload.get("item"), dict):
            text = cls._extract_text_candidate(payload.get("item"))
            if text:
                return text

        return ""

    @staticmethod
    def _item_to_step(item: dict[str, Any]) -> Optional[ToolAction]:
        item_type = str(item.get("type", "") or "")

        if item_type == "command_execution":
            return ToolAction(
                tool="exec",
                tool_input={
                    "command": str(item.get("command", "") or ""),
                    "status": str(item.get("status", "") or ""),
                    "exit_code": item.get("exit_code"),
                },
            )
        if item_type == "web_search":
            return ToolAction(
                tool="search_google_web",
                tool_input={"query": str(item.get("query", "") or "")},
            )
        if item_type == "file_change":
            return ToolAction(
                tool="apply_patch",
                tool_input={
                    "status": str(item.get("status", "") or ""),
                    "changes": item.get("changes", []),
                },
            )
        if item_type == "mcp_tool_call":
            tool_name = str(item.get("tool", "") or "mcp_tool_call")
            tool_input = {
                "server": str(item.get("server", "") or ""),
                "arguments": item.get("arguments"),
                "status": str(item.get("status", "") or ""),
            }
            if "error" in item:
                tool_input["error"] = item.get("error")
            if "result" in item:
                tool_input["result"] = item.get("result")
            return ToolAction(
                tool=tool_name,
                tool_input=tool_input,
            )
        if item_type == "todo_list":
            return ToolAction(
                tool="task_steps_manager",
                tool_input={
                    "action": "update",
                    "items": item.get("items", []),
                },
            )
        return None

    @staticmethod
    def _log_tool_called(tool_name: str, tool_input: Any) -> None:
        try:
            log_event(
                "tool_called",
                payload={
                    "tool": tool_name,
                    "tool_input": tool_input,
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass

    @staticmethod
    def _sync_task_steps_manager(item: dict[str, Any]) -> None:
        try:
            if str(item.get("type", "") or "") != "mcp_tool_call":
                return
            if str(item.get("tool", "") or "").strip() != "task_steps_manager":
                return
            if str(item.get("status", "") or "").strip().lower() != "completed":
                return

            arguments = item.get("arguments")
            if not isinstance(arguments, dict):
                return

            session_id = str(current_session_id() or "").strip()
            if not session_id:
                return
            run_label = str(current_run_label() or "Run 1")

            raw_task_id = arguments.get("task_id")
            task_id: int | None = None
            if raw_task_id not in (None, ""):
                try:
                    task_id = int(raw_task_id)
                except Exception:
                    task_id = None

            TASK_STEPS_STORE.apply(
                session_id=session_id,
                run_label=run_label,
                action=str(arguments.get("action", "") or ""),
                task_id=task_id,
                text=str(arguments.get("text", "") or ""),
                status=str(arguments.get("status", "") or ""),
                tasks_text=str(arguments.get("tasks", "") or ""),
                notes=str(arguments.get("notes", "") or ""),
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
    tools_override: Optional[list[Any]] = None,
    tools_append: Optional[list[Any]] = None,
) -> CodexExecutor:
    try:
        _LOGGER.debug(
            "codex build_executor: memory_summary_max_chars=%s (not used in this backend)",
            int(memory_summary_max_chars),
        )
    except Exception:
        _LOGGER.debug(
            "codex build_executor: memory_summary_max_chars provided (unable to coerce to int in debug log)"
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

    allowed_tool_names: list[str] | None = None
    serialized_tools_override_b64 = serialize_tools_payload(tools_override)
    serialized_tools_append_b64 = serialize_tools_payload(tools_append)
    model_provider = str(config.model.provider or "").strip()
    if not model_provider:
        raise ValueError("model.provider must be defined in config")
    if tools_override is not None:
        allowed_tool_names = _extract_tool_names(list(tools_override))
    elif tools_append:
        base_toolset = AgentsToolset(
            config.tools,
            model_provider=model_provider,
            default_model=config.model.primary,
            social_network_model=config.model.social_network,
            scientific_model=config.model.scientific,
            websearcher_model=config.model.websearcher,
            tester_model=config.model.tester,
            subchack_model=config.model.subchack,
            social_network_max_turns=config.model.social_network_max_turns,
            scientific_max_turns=config.model.scientific_max_turns,
            websearcher_max_turns=config.model.websearcher_max_turns,
            tester_max_turns=config.model.tester_max_turns,
            subchack_max_turns=config.model.subchack_max_turns,
        )
        allowed_tool_names = _extract_tool_names(list(base_toolset.tools) + list(tools_append))

    route = get_openrouter_route(config)
    uses_openrouter_route = route is not None
    fallback_openai_api_key = (
        str(config.credentials.openai_api_key or "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    codex_access_token = (
        str(getattr(config.credentials, "codex_access_token", "") or "").strip()
        or os.environ.get("CODEX_ACCESS_TOKEN", "").strip()
    )
    codex_api_key = route.api_key if route is not None else (codex_access_token or fallback_openai_api_key)
    if not codex_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for OpenRouter-routed Codex models"
            if uses_openrouter_route
            else "CODEX_ACCESS_TOKEN or OPENAI_API_KEY is required when model.provider=codex"
        )

    configured_codex_path = os.environ.get("CODEX_PATH", "").strip() or "codex"
    codex_path = shutil.which(configured_codex_path) or configured_codex_path


    return CodexExecutor(
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _model_name=str(route.model_name if route is not None else config.model.primary),
        _max_turns=int(max_turns or 0),
        _codex_path=codex_path,
        _openai_api_key=codex_api_key,
        _fallback_openai_api_key=fallback_openai_api_key,
        _codex_access_token=codex_access_token,
        _use_codex_access_token=bool(codex_access_token) and not uses_openrouter_route,
        _tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False),
        _allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False)
        if allowed_tool_names is not None
        else "",
        _serialized_tools_override_b64=serialized_tools_override_b64,
        _serialized_tools_append_b64=serialized_tools_append_b64,
        _model_provider=model_provider,
        _default_model=str(config.model.primary or ""),
        _social_network_model=str(config.model.social_network or ""),
        _scientific_model=str(config.model.scientific or ""),
        _websearcher_model=str(config.model.websearcher or ""),
        _tester_model=str(config.model.tester or ""),
        _subchack_model=str(config.model.subchack or ""),
        _social_network_max_turns=int(config.model.social_network_max_turns or 30),
        _scientific_max_turns=int(config.model.scientific_max_turns or 30),
        _websearcher_max_turns=int(config.model.websearcher_max_turns or 30),
        _tester_max_turns=int(config.model.tester_max_turns or 30),
        _subchack_max_turns=int(config.model.subchack_max_turns or 30),
        _min_tools_used=max(0, int(config.tools.min_tools_used or 0)),
        _max_tools_used=max(0, int(config.tools.max_tools_used or 0)),
        _require_task_steps_manager_init_first=bool(
            getattr(config.agent, "require_task_steps_manager_init_first", True)
        ),
        _output_schema_json=(
            json.dumps(getattr(config.agent, "output_schema_json", None), ensure_ascii=False, indent=2)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        _uses_openrouter_route=uses_openrouter_route,
        _openrouter_base_url=str(route.base_url if route is not None else ""),
        _openrouter_http_referer=str((route.headers.get("HTTP-Referer", "") if route else "")),
        _openrouter_app_name=str((route.headers.get("X-Title", "") if route else "")),
    )
