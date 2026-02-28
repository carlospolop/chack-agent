from __future__ import annotations

import json
import logging
import os
import re
import inspect
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_steps_manager_state import (
    STORE as TASK_STEPS_STORE,
    current_run_label,
    current_session_id,
)
from chack_tools.telemetry import log_event

from ..config import ChackConfig


_LOGGER = logging.getLogger("chack.gemini_cli_backend")


_GEMINI_CLI_DENYLIST_TOOLS = {
    "glob",
    "grep_search",
    "list_directory",
    "read_file",
    "run_shell_command",
    "write_file",
    "replace",
    "google_web_search",
    "write_todos",
    "web_fetch",
    "read_many_files",
    "save_memory",
    "get_internal_docs",
    "activate_skill",
    "ask_user",
    "enter_plan_mode",
    "exit_plan_mode",
    "exec",
}


def _sanitize_session_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(session_id or "").strip()).strip()


def _safe_settings_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]


class GeminiCliExecutor:
    def __init__(
        self,
        *,
        conversation: list[dict[str, Any]],
        memory_max_messages: int,
        memory_reset_to_messages: int,
        base_system_prompt: str,
        model_name: str,
        max_turns: int,
        gemini_cli_path: str,
        gemini_api_key: str,
        tools_config_json: str,
        allowed_tools_json: str,
        model_provider: str,
        default_model: str,
        social_network_model: str,
        scientific_model: str,
        websearcher_model: str,
        tester_model: str,
        social_network_max_turns: int,
        scientific_max_turns: int,
        websearcher_max_turns: int,
        tester_max_turns: int,
        min_tools_used: int,
        max_tools_used: int,
        require_task_steps_manager_init_first: bool,
        output_schema_json: str,
        output_schema_name: str,
        output_schema_strict: bool,
    ) -> None:
        self._conversation = conversation
        self._memory_limit = memory_max_messages
        self._memory_reset_to = memory_reset_to_messages
        self._base_system_prompt = base_system_prompt
        self._model_name = str(model_name or "").strip()
        self._max_turns = int(max_turns or 0)
        self._gemini_cli_path = gemini_cli_path
        self._gemini_api_key = gemini_api_key
        self._tools_config_json = tools_config_json
        self._allowed_tools_json = allowed_tools_json
        self._model_provider = model_provider
        self._default_model = default_model
        self._social_network_model = social_network_model
        self._scientific_model = scientific_model
        self._websearcher_model = websearcher_model
        self._tester_model = tester_model
        self._social_network_max_turns = social_network_max_turns
        self._scientific_max_turns = scientific_max_turns
        self._websearcher_max_turns = websearcher_max_turns
        self._tester_max_turns = tester_max_turns
        self._min_tools_used = max(0, int(min_tools_used or 0))
        self._max_tools_used = max(0, int(max_tools_used or 0))
        self._require_task_steps_manager_init_first = bool(
            require_task_steps_manager_init_first
        )
        self._output_schema_json = output_schema_json or ""
        self._output_schema_name = output_schema_name or "output_schema"
        self._output_schema_strict = bool(output_schema_strict)

        self._gemini_home: str | None = None
        self._gemini_session_id: str | None = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_gemini(prompt)

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
                f"- Use at least {self._min_tools_used} non-task tool calls before finalizing."
            )

        if self._max_tools_used > 0:
            policy_lines.append(
                f"- Do not exceed {self._max_tools_used} non-task tool calls in total."
            )

        schema_lines: list[str] = []
        schema_name = self._output_schema_name or "output_schema"
        if self._output_schema_json:
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
            schema_block = "\n\n" + "\n".join(schema_lines)

        prompt_parts = [p for p in (base, user_input, policy_block, schema_block) if p.strip()]
        return "\n".join(prompt_parts)

    def _run_gemini(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_gemini_home_and_config()
        command = self._build_command(prompt)
        env = self._build_env()
        timeout_seconds = int(os.environ.get("CHACK_GEMINI_EXEC_TIMEOUT_SECONDS", "900") or "900")

        _LOGGER.info(
            "Starting Gemini CLI process: model=%s timeout_seconds=%s gemini_session_id=%s",
            self._model_name,
            timeout_seconds,
            self._gemini_session_id or "",
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
            )
        except FileNotFoundError:
            return (
                "ERROR: Gemini CLI executable was not found. "
                f"Configured path: {self._gemini_cli_path!r}. "
                "Install Gemini CLI (e.g. `npm i -g @google/gemini-cli`) or set GEMINI_CLI_PATH to the absolute executable path.",
                [],
                _RawResult(raw_responses=[]),
            )
        except Exception as exc:
            return (
                f"ERROR: Failed to launch Gemini CLI: {type(exc).__name__}: {exc}",
                [],
                _RawResult(raw_responses=[]),
            )

        output_parts: list[str] = []
        steps: list[tuple[ToolAction, Any]] = []
        raw_lines: list[str] = []
        raw_responses: list[Any] = []
        started_at = time.monotonic()
        tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        try:
            while True:
                if (time.monotonic() - started_at) > timeout_seconds:
                    process.kill()
                    return (
                        f"ERROR: Gemini execution timed out after {timeout_seconds}s.",
                        steps,
                        _RawResult(raw_responses=raw_responses),
                    )

                if process.stdout is None:
                    break

                raw_line = process.stdout.readline()
                if raw_line == "" and process.poll() is not None:
                    break
                if not raw_line:
                    time.sleep(0.05)
                    continue

                line = str(raw_line).rstrip("\n")
                if line:
                    raw_lines.append(line)
                event = self._parse_event_line(line)
                if not event:
                    continue

                event_type = str(event.get("type") or "").strip()
                if event_type == "init":
                    session_id = str(event.get("session_id") or "").strip()
                    if session_id:
                        self._gemini_session_id = session_id
                    continue

                if event_type == "message":
                    role = str(event.get("role") or "").strip().lower()
                    content = str(event.get("content", "") or "")
                    if role == "assistant" and content:
                        output_parts.append(content)
                    continue

                if event_type == "tool_use":
                    tool_id = str(event.get("tool_id") or "").strip()
                    tool_name = str(event.get("tool_name") or "").strip()
                    params = event.get("parameters")
                    if not isinstance(params, dict):
                        params = {}
                    if tool_id:
                        tool_calls[tool_id] = (tool_name, params)
                    continue

                if event_type == "tool_result":
                    tool_id = str(event.get("tool_id") or "").strip()
                    status = str(event.get("status") or "").strip()
                    error = event.get("error")
                    output = event.get("output")
                    if tool_id and tool_id in tool_calls:
                        tool_name, tool_input = tool_calls.pop(tool_id)
                    else:
                        tool_name = str(event.get("tool") or "").strip()
                        tool_input = {}

                    step_input = {
                        "tool_id": tool_id,
                        "status": status,
                        **(tool_input or {}),
                    }
                    if output is not None:
                        step_input["result"] = output
                    if error is not None:
                        step_input["error"] = error
                    tool_action = ToolAction(tool=tool_name or "tool", tool_input=step_input)
                    steps.append((tool_action, None))
                    self._log_tool_called(tool_action.tool, tool_action.tool_input)
                    if tool_action.tool == "task_steps_manager":
                        self._sync_task_steps_manager(tool_input, status, error)
                    continue

                if event_type == "result":
                    stats = event.get("stats")
                    if isinstance(stats, dict):
                        usage = {
                            "input_tokens": int(stats.get("input_tokens", 0) or 0),
                            "output_tokens": int(stats.get("output_tokens", 0) or 0),
                            "input_tokens_details": {
                                "cached_tokens": int(stats.get("cached", 0) or 0),
                                "cache_write_tokens": 0,
                            },
                        }
                        raw_responses.append({"usage": usage})
                    if str(event.get("status") or "").strip().lower() == "error":
                        error_obj = event.get("error")
                        details = event.get("message") or ""
                        error_msg = details
                        if isinstance(error_obj, dict):
                            error_text = str(error_obj.get("message", "")).strip()
                            if error_text:
                                error_msg = error_text
                        return (
                            f"ERROR: Gemini result error: {error_msg}",
                            steps,
                            _RawResult(raw_responses=raw_responses),
                        )
                    break

            return_code = process.wait()
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass

        if return_code != 0:
            details = "\n".join(raw_lines).strip() or "No output captured."
            return (
                f"ERROR: Gemini exec failed (exit={return_code}). {details}",
                steps,
                _RawResult(raw_responses=raw_responses),
            )

        response = "".join(output_parts).strip()
        if not response and raw_lines:
            response = "\n".join(raw_lines).strip()
            if response:
                response = response[-4000:]

        return response, steps, _RawResult(raw_responses=raw_responses)

    def _build_command(self, prompt: str) -> list[str]:
        args: list[str] = [self._gemini_cli_path, "-p", prompt, "-o", "stream-json"]
        if self._model_name:
            args.extend(["-m", self._model_name])
        if self._gemini_session_id:
            args.extend(["-r", self._gemini_session_id])
        return args

    def _build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if v is not None}
        env["PYTHONUNBUFFERED"] = "1"
        env["GEMINI_CLI_HOME"] = str(self._gemini_home or os.getcwd())
        if self._gemini_api_key:
            env["GEMINI_API_KEY"] = self._gemini_api_key

        env["CHACK_TOOLS_CONFIG_JSON"] = self._tools_config_json
        env["CHACK_ALLOWED_TOOLS_JSON"] = self._allowed_tools_json
        env["CHACK_MODEL_PROVIDER"] = self._model_provider
        env["CHACK_DEFAULT_MODEL"] = self._default_model
        env["CHACK_SOCIAL_NETWORK_MODEL"] = self._social_network_model
        env["CHACK_SCIENTIFIC_MODEL"] = self._scientific_model
        env["CHACK_WEBSEARCHER_MODEL"] = self._websearcher_model
        env["CHACK_TESTER_MODEL"] = self._tester_model
        env["CHACK_SOCIAL_NETWORK_MAX_TURNS"] = str(self._social_network_max_turns)
        env["CHACK_SCIENTIFIC_MAX_TURNS"] = str(self._scientific_max_turns)
        env["CHACK_WEBSEARCHER_MAX_TURNS"] = str(self._websearcher_max_turns)
        env["CHACK_TESTER_MAX_TURNS"] = str(self._tester_max_turns)
        env["CHACK_MIN_TOOLS_USED"] = str(self._min_tools_used)
        env["CHACK_MAX_TOOLS_USED"] = str(self._max_tools_used)
        env["CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST"] = (
            "1" if self._require_task_steps_manager_init_first else "0"
        )
        env["CHACK_TASK_SESSION_ID"] = str(current_session_id() or "")
        env["CHACK_RUN_LABEL"] = str(current_run_label() or "Run 1")
        env["CHACK_DISABLE_STDOUT_EVENTS"] = "1"
        return env

    def _ensure_gemini_home_and_config(self) -> None:
        if self._gemini_home:
            return
        session = _sanitize_session_id(current_session_id() or "default")
        base = os.path.join(
            os.path.expanduser(os.environ.get("CHACK_GEMINI_HOME", os.path.expanduser("~/.gemini/chack")) or ""),
            session,
        )
        os.makedirs(base, exist_ok=True)
        self._gemini_home = base
        self._write_gemini_settings(base)

    def _write_gemini_settings(self, gemini_home: str) -> None:
        settings_path = os.path.join(gemini_home, "settings.json")
        settings_payload = {
            "model": {
                "name": self._model_name,
                "maxSessionTurns": self._max_turns if self._max_turns > 0 else -1,
            },
            "tools": {
                "core": [],
            },
            "mcpServers": {
                "chack_tools": {
                    "command": sys.executable,
                    "args": ["-m", "chack_agent.backends.chack_tools_mcp_server"],
                    "env": self._gemini_mcp_env_map(),
                }
            },
        }
        with open(settings_path, "w", encoding="utf-8") as handle:
            json.dump(settings_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _gemini_mcp_env_map(self) -> dict[str, str]:
        env_keys = [
            "CHACK_TOOLS_CONFIG_JSON",
            "CHACK_ALLOWED_TOOLS_JSON",
            "CHACK_MODEL_PROVIDER",
            "CHACK_DEFAULT_MODEL",
            "CHACK_SOCIAL_NETWORK_MODEL",
            "CHACK_SCIENTIFIC_MODEL",
            "CHACK_WEBSEARCHER_MODEL",
            "CHACK_TESTER_MODEL",
            "CHACK_SOCIAL_NETWORK_MAX_TURNS",
            "CHACK_SCIENTIFIC_MAX_TURNS",
            "CHACK_WEBSEARCHER_MAX_TURNS",
            "CHACK_TESTER_MAX_TURNS",
            "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST",
            "CHACK_TASK_SESSION_ID",
            "CHACK_RUN_LABEL",
            "CHACK_DISABLE_STDOUT_EVENTS",
            "CHACK_MIN_TOOLS_USED",
            "CHACK_MAX_TOOLS_USED",
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "BRAVE_API_KEY",
            "SERPAPI_API_KEY",
            "FORUMSCOUT_API_KEY",
            "FORUMSCOUT_BASE_URL",
            "GH_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_PROFILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE",
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
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GEMINI_CLI_HOME",
        ]
        env_payload: dict[str, str] = {}
        src_env = self._build_env()
        for key in env_keys:
            value = src_env.get(key)
            if value is None:
                continue
            env_payload[key] = _safe_settings_value(value)

        if self._gemini_api_key:
            env_payload.setdefault("GEMINI_API_KEY", self._gemini_api_key)

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
                payload={
                    "tool": tool_name or "unknown",
                    "tool_input": tool_input,
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
        except Exception:
            pass

    @staticmethod
    def _sync_task_steps_manager(
        arguments: dict[str, Any],
        status: str,
        error: Any,
    ) -> None:
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

            raw_text = arguments.get("text")
            text = str(raw_text or "").strip()
            raw_status = arguments.get("status")
            status_value = str(raw_status or "").strip()
            raw_tasks = arguments.get("tasks")
            tasks = str(raw_tasks or "").strip()
            raw_notes = arguments.get("notes")
            notes = str(raw_notes or "").strip()

            TASK_STEPS_STORE.apply(
                session_id=session_id,
                run_label=run_label,
                action=str(arguments.get("action") or "").strip(),
                task_id=task_id,
                text=text,
                status=status_value,
                tasks_text=tasks,
                notes=notes,
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
) -> GeminiCliExecutor:
    del max_turns
    try:
        _LOGGER.debug(
            "gemini build_executor: memory_summary_max_chars=%s (not used in this backend)",
            int(memory_summary_max_chars),
        )
    except Exception:
        _LOGGER.debug(
            "gemini build_executor: memory_summary_max_chars provided (unable to coerce to int in debug log)"
        )

    def _extract_tool_names(items: list[Any] | None) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for tool in items or []:
            name = str(
                getattr(tool, "name", "") or getattr(tool, "__name__", "") or ""
            ).strip()
            if not name or name in seen:
                continue
            if name in _GEMINI_CLI_DENYLIST_TOOLS:
                continue
            seen.add(name)
            names.append(name)
        return names

    model_provider = str(config.model.provider or "").strip()
    if not model_provider:
        raise ValueError("model.provider must be defined in config")
    if model_provider != "gemini":
        raise ValueError(f"gemini backend requires model.provider='gemini' (got {model_provider!r})")

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
        if "tester_model" in init_params:
            toolset_kwargs["tester_model"] = config.model.tester
        if "tester_max_turns" in init_params:
            toolset_kwargs["tester_max_turns"] = config.model.tester_max_turns
        if "model_provider" in init_params:
            toolset_kwargs["model_provider"] = model_provider
        return toolset_kwargs

    if tools_override is not None:
        allowed_tool_names = _extract_tool_names(list(tools_override))
    elif tools_append:
        base_toolset = AgentsToolset(
            config.tools,
            **_build_toolset_kwargs(),
        )
        allowed_tool_names = _extract_tool_names(
            list(base_toolset.tools) + list(tools_append)
        )
    else:
        base_toolset = AgentsToolset(
            config.tools,
            **_build_toolset_kwargs(),
        )
        allowed_tool_names = _extract_tool_names(list(base_toolset.tools))

    gemini_api_key = (
        str(config.credentials.gemini_api_key or "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_api_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise ValueError(
            "GEMINI_API_KEY is required when model.provider=gemini or set GOOGLE_APPLICATION_CREDENTIALS/GOOGLE_CLOUD credentials."
        )

    configured_gemini_path = os.environ.get("GEMINI_CLI_PATH", "").strip() or "gemini"
    gemini_cli_path = shutil.which(configured_gemini_path) or configured_gemini_path

    return GeminiCliExecutor(
        conversation=[],
        memory_max_messages=memory_max_messages,
        memory_reset_to_messages=memory_reset_to_messages,
        base_system_prompt=system_prompt,
        model_name=str(config.model.primary),
        max_turns=int(config.session.max_turns or 100),
        gemini_cli_path=gemini_cli_path,
        gemini_api_key=gemini_api_key,
        tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False),
        allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False),
        model_provider=model_provider,
        default_model=str(config.model.primary or ""),
        social_network_model=str(config.model.social_network or ""),
        scientific_model=str(config.model.scientific or ""),
        websearcher_model=str(config.model.websearcher or ""),
        tester_model=str(config.model.tester or ""),
        social_network_max_turns=int(config.model.social_network_max_turns or 30),
        scientific_max_turns=int(config.model.scientific_max_turns or 30),
        websearcher_max_turns=int(config.model.websearcher_max_turns or 30),
        tester_max_turns=int(config.model.tester_max_turns or 30),
        min_tools_used=max(0, int(config.tools.min_tools_used or 0)),
        max_tools_used=max(0, int(config.tools.max_tools_used or 0)),
        require_task_steps_manager_init_first=bool(
            getattr(config.agent, "require_task_steps_manager_init_first", True)
        ),
        output_schema_json=(
            json.dumps(config.agent.output_schema_json, ensure_ascii=False)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        output_schema_name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
        output_schema_strict=bool(getattr(config.agent, "output_schema_strict", False)),
    )
