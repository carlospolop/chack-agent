from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
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


_LOGGER = logging.getLogger("chack.codex_backend")


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    _codex_path: str
    _openai_api_key: str
    _tool_profile: str
    _tools_config_json: str
    _allowed_tools_json: str
    _model_provider: str
    _default_model: str
    _social_network_model: str
    _scientific_model: str
    _websearcher_model: str
    _tester_model: str
    _social_network_max_turns: int
    _scientific_max_turns: int
    _websearcher_max_turns: int
    _tester_max_turns: int
    _min_tools_used: int
    _max_tools_used: int
    _require_task_steps_manager_init_first: bool
    _output_schema_json: str
    _output_schema_name: str
    _output_schema_strict: bool
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

        output_schema_block = ""
        if self._output_schema_json:
            schema_name = str(self._output_schema_name or "output_schema").strip() or "output_schema"
            strict_text = "true" if self._output_schema_strict else "false"
            output_schema_block = (
                "\n\n### OUTPUT FORMAT (REQUIRED)\n"
                "Your final response must be valid JSON matching exactly this JSON Schema.\n"
                "Return ONLY the JSON object, with no markdown and no extra text.\n"
                f"Schema name: {schema_name}\n"
                f"Strict: {strict_text}\n"
                f"Schema:\n{self._output_schema_json}"
            )

        if not base:
            return f"{user_input}{policy_block}{output_schema_block}" if (policy_block or output_schema_block) else user_input
        if not user_input:
            return f"{base}{policy_block}{output_schema_block}" if (policy_block or output_schema_block) else base
        return f"{base}{policy_block}{output_schema_block}\n\n### USER REQUEST\n{user_input}"

    def _run_codex(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_codex_home_and_config()
        command = self._build_command()
        env = self._build_env()
        timeout_seconds = int(os.environ.get("CHACK_CODEX_EXEC_TIMEOUT_SECONDS", "900") or "900")
        _LOGGER.info(
            "Starting Codex CLI process: model=%s timeout_seconds=%s thread_id=%s ts=%s",
            self._model_name,
            timeout_seconds,
            self._thread_id or "",
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
            )
        except FileNotFoundError:
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
            return (
                f"ERROR: Failed to launch Codex CLI: {type(exc).__name__}: {exc}",
                [],
                _RawResult(raw_responses=[]),
            )

        steps: list[tuple[ToolAction, Any]] = []
        output = ""
        usage_payload: dict[str, Any] | None = None
        combined_output_lines: list[str] = []
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

            if event_type == "item.completed":
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                item_type = str(item.get("type", "") or "")

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
                    output = str(item.get("text", "") or "")
                    continue

                step = self._item_to_step(item)
                if step is not None:
                    steps.append((step, None))
                    self._log_tool_called(step.tool, step.tool_input)
                    self._sync_task_steps_manager(item)
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

        return_code = process.wait()
        if return_code != 0:
            details = "\n".join(combined_output_lines).strip() or "No error output captured."
            return (
                f"ERROR: Codex exec failed (exit={return_code}).\n{details}",
                steps,
                _RawResult(raw_responses=[]),
            )

        raw_responses: list[Any] = []
        if usage_payload is not None:
            raw_responses.append({"usage": usage_payload})
        return output, steps, _RawResult(raw_responses=raw_responses)

    def _build_command(self) -> list[str]:
        if self._thread_id:
            return [
                self._codex_path,
                "exec",
                "resume",
                "--json",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--model",
                self._model_name,
                self._thread_id,
                "-",
            ]
        return [
            self._codex_path,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            os.getcwd(),
            "--model",
            self._model_name,
            "-",
        ]

    def _build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if v is not None}
        env.setdefault("OPENAI_API_KEY", self._openai_api_key)
        env.setdefault("CODEX_API_KEY", self._openai_api_key)
        if self._codex_home:
            env["CODEX_HOME"] = self._codex_home
        env["CHACK_TOOLS_CONFIG_JSON"] = self._tools_config_json
        env["CHACK_ALLOWED_TOOLS_JSON"] = self._allowed_tools_json
        env["CHACK_TOOL_PROFILE"] = self._tool_profile
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

    def _write_codex_config(self, codex_home: str) -> None:
        os.makedirs(codex_home, exist_ok=True)
        config_path = os.path.join(codex_home, "config.toml")
        python_cmd = sys.executable or "python3"
        env_vars = [
            "CHACK_TOOLS_CONFIG_JSON",
            "CHACK_ALLOWED_TOOLS_JSON",
            "CHACK_TOOL_PROFILE",
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
        ]

        def _toml_string(value: str) -> str:
            return json.dumps(str(value))

        env_vars_toml = "[" + ", ".join(_toml_string(v) for v in env_vars) + "]"
        args_toml = "[" + ", ".join(
            _toml_string(v)
            for v in ["-m", "chack_agent.backends.chack_tools_mcp_server"]
        ) + "]"

        config_body = "\n".join(
            [
                f"model = {_toml_string(self._model_name)}",
                "",
                "[mcp_servers.chack_tools]",
                f"command = {_toml_string(python_cmd)}",
                f"args = {args_toml}",
                f"env_vars = {env_vars_toml}",
                "required = true",
                "startup_timeout_sec = 30",
                "tool_timeout_sec = 120",
            ]
        )
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(config_body + "\n")

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
    tool_profile: str = "all",
    tools_override: Optional[list[Any]] = None,
    tools_append: Optional[list[Any]] = None,
) -> CodexExecutor:
    del max_turns

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
    model_provider = str(config.model.provider or "").strip()
    if not model_provider:
        raise ValueError("model.provider must be defined in config")
    if tools_override is not None:
        allowed_tool_names = _extract_tool_names(list(tools_override))
    elif tools_append:
        base_toolset = AgentsToolset(
            config.tools,
            tool_profile=tool_profile,
            model_provider=model_provider,
            default_model=config.model.primary,
            social_network_model=config.model.social_network,
            scientific_model=config.model.scientific,
            websearcher_model=config.model.websearcher,
            tester_model=config.model.tester,
            social_network_max_turns=config.model.social_network_max_turns,
            scientific_max_turns=config.model.scientific_max_turns,
            websearcher_max_turns=config.model.websearcher_max_turns,
            tester_max_turns=config.model.tester_max_turns,
        )
        allowed_tool_names = _extract_tool_names(list(base_toolset.tools) + list(tools_append))

    openai_api_key = (
        str(config.credentials.openai_api_key or "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when model.provider=codex")

    configured_codex_path = os.environ.get("CODEX_PATH", "").strip() or "codex"
    codex_path = shutil.which(configured_codex_path) or configured_codex_path


    return CodexExecutor(
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _model_name=str(config.model.primary),
        _codex_path=codex_path,
        _openai_api_key=openai_api_key,
        _tool_profile=str(tool_profile or "all"),
        _tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False),
        _allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False)
        if allowed_tool_names is not None
        else "",
        _model_provider=model_provider,
        _default_model=str(config.model.primary or ""),
        _social_network_model=str(config.model.social_network or ""),
        _scientific_model=str(config.model.scientific or ""),
        _websearcher_model=str(config.model.websearcher or ""),
        _tester_model=str(config.model.tester or ""),
        _social_network_max_turns=int(config.model.social_network_max_turns or 30),
        _scientific_max_turns=int(config.model.scientific_max_turns or 30),
        _websearcher_max_turns=int(config.model.websearcher_max_turns or 30),
        _tester_max_turns=int(config.model.tester_max_turns or 30),
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
        _output_schema_name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
        _output_schema_strict=bool(getattr(config.agent, "output_schema_strict", True)),
    )
