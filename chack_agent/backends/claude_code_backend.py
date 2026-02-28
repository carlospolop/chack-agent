from __future__ import annotations

import json
import logging
import os
import inspect
import re
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


_LOGGER = logging.getLogger("chack.claude_code_backend")


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]


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
    _claude_home: str | None = None
    _claude_session_id: str | None = None
    _output_schema: str | None = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_claude(prompt)

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
        if self._output_schema_json:
            schema_lines.append(
                "\n### OUTPUT CONTRACT\nReturn JSON only, exactly one JSON object."
            )
            schema_lines.append("Use schema name: output_schema")
            schema_lines.append("Schema:")
            schema_lines.append(self._output_schema_json)
            schema_lines.append("Match this schema as best as possible.")

        policy_block = ""
        schema_block = ""
        if policy_lines:
            policy_block = "\n\n### TOOL USAGE POLICY\n" + "\n".join(policy_lines)
        if schema_lines:
            schema_block = "\n" + "\n".join(schema_lines)

        prompt_parts = [p for p in (base, user_input, policy_block, schema_block) if p.strip()]
        return "\n".join(prompt_parts)

    def _run_claude(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_claude_home_and_settings()

        command = self._build_command(prompt)
        env = self._build_env()
        timeout_seconds = int(
            os.environ.get("CHACK_CLAUDE_EXEC_TIMEOUT_SECONDS", "900") or "900"
        )

        _LOGGER.info(
            "Starting Claude Code process: model=%s timeout_seconds=%s session_id=%s",
            self._model_name,
            timeout_seconds,
            self._claude_session_id or "",
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

        output_parts: list[str] = []
        steps: list[tuple[ToolAction, Any]] = []
        raw_lines: list[str] = []
        raw_responses: list[Any] = []
        tool_calls: dict[str, tuple[str, Any]] = {}
        started_at = time.monotonic()
        return_seen = False

        try:
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

                if event_type == "system" and subtype == "init":
                    session_id = str(event.get("session_id") or "").strip()
                    if session_id:
                        self._claude_session_id = session_id
                    continue

                if event_type == "assistant":
                    msg = event.get("message")
                    if isinstance(msg, dict):
                        self._extract_message_content(msg, output_parts, tool_calls, steps)
                    continue

                if event_type == "user":
                    msg = event.get("message")
                    if isinstance(msg, dict):
                        self._extract_message_content(msg, output_parts, tool_calls, steps)
                    continue

                if event_type == "result":
                    result_text = str(event.get("result") or "").strip()
                    if result_text:
                        output_parts.append(result_text)
                    return_seen = True

                    usage: dict[str, Any] = {}
                    if "usage" in event and isinstance(event.get("usage"), dict):
                        usage_dict = event["usage"]
                        if isinstance(usage_dict, dict):
                            usage = {
                                "input_tokens": int(usage_dict.get("input_tokens", 0) or 0),
                                "output_tokens": int(usage_dict.get("output_tokens", 0) or 0),
                                "input_tokens_details": {
                                    "cached_tokens": int(
                                        usage_dict.get("cached_tokens", usage_dict.get("cache_read_input_tokens", 0))
                                        or 0
                                    ),
                                    "cache_write_tokens": 0,
                                },
                            }
                    elif "input_tokens" in event or "output_tokens" in event:
                        usage = {
                            "input_tokens": int(event.get("input_tokens", 0) or 0),
                            "output_tokens": int(event.get("output_tokens", 0) or 0),
                            "input_tokens_details": {
                                "cached_tokens": int(
                                    event.get("cached_input_tokens", event.get("cache_read_input_tokens", 0))
                                    or 0
                                ),
                                "cache_write_tokens": 0,
                            },
                        }
                    if usage:
                        raw_responses.append({"usage": usage})

                    if str(event.get("subtype") or "").strip().lower() == "error":
                        return (
                            "ERROR: Claude returned an error in final result event. "
                            + (result_text or "No error text was returned."),
                            steps,
                            _RawResult(raw_responses=raw_responses),
                        )
                    break

                if event_type == "tool_use":
                    self._record_tool_use(event, tool_calls)
                    continue

                if event_type == "tool_result":
                    self._record_tool_result(event, tool_calls, steps)
                    continue
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass

        return_code = process.wait()

        if return_code != 0:
            details = "\n".join(raw_lines).strip() or "No output captured."
            if not return_seen:
                return (
                    f"ERROR: Claude exec failed (exit={return_code}). {details}",
                    steps,
                    _RawResult(raw_responses=raw_responses),
                )

        response = "".join(output_parts).strip()
        if not response and raw_lines:
            response = "\n".join(raw_lines).strip()
            if response:
                response = response[-4000:]

        return response, steps, _RawResult(raw_responses=raw_responses)

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

        if str(step.tool).strip() == "task_steps_manager":
            self._sync_task_steps_manager(tool_input, status)

    def _build_command(self, prompt: str) -> list[str]:
        args: list[str] = [
            self._claude_cli_path,
            "-p",
            prompt,
            "--print",
            "--output-format",
            "stream-json",
            "--tools",
            "",
            "--mcp-config",
            os.path.join(self._claude_home or os.getcwd(), "settings.json"),
            "--strict-mcp-config",
        ]

        if self._max_turns > 0:
            args.extend(["--max-turns", str(self._max_turns)])
        if self._model_name:
            args.extend(["--model", self._model_name])
        if self._claude_session_id:
            args.extend(["--resume", self._claude_session_id])
        if self._output_schema_json:
            args.extend(["--json-schema", self._output_schema_json])

        return args

    def _build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if v is not None}
        env["PYTHONUNBUFFERED"] = "1"

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

        env["ANTHROPIC_API_KEY"] = str(
            os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("CLAUDE_API_KEY", "")
        )

        env.setdefault("AWS_SHARED_CREDENTIALS_FILE", os.environ.get("AWS_SHARED_CREDENTIALS_FILE", ""))
        env.setdefault("AWS_CONFIG_FILE", os.environ.get("AWS_CONFIG_FILE", ""))

        return env

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
        settings_payload = {
            "mcpServers": {
                "chack_tools": {
                    "command": sys.executable,
                    "args": ["-m", "chack_agent.backends.chack_tools_mcp_server"],
                    "env": self._mcp_env_map(),
                }
            }
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

    def _mcp_env_map(self) -> dict[str, str]:
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
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
        ]

        src_env = self._build_env()
        env_payload: dict[str, str] = {}
        for key in env_keys:
            value = src_env.get(key)
            if value is None:
                continue
            env_payload[key] = str(value)

        if self._output_schema is not None:
            env_payload["CHACK_OUTPUT_SCHEMA"] = json.dumps(self._output_schema, ensure_ascii=False)

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
        base_toolset = AgentsToolset(config.tools, **_build_toolset_kwargs())
        allowed_tool_names = _extract_tool_names(list(base_toolset.tools) + list(tools_append))
    else:
        base_toolset = AgentsToolset(config.tools, **_build_toolset_kwargs())
        allowed_tool_names = _extract_tool_names(list(base_toolset.tools))

    configured_claude_path = os.environ.get("CLAUDE_CLI_PATH", "").strip() or "claude"
    claude_cli_path = shutil.which(configured_claude_path) or configured_claude_path

    return ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _model_name=str(config.model.primary),
        _max_turns=int(config.session.max_turns or 100),
        _claude_cli_path=claude_cli_path,
        _tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False),
        _allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False),
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
            json.dumps(config.agent.output_schema_json, ensure_ascii=False)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
    )
