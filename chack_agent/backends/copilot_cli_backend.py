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
from ..live_cost_state import report_live_usage
from ..openrouter_routing import clone_config_for_openrouter, get_openrouter_route
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .tool_payloads import (
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    serialize_tools_payload,
)


_LOGGER = logging.getLogger("chack.copilot_cli_backend")


def _sanitize_session_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(session_id or "").strip()).strip()


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]


class CopilotCliExecutor:
    def __init__(
        self,
        *,
        conversation: list[dict[str, Any]],
        memory_max_messages: int,
        memory_reset_to_messages: int,
        base_system_prompt: str,
        model_name: str,
        max_turns: int,
        copilot_cli_path: str,
        copilot_github_token: str,
        tools_config_json: str,
        allowed_tools_json: str,
        serialized_tools_override_b64: str,
        serialized_tools_append_b64: str,
        model_provider: str,
        default_model: str,
        social_network_model: str,
        scientific_model: str,
        websearcher_model: str,
        tester_model: str,
        subchack_model: str,
        social_network_max_turns: int,
        scientific_max_turns: int,
        websearcher_max_turns: int,
        tester_max_turns: int,
        subchack_max_turns: int,
        min_tools_used: int,
        max_tools_used: int,
        require_task_steps_manager_init_first: bool,
        output_schema_json: str,
    ) -> None:
        self._conversation = conversation
        self._memory_limit = memory_max_messages
        self._memory_reset_to = memory_reset_to_messages
        self._base_system_prompt = base_system_prompt
        self._model_name = str(model_name or "").strip()
        self._max_turns = int(max_turns or 0)
        self._copilot_cli_path = copilot_cli_path
        self._copilot_github_token = copilot_github_token
        self._tools_config_json = tools_config_json
        self._allowed_tools_json = allowed_tools_json
        self._serialized_tools_override_b64 = str(serialized_tools_override_b64 or "")
        self._serialized_tools_append_b64 = str(serialized_tools_append_b64 or "")
        self._model_provider = model_provider
        self._default_model = default_model
        self._social_network_model = social_network_model
        self._scientific_model = scientific_model
        self._websearcher_model = websearcher_model
        self._tester_model = tester_model
        self._subchack_model = subchack_model
        self._social_network_max_turns = social_network_max_turns
        self._scientific_max_turns = scientific_max_turns
        self._websearcher_max_turns = websearcher_max_turns
        self._tester_max_turns = tester_max_turns
        self._subchack_max_turns = subchack_max_turns
        self._min_tools_used = max(0, int(min_tools_used or 0))
        self._max_tools_used = max(0, int(max_tools_used or 0))
        self._require_task_steps_manager_init_first = bool(
            require_task_steps_manager_init_first
        )
        self._output_schema_json = output_schema_json or ""

        self._copilot_home: str | None = None
        self._copilot_session_id: str | None = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_copilot(prompt)

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

    # ------------------------------------------------------------------
    #  Core execution
    # ------------------------------------------------------------------

    def _run_copilot(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_copilot_home_and_config()
        command = self._build_command(prompt)
        env = self._build_env()
        timeout_seconds = int(
            os.environ.get("CHACK_COPILOT_EXEC_TIMEOUT_SECONDS", "900") or "900"
        )

        _LOGGER.info(
            "Starting Copilot CLI process: model=%s timeout_seconds=%s session_id=%s",
            self._model_name,
            timeout_seconds,
            self._copilot_session_id or "",
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
                "ERROR: Copilot CLI executable was not found. "
                f"Configured path: {self._copilot_cli_path!r}. "
                "Install Copilot CLI (e.g. brew install copilot-cli) "
                "or set COPILOT_CLI_PATH to the absolute executable path.",
                [],
                _RawResult(raw_responses=[]),
            )
        except Exception as exc:
            return (
                f"ERROR: Failed to launch Copilot CLI: {type(exc).__name__}: {exc}",
                [],
                _RawResult(raw_responses=[]),
            )

        output_parts: list[str] = []
        delta_parts: list[str] = []
        steps: list[tuple[ToolAction, Any]] = []
        raw_lines: list[str] = []
        raw_responses: list[Any] = []
        tool_calls: dict[str, tuple[str, Any]] = {}
        started_at = time.monotonic()

        try:
            # Close stdin immediately — prompt is passed via -p flag
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass

            while True:
                if (time.monotonic() - started_at) > timeout_seconds:
                    process.kill()
                    return (
                        f"ERROR: Copilot execution timed out after {timeout_seconds}s.",
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

                event_type = str(event.get("type") or "").strip()
                data = event.get("data")
                if not isinstance(data, dict):
                    data = {}

                # -- Session / setup events --------------------------------

                if event_type == "session.tools_updated":
                    # Indicates the model being used
                    continue

                if event_type.startswith("session."):
                    # session.mcp_server_status_changed, session.mcp_servers_loaded, etc.
                    continue

                if event_type == "user.message":
                    continue

                # -- Turn lifecycle ----------------------------------------

                if event_type == "assistant.turn_start":
                    continue

                if event_type == "assistant.turn_end":
                    continue

                # -- Reasoning (ephemeral) ---------------------------------

                if event_type in ("assistant.reasoning_delta", "assistant.reasoning"):
                    continue

                # -- Message deltas (ephemeral) ----------------------------

                if event_type == "assistant.message_delta":
                    delta_text = ""
                    raw_delta = data.get("content") or data.get("delta") or data.get("text")
                    if isinstance(raw_delta, str):
                        delta_text = raw_delta
                    elif isinstance(raw_delta, dict):
                        delta_text = str(raw_delta.get("text") or raw_delta.get("content") or "")
                    elif isinstance(raw_delta, list):
                        for part in raw_delta:
                            if isinstance(part, dict):
                                delta_text += str(part.get("text") or part.get("content") or "")
                            elif isinstance(part, str):
                                delta_text += part
                    if delta_text:
                        delta_parts.append(delta_text)
                    continue

                # -- Complete assistant message ----------------------------

                if event_type == "assistant.message":
                    raw_content = data.get("content")
                    content = ""
                    if isinstance(raw_content, str):
                        content = raw_content.strip()
                    elif isinstance(raw_content, list):
                        # Handle array-format content blocks: [{"type":"text","text":"..."}]
                        for part in raw_content:
                            if isinstance(part, dict):
                                content += str(part.get("text") or part.get("content") or "")
                            elif isinstance(part, str):
                                content += part
                        content = content.strip()
                    elif isinstance(raw_content, dict):
                        content = str(raw_content.get("text") or raw_content.get("content") or "").strip()
                    if content:
                        output_parts.append(content)

                    # Process tool requests
                    tool_requests = data.get("toolRequests")
                    if isinstance(tool_requests, list):
                        for req in tool_requests:
                            if not isinstance(req, dict):
                                continue
                            tool_call_id = str(req.get("toolCallId") or "").strip()
                            tool_name = str(req.get("name") or "").strip()
                            arguments = req.get("arguments")
                            if not isinstance(arguments, dict):
                                arguments = {}
                            if tool_call_id:
                                tool_calls[tool_call_id] = (tool_name, arguments)
                    continue

                # -- Tool execution events ---------------------------------

                if event_type == "tool.execution_start":
                    # Already captured from assistant.message.toolRequests
                    tool_call_id = str(data.get("toolCallId") or "").strip()
                    tool_name = str(data.get("toolName") or "").strip()
                    arguments = data.get("arguments")
                    if not isinstance(arguments, dict):
                        arguments = {}
                    if tool_call_id and tool_call_id not in tool_calls:
                        tool_calls[tool_call_id] = (tool_name, arguments)
                    continue

                if event_type == "tool.execution_complete":
                    tool_call_id = str(data.get("toolCallId") or "").strip()
                    success = bool(data.get("success", True))
                    result_data = data.get("result")
                    if isinstance(result_data, dict):
                        result_content = str(result_data.get("content") or result_data.get("detailedContent") or "")
                    elif result_data is not None:
                        result_content = str(result_data)
                    else:
                        result_content = ""

                    tool_name = ""
                    tool_input: Any = {}
                    if tool_call_id and tool_call_id in tool_calls:
                        tool_name, tool_input = tool_calls.pop(tool_call_id)
                    if not tool_name:
                        tool_name = str(data.get("toolName") or "tool")

                    step_payload = {
                        "tool_id": tool_call_id,
                        "status": "success" if success else "error",
                        "tool_input": tool_input,
                    }
                    if result_content:
                        step_payload["result"] = result_content

                    step = ToolAction(tool=tool_name, tool_input=step_payload)
                    steps.append((step, None))
                    self._log_tool_called(step.tool, step.tool_input)

                    # Sync task steps manager if applicable
                    if tool_name == "task_steps_manager":
                        self._sync_task_steps_manager(
                            tool_input,
                            "success" if success else "error",
                        )
                    continue

                # -- Final result ------------------------------------------

                if event_type == "result":
                    session_id = str(event.get("sessionId") or "").strip()
                    if session_id:
                        self._copilot_session_id = session_id

                    # Capture any content/message in the result event
                    for key in ("content", "message", "text", "output"):
                        result_content = event.get(key) or data.get(key)
                        if isinstance(result_content, str) and result_content.strip():
                            output_parts.append(result_content.strip())
                            break
                        elif isinstance(result_content, list):
                            text_bits = []
                            for part in result_content:
                                if isinstance(part, dict):
                                    text_bits.append(str(part.get("text") or part.get("content") or ""))
                                elif isinstance(part, str):
                                    text_bits.append(part)
                            joined = "".join(text_bits).strip()
                            if joined:
                                output_parts.append(joined)
                                break

                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        api_duration_ms = int(usage.get("totalApiDurationMs", 0) or 0)
                        raw_responses.append({
                            "usage": {
                                "premiumRequests": int(usage.get("premiumRequests", 0) or 0),
                                "totalApiDurationMs": api_duration_ms,
                                "sessionDurationMs": int(usage.get("sessionDurationMs", 0) or 0),
                            }
                        })
                    break

        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass

        return_code = process.wait()

        if return_code != 0:
            details = "\n".join(raw_lines).strip() or "No output captured."
            return (
                f"ERROR: Copilot exec failed (exit={return_code}). {details}",
                steps,
                _RawResult(raw_responses=raw_responses),
            )

        response = "".join(output_parts).strip()
        if not response and delta_parts:
            response = "".join(delta_parts).strip()
        if not response and raw_lines:
            response = "\n".join(raw_lines).strip()
            if response:
                response = response[-4000:]

        return response, steps, _RawResult(raw_responses=raw_responses)

    # ------------------------------------------------------------------
    #  Command / environment building
    # ------------------------------------------------------------------

    def _build_command(self, prompt: str) -> list[str]:
        args: list[str] = [
            self._copilot_cli_path,
            "-p",
            prompt,
            "--allow-all",
            "--output-format",
            "json",
        ]
        if self._model_name:
            args.extend(["--model", self._model_name])
        if self._copilot_session_id:
            args.extend(["--resume", self._copilot_session_id])
        if self._copilot_home:
            mcp_config_path = os.path.join(self._copilot_home, "mcp-config.json")
            if os.path.isfile(mcp_config_path):
                args.extend(["--additional-mcp-config", f"@{mcp_config_path}"])
        return args

    def _build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if v is not None}
        env["PYTHONUNBUFFERED"] = "1"

        # Copilot CLI auth — classic PATs (ghp_) are rejected by copilot CLI,
        # so only pass the token when it is NOT a classic PAT.
        if self._copilot_github_token:
            if self._copilot_github_token.startswith("ghp_"):
                _LOGGER.warning(
                    "Classic PAT (ghp_) detected — not setting COPILOT_GITHUB_TOKEN. "
                    "Copilot CLI requires a fine-grained PAT or OAuth token."
                )
                # Remove any classic-PAT env vars so copilot falls through to
                # its own auth (e.g. `copilot login` or stored OAuth session).
                for key in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
                    env.pop(key, None)
            else:
                env["COPILOT_GITHUB_TOKEN"] = self._copilot_github_token

        if self._copilot_home:
            env["COPILOT_HOME"] = self._copilot_home

        env["CHACK_TOOLS_CONFIG_JSON"] = self._tools_config_json
        env["CHACK_ALLOWED_TOOLS_JSON"] = self._allowed_tools_json
        if self._serialized_tools_override_b64:
            env[CHACK_TOOLS_OVERRIDE_B64_ENV] = self._serialized_tools_override_b64
        if self._serialized_tools_append_b64:
            env[CHACK_TOOLS_APPEND_B64_ENV] = self._serialized_tools_append_b64
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

    # ------------------------------------------------------------------
    #  Home / MCP config
    # ------------------------------------------------------------------

    def _ensure_copilot_home_and_config(self) -> None:
        if self._copilot_home:
            return

        session = _sanitize_session_id(current_session_id() or "default")
        base = os.path.join(
            os.path.expanduser(
                os.environ.get("CHACK_COPILOT_HOME_BASE", os.path.expanduser("~/.copilot/chack"))
            ),
            session,
        )
        os.makedirs(base, exist_ok=True)
        self._copilot_home = base
        self._write_copilot_mcp_config(base)

    def _write_copilot_mcp_config(self, copilot_home: str) -> None:
        mcp_config_path = os.path.join(copilot_home, "mcp-config.json")
        mcp_payload: dict[str, Any] = {
            "mcpServers": {
                "chack_tools": {
                    "command": sys.executable,
                    "args": ["-m", "chack_agent.backends.chack_tools_mcp_server"],
                    "env": self._copilot_mcp_env_map(),
                }
            }
        }
        if self._playwright_mcp_enabled():
            mcp_payload["mcpServers"]["playwright"] = playwright_mcp_server_config()

        with open(mcp_config_path, "w", encoding="utf-8") as handle:
            json.dump(mcp_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _playwright_mcp_enabled(self) -> bool:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        return bool(cfg.get("playwright_enabled")) and playwright_mcp_is_available()

    def _copilot_mcp_env_map(self) -> dict[str, str]:
        env_keys = [
            "CHACK_TOOLS_CONFIG_JSON",
            "CHACK_ALLOWED_TOOLS_JSON",
            "CHACK_TOOLS_OVERRIDE_B64",
            "CHACK_TOOLS_APPEND_B64",
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
            "CHACK_MIN_TOOLS_USED",
            "CHACK_MAX_TOOLS_USED",
            "AISEC_LOCAL_VULN_STORE_PATH",
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "BRAVE_API_KEY",
            "SERPAPI_API_KEY",
            "FORUMSCOUT_API_KEY",
            "FORUMSCOUT_BASE_URL",
            "GH_TOKEN",
            "COPILOT_GITHUB_TOKEN",
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
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
            "GEMINI_API_KEY",
        ]

        src_env = self._build_env()
        env_payload: dict[str, str] = {}
        for key in env_keys:
            value = src_env.get(key)
            if value is None:
                continue
            env_payload[key] = str(value)
        return env_payload

    # ------------------------------------------------------------------
    #  Event parsing helpers
    # ------------------------------------------------------------------

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


# ======================================================================
#  Builder
# ======================================================================


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
) -> CopilotCliExecutor:
    if get_openrouter_route(config) is not None:
        from .openrouter_openai_backend import build_executor as build_openrouter_executor

        return build_openrouter_executor(
            clone_config_for_openrouter(config),
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
            "copilot build_executor: memory_summary_max_chars=%s (not used in this backend)",
            int(memory_summary_max_chars),
        )
    except Exception:
        _LOGGER.debug(
            "copilot build_executor: memory_summary_max_chars provided (unable to coerce to int in debug log)"
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
    if model_provider not in {"copilot", "copilot-cli", "copilot_cli", "gh-copilot", "gh_copilot"}:
        raise ValueError(
            f"copilot backend requires model.provider='copilot' (got {model_provider!r})"
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
        if "subchack_model" in init_params:
            toolset_kwargs["subchack_model"] = config.model.subchack
        if "subchack_max_turns" in init_params:
            toolset_kwargs["subchack_max_turns"] = config.model.subchack_max_turns
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

    # Resolve Copilot GitHub token from config or environment
    copilot_github_token = (
        str(getattr(config.credentials, "copilot_github_token", "") or "").strip()
        or os.environ.get("COPILOT_GITHUB_TOKEN", "").strip()
        or os.environ.get("GH_TOKEN", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )
    if not copilot_github_token:
        raise ValueError(
            "Copilot backend requires a GitHub token. "
            "Set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN, "
            "or configure credentials.copilot_github_token in the config."
        )

    configured_copilot_path = os.environ.get("COPILOT_CLI_PATH", "").strip() or "copilot"
    copilot_cli_path = shutil.which(configured_copilot_path) or configured_copilot_path
    serialized_tools_override_b64 = serialize_tools_payload(tools_override)
    serialized_tools_append_b64 = serialize_tools_payload(tools_append)

    return CopilotCliExecutor(
        conversation=[],
        memory_max_messages=memory_max_messages,
        memory_reset_to_messages=memory_reset_to_messages,
        base_system_prompt=system_prompt,
        model_name=str(config.model.primary),
        max_turns=int(config.session.max_turns or 100),
        copilot_cli_path=copilot_cli_path,
        copilot_github_token=copilot_github_token,
        tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False),
        allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False),
        serialized_tools_override_b64=serialized_tools_override_b64,
        serialized_tools_append_b64=serialized_tools_append_b64,
        model_provider=model_provider,
        default_model=str(config.model.primary or ""),
        social_network_model=str(config.model.social_network or ""),
        scientific_model=str(config.model.scientific or ""),
        websearcher_model=str(config.model.websearcher or ""),
        tester_model=str(config.model.tester or ""),
        subchack_model=str(config.model.subchack or ""),
        social_network_max_turns=int(config.model.social_network_max_turns or 30),
        scientific_max_turns=int(config.model.scientific_max_turns or 30),
        websearcher_max_turns=int(config.model.websearcher_max_turns or 30),
        tester_max_turns=int(config.model.tester_max_turns or 30),
        subchack_max_turns=int(config.model.subchack_max_turns or 30),
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
    )
