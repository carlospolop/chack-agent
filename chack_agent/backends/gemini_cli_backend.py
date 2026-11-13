from __future__ import annotations

import json
import logging
import os
import re
import inspect
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from chack_tools.cancellation import cancellation_requested, register_process, unregister_process
from chack_tools.telemetry import OTEL_CHILD_ENV_NAMES, child_process_env
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.task_steps_manager_state import (
    STORE as TASK_STEPS_STORE,
    current_run_label,
    current_session_id,
)
from chack_tools.telemetry import log_event
from chack_tools.tool_usage_state import effective_max_tools_used

from ..config import ChackConfig
from ..live_cost_state import report_live_usage
from ..output_schema import repair_schema_output
from ..resume_compaction import ResumeCompactionResult
from ..llm_gateway import (
    LLM_GATEWAY_ENV_GEMINI_BASE_URL,
    LLM_GATEWAY_ENV_TYPE,
    clone_config_for_gateway,
    gateway_subprocess_env,
    get_gateway_route,
)
from ..thinking_effort import gemini_thinking_config, normalize_thinking_effort
from .claude_code_backend import _terminate_process_tree
from .cli_process import END_OF_OUTPUT, CliLineReader, cli_exec_timeout_seconds
from .gemini_gateway_relay import GeminiGatewayRelay
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .tool_payloads import (
    configured_mcp_env_names,
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV,
    CHACK_TOOLS_APPEND_NAMES_JSON_ENV,
    augment_subprocess_pythonpath,
    serialize_tools_payload,
    CHACK_ALLOWED_TOOLS_JSON_PATH_ENV,
    CHACK_INLINE_ENV_VALUE_MAX_CHARS,
    CHACK_TOOLS_APPEND_B64_PATH_ENV,
    CHACK_TOOLS_CONFIG_JSON_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_B64_PATH_ENV,
    write_payload_to_file,
)


_LOGGER = logging.getLogger("chack.gemini_cli_backend")

GEMINI_EXEC_TIMEOUT_ENV = "CHACK_GEMINI_EXEC_TIMEOUT_SECONDS"
GEMINI_CANCELLED_OUTPUT = "ERROR: Gemini execution cancelled."
# Gemini CLI ends a run that reaches maxSessionTurns with this error instead
# of an answer. The run is resumed once to ask for the answer; the CLI counts
# turns per process, so that run gets a short budget of its own: the answer,
# plus one tool call the model may still make.
_TURN_LIMIT_ERROR = "ERROR: Gemini result error: Reached max session turns"
_FINAL_ANSWER_TURNS = 2
_FINAL_ANSWER_PROMPT = (
    "You have used all the turns of this run. Do not call any more tools: give your final answer now from "
    "the evidence gathered so far, in the required format, and state what remains unverified."
)


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
}


def _sanitize_session_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(session_id or "").strip()).strip()


def _safe_settings_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _send_prompt(process: subprocess.Popen, prompt: str) -> None:
    """Write the prompt to the CLI's stdin without blocking on the pipe buffer.

    Gemini CLI gives piped input only 500 ms to start arriving, so the write
    starts at launch; a thread keeps prompts larger than the pipe buffer from
    blocking the caller before it begins draining stdout.
    """

    def write() -> None:
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    threading.Thread(target=write, name="gemini-cli-prompt", daemon=True).start()


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class _RawResult:
    raw_responses: list[Any]
    time_to_first_token_seconds: float | None = None
    time_to_first_token_source: str = "unavailable"


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
        serialized_tools_override_b64: str,
        serialized_tools_append_b64: str,
        model_provider: str,
        default_model: str,
        social_network_model: str,
        scientific_model: str,
        websearcher_model: str,
        business_model: str,
        product_model: str,
        cli_model: str,
        subchack_model: str,
        social_network_max_turns: int,
        scientific_max_turns: int,
        websearcher_max_turns: int,
        business_max_turns: int,
        product_max_turns: int,
        cli_max_turns: int,
        subchack_max_turns: int,
        min_tools_used: int,
        max_tools_used: int,
        require_task_steps_manager_init_first: bool,
        output_schema_json: str,
        output_schema_name: str,
        output_schema_strict: bool,
        thinking_effort: str = "high",
        travel_model: str = "",
        travel_max_turns: int = 50,
        serialized_tools_override_names_json: str = "",
        serialized_tools_append_names_json: str = "",
        max_context_tokens: int = 0,
        compaction_threshold_ratio: float = 0.75,
        gemini_base_url: str = "",
        gemini_headers: dict[str, str] | None = None,
        gateway_env: dict[str, str] | None = None,
    ) -> None:
        self._conversation = conversation
        self._memory_limit = memory_max_messages
        self._memory_reset_to = memory_reset_to_messages
        self._base_system_prompt = base_system_prompt
        self._model_name = str(model_name or "").strip()
        self._max_turns = int(max_turns or 0)
        # Replaces max_turns for the run that asks for an answer after the limit.
        self._turn_budget_override = 0
        self._gemini_cli_path = gemini_cli_path
        self._gemini_api_key = gemini_api_key
        # Set when the Gemini API is reached through an LLM gateway.
        self._gemini_base_url = str(gemini_base_url or "").strip()
        self._gemini_headers = dict(gemini_headers or {})
        # Lets the Chack tool server (and the sub-agents it starts) reach the
        # same gateway.
        self._gateway_env = dict(gateway_env or {})
        self._tools_config_json = tools_config_json
        self._allowed_tools_json = allowed_tools_json
        self._serialized_tools_override_b64 = str(serialized_tools_override_b64 or "")
        self._serialized_tools_append_b64 = str(serialized_tools_append_b64 or "")
        self._model_provider = model_provider
        self._default_model = default_model
        self._social_network_model = social_network_model
        self._scientific_model = scientific_model
        self._websearcher_model = websearcher_model
        self._business_model = business_model
        self._product_model = product_model
        self._travel_model = travel_model
        self._cli_model = cli_model
        self._subchack_model = subchack_model
        self._social_network_max_turns = social_network_max_turns
        self._scientific_max_turns = scientific_max_turns
        self._websearcher_max_turns = websearcher_max_turns
        self._business_max_turns = business_max_turns
        self._product_max_turns = product_max_turns
        self._travel_max_turns = travel_max_turns
        self._cli_max_turns = cli_max_turns
        self._subchack_max_turns = subchack_max_turns
        self._min_tools_used = max(0, int(min_tools_used or 0))
        self._max_tools_used = max(0, int(max_tools_used or 0))
        self._require_task_steps_manager_init_first = bool(
            require_task_steps_manager_init_first
        )
        self._output_schema_json = output_schema_json or ""
        self._output_schema_name = output_schema_name or "output_schema"
        self._output_schema_strict = bool(output_schema_strict)
        self._thinking_effort = normalize_thinking_effort(thinking_effort)
        self._serialized_tools_override_names_json = str(
            serialized_tools_override_names_json or ""
        )
        self._serialized_tools_append_names_json = str(
            serialized_tools_append_names_json or ""
        )
        self._max_context_tokens = max(0, int(max_context_tokens or 0))
        self._compaction_threshold_ratio = float(compaction_threshold_ratio or 0)

        self._gemini_home: str | None = None
        self._gemini_session_id: str | None = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_gemini(prompt)
        if output.startswith(_TURN_LIMIT_ERROR) and self._gemini_session_id:
            output, final_steps, final_raw = self._run_final_answer_turn()
            steps.extend(final_steps)
            raw_result.raw_responses.extend(final_raw.raw_responses)
        # The schema is only a prompt contract here. A drifting answer gets one
        # repair turn in the resumed session.
        output, repair_steps, repair_raw = repair_schema_output(
            output,
            schema=self._output_schema_json,
            name=self._output_schema_name,
            strict=self._output_schema_strict,
            resumable=bool(self._gemini_session_id),
            run_turn=self._run_gemini,
            label="gemini",
        )
        steps.extend(repair_steps)
        raw_result.raw_responses.extend(repair_raw)

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

        threshold_tokens = int(self._max_context_tokens * self._compaction_threshold_ratio)
        input_tokens = max(
            (
                int(response.get("usage", {}).get("input_tokens", 0) or 0)
                for response in raw_result.raw_responses
                if isinstance(response, dict) and isinstance(response.get("usage"), dict)
            ),
            default=0,
        )
        if (
            threshold_tokens > 0
            and input_tokens >= threshold_tokens
            and not str(output or "").strip().lower().startswith("error:")
        ):
            log_event(
                "agent_compaction_triggered",
                payload={
                    "backend": "gemini",
                    "input_tokens": input_tokens,
                    "threshold_tokens": threshold_tokens,
                    "max_context_tokens": self._max_context_tokens,
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )
            compacted = self.compact_for_resume()
            raw_result.raw_responses.extend(compacted.raw_responses)
            log_event(
                "agent_compaction_completed" if compacted.succeeded else "agent_compaction_failed",
                payload={
                    "backend": "gemini",
                    "method": compacted.method,
                    "input_tokens": input_tokens,
                    "threshold_tokens": threshold_tokens,
                    "error": compacted.error[:500],
                },
                task_session_id=current_session_id() or "",
                run_label=current_run_label() or "",
            )

        return {
            "output": output,
            "intermediate_steps": steps,
            "raw_result": raw_result,
        }

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)

    def compact_for_resume(
        self, focus_instructions: str = ""
    ) -> ResumeCompactionResult:
        del focus_instructions  # Gemini CLI /compress has no focus argument.
        result = ResumeCompactionResult(
            backend="gemini",
            method="/compress",
        )
        if not self._gemini_session_id:
            return result
        result.attempted = True
        started_at = time.monotonic()
        try:
            output, _steps, raw_result = self._run_gemini("/compress")
            result.raw_responses = list(raw_result.raw_responses or [])
            normalized = str(output or "").strip().lower()
            if normalized.startswith("error:") or "unknown command" in normalized:
                result.error = str(output or "Gemini /compress failed.")
            else:
                result.succeeded = True
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_seconds = max(0.0, time.monotonic() - started_at)
        return result

    def _compose_prompt(self, user_input: str) -> str:
        # The base system prompt is the CLI's system prompt (GEMINI_SYSTEM_MD).
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

        prompt_parts = [p for p in (user_input, policy_block, schema_block) if p.strip()]
        return "\n".join(prompt_parts)

    def _run_final_answer_turn(self) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        _LOGGER.warning(
            "Gemini CLI reached its turn limit (%s) without an answer; resuming session %s for the final answer",
            self._max_turns,
            self._gemini_session_id,
        )
        self._turn_budget_override = _FINAL_ANSWER_TURNS
        try:
            return self._run_gemini(_FINAL_ANSWER_PROMPT)
        finally:
            self._turn_budget_override = 0

    def _run_gemini(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        timeout_seconds = cli_exec_timeout_seconds(GEMINI_EXEC_TIMEOUT_ENV)
        if not self._gemini_base_url:
            return self._run_gemini_cli(prompt, timeout_seconds=timeout_seconds)
        # A gateway reports real token usage only for requests it does not
        # stream, so the CLI reaches it through a relay that un-streams them.
        # The relay also counts each answer's usage as it lands, so a run that
        # is stopped before the CLI prints its stats still reports its calls.
        with GeminiGatewayRelay(self._gemini_base_url, timeout_seconds=timeout_seconds) as relay:
            return self._run_gemini_cli(prompt, timeout_seconds=timeout_seconds, relay=relay)

    def _count_relay_usage(self, relay: GeminiGatewayRelay | None, raw_responses: list[Any]) -> int:
        """Report the usage of the answers the relay passed on since the last call.

        Returns the input tokens counted, so a run can tell a gateway that
        reports no usage at all.
        """
        if relay is None:
            return 0
        input_tokens = 0
        for answer in relay.take_usage():
            usage = {
                "input_tokens": answer["input_tokens"],
                "output_tokens": answer["output_tokens"],
                "input_tokens_details": {"cached_tokens": answer["cached_tokens"], "cache_write_tokens": 0},
            }
            raw_responses.append({"usage": usage})
            input_tokens += answer["input_tokens"]
            report_live_usage(
                self._model_name,
                prompt_tokens=answer["input_tokens"],
                completion_tokens=answer["output_tokens"],
                cached_prompt_tokens=answer["cached_tokens"],
                cache_write_tokens=0,
            )
        return input_tokens

    def _run_gemini_cli(
        self, prompt: str, *, timeout_seconds: int, relay: GeminiGatewayRelay | None = None
    ) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_gemini_home_and_config()
        command = self._build_command()
        env = self._build_env()
        if relay is not None:
            env["GOOGLE_GEMINI_BASE_URL"] = relay.base_url
            for name in ("NO_PROXY", "no_proxy"):
                env[name] = ",".join(part for part in (env.get(name, ""), "127.0.0.1") if part)
        exec_cwd = str(env.get("CHACK_EXEC_CWD", "") or os.environ.get("CHACK_EXEC_CWD", "") or "").strip() or None
        # The Chack tools work in exec_cwd; Gemini CLI itself (which has no
        # native tools here) starts in an empty workspace of its own.
        self._write_gemini_settings(self._gemini_home, tools_cwd=exec_cwd or os.getcwd())
        launch_cwd = self._gemini_workspace_dir()

        _LOGGER.info(
            "Starting Gemini CLI process: model=%s timeout_seconds=%s gemini_session_id=%s cwd=%s tools_cwd=%s",
            self._model_name,
            timeout_seconds,
            self._gemini_session_id or "",
            launch_cwd,
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
                cwd=launch_cwd,
                start_new_session=True,
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

        _send_prompt(process, prompt)
        cancel_registration = register_process(process, _terminate_process_tree)
        reader = CliLineReader(process.stdout)
        output_parts: list[str] = []
        steps: list[tuple[ToolAction, Any]] = []
        raw_lines: list[str] = []
        raw_responses: list[Any] = []
        relay_input_tokens = 0
        started_at = time.monotonic()
        time_to_first_token_seconds: float | None = None
        tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        try:
            while True:
                relay_input_tokens += self._count_relay_usage(relay, raw_responses)
                if cancellation_requested():
                    _terminate_process_tree(process)
                    self._count_relay_usage(relay, raw_responses)
                    return GEMINI_CANCELLED_OUTPUT, steps, _RawResult(raw_responses=raw_responses)
                if timeout_seconds > 0 and (time.monotonic() - started_at) > timeout_seconds:
                    _terminate_process_tree(process)
                    self._count_relay_usage(relay, raw_responses)
                    return (
                        f"ERROR: Gemini execution timed out after {timeout_seconds}s.",
                        steps,
                        _RawResult(raw_responses=raw_responses),
                    )

                raw_line = reader.next_line()
                if raw_line is None:
                    continue
                if raw_line is END_OF_OUTPUT:
                    break

                line = str(raw_line).rstrip("\n")
                if line:
                    raw_lines.append(line)
                event = self._parse_event_line(line)
                if not event:
                    continue

                event_type = str(event.get("type") or "").strip()
                if (
                    time_to_first_token_seconds is None
                    and event_type != "init"
                ):
                    time_to_first_token_seconds = max(
                        0.0,
                        time.monotonic() - started_at,
                    )
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
                    if relay is not None:
                        # The relay counted each answer as it landed; the CLI's
                        # stats are the same calls again.
                        relay_input_tokens += self._count_relay_usage(relay, raw_responses)
                        if not relay_input_tokens:
                            _LOGGER.warning(
                                "The gateway at %s reported no token usage for %s; this run's "
                                "tokens and cost are not counted",
                                self._gemini_base_url,
                                self._model_name,
                            )
                    elif isinstance(stats, dict):
                        usage = {
                            "input_tokens": int(stats.get("input_tokens", 0) or 0),
                            "output_tokens": int(stats.get("output_tokens", 0) or 0),
                            "input_tokens_details": {
                                "cached_tokens": int(stats.get("cached", 0) or 0),
                                "cache_write_tokens": 0,
                            },
                        }
                        report_live_usage(
                            self._model_name,
                            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                            completion_tokens=int(usage.get("output_tokens", 0) or 0),
                            cached_prompt_tokens=int(
                                usage.get("input_tokens_details", {}).get("cached_tokens", 0) or 0
                            ),
                            cache_write_tokens=0,
                        )
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
            unregister_process(cancel_registration)
            if process.poll() is None:
                # Returning early (a result error): leave no CLI or MCP server behind.
                _terminate_process_tree(process)
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass

        self._count_relay_usage(relay, raw_responses)
        if return_code != 0 and cancellation_requested():
            # Cancelled while the loop waited: the cancel stopped the CLI itself.
            return GEMINI_CANCELLED_OUTPUT, steps, _RawResult(raw_responses=raw_responses)
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

        return response, steps, _RawResult(
            raw_responses=raw_responses,
            time_to_first_token_seconds=time_to_first_token_seconds,
            time_to_first_token_source="gemini_first_response_event",
        )

    def _build_command(self) -> list[str]:
        # The prompt goes over stdin (see _send_prompt); Linux caps one argv
        # string at 128 KiB. An empty -p keeps the CLI headless.
        args: list[str] = [self._gemini_cli_path, "-p", "", "-o", "stream-json"]
        if self._model_name:
            args.extend(["-m", self._model_name])
        if self._gemini_session_id:
            args.extend(["-r", self._gemini_session_id])
        return args

    def _set_env_or_file(self, env: dict[str, str], key: str, value: str, *, path_key: str, prefix: str) -> None:
        raw = str(value or "")
        env.pop(key, None)
        env.pop(path_key, None)
        if not raw:
            return
        if len(raw) <= CHACK_INLINE_ENV_VALUE_MAX_CHARS:
            env[key] = raw
            return
        # Large payloads (serialized tool closures) go to files: Linux caps a
        # single environment string at 128 KiB, for the CLI and its MCP server.
        # One file per payload and run keeps _build_env idempotent.
        cache = self.__dict__.setdefault("_payload_files", {})
        path = cache.get((key, raw))
        if not path or not os.path.isfile(path):
            path = write_payload_to_file(raw, prefix=prefix, directory=str(self._gemini_home or os.getcwd()))
            cache[(key, raw)] = path
        env[path_key] = path

    def _build_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if v is not None}
        augment_subprocess_pythonpath(env)
        env["PYTHONUNBUFFERED"] = "1"
        env["GEMINI_CLI_HOME"] = str(self._gemini_home or os.getcwd())
        # Chack's prompt replaces the CLI's built-in coding-assistant prompt
        # (~7k tokens on every request).
        env["GEMINI_SYSTEM_MD"] = self._system_prompt_path()
        # Gemini CLI loads the launch directory's GEMINI.md, .gemini/settings.json
        # (which can run commands), MCP servers, hooks, skills and agents once
        # that directory is trusted, and in headless mode it refuses to start
        # in an untrusted one. It is launched in an empty workspace Chack owns
        # (see _gemini_workspace_dir), so trusting that directory loads
        # nothing from the repository under audit.
        env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
        env.pop("GEMINI_RESTRICTED_MODE", None)
        if self._gemini_api_key:
            env["GEMINI_API_KEY"] = self._gemini_api_key
        if self._gemini_base_url:
            env["GOOGLE_GEMINI_BASE_URL"] = self._gemini_base_url
            env.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
        env.update(self._gateway_env)
        if self._gemini_headers:
            env["GEMINI_CLI_CUSTOM_HEADERS"] = ", ".join(
                f"{name}: {value}" for name, value in self._gemini_headers.items()
            )

        for key, value, path_key, prefix in (
            ("CHACK_TOOLS_CONFIG_JSON", self._tools_config_json,
             CHACK_TOOLS_CONFIG_JSON_PATH_ENV, "chack_tools_config_"),
            ("CHACK_ALLOWED_TOOLS_JSON", self._allowed_tools_json,
             CHACK_ALLOWED_TOOLS_JSON_PATH_ENV, "chack_allowed_tools_"),
            (CHACK_TOOLS_OVERRIDE_B64_ENV, self._serialized_tools_override_b64,
             CHACK_TOOLS_OVERRIDE_B64_PATH_ENV, "chack_tools_override_"),
            (CHACK_TOOLS_APPEND_B64_ENV, self._serialized_tools_append_b64,
             CHACK_TOOLS_APPEND_B64_PATH_ENV, "chack_tools_append_"),
        ):
            self._set_env_or_file(env, key, value, path_key=path_key, prefix=prefix)
        if getattr(self, "_serialized_tools_override_names_json", ""):
            env[CHACK_TOOLS_OVERRIDE_NAMES_JSON_ENV] = self._serialized_tools_override_names_json
        if getattr(self, "_serialized_tools_append_names_json", ""):
            env[CHACK_TOOLS_APPEND_NAMES_JSON_ENV] = self._serialized_tools_append_names_json
        env["CHACK_MODEL_PROVIDER"] = self._model_provider
        env["CHACK_DEFAULT_MODEL"] = self._default_model
        env["CHACK_SOCIAL_NETWORK_MODEL"] = self._social_network_model
        env["CHACK_SCIENTIFIC_MODEL"] = self._scientific_model
        env["CHACK_WEBSEARCHER_MODEL"] = self._websearcher_model
        env["CHACK_BUSINESS_MODEL"] = self._business_model
        env["CHACK_PRODUCT_MODEL"] = self._product_model
        env["CHACK_TRAVEL_MODEL"] = self._travel_model
        env["CHACK_CLI_MODEL"] = self._cli_model
        env["CHACK_SUBCHACK_MODEL"] = self._subchack_model
        env["CHACK_SOCIAL_NETWORK_MAX_TURNS"] = str(self._social_network_max_turns)
        env["CHACK_SCIENTIFIC_MAX_TURNS"] = str(self._scientific_max_turns)
        env["CHACK_WEBSEARCHER_MAX_TURNS"] = str(self._websearcher_max_turns)
        env["CHACK_BUSINESS_MAX_TURNS"] = str(self._business_max_turns)
        env["CHACK_PRODUCT_MAX_TURNS"] = str(self._product_max_turns)
        env["CHACK_TRAVEL_MAX_TURNS"] = str(self._travel_max_turns)
        env["CHACK_CLI_MAX_TURNS"] = str(self._cli_max_turns)
        env["CHACK_SUBCHACK_MAX_TURNS"] = str(self._subchack_max_turns)
        env["CHACK_MIN_TOOLS_USED"] = str(self._min_tools_used)
        env["CHACK_MAX_TOOLS_USED"] = str(effective_max_tools_used(self._max_tools_used))
        env["CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST"] = (
            "1" if self._require_task_steps_manager_init_first else "0"
        )
        env["CHACK_TASK_SESSION_ID"] = str(current_session_id() or "")
        env["CHACK_RUN_LABEL"] = str(current_run_label() or "Run 1")
        # Child processes export into the same place and join this run's trace.
        env.update(child_process_env())
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
        with open(self._system_prompt_path(), "w", encoding="utf-8") as handle:
            handle.write(str(self._base_system_prompt or "").strip() + "\n")

    def _system_prompt_path(self) -> str:
        return os.path.join(str(self._gemini_home or os.getcwd()), "system.md")

    def _gemini_workspace_dir(self) -> str:
        """Empty directory Gemini CLI is launched in, inside the isolated home."""
        workspace = os.path.join(str(self._gemini_home or os.getcwd()), "workspace")
        os.makedirs(workspace, exist_ok=True)
        return workspace

    def _write_gemini_settings(self, gemini_home: str, *, tools_cwd: str | None = None) -> None:
        # GEMINI_CLI_HOME replaces the home directory, so the user settings
        # file is <home>/.gemini/settings.json.
        settings_dir = os.path.join(gemini_home, ".gemini")
        os.makedirs(settings_dir, exist_ok=True)
        settings_path = os.path.join(settings_dir, "settings.json")
        settings_payload = {
            "model": {
                "name": self._model_name,
                "maxSessionTurns": self._turn_budget_override or (self._max_turns if self._max_turns > 0 else -1),
            },
            "modelConfigs": {
                "overrides": [
                    {
                        "match": {"model": self._model_name},
                        "modelConfig": {
                            "generateContentConfig": {
                                "thinkingConfig": gemini_thinking_config(
                                    self._thinking_effort,
                                    self._model_name,
                                )
                            }
                        },
                    }
                ]
            },
            # Gemini CLI's own tools are off: they would work in its empty
            # launch directory, and headless runs drop every tool that needs
            # approval. The model gets Chack's tools (a trusted server, so they
            # run without approval) and, unless turned off, its web search.
            # Skip the listing of up to 200 workspace files the CLI otherwise
            # adds to every conversation; the tools can inspect what matters.
            "context": {"includeDirectoryTree": False},
            "tools": {
                "core": self._gemini_core_tools(),
                "shell": {
                    "inactivityTimeout": int(
                        os.environ.get("CHACK_GEMINI_SHELL_INACTIVITY_TIMEOUT", "300")
                    ),
                },
            },
            "mcpServers": {
                "chack_tools": {
                    "command": sys.executable,
                    "args": ["-m", "chack_agent.backends.chack_tools_mcp_server"],
                    "env": self._gemini_mcp_env_map(),
                    "trust": True,
                    "timeout": int(
                        os.environ.get("CHACK_GEMINI_MCP_TOOL_TIMEOUT_MS", "300000")
                    ),
                }
            },
        }
        # Headless Gemini CLI exits when no auth type is selected.
        settings_payload["security"] = {
            "auth": {"selectedType": "gemini-api-key" if self._gemini_api_key else "vertex-ai"}
        }
        # Prompt sections for features a Chack run never uses.
        settings_payload["skills"] = {"enabled": False}
        settings_payload["general"] = {"topicUpdateNarration": False}
        if tools_cwd:
            settings_payload["mcpServers"]["chack_tools"]["cwd"] = tools_cwd
        if self._playwright_mcp_enabled():
            pw_cfg = playwright_mcp_server_config()
            pw_cfg["timeout"] = int(
                os.environ.get("CHACK_GEMINI_MCP_TOOL_TIMEOUT_MS", "300000")
            )
            settings_payload["mcpServers"]["playwright"] = pw_cfg
        with open(settings_path, "w", encoding="utf-8") as handle:
            json.dump(settings_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _gemini_core_tools(self) -> list[str]:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        tools = ["mcp_chack_tools_*"]
        if self._playwright_mcp_enabled():
            tools.append("mcp_playwright_*")
        if cfg.get("native_web_search_enabled", True):
            tools.append("google_web_search")
        return tools

    def _playwright_mcp_enabled(self) -> bool:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        return bool(cfg.get("playwright_enabled")) and playwright_mcp_is_available()

    def _gemini_mcp_env_map(self) -> dict[str, str]:
        env_keys = [
            "CHACK_TOOLS_CONFIG_JSON",
            "CHACK_ALLOWED_TOOLS_JSON",
            "CHACK_TOOLS_OVERRIDE_B64",
            "CHACK_TOOLS_APPEND_B64",
            "CHACK_TOOLS_CONFIG_JSON_PATH",
            "CHACK_ALLOWED_TOOLS_JSON_PATH",
            "CHACK_TOOLS_OVERRIDE_B64_PATH",
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
            "CHACK_CALLER_PYTHONPATH_JSON",
            "DYNAMIC_POC_VIRTUAL_ENV",
            "CHACK_MODEL_PROVIDER",
            "CHACK_DEFAULT_MODEL",
            "CHACK_SOCIAL_NETWORK_MODEL",
            "CHACK_SCIENTIFIC_MODEL",
            "CHACK_WEBSEARCHER_MODEL",
            "CHACK_BUSINESS_MODEL",
            "CHACK_PRODUCT_MODEL",
            "CHACK_CLI_MODEL",
            "CHACK_SUBCHACK_MODEL",
            "CHACK_SOCIAL_NETWORK_MAX_TURNS",
            "CHACK_SCIENTIFIC_MAX_TURNS",
            "CHACK_WEBSEARCHER_MAX_TURNS",
            "CHACK_BUSINESS_MAX_TURNS",
            "CHACK_PRODUCT_MAX_TURNS",
            "CHACK_CLI_MAX_TURNS",
            "CHACK_SUBCHACK_MAX_TURNS",
            "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST",
            "CHACK_TASK_SESSION_ID",
            "CHACK_RUN_LABEL",
            "CHACK_RESEARCH_MASTER_DIR",
            "CHACK_RESEARCH_DATA_DIR",
            "CHACK_RESEARCH_SAVE_ARTIFACTS",
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
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GEMINI_CLI_HOME",
            "CHACK_EXEC_CWD",
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
        ]
        env_keys.extend(OTEL_CHILD_ENV_NAMES)
        env_keys.extend(configured_mcp_env_names(self._tools_config_json))
        env_payload: dict[str, str] = {}
        src_env = self._build_env()
        for key in env_keys:
            value = src_env.get(key)
            if value is None:
                continue
            env_payload[key] = _safe_settings_value(value)

        if self._gemini_api_key:
            env_payload.setdefault("GEMINI_API_KEY", self._gemini_api_key)

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
    gateway_route = get_gateway_route(config)
    gateway_config = clone_config_for_gateway(config, gateway_route) if gateway_route is not None else None
    if gateway_route is not None and gateway_route.gemini_base_url:
        # The gateway's model names, still run by the Gemini CLI.
        config = replace(gateway_config, model=replace(gateway_config.model, provider="gemini"))
    elif gateway_route is not None:
        from .gateway_responses_backend import build_executor as build_gateway_responses_executor

        _LOGGER.warning(
            "%s serves no Gemini API, so the Gemini CLI cannot run %s; running it on the gateway "
            "Responses backend instead (set %s=litellm or %s to use the Gemini CLI)",
            gateway_route.display_name,
            gateway_route.model_name,
            LLM_GATEWAY_ENV_TYPE,
            LLM_GATEWAY_ENV_GEMINI_BASE_URL,
        )
        return build_gateway_responses_executor(
            gateway_config,
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

    require_task_steps_manager_init_first = bool(
        getattr(config.agent, "require_task_steps_manager_init_first", True)
        and ("task_steps_manager" in allowed_tool_names)
    )

    gemini_api_key = (
        (gateway_route.api_key if gateway_route is not None else "")
        or str(config.credentials.gemini_api_key or "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_api_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise ValueError(
            "GEMINI_API_KEY is required when model.provider=gemini or set GOOGLE_APPLICATION_CREDENTIALS/GOOGLE_CLOUD credentials."
        )

    configured_gemini_path = os.environ.get("GEMINI_CLI_PATH", "").strip() or "gemini"
    gemini_cli_path = shutil.which(configured_gemini_path) or configured_gemini_path
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
        serialized_tools_override_b64=serialized_tools_override_b64,
        serialized_tools_append_b64=serialized_tools_append_b64,
        serialized_tools_override_names_json=serialized_tools_override_names_json,
        serialized_tools_append_names_json=serialized_tools_append_names_json,
        model_provider=model_provider,
        default_model=str(config.model.primary or ""),
        social_network_model=str(config.model.social_network or ""),
        scientific_model=str(config.model.scientific or ""),
        websearcher_model=str(config.model.websearcher or ""),
        business_model=str(config.model.business or ""),
        product_model=str(config.model.product or ""),
        travel_model=str(config.model.travel or ""),
        cli_model=str(config.model.cli or ""),
        subchack_model=str(config.model.subchack or ""),
        social_network_max_turns=int(config.model.social_network_max_turns or 30),
        scientific_max_turns=int(config.model.scientific_max_turns or 30),
        websearcher_max_turns=int(config.model.websearcher_max_turns or 30),
        business_max_turns=int(config.model.business_max_turns or 30),
        product_max_turns=int(config.model.product_max_turns or 30),
        travel_max_turns=int(config.model.travel_max_turns or 40),
        cli_max_turns=int(config.model.cli_max_turns or 30),
        subchack_max_turns=int(config.model.subchack_max_turns or 30),
        min_tools_used=max(0, int(config.tools.min_tools_used or 0)),
        max_tools_used=max(0, int(config.tools.max_tools_used or 0)),
        require_task_steps_manager_init_first=require_task_steps_manager_init_first,
        output_schema_json=(
            json.dumps(config.agent.output_schema_json, ensure_ascii=False)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        output_schema_name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
        output_schema_strict=bool(getattr(config.agent, "output_schema_strict", False)),
        thinking_effort=normalize_thinking_effort(config.agent.thinking_effort),
        max_context_tokens=int(getattr(config.agent, "max_context_tokens", 0) or 0),
        compaction_threshold_ratio=float(
            getattr(config.agent, "compaction_threshold_ratio", 0.75) or 0.75
        ),
        gemini_base_url=gateway_route.gemini_base_url if gateway_route is not None else "",
        gemini_headers=gateway_route.headers if gateway_route is not None else None,
        gateway_env=gateway_subprocess_env(config, gateway_route) if gateway_route is not None else None,
    )
