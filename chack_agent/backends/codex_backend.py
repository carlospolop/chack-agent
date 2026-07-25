from __future__ import annotations

import json
import logging
import os
import re
import selectors
import shutil
import signal
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
from chack_tools.cancellation import cancellation_requested, register_process, unregister_process
from chack_tools.tool_usage_state import effective_max_tools_used

from ..config import ChackConfig
from ..live_cost_state import report_live_usage
from ..openrouter_routing import get_openrouter_route
from ..resume_compaction import ResumeCompactionResult
from ..thinking_effort import codex_thinking_effort, normalize_thinking_effort
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .tool_payloads import (
    CHACK_ALLOWED_TOOLS_JSON_PATH_ENV,
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_APPEND_B64_PATH_ENV,
    CHACK_TOOLS_CONFIG_JSON_PATH_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    CHACK_TOOLS_OVERRIDE_B64_PATH_ENV,
    CHACK_INLINE_ENV_VALUE_MAX_CHARS,
    augment_subprocess_pythonpath,
    serialize_tools_payload,
    write_payload_to_file,
)


_LOGGER = logging.getLogger("chack.codex_backend")

# Per-run codex process timeout, selected by the agent's sub_action so different roles
# (verifiers vs research administrators vs sub-researchers) get different wall-clock caps.
# CHACK_CODEX_EXEC_TIMEOUT_BY_SUBACTION is a JSON map {sub_action: seconds, "default": n};
# falls back to the global CHACK_CODEX_EXEC_TIMEOUT_SECONDS, then 900.
def _resolve_codex_exec_timeout(
    sub_action: str,
    runtime_env: Optional[dict[str, str]] = None,
) -> int:
    runtime_env = runtime_env or {}
    default = int(
        runtime_env.get(
            "CHACK_CODEX_EXEC_TIMEOUT_SECONDS",
            os.environ.get("CHACK_CODEX_EXEC_TIMEOUT_SECONDS", "900"),
        )
        or "900"
    )
    raw = str(
        runtime_env.get(
            "CHACK_CODEX_EXEC_TIMEOUT_BY_SUBACTION",
            os.environ.get("CHACK_CODEX_EXEC_TIMEOUT_BY_SUBACTION", ""),
        )
        or ""
    ).strip()
    if raw:
        try:
            m = json.loads(raw)
            if isinstance(m, dict):
                v = m.get(str(sub_action or "").strip())
                if v is None:
                    v = m.get("default")
                if v is not None:
                    return max(1, int(v))
        except Exception:
            pass
    return default


# Optional host-process callback invoked whenever a codex process times out, so the
# application (e.g. the factchecker) can alert Discord. Called with a dict describing the
# timed-out agent. Runs in the same process/thread that monitors the codex subprocess.
_CODEX_TIMEOUT_HOOK = None


def set_codex_timeout_hook(fn) -> None:
    """Register a callback fired on every codex process timeout. fn receives a dict:
    {sub_action, model, provider, session_id, timeout_seconds}. Errors are swallowed."""
    global _CODEX_TIMEOUT_HOOK
    _CODEX_TIMEOUT_HOOK = fn


def _notify_codex_timeout(info: dict) -> None:
    hook = _CODEX_TIMEOUT_HOOK
    if hook is None:
        return
    try:
        hook(info)
    except Exception:
        pass


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


def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _preview_text(value: Any, *, max_chars: int = 2000) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _readline_when_ready(stream: Any, wait_seconds: float) -> Optional[str]:
    """Read one subprocess line only when its pipe is ready.

    ``TextIO.readline()`` blocks indefinitely when a child stays alive without
    producing output. Waiting on the pipe first lets the caller continue
    enforcing execution deadlines and cancellation while the provider is
    silent.
    """
    selector = selectors.DefaultSelector()
    try:
        selector.register(stream, selectors.EVENT_READ)
        if not selector.select(timeout=max(0.0, float(wait_seconds))):
            return None
        return stream.readline()
    finally:
        selector.close()


def _resolve_codex_exec_cwd(runtime_env: Optional[dict[str, str]] = None) -> str:
    runtime_env = runtime_env or {}
    candidate = str(
        runtime_env.get("CHACK_EXEC_CWD", os.environ.get("CHACK_EXEC_CWD", "")) or ""
    ).strip()
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
    time_to_first_token_seconds: float | None = None
    time_to_first_token_source: str = "unavailable"


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
    _use_existing_codex_auth_file: bool
    _existing_codex_auth_file: str
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
    _self_critique_enabled: bool
    _self_critique_rounds: int
    _min_tools_used: int
    _max_tools_used: int
    _require_task_steps_manager_init_first: bool
    _output_schema_json: str
    _output_schema_strict: bool = True
    _max_context_tokens: int = 0
    _compaction_threshold_ratio: float = 0.50
    _uses_openrouter_route: bool = False
    _openrouter_base_url: str = ""
    _openrouter_http_referer: str = ""
    _openrouter_app_name: str = ""
    _output_schema_path: Optional[str] = None
    _thread_id: Optional[str] = None
    _codex_home: Optional[str] = None
    _disable_native_shell: bool = False
    _disable_native_web_search: bool = False
    _researcher_administrator_model: str = ""
    _sub_action: str = ""
    _researcher_administrator_max_turns: int = 100
    _thinking_effort: str = "high"
    _prompt_only_next_invocation: bool = False
    _travel_model: str = ""
    _travel_max_turns: int = 50
    _runtime_env_json: str = "{}"

    def _runtime_env(self) -> dict[str, str]:
        try:
            value = json.loads(self._runtime_env_json or "{}")
        except (TypeError, ValueError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items() if item is not None}

    def _runtime_env_value(self, name: str, default: str = "") -> str:
        runtime_env = self._runtime_env()
        return str(runtime_env.get(name, os.environ.get(name, default)) or default)

    def suppress_system_prompt_for_next_invocation(self) -> None:
        self._prompt_only_next_invocation = True

    def _disabled_native_tool_args(self) -> list[str]:
        args: list[str] = []
        if self._disable_native_shell:
            args.extend(
                [
                    "--disable",
                    "shell_tool",
                    "--disable",
                    "unified_exec",
                ]
            )
        if self._disable_native_web_search:
            args.extend(
                [
                    "-c",
                    'web_search="disabled"',
                ]
            )
        return args

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

    def compact_for_resume(
        self, focus_instructions: str = ""
    ) -> ResumeCompactionResult:
        result = ResumeCompactionResult(
            backend="codex",
            method="thread/compact/start",
        )
        if not self._thread_id:
            return result
        result.attempted = True
        started_at = time.monotonic()
        try:
            result.raw_responses = self._compact_codex_thread(
                focus_instructions
            )
            result.succeeded = True
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_seconds = max(0.0, time.monotonic() - started_at)
        return result

    def _compact_codex_thread(
        self, focus_instructions: str
    ) -> list[Any]:
        self._ensure_codex_home_and_config()
        command = [self._codex_path, "app-server", "--stdio"]
        env = self._build_env()
        exec_cwd = _resolve_codex_exec_cwd(self._runtime_env())
        timeout_seconds = max(
            30,
            int(
                self._runtime_env_value(
                    "CHACK_CODEX_COMPACTION_TIMEOUT_SECONDS",
                    "300",
                )
                or "300"
            ),
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=exec_cwd or None,
            start_new_session=True,
        )
        cancel_registration = register_process(process, _terminate_process_tree)
        deadline = time.monotonic() + timeout_seconds
        selector = selectors.DefaultSelector()
        compaction_usage: dict[str, Any] = {}
        if process.stdout is None or process.stdin is None:
            _terminate_process_tree(process)
            unregister_process(cancel_registration)
            raise RuntimeError("Codex app-server did not expose stdio pipes")
        selector.register(process.stdout, selectors.EVENT_READ)

        def _send(message: dict[str, Any]) -> None:
            if process.stdin is None:
                raise RuntimeError("Codex app-server stdin closed unexpectedly")
            process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            process.stdin.flush()

        def _wait_for(
            predicate,
            *,
            description: str,
        ) -> dict[str, Any]:
            while time.monotonic() < deadline:
                if cancellation_requested():
                    raise RuntimeError("Codex compaction cancelled")
                if process.poll() is not None:
                    stderr = (
                        process.stderr.read()
                        if process.stderr is not None
                        else ""
                    )
                    raise RuntimeError(
                        f"Codex app-server exited while waiting for {description}: "
                        f"{str(stderr or '').strip()[-1000:]}"
                    )
                remaining = max(0.0, deadline - time.monotonic())
                events = selector.select(timeout=min(1.0, remaining))
                if not events:
                    continue
                line = process.stdout.readline()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                if (
                    str(message.get("method") or "")
                    == "thread/tokenUsage/updated"
                ):
                    params = (
                        message.get("params")
                        if isinstance(message.get("params"), dict)
                        else {}
                    )
                    token_usage = (
                        params.get("tokenUsage")
                        if isinstance(params.get("tokenUsage"), dict)
                        else {}
                    )
                    last = (
                        token_usage.get("last")
                        if isinstance(token_usage.get("last"), dict)
                        else {}
                    )
                    if last:
                        compaction_usage.clear()
                        compaction_usage.update(
                            {
                                "input_tokens": int(
                                    last.get("inputTokens", 0) or 0
                                ),
                                "output_tokens": int(
                                    last.get("outputTokens", 0) or 0
                                ),
                                "input_tokens_details": {
                                    "cached_tokens": int(
                                        last.get(
                                            "cachedInputTokens",
                                            0,
                                        )
                                        or 0
                                    ),
                                    "cache_write_tokens": 0,
                                },
                            }
                        )
                if "error" in message and "id" in message:
                    error = message.get("error")
                    if isinstance(error, dict):
                        error = error.get("message") or error
                    raise RuntimeError(
                        f"Codex app-server request failed: {error}"
                    )
                if predicate(message):
                    return message
            raise TimeoutError(
                f"Codex app-server timed out waiting for {description}"
            )

        try:
            _send(
                {
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {
                            "name": "chack-agent",
                            "title": "Chack pre-resume compactor",
                            "version": "1",
                        },
                        "capabilities": {"experimentalApi": True},
                    },
                }
            )
            _wait_for(
                lambda message: message.get("id") == 1
                and "result" in message,
                description="initialize response",
            )
            _send({"method": "initialized"})
            compact_prompt = (
                "Create a detailed continuation summary of this conversation. "
                "The summary will replace the prior turns and must let the next "
                "agent continue without rereading them."
            )
            if str(focus_instructions or "").strip():
                compact_prompt += "\n\n" + focus_instructions.strip()
            _send(
                {
                    "id": 2,
                    "method": "thread/resume",
                    "params": {
                        "threadId": self._thread_id,
                        "model": self._model_name,
                        "config": {"compact_prompt": compact_prompt},
                    },
                }
            )
            _wait_for(
                lambda message: message.get("id") == 2
                and "result" in message,
                description="thread resume response",
            )
            _send(
                {
                    "id": 3,
                    "method": "thread/compact/start",
                    "params": {"threadId": self._thread_id},
                }
            )
            _wait_for(
                lambda message: message.get("id") == 3
                and "result" in message,
                description="compaction start response",
            )

            def _is_compaction_complete(message: dict[str, Any]) -> bool:
                method = str(message.get("method") or "")
                params = (
                    message.get("params")
                    if isinstance(message.get("params"), dict)
                    else {}
                )
                item = (
                    params.get("item")
                    if isinstance(params.get("item"), dict)
                    else {}
                )
                item_type = str(item.get("type") or "")
                return (
                    method == "item/completed"
                    and item_type == "contextCompaction"
                ) or method == "thread/compacted"

            _wait_for(
                _is_compaction_complete,
                description="compaction completion",
            )
            if compaction_usage:
                report_live_usage(
                    self._model_name,
                    prompt_tokens=int(
                        compaction_usage.get("input_tokens", 0) or 0
                    ),
                    completion_tokens=int(
                        compaction_usage.get("output_tokens", 0) or 0
                    ),
                    cached_prompt_tokens=int(
                        compaction_usage.get(
                            "input_tokens_details",
                            {},
                        ).get("cached_tokens", 0)
                        or 0
                    ),
                    cache_write_tokens=0,
                )
                return [{"usage": dict(compaction_usage)}]
            return []
        finally:
            selector.close()
            try:
                process.stdin.close()
            except Exception:
                pass
            _terminate_process_tree(process)
            unregister_process(cancel_registration)

    def _compose_prompt(self, user_input: str) -> str:
        if self._prompt_only_next_invocation:
            self._prompt_only_next_invocation = False
            return str(user_input or "")
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
        timeout_seconds = _resolve_codex_exec_timeout(self._sub_action, self._runtime_env())
        exec_cwd = _resolve_codex_exec_cwd(self._runtime_env())
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
                start_new_session=True,
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

        cancel_registration = register_process(process, _terminate_process_tree)
        steps: list[tuple[ToolAction, Any]] = []
        output_parts: list[str] = []
        usage_payload: dict[str, Any] | None = None
        combined_output_lines: list[str] = []
        error_messages: list[str] = []
        started_at = time.monotonic()
        time_to_first_token_seconds: float | None = None

        if process.stdin is not None:
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except Exception:
                pass

        while True:
            if cancellation_requested():
                _terminate_process_tree(process)
                unregister_process(cancel_registration)
                self._log_codex_failure(
                    "codex_cli_cancelled",
                    command=command,
                    cwd=exec_cwd,
                    details="\n".join(combined_output_lines).strip()
                    or "Cancelled before Codex produced captured output.",
                )
                return (
                    "ERROR: Codex execution cancelled.",
                    steps,
                    _RawResult(raw_responses=[]),
                )
            if (time.monotonic() - started_at) > timeout_seconds:
                _terminate_process_tree(process)
                unregister_process(cancel_registration)
                self._log_codex_failure(
                    "codex_cli_timeout",
                    command=command,
                    cwd=exec_cwd,
                    details="\n".join(combined_output_lines).strip()
                    or f"Timed out after {timeout_seconds}s with no captured output.",
                )
                _notify_codex_timeout(
                    {
                        "sub_action": str(self._sub_action or ""),
                        "model": str(self._model_name or ""),
                        "provider": str(self._model_provider or ""),
                        "session_id": str(current_session_id() or ""),
                        "timeout_seconds": int(timeout_seconds),
                    }
                )
                return (
                    f"ERROR: Codex execution timed out after {timeout_seconds}s.",
                    steps,
                    _RawResult(raw_responses=[]),
                )
            if process.stdout is None:
                break
            remaining_seconds = max(
                0.0,
                timeout_seconds - (time.monotonic() - started_at),
            )
            line = _readline_when_ready(
                process.stdout,
                min(1.0, remaining_seconds),
            )
            if line is None:
                continue
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
            if (
                time_to_first_token_seconds is None
                and event_type not in {"thread.started", "turn.started", "turn.completed", "error"}
            ):
                time_to_first_token_seconds = max(
                    0.0,
                    time.monotonic() - started_at,
                )
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
        unregister_process(cancel_registration)
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
        elif (output or steps) and usage_payload is None:
            # Codex CLI with copilot tokens doesn't emit turn.completed events,
            # so we never get real token counts.  Estimate from the prompt and
            # output character lengths (≈4 chars/token) so callers at least get a
            # non-zero cost signal.
            est_input = max(len(prompt) // 4, 1)
            est_output = max(len(output) // 4, 1) if output else 0
            estimated = {
                "input_tokens": est_input,
                "output_tokens": est_output,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            }
            raw_responses.append({"usage": estimated})
            report_live_usage(
                self._model_name,
                prompt_tokens=est_input,
                completion_tokens=est_output,
                cached_prompt_tokens=0,
                cache_write_tokens=0,
            )
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
        result = (
            output,
            steps,
            _RawResult(
                raw_responses=raw_responses,
                time_to_first_token_seconds=time_to_first_token_seconds,
                time_to_first_token_source="codex_first_response_event",
            ),
        )
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
            "session limit",
            "usage limit",
            "quota exceeded",
            "credit balance",
        )
        return any(indicator in normalized for indicator in indicators)

    def _build_command(self) -> list[str]:
        exec_cwd = _resolve_codex_exec_cwd()
        effort_args = [
            "--config",
            f'model_reasoning_effort="{codex_thinking_effort(self._thinking_effort)}"',
        ]
        if self._thread_id:
            output_schema_args: list[str] = []
            if self._output_schema_path:
                output_schema_args = ["--output-schema", self._output_schema_path]
            args = [
                self._codex_path,
                "exec",
                "resume",
                "--json",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
            ]
            args.extend(self._disabled_native_tool_args())
            args.extend(effort_args)
            if output_schema_args:
                args.extend(output_schema_args)
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
        args.extend(self._disabled_native_tool_args())
        args.extend(effort_args)
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
        # Config env belongs to this executor. Overlay it after the process env so
        # parallel Chack instances cannot overwrite one another's runtime settings.
        env.update(self._runtime_env())
        augment_subprocess_pythonpath(env)
        if self._uses_openrouter_route:
            env["OPENROUTER_API_KEY"] = self._openai_api_key
            env["OPENROUTER_BASE_URL"] = self._openrouter_base_url
            if self._openrouter_http_referer:
                env["OPENROUTER_HTTP_REFERER"] = self._openrouter_http_referer
            if self._openrouter_app_name:
                env["OPENROUTER_APP_NAME"] = self._openrouter_app_name
        elif self._use_codex_access_token or self._use_existing_codex_auth_file:
            env.pop("OPENAI_API_KEY", None)
            env.pop("CODEX_API_KEY", None)
            env.pop("CODEX_ACCESS_TOKEN", None)
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
        # Shared-MCP routing: when every agent points at one HTTP MCP server
        # (CHACK_CODEX_MCP_URL), give THIS subprocess a per-run bearer token (its
        # session id) under the configured env var so the shared server can route this
        # agent's tool calls to the right identity. Set per-subprocess (not inherited)
        # so parallel same-process agents don't collide on one os.environ value.
        if self._runtime_env_value("CHACK_CODEX_MCP_URL").strip():
            bearer_env_name = (
                self._runtime_env_value("CHACK_CODEX_MCP_BEARER_TOKEN_ENV").strip()
                or "CHACK_CODEX_MCP_BEARER_TOKEN"
            )
            bearer_token = str(current_session_id() or self._thread_id or "").strip()
            if bearer_token:
                env[bearer_env_name] = bearer_token
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
        env["CHACK_SELF_CRITIQUE_ENABLED"] = "1" if self._self_critique_enabled else "0"
        env["CHACK_SELF_CRITIQUE_ROUNDS"] = str(self._self_critique_rounds)
        env["CHACK_MIN_TOOLS_USED"] = str(self._min_tools_used)
        env["CHACK_MAX_TOOLS_USED"] = str(effective_max_tools_used(self._max_tools_used))
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
        payload_dir = self._codex_home or self._runtime_env_value("CHACK_CODEX_HOME_BASE").strip() or ""
        path = write_payload_to_file(raw, prefix=prefix, directory=payload_dir)
        env[path_env_key] = path

    def _ensure_codex_home_and_config(self) -> None:
        if self._codex_home:
            return
        safe_session = re.sub(r"[^A-Za-z0-9._-]", "_", str(current_session_id() or "default"))
        home_base = self._runtime_env_value(
            "CHACK_CODEX_HOME_BASE", os.path.expanduser("~/.codex/chack")
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
            "CHACK_CHATGPT_ASYNC_API_URL",
            "CHACK_CHATGPT_ASYNC_API_SECRET",
            "PYTHONPATH",
            # Preserve any caller-provided execution sandbox in the MCP server.
            # Dynamic-AIgent uses these variables to route PoC package installs
            # into a disposable virtualenv instead of this scanner runtime.
            "PATH",
            "VIRTUAL_ENV",
            "PIP_REQUIRE_VIRTUALENV",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PYTHONNOUSERSITE",
            "DYNAMIC_POC_VIRTUAL_ENV",
            "CHACK_MODEL_PROVIDER",
            "CHACK_DEFAULT_MODEL",
            "CHACK_SOCIAL_NETWORK_MODEL",
            "CHACK_SCIENTIFIC_MODEL",
            "CHACK_WEBSEARCHER_MODEL",
            "CHACK_BUSINESS_MODEL",
            "CHACK_PRODUCT_MODEL",
            "CHACK_TRAVEL_MODEL",
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
            "CHACK_TRAVEL_MAX_TURNS",
            "CHACK_LEGAL_MAX_TURNS",
            "CHACK_DATA_STATISTICS_MAX_TURNS",
            "CHACK_NEWS_MEDIA_MAX_TURNS",
            "CHACK_KNOWLEDGE_GRAPH_MAX_TURNS",
            "CHACK_RELIGIOUS_MAX_TURNS",
            "CHACK_CLI_MAX_TURNS",
            "CHACK_SUBCHACK_MAX_TURNS",
            "CHACK_RESEARCHER_ADMINISTRATOR_MAX_TURNS",
            "CHACK_CODEX_EXEC_TIMEOUT_BY_SUBACTION",
            "CHACK_CODEX_EXEC_TIMEOUT_SECONDS",
            "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST",
            "CHACK_TASK_SESSION_ID",
            "CHACK_RUN_LABEL",
            "CHACK_DISABLE_STDOUT_EVENTS",
            "CHACK_RESEARCH_MASTER_DIR",
            "CHACK_RESEARCH_DATA_DIR",
            "CHACK_RESEARCH_SAVE_ARTIFACTS",
            "AISEC_LOCAL_VULN_STORE_PATH",
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "CODEX_ACCESS_TOKEN",
            # The parent Codex executor may authenticate with a private
            # per-run auth.json instead of exporting a bearer token. Pass the
            # directory to the local MCP subprocess so researcher tools can
            # copy that authenticated session into their own isolated Codex
            # homes. Without this, nested web/travel researchers fail even
            # though the administrator itself is authenticated.
            "CODEX_HOME",
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
            "GEMINI_API_KEY",
            "BRAVE_API_KEY",
            "SERPAPI_API_KEY",
            "FORUMSCOUT_API_KEY",
            "FORUMSCOUT_BASE_URL",
            "BOOKING_API_TOKEN",
            "BOOKING_AFFILIATE_ID",
            "AMADEUS_CLIENT_ID",
            "AMADEUS_CLIENT_SECRET",
            "OPENTRIPMAP_API_KEY",
            "TICKETMASTER_API_KEY",
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

        # Keep the configured model capacity available, but trigger Codex's
        # native summarizing compactor at the configured fraction of it. This is
        # a compaction threshold, NOT a hard context-window cap.
        if self._max_context_tokens > 0:
            ratio = min(
                1.0,
                max(0.05, float(self._compaction_threshold_ratio or 0.50)),
            )
            compact_at = max(1, int(self._max_context_tokens * ratio))
            config_lines.append(
                f"model_auto_compact_token_limit = {compact_at}"
            )

        # System-level instructions to prevent the model from calling
        # non-existent built-in tools like `report_intent` instead of the
        # real MCP tools.
        instructions_text = (
            "CRITICAL: You do NOT have a tool called `report_intent`. "
            "It does not exist. Never attempt to call it. "
            "To report or save a vulnerability finding you MUST call the MCP tool "
            "`chack_tools-save_discovered_vulnerability`. "
            "Any call to `report_intent` will silently discard your finding. "
            "Do not call MCP resource browser helpers such as `list_mcp_resources`, "
            "`list_mcp_resource_templates`, or `read_mcp_resource`; use only the "
            "explicit task tools listed in the prompt."
        )
        config_lines.append(f"instructions = {_toml_string(instructions_text)}")
        chack_mcp_startup_timeout = int(
            self._runtime_env_value("CHACK_CODEX_MCP_STARTUP_TIMEOUT_SECONDS", "120")
            or "120"
        )
        playwright_mcp_startup_timeout = int(
            self._runtime_env_value(
                "CHACK_CODEX_PLAYWRIGHT_MCP_STARTUP_TIMEOUT_SECONDS",
                str(chack_mcp_startup_timeout),
            )
            or str(chack_mcp_startup_timeout)
        )
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
        chack_mcp_tool_timeout = int(
            self._runtime_env_value("CHACK_MCP_TOOL_TIMEOUT_SECONDS", "3600") or "3600"
        )
        chack_shared_mcp_url = self._runtime_env_value("CHACK_CODEX_MCP_URL").strip()
        if chack_shared_mcp_url:
            # Point every codex agent at ONE shared streamable-HTTP MCP server (e.g. a
            # host-process server that holds a shared queue / board), instead of each
            # agent spawning its own stdio server. Custom tools then live on that server.
            shared_mcp_lines = [
                "",
                '[mcp_servers.chack_tools]',
                f"url = {_toml_string(chack_shared_mcp_url)}",
            ]
            bearer_env = (
                self._runtime_env_value("CHACK_CODEX_MCP_BEARER_TOKEN_ENV").strip()
                or "CHACK_CODEX_MCP_BEARER_TOKEN"
            )
            shared_mcp_lines.append(f"bearer_token_env_var = {_toml_string(bearer_env)}")
            shared_mcp_lines.extend(
                [
                    "required = true",
                    f"startup_timeout_sec = {chack_mcp_startup_timeout}",
                    f"tool_timeout_sec = {chack_mcp_tool_timeout}",
                ]
            )
            config_lines.extend(shared_mcp_lines)
        else:
            config_lines.extend(
                [
                    "",
                    '[mcp_servers.chack_tools]',
                    f"command = {_toml_string(python_cmd)}",
                    f"args = {args_toml}",
                    f"env_vars = {env_vars_toml}",
                    "required = true",
                    f"startup_timeout_sec = {chack_mcp_startup_timeout}",
                    f"tool_timeout_sec = {chack_mcp_tool_timeout}",
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
                    f"startup_timeout_sec = {playwright_mcp_startup_timeout}",
                    "tool_timeout_sec = 180",
                ]
            )
        config_body = "\n".join(config_lines)
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(config_body + "\n")

    def _write_codex_auth(self, codex_home: str) -> None:
        if self._use_existing_codex_auth_file:
            if not self._existing_codex_auth_file:
                return
            auth_path = os.path.join(codex_home, "auth.json")
            shutil.copyfile(self._existing_codex_auth_file, auth_path)
            try:
                os.chmod(auth_path, 0o600)
            except OSError:
                pass
            return
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
        # Codex CLI structured output accepts only strict schemas: every
        # property must be required. A non-strict Chack schema intentionally
        # models patch objects whose omitted fields remain unchanged, so
        # passing it through --output-schema causes an API-level 400 before the
        # model can run. Let Chack's normal JSON extraction/schema validation
        # handle these responses instead.
        if not self._output_schema_strict:
            return
        raw = str(self._output_schema_json or "").strip()
        if not raw:
            return
        try:
            schema_obj = json.loads(raw)
        except Exception:
            return
        if not isinstance(schema_obj, dict):
            return
        schema_obj = self._normalize_codex_output_schema(
            schema_obj,
            force_all_required=self._output_schema_strict,
        )
        path = os.path.join(codex_home, "output_schema.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(schema_obj, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            self._output_schema_path = path
        except Exception:
            self._output_schema_path = None

    @classmethod
    def _normalize_codex_output_schema(
        cls,
        schema: Any,
        *,
        force_all_required: bool = True,
    ) -> Any:
        if isinstance(schema, list):
            return [
                cls._normalize_codex_output_schema(
                    item,
                    force_all_required=force_all_required,
                )
                for item in schema
            ]
        if not isinstance(schema, dict):
            return schema

        normalized = {
            key: cls._normalize_codex_output_schema(
                value,
                force_all_required=force_all_required,
            )
            for key, value in schema.items()
        }
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["properties"] = {
                str(key): cls._normalize_codex_output_schema(
                    value,
                    force_all_required=force_all_required,
                )
                for key, value in properties.items()
            }
            if force_all_required:
                normalized["required"] = list(normalized["properties"].keys())
            elif "required" in normalized:
                declared = normalized.get("required")
                normalized["required"] = (
                    [
                        str(key)
                        for key in declared
                        if str(key) in normalized["properties"]
                    ]
                    if isinstance(declared, list)
                    else []
                )
            normalized.setdefault("additionalProperties", False)
        elif "required" in normalized:
            normalized.pop("required", None)
        for union_key in ("anyOf", "allOf", "oneOf"):
            if isinstance(normalized.get(union_key), list):
                normalized[union_key] = [
                    cls._normalize_codex_output_schema(
                        item,
                        force_all_required=force_all_required,
                    )
                    for item in normalized[union_key]
                ]
        if isinstance(normalized.get("items"), dict):
            normalized["items"] = cls._normalize_codex_output_schema(
                normalized["items"],
                force_all_required=force_all_required,
            )
        return normalized

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

    def _existing_codex_auth_file() -> str:
        candidates = []
        configured_home = str(os.environ.get("CODEX_HOME", "") or "").strip()
        if configured_home:
            candidates.append(os.path.join(configured_home, "auth.json"))
        candidates.append(os.path.join(os.path.expanduser("~"), ".codex", "auth.json"))
        for candidate in candidates:
            try:
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    return candidate
            except OSError:
                continue
        return ""

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

    serialized_tools_override_b64 = serialize_tools_payload(tools_override)
    serialized_tools_append_b64 = serialize_tools_payload(tools_append)
    model_provider = str(config.model.provider or "").strip()
    if not model_provider:
        raise ValueError("model.provider must be defined in config")

    configured_tools: list[Any] | None = None

    def _configured_base_tools() -> list[Any]:
        nonlocal configured_tools
        if configured_tools is None:
            base_toolset = AgentsToolset(
                config.tools,
                model_provider=model_provider,
                default_model=config.model.primary,
                social_network_model=config.model.social_network,
                scientific_model=config.model.scientific,
                websearcher_model=config.model.websearcher,
                business_model=config.model.business,
                product_model=config.model.product,
                travel_model=config.model.travel,
                legal_model=config.model.legal,
                data_statistics_model=config.model.data_statistics,
                news_media_model=config.model.news_media,
                knowledge_graph_model=config.model.knowledge_graph,
                religious_model=config.model.religious,
                cli_model=config.model.cli,
                subchack_model=config.model.subchack,
                researcher_administrator_model=config.model.researcher_administrator,
                social_network_max_turns=config.model.social_network_max_turns,
                scientific_max_turns=config.model.scientific_max_turns,
                websearcher_max_turns=config.model.websearcher_max_turns,
                business_max_turns=config.model.business_max_turns,
                product_max_turns=config.model.product_max_turns,
                travel_max_turns=config.model.travel_max_turns,
                legal_max_turns=config.model.legal_max_turns,
                data_statistics_max_turns=config.model.data_statistics_max_turns,
                news_media_max_turns=config.model.news_media_max_turns,
                knowledge_graph_max_turns=config.model.knowledge_graph_max_turns,
                religious_max_turns=config.model.religious_max_turns,
                cli_max_turns=config.model.cli_max_turns,
                subchack_max_turns=config.model.subchack_max_turns,
                researcher_administrator_max_turns=config.model.researcher_administrator_max_turns,
                self_critique_enabled=bool(getattr(config.agent, "self_critique_enabled", False)),
                self_critique_rounds=int(getattr(config.agent, "self_critique_rounds", 0) or 0),
            )
            configured_tools = list(base_toolset.tools)
        return list(configured_tools)

    if tools_override is not None:
        allowed_tool_names = _extract_tool_names(list(tools_override))
    elif tools_append:
        allowed_tool_names = _extract_tool_names(_configured_base_tools() + list(tools_append))
    else:
        allowed_tool_names = _extract_tool_names(_configured_base_tools())

    has_task_steps_manager_tool = (
        "task_steps_manager" in allowed_tool_names
        if allowed_tool_names is not None
        else bool(getattr(config.tools, "task_steps_manager_enabled", True))
    )
    require_task_steps_manager_init_first = bool(
        getattr(config.agent, "require_task_steps_manager_init_first", True)
        and has_task_steps_manager_tool
    )
    disable_native_shell = False
    disable_native_web_search = False
    if allowed_tool_names is not None:
        allowed_set = set(allowed_tool_names)
        disable_native_shell = True
        disable_native_web_search = "search_google_web" not in allowed_set
    if str(os.environ.get("CHACK_DISABLE_CODEX_NATIVE_WEB", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
        disable_native_web_search = True

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
    existing_codex_auth_file = "" if uses_openrouter_route else _existing_codex_auth_file()
    codex_api_key = route.api_key if route is not None else (codex_access_token or fallback_openai_api_key)
    if not codex_api_key and not existing_codex_auth_file:
        raise ValueError(
            "OPENROUTER_API_KEY is required for OpenRouter-routed Codex models"
            if uses_openrouter_route
            else "CODEX_ACCESS_TOKEN, OPENAI_API_KEY, or Codex auth.json is required when model.provider=codex"
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
        _use_existing_codex_auth_file=bool(existing_codex_auth_file) and not codex_access_token and not uses_openrouter_route,
        _existing_codex_auth_file=existing_codex_auth_file,
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
        _sub_action=str(getattr(config.agent, "sub_action", "") or ""),
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
        _thinking_effort=normalize_thinking_effort(config.agent.thinking_effort),
        _self_critique_enabled=bool(getattr(config.agent, "self_critique_enabled", False)),
        _self_critique_rounds=int(getattr(config.agent, "self_critique_rounds", 0) or 0),
        _min_tools_used=max(0, int(config.tools.min_tools_used or 0)),
        _max_tools_used=max(0, int(config.tools.max_tools_used or 0)),
        _require_task_steps_manager_init_first=require_task_steps_manager_init_first,
        _output_schema_json=(
            json.dumps(getattr(config.agent, "output_schema_json", None), ensure_ascii=False, indent=2)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        _output_schema_strict=bool(
            getattr(config.agent, "output_schema_strict", True)
        ),
        _max_context_tokens=int(getattr(config.model, "max_context_tokens", 0) or 0),
        _compaction_threshold_ratio=float(
            getattr(config.agent, "compaction_threshold_ratio", 0.50) or 0.50
        ),
        _uses_openrouter_route=uses_openrouter_route,
        _openrouter_base_url=str(route.base_url if route is not None else ""),
        _openrouter_http_referer=str((route.headers.get("HTTP-Referer", "") if route else "")),
        _openrouter_app_name=str((route.headers.get("X-Title", "") if route else "")),
        _disable_native_shell=disable_native_shell,
        _disable_native_web_search=disable_native_web_search,
        _runtime_env_json=json.dumps(
            {
                "CHACK_EXEC_TIMEOUT": str(config.tools.exec_timeout_seconds),
                "CHACK_EXEC_MAX_OUTPUT": str(config.tools.exec_max_output_chars),
                **(
                    {"CHACK_EXEC_CWD": str(config.tools.exec_cwd)}
                    if str(getattr(config.tools, "exec_cwd", "") or "").strip()
                    else {}
                ),
                **{
                    str(key): str(value)
                    for key, value in (config.env or {}).items()
                    if value is not None
                },
            },
            ensure_ascii=False,
        ),
    )
