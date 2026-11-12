"""Kimi Code and Qwen Code headless CLI transports.

Both CLIs run in an empty per-session workspace. Their native host-file and
shell tools are removed from the model's tool registry; Chack tools arrive over
the same MCP server used by the other CLI backends.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from chack_tools.agents_toolset import AgentsToolset
from chack_tools.cancellation import cancellation_requested, register_process, unregister_process
from chack_tools.task_steps_manager_state import current_run_label, current_session_id
from chack_tools.telemetry import child_process_env, log_event
from chack_tools.tool_usage_state import effective_max_tools_used

from ..config import ChackConfig
from ..live_cost_state import report_live_usage
from ..llm_gateway import clone_config_for_gateway, gateway_subprocess_env, get_gateway_route
from ..output_schema import repair_schema_output
from ..resume_compaction import ResumeCompactionResult
from ..thinking_effort import normalize_thinking_effort
from .cli_process import END_OF_OUTPUT, CliLineReader, cli_exec_timeout_seconds
from .claude_code_backend import _terminate_process_tree
from .opencode_cli_backend import _role_env, _toolset_kwargs
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .tool_payloads import (
    augment_subprocess_pythonpath,
    configured_mcp_env_names,
    serialize_tools_payload,
)
from .web_search_fallback import fallback_web_mcp_url, keyless_web_search_enabled, web_search_enabled

_LOG = logging.getLogger("chack.code_cli_backend")

# Qwen's core allowlist only covers core tools. Disable the other built-ins
# explicitly, including the on-demand bridge that can rediscover deferred tools.
_QWEN_NON_CORE = (
    "get_goal", "update_goal", "tool_call", "tool_search", "read_mcp_resource",
    "agent", "list_agents", "task_stop", "send_message", "skill",
    "manage_memory", "search_memory", "report_findings", "enter_worktree",
    "exit_worktree", "record_artifact", "cron_create", "cron_list",
    "cron_delete", "loop_wakeup", "autofix_pr", "ask_user_question",
    "enter_plan_mode", "exit_plan_mode", "save_memory", "web_fetch",
)
_QWEN_CORE = (
    "run_shell_command", "read_file", "read_many_files", "write_file",
    "edit", "replace", "glob", "grep_search", "list_directory", "search_file_content",
    "web_search", "google_web_search",
)


@dataclass
class ToolAction:
    tool: str
    tool_input: Any


@dataclass
class RawResult:
    raw_responses: list[Any]
    time_to_first_token_seconds: float | None = None
    time_to_first_token_source: str = "unavailable"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    # Qwen's result.usage is already in this shape; accept Anthropic-style
    # cache counters as well without adding them twice when nested details exist.
    nested = value.get("input_tokens_details") or {}
    if not isinstance(nested, dict):
        nested = {}
    prompt = int(value.get("input_tokens", value.get("prompt_tokens", 0)) or 0)
    completion = int(value.get("output_tokens", value.get("completion_tokens", 0)) or 0)
    cached = int(nested.get("cached_tokens", value.get("cache_read_input_tokens", 0)) or 0)
    written = int(nested.get("cache_write_tokens", value.get("cache_creation_input_tokens", 0)) or 0)
    if not prompt and not completion:
        return None
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
    }


class CodeCliExecutor:
    def __init__(
        self, *, backend: str, config: ChackConfig, system_prompt: str,
        cli_path: str, model_name: str, api_key: str, base_url: str,
        tools_override: list[Any] | None, tools_append: list[Any] | None,
        memory_max_messages: int, memory_reset_to_messages: int,
        gateway_env: dict[str, str] | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._cli_path = cli_path
        self._system_prompt = system_prompt
        self._thinking_effort = normalize_thinking_effort(config.agent.thinking_effort)
        self._output_schema = config.agent.output_schema_json
        self._output_schema_name = _clean(config.agent.output_schema_name) or "output_schema"
        self._output_schema_strict = bool(config.agent.output_schema_strict)
        self._session_id: str | None = None
        self._conversation: list[dict[str, str]] = []
        self._memory_limit = memory_max_messages
        self._memory_reset_to = memory_reset_to_messages
        self._gateway_env = dict(gateway_env or {})
        self._tools_config_json = json.dumps(vars(config.tools), ensure_ascii=False, default=str)
        if tools_override is None:
            tools = list(AgentsToolset(
                config.tools, **_toolset_kwargs(config, backend)
            ).tools) + list(tools_append or [])
        else:
            tools = list(tools_override)
        names = [_clean(getattr(x, "name", None) or getattr(x, "__name__", "")) for x in tools]
        self._allowed_tools = list(dict.fromkeys(x for x in names if x and x not in {"Bash", "shell", "run_shell_command"}))
        try:
            self._override_b64 = serialize_tools_payload(tools_override)
        except Exception:
            # Research administrator tools capture unpicklable queue state.
            management = {
                "run_researchers_batch", "start_researchers_async", "list_researcher_jobs",
                "get_researcher_task", "poll_researchers_async", "get_researcher_result",
                "cancel_researcher_task", "retry_researcher_task", "cancel_researchers_async",
            }
            if not (set(self._allowed_tools) & management):
                raise
            self._override_b64 = ""
            self._override_names = json.dumps(self._allowed_tools)
        else:
            self._override_names = ""
        self._append_b64 = serialize_tools_payload(tools_append)
        self._home: str | None = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = _clean(payload.get("input"))
        prompt = self._compose_prompt(user_input)
        output, steps, raw = self._run(prompt)
        # A drifting answer gets one repair turn in the resumed session.
        output, repair_steps, repair_raw = repair_schema_output(
            output,
            schema=self._output_schema,
            name=self._output_schema_name,
            strict=self._output_schema_strict,
            resumable=bool(self._session_id),
            run_turn=self._run,
            label=self.backend,
        )
        steps.extend(repair_steps)
        raw.raw_responses.extend(repair_raw)
        if user_input:
            self._conversation.append({"role": "user", "content": user_input})
        if output:
            self._conversation.append({"role": "assistant", "content": output})
        if self._memory_limit and len(self._conversation) > self._memory_limit:
            self._conversation = self._conversation[-max(1, min(self._memory_limit, self._memory_reset_to or self._memory_limit)):]
        return {"output": output, "intermediate_steps": steps, "raw_result": raw}

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)

    def compact_for_resume(self, focus_instructions: str = "") -> ResumeCompactionResult:
        method = "/compact" if self.backend == "kimi" else "/compress"
        result = ResumeCompactionResult(backend=self.backend, method=method)
        if not self._session_id:
            return result
        result.attempted = True
        started = time.monotonic()
        try:
            command = method + (f" {focus_instructions.strip()}" if self.backend == "kimi" and focus_instructions.strip() else "")
            output, _steps, raw = self._run(command)
            result.raw_responses = list(raw.raw_responses)
            if output.lower().startswith("error:") or "unknown command" in output.lower():
                result.error = output[:1000]
            else:
                result.succeeded = True
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.duration_seconds = time.monotonic() - started
        return result

    def _compose_prompt(self, user_input: str) -> str:
        lines = [user_input]
        if self.config.agent.require_task_steps_manager_init_first and "task_steps_manager" in self._allowed_tools:
            lines.append("First call task_steps_manager with action=init before other tools.")
        minimum = int(self.config.tools.min_tools_used or 0)
        maximum = int(self.config.tools.max_tools_used or 0)
        if minimum:
            lines.append(f"Use at least {minimum} non-task tool calls.")
        if maximum:
            lines.append(f"Use no more than {maximum} non-task tool calls.")
        schema = getattr(self.config.agent, "output_schema_json", None)
        if schema:
            lines.append("Return exactly one JSON object matching this schema:\n" + json.dumps(schema, ensure_ascii=False))
        return "\n\n".join(x for x in lines if x)

    def _ensure_home(self) -> str:
        if self._home:
            return self._home
        base = os.environ.get(f"CHACK_{self.backend.upper()}_HOME", "")
        if not base:
            base = os.path.join(tempfile.gettempdir(), f"chack-{self.backend}-sessions")
        label = re.sub(r"[^A-Za-z0-9._-]", "_", current_session_id() or "default")
        home = os.path.join(os.path.abspath(os.path.expanduser(base)), label)
        os.makedirs(home, mode=0o700, exist_ok=True)
        os.chmod(home, 0o700)
        workspace = os.path.join(home, "workspace")
        os.makedirs(workspace, mode=0o700, exist_ok=True)
        self._home = home
        self._write_config()
        return home

    def _mcp_env(self, env: dict[str, str]) -> dict[str, str]:
        keys = set(configured_mcp_env_names(self._tools_config_json))
        keys.update({"PATH", "PYTHONPATH", "VIRTUAL_ENV", "OPENAI_API_KEY",
                     "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GH_TOKEN",
                     "KIMI_API_KEY", "KIMI_BASE_URL", "QWEN_API_KEY", "QWEN_BASE_URL",
                     "AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION",
                     "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                     "GOOGLE_APPLICATION_CREDENTIALS", "BRAVE_API_KEY", "SERPAPI_API_KEY",
                     "CHACK_CALLER_PYTHONPATH_JSON"})
        return {k: v for k, v in env.items() if k in keys or k.startswith(("CHACK_", "LLM_GATEWAY_", "OTEL_"))}

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        augment_subprocess_pythonpath(env)
        env.update(self._gateway_env)
        env.update(_role_env(self.config))
        env.update(child_process_env())
        env["PYTHONUNBUFFERED"] = "1"
        env["CHACK_TOOLS_CONFIG_JSON"] = self._tools_config_json
        env["CHACK_ALLOWED_TOOLS_JSON"] = json.dumps(self._allowed_tools)
        if self._override_b64:
            env["CHACK_TOOLS_OVERRIDE_B64"] = self._override_b64
        if self._override_names:
            env["CHACK_TOOLS_OVERRIDE_NAMES_JSON"] = self._override_names
        if self._append_b64:
            env["CHACK_TOOLS_APPEND_B64"] = self._append_b64
        env["CHACK_MODEL_PROVIDER"] = self.backend
        env["CHACK_MIN_TOOLS_USED"] = str(max(0, int(self.config.tools.min_tools_used or 0)))
        env["CHACK_MAX_TOOLS_USED"] = str(effective_max_tools_used(int(self.config.tools.max_tools_used or 0)))
        env["CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST"] = (
            "1" if self.config.agent.require_task_steps_manager_init_first and "task_steps_manager" in self._allowed_tools else "0"
        )
        env["CHACK_TASK_SESSION_ID"] = current_session_id() or ""
        env["CHACK_RUN_LABEL"] = current_run_label() or "Run 1"
        env["CHACK_MCP_PARENT_PID"] = str(os.getpid())
        home = self._home or self._ensure_home()
        if self.backend == "kimi":
            env["KIMI_CODE_HOME"] = home
            env["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
            env["KIMI_MODEL_NAME"] = self._model_name
            env["KIMI_MODEL_API_KEY"] = self._api_key
            env["KIMI_MODEL_PROVIDER_TYPE"] = "openai" if self._base_url else "kimi"
            if self._base_url:
                env["KIMI_MODEL_BASE_URL"] = self._base_url
            env["KIMI_CODE_TRUST_WORKSPACE"] = "1"
        else:
            env["QWEN_HOME"] = home
            env["OPENAI_API_KEY"] = self._api_key
            if self._base_url:
                env["OPENAI_BASE_URL"] = self._base_url
            env["OPENAI_MODEL"] = self._model_name
        return env

    def _write_config(self) -> None:
        assert self._home
        env = self._build_env_for_mcp()
        workspace = os.path.join(self._home, "workspace")
        mcp = {
            "chack_tools": {
                "command": sys.executable,
                "args": ["-m", "chack_agent.backends.chack_tools_mcp_server"],
                "cwd": os.environ.get("CHACK_EXEC_CWD") or os.getcwd(),
                "env": self._mcp_env(env),
            }
        }
        if web_search_enabled(self.config) and keyless_web_search_enabled(self.config):
            url = fallback_web_mcp_url()
            if url:
                mcp["web"] = {"url" if self.backend == "kimi" else "httpUrl": url}
        if getattr(self.config.tools, "playwright_enabled", False) and playwright_mcp_is_available():
            mcp["playwright"] = playwright_mcp_server_config()
        if self.backend == "kimi":
            allowed = ["mcp__chack_tools__*"]
            if "web" in mcp:
                allowed.append("mcp__web__*")
            if "playwright" in mcp:
                allowed.append("mcp__playwright__*")
            if web_search_enabled(self.config):
                allowed.extend(["WebSearch", "FetchURL"])
            with open(os.path.join(self._home, "config.toml"), "w", encoding="utf-8") as f:
                f.write("builtin_product_skills = false\nmerge_all_available_skills = false\ntelemetry = false\n")
                f.write("[tools]\nenabled = " + json.dumps(allowed) + "\n")
                f.write("[loop_control]\nmax_steps_per_turn = " + str(max(1, int(self.config.session.max_turns or 100))) + "\n")
                f.write("[thinking]\nenabled = " + ("false" if self._thinking_effort == "none" else "true") + "\n")
                f.write("effort = " + json.dumps("low" if self._thinking_effort in {"none", "minimal"} else self._thinking_effort) + "\n")
            with open(os.path.join(self._home, "mcp.json"), "w", encoding="utf-8") as f:
                json.dump({"mcpServers": mcp}, f)
            agent = os.path.join(self._home, "chack-agent.md")
            with open(agent, "w", encoding="utf-8") as f:
                f.write("---\nname: chack-agent\ndescription: Chack runtime agent\ntools:\n")
                for name in allowed:
                    f.write("  - " + name + "\n")
                f.write("subagents: []\n---\n\n" + self._system_prompt + "\n")
        else:
            # A nonempty core allowlist removes Qwen's file and shell tools.
            # The separate disabled set removes non-core synthetic tools.
            native_web = web_search_enabled(self.config)
            # Qwen sends its reasoning setting as an OpenAI-style request
            # field. Claude gateway routes such as Bakkt's Vertex route
            # reject that field even though the model itself works here.
            reasoning_setting = (
                {} if self._thinking_effort == "none" or "claude" in self._model_name.lower()
                else {"reasoningEffort": "low" if self._thinking_effort == "minimal" else self._thinking_effort}
            )
            settings = {
                "tools": {
                    "core": ["mcp_chack_tools_*"] + (["web_search", "web_fetch"] if native_web else []),
                    "disabled": [name for name in (*_QWEN_NON_CORE, *_QWEN_CORE)
                                 if not (native_web and name in {"web_search", "web_fetch"})],
                    "toolSearch": {"enabled": False},
                },
                "mcpServers": {
                    name: ({**value, "trust": True} if name == "chack_tools" else value)
                    for name, value in mcp.items()
                },
                "general": {"chatRecording": True},
                "privacy": {"usageStatisticsEnabled": False},
                "model": reasoning_setting,
            }
            settings_dir = os.path.join(self._home, "settings.json")
            with open(settings_dir, "w", encoding="utf-8") as f:
                json.dump(settings, f)
            # No project context, hooks, or settings from a repository under audit.
            os.makedirs(os.path.join(workspace, ".qwen"), exist_ok=True)
        for path in (os.path.join(self._home, "config.toml"), os.path.join(self._home, "mcp.json"),
                     os.path.join(self._home, "settings.json"), os.path.join(self._home, "chack-agent.md")):
            if os.path.exists(path):
                os.chmod(path, 0o600)

    def _build_env_for_mcp(self) -> dict[str, str]:
        env = os.environ.copy()
        augment_subprocess_pythonpath(env)
        env.update(self._gateway_env)
        env.update(_role_env(self.config))
        env.update(child_process_env())
        env.update({
            "CHACK_TOOLS_CONFIG_JSON": self._tools_config_json,
            "CHACK_ALLOWED_TOOLS_JSON": json.dumps(self._allowed_tools),
            "CHACK_TOOLS_OVERRIDE_B64": self._override_b64,
            "CHACK_TOOLS_OVERRIDE_NAMES_JSON": self._override_names,
            "CHACK_TOOLS_APPEND_B64": self._append_b64,
            "CHACK_MODEL_PROVIDER": self.backend,
            "CHACK_MIN_TOOLS_USED": str(max(0, int(self.config.tools.min_tools_used or 0))),
            "CHACK_MAX_TOOLS_USED": str(effective_max_tools_used(int(self.config.tools.max_tools_used or 0))),
            "CHACK_REQUIRE_TASK_STEPS_MANAGER_INIT_FIRST": (
                "1" if self.config.agent.require_task_steps_manager_init_first and "task_steps_manager" in self._allowed_tools else "0"
            ),
            "CHACK_TASK_SESSION_ID": current_session_id() or "",
            "CHACK_RUN_LABEL": current_run_label() or "Run 1",
            "CHACK_MCP_PARENT_PID": str(os.getpid()),
            f"{self.backend.upper()}_API_KEY": self._api_key,
        })
        if self._base_url:
            env[f"{self.backend.upper()}_BASE_URL"] = self._base_url
        return env

    def _command(self, prompt: str) -> list[str]:
        cmd = [self._cli_path, "-p", prompt, "--output-format", "stream-json"]
        if self.backend == "kimi":
            if self._session_id:
                cmd += ["--session", self._session_id]
            else:
                cmd += ["--agent-file", os.path.join(self._home or "", "chack-agent.md")]
        else:
            cmd += ["--model", self._model_name, "--system-prompt", self._system_prompt]
            cmd += ["--max-session-turns", str(max(1, int(self.config.session.max_turns or 100)))]
            tool_limit = effective_max_tools_used(int(self.config.tools.max_tools_used or 0))
            if tool_limit > 0:
                # Qwen's built-in web calls bypass the Chack MCP budget.
                # Bound the whole CLI as well, with room for the two Chack
                # management tools that do not consume the non-task budget.
                exempt = sum(name in self._allowed_tools for name in ("task_steps_manager", "check_budget_status"))
                cmd += ["--max-tool-calls", str(tool_limit + exempt)]
            if self._session_id:
                cmd += ["--resume", self._session_id]
        return cmd

    def _run(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], RawResult]:
        self._ensure_home()
        env = self._build_env()
        timeout = cli_exec_timeout_seconds(f"CHACK_{self.backend.upper()}_EXEC_TIMEOUT_SECONDS")
        try:
            process = subprocess.Popen(
                self._command(prompt), cwd=os.path.join(self._home or "", "workspace"),
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True,
            )
        except FileNotFoundError:
            return (f"ERROR: {self.backend} CLI executable not found: {self._cli_path}",
                    [], RawResult([]))
        except Exception as exc:
            return (f"ERROR: {self.backend} CLI failed to launch: {exc}", [], RawResult([]))

        registration = register_process(process, _terminate_process_tree)
        reader = CliLineReader(process.stdout)
        errors = CliLineReader(process.stderr)
        outputs: list[str] = []
        stderr: list[str] = []
        steps: list[tuple[ToolAction, Any]] = []
        raw: list[Any] = []
        calls: dict[str, tuple[str, Any]] = {}
        start = time.monotonic()
        first: float | None = None
        terminal_error = ""
        terminal_result_at: float | None = None
        stopped_after_result = False
        try:
            while True:
                if cancellation_requested():
                    _terminate_process_tree(process)
                    return (f"ERROR: {self.backend} execution cancelled.", steps, RawResult(raw))
                # Qwen prints its terminal result before an optional managed
                # memory drain, which can wait indefinitely after the answer.
                # Give that drain a short grace; the chat record is already
                # durable when the result is emitted.
                if terminal_result_at is not None and time.monotonic() - terminal_result_at > 5:
                    _terminate_process_tree(process)
                    stopped_after_result = True
                    break
                if timeout > 0 and time.monotonic() - start > timeout:
                    _terminate_process_tree(process)
                    return (f"ERROR: {self.backend} execution timed out after {timeout}s.", steps, RawResult(raw))
                error_line = errors.next_line(timeout=0)
                if isinstance(error_line, str) and error_line.strip():
                    stderr.append(error_line.strip())
                    stderr = stderr[-20:]
                line = reader.next_line()
                if line is None:
                    continue
                if line is END_OF_OUTPUT:
                    break
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                if first is None and event.get("type") not in {"system", "stream_event"}:
                    first = time.monotonic() - start
                issue = self._consume_event(event, outputs, steps, raw, calls)
                if issue:
                    terminal_error = issue
                if self.backend == "qwen" and event.get("type") == "result":
                    terminal_result_at = time.monotonic()
            code = process.wait()
            # stdout often closes before stderr has been pumped into its queue.
            # Drain it after exit so API and configuration errors remain visible.
            while True:
                error_line = errors.next_line(timeout=0.2)
                if error_line is END_OF_OUTPUT:
                    break
                if error_line is None:
                    break
                if isinstance(error_line, str) and error_line.strip():
                    stderr.append(error_line.strip())
                    stderr = stderr[-20:]
        finally:
            unregister_process(registration)
            if process.poll() is None:
                _terminate_process_tree(process)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if stopped_after_result and not terminal_error:
            code = 0
        if code != 0 or terminal_error:
            detail = terminal_error or "\n".join(stderr[-5:]) or "CLI exited without a result"
            return f"ERROR: {self.backend} CLI (exit={code}): {detail[:2000]}", steps, RawResult(raw)
        return "".join(outputs).strip(), steps, RawResult(raw, first, f"{self.backend}_first_response_event")

    def _consume_event(
        self, event: dict[str, Any], outputs: list[str],
        steps: list[tuple[ToolAction, Any]], raw: list[Any],
        calls: dict[str, tuple[str, Any]],
    ) -> str:
        kind = _clean(event.get("type"))
        if self.backend == "kimi":
            role = _clean(event.get("role"))
            if role == "meta" and kind == "session.resume_hint":
                self._session_id = _clean(event.get("session_id")) or self._session_id
            if role == "assistant":
                content = event.get("content")
                if isinstance(content, str):
                    outputs.append(content)
                for call in event.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function") or {}
                    if not isinstance(function, dict):
                        function = {}
                    name = _clean(function.get("name") or call.get("name"))
                    arg = function.get("arguments", call.get("arguments", {}))
                    if isinstance(arg, str):
                        try:
                            arg = json.loads(arg)
                        except ValueError:
                            pass
                    calls[_clean(call.get("id"))] = (name, arg)
            if role == "tool":
                # Earlier assistant text is often a tool preamble. The
                # answer is the assistant text after the last tool result.
                outputs.clear()
                ident = _clean(event.get("tool_call_id"))
                name, arg = calls.pop(ident, (_clean(event.get("name")), {}))
                self._record_tool(name, ident, arg, event.get("content"), steps)
            if role == "meta" and kind.endswith("usage"):
                self._record_usage(event.get("usage") or event, raw)
            return ""
        if kind == "system":
            self._session_id = _clean(event.get("session_id")) or self._session_id
        if kind == "assistant":
            message = event.get("message") or {}
            if isinstance(message, dict):
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        outputs.append(str(block.get("text") or ""))
                    elif block.get("type") == "tool_use":
                        calls[_clean(block.get("id"))] = (_clean(block.get("name")), block.get("input") or {})
        if kind in {"user", "tool_result"}:
            message = event.get("message") or {}
            blocks = message.get("content") if isinstance(message, dict) else event.get("content")
            if not isinstance(blocks, list):
                blocks = [blocks]
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    ident = _clean(block.get("tool_use_id"))
                    name, arg = calls.pop(ident, ("tool", {}))
                    self._record_tool(name, ident, arg, block.get("content"), steps)
        if kind == "result":
            self._session_id = _clean(event.get("session_id")) or self._session_id
            self._record_usage(event.get("usage"), raw)
            if isinstance(event.get("result"), str):
                # Earlier assistant events may include intermediate text while
                # Qwen is still running tools or managed memory. The terminal
                # result is the answer the CLI presents to its caller.
                outputs[:] = [event["result"]]
            if _clean(event.get("subtype")) not in {"success", ""} or event.get("is_error"):
                return _clean(event.get("error") or event.get("result") or event.get("subtype"))
        return ""

    def _record_tool(self, name: str, ident: str, arg: Any, result: Any, steps: list[tuple[ToolAction, Any]]) -> None:
        name = name.removeprefix("mcp__chack_tools__").removeprefix("mcp_chack_tools_")
        action = ToolAction(name or "tool", {"tool_id": ident, "tool_input": arg, "result": result})
        steps.append((action, None))
        if name == "task_steps_manager" and isinstance(arg, dict):
            from .opencode_cli_backend import OpencodeCliExecutor
            OpencodeCliExecutor._sync_task_steps_manager(arg)
        try:
            log_event("tool_called", payload={"tool": action.tool, "tool_input": action.tool_input},
                      task_session_id=current_session_id() or "", run_label=current_run_label() or "")
        except Exception:
            pass

    def _record_usage(self, value: Any, raw: list[Any]) -> None:
        usage = _usage(value)
        if usage is None:
            return
        raw.append({"usage": usage})
        details = usage["input_tokens_details"]
        report_live_usage(
            self._model_name,
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            cached_prompt_tokens=details["cached_tokens"],
            cache_write_tokens=details["cache_write_tokens"],
        )


def _build(
    backend: str, config: ChackConfig, *, system_prompt: str,
    memory_max_messages: int, memory_reset_to_messages: int,
    tools_override: list[Any] | None, tools_append: list[Any] | None,
) -> CodeCliExecutor:
    route = get_gateway_route(config)
    if route is not None:
        config = clone_config_for_gateway(config, route)
        model = route.model_name
        key = route.api_key
        base = route.base_url
        gateway_env = gateway_subprocess_env(config, route)
    else:
        model = _clean(config.model.primary)
        field = "kimi_api_key" if backend == "kimi" else "qwen_api_key"
        key = _clean(getattr(config.credentials, field, "")) or _clean(os.environ.get(field.upper()))
        base = _clean(os.environ.get(f"{backend.upper()}_BASE_URL"))
        gateway_env = {}
    if not key:
        raise ValueError(f"{backend} backend requires {field if route is None else 'LLM_GATEWAY_API_KEY'}")
    if backend == "qwen" and not base:
        raise ValueError("qwen backend requires QWEN_BASE_URL with QWEN_API_KEY, or an LLM gateway")
    configured = os.environ.get(f"{backend.upper()}_CLI_PATH", "").strip() or backend
    return CodeCliExecutor(
        backend=backend, config=config, system_prompt=system_prompt,
        cli_path=shutil.which(configured) or configured,
        model_name=model, api_key=key, base_url=base,
        tools_override=tools_override, tools_append=tools_append,
        memory_max_messages=memory_max_messages,
        memory_reset_to_messages=memory_reset_to_messages,
        gateway_env=gateway_env,
    )


def build_kimi_executor(
    config: ChackConfig, *, system_prompt: str, max_turns: int,
    memory_max_messages: int, memory_reset_to_messages: int,
    memory_summary_max_chars: int = 0, tools_override: list[Any] | None = None,
    tools_append: list[Any] | None = None,
) -> CodeCliExecutor:
    del max_turns, memory_summary_max_chars
    return _build("kimi", config, system_prompt=system_prompt,
                  memory_max_messages=memory_max_messages,
                  memory_reset_to_messages=memory_reset_to_messages,
                  tools_override=tools_override, tools_append=tools_append)


def build_qwen_executor(
    config: ChackConfig, *, system_prompt: str, max_turns: int,
    memory_max_messages: int, memory_reset_to_messages: int,
    memory_summary_max_chars: int = 0, tools_override: list[Any] | None = None,
    tools_append: list[Any] | None = None,
) -> CodeCliExecutor:
    del max_turns, memory_summary_max_chars
    return _build("qwen", config, system_prompt=system_prompt,
                  memory_max_messages=memory_max_messages,
                  memory_reset_to_messages=memory_reset_to_messages,
                  tools_override=tools_override, tools_append=tools_append)
