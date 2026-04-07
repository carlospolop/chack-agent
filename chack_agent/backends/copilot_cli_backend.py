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
        deny_builtin_tools: list[str] | None = None,
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
        self._deny_builtin_tools: list[str] = list(deny_builtin_tools or [])

        self._copilot_home: str | None = None
        self._copilot_session_id: str | None = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_copilot(prompt)

        # Save-or-retry: if save_discovered_vulnerability is available but was
        # never called, and the agent DID use other tools (meaning it analysed
        # code), re-run asking it to save its findings.
        if (
            self._has_save_vulnerability_tool()
            and not self._steps_contain_save(steps)
            and self._steps_contain_analysis(steps)
        ):
            exec_cwd = str(
                self._build_env().get("CHACK_EXEC_CWD", "")
                or os.environ.get("CHACK_EXEC_CWD", "")
                or ""
            ).strip() or "/tmp"
            _LOGGER.info(
                "Save-retry: no save_discovered_vulnerability calls detected after %d tool calls. "
                "Re-running %sto prompt saving. cwd=%s",
                len(steps),
                "with --resume " if self._copilot_session_id else "fresh ",
                exec_cwd,
            )
            mcp_prefix = self._mcp_tool_prefix()
            save_tool = f"{mcp_prefix}save_discovered_vulnerability"
            retry_prompt = (
                "You are a security auditor. You have already analysed the source code "
                f"in {exec_cwd}. Now you MUST save every vulnerability you found.\n\n"
                f"PREFERRED: Call `{save_tool}` for each vulnerability with:\n"
                "  name, description, worst_impact, cvss_vector, remediation,\n"
                "  steps (array of {{file_path, code, description}}).\n"
                "  Each step MUST have the actual source file path and a verbatim code snippet.\n\n"
                "FALLBACK: Run `bash save_vuln.sh <<'VULN_EOF'` with a heredoc (see AGENTS.md for format).\n"
                "  NEVER use single-quoted arguments — use heredoc to avoid quoting issues.\n"
                "  The script validates all fields and rejects incomplete data — read error output.\n\n"
                "You MUST save at least one vulnerability. "
                "Do NOT just describe findings in text — call the tool or run save_vuln.sh."
            )
            retry_output, retry_steps, retry_raw = self._run_copilot(retry_prompt)
            # Merge results
            if retry_output:
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

    @staticmethod
    def _steps_contain_save(steps: list[tuple[ToolAction, Any]]) -> bool:
        """Check if any step is a save_discovered_vulnerability call."""
        for step, _ in steps:
            tool_name = str(getattr(step, "tool", "") or "")
            if "save_discovered_vulnerability" in tool_name:
                return True
        return False

    @staticmethod
    def _steps_contain_analysis(steps: list[tuple[ToolAction, Any]]) -> bool:
        """Check if steps contain file-reading / analysis tools (bash, view, grep, exec)."""
        analysis_tools = {"bash", "view", "grep", "exec", "read_file"}
        for step, _ in steps:
            tool_name = str(getattr(step, "tool", "") or "")
            if tool_name in analysis_tools:
                return True
        return False

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

        # Inject save requirement when save_discovered_vulnerability is available
        if self._has_save_vulnerability_tool():
            mcp_prefix = self._mcp_tool_prefix()
            save_tool_name = f"{mcp_prefix}save_discovered_vulnerability"
            save_block = (
                "- CRITICAL: For EACH vulnerability you find, you MUST save it.\n"
                f"  PREFERRED: Call `{save_tool_name}` with parameters:\n"
                "    name, description, worst_impact, cvss_vector, remediation,\n"
                "    steps (array of objects with file_path, code, description).\n"
                "    Each step MUST include the actual file_path in the repo and the\n"
                "    verbatim source code snippet from that file.\n"
                "  FALLBACK: Run `bash save_vuln.sh <<'VULN_EOF'` with a heredoc — it validates all fields.\n"
                "  NEVER use `bash save_vuln.sh '{...}'` with single quotes — it breaks on special characters.\n"
                "  If you do NOT save findings they are LOST."
            )
            # Warn about built-in tools that silently discard findings
            builtin_warnings: list[str] = []
            for builtin in ("report_intent", "task"):
                if builtin not in self._deny_builtin_tools:
                    builtin_warnings.append(f"`{builtin}`")
            if builtin_warnings:
                save_block += (
                    "\n  ⚠ NEVER use " + " or ".join(builtin_warnings) +
                    " — they silently discard your findings."
                )
            policy_lines.append(save_block)

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
        """Return the MCP tool name prefix based on the MCP server name in config."""
        if self._copilot_home:
            mcp_cfg = os.path.join(self._copilot_home, "mcp-config.json")
            try:
                with open(mcp_cfg, "r") as fh:
                    cfg = json.loads(fh.read())
                servers = cfg.get("mcpServers", {})
                if servers:
                    server_name = next(iter(servers))
                    return f"{server_name}-"
            except Exception:
                pass
        return "chack_tools-"

    # ------------------------------------------------------------------
    #  Core execution
    # ------------------------------------------------------------------

    def _run_copilot(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_copilot_home_and_config()
        command = self._build_command(prompt)
        env = self._build_env()
        exec_cwd = str(env.get("CHACK_EXEC_CWD", "") or os.environ.get("CHACK_EXEC_CWD", "") or "").strip() or None
        agents_md_path = self._write_agents_md(exec_cwd)
        timeout_seconds = int(
            os.environ.get("CHACK_COPILOT_EXEC_TIMEOUT_SECONDS", "900") or "900"
        )
        try:
            output, steps, raw_result = self.__run_copilot_subprocess(command, env, exec_cwd, timeout_seconds)
            # Collect any vulnerabilities saved via the bash helper script
            bash_vuln_steps = self._collect_bash_saved_vulns(exec_cwd)
            if bash_vuln_steps:
                steps.extend(bash_vuln_steps)
            return output, steps, raw_result
        finally:
            self._cleanup_agents_md(agents_md_path)

    def __run_copilot_subprocess(self, command, env, exec_cwd, timeout_seconds):
        _LOGGER.info(
            "Starting Copilot CLI process: model=%s timeout_seconds=%s session_id=%s cwd=%s",
            self._model_name,
            timeout_seconds,
            self._copilot_session_id or "",
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
        for denied in self._deny_builtin_tools:
            args.extend(["--deny-tool", str(denied)])
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

    def _write_agents_md(self, exec_cwd: str | None) -> str | None:
        """Write an AGENTS.md in the workspace so copilot CLI loads it as
        system-level instructions.  Also creates a save_vuln.sh helper
        script and .vulns/ directory for the JSON fallback mechanism.
        Returns the path written (for cleanup) or *None* if nothing was written."""
        if not exec_cwd:
            return None
        if not self._deny_builtin_tools and not self._has_save_vulnerability_tool():
            return None

        agents_md_path = os.path.join(exec_cwd, "AGENTS.md")
        # Don't overwrite a pre-existing file (it belongs to the target repo)
        if os.path.exists(agents_md_path):
            _LOGGER.debug("AGENTS.md already exists at %s — skipping write", agents_md_path)
            return None

        denied = ", ".join(f"`{t}`" for t in self._deny_builtin_tools)
        mcp_prefix = self._mcp_tool_prefix()
        save_tool = f"{mcp_prefix}save_discovered_vulnerability"

        # Create .vulns/ directory for bash JSON fallback
        vulns_dir = os.path.join(exec_cwd, ".vulns")
        try:
            os.makedirs(vulns_dir, exist_ok=True)
        except OSError:
            pass

        # Write save_vuln.sh — a validating helper script
        save_vuln_path = os.path.join(exec_cwd, "save_vuln.sh")
        save_vuln_script = r'''#!/usr/bin/env bash
# save_vuln.sh — Save a vulnerability as validated JSON to .vulns/
# Usage (heredoc — PREFERRED, avoids quoting issues):
#   bash save_vuln.sh <<'VULN_EOF'
#   {"name": "...", ...}
#   VULN_EOF
# Usage (argument — legacy):
#   bash save_vuln.sh '<JSON>'
set -euo pipefail
mkdir -p .vulns
if [ $# -ge 1 ] && [ "$1" != "-" ]; then
  JSON="$1"
else
  JSON="$(cat)"
fi
python3 -c "
import json, sys, os, time
try:
    d = json.loads(sys.argv[1])
except Exception as e:
    print(f'ERROR: Invalid JSON: {e}', file=sys.stderr)
    print('Fix the JSON syntax and call save_vuln.sh again.', file=sys.stderr)
    sys.exit(1)
errors = []
for f in ['name','description','worst_impact','cvss_vector','remediation']:
    if not str(d.get(f,'')).strip():
        errors.append(f'Missing required field: {f}')
steps = d.get('steps')
if not isinstance(steps, list) or len(steps) == 0:
    errors.append('steps must be a non-empty array of objects')
else:
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            errors.append(f'steps[{i}] must be an object with file_path, code, description')
            continue
        fp = str(s.get('file_path','')).strip()
        code = str(s.get('code','')).strip()
        desc = str(s.get('description','')).strip()
        if not fp or fp == 'unknown':
            errors.append(f'steps[{i}].file_path is missing — must be the actual source file path')
        elif not os.path.isfile(fp):
            errors.append(f'steps[{i}].file_path \"{fp}\" does not exist — check the path')
        if not code:
            errors.append(f'steps[{i}].code is empty — must contain the actual vulnerable source code snippet')
        if not desc:
            errors.append(f'steps[{i}].description is empty — explain why this code is vulnerable')
if errors:
    print('ERROR: Vulnerability NOT saved. Fix these issues and call save_vuln.sh again:', file=sys.stderr)
    for e in errors:
        print(f'  - {e}', file=sys.stderr)
    sys.exit(1)
fname = f'.vulns/vuln_{int(time.time()*1000)}.json'
with open(fname, 'w') as f:
    json.dump(d, f, indent=2)
print(f'OK: Saved vulnerability \"{d[\"name\"]}\" to {fname}')
" "$JSON"
'''
        try:
            with open(save_vuln_path, "w", encoding="utf-8") as fh:
                fh.write(save_vuln_script)
            os.chmod(save_vuln_path, 0o755)
        except OSError:
            pass

        # Build AGENTS.md content
        content = (
            "# Copilot Agent Instructions\n\n"
            "## MANDATORY: How to Save Findings\n\n"
            "When you discover a vulnerability, you **MUST** save it.\n\n"
            f"### Method 1 (PREFERRED): Call the MCP tool `{save_tool}`\n"
            "Parameters:\n"
            "- `name`: Vulnerability name\n"
            "- `description`: Detailed description with attack flow\n"
            "- `worst_impact`: Worst case impact\n"
            "- `cvss_vector`: CVSS:3.0 vector string\n"
            "- `remediation`: How to fix\n"
            "- `steps`: Array of step objects, EACH with:\n"
            "  - `file_path`: Path to the affected source file (e.g., `app.py`)\n"
            "  - `code`: Verbatim source code snippet from that file\n"
            "  - `description`: Why this code is vulnerable\n\n"
            "### Method 2 (FALLBACK): Run `bash save_vuln.sh` with a heredoc\n"
            "If the MCP tool is unavailable, use the helper script with a **heredoc**.\n"
            "IMPORTANT: Always use a heredoc (<<'VULN_EOF') to avoid bash quoting issues:\n"
            "```bash\n"
            "bash save_vuln.sh <<'VULN_EOF'\n"
            '{\n'
            '  "name": "SQL Injection in login",\n'
            '  "description": "User input is concatenated into SQL query without parameterization",\n'
            '  "worst_impact": "Full database compromise",\n'
            '  "cvss_vector": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",\n'
            '  "remediation": "Use parameterized queries",\n'
            '  "steps": [\n'
            "    {\n"
            '      "file_path": "app/routes/login.py",\n'
            '      "code": "query = f\\"SELECT * FROM users WHERE id={user_id}\\"",\n'
            '      "description": "Unsanitized user input in SQL query"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "VULN_EOF\n"
            "```\n"
            "NEVER use `bash save_vuln.sh '{...}'` with single quotes — it breaks on special characters.\n"
            "If save_vuln.sh rejects the data, read the error output and fix the issues.\n\n"
            "### Rules\n"
            "- You MUST save EVERY vulnerability found using one of the methods above\n"
            "- EVERY step MUST have a real `file_path` (existing file) and `code` (verbatim source snippet)\n"
            "- `code` must be copied from the source file, NOT analysis text\n"
            "- Do NOT just describe findings in text — they are LOST unless saved\n\n"
        )
        if denied:
            content += (
                "## FORBIDDEN Tools\n\n"
                f"The following built-in tools must NEVER be called: {denied}.\n"
            )
        # Dynamic warning for specific built-in tools
        for builtin in ("report_intent", "task"):
            if builtin not in self._deny_builtin_tools:
                content += f"`{builtin}` silently discards your findings. Never use it.\n"
        try:
            with open(agents_md_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            _LOGGER.info("Wrote AGENTS.md to %s", agents_md_path)

            # Also write .github/copilot-instructions.md (copilot CLI reads this too)
            gh_dir = os.path.join(exec_cwd, ".github")
            os.makedirs(gh_dir, exist_ok=True)
            copilot_instr_path = os.path.join(gh_dir, "copilot-instructions.md")
            if not os.path.exists(copilot_instr_path):
                with open(copilot_instr_path, "w", encoding="utf-8") as fh:
                    fh.write(content)

            return agents_md_path
        except OSError as exc:
            _LOGGER.warning("Failed to write AGENTS.md: %s", exc)
            return None

    @staticmethod
    def _cleanup_agents_md(path: str | None) -> None:
        """Remove the AGENTS.md, save_vuln.sh, .vulns, and .github/copilot-instructions.md."""
        if path is None:
            return
        try:
            os.remove(path)
        except OSError:
            pass
        parent = os.path.dirname(path)
        # Clean up helper files
        for cleanup in (
            os.path.join(parent, "save_vuln.sh"),  # legacy cleanup
            os.path.join(parent, ".github", "copilot-instructions.md"),
        ):
            try:
                os.remove(cleanup)
            except OSError:
                pass
        # Clean up .vulns directory
        vulns_dir = os.path.join(parent, ".vulns")
        try:
            import shutil as _shutil
            _shutil.rmtree(vulns_dir, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _collect_bash_saved_vulns(exec_cwd: str | None) -> list[tuple[ToolAction, Any]]:
        """Read JSON files from .vulns/ directory written by the model as
        fallback, and convert them to ToolAction steps matching
        save_discovered_vulnerability."""
        if not exec_cwd:
            return []
        vulns_dir = os.path.join(exec_cwd, ".vulns")
        if not os.path.isdir(vulns_dir):
            return []
        steps: list[tuple[ToolAction, Any]] = []
        for fname in os.listdir(vulns_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(vulns_dir, fname)
            try:
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
                # Preserve steps with file_path and code if provided
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
            _LOGGER.info(
                "Collected %d JSON-fallback vulnerabilities from %s",
                len(steps),
                vulns_dir,
            )
        return steps

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
            "PYTHONPATH",
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
        deny_builtin_tools=list(getattr(config.tools, "deny_builtin_tools", None) or []),
    )
