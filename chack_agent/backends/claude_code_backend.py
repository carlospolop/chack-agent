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
from ..live_cost_state import report_live_usage
from ..openrouter_routing import clone_config_for_openrouter, get_openrouter_route
from ..output_schema import JsonSchemaOutput
from .playwright_mcp import playwright_mcp_is_available, playwright_mcp_server_config
from .tool_payloads import (
    CHACK_TOOLS_APPEND_B64_ENV,
    CHACK_TOOLS_OVERRIDE_B64_ENV,
    serialize_tools_payload,
)


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
    _output_schema_name: str = "output_schema"
    _output_schema_strict: bool = True
    _uses_openrouter_route: bool = False
    _anthropic_api_key: str = ""
    _claude_access_token: str = ""
    _anthropic_base_url: str = ""
    _openrouter_http_referer: str = ""
    _openrouter_app_name: str = ""
    _claude_home: str | None = None
    _claude_session_id: str | None = None
    _output_schema: str | None = None

    def invoke(self, payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
        del context
        user_input = str(payload.get("input", "") or "")
        prompt = self._compose_prompt(user_input)
        output, steps, raw_result = self._run_claude(prompt)
        output = self._normalize_schema_output(output)

        # Save-or-retry: if save_discovered_vulnerability is available but was never called,
        # and the agent DID use other tools (analysis happened), re-run to prompt saving.
        if (
            self._has_save_vulnerability_tool()
            and not self._steps_contain_save(steps)
            and self._steps_contain_analysis(steps)
        ):
            env = self._build_env()
            exec_cwd = str(
                env.get("CHACK_EXEC_CWD", "")
                or os.environ.get("CHACK_EXEC_CWD", "")
                or ""
            ).strip() or "/tmp"
            _LOGGER.info(
                "Save-retry: no save_discovered_vulnerability calls detected. Re-running to prompt saving. cwd=%s",
                exec_cwd,
            )
            save_tool = f"{self._mcp_tool_prefix()}save_discovered_vulnerability"
            retry_prompt = (
                f"You have already analysed the source code in {exec_cwd}. "
                "Now you MUST save every vulnerability you found.\n\n"
                f"Call `{save_tool}` for each vulnerability with:\n"
                "  name, description, worst_impact, cvss_vector, remediation,\n"
                "  steps (array of {{file_path, code, description}}).\n\n"
                "You MUST save at least one vulnerability. Do NOT just describe findings in text."
            )
            retry_output, retry_steps, retry_raw = self._run_claude(retry_prompt)
            retry_output = self._normalize_schema_output(retry_output)
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

    def _normalize_schema_output(self, output: str) -> str:
        """Validate Claude CLI output against the configured schema when present.

        Claude's `--json-schema` flag improves results substantially, but the CLI can
        still emit wrapped text or partially-invalid JSON. Normalize the final payload
        through the same JsonSchemaOutput helper used by the OpenAI/OpenRouter backends
        so downstream agents consistently receive schema-shaped JSON.
        """
        if not self._output_schema_json:
            return output

        raw_schema = str(self._output_schema_json or "").strip()
        if not raw_schema:
            return output

        try:
            schema_obj = json.loads(raw_schema)
        except json.JSONDecodeError:
            _LOGGER.warning("Claude backend received invalid output schema JSON; skipping local validation")
            return output

        if not isinstance(schema_obj, dict):
            return output

        try:
            validator = JsonSchemaOutput(
                schema_obj,
                name=str(self._output_schema_name or "output_schema"),
                strict=bool(self._output_schema_strict),
            )
            validated = validator.validate_json(output or "")
            return json.dumps(validated, ensure_ascii=False)
        except Exception as exc:
            _LOGGER.warning("Claude backend output failed local schema normalization: %s", exc)
            return output

    async def aget_memory_messages(self) -> list[Any]:
        return list(self._conversation)

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
        # Claude Code names MCP tools as mcp__<server>__<tool>
        return "mcp__chack_tools__"

    @staticmethod
    def _steps_contain_save(steps: list[tuple[ToolAction, Any]]) -> bool:
        for step, _ in steps:
            if "save_discovered_vulnerability" in str(getattr(step, "tool", "") or ""):
                return True
        return False

    @staticmethod
    def _steps_contain_analysis(steps: list[tuple[ToolAction, Any]]) -> bool:
        analysis_tools = {"bash", "exec", "read", "grep", "glob"}
        for step, _ in steps:
            if str(getattr(step, "tool", "") or "").lower() in analysis_tools:
                return True
        return False

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
                f"  Call `{save_tool_name}` with parameters:\n"
                "    name, description, worst_impact, cvss_vector, remediation,\n"
                "    steps (array of objects with file_path, code, description).\n"
                "    Each step MUST include the actual file_path in the repo and the\n"
                "    verbatim source code snippet from that file.\n"
                "  If you do NOT save findings they are LOST."
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

    def _run_claude(self, prompt: str) -> tuple[str, list[tuple[ToolAction, Any]], _RawResult]:
        self._ensure_claude_home_and_settings()

        command = self._build_command(prompt)
        env = self._build_env()
        exec_cwd = str(env.get("CHACK_EXEC_CWD", "") or os.environ.get("CHACK_EXEC_CWD", "") or "").strip() or None
        agents_md_path = self._write_agents_md(exec_cwd)
        timeout_seconds = int(
            os.environ.get("CHACK_CLAUDE_EXEC_TIMEOUT_SECONDS", "") or "900"
        )

        _LOGGER.info(
            "Starting Claude Code process: model=%s timeout_seconds=%s session_id=%s cwd=%s",
            self._model_name,
            timeout_seconds,
            self._claude_session_id or "",
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
            # Close stdin immediately — prompt is already in the command args
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass

            while True:
                if timeout_seconds > 0 and (time.monotonic() - started_at) > timeout_seconds:
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
                    structured_output = event.get("structured_output")
                    if structured_output is not None:
                        if isinstance(structured_output, (dict, list)):
                            output_parts.append(json.dumps(structured_output, ensure_ascii=False))
                        else:
                            structured_text = str(structured_output).strip()
                            if structured_text:
                                output_parts.append(structured_text)
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
                        report_live_usage(
                            self._model_name,
                            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                            completion_tokens=int(usage.get("output_tokens", 0) or 0),
                            cached_prompt_tokens=int(
                                usage.get("input_tokens_details", {}).get("cached_tokens", 0) or 0
                            ),
                            cache_write_tokens=int(
                                usage.get("input_tokens_details", {}).get("cache_write_tokens", 0) or 0
                            ),
                        )
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
                    # StructuredOutput tool carries the --json-schema answer
                    _tu_name = str(event.get("name") or "").strip()
                    if _tu_name == "StructuredOutput":
                        _tu_input = event.get("input")
                        if isinstance(_tu_input, dict):
                            output_parts.append(json.dumps(_tu_input))
                        elif _tu_input is not None:
                            output_parts.append(str(_tu_input))
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

        _LOGGER.info(
            "Claude process exit: return_code=%s raw_lines=%d output_parts=%d return_seen=%s",
            return_code,
            len(raw_lines),
            len(output_parts),
            return_seen,
        )
        if raw_lines:
            _LOGGER.info("Claude raw_lines[0]: %s", raw_lines[0][:300] if raw_lines else "")
        if return_code != 0 and raw_lines:
            _LOGGER.warning("Claude stderr/stdout dump (first 1000 chars): %s", "\n".join(raw_lines)[:1000])

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

        # Pick up any JSON-fallback vulns written via save_vuln.sh
        bash_vuln_steps = self._collect_bash_saved_vulns(exec_cwd)
        if bash_vuln_steps:
            steps.extend(bash_vuln_steps)

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
                    # StructuredOutput tool carries the --json-schema answer
                    if tool_name == "StructuredOutput" and tool_input:
                        output_parts.append(
                            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
                        )
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

    def _write_agents_md(self, exec_cwd: str | None) -> str | None:
        """Write AGENTS.md + save_vuln.sh + .vulns/ directory in exec_cwd so the model
        knows how to save vulnerabilities. Only written when save_discovered_vulnerability
        is available. Returns path to AGENTS.md or None."""
        if not exec_cwd or not self._has_save_vulnerability_tool():
            return None
        agents_md_path = os.path.join(exec_cwd, "AGENTS.md")
        if os.path.exists(agents_md_path):
            return agents_md_path

        try:
            mcp_prefix = self._mcp_tool_prefix()
            save_tool_name = f"{mcp_prefix}save_discovered_vulnerability"

            # Create .vulns/ directory for bash JSON fallback
            vulns_dir = os.path.join(exec_cwd, ".vulns")
            os.makedirs(vulns_dir, exist_ok=True)

            # Write save_vuln.sh
            save_vuln_path = os.path.join(exec_cwd, "save_vuln.sh")
            save_vuln_sh = '''#!/usr/bin/env bash
# save_vuln.sh — Save a vulnerability as validated JSON to .vulns/
# Usage: bash save_vuln.sh '<JSON>'
set -e
JSON="$1"
if [ -z "$JSON" ]; then
    echo "Usage: bash save_vuln.sh '<JSON>'" >&2
    exit 1
fi
mkdir -p .vulns
python3 - "$JSON" <<'PYEOF'
import sys, json, time
data = json.loads(sys.argv[1])
errors = []
if not data.get("name"):
    errors.append("missing: name")
if not data.get("description"):
    errors.append("missing: description")
if errors:
    print("ERROR: Vulnerability NOT saved. Fix these issues:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
fname = f".vulns/vuln_{int(time.time()*1000)}.json"
with open(fname, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved to {fname}")
PYEOF
'''
            with open(save_vuln_path, "w", encoding="utf-8") as f:
                f.write(save_vuln_sh)
            os.chmod(save_vuln_path, 0o755)

            agents_md_content = f"""# Agent Instructions

## MANDATORY: Saving Vulnerability Findings

### Method 1 (PREFERRED): Call `{save_tool_name}` directly
Parameters:
  - `name`: vulnerability title (required)
  - `description`: detailed description (required)
  - `worst_impact`: what the attacker gains
  - `cvss_vector`: CVSS:3.0 vector string
  - `remediation`: how to fix it
  - `steps`: JSON array of exploitation steps
  - `is_full_eop`: 1 if full privilege escalation chain, 0 otherwise
  - `poc`: working PoC script (required for is_full_eop=1)
  - `category`: vulnerability class (dylib_injection, xpc_abuse, etc.)
  - `attack_vector`: trigger path

### Method 2 (FALLBACK): Run `bash save_vuln.sh '<JSON>'`
Example:
```bash
bash save_vuln.sh '{{
  "name": "Vulnerability Title",
  "description": "Detailed description",
  "worst_impact": "root code execution",
  "cvss_vector": "CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
  "remediation": "Fix description",
  "steps": [],
  "is_full_eop": 0
}}'
```

**If you do NOT save findings they are LOST. You MUST save every real vulnerability.**
"""
            with open(agents_md_path, "w", encoding="utf-8") as f:
                f.write(agents_md_content)

            # Also write to .github/copilot-instructions.md for compatibility
            gh_dir = os.path.join(exec_cwd, ".github")
            os.makedirs(gh_dir, exist_ok=True)
            gh_instructions = os.path.join(gh_dir, "copilot-instructions.md")
            with open(gh_instructions, "w", encoding="utf-8") as f:
                f.write(agents_md_content)

            _LOGGER.info("Wrote AGENTS.md to %s", agents_md_path)
            return agents_md_path
        except Exception as exc:
            _LOGGER.warning("Failed to write AGENTS.md: %s", exc)
            return None

    @staticmethod
    def _collect_bash_saved_vulns(exec_cwd: str | None) -> list[tuple[ToolAction, Any]]:
        """Read JSON files from .vulns/ directory written by save_vuln.sh and convert to steps."""
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
            _LOGGER.info("Collected %d JSON-fallback vulnerabilities from %s", len(steps), vulns_dir)
        return steps

    def _build_command(self, prompt: str) -> list[str]:
        try:
            tools_cfg = json.loads(self._tools_config_json or "{}")
            _exec_enabled = bool(tools_cfg.get("exec_enabled", False))
        except Exception:
            _exec_enabled = False
        builtin_tools = "Bash" if _exec_enabled else ""

        args: list[str] = [
            self._claude_cli_path,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--tools",
            builtin_tools,
            "--dangerously-skip-permissions",
            "--mcp-config",
            os.path.join(self._claude_home or os.getcwd(), "settings.json"),
            "--strict-mcp-config",
            prompt,
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

        if self._uses_openrouter_route:
            env["ANTHROPIC_API_KEY"] = self._anthropic_api_key
            env["CLAUDE_API_KEY"] = self._anthropic_api_key
            env["ANTHROPIC_AUTH_TOKEN"] = self._anthropic_api_key
            env["ANTHROPIC_BASE_URL"] = self._anthropic_base_url
            env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
            env["OPENROUTER_API_KEY"] = self._anthropic_api_key
            if self._openrouter_http_referer:
                env["OPENROUTER_HTTP_REFERER"] = self._openrouter_http_referer
            if self._openrouter_app_name:
                env["OPENROUTER_APP_NAME"] = self._openrouter_app_name
        else:
            if self._claude_access_token:
                env["CLAUDE_ACCESS_TOKEN"] = self._claude_access_token
                env.pop("ANTHROPIC_API_KEY", None)
                env.pop("CLAUDE_API_KEY", None)
                env.pop("ANTHROPIC_AUTH_TOKEN", None)
            else:
                _api_key = str(
                    self._anthropic_api_key
                    or os.environ.get("ANTHROPIC_API_KEY", "")
                    or os.environ.get("CLAUDE_API_KEY", "")
                )
                if _api_key:
                    env["ANTHROPIC_API_KEY"] = _api_key

        # Allow --dangerously-skip-permissions when running as root inside Docker/CI.
        env.setdefault("IS_SANDBOX", "1")

        env.setdefault("AWS_SHARED_CREDENTIALS_FILE", os.environ.get("AWS_SHARED_CREDENTIALS_FILE", ""))
        env.setdefault("AWS_CONFIG_FILE", os.environ.get("AWS_CONFIG_FILE", ""))

        # Claude Code internal tool timeouts -----------------------------------
        # BASH_DEFAULT_TIMEOUT_MS: default timeout for the Bash tool (default 120s)
        env.setdefault(
            "BASH_DEFAULT_TIMEOUT_MS",
            os.environ.get("CHACK_CLAUDE_BASH_DEFAULT_TIMEOUT_MS", "120000"),
        )
        # BASH_MAX_TIMEOUT_MS: hard cap the model can request for Bash (default 600s)
        env.setdefault(
            "BASH_MAX_TIMEOUT_MS",
            os.environ.get("CHACK_CLAUDE_BASH_MAX_TIMEOUT_MS", "600000"),
        )
        # MCP_TOOL_TIMEOUT: per-tool timeout for MCP calls.
        # Claude Code defaults this to 1e8 ms (~27 hours) which is effectively
        # infinite and can cause the agent to hang. We cap it at 5 minutes.
        env.setdefault(
            "MCP_TOOL_TIMEOUT",
            os.environ.get("CHACK_CLAUDE_MCP_TOOL_TIMEOUT_MS", "300000"),
        )

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
        if self._playwright_mcp_enabled():
            settings_payload["mcpServers"]["playwright"] = playwright_mcp_server_config()

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

    def _playwright_mcp_enabled(self) -> bool:
        try:
            cfg = json.loads(self._tools_config_json or "{}")
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        return bool(cfg.get("playwright_enabled")) and playwright_mcp_is_available()

    def _mcp_env_map(self) -> dict[str, str]:
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
            "CHACK_EXEC_CWD",
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
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
            "OPENROUTER_API_KEY",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_NAME",
            "PYTHONPATH",
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

    require_task_steps_manager_init_first = bool(
        getattr(config.agent, "require_task_steps_manager_init_first", True)
        and ("task_steps_manager" in allowed_tool_names)
    )

    configured_claude_path = os.environ.get("CLAUDE_CLI_PATH", "").strip() or "claude"
    claude_cli_path = shutil.which(configured_claude_path) or configured_claude_path
    serialized_tools_override_b64 = serialize_tools_payload(tools_override)
    serialized_tools_append_b64 = serialize_tools_payload(tools_append)

    route = get_openrouter_route(config)

    return ClaudeCodeExecutor(
        _conversation=[],
        _memory_limit=memory_max_messages,
        _memory_reset_to=memory_reset_to_messages,
        _base_system_prompt=system_prompt,
        _model_name=str(route.model_name if route is not None else config.model.primary),
        _max_turns=int(config.session.max_turns or 100),
        _claude_cli_path=claude_cli_path,
        _tools_config_json=json.dumps(getattr(config.tools, "__dict__", {}), ensure_ascii=False),
        _allowed_tools_json=json.dumps(allowed_tool_names, ensure_ascii=False),
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
        _require_task_steps_manager_init_first=require_task_steps_manager_init_first,
        _output_schema_json=(
            json.dumps(config.agent.output_schema_json, ensure_ascii=False)
            if getattr(config.agent, "output_schema_json", None)
            else ""
        ),
        _output_schema_name=str(getattr(config.agent, "output_schema_name", "") or "output_schema"),
        _output_schema_strict=bool(getattr(config.agent, "output_schema_strict", True)),
        _uses_openrouter_route=route is not None,
        _anthropic_api_key=str(
            route.api_key
            if route is not None
            else (
                os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("CLAUDE_API_KEY", "")
            )
        ),
        _claude_access_token=str("" if route is not None else os.environ.get("CLAUDE_ACCESS_TOKEN", "")),
        _anthropic_base_url=str(route.anthropic_base_url if route is not None else ""),
        _openrouter_http_referer=str((route.headers.get("HTTP-Referer", "") if route else "")),
        _openrouter_app_name=str((route.headers.get("X-Title", "") if route else "")),
    )
