import subprocess
import time
import traceback
import os
from datetime import datetime, timezone

try:
    from agents import function_tool
except ImportError:
    function_tool = None

from .config import ToolsConfig
from .formatting import _truncate
from .run_lifecycle import (
    active_task_session_id,
    register_process_group,
    terminate_process_group,
)
from .telemetry import log_tool_started, log_tool_executed, log_tool_error

class ExecTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _resolve_cwd(self, cwd: str = "") -> str | None:
        candidate = str(cwd or "").strip()
        if candidate:
            return candidate
        candidate = str(getattr(self.config, "exec_cwd", "") or "").strip()
        if candidate:
            return candidate
        candidate = os.environ.get("CHACK_EXEC_CWD", "").strip()
        if candidate:
            return candidate
        return None

    def run(self, command: str, cwd: str = "") -> str:
        timeout = max(1, int(self.config.exec_timeout_seconds or 60))
        max_chars = max(1, int(self.config.exec_max_output_chars or 5000))
        resolved_cwd = self._resolve_cwd(cwd)
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=None,
            cwd=resolved_cwd,
            start_new_session=True,
        )
        register_process_group(active_task_session_id(), process.pid)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_group(process.pid, grace_seconds=0.2)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                process.args,
                timeout,
                output=stdout,
                stderr=stderr,
            )
        output = (stdout or "") + (stderr or "")
        output = output.strip() or "(no output)"
        return _truncate(output, max_chars)


def _build_exec_tool(helper: ExecTool, *, name: str):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override=name)
    def exec_tool(command: str, cwd: str = "") -> str:
        """Execute a shell command locally and return combined output.

        Use this to access local CLIs, curl/wget endpoints, and inspect files using tools like rg, find, ls, cat, head...
        If the output is large or truncated, re-run with grep/jq/sed to narrow it.
        Ideal for gathering evidence or checking system state; avoid destructive commands unless asked.
        This synchronous command has a hard timeout from tools.exec_timeout_seconds (default 60s).
        For long-running processes, start them in the background with explicit log/output files,
        then monitor them with quick follow-up commands such as ps, tail, grep, or cat.

        Args:
            command: The shell command to execute.
            cwd: Optional working directory for this command. If omitted, the tool falls back to
                `tools.exec_cwd`, then `CHACK_EXEC_CWD`.

        Output: Returns SUCCESS/ERROR-style text with the command exit code and combined stdout/stderr.
        Large output may be truncated, so narrow follow-up commands with rg/jq/sed/head/tail when needed.
        """
        tool_input = {"command": command, "cwd": cwd}
        start_ts = log_tool_started(name, tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.run(command, cwd=cwd)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                log_tool_error(
                    name,
                    tool_input,
                    error=error,
                    trace=traceback.format_exc(),
                )
            except Exception:
                pass
            raise
        finally:
            end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            duration_ms = int((time.time() - start_time) * 1000)
            log_tool_executed(
                name,
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    exec_tool.description = (
        f"{exec_tool.description}\n\n"
        "Parameters: Provide command as the local shell command to run and cwd only when it should run from a specific directory.\n"
        "Timeout: Synchronous commands are limited by tools.exec_timeout_seconds (default 60s); run long processes in the background and monitor logs/files with short follow-up commands.\n"
        "Output: Returns SUCCESS/ERROR-style text with the command exit code and combined stdout/stderr. Large output may be truncated; narrow follow-ups with rg/jq/sed/head/tail."
    )
    return exec_tool


def get_exec_tool(helper: ExecTool):
    return _build_exec_tool(helper, name="exec")


def get_controlled_shell_command_tool(helper: ExecTool):
    tool = _build_exec_tool(helper, name="run_shell_command")
    tool.description = (
        f"{tool.description}\n\n"
        "Use this controlled command-execution tool instead of any native shell. "
        "It is the researcher-safe command path with Chack timeouts and tool logging."
    )
    return tool
