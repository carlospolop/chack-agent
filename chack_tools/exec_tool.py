import json
import subprocess
import os
import re
import tempfile

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
from .telemetry import run_with_tool_logging

def _command_env() -> dict[str, str]:
    """The caller's environment for shell commands.

    Inside the Chack MCP server PYTHONPATH also lists the runtime's own import
    roots; a command running another Python would import them and fail.
    OpenTelemetry settings are left out: tools that read them (docker,
    containerd, buildx) would export their own spans into the run's trace.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OTEL_") and key not in {"TRACEPARENT", "TRACESTATE"}
    }
    caller = env.pop("CHACK_CALLER_PYTHONPATH_JSON", None)
    if caller is None:
        return env
    try:
        pythonpath = json.loads(caller)
    except ValueError:
        return env
    if isinstance(pythonpath, str):
        env["PYTHONPATH"] = pythonpath
    else:
        env.pop("PYTHONPATH", None)
    return env


def _read_back(handle) -> str:
    handle.seek(0)
    return handle.read().decode("utf-8", errors="replace")


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
        deny_pattern = os.environ.get("CHACK_EXEC_DENY_REGEX", "").strip()
        if deny_pattern:
            try:
                blocked = re.search(deny_pattern, str(command or ""), flags=re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid CHACK_EXEC_DENY_REGEX: {exc}") from exc
            if blocked:
                raise PermissionError(
                    "Command blocked by the runtime execution policy. Use the assigned runtime "
                    "resources without stopping, rebooting, or reconfiguring shared infrastructure."
                )
        timeout = max(1, int(self.config.exec_timeout_seconds or 60))
        max_chars = max(1, int(self.config.exec_max_output_chars or 5000))
        resolved_cwd = self._resolve_cwd(cwd)
        # Output goes to files, not pipes: a process the command leaves running
        # (`server &`) inherits them, and waiting for pipes to close would hold
        # the call, and with it a tool-server thread, until that process ends.
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=stdout_file,
                stderr=stderr_file,
                env=_command_env(),
                cwd=resolved_cwd,
                start_new_session=True,
            )
            register_process_group(active_task_session_id(), process.pid)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_process_group(process.pid, grace_seconds=0.2)
                process.wait()
                raise subprocess.TimeoutExpired(
                    process.args,
                    timeout,
                    output=_read_back(stdout_file),
                    stderr=_read_back(stderr_file),
                )
            stdout, stderr = _read_back(stdout_file), _read_back(stderr_file)
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
        return run_with_tool_logging(name, tool_input, lambda: helper.run(command, cwd=cwd))

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
