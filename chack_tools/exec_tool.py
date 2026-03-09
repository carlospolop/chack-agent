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
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=None,
            cwd=resolved_cwd,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip() or "(no output)"
        return _truncate(output, max_chars)


def get_exec_tool(helper: ExecTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="exec")
    def exec_tool(command: str, cwd: str = "") -> str:
        """Execute a shell command locally and return combined output.

        Use this to access local CLIs, curl/wget endpoints, and inspect files using tools like rg, find, ls, cat, head...
        If the output is large or truncated, re-run with grep/jq/sed to narrow it.
        Ideal for gathering evidence or checking system state; avoid destructive commands unless asked.

        Args:
            command: The shell command to execute.
            cwd: Optional working directory for this command. If omitted, the tool falls back to
                `tools.exec_cwd`, then `CHACK_EXEC_CWD`.
        """
        tool_input = {"command": command, "cwd": cwd}
        start_ts = log_tool_started("exec", tool_input)
        start_time = time.time()
        error = None
        try:
            return helper.run(command, cwd=cwd)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                log_tool_error(
                    "exec",
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
                "exec",
                tool_input,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                error=error,
            )

    return exec_tool
