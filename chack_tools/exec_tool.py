import os
import subprocess

from langchain_core.tools import tool

from .config import ToolsConfig
from .formatting import _truncate


@tool("exec")
def exec_tool(command: str) -> str:
    """Execute a shell command locally and return combined output."""
    timeout = int(os.environ.get("CHACK_EXEC_TIMEOUT", "120"))
    max_chars = int(os.environ.get("CHACK_EXEC_MAX_OUTPUT", "4000"))
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip() or "(no output)"
    return _truncate(output, max_chars)
