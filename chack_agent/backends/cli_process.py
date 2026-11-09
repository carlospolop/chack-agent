"""Shared pieces for backends that run an agent CLI as a child process.

A backend reads the CLI's JSON events line by line. ``readline`` blocks until
the CLI prints, and a CLI waiting on a model answer prints nothing, so a loop
that checks for cancellation or its deadline only between lines can miss both
for as long as the CLI is quiet. ``CliLineReader`` reads on a thread instead,
so the loop can wake up on its own and check them.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import IO, Optional


# Without an explicit timeout, a CLI run gets the agent's own runtime budget
# (``max_runtime_minutes``, exported as CHACK_BUDGET_MAX_RUNTIME_SECONDS) plus
# this grace, like Codex runs; agents without a budget get the default.
CLI_RUNTIME_GRACE_SECONDS = 300
CLI_DEFAULT_TIMEOUT_SECONDS = 900
BUDGET_MAX_RUNTIME_ENV = "CHACK_BUDGET_MAX_RUNTIME_SECONDS"

# How often a quiet CLI's run loop wakes up to check cancellation and its deadline.
CLI_POLL_SECONDS = 0.2

END_OF_OUTPUT = object()


def cli_exec_timeout_seconds(timeout_env: str) -> int:
    """Wall-clock limit of one CLI run.

    ``timeout_env`` (e.g. CHACK_GEMINI_EXEC_TIMEOUT_SECONDS) wins when set;
    otherwise the agent's runtime budget plus a grace, and only then the default.
    """
    explicit = str(os.environ.get(timeout_env, "") or "").strip()
    if explicit:
        return int(explicit)
    try:
        budget = float(os.environ.get(BUDGET_MAX_RUNTIME_ENV, "") or 0)
    except ValueError:
        budget = 0.0
    if budget > 0:
        return int(budget) + CLI_RUNTIME_GRACE_SECONDS
    return CLI_DEFAULT_TIMEOUT_SECONDS


class CliLineReader:
    """Reads a child's text output on a daemon thread, one line at a time."""

    def __init__(self, stream: Optional[IO[str]]) -> None:
        self._lines: "queue.Queue[object]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        if stream is None:
            self._lines.put(END_OF_OUTPUT)
            return
        self._thread = threading.Thread(target=self._pump, args=(stream,), name="chack-cli-reader", daemon=True)
        self._thread.start()

    def _pump(self, stream: IO[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                self._lines.put(line)
        except (OSError, ValueError):
            # The stream was closed under the reader: the run is over.
            pass
        finally:
            self._lines.put(END_OF_OUTPUT)

    def next_line(self, timeout: float = CLI_POLL_SECONDS) -> object:
        """The next line, ``None`` when none came within ``timeout``, or ``END_OF_OUTPUT``."""
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
