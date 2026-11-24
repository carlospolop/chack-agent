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

# When the machine runs out of memory, Linux kills the process with the highest
# OOM score, normally the biggest. That is often the host process running every
# agent, which loses all of them at once. A raised score makes the kernel pick an
# agent CLI first: one run fails and can be retried. Processes the CLI starts
# afterwards (MCP servers, tool commands) inherit the score.
CLI_OOM_SCORE_ADJ = 500


def prefer_oom_kill(pid: int) -> None:
    """Have the kernel kill this child before its parent when memory runs out.

    Raising a score needs no privilege. Where ``/proc`` is absent (macOS,
    Windows) this does nothing.
    """
    try:
        with open(f"/proc/{pid}/oom_score_adj", "w", encoding="ascii") as handle:
            handle.write(str(CLI_OOM_SCORE_ADJ))
    except OSError:
        pass


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


# A CLI run that prints nothing for this long beyond its longest Chack command
# (``tools.exec_timeout_seconds``) is stopped as hung. The grace covers one
# model answer with the CLI's own retries of a failing provider.
CLI_IDLE_GRACE_SECONDS = 1800


def cli_idle_timeout_seconds(idle_env: str, exec_timeout_seconds: int) -> int:
    """How long one CLI run may print nothing before it is stopped as hung.

    ``idle_env`` (e.g. CHACK_OPENCODE_IDLE_TIMEOUT_SECONDS) wins when set, and 0
    turns the limit off; otherwise the run's longest command plus a grace.
    """
    explicit = str(os.environ.get(idle_env, "") or "").strip()
    if explicit:
        return int(explicit)
    return max(0, int(exec_timeout_seconds or 0)) + CLI_IDLE_GRACE_SECONDS


EXEC_CWD_ENV = "CHACK_EXEC_CWD"
_EXECUTOR_EXEC_CWD_ATTRIBUTE = "chack_exec_cwd"


def pin_exec_cwd(executor: object, config: object) -> None:
    """Record on ``executor`` the exec working directory of the config it was built from.

    Chack exports a run's ``tools.exec_cwd`` as the process-wide CHACK_EXEC_CWD,
    which every agent started later in the same process overwrites. A CLI run
    this executor starts afterwards (a resumed or repair turn) still belongs in
    its own directory, so CLI backends take it from here.
    """
    value = str(getattr(getattr(config, "tools", None), "exec_cwd", "") or "").strip()
    if value and not os.path.isabs(value):
        # export_env has just resolved it against the config's directory.
        value = str(os.environ.get(EXEC_CWD_ENV, "") or "").strip() or os.path.abspath(value)
    try:
        setattr(executor, _EXECUTOR_EXEC_CWD_ATTRIBUTE, value)
    except (AttributeError, TypeError):
        pass


def apply_pinned_exec_cwd(executor: object, env: dict) -> None:
    """Set CHACK_EXEC_CWD in a CLI child's ``env`` to the executor's own directory, when it has one."""
    value = str(getattr(executor, _EXECUTOR_EXEC_CWD_ATTRIBUTE, "") or "")
    if value:
        env[EXEC_CWD_ENV] = value


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
