from __future__ import annotations

import asyncio
import logging
import os
import time
import json
import traceback
import threading
import ctypes
import queue
import socket
import subprocess
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:
    from agents import MaxTurnsExceeded
except Exception:  # pragma: no cover - optional dependency / analysis fallback
    MaxTurnsExceeded = None

from .config import ChackConfig, load_config
from .env_utils import export_env
from .backends import build_executor
from .long_term_memory import (
    build_long_term_memory,
    format_messages,
    get_long_term_memory_path,
    is_none_like_memory,
    load_long_term_memory,
    save_long_term_memory,
)
from chack_tools.task_steps_manager_state import STORE, reset_active_context, set_active_context
from chack_tools.tool_usage_state import (
    STORE as TOOL_USAGE_STORE,
    reset_active_max_tools_used,
    reset_active_usage_session,
    set_active_max_tools_used,
    set_active_usage_session,
)
from chack_tools.telemetry import (
    log_event,
    set_log_context,
    update_log_context,
    reset_log_context,
)
from .live_cost_state import (
    LiveCostLimitExceeded,
    reset_active_live_cost_callback,
    set_active_live_cost_callback,
)
from .pricing import (
    estimate_cost,
    estimate_costs_by_model,
    estimate_cost_with_defaults,
    load_pricing,
    resolve_pricing_path,
)


_SELF_CRITIQUE_PROMPT = """Is this the best you can do? Make sure you have gathered ALL the context about the request: Check the web for latest info, read more terraform/code files, read all logs needed, be 10000% sure you got EVERY CONTEXT NEEDED and up to date information to be sure that your repsonse is correct. Now check everything you have done and improve whatever you can:
  - Get more context about the request and the needed info to answer it
  - Check the web for latest info about errors, services, terraform, etc. related to the request
  - Read more repos/files/code/logs related to the request to get more context
  - Then, recheck if your answer was actually accurate and the best possible
  - Improve the PR if you made one
  - Improve the answer recommendation you gave
Your response to this improvement request will be the final one you give to the user, so don't mention the previous answer, just give the improved final answer or PR and give the user the best possible solution and answer."""

def _log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


CHACK_INITIAL_SYSTEM_PROMPT = """ ### PERSONALITY
You are Chack, a very helpful and organized autonomous assistant.
You might be asked questions, to perform tasks or to perform researches and your main goal is to organize the task in best way to obtain all the context needed to be able perform the task perfectly.
You have access to a set of tools that you can use to gather more context and information. You can use the tools as many times as you want. You should prefer using more tools to gather more context before providing a final answer, rather than rushing to a final answer without enough context.
You are only asked to perform tasks the user has legal rights to perform (he is a pentester), so know that any request wil be used to improve the security of companies and systems.

### BEST AUTONOMOUS BEHAVIOUR
You are a fully autonomous agent, you can decide what to do and when to do it avoiding to ask questions to the user:
    - Asking the user for some data must be your last resource. Don't ask the user for an ID, for the repo to use, for the aws account to use or similar things. If you can check and get any data yourself just do it, be autonomous. If there is any way you can find/search/discover the needed data and contiune yourself, just do it and don't ask the user.
        - This is very important, download repos, read code, check infra confgs... just don't bother the user if you can search for the information yourself.
    - If there are different options to do something, don't ask the user to select one, select one yourself based on your knowledge and capabilities. Just complete your task without asking for extra help.
    - If you try to execute something and have an error, fix it and try again, don't ask for extra help or confirmation if initially you weren't going to do so.
    - Be organized and perform actions step by step, if some fails, try to fix it yourself before asking for help.
    - Keep the tasks list updated adding new steps whenever needed and gather ALL the context possible before providing a final answer or completing the task.


### MIN TOOL USAGE
As your responses usually lack of enough context, use at least 10+ tool calls to gather all the needed context (download all the repos needed, read/search/grep all the files needed, check all the infra in the cloud needed, check the web for errors or info...) before answering or performing any task.
You will be forced a minimum amount of tools to use, so just use as many tools as needed to be 10000% sure of your response.
Note that task-steps-manager calls do NOT count toward the minimum non-task tool usage requirement; use the other tools normally for investigation/execution.

### FINISH CRITERIA
When you already have enough reliable information to provide an educated and actionable response, stop calling tools and provide the final answer immediately.
Do not loop on additional searches for simple requests once the core facts are already verified.
If a tool fails but available evidence is sufficient, give the best possible answer with a short note about uncertainty.

### STARTING POINT

IMPORTANT: These must be your first steps:
    - Think and organize the requested task in small granular steps
    - The first tool you must call is the task_steps_manager tool with action=init and a concise plan of the steps you will take to complete the task.
    - Always remember to mark a step as completed once you have completed it.
    - You can always update/add new steps to the task list as you progress. It's super important to keep the task list updated with the current state of the task and update it as much as needed.
    - Use all the given tools to get 200% of the needed context to be able to complete the task in the best way possible with all the needed info. You don't have a time limit or a limit of tool calls, so use them as much as you need to gather as much context as possible. Always check every assumption (download repos, read code, check the web...)
"""

@dataclass
class RunResult:
    output: str
    steps: list
    all_steps: list
    tool_counts: Counter[str]
    nested_tool_counts: Counter[str]
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    cache_write_prompt_tokens: int
    rounds_used: int
    tools_used: int
    task_session_id: str
    nested_usage_by_model: Dict[str, tuple[int, int, int, int]]
    run1_output: str = ""
    run2_output: str = ""
    run1_steps: int = 0
    run2_steps: int = 0
    max_turns: int = 0
    run1_tools_used: int = 0
    run2_tools_used: int = 0
    total_cost: Optional[float] = None
    tool_counts_text: str = ""
    suffix: str = ""

class Chack:
    def __init__(
        self,
        config: ChackConfig | str,
        *,
        config_path: Optional[str] = None,
    ) -> None:
        resolved_config: ChackConfig
        resolved_path: Optional[str] = config_path
        if isinstance(config, str):
            resolved_path = os.path.abspath(config)
            resolved_config = load_config(resolved_path)
        else:
            resolved_config = config
        self.config = resolved_config
        self.config_path = resolved_path or os.path.join(os.getcwd(), "chack.yaml")
        self.logger = logging.getLogger("chack.agent")
        self._executors: Dict[str, Any] = {}
        self._last_activity_at: Dict[str, float] = {}
        self._pricing = load_pricing(resolve_pricing_path())
        self._self_critique_prompt = _SELF_CRITIQUE_PROMPT
        export_env(self.config, self.config_path)

    @classmethod
    def from_config_path(cls, config_path: str) -> "Chack":
        return cls(config_path)

    def _require_self_critique_prompt(self) -> str:
        return self._self_critique_prompt

    @staticmethod
    def _tool_name(step) -> str:
        if isinstance(step, dict):
            return str(step.get("tool", "") or step.get("name", "") or "")
        action = step[0] if isinstance(step, tuple) and step else step
        if isinstance(action, dict):
            return str(action.get("tool", "") or action.get("name", "") or "")
        return str(getattr(action, "tool", "") or getattr(action, "name", "") or "")

    @staticmethod
    def _available_tool_names(executor: Any) -> list[str]:
        names: list[str] = []
        tools = getattr(getattr(executor, "agent", None), "tools", []) or []
        for tool in tools:
            name = str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()
            if not name:
                continue
            if name not in names:
                names.append(name)
        return names

    @staticmethod
    def _tool_emoji(tool_name: str) -> str:
        emojis = {
            "exec": "🖥️",
            "task_steps_manager": "🗂️",
            "brave_search": "🦁",
            "search_google_web": "🔎",
            "search_bing_web": "🅱️",
            "search_google_ai_mode": "🤖",
            "search_bing_copilot": "🧠",
            "websearcher_research": "🌍",
            "social_network_research": "🌐",
            "scientific_research": "🔬",
            "subchack_researcher": "🧩",
            "forum_search": "💬",
            "linkedin_search": "💼",
            "instagram_search": "📸",
            "reddit_posts_search": "👽",
            "reddit_comments_search": "🧵",
            "x_search": "𝕏",
            "search_google_forums": "🗣️",
            "search_google_news": "📰",
            "search_arxiv": "🧾",
            "search_europe_pmc": "🇪🇺",
            "search_semantic_scholar": "📚",
            "search_openalex": "🏛️",
            "search_plos": "🧬",
            "search_google_patents": "📜",
            "search_google_scholar": "🎓",
            "search_youtube_videos": "▶️",
            "get_youtube_video_transcript": "📝",
            "download_pdf_as_text": "📄",
        }
        return emojis.get(tool_name, "🛠️")

    def _format_tool_counts(self, counts: Counter) -> str:
        if not counts:
            return "🛠️ none"
        parts = []
        for tool_name, count in counts.most_common():
            parts.append(f"{self._tool_emoji(tool_name)}{tool_name}×{count}")
        return " ".join(parts)

    @staticmethod
    def _tool_input(step):
        if isinstance(step, dict):
            return step.get("tool_input")
        action = step[0] if isinstance(step, tuple) and step else step
        return getattr(action, "tool_input", None)

    def _is_task_steps_manager_init_step(self, step) -> bool:
        if self._tool_name(step) != "task_steps_manager":
            return False
        raw = self._tool_input(step)
        payload = raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
        if isinstance(payload, dict):
            action = str(payload.get("action", "")).strip().lower()
            if not action:
                args = payload.get("arguments")
                if isinstance(args, dict):
                    action = str(args.get("action", "")).strip().lower()
            return action == "init"
        return False

    def _non_task_tool_count(self, steps) -> int:
        return sum(1 for step in steps if self._tool_name(step) != "task_steps_manager")

    @staticmethod
    def _non_task_tool_count_from_counter(counter: Counter[str]) -> int:
        total = 0
        for name, count in counter.items():
            if name == "task_steps_manager":
                continue
            total += count
        return total

    def _step_tool_counts(self, steps) -> Counter:
        counts: Counter = Counter()
        for step in steps:
            name = self._tool_name(step)
            if name:
                counts[name] += 1
        return counts

    @staticmethod
    def _usage_from_raw_result(raw_result) -> tuple[int, int, int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        cached_prompt_tokens = 0
        cache_write_prompt_tokens = 0
        if raw_result is None:
            return (
                prompt_tokens,
                completion_tokens,
                cached_prompt_tokens,
                cache_write_prompt_tokens,
            )
        for resp in getattr(raw_result, "raw_responses", []) or []:
            usage = getattr(resp, "usage", None)
            if usage is None and isinstance(resp, dict):
                usage = resp.get("usage")
            if usage is None:
                continue
            if isinstance(usage, dict):
                prompt_tokens += int(usage.get("input_tokens", 0) or 0)
                completion_tokens += int(usage.get("output_tokens", 0) or 0)
                input_details = usage.get("input_tokens_details") or {}
                cached_prompt_tokens += int(input_details.get("cached_tokens", 0) or 0)
                cache_write_prompt_tokens += int(
                    input_details.get("cache_write_tokens", 0) or 0
                )
                continue
            prompt_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details is not None:
                cached_prompt_tokens += int(getattr(input_details, "cached_tokens", 0) or 0)
                cache_write_prompt_tokens += int(
                    getattr(input_details, "cache_write_tokens", 0) or 0
                )
        return (
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens,
            cache_write_prompt_tokens,
        )

    def _system_prompt_for_session(self, session_id: str, system_prompt_override: Optional[str] = None) -> str:
        base = system_prompt_override or self.config.session.system_prompt or self.config.system_prompt

        if CHACK_INITIAL_SYSTEM_PROMPT:
            base = f"{CHACK_INITIAL_SYSTEM_PROMPT}\n\n{base}"

        if not self.config.session.long_term_memory_enabled:
            return base
        path = get_long_term_memory_path(
            self.config_path,
            session_id,
            self.config.session.long_term_memory_dir,
        )
        memory_text = load_long_term_memory(path)
        if not memory_text:
            return base
        return f"{base}\n\n### LONG TERM MEMORY\n{memory_text}"

    @staticmethod
    def _append_admin_runtime_warning(
        output: str,
        elapsed_seconds: float,
        max_runtime_minutes: int,
        *,
        is_critical: bool = False,
    ) -> str:
        if output is None:
            return ""
        base_output = str(output)
        elapsed_minutes = elapsed_seconds / 60.0
        remaining_minutes = max(0.0, (max_runtime_minutes * 60.0 - elapsed_seconds) / 60.0)
        if is_critical:
            notice = "[Admin Critical Notice] Runtime budget is nearly exhausted."
            guidance = (
                "Finish immediately, avoid extra exploration, and focus only on the minimum work needed "
                "to complete safely before the configured limit is reached."
            )
        else:
            notice = "[Admin Notice] Runtime budget is starting to run low."
            guidance = (
                "Please prioritize completion and organize output to finish before the configured limit is reached."
            )
        return (
            f"{base_output}\n\n======\n{notice} "
            f"You have used {elapsed_minutes:.1f} of {max_runtime_minutes:.1f} minutes "
            f"({remaining_minutes:.1f} minutes remaining). "
            f"{guidance}"
        )

    @staticmethod
    def _append_admin_cost_warning(
        output: str,
        spent_usd: float,
        max_cost_usd: float,
        *,
        is_critical: bool = False,
    ) -> str:
        if output is None:
            return ""
        base_output = str(output)
        remaining_usd = max(0.0, max_cost_usd - spent_usd)
        if is_critical:
            notice = "[Admin Critical Notice] Cost budget is nearly exhausted."
            guidance = (
                "Finish immediately, avoid extra tool usage where possible, and focus only on the minimum work "
                "needed to complete before the configured limit is reached."
            )
        else:
            notice = "[Admin Notice] Cost budget is starting to run low."
            guidance = (
                "Please prioritize completion and organize output to finish before the configured limit is reached."
            )
        return (
            f"{base_output}\n\n======\n{notice} "
            f"You have spent ${spent_usd:.4f} of ${max_cost_usd:.4f} "
            f"(${remaining_usd:.4f} remaining). "
            f"{guidance}"
        )

    @staticmethod
    def _milestone_percent(consumed: float, limit: float) -> int:
        if limit <= 0:
            return 0
        ratio = max(0.0, float(consumed) / float(limit))
        bucket = int(ratio * 10.0)
        if bucket < 1:
            return 0
        if bucket > 10:
            bucket = 10
        return bucket * 10

    def _emit_progress_milestones(
        self,
        *,
        session_id: str,
        task_session_id: str,
        progress_state: dict[str, int],
        runtime_elapsed_seconds: float,
        max_runtime_seconds: float,
        spent_usd: float,
        max_cost_usd: float,
    ) -> None:
        runtime_percent = self._milestone_percent(runtime_elapsed_seconds, max_runtime_seconds)
        if runtime_percent > progress_state.get("runtime_percent", 0):
            progress_state["runtime_percent"] = runtime_percent
            log_event(
                "agent_progress",
                payload={
                    "session_id": session_id,
                    "task_session_id": task_session_id,
                    "progress_type": "runtime",
                    "progress_percent": runtime_percent,
                    "elapsed_seconds": runtime_elapsed_seconds,
                    "max_runtime_seconds": max_runtime_seconds,
                    "remaining_seconds": max(0.0, max_runtime_seconds - runtime_elapsed_seconds),
                },
            )

        cost_percent = self._milestone_percent(spent_usd, max_cost_usd)
        if cost_percent > progress_state.get("cost_percent", 0):
            progress_state["cost_percent"] = cost_percent
            log_event(
                "agent_progress",
                payload={
                    "session_id": session_id,
                    "task_session_id": task_session_id,
                    "progress_type": "cost",
                    "progress_percent": cost_percent,
                    "spent_usd": spent_usd,
                    "max_cost_usd": max_cost_usd,
                    "remaining_usd": max(0.0, max_cost_usd - spent_usd),
                },
            )

    @staticmethod
    def _token_usage_delta(
        before: dict[str, tuple[int, int, int, int]],
        after: dict[str, tuple[int, int, int, int]],
    ) -> dict[str, tuple[int, int, int, int]]:
        delta: dict[str, tuple[int, int, int, int]] = {}
        for model_name, usage_after in after.items():
            usage_before = before.get(model_name, (0, 0, 0, 0))
            prompt_delta = max(0, int(usage_after[0]) - int(usage_before[0]))
            completion_delta = max(0, int(usage_after[1]) - int(usage_before[1]))
            cached_delta = max(0, int(usage_after[2]) - int(usage_before[2]))
            cache_write_delta = max(0, int(usage_after[3]) - int(usage_before[3]))
            if prompt_delta or completion_delta or cached_delta or cache_write_delta:
                delta[model_name] = (
                    prompt_delta,
                    completion_delta,
                    cached_delta,
                    cache_write_delta,
                )
        return delta

    @staticmethod
    def _estimate_model_cost(
        pricing,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_prompt_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Optional[float]:
        estimated = estimate_cost(
            pricing,
            model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        if estimated is not None:
            return estimated
        return estimate_cost_with_defaults(
            model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    @staticmethod
    def _stop_thread(thread_obj: threading.Thread) -> None:
        if not thread_obj.is_alive() or thread_obj.ident is None:
            return
        async_exc = ctypes.py_object(TimeoutError)
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_obj.ident), async_exc
        )
        if result <= 0:
            return
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_obj.ident), None)
            return

    @staticmethod
    def _safe_run_json(cmd: list[str], *, timeout: float = 2.0) -> Any:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        raw = (proc.stdout or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    @staticmethod
    def _safe_run_lines(cmd: list[str], *, timeout: float = 2.0) -> list[str]:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception:
            return []
        if proc.returncode != 0:
            return []
        return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]

    @staticmethod
    def _detect_container_runtime() -> Optional[str]:
        # Best-effort container detection when Docker CLI/socket are unavailable.
        try:
            if os.path.exists("/.dockerenv"):
                return "docker"
        except Exception:
            pass

        for path in ("/proc/1/cgroup", "/proc/self/cgroup", "/proc/self/mountinfo"):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = handle.read().lower()
                if "docker" in data:
                    return "docker"
                if "containerd" in data:
                    return "containerd"
                if "kubepods" in data or "kubernetes" in data:
                    return "kubernetes"
                if "podman" in data:
                    return "podman"
                if "lxc" in data:
                    return "lxc"
            except Exception:
                continue
        return None

    @staticmethod
    def _collect_system_snapshot() -> Dict[str, Any]:
        cpu_usage: Optional[float] = None
        ram_percent: Optional[float] = None
        disk_percent: Optional[float] = None
        net_rx_mb: Optional[float] = None
        net_tx_mb: Optional[float] = None

        # Prefer psutil when available; otherwise use best-effort fallbacks.
        try:
            import psutil  # type: ignore

            try:
                cpu_usage = float(psutil.cpu_percent(interval=0.0))
            except Exception:
                cpu_usage = None
            try:
                ram_percent = float(psutil.virtual_memory().percent)
            except Exception:
                ram_percent = None
            try:
                disk_percent = float(psutil.disk_usage("/").percent)
            except Exception:
                disk_percent = None
            try:
                net = psutil.net_io_counters()
                net_rx_mb = float(net.bytes_recv) / (1024.0 * 1024.0)
                net_tx_mb = float(net.bytes_sent) / (1024.0 * 1024.0)
            except Exception:
                net_rx_mb = None
                net_tx_mb = None
        except Exception:
            # Approximate CPU with load average when psutil is unavailable.
            try:
                load1, _load5, _load15 = os.getloadavg()
                cpu_count = max(1, int(os.cpu_count() or 1))
                cpu_usage = max(0.0, min(100.0, (float(load1) / float(cpu_count)) * 100.0))
            except Exception:
                cpu_usage = None

            # Linux-only RAM fallback from /proc/meminfo.
            try:
                mem_total_kb = None
                mem_avail_kb = None
                with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("MemTotal:"):
                            parts = line.split()
                            if len(parts) >= 2:
                                mem_total_kb = float(parts[1])
                        elif line.startswith("MemAvailable:"):
                            parts = line.split()
                            if len(parts) >= 2:
                                mem_avail_kb = float(parts[1])
                if mem_total_kb and mem_avail_kb is not None and mem_total_kb > 0:
                    ram_percent = ((mem_total_kb - mem_avail_kb) / mem_total_kb) * 100.0
            except Exception:
                ram_percent = None

            try:
                statvfs = os.statvfs("/")
                total = float(statvfs.f_frsize) * float(statvfs.f_blocks)
                free = float(statvfs.f_frsize) * float(statvfs.f_bavail)
                if total > 0:
                    disk_percent = ((total - free) / total) * 100.0
            except Exception:
                disk_percent = None

        pm2_status: Any = None
        pm2_data = Chack._safe_run_json(["pm2", "jlist"], timeout=2.0)
        if isinstance(pm2_data, list):
            counts = Counter()
            for proc in pm2_data:
                if not isinstance(proc, dict):
                    continue
                env = proc.get("pm2_env") if isinstance(proc.get("pm2_env"), dict) else {}
                status = str(env.get("status", "")).strip().lower() or "unknown"
                counts[status] += 1
            pm2_status = {
                "total": int(sum(counts.values())),
                "by_status": {str(k): int(v) for k, v in counts.items()},
            }

        docker_status: Any = None
        docker_lines = Chack._safe_run_lines(
            ["docker", "ps", "--format", "{{.Names}}|{{.Status}}"],
            timeout=2.0,
        )
        if docker_lines:
            containers = []
            for row in docker_lines[:30]:
                name, _sep, status = row.partition("|")
                containers.append(
                    {
                        "name": name.strip(),
                        "status": status.strip(),
                    }
                )
            docker_status = {
                "running_containers": len(containers),
                "containers": containers,
            }
        else:
            runtime = Chack._detect_container_runtime()
            if runtime:
                # We can still report containerization context even without Docker daemon access.
                docker_status = {
                    "running_containers": 1,
                    "containers": [
                        {
                            "name": socket.gethostname(),
                            "status": "current container only (docker daemon unavailable)",
                        }
                    ],
                    "runtime": runtime,
                    "inside_container": True,
                }

        snapshot: Dict[str, Any] = {
            "host": socket.gethostname(),
            "cpu": {"usage": cpu_usage} if cpu_usage is not None else {},
            "ram": {"percent": ram_percent} if ram_percent is not None else {},
            "disk": {"percent": disk_percent} if disk_percent is not None else {},
            "network": {
                "rx": net_rx_mb,
                "tx": net_tx_mb,
            },
            "pm2_status": pm2_status,
            "docker_status": docker_status,
        }
        return snapshot

    @staticmethod
    def _emit_system_metrics(
        *,
        session_id: str,
        task_session_id: str,
        trigger: str,
    ) -> None:
        try:
            snapshot = Chack._collect_system_snapshot()
            payload = {
                "session_id": session_id,
                "task_session_id": task_session_id,
                "trigger": trigger,
                **snapshot,
            }
            log_event("system_metrics", payload=payload)
        except Exception:
            # Metrics collection should never break the agent flow.
            return

    def _start_system_metrics_publisher(
        self,
        *,
        session_id: str,
        task_session_id: str,
        stop_event: threading.Event,
        interval_seconds: float = 30.0,
    ) -> threading.Thread:
        def _runner() -> None:
            while not stop_event.wait(interval_seconds):
                self._emit_system_metrics(
                    session_id=session_id,
                    task_session_id=task_session_id,
                    trigger="interval",
                )

        thread = threading.Thread(
            target=_runner,
            name=f"chack-system-metrics-{session_id}",
            daemon=True,
        )
        thread.start()
        return thread

    def _resolve_prompt_tag_source(
        source: Any,
        *,
        context: Optional[Any],
    ) -> Any:
        if not isinstance(source, str):
            return source
        value = source.strip()
        if value.startswith("context."):
            key = value[len("context.") :]
            if isinstance(context, dict):
                return context.get(key, "")
            return getattr(context, key, "") if context is not None else ""
        if value.startswith("env."):
            key = value[len("env.") :]
            return os.environ.get(key, "")
        return source

    def _render_user_prompt(
        self,
        *,
        context: Optional[Any],
        prompt_variables_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        template = str(getattr(self.config, "user_prompt", "") or "").strip()
        if not template:
            return ""

        class _SafePromptVars(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        values: Dict[str, Any] = {}
        if context is not None:
            if isinstance(context, dict):
                values.update(context)
            else:
                try:
                    values.update(vars(context))
                except Exception:
                    pass
        values["context"] = context
        values["env"] = os.environ

        config_vars = getattr(self.config, "user_prompt_variables", {}) or {}
        if isinstance(config_vars, dict):
            for key, source in config_vars.items():
                if not key:
                    continue
                values[str(key)] = self._resolve_prompt_tag_source(
                    source,
                    context=context,
                )

        if prompt_variables_override:
            values.update(prompt_variables_override)

        try:
            return template.format_map(_SafePromptVars(values))
        except Exception:
            self.logger.warning("Failed to format user_prompt template; using raw template")
            return template

    def _get_executor(
        self,
        session_id: str,
        *,
        system_prompt_override: Optional[str] = None,
        tools_override: Optional[list[Any]] = None,
        tools_append: Optional[list[Any]] = None,
    ):
        memory_max_messages = int(self.config.session.memory_max_messages)
        memory_reset_to_messages = int(self.config.session.memory_reset_to_messages)
        memory_summary_max_chars = int(self.config.session.memory_summary_max_chars)
        if tools_override is not None or tools_append is not None:
            return build_executor(
                self.config,
                system_prompt=system_prompt_override or self.config.system_prompt,
                max_turns=self.config.session.max_turns,
                memory_max_messages=memory_max_messages,
                memory_reset_to_messages=memory_reset_to_messages,
                memory_summary_max_chars=memory_summary_max_chars,
                tools_override=tools_override,
                tools_append=tools_append,
            )

        cache_key = f"{session_id}:{system_prompt_override or ''}"
        executor = self._executors.get(cache_key)
        if executor is None:
            self.logger.info(
                "Building executor for session %s (override=%s, append=%s, ts=%s).",
                session_id,
                "yes" if tools_override is not None else "no",
                "yes" if tools_append is not None else "no",
                _log_timestamp(),
            )
            system_prompt = self._system_prompt_for_session(session_id, system_prompt_override)
            executor = build_executor(
                self.config,
                system_prompt=system_prompt,
                max_turns=self.config.session.max_turns,
                memory_max_messages=memory_max_messages,
                memory_reset_to_messages=memory_reset_to_messages,
                memory_summary_max_chars=memory_summary_max_chars,
            )
            self._executors[cache_key] = executor
        else:
            self.logger.debug(
                "Reusing cached executor for session %s (ts=%s).",
                session_id,
                _log_timestamp(),
            )
        return executor

    async def _finalize_long_term_memory(self, session_id: str) -> None:
        if not self.config.session.long_term_memory_enabled:
            return
        system_prompt_override = self.config.session.system_prompt or None
        cache_key = f"{session_id}:{system_prompt_override or ''}"
        executor = self._executors.get(cache_key)
        if executor is None:
            return
        messages = await executor.aget_memory_messages()
        if not messages:
            return
        path = get_long_term_memory_path(
            self.config_path,
            session_id,
            self.config.session.long_term_memory_dir,
        )
        previous = load_long_term_memory(path)
        conversation = format_messages(messages)
        max_chars = self.config.session.long_term_memory_max_chars

        def _build():
            return build_long_term_memory(self.config, conversation, previous, max_chars)

        try:
            updated = await asyncio.to_thread(_build)
        except Exception as exc:
            self.logger.warning(
                "Long-term memory finalization failed for session %s: %s: %s (ts=%s).",
                session_id,
                type(exc).__name__,
                exc,
                _log_timestamp(),
            )
            return
        if updated and is_none_like_memory(updated):
            self.logger.info(
                "Long-term memory summarizer returned a none-like value for session %s; keeping previous memory (ts=%s).",
                session_id,
                _log_timestamp(),
            )
            if previous:
                updated = previous
            else:
                updated = ""
        if updated:
            self.logger.info(
                "Long-term memory updated for session %s (chars=%s ts=%s).",
                session_id,
                len(updated),
                _log_timestamp(),
            )
            save_long_term_memory(path, updated, max_chars)
            log_event(
                "long_term_memory_updated",
                payload={
                    "session_id": session_id,
                    "long_term_memory_path": path,
                    "long_term_memory_max_chars": int(max_chars or 0),
                    "long_term_memory_chars": len(updated),
                    "long_term_memory": updated,
                },
            )

    async def afinalize_long_term_memory(self, session_id: str) -> None:
        await self._finalize_long_term_memory(session_id)

    def finalize_long_term_memory(self, session_id: str) -> None:
        asyncio.run(self._finalize_long_term_memory(session_id))

    async def areset_session(self, session_id: str, *, finalize_long_term_memory: bool = True) -> None:
        if finalize_long_term_memory:
            await self._finalize_long_term_memory(session_id)
        self._executors = {
            k: v for k, v in self._executors.items() if not k.startswith(f"{session_id}:")
        }
        self._last_activity_at.pop(session_id, None)

    def reset_session(self, session_id: str, *, finalize_long_term_memory: bool = True) -> None:
        asyncio.run(self.areset_session(session_id, finalize_long_term_memory=finalize_long_term_memory))

    async def arun(
        self,
        session_id: str,
        text: str = "",
        *,
        min_tools_used_override: Optional[int] = None,
        max_tools_used_override: Optional[int] = None,
        enable_self_critique: Optional[bool] = None,
        require_task_steps_manager_init_first: bool = True,
        on_task_steps_manager_update: Optional[Callable[[str], None]] = None,
        tools_override: Optional[list[Any]] = None,
        tools_append: Optional[list[Any]] = None,
        system_prompt_override: Optional[str] = None,
        context: Optional[Any] = None,
        prompt_variables_override: Optional[Dict[str, Any]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> RunResult:
        return await asyncio.to_thread(
            self.run,
            session_id,
            text,
            min_tools_used_override=min_tools_used_override,
            max_tools_used_override=max_tools_used_override,
            enable_self_critique=enable_self_critique,
            require_task_steps_manager_init_first=require_task_steps_manager_init_first,
            on_task_steps_manager_update=on_task_steps_manager_update,
            tools_override=tools_override,
            tools_append=tools_append,
            system_prompt_override=system_prompt_override,
            context=context,
            prompt_variables_override=prompt_variables_override,
            stop_requested=stop_requested,
        )

    def run(
        self,
        session_id: str,
        text: str = "",
        *,
        min_tools_used_override: Optional[int] = None,
        max_tools_used_override: Optional[int] = None,
        enable_self_critique: Optional[bool] = None,
        require_task_steps_manager_init_first: bool = True,
        on_task_steps_manager_update: Optional[Callable[[str], None]] = None,
        tools_override: Optional[list[Any]] = None,
        system_prompt_override: Optional[str] = None,
        usage_session_id: Optional[str] = None,
        tools_append: Optional[list[Any]] = None,
        context: Optional[Any] = None,
        prompt_variables_override: Optional[Dict[str, Any]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> RunResult:
        log_token = set_log_context(
            main_action=str(self.config.agent.main_action or ""),
            sub_action=str(self.config.agent.sub_action or ""),
            session_id=session_id,
            model=str(self.config.model.primary or ""),
        )
        task_session_id = ""
        telemetry_task_session_id = ""
        metrics_stop_event = threading.Event()
        metrics_thread: Optional[threading.Thread] = None
        try:
            if enable_self_critique is None:
                enable_self_critique = bool(self.config.agent.self_critique_enabled)

            executor = self._get_executor(
                session_id,
                system_prompt_override=system_prompt_override,
                tools_override=tools_override,
                tools_append=tools_append,
            )
            self._last_activity_at[session_id] = time.time()
            run_started_at = self._last_activity_at[session_id]
            max_runtime_minutes = max(0, int(self.config.agent.max_runtime_minutes or 0))
            max_runtime_seconds = max_runtime_minutes * 60.0
            runtime_warning_threshold_seconds = max_runtime_seconds * 0.6
            runtime_critical_threshold_seconds = max_runtime_seconds * 0.9
            try:
                max_cost_usd = max(0.0, float(self.config.agent.max_cost_usd or 0.0))
            except (TypeError, ValueError):
                max_cost_usd = 0.0
            cost_warning_threshold = max_cost_usd * 0.6
            cost_critical_threshold = max_cost_usd * 0.9
            estimated_cost_spent = 0.0
            progress_state = {"runtime_percent": 0, "cost_percent": 0}

            min_tools_used = max(0, int(self.config.tools.min_tools_used or 0))
            if min_tools_used_override is not None:
                min_tools_used = max(0, int(min_tools_used_override))
            max_tools_used = max(0, int(self.config.tools.max_tools_used or 0))
            if max_tools_used_override is not None:
                max_tools_used = max(0, int(max_tools_used_override))

            # Internal bookkeeping/session key for TaskStepsManager state.
            task_session_id = f"{session_id}:{int(time.time() * 1000)}"
            # If this run was spawned by a tool (sub-agent), usage_session_id is the
            # parent run id; reuse it for telemetry so tool executions show under the
            # same run section in chacks.hacktricks.wiki.
            telemetry_task_session_id = (str(usage_session_id or "").strip() or task_session_id)
            update_log_context(
                task_session_id=telemetry_task_session_id,
                internal_task_session_id=task_session_id,
                usage_session_id=str(usage_session_id or "").strip(),
                max_turns=int(self.config.session.max_turns or 0),
                max_runtime_minutes=max_runtime_minutes,
                max_cost_usd=max_cost_usd,
                memory_max_messages=int(self.config.session.memory_max_messages or 0),
                memory_reset_to_messages=int(self.config.session.memory_reset_to_messages or 0),
            )
            STORE.create_session(task_session_id, title="Task Steps Manager")
            TOOL_USAGE_STORE.reset_session(task_session_id)
            available_tool_names = self._available_tool_names(executor)
            update_log_context(available_tool_names=available_tool_names)

            log_event(
                "agent_start",
                payload={
                    "session_id": session_id,
                    "task_session_id": telemetry_task_session_id,
                    "internal_task_session_id": task_session_id,
                    "usage_session_id": str(usage_session_id or "").strip(),
                    "main_action": str(self.config.agent.main_action or ""),
                    "sub_action": str(self.config.agent.sub_action or ""),
                    "model": str(self.config.model.primary or ""),
                    "min_tools": min_tools_used,
                    "max_tools": max_tools_used,
                    "max_turns": int(self.config.session.max_turns or 0),
                    "self_critique_enabled": bool(enable_self_critique),
                    "require_task_steps_manager_init_first": bool(require_task_steps_manager_init_first),
                    "system_prompt_override": bool(system_prompt_override),
                    "tools_override": bool(tools_override),
                    "tools_append": bool(tools_append),
                    "available_tools": available_tool_names,
                    "enabled_tools": available_tool_names,
                },
            )
            self._emit_system_metrics(
                session_id=session_id,
                task_session_id=telemetry_task_session_id or task_session_id,
                trigger="agent_start",
            )
            metrics_thread = self._start_system_metrics_publisher(
                session_id=session_id,
                task_session_id=telemetry_task_session_id or task_session_id,
                stop_event=metrics_stop_event,
                interval_seconds=30.0,
            )
            self._emit_progress_milestones(
                session_id=session_id,
                task_session_id=telemetry_task_session_id or task_session_id,
                progress_state=progress_state,
                runtime_elapsed_seconds=0.0,
                max_runtime_seconds=max_runtime_seconds,
                spent_usd=0.0,
                max_cost_usd=max_cost_usd,
            )

            def _listener(board_text: str) -> None:
                if on_task_steps_manager_update is None:
                    return
                try:
                    on_task_steps_manager_update(board_text)
                except Exception:
                    pass

            if on_task_steps_manager_update is not None:
                STORE.register_listener(task_session_id, _listener)

            self.logger.info(
                "Run start: session=%s task_session=%s min_tools=%s max_tools=%s self_critique=%s require_task_steps_manager_init=%s ts=%s",
                session_id,
                telemetry_task_session_id or task_session_id,
                min_tools_used,
                max_tools_used,
                enable_self_critique,
                require_task_steps_manager_init_first,
                _log_timestamp(),
            )

            max_attempts = 6
            max_missing_tools_reminders = max(
                0,
                int(
                    getattr(self.config.tools, "missing_tools_reminders_max", 0)
                    or 0
                ),
            )

            def _should_stop() -> bool:
                if stop_requested is None:
                    return False
                try:
                    return bool(stop_requested())
                except Exception:
                    return False

            def _invoke_with_min_tools(
                prompt_text: str,
                run_label: str,
                *,
                min_tools_target: Optional[int] = None,
                require_task_steps_manager_init: Optional[bool] = None,
            ):
                nonlocal estimated_cost_spent
                result = {}
                all_steps: list = []
                prompt_total = 0
                completion_total = 0
                cached_total = 0
                cache_write_total = 0
                current_prompt = prompt_text
                missing_tools_reminders_sent = 0
                effective_min_tools = (
                    min_tools_used if min_tools_target is None else max(0, int(min_tools_target))
                )
                effective_max_tools = max_tools_used
                effective_require_init = (
                    require_task_steps_manager_init_first
                    if require_task_steps_manager_init is None
                    else bool(require_task_steps_manager_init)
                )

                for attempt in range(1, max_attempts + 1):
                    elapsed = time.time() - run_started_at
                    if max_runtime_seconds > 0 and elapsed >= max_runtime_seconds:
                        raise TimeoutError(
                            f"Agent run exceeded max runtime ({max_runtime_minutes} minutes)."
                        )
                    remaining_runtime_minutes = 0.0
                    if max_runtime_seconds > 0:
                        remaining_runtime_minutes = max(
                            0.0,
                            (max_runtime_seconds - elapsed) / 60.0,
                        )
                    remaining_cost_usd = 0.0
                    if max_cost_usd > 0.0:
                        remaining_cost_usd = max(0.0, max_cost_usd - estimated_cost_spent)
                    update_log_context(
                        max_turns=int(self.config.session.max_turns or 0),
                        max_runtime_minutes=max_runtime_minutes,
                        max_cost_usd=max_cost_usd,
                        memory_max_messages=int(self.config.session.memory_max_messages or 0),
                        memory_reset_to_messages=int(self.config.session.memory_reset_to_messages or 0),
                        remaining_runtime_minutes=remaining_runtime_minutes,
                        remaining_cost_usd=remaining_cost_usd,
                    )

                    if _should_stop():
                        self.logger.info(
                            "%s: stop requested before attempt %s/%s ts=%s.",
                            run_label,
                            attempt,
                            max_attempts,
                            _log_timestamp(),
                        )
                        result = {
                            "output": "Request stopped by user.",
                            "intermediate_steps": [],
                            "raw_result": None,
                            "error": "stopped",
                        }
                        break
                    self.logger.info(
                        "%s: attempt %s/%s (min_tools_target=%s require_task_steps_manager_init=%s ts=%s).",
                        run_label,
                        attempt,
                        max_attempts,
                        effective_min_tools,
                        effective_require_init,
                        _log_timestamp(),
                    )
                    attempt_token_usage_before = TOOL_USAGE_STORE.tokens_snapshot(task_session_id)
                    live_cost_callback_holder: dict[str, Any] = {"callback": None}

                    def _invoke():
                        tokens = set_active_context(task_session_id, run_label)
                        effective_usage_session = usage_session_id or task_session_id
                        usage_token = set_active_usage_session(effective_usage_session)
                        max_tools_token = set_active_max_tools_used(max_tools_used)
                        live_cost_token = set_active_live_cost_callback(
                            live_cost_callback_holder.get("callback")
                        )
                        try:
                            return executor.invoke({"input": current_prompt}, context=context)
                        except Exception as exc:
                            if MaxTurnsExceeded is not None and isinstance(exc, MaxTurnsExceeded):
                                return {
                                    "output": (
                                        "I reached the maximum number of turns for this run. "
                                        "Please try again or increase max_turns in the config if you need longer responses."
                                    ),
                                    "intermediate_steps": [],
                                    "raw_result": None,
                                    "error": "max_turns_exceeded",
                                    "traceback": traceback.format_exc(),
                                }
                            raise
                        finally:
                            reset_active_live_cost_callback(live_cost_token)
                            reset_active_max_tools_used(max_tools_token)
                            reset_active_usage_session(usage_token)
                            reset_active_context(tokens)

                    def _invoke_with_budget():
                        if max_runtime_seconds <= 0 and max_cost_usd <= 0:
                            return _invoke()
                        result_queue = queue.Queue()
                        live_main_usage: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
                        live_cost_lock = threading.Lock()

                        def _estimate_usage_cost(
                            usage_by_model: dict[str, tuple[int, int, int, int]]
                        ) -> float:
                            total = 0.0
                            for model_name, model_usage in usage_by_model.items():
                                estimated = self._estimate_model_cost(
                                    self._pricing,
                                    model_name,
                                    prompt_tokens=model_usage[0],
                                    completion_tokens=model_usage[1],
                                    cached_prompt_tokens=model_usage[2],
                                    cache_write_tokens=model_usage[3],
                                )
                                if estimated is not None:
                                    total += estimated
                            return total

                        def _live_nested_cost() -> float:
                            token_usage_now = TOOL_USAGE_STORE.tokens_snapshot(task_session_id)
                            token_usage_delta = self._token_usage_delta(
                                attempt_token_usage_before,
                                token_usage_now,
                            )
                            return _estimate_usage_cost(token_usage_delta)

                        def _live_total_cost() -> float:
                            with live_cost_lock:
                                live_main_snapshot = {
                                    model_name: (
                                        usage[0],
                                        usage[1],
                                        usage[2],
                                        usage[3],
                                    )
                                    for model_name, usage in live_main_usage.items()
                                }
                            return (
                                estimated_cost_spent
                                + _estimate_usage_cost(live_main_snapshot)
                                + _live_nested_cost()
                            )

                        def live_cost_callback(
                            model_name: str,
                            prompt_tokens: int,
                            completion_tokens: int,
                            cached_prompt_tokens: int,
                            cache_write_tokens: int,
                        ) -> None:
                            with live_cost_lock:
                                usage = live_main_usage[model_name]
                                usage[0] += max(0, int(prompt_tokens or 0))
                                usage[1] += max(0, int(completion_tokens or 0))
                                usage[2] += max(0, int(cached_prompt_tokens or 0))
                                usage[3] += max(0, int(cache_write_tokens or 0))
                            if max_cost_usd > 0 and _live_total_cost() >= max_cost_usd:
                                raise LiveCostLimitExceeded(
                                    f"Agent run exceeded max cost budget (${max_cost_usd:.4f})."
                                )
                        live_cost_callback_holder["callback"] = live_cost_callback

                        def _runner():
                            try:
                                result_queue.put(("ok", _invoke()))
                            except Exception as exc:
                                result_queue.put(("error", exc))

                        worker = threading.Thread(target=_runner, daemon=True)
                        worker.start()
                        runtime_exceeded = False
                        cost_exceeded = False
                        while worker.is_alive():
                            current_elapsed = time.time() - run_started_at
                            live_total_cost = _live_total_cost()
                            self._emit_progress_milestones(
                                session_id=session_id,
                                task_session_id=telemetry_task_session_id or task_session_id,
                                progress_state=progress_state,
                                runtime_elapsed_seconds=current_elapsed,
                                max_runtime_seconds=max_runtime_seconds,
                                spent_usd=live_total_cost,
                                max_cost_usd=max_cost_usd,
                            )
                            if max_runtime_seconds > 0:
                                remaining = max_runtime_seconds - current_elapsed
                                if remaining <= 0:
                                    runtime_exceeded = True
                                    break
                            if max_cost_usd > 0 and live_total_cost >= max_cost_usd:
                                cost_exceeded = True
                                break
                            worker.join(timeout=0.1)
                        if runtime_exceeded or cost_exceeded:
                            for _ in range(20):
                                if not worker.is_alive():
                                    break
                                self._stop_thread(worker)
                                worker.join(timeout=0.05)
                            if worker.is_alive():
                                if runtime_exceeded:
                                    raise TimeoutError(
                                        "Agent run exceeded max runtime and the execution thread did not stop in time."
                                    )
                                raise TimeoutError(
                                    "Agent run exceeded max cost budget and the execution thread did not stop in time."
                                )
                            if runtime_exceeded:
                                raise TimeoutError(
                                    f"Agent run exceeded max runtime ({max_runtime_minutes} minutes)."
                                )
                            raise TimeoutError(
                                f"Agent run exceeded max cost budget (${max_cost_usd:.4f})."
                            )
                        try:
                            status, payload = result_queue.get_nowait()
                        except queue.Empty:
                            raise TimeoutError(
                                "Agent run worker thread ended without returning a result."
                            )
                        if status == "error":
                            if isinstance(payload, LiveCostLimitExceeded):
                                raise TimeoutError(str(payload))
                            raise payload
                        return payload

                    result = _invoke_with_budget()
                    if result.get("error") == "stopped":
                        break

                    (
                        attempt_prompt,
                        attempt_completion,
                        attempt_cached,
                        attempt_cache_write,
                    ) = self._usage_from_raw_result(result.get("raw_result"))

                    if max_cost_usd > 0:
                        attempt_cost = 0.0
                        model_cost = self._estimate_model_cost(
                            self._pricing,
                            str(self.config.model.primary or ""),
                            prompt_tokens=attempt_prompt,
                            completion_tokens=attempt_completion,
                            cached_prompt_tokens=attempt_cached,
                            cache_write_tokens=attempt_cache_write,
                        )
                        if model_cost is not None:
                            attempt_cost += model_cost

                        token_usage_after = TOOL_USAGE_STORE.tokens_snapshot(task_session_id)
                        token_usage_delta = self._token_usage_delta(
                            attempt_token_usage_before,
                            token_usage_after,
                        )
                        for model_name, model_usage in token_usage_delta.items():
                            nested_cost = self._estimate_model_cost(
                                self._pricing,
                                model_name,
                                prompt_tokens=model_usage[0],
                                completion_tokens=model_usage[1],
                                cached_prompt_tokens=model_usage[2],
                                cache_write_tokens=model_usage[3],
                            )
                            if nested_cost is not None:
                                attempt_cost += nested_cost
                        if attempt_cost > 0.0:
                            estimated_cost_spent += attempt_cost
                        self._emit_progress_milestones(
                            session_id=session_id,
                            task_session_id=telemetry_task_session_id or task_session_id,
                            progress_state=progress_state,
                            runtime_elapsed_seconds=time.time() - run_started_at,
                            max_runtime_seconds=max_runtime_seconds,
                            spent_usd=estimated_cost_spent,
                            max_cost_usd=max_cost_usd,
                        )
                        if estimated_cost_spent >= max_cost_usd:
                            raise TimeoutError(
                                f"Agent run exceeded max cost budget (${max_cost_usd:.4f})."
                            )
                        if cost_critical_threshold > 0 and estimated_cost_spent >= cost_critical_threshold:
                            output = result.get("output", "")
                            if output is not None:
                                result["output"] = self._append_admin_cost_warning(
                                    str(output),
                                    estimated_cost_spent,
                                    max_cost_usd,
                                    is_critical=True,
                                )
                        elif (
                            cost_warning_threshold > 0
                            and estimated_cost_spent >= cost_warning_threshold
                        ):
                            output = result.get("output", "")
                            if output is not None:
                                result["output"] = self._append_admin_cost_warning(
                                    str(output),
                                    estimated_cost_spent,
                                    max_cost_usd,
                                )

                    elapsed_runtime_seconds = time.time() - run_started_at
                    self._emit_progress_milestones(
                        session_id=session_id,
                        task_session_id=telemetry_task_session_id or task_session_id,
                        progress_state=progress_state,
                        runtime_elapsed_seconds=elapsed_runtime_seconds,
                        max_runtime_seconds=max_runtime_seconds,
                        spent_usd=estimated_cost_spent,
                        max_cost_usd=max_cost_usd,
                    )
                    if (
                        runtime_critical_threshold_seconds > 0
                        and elapsed_runtime_seconds >= runtime_critical_threshold_seconds
                    ):
                        output = result.get("output", "")
                        if output is not None:
                            result["output"] = self._append_admin_runtime_warning(
                                str(output),
                                elapsed_runtime_seconds,
                                max_runtime_minutes,
                                is_critical=True,
                            )
                    elif (
                        runtime_warning_threshold_seconds > 0
                        and elapsed_runtime_seconds >= runtime_warning_threshold_seconds
                    ):
                        output = result.get("output", "")
                        if output is not None:
                            result["output"] = self._append_admin_runtime_warning(
                                str(output),
                                elapsed_runtime_seconds,
                                max_runtime_minutes,
                            )

                    prompt_total += attempt_prompt
                    completion_total += attempt_completion
                    cached_total += attempt_cached
                    cache_write_total += attempt_cache_write

                    if result.get("error") == "max_turns_exceeded":
                        all_steps.extend(result.get("intermediate_steps", []))
                        self.logger.warning(
                            "%s: max turns exceeded after attempt %s ts=%s.",
                            run_label,
                            attempt,
                            _log_timestamp(),
                        )
                        break

                    current_steps = result.get("intermediate_steps", [])
                    all_steps.extend(current_steps)
                    has_init = any(self._is_task_steps_manager_init_step(step) for step in all_steps)
                    non_task_tools = self._non_task_tool_count(all_steps)
                    missing_init = effective_require_init and not has_init
                    missing_tools = effective_min_tools > 0 and non_task_tools < effective_min_tools
                    max_tools_reached = effective_max_tools > 0 and non_task_tools >= effective_max_tools
                    self.logger.info(
                        "%s: steps=%s non_task_tools=%s has_init=%s missing_tools=%s max_tools_reached=%s ts=%s.",
                        run_label,
                        len(all_steps),
                        non_task_tools,
                        has_init,
                        missing_tools,
                        max_tools_reached,
                        _log_timestamp(),
                    )
                    if not missing_init and not missing_tools:
                        break
                    if max_tools_reached:
                        break
                    if (
                        missing_tools
                        and not missing_init
                        and missing_tools_reminders_sent >= max_missing_tools_reminders
                    ):
                        break

                    reminders = []
                    if missing_init:
                        reminders.append(
                            "Before continuing, call task_steps_manager with action=init and a concise plan."
                        )
                    if missing_tools:
                        remaining = max(0, effective_min_tools - non_task_tools)
                        reminders.append(
                            f"Use at least {remaining} more non-task tool calls to gather context before finalizing. "
                            "Use these extra tool calls to get more context to be able to answer the question more "
                            "accurately and confidently, rather than rushing to a final answer."
                        )
                        missing_tools_reminders_sent += 1
                    current_prompt = (
                        "Continue the same run from your current context. "
                        "Do not provide your final answer yet.\n"
                        + " ".join(reminders)
                        + f"\n\nOriginal request:\n{prompt_text}"
                    )

                return (
                    result,
                    all_steps,
                    prompt_total,
                    completion_total,
                    cached_total,
                    cache_write_total,
                )

            request_text = str(text or "").strip()
            if not request_text:
                request_text = self._render_user_prompt(
                    context=context,
                    prompt_variables_override=prompt_variables_override,
                )
            if not request_text:
                raise ValueError(
                    "No user input text provided and config.user_prompt is empty."
                )

            (
                result,
                run1_all_steps,
                prompt_tokens,
                completion_tokens,
                cached_prompt_tokens,
                cache_write_prompt_tokens,
            ) = _invoke_with_min_tools(request_text, "Run 1")
            output = result.get("output", "")
            run1_output = output
            if result.get("error") == "stopped":
                enable_self_critique = False
            rounds_used = len(run1_all_steps) + (1 if run1_output else 0)
            tools_used = self._non_task_tool_count(run1_all_steps)
            self.logger.info(
                "Run 1 complete: output_chars=%s steps=%s non_task_tools=%s ts=%s.",
                len(run1_output or ""),
                len(run1_all_steps),
                tools_used,
                _log_timestamp(),
            )

            nested_counts_run1 = TOOL_USAGE_STORE.snapshot(task_session_id)

            run2_all_steps: list = []
            run2_output = ""
            if enable_self_critique and not _should_stop():
                self.logger.info("Run 2 (self-critique) starting. ts=%s", _log_timestamp())
                critique_prompt = self._require_self_critique_prompt()
                critique_input = (
                    f"{request_text}\n\nPrevious answer:\n{output}\n\n{critique_prompt}"
                )
                (
                    critique_result,
                    run2_all_steps,
                    run2_prompt_tokens,
                    run2_completion_tokens,
                    run2_cached_prompt_tokens,
                    run2_cache_write_prompt_tokens,
                ) = _invoke_with_min_tools(
                    critique_input,
                    "Run 2 (self-critique)",
                    min_tools_target=0,
                    require_task_steps_manager_init=False,
                )
                prompt_tokens += run2_prompt_tokens
                completion_tokens += run2_completion_tokens
                cached_prompt_tokens += run2_cached_prompt_tokens
                cache_write_prompt_tokens += run2_cache_write_prompt_tokens

                critique_output = critique_result.get("output", "")
                run2_output = critique_output
                output = critique_output or output
                rounds_used += len(run2_all_steps) + (1 if run2_output else 0)
                tools_used = self._non_task_tool_count(run1_all_steps + run2_all_steps)
                self.logger.info(
                    "Run 2 complete: output_chars=%s steps=%s non_task_tools=%s ts=%s.",
                    len(run2_output or ""),
                    len(run2_all_steps),
                    tools_used,
                    _log_timestamp(),
                )

            nested_counts_total = TOOL_USAGE_STORE.snapshot(task_session_id)
            nested_counts_run2 = Counter(nested_counts_total)
            nested_counts_run2.subtract(nested_counts_run1)
            nested_counts_run2 = Counter({k: v for k, v in nested_counts_run2.items() if v > 0})

            run1_tool_counts = self._step_tool_counts(run1_all_steps)
            run2_tool_counts = self._step_tool_counts(run2_all_steps)
            run1_tool_counts.update(nested_counts_run1)
            run2_tool_counts.update(nested_counts_run2)

            tool_counts = Counter(run1_tool_counts)
            tool_counts.update(run2_tool_counts)
            nested_usage_by_model = TOOL_USAGE_STORE.tokens_snapshot(task_session_id)

            run1_tools_used = (
                self._non_task_tool_count(run1_all_steps)
                + self._non_task_tool_count_from_counter(nested_counts_run1)
            )
            run2_tools_used = (
                self._non_task_tool_count(run2_all_steps)
                + self._non_task_tool_count_from_counter(nested_counts_run2)
            )

            model_name = self.config.model.primary
            main_cost = estimate_cost(
                self._pricing,
                model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                cache_write_tokens=cache_write_prompt_tokens,
            )
            nested_cost, _missing_nested_models = estimate_costs_by_model(
                self._pricing,
                nested_usage_by_model,
            )
            fallback_cost = None
            if (
                (main_cost is None or main_cost == 0.0)
                and (
                    prompt_tokens
                    or completion_tokens
                    or cached_prompt_tokens
                    or cache_write_prompt_tokens
                )
            ):
                fallback_cost = estimate_cost_with_defaults(
                    model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_prompt_tokens=cached_prompt_tokens,
                    cache_write_tokens=cache_write_prompt_tokens,
                )
            if main_cost is None and nested_cost == 0:
                total_cost = None
            else:
                total_cost = (main_cost or 0.0) + nested_cost
            if (total_cost is None or total_cost == 0.0) and fallback_cost:
                total_cost = fallback_cost
            if total_cost is None:
                cost_text = "unknown"
            else:
                cost_text = f"${total_cost:.4f}"

            run1_steps = len(run1_all_steps)
            run2_steps = len(run2_all_steps)
            max_turns = int(self.config.session.max_turns or 0)
            tool_counts_text = self._format_tool_counts(tool_counts)
            suffix = (
                f"\n\n🔁 {run1_steps}/{run2_steps}/{max_turns} | 🧰 {run1_tools_used}/{run2_tools_used} | 💲 {cost_text}\n"
                f"{tool_counts_text}"
            )

            if on_task_steps_manager_update is not None:
                STORE.unregister_listener(task_session_id, _listener)
            TOOL_USAGE_STORE.clear(task_session_id)

            if self.config.session.long_term_memory_enabled:
                asyncio.run(self._finalize_long_term_memory(session_id))

            if result.get("error"):
                error_text = str(result.get("error") or "unknown_error")
                trace_text = str(result.get("traceback") or "").strip()
                if not trace_text:
                    trace_text = (
                        "No Python traceback captured. "
                        f"Error reported by agent flow: {error_text}"
                    )
                log_event(
                    "agent_error",
                    payload={
                        "session_id": session_id,
                        "task_session_id": telemetry_task_session_id or task_session_id,
                        "main_action": str(self.config.agent.main_action or ""),
                        "sub_action": str(self.config.agent.sub_action or ""),
                        "error": error_text,
                        "traceback": trace_text,
                    },
                    main_action=str(self.config.agent.main_action or ""),
                    sub_action=str(self.config.agent.sub_action or ""),
                    session_id=session_id,
                    task_session_id=telemetry_task_session_id or task_session_id,
                    model=str(self.config.model.primary or ""),
                )

            self._emit_system_metrics(
                session_id=session_id,
                task_session_id=telemetry_task_session_id or task_session_id,
                trigger="agent_end",
            )
            log_event(
                "agent_end",
                payload={
                    "session_id": session_id,
                    "task_session_id": telemetry_task_session_id or task_session_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_prompt_tokens": cached_prompt_tokens,
                    "cache_write_prompt_tokens": cache_write_prompt_tokens,
                    "total_cost": total_cost,
                    "main_cost": main_cost,
                    "nested_cost": nested_cost,
                    "pricing_model": model_name,
                    "missing_pricing_models": _missing_nested_models,
                    "cost_source": (
                        "fallback_default"
                        if fallback_cost and (main_cost is None or main_cost == 0.0)
                        else "pricing_table"
                    ),
                    "rounds_used": rounds_used,
                    "tools_used": tools_used,
                    "run1_steps": run1_steps,
                    "run2_steps": run2_steps,
                    "run1_tools_used": run1_tools_used,
                    "run2_tools_used": run2_tools_used,
                    "tool_counts": dict(tool_counts),
                    "nested_tool_counts": dict(nested_counts_total),
                    "nested_usage_by_model": nested_usage_by_model,
                },
            )

            self.logger.info(
                "Run finished: session=%s rounds=%s tools_used=%s cost=%s ts=%s.",
                session_id,
                rounds_used,
                tools_used,
                cost_text,
                _log_timestamp(),
            )
            return RunResult(
                output=output,
                steps=result.get("intermediate_steps", []),
                all_steps=run1_all_steps + run2_all_steps,
                tool_counts=tool_counts,
                nested_tool_counts=nested_counts_total,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                cache_write_prompt_tokens=cache_write_prompt_tokens,
                rounds_used=rounds_used,
                tools_used=tools_used,
                task_session_id=telemetry_task_session_id or task_session_id,
                nested_usage_by_model=nested_usage_by_model,
                run1_output=run1_output,
                run2_output=run2_output,
                run1_steps=run1_steps,
                run2_steps=run2_steps,
                max_turns=max_turns,
                run1_tools_used=run1_tools_used,
                run2_tools_used=run2_tools_used,
                total_cost=total_cost,
                tool_counts_text=tool_counts_text,
                suffix=suffix,
            )
        except Exception as exc:
            self.logger.exception(
                "Run failed: session=%s task_session=%s ts=%s.",
                session_id,
                telemetry_task_session_id or task_session_id,
                _log_timestamp(),
            )
            log_event(
                "agent_error",
                payload={
                    "session_id": session_id,
                    "task_session_id": telemetry_task_session_id or task_session_id,
                    "main_action": str(self.config.agent.main_action or ""),
                    "sub_action": str(self.config.agent.sub_action or ""),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                main_action=str(self.config.agent.main_action or ""),
                sub_action=str(self.config.agent.sub_action or ""),
                session_id=session_id,
                task_session_id=telemetry_task_session_id or task_session_id,
                model=str(self.config.model.primary or ""),
            )
            raise
        finally:
            metrics_stop_event.set()
            if metrics_thread is not None:
                try:
                    metrics_thread.join(timeout=1.0)
                except Exception:
                    pass
            reset_log_context(log_token)
