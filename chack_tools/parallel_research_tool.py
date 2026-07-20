from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict

from .researcher_administrator_agent import (
    RESEARCHER_REGISTRY,
    ResearcherAdministratorAgentTool,
    normalize_researcher_name,
)
from .subagent_config import researcher_response_from_output
from .tool_usage_state import STORE

try:
    from agents import function_tool
except ImportError:
    function_tool = None


MIN_RESEARCH_PROMPT_CHARS = 500
MAX_PARALLEL_RESEARCHERS = 4


class ParallelResearchRequest(TypedDict):
    researcher: str
    prompt: str


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()


def get_parallel_research_tool(researcher_tools: list[Any], *, max_requests: int = 4):
    """Build one root tool that dispatches selected researcher tools concurrently."""
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    tools_by_name = {_tool_name(tool): tool for tool in researcher_tools if _tool_name(tool)}
    configured_max = max(1, min(int(max_requests or 4), MAX_PARALLEL_RESEARCHERS))

    @function_tool(name_override="parallel_research")
    def parallel_research(
        requests: list[ParallelResearchRequest],
    ) -> str:
        """Run up to four selected researchers concurrently.

        Args:
            requests: Array of objects with `researcher` and `prompt`. The
                researcher may be a short name such as travel, websearcher, news_media,
                or social_network, or its full tool name. Every prompt must contain at
                least 500 characters of task-specific context and evidence requirements.
        Output: JSON containing results in request order plus validation/runtime errors.
        """
        if not isinstance(requests, list) or not requests:
            return json.dumps({"worked": False, "results": [], "errors": ["requests must be a non-empty array."]})
        if len(requests) > configured_max:
            return json.dumps({
                "worked": False,
                "results": [],
                "errors": [f"At most {configured_max} researcher requests are allowed."],
            })

        normalized: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, item in enumerate(requests):
            if not isinstance(item, dict):
                errors.append({"index": index, "error": "Each request must be an object."})
                continue
            researcher = normalize_researcher_name(str(item.get("researcher") or item.get("tool") or ""))
            tool_name = RESEARCHER_REGISTRY.get(researcher, ("", ""))[1]
            if not tool_name or tool_name not in tools_by_name:
                errors.append({"index": index, "researcher": researcher, "error": "Researcher is not enabled."})
                continue
            prompt = str(item.get("prompt") or "").strip()
            if len(prompt) < MIN_RESEARCH_PROMPT_CHARS:
                errors.append({
                    "index": index,
                    "researcher": researcher,
                    "error": f"Researcher prompt must be at least {MIN_RESEARCH_PROMPT_CHARS} characters.",
                })
                continue
            normalized.append({
                "index": index,
                "researcher": researcher,
                "tool_name": tool_name,
                "prompt": prompt,
            })
        if errors or not normalized:
            return json.dumps({"worked": False, "results": [], "errors": errors}, ensure_ascii=False)

        def run_one(row: dict[str, Any], context: contextvars.Context) -> dict[str, Any]:
            def invoke() -> dict[str, Any]:
                # The parent agent sees `parallel_research` as its root tool. Record
                # each selected researcher in the same run-scoped usage ledger so
                # callers can audit which nested researchers were actually invoked.
                STORE.add(row["tool_name"])
                output = ResearcherAdministratorAgentTool._invoke_tool_sync(
                    tools_by_name[row["tool_name"]],
                    {"prompt": row["prompt"], "save_artifacts": False},
                )
                result = {
                    "index": row["index"],
                    "researcher": row["researcher"],
                    "researcher_tool": row["tool_name"],
                    "output": output,
                }
                parsed = researcher_response_from_output(row["tool_name"], output)
                if parsed is not None:
                    result["parsed_response"] = parsed
                return result

            return context.run(invoke)

        worker_count = max(1, min(configured_max, len(normalized)))
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="parallel-research") as executor:
            futures = {
                executor.submit(run_one, row, contextvars.copy_context()): row
                for row in normalized
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({
                        "index": row["index"],
                        "researcher": row["researcher"],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        results.sort(key=lambda item: int(item["index"]))
        return json.dumps(
            {"worked": bool(results) and not errors, "results": results, "errors": errors},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    parallel_research.description += (
        "\n\nUse this when two or more independent research tasks can run concurrently. "
        "Each request chooses its researcher explicitly and every prompt is hard-rejected below 500 characters. "
        "The tool accepts at most four requests and returns all outputs to the calling agent for synthesis.\n"
        "Parameters: requests is an array of researcher/prompt objects. No other parameters are accepted.\n"
        "Output: JSON with worked, request-ordered researcher results, parsed responses when available, and errors."
    )
    return parallel_research
