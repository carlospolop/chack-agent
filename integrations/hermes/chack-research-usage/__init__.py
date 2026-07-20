"""Deterministic per-turn Chack researcher usage footer for Hermes.

The Chack queue result carries an exact structured ledger. This plugin observes
queue tool returns and appends a compact footer to the final response. Because
``transform_llm_output`` runs for gateway and cron turns, progress messages stay
clean and the aggregate appears exactly once on the final delivery.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_QUEUE_TOOL_RE = re.compile(r"(?:^|_+)researcher_queue$")
_ALLOWED_ARTIFACT_ROOTS = (Path("/tmp/chack-research-data").resolve(),)
_MAX_RESULT_CHARS = 2_000_000
_MAX_ADMIN_FILE_BYTES = 5_000_000

_DISPLAY_NAMES = {
    "prochatgpt_researcher": "chatgptpro",
    "deepchatgpt_researcher": "chatgptdeep",
    "websearcher_research": "webresearcher",
    "scientific_research": "scientific",
    "business_research": "business",
    "product_research": "product",
    "travel_research": "travel",
    "social_network_research": "socialnetwork",
    "legal_research": "legal",
    "data_statistics_research": "datastatistics",
    "news_media_research": "newsmedia",
    "knowledge_graph_research": "knowledgegraph",
    "religious_research": "religious",
    "cli_research": "cli",
}
_DISPLAY_PRIORITY = {
    "chatgptpro": 0,
    "chatgptdeep": 1,
    "webresearcher": 2,
}


@dataclass
class _TurnUsage:
    turn_id: str = ""
    queue_calls: int = 0
    administrator_calls: int = 0
    researcher_calls: dict[str, int] = field(default_factory=dict)
    complete: bool = True
    seen_tool_call_ids: set[str] = field(default_factory=set)


_LOCK = threading.RLock()
_TURNS: dict[str, _TurnUsage] = {}


def _state_key(session_id: str, task_id: str = "") -> str:
    session = str(session_id or "").strip()
    return f"session:{session}" if session else f"task:{str(task_id or '').strip()}"


def _is_queue_tool(tool_name: str) -> bool:
    name = str(tool_name or "").strip()
    return name == "researcher_queue" or bool(_QUEUE_TOOL_RE.search(name))


def _positive_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            result[name] = result.get(name, 0) + count
    return result


def _first_json(text: str) -> Any:
    source = str(text or "")[:_MAX_RESULT_CHARS]
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = source.find("{", cursor)
        if start < 0:
            return None
        try:
            value, _ = decoder.raw_decode(source[start:])
            return value
        except json.JSONDecodeError:
            cursor = start + 1


def _find_queue_payload(result: Any) -> dict[str, Any] | None:
    pending = [result]
    seen_strings: set[str] = set()
    while pending:
        value = pending.pop(0)
        if isinstance(value, dict):
            if isinstance(value.get("researcher_usage"), dict) or isinstance(value.get("researches"), list):
                return value
            for key in ("result", "structuredContent", "content", "output"):
                nested = value.get(key)
                if nested not in (None, ""):
                    pending.append(nested)
            continue
        if isinstance(value, list):
            pending.extend(value)
            continue
        if not isinstance(value, str) or value in seen_strings:
            continue
        seen_strings.add(value)
        decoded = _first_json(value)
        if decoded is not None:
            pending.append(decoded)
    return None


def _safe_admin_json(path_value: Any) -> dict[str, Any] | None:
    try:
        path = Path(str(path_value or "")).expanduser().resolve(strict=True)
        if path.name != "admin_output.json":
            return None
        if not any(path.is_relative_to(root) for root in _ALLOWED_ARTIFACT_ROOTS):
            return None
        if not path.is_file() or path.stat().st_size > _MAX_ADMIN_FILE_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _row_counts(row: dict[str, Any]) -> tuple[dict[str, int], bool]:
    raw = row.get("researcher_call_counts")
    if isinstance(raw, dict):
        return _positive_counts(raw), row.get("researcher_usage_complete") is not False

    output_files = row.get("output_files")
    if isinstance(output_files, dict):
        for path_value in output_files.values():
            payload = _safe_admin_json(path_value)
            if payload is not None and isinstance(payload.get("researcher_call_counts"), dict):
                return _positive_counts(payload["researcher_call_counts"]), True
    return {}, False


def _usage_from_payload(payload: dict[str, Any]) -> tuple[int, dict[str, int], bool]:
    usage = payload.get("researcher_usage")
    if isinstance(usage, dict):
        try:
            administrators = max(0, int(usage.get("administrator_calls") or 0))
        except (TypeError, ValueError):
            administrators = 0
        return (
            administrators,
            _positive_counts(usage.get("researcher_call_counts")),
            usage.get("complete") is True,
        )

    rows = [row for row in (payload.get("researches") or []) if isinstance(row, dict)]
    counts: dict[str, int] = {}
    complete = True
    for row in rows:
        row_counts, row_complete = _row_counts(row)
        complete = complete and row_complete
        for name, count in row_counts.items():
            counts[name] = counts.get(name, 0) + count
    return len(rows), counts, complete


def _display_name(tool_name: str) -> str:
    if tool_name in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[tool_name]
    clean = re.sub(r"[^a-z0-9_-]+", "", str(tool_name or "").lower())
    for suffix in ("_researcher", "_research"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    return clean[:40] or "unknown"


def _footer(usage: _TurnUsage) -> str:
    parts = [f"queue x{usage.queue_calls}", f"admin x{usage.administrator_calls}"]
    displayed: dict[str, int] = {}
    for tool_name, count in usage.researcher_calls.items():
        name = _display_name(tool_name)
        displayed[name] = displayed.get(name, 0) + count
    ordered = sorted(displayed.items(), key=lambda item: (_DISPLAY_PRIORITY.get(item[0], 100), item[0]))
    parts.extend(f"{name} x{count}" for name, count in ordered)
    label = "Chack usage" if usage.complete else "Chack usage (partial accounting)"
    return f"_{label}: {' · '.join(parts)}_"


def _pre_llm_call(*, session_id: str = "", task_id: str = "", turn_id: str = "", **_: Any) -> None:
    key = _state_key(session_id, task_id)
    with _LOCK:
        _TURNS[key] = _TurnUsage(turn_id=str(turn_id or ""))


def _post_tool_call(
    *,
    tool_name: str,
    result: Any,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
    status: str = "",
    **_: Any,
) -> None:
    if not _is_queue_tool(tool_name):
        return
    key = _state_key(session_id, task_id)
    with _LOCK:
        usage = _TURNS.setdefault(key, _TurnUsage(turn_id=str(turn_id or "")))
        if usage.turn_id and turn_id and usage.turn_id != str(turn_id):
            usage = _TurnUsage(turn_id=str(turn_id))
            _TURNS[key] = usage
        call_id = str(tool_call_id or "").strip()
        if call_id and call_id in usage.seen_tool_call_ids:
            return
        if call_id:
            usage.seen_tool_call_ids.add(call_id)
        usage.queue_calls += 1

        payload = _find_queue_payload(result)
        if payload is None:
            usage.complete = False
            return
        administrators, researcher_counts, complete = _usage_from_payload(payload)
        usage.administrator_calls += administrators
        for name, count in researcher_counts.items():
            usage.researcher_calls[name] = usage.researcher_calls.get(name, 0) + count
        usage.complete = usage.complete and complete and str(status or "").lower() not in {"error", "failed"}


def _transform_llm_output(*, response_text: str, session_id: str = "", **_: Any) -> str | None:
    key = _state_key(session_id)
    with _LOCK:
        usage = _TURNS.pop(key, None)
    if usage is None or usage.queue_calls <= 0:
        return None
    return f"{str(response_text or '').rstrip()}\n\n{_footer(usage)}"


def register(ctx: Any) -> None:
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
