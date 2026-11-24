from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import contextvars
import hashlib
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    from agents import function_tool
except ImportError:
    function_tool = None

from .config import ToolsConfig
from .run_cache import RunScopedCache
from .tool_call_policy import reuse_tool_results
from .telemetry import run_with_tool_logging


ARTIFACT_MANIFEST_FILENAME = "_artifact_manifest.jsonl"
_RESEARCH_DATA_DIR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chack_research_data_dir",
    default="",
)
_RESEARCH_MASTER_DIR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chack_research_master_dir",
    default="",
)
_ARTIFACT_MANIFEST_LOCK = threading.RLock()


def _set_param_descriptions(tool: Any, descriptions: dict[str, str]):
    schema = getattr(tool, "params_json_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, description in descriptions.items():
            if isinstance(properties.get(name), dict):
                properties[name]["description"] = description
    current = str(getattr(tool, "description", "") or "").strip()
    if current and "Output:" not in current:
        tool.description = (
            f"{current}\n\n"
            "Parameters: Use the schema descriptions to choose artifact paths, globs, line ranges, search text, regex behavior, limits, and truncation caps.\n"
            "Output: Returns SUCCESS/ERROR text containing artifact paths, selected file contents, or grep matches from this research run's evidence folder."
        )
    return tool


def research_artifacts_root() -> str:
    return (_RESEARCH_DATA_DIR.get() or os.environ.get("CHACK_RESEARCH_DATA_DIR", "")).strip()


def research_artifacts_master_root() -> str:
    return (_RESEARCH_MASTER_DIR.get() or os.environ.get("CHACK_RESEARCH_MASTER_DIR", "")).strip()


def set_research_artifact_context(data_dir: str = "", master_dir: str = ""):
    data_token = _RESEARCH_DATA_DIR.set(str(data_dir or "").strip())
    master_token = _RESEARCH_MASTER_DIR.set(str(master_dir or "").strip())
    return data_token, master_token


def reset_research_artifact_context(tokens) -> None:
    data_token, master_token = tokens
    _RESEARCH_MASTER_DIR.reset(master_token)
    _RESEARCH_DATA_DIR.reset(data_token)


@contextmanager
def research_artifact_context(data_dir: str = "", master_dir: str = "") -> Iterator[None]:
    tokens = set_research_artifact_context(data_dir, master_dir)
    try:
        yield
    finally:
        reset_research_artifact_context(tokens)


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _infer_source_url(value: Any) -> str:
    seen: set[int] = set()
    keys = (
        "source_url",
        "url",
        "link",
        "href",
        "final_url",
        "requested_url",
        "pdf_url",
        "landing_page_url",
        "html_url",
    )

    def walk(item: Any, depth: int = 0) -> str:
        if depth > 4:
            return ""
        if isinstance(item, dict):
            ident = id(item)
            if ident in seen:
                return ""
            seen.add(ident)
            for key in keys:
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip().lower().startswith(("http://", "https://")):
                    return raw.strip()
            for raw in item.values():
                found = walk(raw, depth + 1)
                if found:
                    return found
        elif isinstance(item, list):
            for raw in item[:20]:
                found = walk(raw, depth + 1)
                if found:
                    return found
        elif isinstance(item, str):
            match = re.search(r"https?://[^\s)>\"]+", item.strip())
            if match:
                return match.group(0).rstrip(".,;")
        return ""

    return walk(value)


def record_research_artifact(
    path: str | Path,
    *,
    source_url: str = "",
    provenance: str = "",
    tool: str = "",
    kind: str = "",
    label: str = "",
) -> None:
    root = research_artifacts_root()
    if not root:
        return
    try:
        root_path = Path(root).expanduser().resolve()
        file_path = Path(path).expanduser().resolve()
        rel = str(file_path.relative_to(root_path))
    except (OSError, ValueError):
        return
    if rel == ARTIFACT_MANIFEST_FILENAME:
        return
    row = {
        "filename": rel,
        "source_url": str(source_url or "").strip(),
        "provenance": str(provenance or "").strip(),
        "tool": str(tool or "").strip(),
        "kind": str(kind or "").strip(),
        "label": str(label or "").strip(),
    }
    if not row["source_url"]:
        row["source_url"] = _infer_source_url({"provenance": provenance, "label": label})
    manifest = root_path / ARTIFACT_MANIFEST_FILENAME
    with _ARTIFACT_MANIFEST_LOCK:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        if manifest.is_file():
            current = _compact_json(row)
            for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip() == current:
                    return
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(_compact_json(row) + "\n")


def record_research_json_artifact(
    path: str | Path,
    payload: Any,
    *,
    provenance: str = "",
    tool: str = "",
    kind: str = "",
    label: str = "",
) -> None:
    record_research_artifact(
        path,
        source_url=_infer_source_url(payload),
        provenance=provenance,
        tool=tool,
        kind=kind,
        label=label,
    )


def remove_research_artifact_manifest_entry(evidence_dir: str | Path, filename: str | Path) -> None:
    root = Path(str(evidence_dir or "")).expanduser()
    rel = str(filename or "").strip()
    if not rel:
        return
    manifest = root / ARTIFACT_MANIFEST_FILENAME
    if not manifest.is_file():
        return
    with _ARTIFACT_MANIFEST_LOCK:
        kept: list[str] = []
        changed = False
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if isinstance(payload, dict) and str(payload.get("filename") or "").strip() == rel:
                changed = True
                continue
            kept.append(line)
        if changed:
            manifest.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")


def research_artifact_manifest(evidence_dir: str) -> dict[str, dict[str, str]]:
    root = Path(str(evidence_dir or "")).expanduser()
    manifest = root / ARTIFACT_MANIFEST_FILENAME
    with _ARTIFACT_MANIFEST_LOCK:
        if not manifest.is_file():
            return {}
        rows: dict[str, dict[str, str]] = {}
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            filename = str(payload.get("filename") or "").strip()
            if not filename:
                continue
            rows[filename] = {
                "source_url": str(payload.get("source_url") or "").strip(),
                "provenance": str(payload.get("provenance") or "").strip(),
                "tool": str(payload.get("tool") or "").strip(),
                "kind": str(payload.get("kind") or "").strip(),
                "label": str(payload.get("label") or "").strip(),
            }
        return rows


def register_untracked_research_artifacts(evidence_dir: str | Path) -> int:
    """Register evidence files that were created without an artifact-aware tool."""
    root = Path(str(evidence_dir or "")).expanduser()
    if not root.is_dir():
        return 0
    try:
        root = root.resolve()
    except OSError:
        return 0
    with _ARTIFACT_MANIFEST_LOCK:
        known = set(research_artifact_manifest(str(root)).keys())
        added = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = str(path.resolve().relative_to(root))
            except (OSError, ValueError):
                continue
            if rel == ARTIFACT_MANIFEST_FILENAME or rel in known:
                continue
            row = {
                "filename": rel,
                "source_url": "",
                "provenance": f"unregistered local evidence file created during researcher execution: {rel}",
                "tool": "unregistered_file_fallback",
                "kind": "unregistered-file",
                "label": rel,
            }
            manifest = root / ARTIFACT_MANIFEST_FILENAME
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(_compact_json(row) + "\n")
            known.add(rel)
            added += 1
        return added


def cleanup_research_artifacts(path: str, *, save_artifacts: bool) -> None:
    if save_artifacts:
        return
    root = str(path or "").strip()
    if not root:
        return
    resolved = Path(root).expanduser().resolve()
    # A researcher_administrator owns the lifecycle of its master evidence folder.
    # Per-type subfolders live inside it and are shared by sibling researchers, so
    # an individual researcher must never delete them mid-run; only the
    # administrator itself (path == master) is allowed to clean the whole tree.
    master = research_artifacts_master_root()
    if master:
        master_resolved = Path(master).expanduser().resolve()
        if resolved != master_resolved:
            try:
                resolved.relative_to(master_resolved)
                return
            except ValueError:
                pass
    tmp_root = Path("/tmp/chack-research-data").resolve()
    try:
        resolved.relative_to(tmp_root)
    except ValueError:
        return
    shutil.rmtree(resolved, ignore_errors=True)


DEFAULT_READ_MAX_CHARS = 6000
DEFAULT_CONTEXT_LINES = 10
DEFAULT_GREP_MAX_MATCHES = 20
DEFAULT_GREP_MAX_CHARS = 6000
# A matching line longer than this is shown as a window around the match.
_LONG_LINE_CHARS = 600
_LINE_WINDOW_CHARS = 240
REPEAT_CACHE_ENTRIES = 256


def _clip_span(line: str, start: int, end: int) -> str:
    if len(line) <= _LONG_LINE_CHARS:
        return line
    left = max(0, start - _LINE_WINDOW_CHARS)
    right = min(len(line), max(end, start) + _LINE_WINDOW_CHARS)
    prefix = f"[...{left} chars] " if left else ""
    suffix = f" [{len(line) - right} chars...]" if right < len(line) else ""
    return f"{prefix}{line[left:right]}{suffix}"


def _clip_around(line: str, lowered_needle: str) -> str:
    position = line.lower().find(lowered_needle) if lowered_needle else -1
    if position < 0:
        return _clip_span(line, 0, 0)
    return _clip_span(line, position, position + len(lowered_needle))


class ResearchArtifactsTool:
    def __init__(self, config: ToolsConfig, root: str = ""):
        self.config = config
        # When set, these tools are pinned to this directory regardless of the
        # per-run research context/env. Used to expose the file tools over a fixed
        # folder (e.g. a shared researcher-queue evidence folder for verifiers).
        self._explicit_root = str(root or "").strip()

    def _root(self) -> Path:
        root = self._explicit_root or research_artifacts_root()
        if not root:
            raise ValueError("CHACK_RESEARCH_DATA_DIR is not configured")
        path = Path(root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_inside_root(self, value: str) -> Path:
        root = self._root()
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("path cannot be empty")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("path must be inside CHACK_RESEARCH_DATA_DIR") from exc
        if not resolved.is_file():
            raise ValueError(f"file not found: {resolved}")
        return resolved

    def list_files(self, glob: str = "*", max_results: int = 200) -> str:
        root = self._root()
        pattern = str(glob or "*").strip() or "*"
        limit = max(1, min(int(max_results or 200), 1000))
        rows: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            if not fnmatch.fnmatch(rel, pattern) and not fnmatch.fnmatch(path.name, pattern):
                continue
            stat = path.stat()
            rows.append(f"{rel}\t{stat.st_size} bytes")
            if len(rows) >= limit:
                break
        if not rows:
            return f"SUCCESS: No research artifacts matched '{pattern}' in {root}"
        return "SUCCESS: Research artifacts:\n" + "\n".join(rows)

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 0,
        around_text: str = "",
        context_lines: int = DEFAULT_CONTEXT_LINES,
        max_chars: int = DEFAULT_READ_MAX_CHARS,
    ) -> str:
        file_path = self._resolve_inside_root(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        budget = max(200, int(max_chars or DEFAULT_READ_MAX_CHARS))
        if around_text:
            needle = str(around_text)
            lowered = needle.lower()
            matches = [idx for idx, line in enumerate(lines) if lowered in line.lower()]
            if not matches:
                return f"SUCCESS: No matches for '{needle}' in {file_path} ({total} lines)"
            context = max(0, min(int(context_lines or 0), 200))
            # Overlapping windows are merged so no line is sent twice.
            windows: list[list[int]] = []
            for match in matches:
                start, end = max(0, match - context), min(total, match + context + 1)
                if windows and start <= windows[-1][1]:
                    windows[-1][1] = max(windows[-1][1], end)
                else:
                    windows.append([start, end])
            out = [f"SUCCESS: {len(matches)} line(s) match '{needle}' in {file_path} ({total} lines)"]
            used = len(out[0])
            for shown, (start, end) in enumerate(windows):
                rows = [f"--- lines {start + 1}-{end} ---"] + [
                    f"{idx + 1}: {_clip_around(lines[idx], lowered)}" for idx in range(start, end)
                ]
                chunk = "\n".join(rows)
                if used + len(chunk) > budget and shown:
                    out.append(
                        f"[{len(windows) - shown} more match window(s) not shown; "
                        f"next starts at line {start + 1}. Narrow around_text or read that line range.]"
                    )
                    break
                out.append(chunk[: max(0, budget - used)])
                used += len(chunk) + 1
            return "\n".join(out)

        start = max(1, int(start_line or 1))
        end = int(end_line or 0)
        if end <= 0 or end > total:
            end = total
        if total and start > end:
            return f"ERROR: start_line ({start}) cannot be greater than end_line ({end}); the file has {total} lines"
        rows: list[str] = []
        used = 0
        last = start - 1
        for idx in range(start - 1, end):
            row = f"{idx + 1}: {lines[idx]}"
            if used + len(row) > budget:
                if not rows:
                    rows.append(row[:budget] + f" [line {idx + 1} cut at {budget} chars; use around_text to read inside it]")
                    last = idx + 1
                break
            rows.append(row)
            used += len(row) + 1
            last = idx + 1
        header = f"SUCCESS: Read {file_path} lines {start}-{last} of {total} ({len(text)} chars in file)"
        footer = ""
        if last < end:
            footer = (
                f"\n[Stopped at max_chars. Continue with start_line={last + 1}, "
                "or use around_text / grep_research_artifacts to jump to what you need.]"
            )
        return header + "\n" + "\n".join(rows) + footer

    def grep(
        self,
        pattern: str,
        glob: str = "*",
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_matches: int = DEFAULT_GREP_MAX_MATCHES,
        max_chars: int = DEFAULT_GREP_MAX_CHARS,
    ) -> str:
        root = self._root()
        raw_pattern = str(pattern or "")
        if not raw_pattern:
            return "ERROR: pattern cannot be empty"
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(raw_pattern, flags)
        except re.error as exc:
            return f"ERROR: invalid regex ({exc})"
        file_pattern = str(glob or "*").strip() or "*"
        context = max(0, min(int(context_lines or 0), 20))
        limit = max(1, min(int(max_matches or DEFAULT_GREP_MAX_MATCHES), 500))
        budget = max(200, int(max_chars or DEFAULT_GREP_MAX_CHARS))
        rows: list[str] = []
        used = 0
        matches = 0
        stopped = ""
        for file_path in sorted(root.rglob("*")):
            if stopped:
                break
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(root))
            if not fnmatch.fnmatch(rel, file_pattern) and not fnmatch.fnmatch(file_path.name, file_pattern):
                continue
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for idx, line in enumerate(lines):
                found = regex.search(line)
                if not found:
                    continue
                if matches >= limit:
                    stopped = f"max_matches={limit}"
                    break
                start = max(0, idx - context)
                end = min(len(lines), idx + context + 1)
                block = []
                for row_idx in range(start, end):
                    marker = ":" if row_idx == idx else "-"
                    shown = _clip_span(lines[row_idx], found.start(), found.end()) if row_idx == idx else _clip_span(lines[row_idx], 0, 0)
                    block.append(f"{rel}{marker}{row_idx + 1}: {shown}")
                chunk = "\n".join(block)
                if used + len(chunk) > budget and rows:
                    stopped = f"max_chars={budget}"
                    break
                rows.append(chunk)
                used += len(chunk) + 1
                matches += 1
        if not rows:
            return f"SUCCESS: No grep matches for '{raw_pattern}' in {root}"
        if stopped:
            return (
                f"SUCCESS: First {matches} grep matches (stopped at {stopped}; more exist: "
                "narrow the pattern or glob, or read a file around a match)\n" + "\n".join(rows)
            )
        return f"SUCCESS: {matches} grep matches\n" + "\n".join(rows)

    def delete_file(self, path: str) -> str:
        root = self._root()
        file_path = self._resolve_inside_root(path)
        rel = file_path.relative_to(root)
        file_path.unlink()
        remove_research_artifact_manifest_entry(root, rel)
        return f"SUCCESS: Deleted research artifact {rel}"

    def register_file(
        self,
        path: str,
        source_url: str = "",
        provenance: str = "",
        tool: str = "exec",
        kind: str = "manual",
        label: str = "",
    ) -> str:
        root = self._root()
        file_path = self._resolve_inside_root(path)
        rel = file_path.relative_to(root)
        record_research_artifact(
            file_path,
            source_url=source_url,
            provenance=provenance,
            tool=tool,
            kind=kind,
            label=label or str(rel),
        )
        return f"SUCCESS: Registered research artifact {rel}"


def get_research_artifact_tools(helper: ResearchArtifactsTool, *, readonly: bool = False) -> list[Any]:
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    # Results already returned in this run, per tool list (one run) and run
    # scope: an identical call whose result has not changed gets a short notice
    # once; asking again returns the text, so nothing becomes unreachable.
    served = RunScopedCache(REPEAT_CACHE_ENTRIES)

    def once(tool: str, tool_input: dict, produce) -> str:
        def run() -> str:
            result = produce()
            if not isinstance(result, str) or not result.startswith("SUCCESS"):
                return result
            key = (tool, json.dumps(tool_input, sort_keys=True, default=str))
            digest = hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()
            earlier = served.get(key)
            if earlier is not None and earlier[0] == digest and not earlier[1]:
                served.put(key, (digest, True))
                return (
                    f"[Unchanged since the identical {tool} call earlier in this run "
                    f"({result.splitlines()[0][len('SUCCESS: '):]}); not repeated. "
                    "Call it again with the same arguments if you need that text once more.]"
                )
            served.put(key, (digest, False))
            return result

        return run_with_tool_logging(tool, tool_input, run)

    @function_tool(name_override="list_research_artifacts")
    def list_research_artifacts(glob: str = "*", max_results: int = 200) -> str:
        """List files saved in this research run's evidence folder."""
        return once(
            "list_research_artifacts",
            {"glob": glob, "max_results": max_results},
            lambda: helper.list_files(glob=glob, max_results=max_results),
        )

    @function_tool(name_override="read_research_artifact")
    def read_research_artifact(
        path: str,
        start_line: int = 1,
        end_line: int = 0,
        around_text: str = "",
        context_lines: int = DEFAULT_CONTEXT_LINES,
        max_chars: int = DEFAULT_READ_MAX_CHARS,
    ) -> str:
        """Read a saved artifact by relative/absolute path, optionally by line range or around matching text."""
        return once(
            "read_research_artifact",
            {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "around_text": around_text,
                "context_lines": context_lines,
                "max_chars": max_chars,
            },
            lambda: helper.read_file(
                path=path,
                start_line=start_line,
                end_line=end_line,
                around_text=around_text,
                context_lines=context_lines,
                max_chars=max_chars,
            ),
        )

    @function_tool(name_override="grep_research_artifacts")
    def grep_research_artifacts(
        pattern: str,
        glob: str = "*",
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_matches: int = DEFAULT_GREP_MAX_MATCHES,
        max_chars: int = DEFAULT_GREP_MAX_CHARS,
    ) -> str:
        """Regex-search saved artifacts inside this research run's evidence folder."""
        return once(
            "grep_research_artifacts",
            {
                "pattern": pattern,
                "glob": glob,
                "case_sensitive": case_sensitive,
                "context_lines": context_lines,
                "max_matches": max_matches,
                "max_chars": max_chars,
            },
            lambda: helper.grep(
                pattern=pattern,
                glob=glob,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                max_matches=max_matches,
                max_chars=max_chars,
            ),
        )

    @function_tool(name_override="delete_research_artifact")
    def delete_research_artifact(path: str) -> str:
        """Delete one saved artifact inside this research run's evidence folder when it was not useful."""
        return run_with_tool_logging(
            "delete_research_artifact",
            {"path": path},
            lambda: helper.delete_file(path=path),
        )

    @function_tool(name_override="register_research_artifact")
    def register_research_artifact(
        path: str,
        source_url: str = "",
        provenance: str = "",
        tool: str = "exec",
        kind: str = "manual",
        label: str = "",
    ) -> str:
        """Register an existing evidence file that was created by exec or another non-artifact-aware tool."""
        return run_with_tool_logging(
            "register_research_artifact",
            {
                "path": path,
                "source_url": source_url,
                "provenance": provenance,
                "tool": tool,
                "kind": kind,
                "label": label,
            },
            lambda: helper.register_file(
                path=path,
                source_url=source_url,
                provenance=provenance,
                tool=tool,
                kind=kind,
                label=label,
            ),
        )

    built = [
        _set_param_descriptions(list_research_artifacts, {
            "glob": "File glob pattern relative to this research run's evidence folder, such as *, *.txt, or web-pages/*.html.",
            "max_results": "Maximum number of saved artifact paths to list.",
        }),
        _set_param_descriptions(read_research_artifact, {
            "path": "Relative artifact path from list_research_artifacts, or an absolute path inside this run's evidence folder.",
            "start_line": "First 1-based line number to read when using a line range.",
            "end_line": "Last 1-based line number to read; 0 reads from start_line until the file end or max_chars, and the reply says which start_line continues.",
            "around_text": "Optional literal text to find first, then read surrounding lines instead of a fixed line range.",
            "context_lines": "Number of lines before and after around_text to include when around_text is provided.",
            "max_chars": "Maximum number of characters to return from the artifact read.",
        }),
        _set_param_descriptions(grep_research_artifacts, {
            "pattern": "Regular expression to search for inside saved artifacts.",
            "glob": "File glob pattern limiting which artifacts are searched, such as *.txt, **/*.json, or web-pages/*.",
            "case_sensitive": "Whether the regular expression match should be case-sensitive.",
            "context_lines": "Number of surrounding lines to include before and after each matching line.",
            "max_matches": "Maximum number of matching lines to return before truncating.",
            "max_chars": "Maximum number of characters to return from the grep output.",
        }),
        _set_param_descriptions(delete_research_artifact, {
            "path": "Relative artifact path from list_research_artifacts, or an absolute path inside this run's evidence folder. Only delete files you did not actually use or need as evidence.",
        }),
        _set_param_descriptions(register_research_artifact, {
            "path": "Relative artifact path from list_research_artifacts, or an absolute path inside this run's evidence folder. The file must already exist.",
            "source_url": "Original URL or API endpoint used to create this file, if known. For shell downloads, use the URL passed to curl/wget or the dataset/API URL.",
            "provenance": "Short provenance note describing how the file was created, such as 'exec curl <url>' or 'exec analysis output from worldbank json'.",
            "tool": "Tool or method that created the file. Use exec for shell-created files, or the specific source tool name if applicable.",
            "kind": "Artifact family/category such as dataset, raw-html, pdf, extracted-text, calculation, or manual.",
            "label": "Human-readable label for this artifact; defaults to the filename when empty.",
        }),
    ]
    for tool in built[:3]:
        # Repeats are handled by ``once`` above, not by generic result reuse.
        reuse_tool_results(tool, False)
    if readonly:
        # Browse-only subset: list + read + grep (no delete/register). Used to give
        # non-researcher verifier agents safe access to a folder.
        return built[:3]
    return built


def add_research_artifact_tools(tools: list[Any], config: ToolsConfig, *, root: str = "") -> None:
    """Add artifact tools, optionally pinned to one explicit evidence root.

    Researchers normally use the ContextVar-backed root because each worker owns
    its own evidence folder. Orchestrators that inspect a shared master folder
    must pass ``root`` so a child researcher cannot redirect their file tools by
    changing the inherited research context.
    """
    tools.extend(get_research_artifact_tools(ResearchArtifactsTool(config, root=root)))
    # Every administrator and specialist receives the same read-only retrieval
    # surface when the per-run knowledge policy permits it. Imports stay local
    # to avoid a config/artifact/knowledge module cycle at startup.
    from .knowledge_store import add_knowledge_read_tools

    add_knowledge_read_tools(tools, config)


def get_readonly_file_tools_for_root(config: ToolsConfig, root: str) -> list[Any]:
    """Build list/read/grep tools pinned to an explicit folder (no delete/register).

    Lets a non-researcher agent browse a fixed directory — e.g. the shared
    researcher-queue evidence folder that holds every queued research's files.
    """
    return get_research_artifact_tools(ResearchArtifactsTool(config, root=root), readonly=True)
