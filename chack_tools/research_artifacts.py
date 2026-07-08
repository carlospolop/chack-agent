from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import contextvars
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    from agents import function_tool
except ImportError:
    function_tool = None

from .config import ToolsConfig
from .formatting import _truncate
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


class ResearchArtifactsTool:
    def __init__(self, config: ToolsConfig, root: str = ""):
        self.config = config
        # When set, these tools are pinned to this directory regardless of the
        # per-run research context/env. Used to expose the file tools over a fixed
        # folder (e.g. the shared researcher-queue evidence folder for factcheckers).
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
        context_lines: int = 20,
        max_chars: int = 12000,
    ) -> str:
        file_path = self._resolve_inside_root(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if around_text:
            needle = str(around_text)
            lowered = needle.lower()
            matches = [idx for idx, line in enumerate(lines) if lowered in line.lower()]
            if not matches:
                return f"SUCCESS: No matches for '{needle}' in {file_path}"
            context = max(0, min(int(context_lines or 20), 200))
            chunks: list[str] = []
            for match in matches[:20]:
                start = max(0, match - context)
                end = min(len(lines), match + context + 1)
                chunks.append(f"--- lines {start + 1}-{end} ---")
                chunks.extend(f"{idx + 1}: {lines[idx]}" for idx in range(start, end))
            body = "\n".join(chunks)
            return _truncate(f"SUCCESS: Matches in {file_path}\n{body}", max(1, int(max_chars or 12000)))

        total = len(lines)
        start = max(1, int(start_line or 1))
        end = int(end_line or 0)
        if end <= 0 or end > total:
            end = total
        if start > end:
            return f"ERROR: start_line ({start}) cannot be greater than end_line ({end})"
        body = "\n".join(f"{idx + 1}: {lines[idx]}" for idx in range(start - 1, end))
        return _truncate(
            f"SUCCESS: Read {file_path} lines {start}-{end} of {total}\n{body}",
            max(1, int(max_chars or 12000)),
        )

    def grep(
        self,
        pattern: str,
        glob: str = "*",
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_matches: int = 50,
        max_chars: int = 12000,
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
        limit = max(1, min(int(max_matches or 50), 500))
        rows: list[str] = []
        matches = 0
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(root))
            if not fnmatch.fnmatch(rel, file_pattern) and not fnmatch.fnmatch(file_path.name, file_pattern):
                continue
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for idx, line in enumerate(lines):
                if not regex.search(line):
                    continue
                start = max(0, idx - context)
                end = min(len(lines), idx + context + 1)
                for row_idx in range(start, end):
                    marker = ":" if row_idx == idx else "-"
                    rows.append(f"{rel}{marker}{row_idx + 1}: {lines[row_idx]}")
                matches += 1
                if matches >= limit:
                    body = "\n".join(rows)
                    return _truncate(f"SUCCESS: Grep matches (truncated at {limit})\n{body}", max(1, int(max_chars or 12000)))
        if not rows:
            return f"SUCCESS: No grep matches for '{raw_pattern}' in {root}"
        return _truncate("SUCCESS: Grep matches\n" + "\n".join(rows), max(1, int(max_chars or 12000)))

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

    @function_tool(name_override="list_research_artifacts")
    def list_research_artifacts(glob: str = "*", max_results: int = 200) -> str:
        """List files saved in this research run's evidence folder."""
        return run_with_tool_logging(
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
        context_lines: int = 20,
        max_chars: int = 12000,
    ) -> str:
        """Read a saved artifact by relative/absolute path, optionally by line range or around matching text."""
        return run_with_tool_logging(
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
        max_matches: int = 50,
        max_chars: int = 12000,
    ) -> str:
        """Regex-search saved artifacts inside this research run's evidence folder."""
        return run_with_tool_logging(
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
            "end_line": "Last 1-based line number to read; use 0 to read from start_line to the file end or max_chars limit.",
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
    if readonly:
        # Browse-only subset: list + read + grep (no delete/register). Used to give
        # non-researcher agents (e.g. factchecker verifiers) safe access to a folder.
        return built[:3]
    return built


def add_research_artifact_tools(tools: list[Any], config: ToolsConfig) -> None:
    tools.extend(get_research_artifact_tools(ResearchArtifactsTool(config)))


def get_readonly_file_tools_for_root(config: ToolsConfig, root: str) -> list[Any]:
    """Build list/read/grep tools pinned to an explicit folder (no delete/register).

    Lets a non-researcher agent browse a fixed directory — e.g. the shared
    researcher-queue evidence folder that holds every queued research's files.
    """
    return get_research_artifact_tools(ResearchArtifactsTool(config, root=root), readonly=True)
