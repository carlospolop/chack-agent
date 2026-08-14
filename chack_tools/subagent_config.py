from __future__ import annotations

import json
import os
import re
import contextvars
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from time import time
from typing import Any, Callable, List, Mapping
from uuid import uuid4

from .config import ToolsConfig as BaseToolsConfig
from .research_artifacts import research_artifacts_master_root


_ACTIVE_RESEARCHER_RESPONSE_COLLECTOR: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "chack_active_researcher_response_collector",
    default=None,
)


RESEARCHER_COMMON_SYSTEM_PROMPT = """### RESEARCHER SPECIALIZATION
You are an expert objective researcher. You will be asked to research a topic, and you must use the available tools as many times as needed to find all relevant data about that topic.
You are objective and only looking for the real truth. Treat social norms and common sense as hypotheses to test, never as proof.

- Stay source-first. Treat search results, snippets, summaries, model-generated answers, and social claims as discovery leads until supported by inspectable evidence.
    - Be careful with duplicate sources: if five sources repeat the same hypothesis but all trace back to one study, count that as one underlying source.
- Do not use common sense, social normality, reputational assumptions, or institutional familiarity as filters for what deserves investigation. Use them only as hypotheses to test against evidence.
- Preserve the full evidentiary trail whenever tooling allows it. Search/list tools are discovery aids; content/detail/fetch/download/transcript tools should create inspectable artifacts while the run is active.
- Prefer primary, original, or directly inspectable sources. When using secondary sources, label them as such and keep their provenance.
- Clearly separate observed facts, source claims, inferences, uncertainty, contradictions, and missing evidence.
- The final answer has two layers. `overall_summary`, `findings`, `gaps`, and `open_topics` are the compact digest the parent normally sees. `full_research_review` is the complete evidence-backed record used for audit and downstream synthesis.
- Do not put source lists or assessment labels in the digest. Each `findings[].claim` names the investigated claim and its `summary` explains what was found, how it affects that claim, and any material contradiction or uncertainty. Keep citations, URLs, source provenance, and detailed reasoning in `full_research_review` and preserved artifacts.
- `gaps` are missing evidence that limits the current conclusion. `open_topics` are optional, concrete follow-up investigations that could add value but are not required to close the current claim. Do not repeat a gap as an open topic, and return an empty list when there is no worthwhile follow-up.
- Digest limits are strict: `failure_reason` <= 500 characters; `overall_summary` <= 1000; at most 8 findings; each claim 30-220 and each finding summary 100-600; at most 5 gaps of 20-240 characters each; at most 5 open topics of 30-250 characters each.
- Strong evidence survives serious attempts to disprove it. Actively look for disconfirming evidence, opposing sources, methodological weaknesses, and alternative explanations before concluding.
- Return only the configured JSON output object. Never omit relevant evidence from `full_research_review` merely to make the digest shorter. When research succeeds, make `full_research_review` at least 2000 characters if the evidence reasonably supports that much detail.
"""


RESEARCHER_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "research_worked": {
            "type": "boolean",
            "description": "True when the delegated research completed enough to provide a useful evidence-backed review; false when blocked or failed.",
        },
        "failure_reason": {
            "type": "string",
            "maxLength": 500,
            "description": "Empty when research_worked is true. If false, explain the blocker or failure clearly in at most 500 characters.",
        },
        "overall_summary": {
            "type": "string",
            "maxLength": 1000,
            "description": "Compact overall conclusion, at most 1000 characters. Do not repeat every finding.",
        },
        "findings": {
            "type": "array",
            "maxItems": 8,
            "description": "Up to 8 self-contained findings. No source arrays and no assessment labels.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim": {
                        "type": "string",
                        "minLength": 30,
                        "maxLength": 220,
                        "description": "30-220 characters naming the concrete claim investigated.",
                    },
                    "summary": {
                        "type": "string",
                        "minLength": 100,
                        "maxLength": 600,
                        "description": "100-600 characters explaining what was found, how it affects the claim, and material caveats or contradictions.",
                    },
                },
                "required": ["claim", "summary"],
            },
        },
        "gaps": {
            "type": "array",
            "maxItems": 5,
            "description": "Up to 5 unresolved evidence gaps. Empty when there are no material gaps.",
            "items": {
                "type": "string",
                "minLength": 20,
                "maxLength": 240,
            },
        },
        "open_topics": {
            "type": "array",
            "maxItems": 5,
            "description": "Up to 5 concrete, non-duplicative follow-up investigations that could add value beyond the current conclusion. Empty when no further research is worthwhile.",
            "items": {
                "type": "string",
                "minLength": 30,
                "maxLength": 250,
            },
        },
        "full_research_review": {
            "type": "string",
            "description": "The complete evidence-backed review, including citations, URLs, provenance, contradictions, uncertainty, and detailed reasoning. Never shorten this to satisfy digest limits.",
        },
        "evidence_data_path": {
            "type": "string",
            "description": "Absolute local path to the evidence directory when artifacts were preserved; empty string when the run was configured to delete temporary artifacts.",
        },
        "key_artifacts": {
            "type": "array",
            "description": "Every preserved evidence file that remains useful for the review. If a saved file was not useful, delete it before finalizing instead of omitting it.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Saved artifact filename relative to evidence_data_path. Do not use an absolute path.",
                    },
                    "source_url": {
                        "type": "string",
                        "description": "Original source URL or API/query provenance, if available.",
                    },
                    "description": {
                        "type": "string",
                        "minLength": 100,
                        "maxLength": 300,
                        "description": "100-300 characters explaining what evidence this file contains and how it was used in the review.",
                    },
                },
                "required": ["filename", "source_url", "description"],
            },
        },
    },
    "required": [
        "research_worked",
        "failure_reason",
        "overall_summary",
        "findings",
        "gaps",
        "open_topics",
        "full_research_review",
        "evidence_data_path",
        "key_artifacts",
    ],
}


RESEARCHER_OUTPUT_SCHEMA_NO_ARTIFACTS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "research_worked": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["research_worked"]),
        "failure_reason": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["failure_reason"]),
        "overall_summary": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["overall_summary"]),
        "findings": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["findings"]),
        "gaps": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["gaps"]),
        "open_topics": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["open_topics"]),
        "full_research_review": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["full_research_review"]),
    },
    "required": [
        "research_worked",
        "failure_reason",
        "overall_summary",
        "findings",
        "gaps",
        "open_topics",
        "full_research_review",
    ],
}


ARTIFACT_RECONCILIATION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "key_artifacts": {
            "type": "array",
            "description": "Only artifact records requested by the reconciliation prompt. Include missing files and replacements for listed files with invalid metadata.",
            "items": deepcopy(RESEARCHER_OUTPUT_SCHEMA["properties"]["key_artifacts"]["items"]),
        },
        "delete_artifacts": {
            "type": "array",
            "description": "Relative filenames of saved artifacts that were not useful and should be deleted from the evidence folder.",
            "items": {"type": "string"},
        },
    },
    "required": ["key_artifacts", "delete_artifacts"],
}


def researcher_output_schema(*, preserve_artifacts: bool) -> dict[str, Any]:
    return deepcopy(RESEARCHER_OUTPUT_SCHEMA if preserve_artifacts else RESEARCHER_OUTPUT_SCHEMA_NO_ARTIFACTS)


def _compact_counter(counter: Mapping[str, Any] | None) -> dict[str, int]:
    rows: dict[str, int] = {}
    for key, value in (counter or {}).items():
        name = str(key or "").strip()
        if not name:
            continue
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            rows[name] = rows.get(name, 0) + count
    return dict(sorted(rows.items()))


def _json_dumps_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _digest_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _legacy_digest_finding(full_review: str) -> list[dict[str, str]]:
    excerpt = _digest_text(full_review, 470)
    if not excerpt:
        return []
    first_sentence = re.split(r"(?<=[.!?])\s+", excerpt, maxsplit=1)[0].strip()
    claim = _digest_text(first_sentence, 220)
    if len(claim) < 30:
        claim = "The delegated researcher returned a substantive result"
    summary = _digest_text(
        "The preserved full researcher review reports the following finding: "
        f"{excerpt} Consult the complete review and artifacts for its evidence, citations, and caveats.",
        600,
    )
    return [{"claim": claim, "summary": summary}]


def normalize_researcher_response_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical two-layer researcher response.

    Older researchers used ``final_research_review`` as their only semantic field.
    It is promoted losslessly to ``full_research_review`` and a bounded compatibility
    digest is derived. The full review is never truncated; only digest fields are.
    """

    row = deepcopy(dict(payload or {}))
    legacy_full = row.pop("final_research_review", None)
    full_review = str(row.get("full_research_review") or legacy_full or "")
    row["full_research_review"] = full_review
    row["research_worked"] = row.get("research_worked") is True
    row["failure_reason"] = _digest_text(row.get("failure_reason"), 500)

    overall_summary = _digest_text(row.get("overall_summary"), 1000)
    if not overall_summary and full_review:
        overall_summary = _digest_text(full_review, 1000)
    row["overall_summary"] = overall_summary

    findings: list[dict[str, str]] = []
    raw_findings = row.get("findings")
    if isinstance(raw_findings, list):
        for raw_finding in raw_findings[:8]:
            if not isinstance(raw_finding, Mapping):
                continue
            claim = _digest_text(raw_finding.get("claim"), 220)
            summary = _digest_text(raw_finding.get("summary"), 600)
            if not claim or not summary:
                continue
            if len(claim) < 30:
                claim = _digest_text(f"Investigated claim reported by the researcher: {claim}", 220)
            if len(summary) < 100:
                summary = _digest_text(
                    f"The researcher found the following about this claim: {summary} "
                    "The complete review retains the supporting evidence and any additional caveats.",
                    600,
                )
            findings.append({"claim": claim, "summary": summary})
    if not findings and full_review:
        findings = _legacy_digest_finding(full_review)
    row["findings"] = findings

    gaps: list[str] = []
    raw_gaps = row.get("gaps")
    if isinstance(raw_gaps, list):
        for raw_gap in raw_gaps[:5]:
            gap = _digest_text(raw_gap, 240)
            if not gap:
                continue
            if len(gap) < 20:
                gap = _digest_text(f"Unresolved evidence gap: {gap}", 240)
            gaps.append(gap)
    row["gaps"] = gaps

    open_topics: list[str] = []
    gap_markers = {gap.casefold() for gap in gaps}
    seen_topics: set[str] = set()
    raw_open_topics = row.get("open_topics")
    if isinstance(raw_open_topics, list):
        for raw_topic in raw_open_topics:
            topic = _digest_text(raw_topic, 250)
            if not topic or topic.casefold() in gap_markers:
                continue
            if len(topic) < 30:
                topic = _digest_text(f"Suggested follow-up investigation: {topic}", 250)
            marker = topic.casefold()
            if marker in seen_topics:
                continue
            seen_topics.add(marker)
            open_topics.append(topic)
            if len(open_topics) >= 5:
                break
    row["open_topics"] = open_topics
    return row


def compact_researcher_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a full researcher response into the bounded LLM-facing digest."""

    row = normalize_researcher_response_payload(payload)
    digest: dict[str, Any] = {
        "research_worked": row["research_worked"],
        "failure_reason": row["failure_reason"],
        "overall_summary": row["overall_summary"],
        "findings": deepcopy(row["findings"]),
        "gaps": list(row["gaps"]),
        "open_topics": list(row["open_topics"]),
    }
    researcher_tool = str(row.get("researcher_tool") or "").strip()
    if researcher_tool:
        digest["researcher_tool"] = researcher_tool
    return digest


def append_research_tool_usage(output: str, tool_counts: Mapping[str, Any] | None) -> str:
    """Append tool-call counts to a researcher JSON result without asking the model to write them."""
    payload = _json_from_research_output(output)
    if payload is None:
        return output
    payload = normalize_researcher_response_payload(payload)
    counts = _compact_counter(tool_counts)
    payload["tool_call_counts"] = counts
    payload["total_tool_calls"] = int(sum(counts.values()))
    return _json_dumps_compact(payload)


def begin_researcher_response_collection():
    collector: list[dict[str, Any]] = []
    token = _ACTIVE_RESEARCHER_RESPONSE_COLLECTOR.set(collector)
    return token, collector


def end_researcher_response_collection(token) -> None:
    _ACTIVE_RESEARCHER_RESPONSE_COLLECTOR.reset(token)


def record_researcher_response(researcher_tool: str, output: str) -> None:
    collector = _ACTIVE_RESEARCHER_RESPONSE_COLLECTOR.get()
    if collector is None:
        return
    responses = researcher_responses_from_output(researcher_tool, output)
    if responses:
        collector.extend(responses)
        return
    collector.append(
        {
            "research_worked": False,
            "failure_reason": "Researcher did not return parseable JSON.",
            "overall_summary": "The researcher returned unparseable output; the full response was preserved for inspection.",
            "findings": [],
            "gaps": ["The researcher response could not be parsed into the configured structured output."],
            "open_topics": [],
            "full_research_review": str(output or "").strip(),
            "researcher_tool": str(researcher_tool or "").strip(),
        }
    )


def researcher_responses_from_output(researcher_tool: str, output: Any) -> list[dict[str, Any]]:
    text = str(output or "").strip()
    if not text:
        return []
    payload = _json_from_research_output(output)
    if payload is not None:
        if "result" in payload and not any(
            key in payload
            for key in (
                "research_worked",
                "failure_reason",
                "final_research_review",
                "overall_summary",
                "findings",
                "gaps",
                "open_topics",
                "full_research_review",
                "evidence_data_path",
                "key_artifacts",
                "tool_call_counts",
            )
        ):
            inner = _json_from_research_output(payload.get("result"))
            if inner is None:
                return []
            payload = inner
        payload = normalize_researcher_response_payload(payload)
        payload.setdefault("researcher_tool", str(researcher_tool or "").strip())
        return [payload]

    responses: list[dict[str, Any]] = []
    matches = list(
        re.finditer(
            r"SUBAGENT_RESULT_(\d+):\s*(.*?)(?=\n\s*SUBAGENT_RESULT_\d+:|\Z)",
            text,
            flags=re.DOTALL,
        )
    )
    for match in matches:
        parsed = _json_from_research_output(match.group(2))
        if parsed is None:
            continue
        parsed = normalize_researcher_response_payload(parsed)
        parsed.setdefault("researcher_tool", str(researcher_tool or "").strip())
        parsed.setdefault("batch_result_index", int(match.group(1)))
        responses.append(parsed)
    return responses


def researcher_response_from_output(researcher_tool: str, output: Any) -> dict[str, Any] | None:
    responses = researcher_responses_from_output(researcher_tool, output)
    return responses[0] if responses else None


def aggregate_tool_call_counts(responses: list[dict[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for response in responses:
        if not isinstance(response, dict):
            continue
        counts = response.get("tool_call_counts")
        if not isinstance(counts, dict):
            continue
        total.update(_compact_counter(counts))
    return dict(sorted((name, int(count)) for name, count in total.items() if int(count) > 0))


OBJECTIVE_EVIDENCE_RULES = """### OBJECTIVITY AND EVIDENCE RULES
- Stay strictly objective and use critical thinking. Do not dismiss, down-rank, or ignore a lead because it feels unlikely, socially abnormal, counterintuitive, fringe, embarrassing, or "impossible".
- Give every relevant claim a chance until the evidence shows otherwise. Treat common sense and social normality as hypotheses, not filters.
- Prioritize only by evidence quality: primary sources, reproducible artifacts, direct observations, timestamps, provenance, and corroboration. Never prioritize a source type, institution, community, or narrative without proof.
- Preserve surprising or contradictory evidence. If a lead is weak, label it as weak; do not erase it.
- Download or save every relevant content/detail source artifact you rely on when tooling allows it: PDFs, rendered pages, HTML, text extracts, JSON/CSV data, logs, screenshots, transcripts, and command outputs. Search/list outputs do not need durable files unless they are the primary evidence.
- Use the `CHACK_RESEARCH_DATA_DIR` environment variable as the root evidence directory when it is available. Keep downloaded or generated artifacts there, organized in subdirectories if useful.
- Use the artifact list/read/grep tools to inspect saved evidence before concluding.
- If you use command execution or another non-artifact-aware method to create/download a file, call `register_research_artifact` for each useful file with its source URL/provenance so the manifest can be audited.
- When artifacts are preserved and the configured schema includes `key_artifacts`, your final `key_artifacts` must account for every useful file left in the evidence directory, using filenames relative to `evidence_data_path` rather than full paths. If a file was downloaded/generated but was not actually useful, delete it with `delete_research_artifact` before finalizing instead of leaving it unlisted. For each retained file, include source_url/provenance and a 100-300 character description explaining what evidence the file contains and how you used it.
- Your final JSON must include evidence directory/artifact metadata only when the configured schema asks for those fields. If artifacts are temporary for this run, do not spend output on file paths, source URLs, or artifact descriptions.
- You have access to several tools, use all them as much as you need to find all the useful data. Even if they don't look useful for something give them a chance and see if you find unexpected data in unexpected places.
"""


def researcher_system_prompt(specific_prompt: str) -> str:
    return f"{RESEARCHER_COMMON_SYSTEM_PROMPT.rstrip()}\n\n{str(specific_prompt or '').strip()}"


def _safe_path_part(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text or fallback


def research_master_dir() -> str:
    """Root evidence folder created by a running researcher_administrator.

    When set, every researcher launched underneath the administrator groups its
    downloads under ``<master>/<kind>`` so several researchers of the same type
    share one folder and can see what the others already found.
    """
    return research_artifacts_master_root()


def create_research_master_dir(session_id: str = "") -> str:
    """Create the top-level master evidence folder owned by an administrator run."""
    safe_session = _safe_path_part(session_id, "session")
    path = os.path.join(
        "/tmp",
        "chack-research-data",
        safe_session,
        f"administrator-{int(time() * 1000)}-{uuid4().hex[:8]}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def create_subagent_evidence_dir(kind: str, session_id: str = "") -> str:
    safe_kind = _safe_path_part(kind, "subagent")
    master = research_master_dir()
    if master:
        # Under a researcher_administrator run, all researchers of the same type
        # share one per-type folder so siblings see each other's downloads.
        path = os.path.join(master, safe_kind)
        os.makedirs(path, exist_ok=True)
        return path
    safe_session = _safe_path_part(session_id, "session")
    path = os.path.join(
        "/tmp",
        "chack-research-data",
        safe_session,
        f"{safe_kind}-{int(time() * 1000)}-{uuid4().hex[:8]}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def create_subagent_session_id(kind: str, parent_session_id: str = "") -> str:
    """Create an isolated backend session id for one delegated researcher run.

    Retry-hard and artifact-reconciliation followups inside that researcher reuse
    this id, but sibling researchers must not share it or they can leak context,
    tools, or artifacts across specialist boundaries.
    """
    safe_kind = _safe_path_part(kind, "subagent")
    safe_parent = _safe_path_part(parent_session_id, "session")
    return f"{safe_parent}:{safe_kind}:{int(time() * 1000)}:{uuid4().hex[:8]}"


def append_evidence_dir_instruction(
    prompt: str,
    evidence_dir: str,
    start_sentence: str,
    *,
    save_artifacts: bool = False,
    request_artifact_metadata: bool | None = None,
) -> str:
    master = research_master_dir()
    # A master folder still preserves files during an administrator run so
    # sibling researchers can inspect them. Final artifact metadata is only
    # requested when save_artifacts is true.
    effective_save = bool(save_artifacts)
    metadata_requested = bool(effective_save if request_artifact_metadata is None else request_artifact_metadata)
    if effective_save and metadata_requested:
        persistence = "The caller requested preserved evidence files. Keep all important content/detail artifacts in this directory and return the directory in `evidence_data_path` with key artifact filenames in `key_artifacts`."
    elif effective_save:
        persistence = "The caller requested preserved evidence files. Keep all important content/detail artifacts in this directory, but do not include artifact metadata in the final JSON unless the configured schema asks for it."
    else:
        persistence = "This evidence directory is temporary and will be deleted after the run. Use it during the run to inspect artifacts, but do not include evidence paths, artifact source URLs, or artifact descriptions in the final JSON."
    shared_note = ""
    if master:
        shared_note = (
            "You are running under a research administrator. This evidence folder is shared by every "
            "researcher of your type for this run, so it may already contain artifacts downloaded by "
            "sibling researchers. Inspect the existing files with the artifact list/read/grep tools before "
            "duplicating work, and keep adding any new artifacts you collect here.\n"
        )
    preserved_finalization = (
        (
            "Before finalizing with preserved artifacts, ensure every useful file left in the directory appears in `key_artifacts` with filename relative to evidence_data_path, source_url/provenance, and description. "
            "If a file was saved but not useful, delete it with `delete_research_artifact` rather than leaving unlisted evidence behind.\n"
        )
        if metadata_requested
        else ""
    )
    return (
        f"{str(prompt or '').rstrip()}\n\n"
        "### Evidence collection\n"
        f"Use this evidence data path for this delegated run: {evidence_dir}\n"
        f"{shared_note}"
        "Content/detail/fetch/download/transcript tools should save relevant source artifacts there when tooling allows it. "
        "Do not base final claims on memory, search snippets, or bare result lists when a content/detail/fetch/download/transcript tool can retrieve the underlying source; fetch the underlying source first and inspect the saved artifact. "
        "Search/list tools are discovery aids and do not need durable files unless the search response itself is primary evidence. "
        "Use the artifact list/read/grep tools to inspect files in that directory. "
        "If you create/download files with exec or any non-artifact-aware tool, call register_research_artifact for each useful file with source URL/provenance.\n"
        f"{preserved_finalization}"
        f"{persistence}\n\n"
        f"{start_sentence}"
    )


def _json_from_research_output(output: str) -> dict[str, Any] | None:
    text = str(output or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _relative_evidence_files(evidence_dir: str) -> list[tuple[str, int]]:
    from .research_artifacts import ARTIFACT_MANIFEST_FILENAME

    root = Path(str(evidence_dir or "")).expanduser()
    if not root.is_dir():
        return []
    resolved_root = root.resolve()
    rows: list[tuple[str, int]] = []
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(resolved_root))
        except ValueError:
            continue
        if rel == ARTIFACT_MANIFEST_FILENAME:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        rows.append((rel, int(size)))
    return rows


def _covered_artifact_files(payload: dict[str, Any], evidence_dir: str) -> set[str]:
    root = Path(str(evidence_dir or "")).expanduser().resolve()
    covered: set[str] = set()
    items = payload.get("key_artifacts")
    if not isinstance(items, list):
        return covered
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("filename") or item.get("path") or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            covered.add(str(resolved.relative_to(root)))
        elif resolved.is_dir():
            for path in resolved.rglob("*"):
                if path.is_file():
                    try:
                        covered.add(str(path.resolve().relative_to(root)))
                    except (OSError, ValueError):
                        continue
    return covered


def _fill_artifact_sources(payload: dict[str, Any], evidence_dir: str) -> dict[str, Any]:
    from .research_artifacts import research_artifact_manifest

    metadata = research_artifact_manifest(evidence_dir)
    if not metadata:
        return payload
    items = payload.get("key_artifacts")
    if not isinstance(items, list):
        return payload
    root = Path(str(evidence_dir or "")).expanduser().resolve()
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_url") or "").strip():
            continue
        raw_path = str(item.get("filename") or item.get("path") or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            rel = str(candidate.resolve().relative_to(root))
        except (OSError, ValueError):
            rel = raw_path
        meta = metadata.get(rel) or {}
        source = str(meta.get("source_url") or meta.get("provenance") or "").strip()
        if source:
            item["source_url"] = source
    return payload


def _artifact_metadata_issues(payload: dict[str, Any], evidence_dir: str) -> list[str]:
    root = Path(str(evidence_dir or "")).expanduser().resolve()
    rows: list[str] = []
    items = payload.get("key_artifacts")
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("filename") or item.get("path") or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            rel = str(candidate.resolve().relative_to(root))
        except (OSError, ValueError):
            rel = raw_path
        description = " ".join(str(item.get("description") or "").split())
        source = str(item.get("source_url") or "").strip()
        if not source:
            rows.append(f"- {rel}: source_url/provenance is empty.")
        if len(description) < 100 or len(description) > 300:
            rows.append(f"- {rel}: description is {len(description)} characters; required 100-300.")
    return rows


def _json_with_filled_artifact_sources(output: str, evidence_dir: str) -> str:
    payload = _json_from_research_output(output)
    if payload is None:
        return output
    filled = _fill_artifact_sources(payload, evidence_dir)
    return _json_dumps_compact(filled)


def _delete_artifact_file(evidence_dir: str, filename: str) -> bool:
    root = Path(str(evidence_dir or "")).expanduser().resolve()
    raw = str(filename or "").strip()
    if not raw:
        return False
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return False
    if not resolved.is_file():
        return False
    try:
        resolved.unlink()
        try:
            from .research_artifacts import remove_research_artifact_manifest_entry

            remove_research_artifact_manifest_entry(root, str(resolved.relative_to(root)))
        except Exception:
            pass
        return True
    except OSError:
        return False


def _merge_artifact_reconciliation_patch(
    output: str,
    patch_output: str,
    evidence_dir: str,
) -> str:
    payload = _json_from_research_output(output)
    patch = _json_from_research_output(patch_output)
    if payload is None or patch is None:
        return output
    payload = _fill_artifact_sources(payload, evidence_dir)
    root = Path(str(evidence_dir or "")).expanduser().resolve()
    delete_set: set[str] = set()
    for raw in patch.get("delete_artifacts") or []:
        raw_text = str(raw or "").strip()
        if not raw_text:
            continue
        candidate = Path(raw_text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            rel = str(candidate.resolve().relative_to(root))
        except (OSError, ValueError):
            rel = raw_text
        if _delete_artifact_file(evidence_dir, rel):
            delete_set.add(rel)
    existing: dict[str, dict[str, Any]] = {}
    for item in payload.get("key_artifacts") or []:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or item.get("path") or "").strip()
        if filename and filename not in delete_set:
            existing[filename] = dict(item)
    patch_records = patch.get("key_artifacts") if isinstance(patch.get("key_artifacts"), list) else []
    for item in patch_records:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or item.get("path") or "").strip()
        if not filename or filename in delete_set:
            continue
        record = {
            "filename": filename,
            "source_url": str(item.get("source_url") or "").strip(),
            "description": " ".join(str(item.get("description") or "").split()),
        }
        existing[filename] = record
    payload["key_artifacts"] = list(existing.values())
    return _json_with_filled_artifact_sources(_json_dumps_compact(payload), evidence_dir)


def artifact_reconciliation_prompt(
    *,
    output: str,
    evidence_dir: str,
    max_missing_files: int = 300,
) -> str:
    payload = _json_from_research_output(output)
    if payload is None:
        return ""
    payload = _fill_artifact_sources(payload, evidence_dir)
    files = _relative_evidence_files(evidence_dir)
    if not files:
        return ""
    covered = _covered_artifact_files(payload, evidence_dir)
    missing = [(rel, size) for rel, size in files if rel not in covered]
    metadata_issues = _artifact_metadata_issues(payload, evidence_dir)
    if not missing and not metadata_issues:
        return ""
    from .research_artifacts import ARTIFACT_MANIFEST_FILENAME, research_artifact_manifest

    metadata = research_artifact_manifest(evidence_dir)
    limit = max(1, int(max_missing_files or 300))
    shown = missing[:limit]
    omitted = len(missing) - len(shown)
    missing_rows: list[str] = []
    for rel, size in shown:
        meta = metadata.get(rel) or {}
        source = str(meta.get("source_url") or meta.get("provenance") or "").strip()
        tool = str(meta.get("tool") or meta.get("kind") or "").strip()
        hint = ""
        if source and tool:
            hint = f" source/provenance={source}; tool={tool}"
        elif source:
            hint = f" source/provenance={source}"
        elif tool:
            hint = f" tool={tool}"
        missing_rows.append(f"- {rel} ({size} bytes){hint}")
    missing_lines = "\n".join(missing_rows)
    if omitted > 0:
        missing_lines += f"\n- ... {omitted} additional unlisted files; use list_research_artifacts to inspect them."
    issue_lines = "\n".join(metadata_issues)
    if not missing_lines:
        missing_lines = "- No missing files; fix the metadata issues below."
    issues_block = (
        f"These listed artifacts have invalid metadata:\n{issue_lines}\n\n"
        if issue_lines
        else ""
    )
    return (
        "Continue the same researcher session. Do not redo broad research.\n\n"
        "### Artifact reconciliation required\n"
        f"Evidence directory: {evidence_dir}\n"
        f"The directory currently contains {len(files)} file(s), but the previous final JSON accounts for "
        f"{len(covered)} file(s) in `key_artifacts`.\n\n"
        f"Ignore `{ARTIFACT_MANIFEST_FILENAME}` if present; it is runtime metadata, not evidence to list.\n"
        "These saved files are not accounted for:\n"
        f"{missing_lines}\n\n"
        f"{issues_block}"
        "For each unaccounted file or invalid artifact metadata entry, do exactly one of these:\n"
        "1. If it was useful evidence, inspect it as needed and include it in `key_artifacts` with `filename` relative to the evidence directory, source_url/provenance, and a 100-300 character description explaining what the file contains and how it was used. Use the source/provenance hints above when present. Do not include full file paths.\n"
        "2. If it was not actually useful for the research, put its relative filename in `delete_artifacts`; the runtime will delete it. You may also call `delete_research_artifact` if you need to inspect/delete immediately.\n\n"
        "Return only the artifact reconciliation JSON object requested by the schema: `key_artifacts` for the missing/replacement artifact records and `delete_artifacts` for useless files. "
        "Do not repeat the full research review, do not rewrite the complete researcher result, and do not include files that already had valid metadata."
    )


def reconcile_research_artifacts(
    output: str,
    *,
    evidence_dir: str,
    save_artifacts: bool,
    run_followup: Callable[[str], str],
) -> str:
    if not save_artifacts:
        return output
    try:
        from .research_artifacts import register_untracked_research_artifacts

        register_untracked_research_artifacts(evidence_dir)
    except Exception:
        pass
    output = _json_with_filled_artifact_sources(output, evidence_dir)
    prompt = artifact_reconciliation_prompt(output=output, evidence_dir=evidence_dir)
    if not prompt:
        return output
    try:
        try:
            revised = run_followup(prompt, output_schema_json=ARTIFACT_RECONCILIATION_OUTPUT_SCHEMA)
        except TypeError:
            revised = run_followup(prompt)
    except Exception:
        return output
    revised_text = str(revised or "").strip() or output
    return _merge_artifact_reconciliation_patch(output, revised_text, evidence_dir)


def enforce_prompt_str_or_list_schema(tool: Any) -> Any:
    """Make prompt schema OpenAI-compatible while allowing string or list input."""
    schema = getattr(tool, "params_json_schema", None)
    if not isinstance(schema, dict):
        return tool
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return tool
    prompt_schema = properties.get("prompt")
    if not isinstance(prompt_schema, dict):
        return tool
    prompt_schema.pop("anyOf", None)
    prompt_schema["type"] = ["string", "array"]
    prompt_schema["items"] = {"type": "string"}
    return tool


def _build_tools_config(base: BaseToolsConfig, overrides: Mapping[str, Any] | None) -> BaseToolsConfig:
    allowed = set(getattr(BaseToolsConfig, "__dataclass_fields__", {}).keys())
    data = {k: v for k, v in dict(base.__dict__).items() if k in allowed}
    for key, value in (overrides or {}).items():
        if key in data:
            data[key] = value
    return BaseToolsConfig(**data)


def _scaled_limit_int(value: float, ratio: float, minimum: int) -> int:
    raw = max(0.0, float(value or 0.0))
    if raw <= 0.0:
        return 0
    return max(minimum, int(raw * ratio))


def _scaled_limit_float(value: float, ratio: float) -> float:
    raw = max(0.0, float(value or 0.0))
    if raw <= 0.0:
        return 0.0
    return raw * ratio


def inherit_subagent_limits(
    *,
    default_max_turns: int,
    parent_max_turns: int,
    parent_remaining_runtime_minutes: float,
    parent_remaining_cost_usd: float,
    runtime_ratio: float = 1.0 / 3.0,
    runtime_cap_minutes: int = 20,
    cost_ratio: float = 1.0 / 3.0,
) -> tuple[int, int, float]:
    # Child turns cap: 1/2 of parent max turns.
    parent_turns_cap = _scaled_limit_int(parent_max_turns, 0.5, minimum=2)
    effective_max_turns = max(2, int(default_max_turns or 30))
    if parent_turns_cap > 0:
        effective_max_turns = min(effective_max_turns, parent_turns_cap)

    # Child runtime/cost cap: keep enough parent budget for polling,
    # cross-pollination, cancellation, and final synthesis. In-process
    # administrator runs serialize child researchers for artifact isolation, so
    # one child must not consume most of the parent runtime.
    effective_runtime_minutes = _scaled_limit_int(
        parent_remaining_runtime_minutes,
        runtime_ratio,
        minimum=1,
    )
    if effective_runtime_minutes > 0 and int(runtime_cap_minutes or 0) > 0:
        effective_runtime_minutes = min(effective_runtime_minutes, int(runtime_cap_minutes))
    effective_cost_usd = _scaled_limit_float(parent_remaining_cost_usd, cost_ratio)
    return effective_max_turns, effective_runtime_minutes, effective_cost_usd


def subagent_launch_block_reason(
    *,
    parent_original_runtime_minutes: int,
    parent_remaining_runtime_minutes: float,
    parent_original_cost_usd: float,
    parent_remaining_cost_usd: float,
) -> str | None:
    runtime_limited = max(0, int(parent_original_runtime_minutes or 0)) > 0
    cost_limited = max(0.0, float(parent_original_cost_usd or 0.0)) > 0.0

    if runtime_limited:
        original_runtime = max(0, int(parent_original_runtime_minutes or 0))
        remaining_runtime = max(0.0, float(parent_remaining_runtime_minutes or 0.0))
        runtime_floor = max(10.0, float(original_runtime) / 3.0)
        if remaining_runtime < runtime_floor:
            return (
                "ERROR: cannot launch delegated agent. "
                f"Parent remaining runtime is too low ({remaining_runtime:.2f} min, "
                f"requires at least {runtime_floor:.2f} min) to launch tools that run autonomous agents."
            )

    if cost_limited:
        original_cost = max(0.0, float(parent_original_cost_usd or 0.0))
        remaining_cost = max(0.0, float(parent_remaining_cost_usd or 0.0))
        cost_floor = max(1.0, original_cost / 3.0)
        if remaining_cost < cost_floor:
            return (
                "ERROR: cannot launch delegated agent. "
                f"Parent remaining budget is too low (${remaining_cost:.4f}, "
                f"requires at least ${cost_floor:.4f}) to launch tools that run autonomous agents."
            )
    return None


def validate_subagent_instruction_length(prompt: str, *, min_chars: int = 500) -> str | None:
    text = str(prompt or "").strip()
    if not text:
        return "INPUT_REJECTED: prompt cannot be empty"
    if len(text) < int(min_chars):
        return (
            "INPUT_REJECTED: delegated sub-agent launch blocked. "
            f"Provide at least {int(min_chars)} characters of detailed instructions "
            f"(received {len(text)})."
            f"Use the extra chars to indicate more details on the goals of the agents, expected example responses/information, or any other relevant data. The more specific you are, the better."
        )
    return None


def normalize_subagent_prompts(
    prompt_input: Any,
    *,
    min_chars: int = 500,
    max_prompts: int = 3,
) -> tuple[List[str], str | None]:
    prompts: List[str]
    if isinstance(prompt_input, list):
        prompts = [str(item or "").strip() for item in prompt_input]
    else:
        prompts = [str(prompt_input or "").strip()]

    prompts = [item for item in prompts if item]
    if not prompts:
        return [], "INPUT_REJECTED: prompt cannot be empty"
    if len(prompts) > int(max_prompts):
        return [], (
            "INPUT_REJECTED: delegated sub-agent launch blocked. "
            f"You can provide at most {int(max_prompts)} prompts."
        )

    for idx, text in enumerate(prompts, start=1):
        guard = validate_subagent_instruction_length(text, min_chars=min_chars)
        if guard:
            return [], f"{guard} (prompt #{idx})"
    return prompts, None


def run_parallel_subagent_prompts(
    prompts: List[str],
    runner: Callable[[str], str],
) -> str:
    if len(prompts) == 1:
        return runner(prompts[0])

    results: dict[int, str] = {}
    # Researcher-specific validators remain the authority on batch size. Most
    # researchers cap calls at three; the brokered ChatGPT tools deliberately
    # allow five because the outbound workstation worker has five isolated
    # browser slots.
    max_workers = min(5, len(prompts))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(runner, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = str(future.result() or "").strip() or "ERROR: empty sub-agent output."
            except Exception as exc:
                results[idx] = f"ERROR: sub-agent batch worker failed ({exc})"

    chunks: List[str] = []
    for idx in range(len(prompts)):
        output = results.get(idx, "ERROR: missing sub-agent output.")
        chunks.append(f"SUBAGENT_RESULT_{idx + 1}:\n{output}")
    return "\n\n".join(chunks)


def build_subagent_config(
    base_tools: BaseToolsConfig,
    *,
    model_name: str,
    model_provider: str,
    max_turns: int,
    system_prompt: str,
    overrides: Mapping[str, Any] | None = None,
) -> ChackConfig:
    from chack_agent import (
        AgentConfig,
        ChackConfig,
        CredentialsConfig,
        LoggingConfig,
        ModelConfig,
        SessionConfig,
        ToolsConfig as AgentToolsConfig,
    )

    def _resolve_alias(name: str, *, provider: str, fallback: str = "") -> str:
        raw = str(name or "").strip() or fallback
        if not raw:
            return ""
        try:
            from chack_agent.model_aliases import resolve_model_alias

            return resolve_model_alias(raw, provider=provider)
        except Exception:
            return raw

    overrides = dict(overrides or {})
    prompt = str(overrides.get("system_prompt") or system_prompt).strip() or system_prompt
    if "### RESEARCHER SPECIALIZATION" not in prompt:
        prompt = researcher_system_prompt(prompt)

    model_overrides = overrides.get("model") or {}
    provider = str(model_overrides.get("provider") or model_provider or "").strip()
    if not provider:
        raise ValueError("model_provider must be defined for sub-agent config")
    model_primary = _resolve_alias(
        str(model_overrides.get("primary") or model_name or "").strip(),
        provider=provider,
    )
    model = ModelConfig(
        primary=model_primary,
        provider=provider,
        max_context_tokens=int(model_overrides.get("max_context_tokens") or 0),
        social_network=_resolve_alias(
            str(model_overrides.get("social_network") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        scientific=_resolve_alias(
            str(model_overrides.get("scientific") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        websearcher=_resolve_alias(
            str(model_overrides.get("websearcher") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        business=_resolve_alias(
            str(model_overrides.get("business") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        product=_resolve_alias(
            str(model_overrides.get("product") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        travel=_resolve_alias(
            str(model_overrides.get("travel") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        legal=_resolve_alias(
            str(model_overrides.get("legal") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        data_statistics=_resolve_alias(
            str(model_overrides.get("data_statistics") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        news_media=_resolve_alias(
            str(model_overrides.get("news_media") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        knowledge_graph=_resolve_alias(
            str(model_overrides.get("knowledge_graph") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        religious=_resolve_alias(
            str(model_overrides.get("religious") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        cli=_resolve_alias(
            str(model_overrides.get("cli") or ""),
            provider=provider,
            fallback="CHEAP_BUT_QUALITY",
        ),
        subchack=_resolve_alias(
            str(model_overrides.get("subchack") or ""),
            provider=provider,
            fallback="",
        ),
        researcher_administrator=_resolve_alias(
            str(model_overrides.get("researcher_administrator") or ""),
            provider=provider,
            fallback="",
        ),
        social_network_max_turns=int(model_overrides.get("social_network_max_turns") or 30),
        scientific_max_turns=int(model_overrides.get("scientific_max_turns") or 30),
        websearcher_max_turns=int(model_overrides.get("websearcher_max_turns") or 30),
        business_max_turns=int(model_overrides.get("business_max_turns") or 30),
        product_max_turns=int(model_overrides.get("product_max_turns") or 30),
        travel_max_turns=int(model_overrides.get("travel_max_turns") or 40),
        legal_max_turns=int(model_overrides.get("legal_max_turns") or 30),
        data_statistics_max_turns=int(model_overrides.get("data_statistics_max_turns") or 30),
        news_media_max_turns=int(model_overrides.get("news_media_max_turns") or 30),
        knowledge_graph_max_turns=int(model_overrides.get("knowledge_graph_max_turns") or 30),
        religious_max_turns=int(model_overrides.get("religious_max_turns") or 30),
        cli_max_turns=int(model_overrides.get("cli_max_turns") or 30),
        subchack_max_turns=int(model_overrides.get("subchack_max_turns") or 30),
        researcher_administrator_max_turns=int(model_overrides.get("researcher_administrator_max_turns") or 100),
    )

    env_overrides = overrides.get("env") or {}
    preserve_artifacts = str(env_overrides.get("CHACK_RESEARCH_SAVE_ARTIFACTS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    default_output_schema = researcher_output_schema(preserve_artifacts=preserve_artifacts)

    agent_overrides = overrides.get("agent") or {}
    sub_action = str(agent_overrides.get("sub_action") or "").strip().lower()
    role_agent_fields = {
        "social": "social_network_agent",
        "social_network": "social_network_agent",
        "scientific": "scientific_agent",
        "webresearcher": "websearcher_agent",
        "websearcher": "websearcher_agent",
        "business": "business_agent",
        "product": "product_agent",
        "travel": "travel_agent",
        "legal": "legal_agent",
        "data_statistics": "data_statistics_agent",
        "news_media": "news_media_agent",
        "knowledge_graph": "knowledge_graph_agent",
        "religious": "religious_agent",
        "cli": "cli_agent",
        "subchack": "subchack_agent",
        "researcher_administrator": "researcher_administrator_agent",
        "researcher_queue_merge": "researcher_queue_agent",
    }
    role_settings = getattr(
        base_tools,
        role_agent_fields.get(sub_action, ""),
        {},
    )
    if not isinstance(role_settings, Mapping):
        role_settings = {}
    from chack_agent.thinking_effort import validate_thinking_effort

    thinking_effort = validate_thinking_effort(
        agent_overrides.get("thinking_effort")
        or role_settings.get("thinking_effort")
        or "high",
        model=model_primary,
        setting="thinking_effort",
    )
    agent = AgentConfig(
        thinking_effort=thinking_effort,
        self_critique_enabled=bool(agent_overrides.get("self_critique_enabled", False)),
        self_critique_rounds=int(agent_overrides.get("self_critique_rounds") or 0),
        max_runtime_minutes=int(agent_overrides.get("max_runtime_minutes") or 0),
        max_cost_usd=float(agent_overrides.get("max_cost_usd") or 0.0),
        compaction_threshold_ratio=float(agent_overrides.get("compaction_threshold_ratio") or 0.75),
        compaction_model=str(agent_overrides.get("compaction_model") or ""),
        main_action=str(agent_overrides.get("main_action") or ""),
        sub_action=str(agent_overrides.get("sub_action") or ""),
        output_schema_json=deepcopy(agent_overrides.get("output_schema_json") or default_output_schema),
        output_schema_name=str(agent_overrides.get("output_schema_name") or "researcher_result"),
        output_schema_strict=bool(agent_overrides.get("output_schema_strict", True)),
    )

    session_overrides = overrides.get("session") or {}
    session = SessionConfig(
        max_turns=int(session_overrides.get("max_turns") or max_turns),
        long_term_memory_enabled=bool(
            session_overrides.get("long_term_memory_enabled", False)
        ),
        long_term_memory_max_chars=int(session_overrides.get("long_term_memory_max_chars") or 0),
        long_term_memory_dir=str(session_overrides.get("long_term_memory_dir") or ""),
        system_prompt="",
    )

    tools = _build_tools_config(base_tools, overrides.get("tools") or {})
    logging_overrides = overrides.get("logging") or {}
    logging = LoggingConfig(level=str(logging_overrides.get("level") or "INFO"))
    env = env_overrides

    return ChackConfig(
        model=model,
        agent=agent,
        session=session,
        tools=tools,
        credentials=CredentialsConfig(),
        logging=logging,
        system_prompt=prompt,
        env=env,
    )
