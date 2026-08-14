from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chack_tools.business_research_agent import BusinessResearchAgentTool
from chack_tools.cli_research_agent import CliResearchAgentTool
from chack_tools.config import ToolsConfig
from chack_tools.product_research_agent import ProductResearchAgentTool
from chack_tools.researcher_administrator_agent import RESEARCHER_REGISTRY
from chack_tools.scientific_research_agent import ScientificResearchAgentTool
from chack_tools.social_network_agent import SocialNetworkAgentTool
from chack_tools.telemetry.context import reset_log_context, set_log_context
from chack_tools.travel_research_agent import TravelResearchAgentTool
from chack_tools.websearcher_agent import WebSearcherAgentTool
from chack_tools.open_research_agents import (
    build_data_statistics_agent,
    build_knowledge_graph_agent,
    build_legal_agent,
    build_news_media_agent,
    build_religious_agent,
)

ORDINARY = [
    name for name in RESEARCHER_REGISTRY
    if name not in {"deepchatgpt", "prochatgpt", "chatgptxhigh"}
]

PROMPTS: dict[str, str] = {
    "scientific": """Investigate the current scientific and translational evidence for solid-state lithium-metal batteries across sulfide, oxide, and polymer electrolytes. Cover dendrite and interface mechanisms, pressure and manufacturing constraints, cycle-life and fast-charge claims, independent peer-reviewed evidence versus preprints, patents, retraction or correction signals, and the gap between laboratory cells and commercial EV deployment. Use several relevant scientific databases, DOI/provenance checks, accessible full text or PDFs, and disconfirming studies. Preserve source-level artifacts. Return a substantive JSON review separating established measurements, disputed claims, limitations, and what future experiment would change confidence.""",
    "business": """Research the current commercial reality of solid-state batteries for EVs and consumer electronics. Identify the legal entities, ownership or partnerships, financing and manufacturing milestones, automaker or supplier commitments, announced production dates, delays, and revenue or capacity claims. Verify company statements against SEC or equivalent filings, investor documents, patents, independent technical reporting, and market data. Resolve similarly named companies and distinguish pilot lines from mass production. Preserve source artifacts and explicitly test bullish claims against skeptical commercial evidence, contradictions, and missing disclosures. Return a detailed evidence-backed JSON review with dates and source provenance.""",
    "product": """Research consumer and industrial product evidence for solid-state battery products and near-term EV cells. Compare named products, battery formats, claimed energy density, cycle life, charging, safety, warranty or recall information, certification or regulatory signals, patent and supplier relationships, and whether a claim refers to a prototype, laboratory cell, pilot product, or available product. Use product-detail, safety/recall, vulnerability or standards, shopping/marketplace, image/video and primary web evidence where relevant; fetch core source pages rather than relying on snippets. Preserve artifacts and identify counterfeit, marketing, identifier, and availability risks. Return a confidence-graded JSON review with contradictions and gaps.""",
    "travel": """Research a realistic two-week rail-first trip through Spain, France, and Italy for the current season. Compare two or three route variants using official rail/operator and government sources first. Check border rules, reservations, strike or disruption risk, accessibility, seasonal weather, night-train or high-speed constraints, realistic transfer times, and sustainable alternatives to short-haul flights. Do not invent live schedules or prices: record exact dates, source timestamps, caveats, and what must be rechecked at booking time. Preserve source artifacts and distinguish official facts, current reports, and planning assumptions. Return substantive JSON findings with route trade-offs and unresolved operational risks.""",
    "websearcher": """Research whether major consumer VPN providers' no-log and privacy claims are supported by inspectable current evidence. Compare independent audits and technical policies with court or law-enforcement disclosures, ownership/entity changes, breach or incident history, app and website telemetry or trackers, pricing and affiliate incentives, and credible critical reporting. Use multiple independent searches, fetch primary pages and archived material when useful, and separate direct evidence from repeated marketing or anecdotal claims. Preserve source/detail artifacts, record dates and URLs, actively seek disconfirming evidence, and return a detailed JSON review with evidence quality, contradictions, and remaining gaps.""",
    "social_network": """Investigate the public social and forum evidence around consumer VPN privacy and no-log claims. Identify recurring user reports, security or privacy allegations, promotion and affiliate patterns, coordinated or duplicated claims, geographic and time differences, and credible counterexamples. Use social/forum/news/video sources only as provenance-labeled evidence, resolve whether posts refer to the same incident, and fetch underlying pages where possible. Preserve artifacts, do not treat popularity as truth, compare anecdotal claims with official audits or technical evidence, and return a substantive JSON review that clearly marks weak signals, corroborated events, contradictions, and gaps.""",
    "legal": """Research the current legal and regulatory framework relevant to AI data-center electricity and water use in Spain and the European Union. Cover permitting, environmental impact assessment, water concessions, grid connection, data-center or AI policy, disclosure and sustainability obligations, municipal and regional competence, enforcement or litigation signals, and exact effective dates. Prefer BOE, EUR-Lex, Commission, regulator, court, and municipal primary documents; download or fetch underlying texts, track jurisdiction and document identifiers, and distinguish binding law from proposals and commentary. Preserve artifacts and return a detailed JSON review with practical implications, conflicts, and missing legal facts.""",
    "data_statistics": """Produce an evidence-backed data investigation of electricity demand, water use, and data-center or AI infrastructure in Spain and the EU. Locate primary datasets and indicators from grid operators, Eurostat, national authorities, water agencies, and credible research; record units, geography, date ranges, definitions, update dates, methodology, and raw values. Use commands when needed to fetch and inspect JSON/CSV and calculate clearly reproducible comparisons. Reconcile incompatible denominators and preserve conflicting datasets instead of smoothing them. Save raw data artifacts and return a detailed JSON review separating measured data, modeled estimates, company projections, and unknowns.""",
    "news_media": """Research the current news and media record about hyperscale or AI data-center projects in Spain and the EU, especially electricity, water, permits, local opposition, grid constraints, and company sustainability claims. Build a dated timeline from original reporting, official announcements, filings, interviews, video or archived pages, and independent follow-up. Detect syndication and duplicated stories, distinguish reported facts from headlines and speculation, and seek sources that contradict the dominant narrative. Preserve source artifacts and return a detailed JSON review with publisher provenance, exact dates, unresolved discrepancies, and evidence quality.""",
    "knowledge_graph": """Resolve the entities and relationships involved in AI/data-center expansion in Spain and the EU: companies, subsidiaries, projects, municipalities, utilities, regulators, investors, suppliers, and public datasets. Use structured entity and graph sources to collect identifiers, aliases, official URLs, locations, ownership or partnership relationships, and provenance, then verify important links against primary pages or filings. Preserve ambiguous candidates rather than guessing, record changed names and dates, and return a substantive JSON review of the resolved graph, conflicts, and missing identifiers.""",
    "religious": """Research how major religious traditions and secular bioethics institutions discuss AI-assisted embryo selection and genetic screening. Retrieve exact primary-text or authoritative passages where available, compare institutional commentary with medical/bioethics evidence, and cover current regulatory or public controversy in the US, EU, and Spain. Include Catholic, Protestant, Jewish, Muslim, Hindu, Buddhist, and secular perspectives only when sources are inspectable; distinguish doctrine, interpretation, clinical evidence, commercial fertility claims, and advocacy. Preserve text/source artifacts and return a detailed JSON review with citations, differences within traditions, contradictions, and gaps.""",
    "cli": """Perform a safe reproducible CLI-backed investigation of current npm package metadata and public advisories for dependency-confusion or typosquatting risk affecting popular AI developer tooling. Use commands to query registries, inspect JSON, compare package names, maintainers, versions, publish dates, downloads, integrity metadata, and known advisories; do not install or execute untrusted packages. Cross-check public security advisories and maintainers' official information with web/entity context. Save command outputs and scripts as artifacts, report exact commands and observed outputs, and return substantive JSON separating command-verified facts, leads, limitations, and safe next checks.""",
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_output(raw: str) -> dict[str, Any]:
    """Extract the terminal researcher object from plain or narrated output.

    Claude may return a valid JSON object after a short narrative or inside a
    fenced code block. A greedy ``\\{.*\\}`` regex cannot parse that reliably:
    it spans prose, nested objects, and multiple candidates. Use the JSON
    decoder's balanced-object parser and prefer the candidate with the
    researcher-result keys.
    """
    text = str(raw or "").strip()
    if not text:
        return {}

    def candidates(value: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        found: list[dict[str, Any]] = []
        for index, char in enumerate(value):
            if char != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(value[index:])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                found.append(parsed)
        return found

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    found = candidates(text)
    if not found:
        return {}
    result_keys = {"research_worked", "findings", "full_research_review"}
    return max(
        found,
        key=lambda value: (
            len(result_keys.intersection(value)),
            value.get("research_worked") is True,
            bool(str(value.get("full_research_review") or "").strip()),
            len(value),
        ),
    )


def artifact_stats(root: Path) -> dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
    return {
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "latest_mtime": max((p.stat().st_mtime for p in files), default=0.0),
    }


def matrix_parent_max_turns(child_max_turns: int) -> int:
    """Give the synthetic matrix parent enough turns for the requested child budget.

    Production subagents intentionally inherit at most half of their parent's
    turn budget.  A direct specialist matrix has no real parent, so using the
    requested child budget as the synthetic parent budget silently halves it.
    """

    return max(4, max(2, int(child_max_turns or 0)) * 2)


def make_config(researcher: str, thinking_effort: str = "high") -> ToolsConfig:
    cfg = ToolsConfig(task_steps_manager_enabled=False, exec_enabled=True, pdf_text_enabled=True)
    # The direct specialist builders use role-specific flags. Enabling the full
    # source surface avoids false passes caused by a missing optional family;
    # budgets below remain finite and the provider controls actual selection.
    for field in fields(cfg):
        if field.name.endswith("_enabled"):
            setattr(cfg, field.name, True)
        elif field.name.endswith("_max_tools_used"):
            setattr(cfg, field.name, 12)
        elif field.name.endswith("_max_results"):
            setattr(cfg, field.name, 5)
    for field_name in (
        "deepchatgpt_enabled",
        "prochatgpt_enabled",
        "chatgptxhigh_enabled",
        "subchack_enabled",
        "parallel_research_enabled",
        "researcher_administrator_enabled",
        "researcher_queue_enabled",
    ):
        if hasattr(cfg, field_name):
            setattr(cfg, field_name, False)
    cfg.max_tools_used = 12
    cfg.exec_timeout_seconds = 60
    cfg.playwright_enabled = False
    cfg.researcher_administrator_researchers = []
    cfg.researcher_administrator_required_researchers = []
    # Specialist builders read the role-specific agent mapping when constructing
    # the real ChackConfig. Keep the matrix's requested reasoning level explicit
    # instead of silently relying on the production default.
    for field_name in (
        "social_network_agent",
        "scientific_agent",
        "websearcher_agent",
        "business_agent",
        "product_agent",
        "travel_agent",
        "legal_agent",
        "data_statistics_agent",
        "news_media_agent",
        "knowledge_graph_agent",
        "religious_agent",
        "cli_agent",
        "subchack_agent",
        "researcher_administrator_agent",
        "researcher_queue_agent",
    ):
        if hasattr(cfg, field_name):
            setattr(cfg, field_name, {"thinking_effort": thinking_effort})
    return cfg


def build_helper(
    researcher: str,
    provider: str,
    model: str,
    max_turns: int,
    thinking_effort: str,
):
    cfg = make_config(researcher, thinking_effort)
    kwargs = dict(
        model_name=model,
        fallback_model=model,
        model_provider=provider,
        max_turns=max_turns,
    )
    if researcher == "scientific":
        return ScientificResearchAgentTool(cfg, **kwargs)
    if researcher == "business":
        return BusinessResearchAgentTool(cfg, **kwargs)
    if researcher == "product":
        return ProductResearchAgentTool(cfg, **kwargs)
    if researcher == "travel":
        return TravelResearchAgentTool(cfg, **kwargs)
    if researcher == "websearcher":
        return WebSearcherAgentTool(cfg, **kwargs)
    if researcher == "social_network":
        return SocialNetworkAgentTool(cfg, **kwargs)
    if researcher == "cli":
        return CliResearchAgentTool(cfg, **kwargs)
    builders = {
        "legal": build_legal_agent,
        "data_statistics": build_data_statistics_agent,
        "news_media": build_news_media_agent,
        "knowledge_graph": build_knowledge_graph_agent,
        "religious": build_religious_agent,
    }
    if researcher in builders:
        return builders[researcher](cfg, **kwargs)
    raise ValueError(f"Unsupported ordinary researcher: {researcher}")


def case_gate(parsed: dict[str, Any], evidence: Path, raw: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not isinstance(parsed.get("research_worked"), bool):
        failures.append("research_worked is not an explicit boolean")
    if parsed.get("research_worked") is not True:
        failures.append("research_worked is not true")
    findings = parsed.get("findings")
    if not isinstance(findings, list) or not findings:
        failures.append("findings are empty")
    review = str(parsed.get("full_research_review") or "").strip()
    if len(review) < 500 or review.lower() == "placeholder":
        failures.append("full_research_review is missing or not substantive")
    if not raw.strip():
        failures.append("raw output is empty")
    if not evidence.is_dir():
        failures.append("evidence directory is missing")
    else:
        stats = artifact_stats(evidence)
        if stats["file_count"] <= 0:
            failures.append("evidence directory has no files")
    return not failures, failures


def child_case(args: argparse.Namespace) -> int:
    researcher = args.case
    case_dir = Path(args.out_dir).expanduser().resolve() / researcher
    evidence_root = case_dir / "evidence"
    case_dir.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    # Specialist researchers create per-kind folders below the administrator
    # root. Setting only CHACK_RESEARCH_DATA_DIR leaves that layout decision on
    # the default /tmp tree, so the harness cannot verify requested artifacts.
    os.environ["CHACK_RESEARCH_MASTER_DIR"] = str(evidence_root)
    os.environ["CHACK_RESEARCH_DATA_DIR"] = str(evidence_root)
    os.environ["CHACK_RESEARCH_SAVE_ARTIFACTS"] = "1"
    if args.provider == "claude":
        # The Hermes gateway uses an isolated HOME. Claude Code's authenticated
        # login is intentionally read from the host user's CLI home instead.
        os.environ["HOME"] = "/home/tester"
        os.environ["USER"] = "tester"
        os.environ["LOGNAME"] = "tester"
    if args.provider == "claude":
        # Claude runs through its authenticated host CLI home. Do not leak a
        # Codex bearer token into that subprocess, but preserve it for the Codex
        # provider below when the harness is exercising GPT researchers.
        os.environ.pop("CODEX_ACCESS_TOKEN", None)
    elif args.provider == "codex":
        # The gateway may export a stale short-lived bearer token while the host
        # Codex login still has a valid refreshable auth.json. Force the matrix
        # to exercise the same refreshable account session instead of generating
        # an isolated auth.json with an empty refresh_token.
        os.environ.pop("CODEX_ACCESS_TOKEN", None)
        os.environ["CODEX_HOME"] = "/home/tester/.codex"
    token = set_log_context(
        session_id=f"matrix-{researcher}-{os.getpid()}",
        max_turns=matrix_parent_max_turns(args.max_turns),
        max_runtime_minutes=args.runtime_minutes,
        remaining_runtime_minutes=args.runtime_minutes,
        max_cost_usd=0,
        remaining_cost_usd=0,
        memory_max_messages=16,
        memory_reset_to_messages=8,
        main_action="researcher_matrix_eval",
        sub_action=researcher,
    )
    started = time.time()
    raw = ""
    error = ""
    try:
        helper = build_helper(
            researcher,
            args.provider,
            args.model,
            args.max_turns,
            args.thinking_effort,
        )
        raw = str(helper.run(PROMPTS[researcher], save_artifacts=True) or "")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raw = f"ERROR: {error}"
    finally:
        reset_log_context(token)
    elapsed = time.time() - started
    parsed = parse_output(raw)
    evidence_path = str(parsed.get("evidence_data_path") or "").strip()
    evidence = Path(evidence_path).expanduser() if evidence_path else evidence_root
    if not evidence.is_dir():
        candidates = sorted(case_dir.rglob("researcher_outputs"))
        if candidates:
            evidence = candidates[0].parent
    passed, failures = case_gate(parsed, evidence, raw)
    summary = {
        "researcher": researcher,
        "tool_name": RESEARCHER_REGISTRY[researcher][1],
        "provider": args.provider,
        "model": args.model,
        "thinking_effort": args.thinking_effort,
        "elapsed_seconds": round(elapsed, 2),
        "terminal": True,
        "error": error,
        "research_worked": parsed.get("research_worked"),
        "failure_reason": parsed.get("failure_reason", ""),
        "findings": len(parsed.get("findings") or []) if isinstance(parsed.get("findings"), list) else 0,
        "review_chars": len(str(parsed.get("full_research_review") or "")),
        "tool_call_counts": parsed.get("tool_call_counts") or {},
        "total_tool_calls": parsed.get("total_tool_calls", 0),
        "evidence_data_path": str(evidence),
        "artifact_stats": artifact_stats(evidence),
        "functional_pass": passed,
        "validation_failures": failures,
    }
    (case_dir / "raw_output.txt").write_text(raw, encoding="utf-8")
    (case_dir / "parsed_output.json").write_text(compact(parsed) + "\n", encoding="utf-8")
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(compact(summary), flush=True)
    return 0 if passed else 1


def proc_snapshot(pid: int, case_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    by_pid: dict[int, dict[str, Any]] = {}
    proc_root = Path("/proc")
    for entry in proc_root.iterdir() if proc_root.is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            raw_stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            close = raw_stat.rfind(")")
            fields_ = raw_stat[close + 2 :].split()
            child_pid = int(entry.name)
            parent = int(fields_[1])
            pgrp = int(fields_[2])
            sid = int(fields_[3])
            state = fields_[0]
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            wchan = (entry / "wchan").read_text(encoding="utf-8", errors="replace").strip()
            row = {"pid": child_pid, "ppid": parent, "pgid": pgrp, "sid": sid, "state": state, "wchan": wchan, "cmd": cmd[:500]}
            by_pid[child_pid] = row
        except (OSError, ValueError, IndexError):
            continue
    owned: set[int] = {pid}
    changed = True
    while changed:
        changed = False
        for child_pid, row in by_pid.items():
            if row["ppid"] in owned and child_pid not in owned:
                owned.add(child_pid)
                changed = True
    records = [by_pid[item] for item in sorted(owned) if item in by_pid]
    (case_dir / "diagnostic_latest.json").write_text(
        json.dumps({"captured_at": time.time(), "root_pid": pid, "processes": records, "artifacts": artifact_stats(case_dir)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"processes": records, "artifacts": artifact_stats(case_dir), "captured_at": time.time()}


def terminate_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=8)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass


def run_one(args: argparse.Namespace, researcher: str, out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / researcher
    case_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        researcher,
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--out-dir",
        str(out_dir),
        "--max-turns",
        str(args.max_turns),
        "--runtime-minutes",
        str(args.runtime_minutes),
        "--thinking-effort",
        args.thinking_effort,
    ]
    log_path = case_dir / "harness.log"
    with log_path.open("w", encoding="utf-8") as log:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        if args.provider == "claude":
            env.update({"HOME": "/home/tester", "USER": "tester", "LOGNAME": "tester"})
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        started = time.monotonic()
        initial = artifact_stats(case_dir)
        deadline = started + args.initial_timeout
        checkpoint = False
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if not checkpoint and elapsed >= 900:
                snap = proc_snapshot(process.pid, case_dir)
                active_descendants = [row for row in snap["processes"] if row["pid"] != process.pid and row["state"] != "Z"]
                growth = snap["artifacts"]["file_count"] > initial["file_count"] or snap["artifacts"]["bytes"] > initial["bytes"]
                checkpoint_payload = {
                    "elapsed_seconds": round(elapsed, 2),
                    "active_descendants": active_descendants,
                    "artifacts_growing": growth,
                    "decision": "continue_active_work" if active_descendants or growth else "terminate_lifecycle_hang",
                }
                (case_dir / "diagnostic_at_15m.json").write_text(json.dumps(checkpoint_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(compact({"checkpoint": researcher, **checkpoint_payload}), flush=True)
                checkpoint = True
                if active_descendants or growth:
                    deadline = started + args.extended_timeout
                else:
                    terminate_group(process)
                    break
            if time.monotonic() >= deadline:
                terminate_group(process)
                break
            time.sleep(5)
    returncode = process.poll()
    summary_path = case_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                summary["harness_exit_code"] = returncode
                return summary
        except Exception:
            pass
    snapshot = proc_snapshot(process.pid, case_dir) if process.poll() is None else None
    summary = {
        "researcher": researcher,
        "provider": args.provider,
        "model": args.model,
        "thinking_effort": args.thinking_effort,
        "terminal": False,
        "functional_pass": False,
        "harness_exit_code": returncode,
        "failure_reason": "case process ended without a valid terminal summary",
        "diagnostic": snapshot,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def resume_summary_matches(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    """Only reuse a pass produced by the exact requested provider configuration."""
    return bool(
        isinstance(summary, dict)
        and summary.get("functional_pass") is True
        and str(summary.get("provider") or "") == str(args.provider)
        and str(summary.get("model") or "") == str(args.model)
        and str(summary.get("thinking_effort") or "") == str(args.thinking_effort)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=ORDINARY)
    parser.add_argument("--provider", default="claude")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--thinking-effort", choices=("high", "max"), default="high")
    parser.add_argument("--out-dir", default="/tmp/chack-live-researcher-matrix-v2")
    parser.add_argument("--max-turns", type=int, default=18)
    parser.add_argument("--runtime-minutes", type=int, default=50)
    parser.add_argument("--initial-timeout", type=int, default=1800)
    parser.add_argument("--extended-timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.case:
        return child_case(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for researcher in ORDINARY:
        summary_path = out_dir / researcher / "summary.json"
        if args.resume and summary_path.is_file():
            try:
                old = json.loads(summary_path.read_text(encoding="utf-8"))
                if resume_summary_matches(old, args):
                    rows.append(old)
                    print(compact({"skip": researcher, "functional_pass": True}), flush=True)
                    continue
            except Exception:
                pass
        print(compact({"start": researcher, "provider": args.provider, "model": args.model}), flush=True)
        row = run_one(args, researcher, out_dir)
        rows.append(row)
        print(compact({"done": researcher, "functional_pass": row.get("functional_pass"), "terminal": row.get("terminal"), "elapsed_seconds": row.get("elapsed_seconds"), "failure_reason": row.get("failure_reason", "")}), flush=True)
        (out_dir / "matrix_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [row for row in rows if row.get("functional_pass") is not True]
    print(compact({"matrix": "ordinary", "cases": len(rows), "passed": len(rows) - len(failures), "failed": len(failures), "out_dir": str(out_dir)}), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
