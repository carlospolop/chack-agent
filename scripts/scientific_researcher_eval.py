from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chack_agent import Chack
from chack_tools.config import ToolsConfig
from chack_tools.open_research_sources import OpenResearchTool
from chack_tools.pdf_text import PdfTextTool
from chack_tools.scientific_research_agent import (
    _SCIENTIFIC_AGENT_SYSTEM_PROMPT,
    ScientificResearchAgentTool,
)
from chack_tools.scientific_search import ScientificSearchTool
from chack_tools.subagent_config import append_evidence_dir_instruction, build_subagent_config


ESSENTIAL_SHARED_TOOL_NAMES = [
    "download_pmc_full_text",
    "download_ncbi_bookshelf",
    "download_medrxiv_full_text",
    "biorxiv_download",
    "download_pdf_as_text",
    "exec",
]
ARTIFACT_TOOL_NAMES = [
    "list_research_artifacts",
    "read_research_artifact",
    "grep_research_artifacts",
]
SOURCE_TOOL_FAMILIES = [
    ("arxiv", {"search_arxiv"}),
    ("europe_pmc", {"search_europe_pmc"}),
    ("pmc", {"search_pmc_full_text"}),
    ("ncbi_bookshelf", {"search_ncbi_bookshelf"}),
    ("semantic_scholar", {"search_semantic_scholar"}),
    ("openalex", {"search_openalex"}),
    ("plos", {"search_plos"}),
    ("google_patents", {"search_google_patents", "search_google_patents_details"}),
    ("google_scholar", {"search_google_scholar", "search_google_scholar_cite"}),
    ("youtube", {"search_youtube_videos", "get_youtube_video_details", "get_youtube_video_transcript"}),
    ("medrxiv_native", {"search_medrxiv_preprints"}),
    ("crossref", {"crossref_search", "crossref_doi_lookup"}),
    ("clinicaltrials", {"clinicaltrials_search", "clinicaltrial_get"}),
    ("biorxiv", {"biorxiv_search"}),
    ("retractions", {"retraction_watch"}),
    ("pubchem", {"pubchem_search"}),
]
ARTIFACT_DIR_TO_TOOL = {
    "pdf-text": "download_pdf_as_text",
    "pmc-full-text": "download_pmc_full_text",
    "ncbi-bookshelf": "download_ncbi_bookshelf",
    "medrxiv-full-text": "download_medrxiv_full_text",
    "medrxiv-pdf": "biorxiv_download",
    "biorxiv-pdf": "biorxiv_download",
    "youtube-transcripts": "get_youtube_video_transcript",
    "crossref-search": "crossref_search",
    "crossref-doi": "crossref_doi_lookup",
    "crossref-retractions": "retraction_watch",
    "clinicaltrials-search": "clinicaltrials_search",
    "clinicaltrials-study": "clinicaltrial_get",
    "biorxiv-search": "biorxiv_search",
    "medrxiv-search": "biorxiv_search",
    "pubchem": "pubchem_search",
}
BENCHMARK_MAX_TURNS = 18
BENCHMARK_MAX_RUNTIME_MINUTES = 12
BENCHMARK_MAX_TOOLS_USED = 24


TOPICS = {
    "microplastics_health": (
        "Research the scientific evidence about microplastics and nanoplastics in humans. Focus on detection in blood, "
        "placenta, lungs, gut, and arterial plaques; contamination and measurement limitations; mechanistic toxicology; "
        "epidemiology or clinical-outcome evidence; and what remains unproven. Use multiple scientific databases, DOI "
        "metadata, retraction/update checks, and download accessible full text or PDFs where possible. Separate peer-reviewed "
        "papers from preprints, reviews from primary studies, and strong evidence from speculative claims."
    ),
    "long_covid_mitochondria": (
        "Research evidence for mitochondrial, metabolic, and bioenergetic dysfunction in long COVID. Include human omics "
        "studies, muscle or exercise physiology papers, immune/metabolic hypotheses, trials or interventions when present, "
        "contradictory findings, and whether proposed mechanisms are causal or correlational. Use paper search, preprints, "
        "clinical-trial tools, DOI/provenance checks, and download accessible full text/PDF evidence."
    ),
    "solid_state_batteries": (
        "Research progress and remaining barriers for solid-state lithium-metal batteries across sulfide, oxide, and polymer "
        "electrolytes. Cover dendrites, interface degradation, pressure and manufacturing constraints, cycle-life claims, "
        "commercialization evidence, relevant patents, and independent reviews. Use scientific search engines, patents, DOI "
        "metadata, accessible papers, and full-text/PDF downloads; explicitly challenge optimistic commercial claims."
    ),
}


def load_env(path: Path) -> None:
    if path.exists():
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    os.environ.pop("CODEX_ACCESS_TOKEN", None)
    os.environ.setdefault("CHACK_DISABLE_STDOUT_EVENTS", "1")
    os.environ.setdefault("CHACK_DISABLE_CODEX_NATIVE_WEB", "1")


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "").strip()


def partition_source_tools(seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    families = list(SOURCE_TOOL_FAMILIES)
    rng.shuffle(families)
    split = (len(families) + 1) // 2
    left = set().union(*(tools for _, tools in families[:split]))
    right = set().union(*(tools for _, tools in families[split:]))
    return left, right


def base_tools_config() -> ToolsConfig:
    return ToolsConfig(
        task_steps_manager_enabled=False,
        exec_enabled=True,
        pdf_text_enabled=True,
        scientific_enabled=True,
        scientific_max_results=6,
        scientific_max_tools_used=BENCHMARK_MAX_TOOLS_USED,
        max_tools_used=BENCHMARK_MAX_TOOLS_USED,
        min_tools_used=0,
    )


def build_tools(provider: str, model: str, selected_sources: set[str] | None) -> list[Any]:
    helper = ScientificResearchAgentTool(
        base_tools_config(),
        model_name=model,
        fallback_model=model,
        model_provider=provider,
        max_turns=BENCHMARK_MAX_TURNS,
    )
    tools = helper._build_subagent_tools()
    if selected_sources is None:
        return tools
    allowed = set(selected_sources) | set(ESSENTIAL_SHARED_TOOL_NAMES) | set(ARTIFACT_TOOL_NAMES)
    return [tool for tool in tools if tool_name(tool) in allowed]


def parse_json_output(output: str) -> dict[str, Any]:
    text = str(output or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def extract_tool_counts(result: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for step in getattr(result, "intermediate_steps", []) or []:
        item = step[0] if isinstance(step, (tuple, list)) and step else step
        name = str(getattr(item, "tool", "") or getattr(item, "name", "") or "")
        if not name and isinstance(item, dict):
            name = str(item.get("tool") or item.get("name") or "")
        if name:
            counts[name] += 1
    return counts


def artifact_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"file_count": 0, "bytes": 0, "dirs": []}
    files = [p for p in path.rglob("*") if p.is_file()]
    dirs = sorted({str(p.parent.relative_to(path)) for p in files if p.parent != path})
    return {
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "dirs": dirs,
    }


def infer_tool_counts_from_artifacts(artifacts: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for dirname in artifacts.get("dirs") or []:
        first = str(dirname).split("/", 1)[0]
        tool = ARTIFACT_DIR_TO_TOOL.get(first)
        if tool:
            counts[tool] += 1
    return counts


def score_run(output: str, parsed: dict[str, Any], counts: Counter[str], artifacts: dict[str, Any], elapsed: float) -> float:
    review = str(
        parsed.get("full_research_review")
        or parsed.get("final_research_review")
        or parsed.get("final_review")
        or output
        or ""
    )
    worked = parsed.get("worked")
    if isinstance(worked, str):
        worked_ok = worked.strip().lower() in {"true", "yes", "1", "worked", "success"}
    else:
        worked_ok = bool(worked) if worked is not None else "ERROR:" not in output[:200]
    urls = len(set(re.findall(r"https?://[^\s)>\"]+", output)))
    dois = len(set(re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", output)))
    pmcids = len(set(re.findall(r"PMC\d{5,}", output, flags=re.I)))
    ncts = len(set(re.findall(r"NCT\d{8}", output, flags=re.I)))
    full_text_dirs = {"pdf-text", "pmc-full-text", "ncbi-bookshelf", "medrxiv-full-text", "biorxiv-pdf", "medrxiv-pdf"}
    full_text_hits = sum(1 for d in artifacts.get("dirs") or [] if str(d).split("/", 1)[0] in full_text_dirs)
    score = 0.0
    score += 25.0 if worked_ok else -25.0
    score += min(22.0, sum(counts.values()) * 1.6)
    score += min(16.0, len(counts) * 2.0)
    score += min(16.0, int(artifacts.get("file_count") or 0) * 0.6)
    score += min(12.0, full_text_hits * 3.0)
    score += min(12.0, (dois + pmcids + ncts) * 0.9)
    score += min(8.0, urls * 0.45)
    score += min(20.0, len(review) / 500.0)
    if "ERROR:" in output:
        score -= 10.0
    if elapsed > 0:
        score -= min(8.0, elapsed / 600.0)
    return round(score, 2)


def make_config(provider: str, model: str, evidence_dir: Path, max_turns: int, self_critique_rounds: int = 0):
    overrides = {
        "agent": {
            "self_critique_enabled": bool(self_critique_rounds),
            "self_critique_rounds": max(0, int(self_critique_rounds or 0)),
            "max_runtime_minutes": BENCHMARK_MAX_RUNTIME_MINUTES,
            "max_cost_usd": 0,
            "main_action": "scientific_researcher_eval",
            "sub_action": "scientific",
        },
        "session": {
            "max_turns": max_turns,
            "memory_max_messages": 12,
            "memory_reset_to_messages": 6,
            "long_term_memory_enabled": False,
            "long_term_memory_max_chars": 0,
            "long_term_memory_dir": "",
        },
        "tools": {
            "max_tools_used": BENCHMARK_MAX_TOOLS_USED,
            "scientific_max_tools_used": BENCHMARK_MAX_TOOLS_USED,
        },
        "env": {
            "CHACK_RESEARCH_DATA_DIR": str(evidence_dir),
            "CHACK_RESEARCH_SAVE_ARTIFACTS": "1",
        },
    }
    return build_subagent_config(
        base_tools_config(),
        model_name=model,
        model_provider=provider,
        max_turns=max_turns,
        system_prompt=_SCIENTIFIC_AGENT_SYSTEM_PROMPT,
        overrides=overrides,
    )


def run_case(case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    case_id = str(case["id"])
    case_dir = out_dir / case_id
    evidence_dir = case_dir / "artifacts"
    case_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    tools = build_tools(case["provider"], case["model"], case.get("selected_sources"))
    base_prompt = TOPICS[case["topic"]]
    previous_case_id = str(case.get("previous_case_id") or "").strip()
    if previous_case_id:
        previous_output_path = out_dir / previous_case_id / "output.txt"
        previous_output = previous_output_path.read_text(encoding="utf-8", errors="replace") if previous_output_path.exists() else ""
        if len(previous_output) > 14000:
            previous_output = previous_output[:14000] + "\n...[previous output truncated for prompt budget]..."
        base_prompt = (
            f"{base_prompt}\n\n"
            "This is pass 2 of a two-cheap-agent sequential scientific review. The previous cheap researcher found this:\n"
            f"{previous_output or '[previous output missing]'}\n\n"
            "Start from that work. Verify the strongest claims, fetch missing full texts or PDFs, look for contradictory studies, "
            "check DOI/provenance/retraction/trial signals, and improve the final review. Explicitly say what you confirmed, "
            "corrected, and newly added beyond pass 1."
        )
    prompt = append_evidence_dir_instruction(
        base_prompt,
        str(evidence_dir),
        "Start now. Use your available scientific tools aggressively, download/read accessible full content, and return only the required JSON schema.",
        save_artifacts=True,
    )
    self_critique_rounds = max(0, int(case.get("self_critique_rounds") or 0))
    config = make_config(
        case["provider"],
        case["model"],
        evidence_dir,
        max_turns=int(case.get("max_turns") or BENCHMARK_MAX_TURNS),
        self_critique_rounds=self_critique_rounds,
    )
    started = time.time()
    output = ""
    error = ""
    counts: Counter[str] = Counter()
    try:
        result = Chack(config).run(
            session_id=f"scientific-eval-{case_id}",
            text=prompt,
            min_tools_used_override=0,
            max_tools_used_override=BENCHMARK_MAX_TOOLS_USED,
            enable_self_critique=bool(self_critique_rounds),
            self_critique_rounds_override=self_critique_rounds,
            require_task_steps_manager_init_first=False,
            tools_override=tools,
            system_prompt_override=config.system_prompt,
        )
        output = str(result.output or "").strip()
        counts = extract_tool_counts(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        output = f"ERROR: {error}"
    elapsed = time.time() - started
    parsed = parse_json_output(output)
    artifacts = artifact_stats(evidence_dir)
    if not counts:
        counts = infer_tool_counts_from_artifacts(artifacts)
    serializable_case = dict(case)
    if isinstance(serializable_case.get("selected_sources"), set):
        serializable_case["selected_sources"] = sorted(serializable_case["selected_sources"])
    summary = {
        **serializable_case,
        "tool_names": [tool_name(tool) for tool in tools],
        "elapsed_seconds": round(elapsed, 2),
        "error": error,
        "output_chars": len(output),
        "parsed_worked": parsed.get("worked"),
        "failure_reason": parsed.get("failure_reason") or parsed.get("reason") or "",
        "self_critique_rounds": self_critique_rounds,
        "run1_steps": getattr(result, "run1_steps", 0) if "result" in locals() else 0,
        "run2_steps": getattr(result, "run2_steps", 0) if "result" in locals() else 0,
        "run1_tools_used": getattr(result, "run1_tools_used", 0) if "result" in locals() else 0,
        "run2_tools_used": getattr(result, "run2_tools_used", 0) if "result" in locals() else 0,
        "run2_output_chars": len(str(getattr(result, "run2_output", "") or "")) if "result" in locals() else 0,
        "tool_counts": dict(counts),
        "total_tool_calls": sum(counts.values()),
        "unique_tools_used": len(counts),
        "artifacts": artifacts,
        "url_count": len(set(re.findall(r"https?://[^\s)>\"]+", output))),
        "doi_count": len(set(re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", output))),
        "pmcid_count": len(set(re.findall(r"PMC\d{5,}", output, flags=re.I))),
        "nct_count": len(set(re.findall(r"NCT\d{8}", output, flags=re.I))),
    }
    summary["score"] = score_run(output, parsed, counts, artifacts, elapsed)
    (case_dir / "output.txt").write_text(output + "\n", encoding="utf-8")
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_cases(max_runs: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    split_a, split_b = partition_source_tools(90713)
    backends = [
        ("codex", "gpt-5.4", "full"),
        ("codex", "gpt-5.4-mini", "mini"),
        ("claude", "claude-opus-4-8", "full"),
        ("claude", "claude-haiku-4-5", "mini"),
    ]
    for topic_key in TOPICS:
        for provider, model, quality in backends:
            cases.append({
                "id": f"{topic_key}__{provider}__{quality}",
                "topic": topic_key,
                "provider": provider,
                "model": model,
                "scenario": f"{quality}_full_tools",
                "max_turns": BENCHMARK_MAX_TURNS,
            })
        cases.extend([
            {
                "id": f"{topic_key}__codex__mini_split_a",
                "topic": topic_key,
                "provider": "codex",
                "model": "gpt-5.4-mini",
                "scenario": "split_a",
                "selected_sources": split_a,
                "max_turns": BENCHMARK_MAX_TURNS,
            },
            {
                "id": f"{topic_key}__codex__mini_split_b",
                "topic": topic_key,
                "provider": "codex",
                "model": "gpt-5.4-mini",
                "scenario": "split_b",
                "selected_sources": split_b,
                "max_turns": BENCHMARK_MAX_TURNS,
            },
            {
                "id": f"{topic_key}__codex__mini_all_tools_chain_1",
                "topic": topic_key,
                "provider": "codex",
                "model": "gpt-5.4-mini",
                "scenario": "cheap_all_tools_chain_1",
                "max_turns": BENCHMARK_MAX_TURNS,
            },
            {
                "id": f"{topic_key}__codex__mini_all_tools_chain_2",
                "topic": topic_key,
                "provider": "codex",
                "model": "gpt-5.4-mini",
                "scenario": "cheap_all_tools_chain_2",
                "previous_case_id": f"{topic_key}__codex__mini_all_tools_chain_1",
                "max_turns": BENCHMARK_MAX_TURNS,
            },
            {
                "id": f"{topic_key}__claude__haiku_split_a",
                "topic": topic_key,
                "provider": "claude",
                "model": "claude-haiku-4-5",
                "scenario": "split_a",
                "selected_sources": split_a,
                "max_turns": BENCHMARK_MAX_TURNS,
            },
            {
                "id": f"{topic_key}__claude__haiku_split_b",
                "topic": topic_key,
                "provider": "claude",
                "model": "claude-haiku-4-5",
                "scenario": "split_b",
                "selected_sources": split_b,
                "max_turns": BENCHMARK_MAX_TURNS,
            },
            {
                "id": f"{topic_key}__claude__haiku_all_tools_chain_1",
                "topic": topic_key,
                "provider": "claude",
                "model": "claude-haiku-4-5",
                "scenario": "cheap_all_tools_chain_1",
                "max_turns": BENCHMARK_MAX_TURNS,
            },
            {
                "id": f"{topic_key}__claude__haiku_all_tools_chain_2",
                "topic": topic_key,
                "provider": "claude",
                "model": "claude-haiku-4-5",
                "scenario": "cheap_all_tools_chain_2",
                "previous_case_id": f"{topic_key}__claude__haiku_all_tools_chain_1",
                "max_turns": BENCHMARK_MAX_TURNS,
            },
        ])
    return cases[:max_runs] if max_runs else cases


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda item: item.get("score", -999), reverse=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Scientific Researcher Eval Summary",
        "",
        "| Rank | Case | Provider | Model | Scenario | Topic | Score | Tool calls | Unique tools | Artifacts | DOI/PMC/NCT | URLs | Worked | Error |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        evidence_ids = int(row.get("doi_count", 0)) + int(row.get("pmcid_count", 0)) + int(row.get("nct_count", 0))
        lines.append(
            "| {rank} | {id} | {provider} | {model} | {scenario} | {topic} | {score} | {calls} | {unique} | {files} | {ids} | {urls} | {worked} | {error} |".format(
                rank=idx,
                id=row.get("id", ""),
                provider=row.get("provider", ""),
                model=row.get("model", ""),
                scenario=row.get("scenario", ""),
                topic=row.get("topic", ""),
                score=row.get("score", ""),
                calls=row.get("total_tool_calls", 0),
                unique=row.get("unique_tools_used", 0),
                files=(row.get("artifacts") or {}).get("file_count", 0),
                ids=evidence_ids,
                urls=row.get("url_count", 0),
                worked=row.get("parsed_worked", ""),
                error=str(row.get("error", "") or "").replace("|", "/")[:80],
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def first_match(pattern: str, text: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    return match.group(1) if match else ""


def smoke_test_tools(out_dir: Path) -> list[dict[str, Any]]:
    smoke_dir = out_dir / "smoke_artifacts"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CHACK_RESEARCH_DATA_DIR"] = str(smoke_dir)
    os.environ["CHACK_RESEARCH_SAVE_ARTIFACTS"] = "1"
    config = base_tools_config()
    sci = ScientificSearchTool(config)
    open_src = OpenResearchTool(config)
    pdf = PdfTextTool(config)
    state: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    def run(name: str, func: Callable[[], str]) -> str:
        started = time.time()
        try:
            output = func()
            status = "pass" if not output.startswith("ERROR:") else "fail"
        except Exception as exc:
            output = f"ERROR: {type(exc).__name__}: {exc}"
            status = "fail"
        row = {
            "name": name,
            "status": status,
            "elapsed_seconds": round(time.time() - started, 2),
            "output_preview": output[:900],
        }
        rows.append(row)
        return output

    arxiv = run("search_arxiv", lambda: sci.search_arxiv("solid state lithium metal battery", max_results=2, timeout_seconds=20))
    state["pdf_url"] = first_match(r"(https?://\S+?\.pdf)", arxiv)
    if state["pdf_url"]:
        run("download_pdf_as_text", lambda: pdf.download_pdf_as_text(state["pdf_url"], max_chars=1200, timeout_seconds=30))
    else:
        rows.append({"name": "download_pdf_as_text", "status": "skip", "elapsed_seconds": 0, "output_preview": "No arXiv PDF URL found."})

    run("search_europe_pmc", lambda: sci.search_europe_pmc("long covid mitochondrial dysfunction", page_size=2, timeout_seconds=20))
    pmc = run("search_pmc_full_text", lambda: sci.search_pmc_full_text("microplastics human placenta", max_results=2, timeout_seconds=20))
    state["pmcid"] = first_match(r"(PMC\d{5,})", pmc)
    if state["pmcid"]:
        run("download_pmc_full_text", lambda: sci.download_pmc_full_text(state["pmcid"], timeout_seconds=30))
    else:
        rows.append({"name": "download_pmc_full_text", "status": "skip", "elapsed_seconds": 0, "output_preview": "No PMCID found."})

    books = run("search_ncbi_bookshelf", lambda: sci.search_ncbi_bookshelf("mitochondrial dysfunction", max_results=2, timeout_seconds=20))
    state["nbk"] = first_match(r"(NBK\d+)", books)
    if state["nbk"]:
        run("download_ncbi_bookshelf", lambda: sci.download_ncbi_bookshelf(state["nbk"], timeout_seconds=30))
    else:
        rows.append({"name": "download_ncbi_bookshelf", "status": "skip", "elapsed_seconds": 0, "output_preview": "No NBK accession found."})

    run("search_semantic_scholar", lambda: sci.search_semantic_scholar("long covid mitochondrial dysfunction", limit=2, timeout_seconds=20))
    run("search_openalex", lambda: sci.search_openalex("solid state lithium metal battery", page=1, per_page=2, timeout_seconds=20))
    run("search_plos", lambda: sci.search_plos("microplastics human health", rows=2, timeout_seconds=20))
    patents = run("search_google_patents", lambda: sci.search_google_patents("solid state lithium battery interface", page=1, num=10, timeout_seconds=20))
    patent_id = first_match(r"patents\.google\.com/(patent/[^ \n]+)", patents)
    if patent_id:
        run("search_google_patents_details", lambda: sci.search_google_patents_details(patent_id, timeout_seconds=20))
    else:
        rows.append({"name": "search_google_patents_details", "status": "skip", "elapsed_seconds": 0, "output_preview": "No patent id found."})

    scholar = run("search_google_scholar", lambda: sci.search_google_scholar("long covid mitochondrial dysfunction", num=2, timeout_seconds=20))
    result_id = first_match(r"result_id: ([A-Za-z0-9_-]+)", scholar)
    if result_id:
        run("search_google_scholar_cite", lambda: sci.search_google_scholar_cite(result_id, timeout_seconds=20))
    else:
        rows.append({"name": "search_google_scholar_cite", "status": "skip", "elapsed_seconds": 0, "output_preview": "No Scholar result_id found."})

    youtube = run("search_youtube_videos", lambda: sci.search_youtube_videos("long COVID mitochondrial dysfunction lecture", limit=2, timeout_seconds=20))
    video_id = first_match(r"video_id: ([A-Za-z0-9_-]{6,})", youtube)
    if video_id:
        run("get_youtube_video_details", lambda: sci.get_youtube_video_details(video_id, timeout_seconds=30))
        run("get_youtube_video_transcript", lambda: sci.get_youtube_video_transcript(video_id, max_segments=5, timeout_seconds=30))
    else:
        rows.append({"name": "get_youtube_video_details", "status": "skip", "elapsed_seconds": 0, "output_preview": "No video id found."})
        rows.append({"name": "get_youtube_video_transcript", "status": "skip", "elapsed_seconds": 0, "output_preview": "No video id found."})

    medrxiv = run("search_medrxiv_preprints", lambda: sci.search_medrxiv_preprints("long covid", "2024-01-01", "2026-07-06", max_results=2, timeout_seconds=30))
    jats = first_match(r"Full-text JATS XML: (https?://\S+)", medrxiv)
    if jats:
        run("download_medrxiv_full_text", lambda: sci.download_medrxiv_full_text(jats, timeout_seconds=30))
    else:
        rows.append({"name": "download_medrxiv_full_text", "status": "skip", "elapsed_seconds": 0, "output_preview": "No medRxiv JATS URL found."})

    crossref = run("crossref_search", lambda: open_src.search_crossref("long covid mitochondrial dysfunction", rows=2, from_year="2020", timeout_seconds=20))
    doi = first_match(r"DOI: (10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", crossref)
    if doi:
        run("crossref_doi_lookup", lambda: open_src.lookup_crossref_doi(doi, timeout_seconds=20))
    else:
        rows.append({"name": "crossref_doi_lookup", "status": "skip", "elapsed_seconds": 0, "output_preview": "No DOI found."})

    trials = run("clinicaltrials_search", lambda: open_src.search_clinical_trials("long covid mitochondrial dysfunction", max_results=2, timeout_seconds=20))
    nct = first_match(r"(NCT\d{8})", trials)
    if nct:
        run("clinicaltrial_get", lambda: open_src.get_clinical_trial(nct, timeout_seconds=20))
    else:
        rows.append({"name": "clinicaltrial_get", "status": "skip", "elapsed_seconds": 0, "output_preview": "No NCT id found."})

    bio = run("biorxiv_search", lambda: open_src.search_biorxiv("covid", server="medrxiv", from_date="2026-01-01", to_date="2026-07-06", max_results=2, timeout_seconds=30))
    bio_doi = first_match(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", bio)
    if bio_doi:
        run("biorxiv_download", lambda: open_src.download_biorxiv_pdf(bio_doi, server="medrxiv", timeout_seconds=30))
    else:
        rows.append({"name": "biorxiv_download", "status": "skip", "elapsed_seconds": 0, "output_preview": "No bioRxiv/medRxiv DOI found."})

    run("retraction_watch", lambda: open_src.search_crossref_retractions("long covid", rows=2, timeout_seconds=20))
    run("pubchem_search", lambda: open_src.search_pubchem("semaglutide", max_results=2, timeout_seconds=20))
    (out_dir / "smoke_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="/Users/carlospolop/git/chack/.env")
    parser.add_argument("--out-dir", default=str(ROOT / ".benchmarks" / "scientific_researcher_eval"))
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    load_env(Path(args.env_file))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_smoke:
        smoke = smoke_test_tools(out_dir)
        failed = [row for row in smoke if row.get("status") == "fail"]
        skipped = [row for row in smoke if row.get("status") == "skip"]
        print(f"SMOKE pass={len(smoke) - len(failed) - len(skipped)} fail={len(failed)} skip={len(skipped)}", flush=True)
        for row in failed:
            print(f"SMOKE FAIL {row['name']}: {row['output_preview'][:200]}", flush=True)
    cases = build_cases(args.max_runs or None)
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        summary_path = out_dir / str(case["id"]) / "summary.json"
        if args.resume and summary_path.exists():
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(f"[{idx}/{len(cases)}] SKIP {case['id']}", flush=True)
            continue
        print(f"[{idx}/{len(cases)}] RUN {case['id']} tools={case.get('selected_sources') or 'all'}", flush=True)
        row = run_case(case, out_dir)
        rows.append(row)
        print(
            f"[{idx}/{len(cases)}] DONE {case['id']} score={row['score']} "
            f"calls={row['total_tool_calls']} artifacts={(row['artifacts'] or {}).get('file_count')} "
            f"error={row['error'] or '-'}",
            flush=True,
        )
        write_summary(out_dir, rows)
    write_summary(out_dir, rows)
    print(f"SUMMARY {out_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
