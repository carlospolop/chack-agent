from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chack_agent import Chack
from chack_tools.config import ToolsConfig
from chack_tools.subagent_config import (
    append_evidence_dir_instruction,
    build_subagent_config,
)
from chack_tools.websearcher_agent import (
    _WEBSEARCHER_AGENT_SYSTEM_PROMPT,
    WebSearcherAgentTool,
)


ESSENTIAL_SHARED_TOOL_NAMES = [
    "fetch_url_text",
]
ARTIFACT_TOOL_NAMES = [
    "list_research_artifacts",
    "read_research_artifact",
    "grep_research_artifacts",
]
SOURCE_TOOL_FAMILIES = [
    ("archive", {"web_archive_search", "wayback_fetch"}),
    ("gdelt", {"gdelt_news_search"}),
    ("brave", {"brave_search"}),
    ("google_web", {"search_google_web"}),
    ("bing_web", {"search_bing_web"}),
    ("google_ai_mode", {"search_google_ai_mode"}),
    ("bing_copilot", {"search_bing_copilot"}),
]
SOURCE_TOOL_NAMES = sorted(set().union(*(tools for _, tools in SOURCE_TOOL_FAMILIES)))
ARTIFACT_DIR_TO_TOOL = {
    "web-pages": "fetch_url_text",
    "wayback-cdx": "web_archive_search",
    "wayback-captures": "wayback_fetch",
    "gdelt-news": "gdelt_news_search",
    "google-web": "search_google_web",
    "bing": "search_bing_web",
    "google_ai_mode": "search_google_ai_mode",
    "bing_copilot": "search_bing_copilot",
}


TOPICS = {
    "pq_tls": (
        "Research the current state of post-quantum cryptography migration for TLS and browser/cloud ecosystems. "
        "Find primary or near-primary sources on NIST/FIPS standardization status, hybrid KEM experiments or deployment, "
        "browser and CDN/server support, cloud provider migration timelines, and credible critiques about operational risks. "
        "Use multiple independent searches, fetch the most relevant pages, check for archived versions of at least one important source, "
        "and separate confirmed facts from unresolved or vendor-claimed items. The final review should compare evidence quality, identify "
        "contradictions, include URLs, and mention saved artifact paths when available."
    ),
    "ai_datacenter_eu": (
        "Research the evidence around AI/data-center electricity and water impact in Spain and the broader European Union. "
        "Find official regulatory, grid, environmental, municipal, company, and reputable news sources about announced projects, "
        "community opposition, water usage, grid constraints, renewable-energy claims, and policy responses. Use search and page fetching, "
        "look for archived context where pages may have changed, and explicitly compare company claims with public authority or civil-society data. "
        "The final review should identify the strongest sources, quantify claims only when source-backed, include URLs, and note saved evidence paths."
    ),
    "glp1_supply": (
        "Research GLP-1 weight-loss/diabetes drug supply, pricing, counterfeit, and telehealth-market risks. Look for regulator warnings, "
        "manufacturer supply updates, pharmacy shortage information, enforcement actions, adverse-event or patient-safety reports, and reporting "
        "on compounded or counterfeit semaglutide/tirzepatide. Use several search strategies, fetch key source pages, preserve artifacts, and avoid "
        "assuming one side is correct without evidence. The final review should distinguish official warnings from market commentary, include URLs, "
        "and explain which claims are well supported, contested, or weakly evidenced."
    ),
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    # The checked .env currently has a Codex token that the CLI rejects; let Codex use its auth.json.
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
        playwright_enabled=True,
        brave_max_results=6,
        serpapi_web_max_results=6,
        open_research_max_results=8,
        websearcher_max_tools_used=24,
        max_tools_used=24,
        min_tools_used=0,
    )


def build_tools(provider: str, model: str, selected_sources: set[str] | None) -> list[Any]:
    helper = WebSearcherAgentTool(
        base_tools_config(),
        model_name=model,
        fallback_model=model,
        model_provider=provider,
        max_turns=24,
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
    review = str(parsed.get("final_research_review") or parsed.get("final_review") or output or "")
    worked = parsed.get("worked")
    if isinstance(worked, str):
        worked_ok = worked.strip().lower() in {"true", "yes", "1", "worked", "success"}
    else:
        worked_ok = bool(worked) if worked is not None else "ERROR:" not in output[:200]
    urls = len(set(re.findall(r"https?://[^\s)>\"]+", output)))
    score = 0.0
    score += 25.0 if worked_ok else -25.0
    score += min(25.0, sum(counts.values()) * 2.0)
    score += min(15.0, len(counts) * 2.5)
    score += min(15.0, int(artifacts.get("file_count") or 0) * 0.75)
    score += min(10.0, urls * 0.75)
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
            "max_runtime_minutes": 25,
            "max_cost_usd": 0,
            "main_action": "webresearcher_eval",
            "sub_action": "webresearcher",
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
            "max_tools_used": 24,
            "websearcher_max_tools_used": 24,
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
        system_prompt=_WEBSEARCHER_AGENT_SYSTEM_PROMPT,
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
        previous_output = ""
        if previous_output_path.exists():
            previous_output = previous_output_path.read_text(encoding="utf-8", errors="replace")
        if len(previous_output) > 14000:
            previous_output = previous_output[:14000] + "\n...[previous output truncated for prompt budget]..."
        base_prompt = (
            f"{base_prompt}\n\n"
            "This is pass 2 of a two-cheap-agent sequential review. The previous cheap researcher found the following:\n"
            f"{previous_output or '[previous output missing]'}\n\n"
            "Your job is to start from that work, not merely repeat it. Verify the strongest claims, find missing primary sources, "
            "look for contradictions or antagonistic evidence, fetch additional relevant pages, preserve artifacts, and produce a better final report. "
            "Explicitly say what you confirmed, what you corrected, and what new evidence you added beyond pass 1."
        )
    prompt = append_evidence_dir_instruction(
        base_prompt,
        str(evidence_dir),
        "Start now. Use your available web tools aggressively and return only the required JSON schema.",
        save_artifacts=True,
    )
    self_critique_rounds = max(0, int(case.get("self_critique_rounds") or 0))
    config = make_config(
        case["provider"],
        case["model"],
        evidence_dir,
        max_turns=int(case.get("max_turns") or 24),
        self_critique_rounds=self_critique_rounds,
    )
    started = time.time()
    output = ""
    error = ""
    counts: Counter[str] = Counter()
    try:
        result = Chack(config).run(
            session_id=f"webresearcher-eval-{case_id}",
            text=prompt,
            min_tools_used_override=0,
            max_tools_used_override=24,
            enable_self_critique=bool(self_critique_rounds),
            self_critique_rounds_override=self_critique_rounds,
            require_task_steps_manager_init_first=False,
            required_tool_names=["fetch_url_text"],
            required_tool_call_attempts=2,
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
    }
    summary["score"] = score_run(output, parsed, counts, artifacts, elapsed)
    (case_dir / "output.txt").write_text(output + "\n", encoding="utf-8")
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_cases(max_runs: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    split_a, split_b = partition_source_tools(54821)
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
                "max_turns": 24,
            })
        cases.append({
            "id": f"{topic_key}__codex__mini_split_a",
            "topic": topic_key,
            "provider": "codex",
            "model": "gpt-5.4-mini",
            "scenario": "split_a",
            "selected_sources": split_a,
            "max_turns": 24,
        })
        cases.append({
            "id": f"{topic_key}__codex__mini_split_b",
            "topic": topic_key,
            "provider": "codex",
            "model": "gpt-5.4-mini",
            "scenario": "split_b",
            "selected_sources": split_b,
            "max_turns": 24,
        })
        cases.append({
            "id": f"{topic_key}__codex__mini_all_tools_chain_1",
            "topic": topic_key,
            "provider": "codex",
            "model": "gpt-5.4-mini",
            "scenario": "cheap_all_tools_chain_1",
            "max_turns": 24,
        })
        cases.append({
            "id": f"{topic_key}__codex__mini_all_tools_chain_2",
            "topic": topic_key,
            "provider": "codex",
            "model": "gpt-5.4-mini",
            "scenario": "cheap_all_tools_chain_2",
            "previous_case_id": f"{topic_key}__codex__mini_all_tools_chain_1",
            "max_turns": 24,
        })
        cases.append({
            "id": f"{topic_key}__claude__haiku_split_a",
            "topic": topic_key,
            "provider": "claude",
            "model": "claude-haiku-4-5",
            "scenario": "split_a",
            "selected_sources": split_a,
            "max_turns": 24,
        })
        cases.append({
            "id": f"{topic_key}__claude__haiku_split_b",
            "topic": topic_key,
            "provider": "claude",
            "model": "claude-haiku-4-5",
            "scenario": "split_b",
            "selected_sources": split_b,
            "max_turns": 24,
        })
        cases.append({
            "id": f"{topic_key}__claude__haiku_all_tools_chain_1",
            "topic": topic_key,
            "provider": "claude",
            "model": "claude-haiku-4-5",
            "scenario": "cheap_all_tools_chain_1",
            "max_turns": 24,
        })
        cases.append({
            "id": f"{topic_key}__claude__haiku_all_tools_chain_2",
            "topic": topic_key,
            "provider": "claude",
            "model": "claude-haiku-4-5",
            "scenario": "cheap_all_tools_chain_2",
            "previous_case_id": f"{topic_key}__claude__haiku_all_tools_chain_1",
            "max_turns": 24,
        })
    return cases[:max_runs] if max_runs else cases


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda item: item.get("score", -999), reverse=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Web Researcher Eval Summary",
        "",
        "| Rank | Case | Provider | Model | Scenario | Topic | Score | Tool calls | Unique tools | Artifacts | URLs | Worked | Error |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {id} | {provider} | {model} | {scenario} | {topic} | {score} | {calls} | {unique} | {files} | {urls} | {worked} | {error} |".format(
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
                urls=row.get("url_count", 0),
                worked=row.get("parsed_worked", ""),
                error=str(row.get("error", "") or "").replace("|", "/")[:80],
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="/Users/carlospolop/git/chack/.env")
    parser.add_argument("--out-dir", default=str(ROOT / ".benchmarks" / "webresearcher_eval"))
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_env(Path(args.env_file))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
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
