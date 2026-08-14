from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chack_tools.config import ToolsConfig
from chack_tools.researcher_administrator_agent import (
    RESEARCHER_REGISTRY,
    ResearcherAdministratorAgentTool,
)
from chack_tools.telemetry.context import reset_log_context, set_log_context


_BROWSER_RESEARCHERS = {"deepchatgpt", "prochatgpt", "chatgptxhigh"}
# The normal CLI/provider matrix is derived from the runtime registry. Browser
# researchers are a separate authenticated-UI acceptance surface.
RESEARCHERS = [
    short for short in RESEARCHER_REGISTRY
    if short not in _BROWSER_RESEARCHERS
]


TOPICS = {
    "consumer_vpn_privacy": (
        "Research whether major consumer VPN providers' privacy and no-log marketing claims are well supported. Focus on "
        "independent audits, court or law-enforcement disclosures, ownership/entity structure, breach or incident history, "
        "tracker/telemetry claims in apps or websites, pricing/affiliate incentives, and credible critical reporting. Use web/news, "
        "business/entity, product/security, and social/forum sources where relevant. Preserve evidence, compare marketing claims "
        "against antagonistic evidence, and separate verified facts, plausible but weak claims, and gaps."
    ),
    "melatonin_gummies_children": (
        "Research the evidence and risks around melatonin gummies marketed or used for children in the US/EU. Cover scientific "
        "and clinical safety evidence, poison-control or adverse-event signals, regulatory guidance or warnings, product labeling "
        "and recall/quality issues, retailer/brand market behavior, and parent/social claims where useful. Use scientific, legal/regulatory, "
        "product, web/news, and social sources. Preserve evidence and clearly rank claims by evidence quality. Explicitly compare official "
        "regulator guidance, peer-reviewed or clinical evidence, product test/label evidence, market claims, and anecdotal parent reports; "
        "identify contradictions, missing source families, and what evidence would most change confidence."
    ),
    "ev_charger_reliability": (
        "Research public evidence about reliability problems in public EV charging networks, with emphasis on the US and EU. Cover "
        "government or standards data, company/network claims, consumer reports or social/forum evidence, product/recall/security issues, "
        "and business incentives or market concentration. Use web/news, business, product, data/statistics if useful, and social/forum sources. "
        "Preserve evidence, look for counterevidence to both optimistic and pessimistic narratives, and summarize actionable gaps."
    ),
    "ai_datacenter_spain": (
        "Research the evidence around AI/data-center electricity and water impact in Spain and the broader "
        "European Union, with a special focus on announced hyperscaler projects, regional grid and water constraints, "
        "company sustainability claims, municipal/regulatory filings, opposition from local communities, and credible "
        "news or NGO investigations. Use multiple specialized researchers: web/news for public reporting and official "
        "pages, legal for Spanish/EU regulatory and permitting context, business for company/project claims, data/statistics "
        "for energy/water/public datasets, knowledge graph for entity relationships, and social/network sources when there "
        "are claims of local opposition or public campaigns. Preserve source-level evidence, compare company claims against "
        "public authority or civil-society evidence, actively look for antagonistic evidence, and finish with a confidence-graded "
        "synthesis that separates established facts, contested claims, weak claims, and remaining gaps."
    ),
    "glp1_market_safety": (
        "Research GLP-1 weight-loss and diabetes drug supply, pricing, compounded/counterfeit product, telehealth-market, "
        "regulatory, and patient-safety risks. Cover semaglutide and tirzepatide, major manufacturers, pharmacies/compounders, "
        "regulator warnings, shortage or supply updates, enforcement actions, adverse-event or poison-control evidence, media "
        "investigations, market incentives, and social-network signals when they reveal non-official user claims or promotion patterns. "
        "Use scientific sources for safety evidence, legal/regulatory sources for warnings and enforcement, business/product sources "
        "for market and product availability, web/news for current reporting, and knowledge graph for entity disambiguation. Preserve "
        "source-level data, do not assume official or industry claims are true without checking counterevidence, and provide a final "
        "review that clearly separates official evidence, peer-reviewed/clinical evidence, market commentary, anecdotal signals, and gaps."
    ),
    "solid_state_batteries": (
        "Research the current commercial reality of solid-state batteries for EVs and consumer electronics: which companies claim "
        "near-term deployment, what technical milestones are independently evidenced, what patents or papers support or undermine "
        "the claims, which automakers or suppliers are involved, what production timelines have slipped, and where investment or "
        "media hype may exceed evidence. Use web/news, scientific, business, product, patents via the relevant researcher tools, "
        "knowledge graph for company/entity links, and data/statistics where market or production numbers are available. Preserve "
        "source-level evidence and test the strongest bullish claims against skeptical technical, commercial, and regulatory sources. "
        "The administrator conclusion should rank claims by evidence strength and identify exactly what would change the confidence."
    ),
    "religious_ai_bioethics": (
        "Research how major religious traditions and secular regulators discuss AI-assisted embryo selection or genetic screening. "
        "Cover primary religious texts or authoritative commentary where available, bioethics or medical literature, legal/regulatory "
        "status in the US/EU/Spain, advocacy or public controversy, business/product claims from fertility clinics or genetic-testing "
        "companies, knowledge-graph/entity disambiguation for organizations, and credible news/social evidence of public campaigns. "
        "Use religious, scientific, legal/regulatory, web/news, business, knowledge graph, and social sources when relevant. Preserve "
        "evidence and separate direct primary text, official institutional guidance, peer-reviewed evidence, commercial marketing, and weak claims."
    ),
    "cli_repro_research": (
        "Research whether current public package/data sources support a reproducible local check of recent npm package typosquatting "
        "or dependency-confusion risk around popular AI developer tooling. Use the CLI researcher to run safe local commands, query package "
        "registries or public metadata, inspect downloaded JSON/text artifacts, and summarize reproducible command evidence. Also use web/news, "
        "product/security, business/entity, and knowledge graph researchers as needed for context, advisories, maintainers, ownership, and public "
        "reporting. Preserve evidence and clearly distinguish command-verified facts from search-result leads and commentary."
    ),
    "travel_sustainable_europe": (
        "Research a bounded travel-planning question for a two-week rail-first trip through Spain, France, and Italy in the current "
        "season. Compare realistic route options, border and rail-operator constraints, reservation requirements, disruption or strike "
        "risks, seasonal weather, accessibility, and major sustainable alternatives to short-haul flights. Use official rail/operator "
        "and government sources first, then current travel reporting and traveler evidence only as supporting context. Preserve exact "
        "dates, fares or availability caveats, source provenance, contradictions, and unresolved details; do not invent live schedules."
    ),
}

SPLIT_PRIORITY_SUFFIX = (
    "\n\n### Evaluation variant: complementary duplicate researchers\n"
    "When a researcher type is clearly relevant and budget allows, launch two independent runs of that same researcher. "
    "Give both access to all their normal tools, but ask each run to prioritize a different half of SOURCE/DISCOVERY tools. "
    "Do not split utility/content access tools such as fetch_url_text, playwright_fetch, PDF/text extractors, transcript fetchers, artifact list/read/grep, or detail/download tools; both runs may use those whenever needed. "
    "The first run should prioritize official/primary/index/database sources where available. The second run should prioritize web/news/social/commercial/contradictory or independent sources where available. "
    "After both return, compare overlap, contradictions, and missing source families before concluding."
)


BACKEND_PROFILES = {
    "codex": {
        "provider": "codex",
        "admin_model": "gpt-5.5",
        "researcher_model": "gpt-5.4-mini",
    },
    "claude": {
        "provider": "claude",
        "admin_model": "claude-opus-4-8",
        "researcher_model": "claude-haiku-4-5",
    },
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
    # Prefer the already-authenticated local Codex CLI session when present.
    # Some .env access tokens are short-lived and make Codex reject otherwise valid local auth.
    os.environ.pop("CODEX_ACCESS_TOKEN", None)
    os.environ.setdefault("CHACK_DISABLE_STDOUT_EVENTS", "1")
    os.environ.setdefault("CHACK_DISABLE_CODEX_NATIVE_WEB", "1")
    os.environ.setdefault("CHACK_MCP_TOOL_MAX_TOKENS", "50000")
    os.environ.setdefault("CHACK_CODEX_EXEC_TIMEOUT_SECONDS", "3600")
    os.environ.setdefault("CHACK_CODEX_MCP_STARTUP_TIMEOUT_SECONDS", "180")
    os.environ.setdefault("CHACK_CLAUDE_EXEC_TIMEOUT_SECONDS", "3600")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_json_output(output: str) -> dict[str, Any]:
    text = str(output or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def artifact_stats(path: str) -> dict[str, Any]:
    root = Path(path)
    if not path or not root.exists():
        return {"file_count": 0, "bytes": 0, "dirs": []}
    files = [p for p in root.rglob("*") if p.is_file()]
    dirs = sorted({str(p.parent.relative_to(root)) for p in files if p.parent != root})
    return {
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "dirs": dirs,
    }


def validate_case(
    summary: dict[str, Any],
    parsed: dict[str, Any],
    raw: str,
) -> dict[str, Any]:
    """Apply independent case-level gates; reject schema-shaped/null rows."""
    failures: list[str] = []
    expected_researchers = [str(value) for value in summary.get("enabled_researchers") or []]
    expected_tools = [
        RESEARCHER_REGISTRY[name][1]
        for name in expected_researchers
        if name in RESEARCHER_REGISTRY
    ]
    parse_ok = isinstance(parsed, dict) and isinstance(parsed.get("research_worked"), bool)
    if not parse_ok:
        failures.append("administrator output is missing an explicit boolean research_worked")
    if not str(raw or "").strip():
        failures.append("administrator returned empty output")

    responses = parsed.get("researcher_responses") if isinstance(parsed.get("researcher_responses"), list) else []
    researcher_calls = parsed.get("researcher_call_counts") if isinstance(parsed.get("researcher_call_counts"), dict) else {}
    tool_counts = parsed.get("researcher_tool_call_counts") if isinstance(parsed.get("researcher_tool_call_counts"), dict) else {}
    if not responses:
        failures.append("no terminal researcher response was returned")
    if not tool_counts:
        failures.append("no researcher tool calls were observed")
    for tool_name in expected_tools:
        if int(researcher_calls.get(tool_name) or 0) < 1:
            failures.append(f"researcher call was not observed for {tool_name}")

    conclusions = str(parsed.get("administrator_conclusions") or "").strip()
    if len(conclusions) < 40:
        failures.append("administrator synthesis is not substantive")
    for response in responses:
        if not isinstance(response, dict):
            failures.append("researcher response is not an object")
            continue
        tool_name = str(response.get("researcher_tool") or "unknown")
        if response.get("research_worked") is not True:
            failures.append(f"{tool_name} did not report research_worked=true")
        findings = response.get("findings")
        if not isinstance(findings, list) or not findings:
            failures.append(f"{tool_name} has no findings in its digest")

    evidence_path = str(summary.get("evidence_data_path") or "")
    artifact_info = summary.get("artifact_stats") if isinstance(summary.get("artifact_stats"), dict) else {}
    artifact_files = int(artifact_info.get("file_count") or 0)
    full_records: dict[str, dict[str, Any]] = {}
    raw_files: list[Path] = []
    source_files: list[Path] = []
    if summary.get("save_artifacts"):
        if not evidence_path or not Path(evidence_path).expanduser().is_dir():
            failures.append("preserved evidence_data_path is missing or not a directory")
        else:
            root = Path(evidence_path).expanduser()
            if artifact_files <= 0:
                failures.append("preserved evidence workspace is empty")
            output_dir = root / "researcher_outputs"

            def persisted_response_rank(record: dict[str, Any]) -> tuple[bool, bool, bool, int]:
                review = str(record.get("full_research_review") or "").strip()
                findings = record.get("findings")
                return (
                    record.get("research_worked") is True,
                    not str(record.get("failure_reason") or "").strip(),
                    isinstance(findings, list) and bool(findings),
                    len(review),
                )

            for path in sorted(output_dir.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    failures.append(f"full parsed researcher output is unreadable: {path.name}")
                    continue
                if isinstance(record, dict) and str(record.get("researcher_tool") or "").strip():
                    tool_name = str(record["researcher_tool"])
                    existing = full_records.get(tool_name)
                    # A lifecycle probe/retry can legitimately persist failed and
                    # successful attempts for the same researcher. Validate the
                    # strongest terminal response instead of allowing filename
                    # order to make a later cancellation overwrite the success.
                    if existing is None or persisted_response_rank(record) > persisted_response_rank(existing):
                        full_records[tool_name] = record
            raw_files = sorted(output_dir.glob("*.raw.txt"))
            ignored_parts = {"researcher_outputs", "researcher_jobs", "admin_output.json"}
            source_files = [
                path for path in root.rglob("*")
                if path.is_file()
                and path.name != "_artifact_manifest.jsonl"
                and not any(part in ignored_parts for part in path.relative_to(root).parts)
            ]
            if not full_records:
                failures.append("full parsed researcher output was not persisted")
            if not raw_files:
                failures.append("exact raw researcher output was not persisted")
            if not source_files:
                failures.append("no source/detail artifact was persisted")
            for tool_name in expected_tools:
                record = full_records.get(tool_name)
                if record is None:
                    failures.append(f"full response is missing for {tool_name}")
                    continue
                if record.get("research_worked") is not True:
                    failures.append(f"persisted response for {tool_name} is not successful")
                if len(str(record.get("full_research_review") or "").strip()) < 20:
                    failures.append(f"persisted response for {tool_name} has no substantive full review")
                if not isinstance(record.get("findings"), list) or not record.get("findings"):
                    failures.append(f"persisted response for {tool_name} has no findings")

            for ledger_path in sorted((root / "researcher_jobs").glob("*.json")):
                try:
                    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                except Exception:
                    failures.append(f"researcher ledger is unreadable: {ledger_path.name}")
                    continue
                for task in ledger.get("tasks") or []:
                    status = str(task.get("status") or "")
                    if status not in {"done", "error", "cancelled", "deadline_exceeded"}:
                        failures.append(f"researcher task remained non-terminal: {task.get('task_id')}")
                    if task.get("execution_active"):
                        failures.append(f"researcher task remained physically active: {task.get('task_id')}")

    harness_failure = not parse_ok and "bootstrapping phase" in str(raw or "")
    acceptance_pass = not failures and parsed.get("research_worked") is True
    return {
        "parse_ok": parse_ok,
        "terminal": parse_ok,
        "functional_pass": acceptance_pass,
        "acceptance_pass": acceptance_pass,
        "harness_failure": harness_failure,
        "validation_failures": failures,
    }


def base_tools_config(*, researcher_tool_budget: int, admin_tool_budget: int, max_results: int) -> ToolsConfig:
    max_results = max(2, int(max_results or 4))
    researcher_tool_budget = max(3, int(researcher_tool_budget or 8))
    admin_tool_budget = max(1, int(admin_tool_budget or 8))
    return ToolsConfig(
        task_steps_manager_enabled=False,
        playwright_enabled=True,
        playwright_timeout_seconds=45,
        playwright_max_output_chars=20000,
        brave_enabled=True,
        brave_max_results=max_results,
        serpapi_google_web_enabled=True,
        serpapi_bing_web_enabled=True,
        serpapi_bing_copilot_enabled=True,
        serpapi_web_max_results=max_results,
        open_research_max_results=max_results,
        open_research_fetch_url_text_enabled=True,
        open_research_web_archive_enabled=True,
        open_research_gdelt_enabled=True,
        open_research_federal_register_enabled=True,
        open_research_world_bank_enabled=True,
        open_research_wikidata_enabled=True,
        websearcher_enabled=True,
        websearcher_brave_enabled=True,
        websearcher_google_web_enabled=True,
        websearcher_bing_web_enabled=True,
        websearcher_google_ai_mode_enabled=True,
        websearcher_bing_copilot_enabled=True,
        websearcher_web_archive_enabled=True,
        websearcher_gdelt_enabled=True,
        websearcher_fetch_url_text_enabled=True,
        scientific_enabled=True,
        scientific_max_results=max_results,
        scientific_arxiv_enabled=True,
        scientific_europe_pmc_enabled=True,
        scientific_pmc_full_text_enabled=True,
        scientific_ncbi_bookshelf_enabled=True,
        scientific_semantic_scholar_enabled=True,
        scientific_openalex_enabled=True,
        scientific_plos_enabled=True,
        scientific_google_patents_enabled=True,
        scientific_google_patents_details_enabled=True,
        scientific_google_scholar_enabled=True,
        scientific_google_scholar_cite_enabled=True,
        scientific_medrxiv_enabled=True,
        scientific_crossref_enabled=True,
        scientific_clinicaltrials_enabled=True,
        scientific_biorxiv_enabled=True,
        scientific_retraction_watch_enabled=True,
        scientific_pubchem_enabled=True,
        scientific_pdf_text_enabled=True,
        business_enabled=True,
        business_max_results=max_results,
        business_sec_enabled=True,
        business_gleif_enabled=True,
        business_google_finance_enabled=True,
        business_google_finance_markets_enabled=True,
        business_google_web_enabled=True,
        business_bing_web_enabled=True,
        business_google_news_enabled=True,
        business_google_trends_enabled=True,
        business_google_patents_enabled=True,
        business_google_patents_details_enabled=True,
        business_google_maps_enabled=True,
        business_google_maps_reviews_enabled=True,
        business_yelp_enabled=True,
        business_apple_maps_enabled=True,
        business_google_ads_enabled=True,
        business_google_ads_transparency_enabled=True,
        business_google_shopping_enabled=True,
        business_google_shopping_light_enabled=True,
        business_google_immersive_product_enabled=True,
        business_amazon_enabled=True,
        business_walmart_enabled=True,
        business_ebay_enabled=True,
        business_home_depot_enabled=True,
        business_tripadvisor_enabled=True,
        business_cpsc_enabled=True,
        business_federal_register_enabled=True,
        business_wikidata_enabled=True,
        business_playwright_enabled=True,
        product_enabled=True,
        product_max_results=max_results,
        product_serpapi_enabled=True,
        product_google_lens_enabled=True,
        product_open_food_facts_enabled=True,
        product_openfda_enabled=True,
        product_nvd_enabled=True,
        product_cpsc_enabled=True,
        product_cisa_kev_enabled=True,
        product_osv_enabled=True,
        product_google_shopping_enabled=True,
        product_google_shopping_light_enabled=True,
        product_google_immersive_product_enabled=True,
        product_amazon_enabled=True,
        product_walmart_enabled=True,
        product_ebay_enabled=True,
        product_home_depot_enabled=True,
        product_google_trends_enabled=True,
        product_google_patents_enabled=True,
        product_youtube_enabled=True,
        product_playwright_enabled=True,
        legal_enabled=True,
        legal_boe_enabled=True,
        legal_federal_register_enabled=True,
        legal_wikidata_enabled=True,
        data_statistics_enabled=True,
        data_statistics_world_bank_enabled=True,
        data_statistics_wikidata_enabled=True,
        news_media_enabled=True,
        knowledge_graph_enabled=True,
        knowledge_graph_wikidata_enabled=True,
        social_network_enabled=True,
        social_network_forum_search_enabled=True,
        social_network_linkedin_enabled=True,
        social_network_instagram_enabled=True,
        social_network_reddit_posts_enabled=True,
        social_network_reddit_comments_enabled=True,
        social_network_x_enabled=True,
        social_network_google_forums_enabled=True,
        social_network_google_news_enabled=True,
        social_network_google_trends_enabled=True,
        social_network_google_trending_now_enabled=True,
        social_network_google_videos_enabled=True,
        social_network_instagram_profile_enabled=True,
        social_network_facebook_profile_enabled=True,
        social_network_youtube_video_details_enabled=True,
        social_network_mastodon_enabled=True,
        social_network_tiktok_web_enabled=True,
        social_network_bluesky_web_enabled=True,
        religious_enabled=True,
        religious_bible_enabled=True,
        religious_sefaria_enabled=True,
        religious_quran_enabled=True,
        religious_gita_enabled=True,
        religious_hadith_enabled=True,
        religious_suttacentral_enabled=True,
        religious_wikidata_enabled=True,
        cli_enabled=True,
        cli_exec_enabled=True,
        cli_brave_enabled=True,
        cli_google_web_enabled=True,
        researcher_administrator_enabled=True,
        researcher_administrator_researchers=RESEARCHERS,
        min_tools_used=0,
        max_tools_used=0,
        websearcher_max_tools_used=researcher_tool_budget,
        scientific_max_tools_used=researcher_tool_budget,
        business_max_tools_used=researcher_tool_budget,
        product_max_tools_used=researcher_tool_budget,
        legal_max_tools_used=researcher_tool_budget,
        data_statistics_max_tools_used=researcher_tool_budget,
        news_media_max_tools_used=researcher_tool_budget,
        knowledge_graph_max_tools_used=researcher_tool_budget,
        social_network_max_tools_used=researcher_tool_budget,
        religious_max_tools_used=researcher_tool_budget,
        cli_max_tools_used=researcher_tool_budget,
        researcher_administrator_max_tools_used=admin_tool_budget,
    )


def make_helper(
    profile: dict[str, str],
    *,
    admin_turns: int,
    researcher_turns: int,
    researcher_tool_budget: int,
    admin_tool_budget: int,
    max_results: int,
    researchers: list[str] | None = None,
) -> ResearcherAdministratorAgentTool:
    researcher_model = profile["researcher_model"]
    active_researchers = list(researchers or RESEARCHERS)
    researcher_models = {name: researcher_model for name in active_researchers}
    researcher_max_turns = {name: researcher_turns for name in active_researchers}
    config = base_tools_config(
        researcher_tool_budget=researcher_tool_budget,
        admin_tool_budget=admin_tool_budget,
        max_results=max_results,
    )
    config.researcher_administrator_researchers = active_researchers
    return ResearcherAdministratorAgentTool(
        config,
        model_name=profile["admin_model"],
        fallback_model=profile["admin_model"],
        model_provider=profile["provider"],
        max_turns=admin_turns,
        researchers=active_researchers,
        required_researchers=active_researchers,
        researcher_model_overrides=researcher_models,
        researcher_max_turns_overrides=researcher_max_turns,
        social_network_model=researcher_model,
        scientific_model=researcher_model,
        websearcher_model=researcher_model,
        business_model=researcher_model,
        product_model=researcher_model,
        legal_model=researcher_model,
        data_statistics_model=researcher_model,
        news_media_model=researcher_model,
        knowledge_graph_model=researcher_model,
        religious_model=researcher_model,
        cli_model=researcher_model,
        social_network_max_turns=researcher_turns,
        scientific_max_turns=researcher_turns,
        websearcher_max_turns=researcher_turns,
        business_max_turns=researcher_turns,
        product_max_turns=researcher_turns,
        legal_max_turns=researcher_turns,
        data_statistics_max_turns=researcher_turns,
        news_media_max_turns=researcher_turns,
        knowledge_graph_max_turns=researcher_turns,
        religious_max_turns=researcher_turns,
        cli_max_turns=researcher_turns,
    )


def score_result(parsed: dict[str, Any], raw: str, artifacts: dict[str, Any], elapsed: float) -> float:
    conclusions = str(parsed.get("administrator_conclusions") or "")
    responses = parsed.get("researcher_responses") if isinstance(parsed.get("researcher_responses"), list) else []
    researcher_calls = parsed.get("researcher_call_counts") if isinstance(parsed.get("researcher_call_counts"), dict) else {}
    tool_counts = parsed.get("researcher_tool_call_counts") if isinstance(parsed.get("researcher_tool_call_counts"), dict) else {}
    urls = len(set(re.findall(r"https?://[^\s)>\"]+", raw)))
    worked = parsed.get("research_worked")
    score = 0.0
    score += 30.0 if worked is True else -25.0
    score += min(20.0, len(responses) * 3.0)
    score += min(15.0, len(researcher_calls) * 2.0)
    score += min(20.0, sum(int(v or 0) for v in tool_counts.values()) * 0.4)
    score += min(15.0, len(tool_counts) * 1.0)
    score += min(15.0, int(artifacts.get("file_count") or 0) * 0.4)
    score += min(10.0, urls * 0.5)
    score += min(20.0, len(conclusions) / 400.0)
    score -= min(8.0, elapsed / 900.0)
    if "ERROR:" in raw[:500]:
        score -= 20.0
    return round(score, 2)


def run_case(
    backend_name: str,
    topic_name: str,
    prompt: str,
    *,
    out_dir: Path,
    admin_turns: int,
    researcher_turns: int,
    researcher_tool_budget: int,
    admin_tool_budget: int,
    max_results: int,
    admin_model_override: str = "",
    researcher_model_override: str = "",
    prompt_variant: str = "baseline",
    max_runtime_minutes: int = 20,
    researchers: list[str] | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    profile = dict(BACKEND_PROFILES[backend_name])
    if admin_model_override:
        profile["admin_model"] = admin_model_override
    if researcher_model_override:
        profile["researcher_model"] = researcher_model_override
    prompt_variant = str(prompt_variant or "baseline").strip() or "baseline"
    if prompt_variant == "split_priority":
        prompt = f"{prompt.rstrip()}{SPLIT_PRIORITY_SUFFIX}"
    elif prompt_variant != "baseline":
        raise ValueError(f"unknown prompt_variant: {prompt_variant}")
    run_id = f"{backend_name}_{prompt_variant}_{topic_name}_{uuid.uuid4().hex[:8]}"
    case_dir = out_dir / run_id
    case_dir.mkdir(parents=True, exist_ok=True)
    helper = make_helper(
        profile,
        admin_turns=admin_turns,
        researcher_turns=researcher_turns,
        researcher_tool_budget=researcher_tool_budget,
        admin_tool_budget=admin_tool_budget,
        max_results=max_results,
        researchers=researchers,
    )
    token = set_log_context(
        session_id=f"eval:{run_id}",
        max_turns=admin_turns,
        max_runtime_minutes=max_runtime_minutes,
        remaining_runtime_minutes=max_runtime_minutes,
        max_cost_usd=0,
        remaining_cost_usd=0,
        memory_max_messages=16,
        memory_reset_to_messages=8,
        main_action="researcher_administrator_eval",
    )
    started = time.time()
    try:
        raw = helper.run(prompt, save_artifacts=save_artifacts)
    except Exception as exc:
        raw = f"ERROR: {type(exc).__name__}: {exc}"
    finally:
        reset_log_context(token)
    elapsed = time.time() - started
    parsed = parse_json_output(raw)
    evidence_path = str(parsed.get("evidence_data_path") or "")
    artifacts = artifact_stats(evidence_path)
    summary = {
        "run_id": run_id,
        "backend": backend_name,
        "admin_model": profile["admin_model"],
        "researcher_model": profile["researcher_model"],
        "enabled_researchers": list(researchers or RESEARCHERS),
        "save_artifacts": bool(save_artifacts),
        "topic": topic_name,
        "prompt_variant": prompt_variant,
        "elapsed_seconds": round(elapsed, 2),
        "research_worked": parsed.get("research_worked"),
        "failure_reason": parsed.get("failure_reason", ""),
        "conclusion_chars": len(str(parsed.get("administrator_conclusions") or "")),
        "researcher_responses": len(parsed.get("researcher_responses") or []),
        "researcher_call_counts": parsed.get("researcher_call_counts") or {},
        "researcher_tool_call_counts": parsed.get("researcher_tool_call_counts") or {},
        "artifact_stats": artifacts,
        "evidence_data_path": evidence_path,
        "score": score_result(parsed, raw, artifacts, elapsed),
    }
    summary.update(validate_case(summary, parsed, raw))
    (case_dir / "raw_output.json").write_text(raw, encoding="utf-8")
    (case_dir / "parsed_output.json").write_text(compact_json(parsed), encoding="utf-8")
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate researcher_administrator with complex research prompts.")
    parser.add_argument("--env-file", default="/Users/carlospolop/git/chack/.env")
    parser.add_argument("--out-dir", default=".benchmarks/researcher_administrator_eval")
    parser.add_argument("--backends", nargs="+", choices=sorted(BACKEND_PROFILES), default=["codex", "claude"])
    parser.add_argument("--topics", nargs="+", choices=sorted(TOPICS), default=list(TOPICS))
    parser.add_argument("--prompt-variant", choices=["baseline", "split_priority"], default="baseline")
    parser.add_argument("--admin-model", default="", help="Override profile administrator model.")
    parser.add_argument("--researcher-model", default="", help="Override profile researcher model.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of topic runs per backend; 0 means all.")
    parser.add_argument("--admin-turns", type=int, default=28)
    parser.add_argument("--researcher-turns", type=int, default=18)
    parser.add_argument("--admin-tool-budget", type=int, default=8)
    parser.add_argument("--researcher-tool-budget", type=int, default=8)
    parser.add_argument("--max-results", type=int, default=4)
    parser.add_argument("--max-runtime-minutes", type=int, default=20)
    parser.add_argument("--researchers", nargs="+", choices=RESEARCHERS, default=RESEARCHERS)
    parser.add_argument("--no-save-artifacts", action="store_true")
    args = parser.parse_args()

    load_env(Path(args.env_file).expanduser())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    invalid_cases = 0
    for backend in args.backends:
        topics = list(args.topics)
        if args.limit > 0:
            topics = topics[: args.limit]
        for topic in topics:
            print(f"RUN {backend} {topic}", flush=True)
            summary = run_case(
                backend,
                topic,
                TOPICS[topic],
                out_dir=out_dir,
                admin_turns=args.admin_turns,
                researcher_turns=args.researcher_turns,
                researcher_tool_budget=args.researcher_tool_budget,
                admin_tool_budget=args.admin_tool_budget,
                max_results=args.max_results,
                admin_model_override=args.admin_model,
                researcher_model_override=args.researcher_model,
                prompt_variant=args.prompt_variant,
                max_runtime_minutes=args.max_runtime_minutes,
                researchers=args.researchers,
                save_artifacts=not args.no_save_artifacts,
            )
            all_summaries.append(summary)
            print(compact_json(summary), flush=True)
            if not summary.get("acceptance_pass"):
                invalid_cases += 1

    summary_path = out_dir / "summary.json"
    existing_summaries: list[dict[str, Any]] = []
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing_summaries = [row for row in loaded if isinstance(row, dict)]
        except Exception:
            existing_summaries = []
    summary_path.write_text(
        json.dumps(existing_summaries + all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"WROTE {summary_path}", flush=True)
    if invalid_cases:
        print(f"ACCEPTANCE FAILED: {invalid_cases} case(s) did not satisfy the independent gates.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
