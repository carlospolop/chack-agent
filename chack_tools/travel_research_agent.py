from __future__ import annotations

import os
from typing import Any, Optional

from .business_search import (
    BusinessSearchTool,
    get_apple_maps_place_tool,
    get_apple_maps_search_tool,
    get_google_maps_reviews_tool,
    get_google_maps_search_tool,
    get_tripadvisor_place_tool,
    get_tripadvisor_reviews_tool,
    get_tripadvisor_search_tool,
    get_yelp_business_search_tool,
    get_yelp_place_tool,
    get_yelp_reviews_tool,
)
from .config import ToolsConfig
from .forumscout_search import (
    ForumScoutTool,
    get_forum_search_tool,
    get_google_forums_search_tool,
    get_google_news_search_tool,
    get_reddit_comments_search_tool,
    get_reddit_posts_search_tool,
)
from .open_research_sources import (
    OpenResearchTool,
    get_fetch_url_text_tool,
    get_gdelt_news_search_tool,
    get_wikidata_entity_search_tool,
    get_wikidata_sparql_tool,
)
from .open_travel_search import (
    OpenTravelSearchTool,
    get_open_meteo_air_quality_tool,
    get_open_meteo_marine_tool,
    get_public_holidays_tool,
    get_ticketmaster_events_tool,
    get_transitous_route_tool,
    get_travel_currency_tool,
    get_wikivoyage_search_tool,
)
from .playwright_fetch import PlaywrightFetchTool, get_playwright_fetch_tool, is_playwright_available
from .research_artifacts import (
    add_research_artifact_tools,
    cleanup_research_artifacts,
    reset_research_artifact_context,
    set_research_artifact_context,
)
from .serpapi_keys import has_serpapi_keys
from .serpapi_web_search import (
    SerpApiWebSearchTool,
    get_bing_web_search_tool,
    get_google_web_search_tool,
)
from .subagent_config import (
    OBJECTIVE_EVIDENCE_RULES,
    append_evidence_dir_instruction,
    append_research_tool_usage,
    build_subagent_config,
    create_subagent_evidence_dir,
    create_subagent_session_id,
    enforce_prompt_str_or_list_schema,
    inherit_subagent_limits,
    normalize_subagent_prompts,
    record_researcher_response,
    reconcile_research_artifacts,
    run_parallel_subagent_prompts,
    subagent_launch_block_reason,
)
from .task_steps_manager_state import current_session_id
from .task_steps_manager_tool import TaskStepsManagerTool, get_task_steps_manager_tool
from .telemetry import current_log_context, run_with_tool_logging
from .travel_search import (
    TravelSearchTool,
    get_amadeus_hotel_prices_tool,
    get_amadeus_hotel_sentiments_tool,
    get_booking_cities_tool,
    get_booking_stay_details_tool,
    get_booking_stay_reviews_tool,
    get_booking_stays_search_tool,
    get_google_flights_search_tool,
    get_google_stay_details_tool,
    get_google_stay_reviews_tool,
    get_google_stays_search_tool,
    get_google_travel_explore_tool,
    get_open_meteo_forecast_tool,
    get_opentripmap_place_details_tool,
    get_opentripmap_places_search_tool,
)

try:
    from agents import function_tool
except ImportError:
    function_tool = None


_TRAVEL_AGENT_SYSTEM_PROMPT = """### RULES
- Your only job is evidence-backed trip research: flight options and prices, hotels and vacation rentals, property details, reviews/opinions, destinations, local places, weather, disruptions, and feasible itineraries.
- Begin by extracting the traveler's hard constraints: origin/destination, dates or flexibility, travelers/rooms, currency, budget, baggage, cabin, stop tolerance, lodging type, accessibility, interests, and pace. State missing constraints instead of inventing them.
- For flights, use structured flight results and price insights. Compare total price, duration, stops, times, airports, baggage/fare caveats, and emissions where available. Prices are volatile snapshots: record query date/parameters and require rechecking before purchase.
- Preserve the public, key-free Google Travel research URL emitted with each structured flight/stay query. Never expose a SerpAPI request URL or API key. A local artifact path proves internal provenance but is not a public citation URL.
- For lodging, search both hotels and vacation rentals when relevant. Compare nightly and total rates, taxes/fee caveats, cancellation, location, rating volume, amenities, and booking sources. Prefer several independent structured sources when credentials permit: Google Hotels, Booking.com Demand API, and Amadeus offers/sentiments. Retrieve property details and recent/lowest reviews for finalists; do not imply that a vacation rental is an Airbnb listing unless the source identifies Airbnb. Airbnb has no open public shopping API, so label direct Airbnb web evidence as non-normalized and recheck it on Airbnb.
- Opinions are subjective evidence. Triangulate structured reviews with Tripadvisor/Maps/Yelp and forums or Reddit when available; separate recurring themes from isolated complaints, sponsored content, and stale reviews.
- Use official airline, hotel, transport, tourism, government, or venue pages for schedules, restrictions, and policies. Fetch the underlying source before relying on it. Use rendered-page access when normal fetching cannot read a JavaScript page.
- Check destination weather only when dates fall within the forecast horizon, and clearly label forecasts as predictions. For distant dates, research seasonal climate separately instead of presenting a forecast. For beach, surf, sailing, or ferry plans, check marine forecasts while warning that models do not replace local flags, operators, or navigation guidance.
- Build itineraries that are geographically and temporally feasible. Use structured public-transport routing when coverage exists, then include transfer/check-in buffers, opening-day uncertainty, recovery time after long flights, and a cost breakdown with current reference-currency conversion plus inclusions/exclusions.
- Use Wikivoyage as an openly licensed orientation source, not as authority for volatile opening hours, prices, entry rules, or safety. Consider destination air quality and pollen when relevant, clearly treating forecasts as predictions rather than medical advice. Check public holidays for closure/crowd risk, and use event inventory only as partial coverage that must be confirmed with the official organizer.
- For cultural or historical travel, use the scoped Wikidata entity and SPARQL lookups to disambiguate places, aliases, coordinates, heritage relationships, and identifiers. Treat graph claims as orientation evidence and verify consequential historical or visitor information with official heritage, museum, archive, or tourism sources.
- Never book, purchase, submit traveler data, or claim availability is guaranteed. Mention sources, and mention artifact filenames only when artifacts are preserved, without naming internal tool names.
""" + OBJECTIVE_EVIDENCE_RULES


class TravelResearchAgentTool:
    def __init__(
        self,
        config: ToolsConfig,
        model_name: str = "",
        fallback_model: str = "",
        model_provider: str = "",
        max_turns: int = 40,
        self_critique_enabled: bool = False,
        self_critique_rounds: int = 0,
    ):
        self.config = config
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.model_provider = str(model_provider or "").strip()
        if not self.model_provider:
            raise ValueError("model_provider must be defined")
        self.max_turns = max(2, int(max_turns or 40))
        self.self_critique_rounds = max(0, int(self_critique_rounds or 0))
        self.self_critique_enabled = bool(self_critique_enabled or self.self_critique_rounds > 0)
        self.travel = TravelSearchTool(config)
        self.business = BusinessSearchTool(config)
        self.web = SerpApiWebSearchTool(config)
        self.forum = ForumScoutTool(config)
        self.open = OpenResearchTool(config)
        self.open_travel = OpenTravelSearchTool(config)

    def _resolved_model(self) -> Optional[str]:
        configured = str(self.model_name or "").strip()
        if configured:
            return configured
        fallback = str(self.fallback_model or "").strip()
        return fallback or None

    def _build_subagent_tools(self):
        if function_tool is None:
            raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")
        tools = []
        if getattr(self.config, "task_steps_manager_enabled", True):
            tools.append(get_task_steps_manager_tool(TaskStepsManagerTool(self.config)))

        # Open-Meteo and page/news access work without SerpAPI.
        tools.append(get_open_meteo_forecast_tool(self.travel))
        tools.append(get_open_meteo_air_quality_tool(self.open_travel))
        tools.append(get_open_meteo_marine_tool(self.open_travel))
        tools.append(get_public_holidays_tool(self.open_travel))
        tools.append(get_travel_currency_tool(self.open_travel))
        tools.append(get_wikivoyage_search_tool(self.open_travel))
        tools.append(get_transitous_route_tool(self.open_travel))
        tools.append(get_fetch_url_text_tool(self.open))
        tools.append(get_gdelt_news_search_tool(self.open))
        tools.append(get_wikidata_entity_search_tool(self.open))
        tools.append(get_wikidata_sparql_tool(self.open))

        if os.environ.get("TICKETMASTER_API_KEY"):
            tools.append(get_ticketmaster_events_tool(self.open_travel))

        if os.environ.get("BOOKING_API_TOKEN") and os.environ.get("BOOKING_AFFILIATE_ID"):
            tools.extend(
                [
                    get_booking_cities_tool(self.travel),
                    get_booking_stays_search_tool(self.travel),
                    get_booking_stay_details_tool(self.travel),
                    get_booking_stay_reviews_tool(self.travel),
                ]
            )
        if os.environ.get("AMADEUS_CLIENT_ID") and os.environ.get("AMADEUS_CLIENT_SECRET"):
            tools.extend(
                [
                    get_amadeus_hotel_prices_tool(self.travel),
                    get_amadeus_hotel_sentiments_tool(self.travel),
                ]
            )
        if os.environ.get("OPENTRIPMAP_API_KEY"):
            tools.extend(
                [
                    get_opentripmap_places_search_tool(self.travel),
                    get_opentripmap_place_details_tool(self.travel),
                ]
            )

        if has_serpapi_keys(os.environ.get("SERPAPI_API_KEY", "")):
            tools.extend(
                [
                    get_google_flights_search_tool(self.travel),
                    get_google_travel_explore_tool(self.travel),
                    get_google_stays_search_tool(self.travel),
                    get_google_stay_details_tool(self.travel),
                    get_google_stay_reviews_tool(self.travel),
                    get_google_maps_search_tool(self.business),
                    get_google_maps_reviews_tool(self.business),
                    get_yelp_business_search_tool(self.business),
                    get_yelp_place_tool(self.business),
                    get_yelp_reviews_tool(self.business),
                    get_apple_maps_search_tool(self.business),
                    get_apple_maps_place_tool(self.business),
                    get_tripadvisor_search_tool(self.business),
                    get_tripadvisor_place_tool(self.business),
                    get_tripadvisor_reviews_tool(self.business),
                    get_google_web_search_tool(self.web),
                    get_bing_web_search_tool(self.web),
                    get_google_forums_search_tool(self.forum),
                    get_google_news_search_tool(self.forum),
                ]
            )

        # ForumScout calls return a clear configuration error when no key exists,
        # while still allowing deployments to add the key without changing tools.
        tools.extend(
            [
                get_forum_search_tool(self.forum),
                get_reddit_posts_search_tool(self.forum),
                get_reddit_comments_search_tool(self.forum),
            ]
        )
        if self.config.playwright_enabled and is_playwright_available():
            tools.append(get_playwright_fetch_tool(PlaywrightFetchTool(self.config)))
        add_research_artifact_tools(tools, self.config)
        return tools

    def _run_single(self, prompt: str, ctx: dict[str, Any], save_artifacts: bool = False) -> str:
        tools = self._build_subagent_tools()
        launch_block = subagent_launch_block_reason(
            parent_original_runtime_minutes=int(ctx.get("max_runtime_minutes") or 0),
            parent_remaining_runtime_minutes=float(ctx.get("remaining_runtime_minutes") or 0.0),
            parent_original_cost_usd=float(ctx.get("max_cost_usd") or 0.0),
            parent_remaining_cost_usd=float(ctx.get("remaining_cost_usd") or 0.0),
        )
        if launch_block:
            return launch_block
        effective_max_turns, effective_runtime_minutes, effective_cost_usd = inherit_subagent_limits(
            default_max_turns=self.max_turns,
            parent_max_turns=int(ctx.get("max_turns") or 0),
            parent_remaining_runtime_minutes=float(ctx.get("remaining_runtime_minutes") or 0.0),
            parent_remaining_cost_usd=float(ctx.get("remaining_cost_usd") or 0.0),
        )
        parent_memory_max_messages = max(1, int(ctx.get("memory_max_messages") or 8))
        parent_memory_reset_to_messages = max(
            1,
            int(ctx.get("memory_reset_to_messages") or parent_memory_max_messages),
        )
        parent_root_session_id = str(ctx.get("session_id") or "").strip()
        evidence_dir = create_subagent_evidence_dir("travel", parent_root_session_id)
        prompt = append_evidence_dir_instruction(
            prompt,
            evidence_dir,
            "Now research the trip using structured flight/stay prices, details, reviews, local sources, weather, and official web pages before synthesizing an itinerary!",
            save_artifacts=save_artifacts,
        )
        max_tools = int(getattr(self.config, "travel_max_tools_used", 0) or 0)
        overrides = {
            "agent": {
                "self_critique_enabled": self.self_critique_enabled,
                "self_critique_rounds": self.self_critique_rounds,
                "max_runtime_minutes": effective_runtime_minutes,
                "max_cost_usd": effective_cost_usd,
                "sub_action": "travel",
            },
            "session": {
                "max_turns": effective_max_turns,
                "memory_max_messages": parent_memory_max_messages,
                "memory_reset_to_messages": parent_memory_reset_to_messages,
                "long_term_memory_enabled": False,
                "long_term_memory_max_chars": 0,
                "long_term_memory_dir": "",
            },
            "tools": {
                "travel_enabled": True,
                "travel_google_flights_enabled": True,
                "travel_google_travel_explore_enabled": True,
                "travel_google_hotels_enabled": True,
                "travel_google_hotels_reviews_enabled": True,
                "travel_booking_enabled": True,
                "travel_amadeus_enabled": True,
                "travel_opentripmap_enabled": True,
                "travel_open_meteo_enabled": True,
                "travel_open_meteo_air_quality_enabled": True,
                "travel_open_meteo_marine_enabled": True,
                "travel_public_holidays_enabled": True,
                "travel_ticketmaster_enabled": bool(os.environ.get("TICKETMASTER_API_KEY")),
                "travel_frankfurter_enabled": True,
                "travel_wikivoyage_enabled": True,
                "travel_transitous_enabled": True,
                "travel_max_tools_used": max_tools,
                "max_tools_used": max_tools,
                "business_google_maps_enabled": True,
                "business_google_maps_reviews_enabled": True,
                "business_yelp_enabled": True,
                "business_apple_maps_enabled": True,
                "business_tripadvisor_enabled": True,
                "serpapi_google_web_enabled": True,
                "serpapi_bing_web_enabled": True,
                "social_network_google_forums_enabled": True,
                "social_network_google_news_enabled": True,
                "open_research_fetch_url_text_enabled": True,
                "open_research_gdelt_enabled": True,
                "open_research_wikidata_enabled": True,
                "knowledge_graph_enabled": False,
                "business_enabled": False,
                "product_enabled": False,
                "scientific_enabled": False,
                "social_network_enabled": False,
                "websearcher_enabled": False,
                "exec_enabled": False,
                "pdf_text_enabled": False,
            },
            "env": {
                "CHACK_RESEARCH_DATA_DIR": evidence_dir,
                "CHACK_RESEARCH_SAVE_ARTIFACTS": "1" if save_artifacts else "0",
            },
        }
        main_action = str(ctx.get("main_action") or "").strip()
        if main_action:
            overrides["agent"]["main_action"] = main_action
        config = build_subagent_config(
            self.config,
            model_name=self._resolved_model() or "",
            model_provider=self.model_provider,
            max_turns=effective_max_turns,
            system_prompt=_TRAVEL_AGENT_SYSTEM_PROMPT,
            overrides=overrides,
        )
        parent_task_session_id = current_session_id()
        subagent_session_id = create_subagent_session_id("travel", parent_root_session_id)
        from chack_agent import Chack

        chack = Chack(config)
        artifact_context_tokens = set_research_artifact_context(
            evidence_dir,
            os.environ.get("CHACK_RESEARCH_MASTER_DIR", "").strip(),
        )
        try:
            result = chack.run(
                session_id=subagent_session_id,
                text=prompt,
                min_tools_used_override=0,
                max_tools_used_override=max_tools,
                enable_self_critique=None,
                require_task_steps_manager_init_first=bool(
                    getattr(self.config, "task_steps_manager_enabled", True)
                ),
                tools_override=tools,
                system_prompt_override=config.system_prompt,
                usage_session_id=parent_task_session_id,
            )
            output = result.output.strip() if result.output else "ERROR: sub-agent returned an empty response."
            if output.startswith("ERROR:"):
                return output
            tool_counts = result.tool_counts.copy()

            def _run_followup(followup: str, output_schema_json=None) -> str:
                followup_result = chack.run(
                    session_id=subagent_session_id,
                    text=followup,
                    min_tools_used_override=0,
                    max_tools_used_override=max_tools,
                    enable_self_critique=False,
                    self_critique_rounds_override=0,
                    require_task_steps_manager_init_first=False,
                    tools_override=tools,
                    system_prompt_override=config.system_prompt,
                    usage_session_id=parent_task_session_id,
                    output_schema_json_override=output_schema_json,
                )
                tool_counts.update(followup_result.tool_counts)
                return (followup_result.output or "").strip()

            output = reconcile_research_artifacts(
                output,
                evidence_dir=evidence_dir,
                save_artifacts=bool(
                    save_artifacts
                    and getattr(self.config, "research_strict_artifact_manifest", True)
                ),
                run_followup=_run_followup,
            )
            return append_research_tool_usage(output, tool_counts)
        finally:
            cleanup_research_artifacts(evidence_dir, save_artifacts=save_artifacts)
            reset_research_artifact_context(artifact_context_tokens)

    def run(self, prompt: str | list[str], save_artifacts: bool = False) -> str:
        prompts, error = normalize_subagent_prompts(prompt, min_chars=500, max_prompts=3)
        if error:
            return error
        ctx = current_log_context()
        return run_parallel_subagent_prompts(
            prompts,
            lambda item: self._run_single(item, ctx, save_artifacts=save_artifacts),
        )


def get_travel_research_tool(helper: TravelResearchAgentTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="travel_research")
    def travel_research(prompt: str | list[str], save_artifacts: bool = False) -> str:
        """Run a dedicated travel-research sub-agent.

        Use it for evidence-backed flight and lodging price comparisons, hotels/vacation
        rentals, reviews and traveler opinions, destination research, weather, disruption
        checks, cost estimates, and feasible itineraries.

        Args:
            prompt: One detailed request or up to 3 detailed requests, each at least 500 characters. Include origin/destination, dates/flexibility, traveler counts, currency/budget, transport/lodging constraints, interests, evidence expectations, and caveats to test.
            save_artifacts: Preserve the auditable evidence folder when true.

        Output: Standard Chack researcher JSON with worked status, conclusions, source-backed comparisons, caveats, and artifact metadata when preserved.
        """
        tool_input = {"prompt": prompt, "save_artifacts": save_artifacts}
        try:
            return run_with_tool_logging(
                "travel_research",
                tool_input,
                lambda: _run_and_record(
                    "travel_research",
                    helper.run(prompt=prompt, save_artifacts=save_artifacts),
                ),
            )
        except Exception as exc:
            return f"ERROR: travel_research failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(travel_research)
    tool.description = (
        f"{tool.description}\n\n"
        "Parameters: Provide one detailed prompt or up to three detailed prompts and choose whether to preserve artifacts.\n"
        "Output: Standard researcher JSON with worked status, evidence-backed travel conclusions, caveats, tool usage, and preserved artifact metadata when requested.\n"
        "Prices and availability are snapshots, not booking guarantees. The researcher does not purchase or submit traveler information."
    )
    return tool


def _run_and_record(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output
