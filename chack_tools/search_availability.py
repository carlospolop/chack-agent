"""Register keyed search tools only when their key can serve a request.

A tool whose provider key is missing (or, for SerpAPI, whose every key is
exhausted) only answers "key not configured"/"no usable key", so offering it
wastes the agent's turns. Every tool list a research agent receives passes
through :func:`drop_unavailable_search_tools`, evaluated when the list is
built, i.e. once per agent run: SerpAPI tools come back on a later run once a
key is usable again (``serpapi_keys`` re-checks parked keys after its TTL).

When an agent that should search the web is left without any paid engine,
:func:`with_keyless_web_search` adds the keyless ``web_search`` tool
(``exa_web_search``) unless ``keyless_web_search_enabled`` or
``native_web_search_enabled`` is false.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from .serpapi_keys import usable_serpapi_keys

# Tools whose only backend is SerpAPI.
SERPAPI_TOOL_NAMES = frozenset(
    {
        # General web search.
        "search_google_web",
        "search_bing_web",
        "search_google_ai_mode",
        "search_bing_copilot",
        # Scientific / video.
        "search_google_patents",
        "search_google_patents_details",
        "search_google_scholar",
        "search_google_scholar_cite",
        "search_youtube_videos",
        "get_youtube_video_details",
        "get_youtube_video_transcript",
        # Social / news.
        "search_google_forums",
        "search_google_news",
        "search_google_trends",
        "search_google_trends_trending_now",
        "search_google_videos",
        "get_instagram_profile",
        "get_facebook_profile",
        "tiktok_web_search",
        "bluesky_web_search",
        # Business / product.
        "search_google_finance",
        "search_google_finance_markets",
        "search_google_maps_businesses",
        "get_google_maps_reviews",
        "search_yelp_businesses",
        "get_yelp_place",
        "get_yelp_reviews",
        "search_apple_maps_businesses",
        "get_apple_maps_place",
        "search_google_ads",
        "search_google_ads_transparency",
        "search_google_shopping",
        "search_google_shopping_light",
        "get_google_immersive_product",
        "search_amazon_products",
        "get_amazon_product",
        "search_walmart_products",
        "get_walmart_product",
        "search_ebay_products",
        "get_ebay_product",
        "search_home_depot_products",
        "get_home_depot_product",
        "search_tripadvisor",
        "get_tripadvisor_place",
        "get_tripadvisor_reviews",
        "search_google_lens_products",
        # Travel.
        "search_google_flights",
        "explore_google_travel_destinations",
        "search_google_stays",
        "get_google_stay_details",
        "get_google_stay_reviews",
    }
)

# Tools that need every listed environment variable set.
_FORUMSCOUT = ("FORUMSCOUT_API_KEY",)
_BOOKING = ("BOOKING_API_TOKEN", "BOOKING_AFFILIATE_ID")
_AMADEUS = ("AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET")
_OPENTRIPMAP = ("OPENTRIPMAP_API_KEY",)
KEYED_TOOL_ENV: dict[str, tuple[str, ...]] = {
    "brave_search": ("BRAVE_API_KEY",),
    "forum_search": _FORUMSCOUT,
    "linkedin_search": _FORUMSCOUT,
    "instagram_search": _FORUMSCOUT,
    "reddit_posts_search": _FORUMSCOUT,
    "reddit_comments_search": _FORUMSCOUT,
    "x_search": _FORUMSCOUT,
    "find_booking_cities": _BOOKING,
    "search_booking_stays": _BOOKING,
    "get_booking_stay_details": _BOOKING,
    "get_booking_stay_reviews": _BOOKING,
    "search_amadeus_hotel_prices": _AMADEUS,
    "get_amadeus_hotel_sentiments": _AMADEUS,
    "search_destination_places": _OPENTRIPMAP,
    "get_destination_place_details": _OPENTRIPMAP,
    "search_ticketmaster_events": ("TICKETMASTER_API_KEY",),
}

# Tools that search the open web for any query.
GENERAL_WEB_SEARCH_TOOL_NAMES = frozenset({"search_google_web", "search_bing_web", "brave_search", "web_search"})


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")


def _env_set(names: Iterable[str]) -> bool:
    return all(str(os.environ.get(name, "") or "").strip() for name in names)


def serpapi_usable() -> bool:
    """At least one configured SerpAPI key is not known to be exhausted."""
    return bool(usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", "")))


def drop_unavailable_search_tools(tools: list[Any]) -> list[Any]:
    """Return ``tools`` without the keyed tools that cannot serve a request now."""
    has_serpapi: bool | None = None
    kept: list[Any] = []
    for tool in tools:
        name = tool_name(tool)
        if name in SERPAPI_TOOL_NAMES:
            if has_serpapi is None:
                has_serpapi = serpapi_usable()
            if not has_serpapi:
                continue
        required = KEYED_TOOL_ENV.get(name)
        if required and not _env_set(required):
            continue
        kept.append(tool)
    return kept


def keyless_web_search_allowed(config: Any) -> bool:
    return bool(getattr(config, "native_web_search_enabled", True)) and bool(
        getattr(config, "keyless_web_search_enabled", True)
    )


def with_keyless_web_search(tools: list[Any], config: Any) -> list[Any]:
    """Add the keyless ``web_search`` tool when ``tools`` has no general web search."""
    if any(tool_name(tool) in GENERAL_WEB_SEARCH_TOOL_NAMES for tool in tools):
        return tools
    if not keyless_web_search_allowed(config):
        return tools
    from .exa_web_search import get_web_search_tools

    return list(tools) + [tool for tool in get_web_search_tools() if tool_name(tool) == "web_search"]


def available_research_tools(tools: list[Any], config: Any, *, web_search: bool = False) -> list[Any]:
    """Drop unusable keyed tools; with ``web_search``, keep one working web search."""
    kept = drop_unavailable_search_tools(tools)
    return with_keyless_web_search(kept, config) if web_search else kept


def keyless_web_search_note(tools: list[Any]) -> str:
    """A prompt line for runs whose only general web search is the keyless one."""
    names = {tool_name(tool) for tool in tools}
    if "web_search" not in names or names & (GENERAL_WEB_SEARCH_TOOL_NAMES - {"web_search"}):
        return ""
    return (
        "Web search in this run: `web_search` is the only search engine available; use it for discovery, "
        "then fetch and read the pages it finds."
    )
