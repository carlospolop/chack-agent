"""Opt-in live smoke tests for travel providers.

Run with:
    CHACK_LIVE_TRAVEL=1 python -m pytest -q tests/live_travel_api_test.py

Only providers whose credentials are present are exercised. The keyless test
always runs when the live-test switch is enabled.
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta

import pytest

from chack_tools.config import ToolsConfig
from chack_tools.forumscout_search import ForumScoutTool
from chack_tools.open_travel_search import OpenTravelSearchTool
from chack_tools.travel_search import TravelSearchTool


pytestmark = pytest.mark.skipif(
    os.environ.get("CHACK_LIVE_TRAVEL") != "1",
    reason="set CHACK_LIVE_TRAVEL=1 to run external travel-provider smoke tests",
)


def _dates(days_from_now: int = 30, nights: int = 3) -> tuple[str, str]:
    check_in = date.today() + timedelta(days=days_from_now)
    return check_in.isoformat(), (check_in + timedelta(days=nights)).isoformat()


def _assert_success(output: str) -> None:
    assert output.startswith("SUCCESS:"), output


def test_live_keyless_travel_apis(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    check_in, _ = _dates(days_from_now=1)
    helper = OpenTravelSearchTool(ToolsConfig(travel_max_results=2))
    weather = TravelSearchTool(ToolsConfig(travel_max_results=2))

    _assert_success(weather.get_weather_forecast("Madrid", forecast_days=3))
    _assert_success(helper.get_air_quality("Madrid", forecast_days=2))
    _assert_success(helper.convert_currency("EUR", "USD", amount=100, provider="ECB"))
    _assert_success(helper.search_wikivoyage("Madrid", max_results=2))
    _assert_success(helper.get_marine_forecast("Palma de Mallorca", forecast_days=2))
    holiday_start, holiday_end = _dates(days_from_now=1, nights=30)
    _assert_success(helper.get_public_holidays("ES", holiday_start, holiday_end))
    _assert_success(
        helper.plan_public_transport(
            "Puerta del Sol, Madrid",
            "Avenida de America, Madrid",
            departure_time=f"{check_in}T09:00:00+02:00",
            max_results=2,
        )
    )


@pytest.mark.skipif(not os.environ.get("SERPAPI_API_KEY"), reason="SERPAPI_API_KEY missing")
def test_live_serpapi_travel_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    check_in, check_out = _dates()
    helper = TravelSearchTool(ToolsConfig(travel_max_results=2))

    _assert_success(
        helper.search_flights(
            "MAD", "BCN", check_in, return_date=check_out, currency="EUR", max_results=2
        )
    )
    _assert_success(helper.explore_destinations("MAD", currency="EUR", max_results=2))
    hotels = helper.search_stays(
        "Madrid", check_in, check_out, currency="EUR", max_results=2
    )
    _assert_success(hotels)
    _assert_success(
        helper.search_stays(
            "Madrid",
            check_in,
            check_out,
            currency="EUR",
            vacation_rentals=True,
            max_results=2,
        )
    )
    token = re.search(r"property_token:\s*(\S+)", hotels)
    assert token, hotels
    _assert_success(
        helper.get_stay_details(
            "Madrid", token.group(1), check_in, check_out, currency="EUR"
        )
    )
    _assert_success(helper.get_stay_reviews(token.group(1), max_results=2))


@pytest.mark.skipif(
    not (os.environ.get("FORUMSCOUT_API_KEY") and os.environ.get("SERPAPI_API_KEY")),
    reason="ForumScout/SerpAPI credentials missing",
)
def test_live_travel_opinion_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    helper = ForumScoutTool(ToolsConfig(forumscout_max_results=2))
    _assert_success(helper.forum_search("Madrid hotel neighborhood opinions", time="year"))
    _assert_success(
        helper.reddit_posts_search("Madrid hotels best neighborhood", sort_by="relevance")
    )
    _assert_success(helper.reddit_comments_search("Madrid hotel neighborhood", sort_by="score"))


@pytest.mark.skipif(
    not (os.environ.get("BOOKING_API_TOKEN") and os.environ.get("BOOKING_AFFILIATE_ID")),
    reason="Booking.com Demand API credentials missing",
)
def test_live_booking_demand_api(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    check_in, check_out = _dates(days_from_now=60)
    helper = TravelSearchTool(ToolsConfig(travel_max_results=2))
    cities = helper.find_booking_cities("nl", "Amsterdam", max_results=2)
    _assert_success(cities)
    city_id = re.search(r"city_id:\s*(-?\d+)", cities)
    assert city_id, cities
    stays = helper.search_booking_stays(
        int(city_id.group(1)), check_in, check_out, "nl", currency="EUR", max_results=2
    )
    _assert_success(stays)
    accommodation_id = re.search(r"accommodation_id:\s*(\d+)", stays)
    if accommodation_id:
        _assert_success(helper.get_booking_stay_details(accommodation_id.group(1)))
        _assert_success(helper.get_booking_stay_reviews(int(accommodation_id.group(1)), max_results=2))


@pytest.mark.skipif(
    not (os.environ.get("AMADEUS_CLIENT_ID") and os.environ.get("AMADEUS_CLIENT_SECRET")),
    reason="Amadeus credentials missing",
)
def test_live_amadeus_hotel_apis(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    check_in, check_out = _dates(days_from_now=45)
    helper = TravelSearchTool(ToolsConfig(travel_max_results=3))
    offers = helper.search_amadeus_hotel_prices(
        "PAR", check_in, check_out, currency="EUR", max_results=3
    )
    _assert_success(offers)
    hotel_ids = re.findall(r"hotel_id:\s*([A-Z0-9]+)", offers)
    if hotel_ids:
        _assert_success(helper.get_amadeus_hotel_sentiments(",".join(hotel_ids[:3])))


@pytest.mark.skipif(not os.environ.get("OPENTRIPMAP_API_KEY"), reason="OpenTripMap key missing")
def test_live_opentripmap(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    helper = TravelSearchTool(ToolsConfig(travel_max_results=3))
    places = helper.search_opentripmap_places("Paris", kinds="museums", max_results=3)
    _assert_success(places)
    xid = re.search(r"xid:\s*(\S+)", places)
    assert xid, places
    _assert_success(helper.get_opentripmap_place_details(xid.group(1)))


@pytest.mark.skipif(not os.environ.get("TICKETMASTER_API_KEY"), reason="TICKETMASTER_API_KEY missing")
def test_live_ticketmaster_events(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    start, end = _dates(days_from_now=14, nights=30)
    helper = OpenTravelSearchTool(ToolsConfig(travel_max_results=2))
    _assert_success(helper.search_ticketmaster_events("Madrid", start, end, max_results=2))
