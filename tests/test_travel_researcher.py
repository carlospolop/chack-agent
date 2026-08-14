import json
import asyncio

from agents.tool import ToolContext
from chack_tools.agents_toolset import AgentsToolset
from chack_tools.config import ToolsConfig
from chack_tools.researcher_administrator_agent import (
    RESEARCHER_REGISTRY,
    ResearcherAdministratorAgentTool,
)
from chack_tools.open_travel_search import OpenTravelSearchTool, get_travel_currency_tool
from chack_tools.forumscout_search import ForumScoutTool
from chack_tools.travel_research_agent import TravelResearchAgentTool
from chack_tools.travel_search import TravelSearchTool, get_google_flights_search_tool


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            response = type("Response", (), {"status_code": self.status_code})()
            raise requests.exceptions.HTTPError(response=response)


def _tool_names(tools):
    return {
        str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")
        for tool in tools
    }


def test_forumscout_empty_base_url_uses_default(monkeypatch):
    monkeypatch.setenv("FORUMSCOUT_BASE_URL", "")
    assert ForumScoutTool(ToolsConfig())._base_url() == "https://forumscout.app"


def test_forumscout_retries_transient_timeout(monkeypatch, tmp_path):
    import requests

    monkeypatch.setenv("FORUMSCOUT_API_KEY", "forum-key")
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("chack_tools.forumscout_search.time_module.sleep", lambda _seconds: None)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise requests.exceptions.Timeout()
        return FakeResponse([
            {"title": "Madrid areas", "url": "https://example.test/thread", "snippet": "Local opinions"}
        ])

    monkeypatch.setattr("chack_tools.forumscout_search.requests.get", fake_get)
    result = ForumScoutTool(ToolsConfig(forumscout_max_results=2)).forum_search("Madrid hotels")

    assert len(calls) == 2
    assert calls[1][1]["params"]["keyword"] == "Madrid hotels"
    assert result.startswith("SUCCESS: ForumScout results")


def test_forumscout_treats_documented_empty_message_as_no_results(monkeypatch, tmp_path):
    monkeypatch.setenv("FORUMSCOUT_API_KEY", "forum-key")
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        "chack_tools.forumscout_search.requests.get",
        lambda *_args, **_kwargs: FakeResponse("No posts found"),
    )

    result = ForumScoutTool(ToolsConfig()).forum_search("quiet query")

    assert result.startswith("SUCCESS: No ForumScout results found")


def test_open_travel_retries_transient_provider_failures(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            return FakeResponse({}, status_code=503)
        return FakeResponse({"ok": True})

    monkeypatch.setattr("chack_tools.open_travel_search.requests.get", fake_get)
    monkeypatch.setattr("chack_tools.open_travel_search.time.sleep", lambda _seconds: None)
    assert OpenTravelSearchTool._get_json("https://example.test", {}, 5, "Example") == {"ok": True}
    assert len(calls) == 2


def test_travel_function_wrappers_do_not_leak_helper_closure():
    class TravelStub:
        def search_flights(self, **kwargs):
            assert "helper" not in kwargs
            return f"SUCCESS: {kwargs['departure_id']} to {kwargs['arrival_id']}"

    class OpenTravelStub:
        def convert_currency(self, **kwargs):
            assert "helper" not in kwargs
            return f"SUCCESS: {kwargs['base_currency']} to {kwargs['quote_currency']}"

    flight_tool = get_google_flights_search_tool(TravelStub())
    currency_tool = get_travel_currency_tool(OpenTravelStub())
    flight_context = ToolContext(None, None, "search_google_flights", "call-flight", "{}")
    currency_context = ToolContext(None, None, "convert_travel_currency", "call-currency", "{}")
    flight = asyncio.run(
        flight_tool.on_invoke_tool(
            flight_context,
            json.dumps(
                {
                    "departure_id": "MAD",
                    "arrival_id": "BCN",
                    "outbound_date": "2026-08-10",
                }
            ),
        )
    )
    currency = asyncio.run(
        currency_tool.on_invoke_tool(
            currency_context, json.dumps({"base_currency": "EUR", "quote_currency": "USD"})
        )
    )

    assert flight == "SUCCESS: MAD to BCN"
    assert currency == "SUCCESS: EUR to USD"


def test_google_flights_params_price_insights_and_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs["params"]
        return FakeResponse(
            {
                "search_parameters": {"currency": "EUR"},
                "price_insights": {
                    "lowest_price": 180,
                    "price_level": "low",
                    "typical_price_range": [220, 310],
                },
                "best_flights": [
                    {
                        "price": 190,
                        "total_duration": 165,
                        "flights": [
                            {
                                "departure_airport": {"id": "MAD", "time": "2026-09-01 08:00"},
                                "arrival_airport": {"id": "CDG", "time": "2026-09-01 10:45"},
                                "airline": "Example Air",
                                "flight_number": "EA 101",
                            }
                        ],
                        "departure_token": "return-token",
                    }
                ],
            }
        )

    monkeypatch.setattr("chack_tools.travel_search.requests.get", fake_get)
    helper = TravelSearchTool(ToolsConfig(travel_max_results=5))

    result = helper.search_flights(
        "MAD",
        "CDG",
        "2026-09-01",
        return_date="2026-09-08",
        currency="eur",
        stops=1,
        sort_by=2,
        bags=1,
        deep_search=True,
    )

    params = captured["params"]
    assert captured["url"] == "https://serpapi.com/search"
    assert params["engine"] == "google_flights"
    assert params["type"] == 1
    assert params["return_date"] == "2026-09-08"
    assert params["currency"] == "EUR"
    assert params["stops"] == 1
    assert params["sort_by"] == 2
    assert params["deep_search"] == "true"
    assert params["api_key"] == "serp-key"
    assert "Price insight: lowest 180 EUR" in result
    assert "Example Air EA 101" in result
    assert "selection_token: return-token" in result
    assert "https://www.google.com/travel/flights?" in result and "curr=EUR" in result
    assert list((tmp_path / "google-flights").glob("MAD_CDG_2026-09-01_2026-09-08_*.json"))


def test_google_stays_vacation_rentals_and_review_details(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    calls = []

    def fake_get(url, **kwargs):
        params = kwargs["params"]
        calls.append(params)
        if params["engine"] == "google_hotels_reviews":
            return FakeResponse(
                {
                    "reviews": [
                        {
                            "rating": 2,
                            "best_rating": 5,
                            "date": "a week ago",
                            "source": "Google",
                            "user": {"name": "Traveler"},
                            "description": "Noisy at night",
                            "subratings": {"rooms": 2, "location": 5},
                        }
                    ]
                }
            )
        if "property_token" in params:
            return FakeResponse(
                {
                    "property": {
                        "name": "Central Apartment",
                        "description": "Walkable apartment",
                        "rate_per_night": {"lowest": "€120"},
                        "prices": [{"source": "Example", "rate_per_night": "€120"}],
                    }
                }
            )
        return FakeResponse(
            {
                "search_parameters": {"currency": "EUR"},
                "properties": [
                    {
                        "type": "vacation rental",
                        "name": "Central Apartment",
                        "rate_per_night": {"lowest": "€120"},
                        "total_rate": {"lowest": "€480"},
                        "overall_rating": 4.4,
                        "reviews": 88,
                        "essential_info": ["Sleeps 4", "2 bedrooms"],
                        "property_token": "property-token",
                        "prices": [{"source": "Example", "rate_per_night": {"lowest": "€120"}}],
                    }
                ],
                "serpapi_pagination": {"next_page_token": "next-token"},
            }
        )

    monkeypatch.setattr("chack_tools.travel_search.requests.get", fake_get)
    helper = TravelSearchTool(ToolsConfig(travel_max_results=5))

    search = helper.search_stays(
        "Paris",
        "2026-09-01",
        "2026-09-05",
        vacation_rentals=True,
        bedrooms=2,
        bathrooms=1,
        sort_by=3,
        currency="EUR",
    )
    details = helper.get_stay_details(
        "Paris",
        "property-token",
        "2026-09-01",
        "2026-09-05",
        currency="EUR",
    )
    reviews = helper.get_stay_reviews("property-token", sort_by=4)

    assert calls[0]["engine"] == "google_hotels"
    assert calls[0]["vacation_rentals"] == "true"
    assert calls[0]["bedrooms"] == 2
    assert calls[0]["bathrooms"] == 1
    assert calls[1]["property_token"] == "property-token"
    assert calls[2]["engine"] == "google_hotels_reviews"
    assert calls[2]["sort_by"] == 4
    assert "Central Apartment" in search
    assert "https://www.google.com/travel/search?" in search and "checkin=2026-09-01" in search
    assert "total €480" in search
    assert "Next page token: next-token" in search
    assert "Walkable apartment" in details
    assert "Noisy at night" in reviews
    assert list((tmp_path / "google-vacation-rentals").glob("Paris_2026-09-01_2026-09-05_*.json"))
    assert list((tmp_path / "google-hotel-details").glob("property-token_*.json"))
    assert list((tmp_path / "google-hotel-reviews").glob("property-token_*.json"))


def test_open_meteo_forecast_is_keyless_and_writes_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        if "geocoding-api" in url:
            return FakeResponse(
                {
                    "results": [
                        {
                            "name": "Madrid",
                            "admin1": "Madrid",
                            "country": "Spain",
                            "latitude": 40.4,
                            "longitude": -3.7,
                        }
                    ]
                }
            )
        return FakeResponse(
            {
                "daily_units": {
                    "temperature_2m_max": "°C",
                    "precipitation_probability_max": "%",
                    "precipitation_sum": "mm",
                    "wind_speed_10m_max": "km/h",
                },
                "daily": {
                    "time": ["2026-07-18"],
                    "weather_code": [1],
                    "temperature_2m_min": [20],
                    "temperature_2m_max": [34],
                    "precipitation_probability_max": [5],
                    "precipitation_sum": [0],
                    "wind_speed_10m_max": [18],
                    "uv_index_max": [9],
                },
            }
        )

    monkeypatch.setattr("chack_tools.travel_search.requests.get", fake_get)
    helper = TravelSearchTool(ToolsConfig())
    result = helper.get_weather_forecast("Madrid", forecast_days=10)

    assert calls[0][0] == "https://geocoding-api.open-meteo.com/v1/search"
    assert calls[1][0] == "https://api.open-meteo.com/v1/forecast"
    assert calls[1][1]["forecast_days"] == 10
    assert "Madrid, Madrid, Spain" in result
    assert "20-34 °C" in result
    assert list((tmp_path / "open-meteo").glob("Madrid_*.json"))


def test_open_meteo_retries_transient_503(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("chack_tools.travel_search.time.sleep", lambda _seconds: None)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return FakeResponse({}, status_code=503)
        if "geocoding-api" in url:
            return FakeResponse({"results": [{"name": "Madrid", "latitude": 40.4, "longitude": -3.7}]})
        return FakeResponse({"daily_units": {}, "daily": {"time": []}})

    monkeypatch.setattr("chack_tools.travel_search.requests.get", fake_get)
    result = TravelSearchTool(ToolsConfig()).get_weather_forecast("Madrid")

    assert calls.count("https://geocoding-api.open-meteo.com/v1/search") == 2
    assert result.startswith("SUCCESS: Open-Meteo forecast")


def test_booking_demand_api_search_details_and_reviews(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOOKING_API_TOKEN", "booking-token")
    monkeypatch.setenv("BOOKING_AFFILIATE_ID", "12345")
    monkeypatch.setenv("BOOKING_DEMAND_API_BASE_URL", "https://booking.test/3.1")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("common/locations/cities"):
            return FakeResponse(
                {"data": [{"id": -2140479, "name": {"en-gb": "Amsterdam"}}], "metadata": {}}
            )
        if url.endswith("accommodations/search"):
            return FakeResponse(
                {
                    "data": [
                        {
                            "id": 10507360,
                            "name": "Canal Hotel",
                            "currency": "EUR",
                            "price": {"book": 321.5},
                            "review_score": 8.9,
                            "url": "https://booking.example/hotel",
                        }
                    ]
                }
            )
        if url.endswith("accommodations/details"):
            return FakeResponse(
                {"data": [{"id": 10507360, "name": "Canal Hotel", "description": {"text": "Central"}}]}
            )
        return FakeResponse(
            {
                "data": [
                    {
                        "reviews": [
                            {"score": 9, "title": "Great stay", "pros": "Walkable", "cons": "Small room"}
                        ]
                    }
                ]
            }
        )

    monkeypatch.setattr("chack_tools.travel_search.requests.post", fake_post)
    helper = TravelSearchTool(ToolsConfig(travel_max_results=5))

    cities = helper.find_booking_cities("nl", "amster")
    stays = helper.search_booking_stays(
        -2140479,
        "2026-09-01",
        "2026-09-04",
        "es",
        children_ages="7",
        min_price=100,
        max_price=500,
    )
    details = helper.get_booking_stay_details("10507360")
    reviews = helper.get_booking_stay_reviews(10507360)

    assert calls[0][1]["headers"]["Authorization"] == "Bearer booking-token"
    assert calls[0][1]["headers"]["X-Affiliate-Id"] == "12345"
    search_body = calls[1][1]["json"]
    assert search_body["city"] == -2140479
    assert search_body["guests"]["children"] == [7]
    assert search_body["price"] == {"minimum": 100, "maximum": 500}
    assert calls[2][1]["json"]["extras"] == ["description", "facilities", "photos", "rooms"]
    assert calls[3][1]["json"]["accommodations"] == [10507360]
    assert "city_id: -2140479" in cities
    assert "price: 321.5 EUR" in stays
    assert "Central" in details
    assert "Small room" in reviews
    assert list((tmp_path / "booking-stays").glob("-2140479_2026-09-01_2026-09-04_*.json"))
    assert list((tmp_path / "booking-stay-reviews").glob("10507360_*.json"))


def test_amadeus_hotel_prices_and_sentiments(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AMADEUS_CLIENT_ID", "client-id")
    monkeypatch.setenv("AMADEUS_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("AMADEUS_BASE_URL", "https://amadeus.test")
    posts = []
    gets = []

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return FakeResponse({"access_token": "access-token", "expires_in": 1800})

    def fake_get(url, **kwargs):
        gets.append((url, kwargs))
        if url.endswith("hotels/by-city"):
            return FakeResponse(
                {"data": [{"hotelId": "HLPAR266", "name": "Paris Hotel"}]}
            )
        if url.endswith("hotel-offers"):
            return FakeResponse(
                {
                    "data": [
                        {
                            "available": True,
                            "hotel": {"hotelId": "HLPAR266", "name": "Paris Hotel"},
                            "offers": [
                                {
                                    "id": "OFFER1",
                                    "price": {"total": "420.00", "currency": "EUR"},
                                    "room": {"description": {"text": "Queen room"}},
                                }
                            ],
                        }
                    ]
                }
            )
        return FakeResponse(
            {
                "data": [
                    {
                        "hotelId": "HLPAR266",
                        "overallRating": 91,
                        "numberOfReviews": 218,
                        "numberOfRatings": 278,
                        "sentiments": {"location": 98, "service": 92},
                    }
                ]
            }
        )

    monkeypatch.setattr("chack_tools.travel_search.requests.post", fake_post)
    monkeypatch.setattr("chack_tools.travel_search.requests.get", fake_get)
    helper = TravelSearchTool(ToolsConfig(travel_max_results=5))

    prices = helper.search_amadeus_hotel_prices(
        "PAR", "2026-09-01", "2026-09-04", currency="EUR"
    )
    sentiments = helper.get_amadeus_hotel_sentiments("HLPAR266")

    assert len(posts) == 1
    assert posts[0][1]["data"]["grant_type"] == "client_credentials"
    assert all(call[1]["headers"]["Authorization"] == "Bearer access-token" for call in gets)
    assert gets[0][1]["params"]["cityCode"] == "PAR"
    assert gets[1][1]["params"]["hotelIds"] == "HLPAR266"
    assert gets[1][1]["params"]["checkOutDate"] == "2026-09-04"
    assert "420.00 EUR" in prices
    assert "overall: 91/100" in sentiments
    assert list((tmp_path / "amadeus-hotel-prices").glob("PAR_2026-09-01_2026-09-04_*.json"))
    assert list((tmp_path / "amadeus-hotel-sentiments").glob("HLPAR266_*.json"))


def test_opentripmap_places_and_details(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "otm-key")
    monkeypatch.setenv("OPENTRIPMAP_BASE_URL", "https://opentripmap.test/0.1")
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        if url.endswith("/geoname"):
            return FakeResponse({"name": "Paris", "lat": 48.8566, "lon": 2.3522})
        if "/radius" in url:
            return FakeResponse(
                [{"xid": "N123", "name": "Example Museum", "kinds": "museums", "rate": 3, "dist": 450}]
            )
        return FakeResponse(
            {
                "xid": "N123",
                "name": "Example Museum",
                "address": {"city": "Paris"},
                "wikipedia_extracts": {"text": "A notable museum."},
            }
        )

    monkeypatch.setattr("chack_tools.travel_search.requests.get", fake_get)
    helper = TravelSearchTool(ToolsConfig())
    places = helper.search_opentripmap_places("Paris", kinds="museums")
    details = helper.get_opentripmap_place_details("N123")

    assert calls[0][1]["apikey"] == "otm-key"
    assert calls[1][1]["kinds"] == "museums"
    assert calls[1][1]["lat"] == 48.8566
    assert "Example Museum" in places
    assert "A notable museum" in details
    assert list((tmp_path / "opentripmap-places").glob("Paris_*.json"))
    assert list((tmp_path / "opentripmap-place-details").glob("N123_*.json"))


def test_keyless_air_quality_currency_and_wikivoyage(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if "geocoding-api.open-meteo.com" in url:
            return FakeResponse(
                {"results": [{"name": "Madrid", "country": "Spain", "latitude": 40.4, "longitude": -3.7}]}
            )
        if "air-quality-api.open-meteo.com" in url:
            return FakeResponse(
                {
                    "timezone_abbreviation": "CEST",
                    "current_units": {"european_aqi": "EAQI", "pm2_5": "ug/m3"},
                    "current": {"time": "2026-07-19T12:00", "european_aqi": 26, "pm2_5": 4.3},
                    "hourly": {
                        "european_aqi": [20, 30],
                        "us_aqi": [18, 28],
                        "pm10": [8, 10],
                        "pm2_5": [4, 6],
                        "grass_pollen": [5, 7],
                    },
                }
            )
        if "frankfurter" in url:
            return FakeResponse({"date": "2026-07-18", "base": "EUR", "quote": "USD", "rate": 1.15})
        return FakeResponse(
            {
                "query": {
                    "pages": [
                        {
                            "title": "Madrid",
                            "fullurl": "https://en.wikivoyage.org/wiki/Madrid",
                            "extract": "Madrid is Spain's capital and a major cultural destination.",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("chack_tools.open_travel_search.requests.get", fake_get)
    helper = OpenTravelSearchTool(ToolsConfig(travel_max_results=5))

    air = helper.get_air_quality("Madrid", forecast_days=4)
    currency = helper.convert_currency("EUR", "USD", amount=100, provider="ECB")
    guide = helper.search_wikivoyage("Madrid")

    assert calls[1][1]["params"]["forecast_days"] == 4
    assert calls[2][0].endswith("/v2/rate/EUR/USD")
    assert calls[2][1]["params"]["providers"] == "ECB"
    assert calls[3][1]["params"]["generator"] == "search"
    assert "european_aqi: 26" in air
    assert "100 EUR = 115.0000 USD" in currency
    assert "Madrid is Spain's capital" in guide
    assert list((tmp_path / "open-meteo-air-quality").glob("Madrid_*.json"))
    assert list((tmp_path / "frankfurter-rates").glob("EUR_USD_2026-07-18_*.json"))
    assert list((tmp_path / "wikivoyage-guides").glob("en_Madrid_*.json"))


def test_transitous_public_transport_plan(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/geocode"):
            text = kwargs["params"]["text"]
            if text == "Puerta del Sol, Madrid":
                return FakeResponse([{"name": "Puerta del Sol", "lat": 40.4168, "lon": -3.7038}])
            return FakeResponse([{"name": "Avenida de America", "lat": 40.4379, "lon": -3.6795}])
        return FakeResponse(
            {
                "itineraries": [
                    {
                        "startTime": "2026-07-20T07:00:00Z",
                        "endTime": "2026-07-20T07:25:00Z",
                        "duration": 1500,
                        "transfers": 0,
                        "legs": [
                            {
                                "mode": "SUBWAY",
                                "routeShortName": "L2",
                                "agencyName": "Metro de Madrid",
                                "from": {"name": "Sol"},
                                "to": {"name": "Avenida de America"},
                                "startTime": "2026-07-20T07:00:00Z",
                                "endTime": "2026-07-20T07:25:00Z",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("chack_tools.open_travel_search.requests.get", fake_get)
    helper = OpenTravelSearchTool(ToolsConfig(travel_max_results=3))
    result = helper.plan_public_transport(
        "Puerta del Sol, Madrid",
        "Avenida de America, Madrid",
        departure_time="2026-07-20T09:00:00+02:00",
        max_transfers=1,
    )

    plan_params = calls[2][1]["params"]
    assert plan_params["fromPlace"] == "40.4168,-3.7038"
    assert plan_params["toPlace"] == "40.4379,-3.6795"
    assert plan_params["maxTransfers"] == 1
    assert calls[0][1]["headers"]["User-Agent"].startswith("chack-agent/")
    assert "25 min" in result
    assert "SUBWAY L2 Metro de Madrid" in result
    assert list((tmp_path / "transitous-routes").glob("Puerta_del_Sol_Madrid_Avenida_de_America_Madrid_*.json"))


def test_marine_holidays_and_ticketmaster_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("CHACK_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TICKETMASTER_API_KEY", "tm-key")
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        if "geocoding-api" in url:
            return FakeResponse({"results": [{"name": "Palma", "country": "Spain", "latitude": 39.57, "longitude": 2.65}]})
        if "marine-api" in url:
            return FakeResponse({
                "daily": {"time": ["2026-07-20"], "wave_height_max": [1.2], "swell_wave_height_max": [0.8], "wave_period_max": [7.0]},
                "daily_units": {"wave_height_max": "m", "swell_wave_height_max": "m", "wave_period_max": "s"},
                "hourly": {"sea_surface_temperature": [25.1, 25.7]}, "hourly_units": {"sea_surface_temperature": "°C"},
            })
        if "date.nager.at" in url:
            return FakeResponse([{"date": "2026-08-15", "name": "Assumption Day", "nationalHoliday": True, "subdivisionCodes": None, "holidayTypes": ["Public"]}])
        return FakeResponse({"_embedded": {"events": [{
            "name": "Summer Festival", "url": "https://ticketmaster.example/event",
            "dates": {"start": {"localDate": "2026-08-15", "localTime": "20:00:00"}, "status": {"code": "onsale"}},
            "_embedded": {"venues": [{"name": "Harbour Stage"}]},
        }]}})

    monkeypatch.setattr("chack_tools.open_travel_search.requests.get", fake_get)
    helper = OpenTravelSearchTool(ToolsConfig(travel_max_results=3))
    marine = helper.get_marine_forecast("Palma", forecast_days=3)
    holidays = helper.get_public_holidays("ES", "2026-08-10", "2026-08-20")
    events = helper.search_ticketmaster_events("Palma", "2026-08-10", "2026-08-20", keyword="festival")

    assert calls[1][0] == "https://marine-api.open-meteo.com/v1/marine"
    assert calls[1][1]["cell_selection"] == "sea"
    assert calls[2][0].endswith("/ES/2026")
    assert calls[3][1]["apikey"] == "tm-key"
    assert "Sea-surface temperature range: 25.1–25.7 °C" in marine
    assert "Assumption Day" in holidays
    assert "Summer Festival" in events and "Harbour Stage" in events
    assert list((tmp_path / "open-meteo-marine").glob("Palma_*.json"))
    assert list((tmp_path / "nager-public-holidays").glob("ES_2026-08-10_2026-08-20_*.json"))
    assert list((tmp_path / "ticketmaster-events").glob("Palma_2026-08-10_2026-08-20_festival_*.json"))


def test_direct_travel_tools_and_researcher_are_registered(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    monkeypatch.setenv("TICKETMASTER_API_KEY", "tm-key")
    config = ToolsConfig(
        travel_enabled=True,
        travel_google_flights_enabled=True,
        travel_google_travel_explore_enabled=True,
        travel_google_hotels_enabled=True,
        travel_google_hotels_reviews_enabled=True,
        travel_booking_enabled=True,
        travel_amadeus_enabled=True,
        travel_opentripmap_enabled=True,
        travel_open_meteo_enabled=True,
        travel_open_meteo_air_quality_enabled=True,
        travel_open_meteo_marine_enabled=True,
        travel_public_holidays_enabled=True,
        travel_ticketmaster_enabled=True,
        travel_frankfurter_enabled=True,
        travel_wikivoyage_enabled=True,
        travel_transitous_enabled=True,
    )
    names = _tool_names(AgentsToolset(config, model_provider="openai").tools)

    assert {
        "search_google_flights",
        "explore_google_travel_destinations",
        "search_google_stays",
        "get_google_stay_details",
        "get_google_stay_reviews",
        "find_booking_cities",
        "search_booking_stays",
        "get_booking_stay_details",
        "get_booking_stay_reviews",
        "search_amadeus_hotel_prices",
        "get_amadeus_hotel_sentiments",
        "search_destination_places",
        "get_destination_place_details",
        "get_destination_weather",
        "get_destination_air_quality",
        "get_destination_marine_forecast",
        "get_destination_public_holidays",
        "search_ticketmaster_events",
        "convert_travel_currency",
        "search_wikivoyage_guides",
        "plan_public_transport_trip",
        "travel_research",
    }.issubset(names)
    assert RESEARCHER_REGISTRY["travel"] == ("travel_enabled", "travel_research")


def test_travel_researcher_capability_set(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    monkeypatch.setenv("BOOKING_API_TOKEN", "booking-token")
    monkeypatch.setenv("BOOKING_AFFILIATE_ID", "12345")
    monkeypatch.setenv("AMADEUS_CLIENT_ID", "amadeus-id")
    monkeypatch.setenv("AMADEUS_CLIENT_SECRET", "amadeus-secret")
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "otm-key")
    monkeypatch.setenv("TICKETMASTER_API_KEY", "tm-key")
    helper = TravelResearchAgentTool(
        ToolsConfig(task_steps_manager_enabled=False),
        model_provider="openai",
    )
    names = _tool_names(helper._build_subagent_tools())

    assert "search_google_flights" in names
    assert "search_google_stays" in names
    assert "get_google_stay_details" in names
    assert "get_google_stay_reviews" in names
    assert "search_booking_stays" in names
    assert "get_booking_stay_reviews" in names
    assert "search_amadeus_hotel_prices" in names
    assert "get_amadeus_hotel_sentiments" in names
    assert "search_destination_places" in names
    assert "get_destination_place_details" in names
    assert "search_tripadvisor" in names
    assert "search_google_maps_businesses" in names
    assert "get_destination_weather" in names
    assert "get_destination_air_quality" in names
    assert "get_destination_marine_forecast" in names
    assert "get_destination_public_holidays" in names
    assert "search_ticketmaster_events" in names
    assert "convert_travel_currency" in names
    assert "search_wikivoyage_guides" in names
    assert "plan_public_transport_trip" in names
    assert "wikidata_entity_search" in names
    assert "wikidata_sparql" in names
    assert "knowledge_graph_research" not in names

    for tool in helper._build_subagent_tools():
        description = str(getattr(tool, "description", "") or "")
        assert "Output:" in description, getattr(tool, "name", "")
        for param_name, param_schema in (
            (getattr(tool, "params_json_schema", {}) or {}).get("properties", {}).items()
        ):
            assert param_schema.get("description"), f"{getattr(tool, 'name', '')}.{param_name}"


def test_research_administrator_can_force_enable_travel(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    helper = ResearcherAdministratorAgentTool(
        ToolsConfig(task_steps_manager_enabled=False),
        model_provider="openai",
        researchers=["travel"],
    )

    assert helper._enabled_researchers() == ["travel"]
    names = _tool_names(helper._build_subagent_tools(["travel"]))
    # Ordinary researchers enter through the supervised async boundary; the
    # raw travel tool remains a private dependency of that job.
    assert "travel_research" not in names
    assert "run_researchers_batch" not in names
    assert "start_researchers_async" in names
    assert "poll_researchers_async" in names
