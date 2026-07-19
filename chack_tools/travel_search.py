from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from typing import Any, Optional
from urllib.parse import urlencode
from uuid import uuid4

import requests

from .config import ToolsConfig
from .research_artifacts import record_research_json_artifact, research_artifacts_root
from .serpapi_keys import (
    is_serpapi_rate_limited,
    note_serpapi_response_error,
    usable_serpapi_keys,
)
from .telemetry import run_with_tool_logging

try:
    from agents import function_tool
except ImportError:
    function_tool = None


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FLIGHT_CLASSES = {1, 2, 3, 4}
_FLIGHT_SORTS = {1, 2, 3, 4, 5, 6}
_FLIGHT_STOPS = {0, 1, 2, 3}
_HOTEL_SORTS = {0, 3, 8, 13}
_HOTEL_RATINGS = {0, 7, 8, 9}
_REVIEW_SORTS = {1, 2, 3, 4}


def _run_logged(tool: str, tool_input: dict[str, Any], func):
    try:
        return run_with_tool_logging(tool, tool_input, func)
    except Exception as exc:
        return f"ERROR: {tool} failed ({exc})"


def _wrapper_values(values: dict[str, Any]) -> dict[str, Any]:
    """Remove closure cells that CPython may expose through nested-function locals()."""
    return {name: value for name, value in values.items() if name != "helper"}


def _with_travel_output(tool):
    descriptions = {
        "departure_id": "IATA airport code(s) or Google location ID for the origin.",
        "arrival_id": "IATA airport code(s) or Google location ID for the destination; optional for open-ended exploration.",
        "outbound_date": "Outbound travel date in YYYY-MM-DD format.",
        "return_date": "Return date in YYYY-MM-DD format; leave empty for one-way travel.",
        "currency": "Three-letter currency code used for returned prices, such as USD or EUR.",
        "travel_class": "Cabin: 1 economy, 2 premium economy, 3 business, or 4 first.",
        "adults": "Number of adult travelers or guests.",
        "children": "Number of child travelers or guests.",
        "infants_in_seat": "Number of infants traveling in their own seats.",
        "infants_on_lap": "Number of lap infants; cannot exceed the adult count.",
        "stops": "Flight stops filter: 0 any, 1 nonstop, 2 at most one, 3 at most two.",
        "sort_by": "Endpoint-specific sort code documented in the tool description.",
        "max_price": "Optional maximum price in the selected currency.",
        "min_price": "Optional minimum nightly lodging price in the selected currency.",
        "bags": "Number of carry-on bags, limited by eligible passenger count.",
        "include_airlines": "Comma-separated IATA airline codes or alliance names to include.",
        "exclude_airlines": "Comma-separated IATA airline codes or alliance names to exclude.",
        "deep_search": "Ask SerpAPI for browser-equivalent flight results at higher latency.",
        "gl": "Optional two-letter Google country code for localization.",
        "hl": "Optional two-letter result language code.",
        "max_results": "Maximum number of compact results to return, capped by configuration.",
        "timeout_seconds": "HTTP request timeout in seconds.",
        "query": "Destination, property, neighborhood, or lodging search text.",
        "check_in_date": "Lodging check-in date in YYYY-MM-DD format.",
        "check_out_date": "Lodging check-out date in YYYY-MM-DD format; must be after check-in.",
        "children_ages": "Comma-separated ages from 1 to 17, one value per child.",
        "vacation_rentals": "Search vacation-rental inventory instead of hotels when true.",
        "bedrooms": "Minimum bedrooms for vacation rentals.",
        "bathrooms": "Minimum bathrooms for vacation rentals.",
        "rating": "Google Hotels rating filter: 0 none, 7 for 3.5+, 8 for 4.0+, or 9 for 4.5+.",
        "hotel_class": "Comma-separated hotel star classes such as 3,4,5.",
        "free_cancellation": "Require free cancellation for hotel results when true.",
        "next_page_token": "Opaque pagination token returned by a previous call.",
        "property_token": "Opaque Google Hotels property token returned by stay search.",
        "source_number": "Review source filter: 0 all, -1 Google, or a property-specific source number.",
        "category_token": "Optional review category token returned by hotel details.",
        "location": "Human-readable destination name to geocode for weather.",
        "forecast_days": "Forecast length from 1 to 16 days.",
        "temperature_unit": "Temperature unit, celsius or fahrenheit.",
        "wind_speed_unit": "Wind unit: kmh, ms, mph, or kn.",
        "country": "Lowercase ISO 3166-1 alpha-2 country code used by Booking.com.",
        "language": "Provider language code, such as en-gb for Booking.com or en for OpenTripMap.",
        "city_query": "City name fragment used to find the Booking.com city identifier required by stay search.",
        "city_id": "Signed Booking.com city identifier returned by find_booking_cities.",
        "booker_country": "Lowercase ISO 3166-1 alpha-2 country code for the traveler making the search.",
        "platform": "Booking.com client platform: desktop, mobile, tablet, ios, or android.",
        "rooms": "Number of hotel rooms requested.",
        "accommodation_ids": "Comma-separated Booking.com accommodation identifiers.",
        "accommodation_id": "Booking.com accommodation identifier returned by stay search.",
        "city_code": "Three-letter IATA city code used by Amadeus, such as PAR or MAD.",
        "radius_km": "Search radius in kilometers around the destination city center.",
        "hotel_ids": "Comma-separated Amadeus hotel identifiers returned by hotel-price search.",
        "destination": "Destination city or place name to resolve with OpenTripMap.",
        "radius_meters": "Point-of-interest search radius in meters.",
        "kinds": "Optional comma-separated OpenTripMap place categories such as museums,architecture.",
        "minimum_rating": "OpenTripMap popularity filter from 1 (all) to 3 (most notable).",
        "xid": "OpenTripMap place identifier returned by destination-place search.",
        "max_pages": "Maximum number of provider pagination pages to inspect.",
    }
    schema = getattr(tool, "params_json_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, property_schema in properties.items():
            if isinstance(property_schema, dict) and not property_schema.get("description"):
                property_schema["description"] = descriptions.get(
                    name,
                    f"Value for the {name.replace('_', ' ')} travel search parameter.",
                )
    current = str(getattr(tool, "description", "") or "").strip()
    tool.description = (
        f"{current}\n\n"
        "Parameters: Provide endpoint-specific travel dates, traveler counts, locale, currency, filters, pagination, and timeout values as described in the schema.\n"
        "Output: Compact SUCCESS/ERROR text with price/review/forecast records and an Artifact JSON path containing the complete raw response."
    ).strip()
    return tool


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short(value: Any, max_chars: int = 360) -> str:
    clean = " ".join(str(value or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(filter(None, (_as_text(item) for item in value)))
    if isinstance(value, dict):
        for key in ("lowest", "price", "amount", "value", "name", "title", "text", "description"):
            text = _as_text(value.get(key))
            if text:
                return text
    return ""


def _safe_filename(value: str, fallback: str = "travel-data") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _artifact_dir(kind: str) -> str:
    root = research_artifacts_root()
    base = os.path.join(root, kind) if root else os.path.join("/tmp", "chack-travel", kind)
    os.makedirs(base, exist_ok=True)
    return base


def _write_json_artifact(kind: str, label: str, payload: Any) -> str:
    path = os.path.join(_artifact_dir(kind), f"{_safe_filename(label)}_{uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    record_research_json_artifact(
        path,
        payload,
        provenance=f"{kind}:{label}",
        kind=kind,
        label=label,
    )
    return path


def _validate_date(value: str, label: str) -> str:
    raw = str(value or "").strip()
    if not _ISO_DATE.match(raw):
        return f"ERROR: {label} must use YYYY-MM-DD format"
    try:
        date.fromisoformat(raw)
    except ValueError:
        return f"ERROR: {label} is not a valid calendar date"
    return ""


def _format_minutes(value: Any) -> str:
    minutes = _coerce_int(value, -1)
    if minutes < 0:
        return ""
    hours, remainder = divmod(minutes, 60)
    return f"{hours}h {remainder}m" if hours else f"{remainder}m"


class TravelSearchTool:
    """Travel price, accommodation, review, destination, and weather APIs."""

    def __init__(self, config: ToolsConfig):
        self.config = config
        self._amadeus_access_token = ""
        self._amadeus_access_token_expires_at = 0.0

    def _max_results(self, requested: Optional[int]) -> int:
        configured = _clamp(
            _coerce_int(getattr(self.config, "travel_max_results", 10), 10),
            1,
            30,
        )
        if requested is None:
            return configured
        return _clamp(_coerce_int(requested, configured), 1, 30)

    def _serpapi_request(self, params: dict[str, Any], timeout_seconds: int = 30) -> Any:
        api_keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if not api_keys:
            return "ERROR: No usable SerpAPI key (not configured or all keys exhausted)."
        for index, api_key in enumerate(api_keys):
            request_params = dict(params)
            request_params["api_key"] = api_key
            request_params["output"] = "json"
            try:
                response = requests.get(
                    "https://serpapi.com/search",
                    params=request_params,
                    timeout=max(5, int(timeout_seconds or 30)),
                )
            except requests.exceptions.Timeout:
                return "ERROR: SerpAPI request timed out"
            except requests.exceptions.ConnectionError:
                return "ERROR: Failed to connect to SerpAPI"
            if response.status_code >= 400:
                body = _short(response.text, 220)
                note_serpapi_response_error(api_key, response.status_code, body)
                if is_serpapi_rate_limited(response.status_code, body) and index < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI returned HTTP {response.status_code} ({body})"
            try:
                payload = response.json()
            except ValueError:
                return "ERROR: SerpAPI returned invalid JSON"
            if isinstance(payload, dict) and payload.get("error"):
                error_text = str(payload.get("error") or "")
                note_serpapi_response_error(api_key, response.status_code, error_text)
                if is_serpapi_rate_limited(response.status_code, error_text) and index < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI error ({error_text})"
            return payload
        return "ERROR: All configured SerpAPI keys are rate limited."

    def search_flights(
        self,
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        return_date: str = "",
        currency: str = "USD",
        travel_class: int = 1,
        adults: int = 1,
        children: int = 0,
        infants_in_seat: int = 0,
        infants_on_lap: int = 0,
        stops: int = 0,
        sort_by: int = 1,
        max_price: Optional[int] = None,
        bags: int = 0,
        include_airlines: str = "",
        exclude_airlines: str = "",
        deep_search: bool = False,
        gl: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        departure_id = str(departure_id or "").strip()
        arrival_id = str(arrival_id or "").strip()
        if not departure_id or not arrival_id:
            return "ERROR: departure_id and arrival_id are required"
        date_error = _validate_date(outbound_date, "outbound_date")
        if date_error:
            return date_error
        return_date = str(return_date or "").strip()
        if return_date:
            date_error = _validate_date(return_date, "return_date")
            if date_error:
                return date_error
            if date.fromisoformat(return_date) < date.fromisoformat(outbound_date):
                return "ERROR: return_date must be on or after outbound_date"
        if travel_class not in _FLIGHT_CLASSES:
            return "ERROR: travel_class must be 1 (economy), 2, 3, or 4 (first)"
        if stops not in _FLIGHT_STOPS:
            return "ERROR: stops must be 0 (any), 1 (nonstop), 2, or 3"
        if sort_by not in _FLIGHT_SORTS:
            return "ERROR: sort_by must be between 1 and 6"
        if include_airlines.strip() and exclude_airlines.strip():
            return "ERROR: include_airlines and exclude_airlines cannot be used together"
        adults = _clamp(_coerce_int(adults, 1), 1, 9)
        children = _clamp(_coerce_int(children, 0), 0, 8)
        infants_in_seat = _clamp(_coerce_int(infants_in_seat, 0), 0, 8)
        infants_on_lap = _clamp(_coerce_int(infants_on_lap, 0), 0, adults)
        bags = _clamp(_coerce_int(bags, 0), 0, adults + children + infants_in_seat)
        params: dict[str, Any] = {
            "engine": "google_flights",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date,
            "type": 1 if return_date else 2,
            "currency": str(currency or "USD").upper(),
            "travel_class": travel_class,
            "adults": adults,
            "children": children,
            "infants_in_seat": infants_in_seat,
            "infants_on_lap": infants_on_lap,
            "stops": stops,
            "sort_by": sort_by,
            "bags": bags,
        }
        if return_date:
            params["return_date"] = return_date
        if max_price is not None and _coerce_int(max_price, 0) > 0:
            params["max_price"] = _coerce_int(max_price, 0)
        if include_airlines.strip():
            params["include_airlines"] = include_airlines.strip().upper()
        if exclude_airlines.strip():
            params["exclude_airlines"] = exclude_airlines.strip().upper()
        if deep_search:
            params["deep_search"] = "true"
        if str(gl or "").strip():
            params["gl"] = str(gl).lower()
        if str(hl or "").strip():
            params["hl"] = str(hl).lower()

        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        label = f"{departure_id}_{arrival_id}_{outbound_date}_{return_date or 'one-way'}"
        artifact = _write_json_artifact("google-flights", label, payload)
        currency_code = str((payload.get("search_parameters") or {}).get("currency") or params["currency"])
        insights = payload.get("price_insights") or {}
        lines = [
            f"SUCCESS: Google Flights {departure_id} -> {arrival_id} on {outbound_date}"
            + (f" returning {return_date}:" if return_date else " one-way:"),
        ]
        research_query = f"Flights from {departure_id} to {arrival_id} on {outbound_date}"
        if return_date:
            research_query += f" returning {return_date}"
        lines.append(
            "Public research URL (re-run to verify inventory): https://www.google.com/travel/flights?"
            + urlencode({"q": research_query, "curr": params["currency"]})
        )
        if isinstance(insights, dict) and insights:
            typical = insights.get("typical_price_range") or []
            typical_text = "-".join(str(item) for item in typical[:2]) if isinstance(typical, list) else ""
            lines.append(
                f"Price insight: lowest {insights.get('lowest_price', '')} {currency_code}; "
                f"level {insights.get('price_level', '')}; typical {typical_text} {currency_code}".strip()
            )
        rows: list[tuple[str, dict[str, Any]]] = []
        rows.extend(("best", item) for item in (payload.get("best_flights") or []) if isinstance(item, dict))
        rows.extend(("other", item) for item in (payload.get("other_flights") or []) if isinstance(item, dict))
        limit = self._max_results(max_results)
        for index, (category, item) in enumerate(rows[:limit], start=1):
            segments = item.get("flights") or []
            segment_texts = []
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                departure = segment.get("departure_airport") or {}
                arrival = segment.get("arrival_airport") or {}
                segment_texts.append(
                    f"{departure.get('id', '')} {departure.get('time', '')} -> "
                    f"{arrival.get('id', '')} {arrival.get('time', '')} "
                    f"({segment.get('airline', '')} {segment.get('flight_number', '')})"
                )
            layovers = item.get("layovers") or []
            layover_text = ", ".join(
                f"{part.get('id', '')} {_format_minutes(part.get('duration'))}"
                for part in layovers
                if isinstance(part, dict)
            )
            lines.append(
                f"{index}. [{category}] {item.get('price', '')} {currency_code} | "
                f"{_format_minutes(item.get('total_duration'))} | {'; '.join(segment_texts)}"
            )
            if layover_text:
                lines.append(f"   layovers: {layover_text}")
            emissions = item.get("carbon_emissions") or {}
            if isinstance(emissions, dict) and emissions.get("difference_percent") is not None:
                lines.append(f"   emissions vs typical: {emissions.get('difference_percent')}%")
            token = item.get("departure_token") or item.get("booking_token") or ""
            if token:
                lines.append(f"   selection_token: {token}")
        if not rows:
            lines.append("No best_flights or other_flights returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def explore_destinations(
        self,
        departure_id: str,
        arrival_id: str = "",
        currency: str = "USD",
        gl: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        departure_id = str(departure_id or "").strip()
        if not departure_id:
            return "ERROR: departure_id is required"
        params: dict[str, Any] = {
            "engine": "google_travel_explore",
            "departure_id": departure_id,
            "currency": str(currency or "USD").upper(),
        }
        if str(arrival_id or "").strip():
            params["arrival_id"] = str(arrival_id).strip()
        if str(gl or "").strip():
            params["gl"] = str(gl).lower()
        if str(hl or "").strip():
            params["hl"] = str(hl).lower()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact(
            "google-travel-explore",
            f"{departure_id}_{arrival_id or 'anywhere'}",
            payload,
        )
        currency_code = str((payload.get("search_parameters") or {}).get("currency") or params["currency"])
        rows = payload.get("destinations") or payload.get("flights") or []
        lines = [f"SUCCESS: Google Travel Explore from {departure_id} (prices in {currency_code}):"]
        lines.append(
            "Public research URL (re-run to verify inventory): https://www.google.com/travel/explore?"
            + urlencode({"q": f"Flights from {departure_id} to {arrival_id or 'anywhere'}", "curr": params["currency"]})
        )
        for index, item in enumerate(rows[: self._max_results(max_results)], start=1):
            if not isinstance(item, dict):
                continue
            destination = item.get("name") or (item.get("arrival_airport") or {}).get("name") or arrival_id
            lines.append(
                f"{index}. {destination} | {item.get('start_date', payload.get('start_date', ''))} to "
                f"{item.get('end_date', payload.get('end_date', ''))} | flight {item.get('flight_price', item.get('price', ''))} "
                f"{currency_code} | hotel/night {item.get('hotel_price', '')} {currency_code} | "
                f"stops {item.get('number_of_stops', '')} | airline {item.get('airline', '')}"
            )
            if item.get("link"):
                lines.append(f"   {item.get('link')}")
        if not rows:
            lines.append("No destinations or flights returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_stays(
        self,
        query: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        children: int = 0,
        children_ages: str = "",
        currency: str = "USD",
        vacation_rentals: bool = False,
        bedrooms: int = 0,
        bathrooms: int = 0,
        sort_by: int = 0,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        rating: int = 0,
        hotel_class: str = "",
        free_cancellation: bool = False,
        gl: str = "",
        hl: str = "",
        next_page_token: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query is required"
        for value, label in ((check_in_date, "check_in_date"), (check_out_date, "check_out_date")):
            date_error = _validate_date(value, label)
            if date_error:
                return date_error
        if date.fromisoformat(check_out_date) <= date.fromisoformat(check_in_date):
            return "ERROR: check_out_date must be after check_in_date"
        if sort_by not in _HOTEL_SORTS:
            return "ERROR: sort_by must be 0 (relevance), 3 (price), 8 (rating), or 13 (reviews)"
        if rating not in _HOTEL_RATINGS:
            return "ERROR: rating must be 0, 7 (3.5+), 8 (4.0+), or 9 (4.5+)"
        adults = _clamp(_coerce_int(adults, 2), 1, 10)
        children = _clamp(_coerce_int(children, 0), 0, 10)
        ages = [part.strip() for part in str(children_ages or "").split(",") if part.strip()]
        if ages and len(ages) != children:
            return "ERROR: children_ages count must match children"
        if any(not age.isdigit() or not 1 <= int(age) <= 17 for age in ages):
            return "ERROR: every children_ages value must be between 1 and 17"
        params: dict[str, Any] = {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "children": children,
            "currency": str(currency or "USD").upper(),
        }
        if ages:
            params["children_ages"] = ",".join(ages)
        if vacation_rentals:
            params["vacation_rentals"] = "true"
            if _coerce_int(bedrooms, 0) > 0:
                params["bedrooms"] = _coerce_int(bedrooms, 0)
            if _coerce_int(bathrooms, 0) > 0:
                params["bathrooms"] = _coerce_int(bathrooms, 0)
        else:
            if str(hotel_class or "").strip():
                params["hotel_class"] = str(hotel_class).strip()
            if free_cancellation:
                params["free_cancellation"] = "true"
        if sort_by:
            params["sort_by"] = sort_by
        if rating:
            params["rating"] = rating
        if min_price is not None and _coerce_int(min_price, -1) >= 0:
            params["min_price"] = _coerce_int(min_price, 0)
        if max_price is not None and _coerce_int(max_price, 0) > 0:
            params["max_price"] = _coerce_int(max_price, 0)
        if str(gl or "").strip():
            params["gl"] = str(gl).lower()
        if str(hl or "").strip():
            params["hl"] = str(hl).lower()
        if str(next_page_token or "").strip():
            params["next_page_token"] = str(next_page_token).strip()

        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        kind = "google-vacation-rentals" if vacation_rentals else "google-hotels"
        artifact = _write_json_artifact(kind, f"{query}_{check_in_date}_{check_out_date}", payload)
        currency_code = str((payload.get("search_parameters") or {}).get("currency") or params["currency"])
        rows = payload.get("properties") or []
        lines = [
            f"SUCCESS: {'Vacation rentals' if vacation_rentals else 'Hotels'} for '{query}' "
            f"from {check_in_date} to {check_out_date} ({currency_code}):"
        ]
        lines.append(
            "Public research URL (re-run to verify inventory): https://www.google.com/travel/search?"
            + urlencode(
                {
                    "q": ("Vacation rentals" if vacation_rentals else "Hotels") + f" in {query}",
                    "checkin": check_in_date, "checkout": check_out_date, "curr": params["currency"],
                }
            )
        )
        for index, item in enumerate(rows[: self._max_results(max_results)], start=1):
            if not isinstance(item, dict):
                continue
            nightly = item.get("rate_per_night") or {}
            total = item.get("total_rate") or {}
            lines.append(
                f"{index}. {item.get('name', '')} | type {item.get('type', '')} | "
                f"night {_as_text(nightly)} | total {_as_text(total)} | rating {item.get('overall_rating', '')} "
                f"({item.get('reviews', '')} reviews) | class {item.get('hotel_class', '')}"
            )
            address = item.get("address") or ""
            if address:
                lines.append(f"   address: {_short(address, 240)}")
            amenities = item.get("amenities") or []
            if amenities:
                lines.append(f"   amenities: {_short(_as_text(amenities), 320)}")
            prices = item.get("prices") or []
            price_sources = []
            for price in prices[:5]:
                if isinstance(price, dict):
                    price_sources.append(
                        f"{price.get('source', '')}: {_as_text(price.get('rate_per_night') or price.get('price') or price.get('total_rate'))}"
                    )
            if price_sources:
                lines.append(f"   sellers: {'; '.join(price_sources)}")
            if item.get("property_token"):
                lines.append(f"   property_token: {item.get('property_token')}")
            if item.get("link"):
                lines.append(f"   {item.get('link')}")
        if not rows:
            lines.append("No properties returned.")
        pagination = payload.get("serpapi_pagination") or {}
        if isinstance(pagination, dict) and pagination.get("next_page_token"):
            lines.append(f"Next page token: {pagination.get('next_page_token')}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_stay_details(
        self,
        query: str,
        property_token: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        children: int = 0,
        currency: str = "USD",
        gl: str = "",
        hl: str = "",
        timeout_seconds: int = 30,
    ) -> str:
        query = str(query or "").strip()
        property_token = str(property_token or "").strip()
        if not query or not property_token:
            return "ERROR: query and property_token are required"
        for value, label in ((check_in_date, "check_in_date"), (check_out_date, "check_out_date")):
            date_error = _validate_date(value, label)
            if date_error:
                return date_error
        params: dict[str, Any] = {
            "engine": "google_hotels",
            "q": query,
            "property_token": property_token,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": _clamp(_coerce_int(adults, 2), 1, 10),
            "children": _clamp(_coerce_int(children, 0), 0, 10),
            "currency": str(currency or "USD").upper(),
        }
        if str(gl or "").strip():
            params["gl"] = str(gl).lower()
        if str(hl or "").strip():
            params["hl"] = str(hl).lower()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-hotel-details", property_token, payload)
        item = payload.get("property") or payload.get("property_results") or {}
        if not isinstance(item, dict) or not item:
            properties = payload.get("properties") or []
            item = properties[0] if properties and isinstance(properties[0], dict) else payload
        lines = [f"SUCCESS: Stay details for {item.get('name', query)}:"]
        lines.append(
            "Public research URL (re-run to verify inventory): https://www.google.com/travel/search?"
            + urlencode({"q": str(item.get("name") or query), "checkin": check_in_date, "checkout": check_out_date, "curr": params["currency"]})
        )
        for label, key in (
            ("type", "type"),
            ("address", "address"),
            ("phone", "phone"),
            ("description", "description"),
            ("rating", "overall_rating"),
            ("reviews", "reviews"),
            ("nightly rate", "rate_per_night"),
            ("total rate", "total_rate"),
            ("check-in", "check_in_time"),
            ("check-out", "check_out_time"),
            ("amenities", "amenities"),
            ("nearby places", "nearby_places"),
        ):
            value = _as_text(item.get(key))
            if value:
                lines.append(f"{label}: {_short(value, 900)}")
        prices = item.get("prices") or []
        if prices:
            lines.append(f"booking prices: {_short(json.dumps(prices[:10], ensure_ascii=False), 1800)}")
        if item.get("link"):
            lines.append(f"property URL: {item.get('link')}")
        lines.append(f"property_token: {property_token}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_stay_reviews(
        self,
        property_token: str,
        sort_by: int = 1,
        source_number: int = 0,
        category_token: str = "",
        next_page_token: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        property_token = str(property_token or "").strip()
        if not property_token:
            return "ERROR: property_token is required"
        if sort_by not in _REVIEW_SORTS:
            return "ERROR: sort_by must be 1 (helpful), 2 (recent), 3 (highest), or 4 (lowest)"
        params: dict[str, Any] = {
            "engine": "google_hotels_reviews",
            "property_token": property_token,
            "sort_by": sort_by,
            "source_number": _coerce_int(source_number, 0),
        }
        if str(category_token or "").strip():
            params["category_token"] = str(category_token).strip()
        if str(next_page_token or "").strip():
            params["next_page_token"] = str(next_page_token).strip()
        if str(hl or "").strip():
            params["hl"] = str(hl).lower()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-hotel-reviews", property_token, payload)
        reviews = payload.get("reviews") or []
        lines = [f"SUCCESS: Stay reviews for property {property_token}:"]
        for index, item in enumerate(reviews[: self._max_results(max_results)], start=1):
            if not isinstance(item, dict):
                continue
            user = item.get("user") or {}
            user_name = user.get("name", "") if isinstance(user, dict) else _as_text(user)
            text = item.get("description") or item.get("text") or item.get("snippet") or ""
            subratings = item.get("subratings") or {}
            lines.append(
                f"{index}. {item.get('rating', '')}/{item.get('best_rating', 5)} | "
                f"{item.get('date', '')} | {item.get('source', '')} | {user_name}"
            )
            if text:
                lines.append(f"   {_short(text, 700)}")
            if subratings:
                lines.append(f"   subratings: {_short(json.dumps(subratings, ensure_ascii=False), 500)}")
        if not reviews:
            lines.append("No reviews returned.")
        pagination = payload.get("serpapi_pagination") or {}
        if isinstance(pagination, dict) and pagination.get("next_page_token"):
            lines.append(f"Next page token: {pagination.get('next_page_token')}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def _booking_request(
        self,
        path: str,
        body: dict[str, Any],
        timeout_seconds: int = 30,
    ) -> Any:
        token = str(os.environ.get("BOOKING_API_TOKEN", "") or "").strip()
        affiliate_id = str(os.environ.get("BOOKING_AFFILIATE_ID", "") or "").strip()
        if not token or not affiliate_id:
            return "ERROR: Booking.com Demand API requires BOOKING_API_TOKEN and BOOKING_AFFILIATE_ID."
        base_url = str(
            os.environ.get("BOOKING_DEMAND_API_BASE_URL", "https://demandapi.booking.com/3.1")
            or "https://demandapi.booking.com/3.1"
        ).rstrip("/")
        try:
            response = requests.post(
                f"{base_url}/{path.lstrip('/')}",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "X-Affiliate-Id": affiliate_id,
                },
                json=body,
                timeout=max(5, int(timeout_seconds or 30)),
            )
        except requests.exceptions.Timeout:
            return "ERROR: Booking.com Demand API request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Booking.com Demand API"
        if response.status_code >= 400:
            return f"ERROR: Booking.com Demand API returned HTTP {response.status_code} ({_short(response.text, 300)})"
        try:
            return response.json()
        except ValueError:
            return "ERROR: Booking.com Demand API returned invalid JSON"

    def find_booking_cities(
        self,
        country: str,
        city_query: str,
        language: str = "en-gb",
        max_results: Optional[int] = None,
        max_pages: int = 3,
        timeout_seconds: int = 30,
    ) -> str:
        country = str(country or "").strip().lower()
        query = str(city_query or "").strip().casefold()
        if not re.fullmatch(r"[a-z]{2}", country):
            return "ERROR: country must be a lowercase two-letter ISO code"
        if not query:
            return "ERROR: city_query is required"
        matches: list[dict[str, Any]] = []
        responses: list[Any] = []
        page = ""
        for _ in range(_clamp(_coerce_int(max_pages, 3), 1, 10)):
            body: dict[str, Any] = {
                "country": country,
                "languages": [str(language or "en-gb").strip().lower()],
                "rows": 1000,
            }
            if page:
                body["page"] = page
            payload = self._booking_request("common/locations/cities", body, timeout_seconds)
            if isinstance(payload, str):
                return payload
            responses.append(payload)
            for item in payload.get("data") or []:
                if not isinstance(item, dict):
                    continue
                names = item.get("name") or item.get("names") or ""
                names_text = (
                    ", ".join(str(value) for value in names.values())
                    if isinstance(names, dict)
                    else _as_text(names)
                )
                if query in names_text.casefold():
                    matches.append(item)
            metadata = payload.get("metadata") or {}
            page = str(metadata.get("next_page") or metadata.get("next_page_token") or "")
            if not page or len(matches) >= self._max_results(max_results):
                break
        artifact = _write_json_artifact(
            "booking-cities", f"{country}_{city_query}", {"pages": responses, "matches": matches}
        )
        lines = [f"SUCCESS: Booking.com city matches for {city_query}, {country.upper()}:"]
        for index, item in enumerate(matches[: self._max_results(max_results)], start=1):
            names = item.get("name") or item.get("names") or ""
            names_text = (
                ", ".join(str(value) for value in names.values())
                if isinstance(names, dict)
                else _as_text(names)
            )
            lines.append(
                f"{index}. {names_text or 'Unknown city'} | "
                f"city_id: {item.get('id', '')} | region: {_as_text(item.get('region'))}"
            )
        if not matches:
            lines.append("No matching cities found in the inspected pages; try a broader name or more pages.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_booking_stays(
        self,
        city_id: int,
        check_in_date: str,
        check_out_date: str,
        booker_country: str,
        adults: int = 2,
        rooms: int = 1,
        children_ages: str = "",
        currency: str = "EUR",
        platform: str = "desktop",
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        for value, label in ((check_in_date, "check_in_date"), (check_out_date, "check_out_date")):
            error = _validate_date(value, label)
            if error:
                return error
        if date.fromisoformat(check_out_date) <= date.fromisoformat(check_in_date):
            return "ERROR: check_out_date must be after check_in_date"
        country = str(booker_country or "").strip().lower()
        if not re.fullmatch(r"[a-z]{2}", country):
            return "ERROR: booker_country must be a lowercase two-letter ISO code"
        platform = str(platform or "desktop").strip().lower()
        if platform not in {"desktop", "mobile", "tablet", "ios", "android"}:
            return "ERROR: platform must be desktop, mobile, tablet, ios, or android"
        ages = []
        if str(children_ages or "").strip():
            try:
                ages = [int(value.strip()) for value in str(children_ages).split(",") if value.strip()]
            except ValueError:
                return "ERROR: children_ages must be comma-separated integers"
            if any(value < 0 or value > 17 for value in ages):
                return "ERROR: every child age must be between 0 and 17"
        body: dict[str, Any] = {
            "city": int(city_id),
            "booker": {"country": country, "platform": platform},
            "checkin": check_in_date,
            "checkout": check_out_date,
            "currency": str(currency or "EUR").upper(),
            "guests": {
                "number_of_adults": _clamp(_coerce_int(adults, 2), 1, 30),
                "number_of_rooms": _clamp(_coerce_int(rooms, 1), 1, 30),
            },
        }
        if ages:
            body["guests"]["children"] = ages
        if min_price is not None or max_price is not None:
            body["price"] = {}
            if min_price is not None:
                body["price"]["minimum"] = max(0, int(min_price))
            if max_price is not None:
                body["price"]["maximum"] = max(0, int(max_price))
        payload = self._booking_request("accommodations/search", body, timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact(
            "booking-stays", f"{city_id}_{check_in_date}_{check_out_date}", payload
        )
        data = payload.get("data") or []
        lines = [f"SUCCESS: Booking.com stays for city {city_id} ({check_in_date} to {check_out_date}):"]
        for index, item in enumerate(data[: self._max_results(max_results)], start=1):
            if not isinstance(item, dict):
                continue
            price = item.get("price") or {}
            amount = _as_text(price.get("book") if isinstance(price, dict) else price) or _as_text(price)
            lines.append(
                f"{index}. {item.get('name') or item.get('id') or 'Accommodation'} | "
                f"accommodation_id: {item.get('id', '')} | price: {amount or 'not returned'} "
                f"{item.get('currency') or body['currency']} | score: {_as_text(item.get('review_score') or item.get('rating'))}"
            )
            url = item.get("url") or item.get("deep_link_url")
            if url:
                lines.append(f"   URL: {_as_text(url)}")
        if not data:
            lines.append("No available stays returned.")
        metadata = payload.get("metadata") or {}
        if metadata.get("next_page") or metadata.get("next_page_token"):
            lines.append(f"Next page token: {metadata.get('next_page') or metadata.get('next_page_token')}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_booking_stay_details(
        self,
        accommodation_ids: str,
        language: str = "en-gb",
        timeout_seconds: int = 30,
    ) -> str:
        try:
            ids = [int(value.strip()) for value in str(accommodation_ids or "").split(",") if value.strip()]
        except ValueError:
            return "ERROR: accommodation_ids must be comma-separated integers"
        if not ids:
            return "ERROR: accommodation_ids is required"
        ids = ids[:100]
        body = {
            "accommodations": ids,
            "extras": ["description", "facilities", "photos", "rooms"],
            "languages": [str(language or "en-gb").strip().lower()],
        }
        payload = self._booking_request("accommodations/details", body, timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("booking-stay-details", "_".join(map(str, ids[:5])), payload)
        data = payload.get("data") or []
        lines = ["SUCCESS: Booking.com accommodation details:"]
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"{index}. {item.get('name') or item.get('id')} | accommodation_id: {item.get('id', '')} | "
                f"address: {_short(_as_text(item.get('address')), 300)} | score: {_as_text(item.get('review_score'))}"
            )
            description = _as_text(item.get("description"))
            if description:
                lines.append(f"   {_short(description, 900)}")
            facilities = _as_text(item.get("facilities"))
            if facilities:
                lines.append(f"   facilities: {_short(facilities, 600)}")
        if not data:
            lines.append("No details returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_booking_stay_reviews(
        self,
        accommodation_id: int,
        language: str = "en-gb",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        rows = self._max_results(max_results)
        body = {
            "accommodations": [int(accommodation_id)],
            "languages": [str(language or "en-gb").strip().lower()],
            "rows": max(10, ((rows + 9) // 10) * 10),
        }
        payload = self._booking_request("accommodations/reviews", body, timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("booking-stay-reviews", str(accommodation_id), payload)
        data = payload.get("data") or []
        reviews: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("reviews"), list):
                reviews.extend(value for value in item["reviews"] if isinstance(value, dict))
            elif isinstance(item, dict):
                reviews.append(item)
        lines = [f"SUCCESS: Booking.com reviews for accommodation {accommodation_id}:"]
        for index, item in enumerate(reviews[:rows], start=1):
            score = item.get("score") or item.get("review_score") or item.get("rating") or ""
            title = _as_text(item.get("title"))
            positive = _as_text(item.get("pros") or item.get("positive"))
            negative = _as_text(item.get("cons") or item.get("negative"))
            text = _as_text(item.get("text") or item.get("content"))
            lines.append(f"{index}. score {score} | {_short(title, 180)}")
            if positive:
                lines.append(f"   positive: {_short(positive, 500)}")
            if negative:
                lines.append(f"   negative: {_short(negative, 500)}")
            if text:
                lines.append(f"   {_short(text, 700)}")
        if not reviews:
            lines.append("No reviews returned, or this affiliate account lacks review-endpoint permission.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def _amadeus_token(self, timeout_seconds: int = 30) -> Any:
        if self._amadeus_access_token and time.monotonic() < self._amadeus_access_token_expires_at:
            return self._amadeus_access_token
        client_id = str(os.environ.get("AMADEUS_CLIENT_ID", "") or "").strip()
        client_secret = str(os.environ.get("AMADEUS_CLIENT_SECRET", "") or "").strip()
        if not client_id or not client_secret:
            return "ERROR: Amadeus requires AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET."
        base_url = str(os.environ.get("AMADEUS_BASE_URL", "https://test.api.amadeus.com") or "").rstrip("/")
        try:
            response = requests.post(
                f"{base_url}/v1/security/oauth2/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=max(5, int(timeout_seconds or 30)),
            )
        except requests.exceptions.Timeout:
            return "ERROR: Amadeus authentication timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Amadeus"
        if response.status_code >= 400:
            return f"ERROR: Amadeus authentication returned HTTP {response.status_code} ({_short(response.text, 300)})"
        try:
            payload = response.json()
        except ValueError:
            return "ERROR: Amadeus authentication returned invalid JSON"
        token = str(payload.get("access_token") or "")
        if not token:
            return "ERROR: Amadeus authentication response did not contain an access token"
        self._amadeus_access_token = token
        self._amadeus_access_token_expires_at = time.monotonic() + max(
            30, _coerce_int(payload.get("expires_in"), 1800) - 30
        )
        return token

    def _amadeus_get(self, path: str, params: dict[str, Any], timeout_seconds: int = 30) -> Any:
        token = self._amadeus_token(timeout_seconds)
        if isinstance(token, str) and token.startswith("ERROR:"):
            return token
        base_url = str(os.environ.get("AMADEUS_BASE_URL", "https://test.api.amadeus.com") or "").rstrip("/")
        try:
            response = requests.get(
                f"{base_url}/{path.lstrip('/')}",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                params=params,
                timeout=max(5, int(timeout_seconds or 30)),
            )
        except requests.exceptions.Timeout:
            return "ERROR: Amadeus API request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Amadeus API"
        if response.status_code >= 400:
            return f"ERROR: Amadeus API returned HTTP {response.status_code} ({_short(response.text, 300)})"
        try:
            return response.json()
        except ValueError:
            return "ERROR: Amadeus API returned invalid JSON"

    def search_amadeus_hotel_prices(
        self,
        city_code: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        rooms: int = 1,
        currency: str = "EUR",
        radius_km: int = 5,
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        city_code = str(city_code or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", city_code):
            return "ERROR: city_code must be a three-letter IATA city code"
        for value, label in ((check_in_date, "check_in_date"), (check_out_date, "check_out_date")):
            error = _validate_date(value, label)
            if error:
                return error
        if date.fromisoformat(check_out_date) <= date.fromisoformat(check_in_date):
            return "ERROR: check_out_date must be after check_in_date"
        limit = self._max_results(max_results)
        hotels_payload = self._amadeus_get(
            "v1/reference-data/locations/hotels/by-city",
            {
                "cityCode": city_code,
                "radius": _clamp(_coerce_int(radius_km, 5), 1, 100),
                "radiusUnit": "KM",
                "hotelSource": "ALL",
            },
            timeout_seconds,
        )
        if isinstance(hotels_payload, str):
            return hotels_payload
        hotels = [item for item in hotels_payload.get("data") or [] if isinstance(item, dict)][:limit]
        hotel_ids = [str(item.get("hotelId") or "") for item in hotels if item.get("hotelId")]
        offers_payload: Any = {"data": []}
        if hotel_ids:
            offers_payload = self._amadeus_get(
                "v3/shopping/hotel-offers",
                {
                    "hotelIds": ",".join(hotel_ids),
                    "adults": _clamp(_coerce_int(adults, 2), 1, 9),
                    "checkInDate": check_in_date,
                    "checkOutDate": check_out_date,
                    "roomQuantity": _clamp(_coerce_int(rooms, 1), 1, 9),
                    "currency": str(currency or "EUR").upper(),
                },
                timeout_seconds,
            )
            if isinstance(offers_payload, str):
                return offers_payload
        combined = {"hotel_list": hotels_payload, "hotel_offers": offers_payload}
        artifact = _write_json_artifact(
            "amadeus-hotel-prices", f"{city_code}_{check_in_date}_{check_out_date}", combined
        )
        lines = [f"SUCCESS: Amadeus hotel offers for {city_code} ({check_in_date} to {check_out_date}):"]
        for index, item in enumerate((offers_payload.get("data") or [])[:limit], start=1):
            if not isinstance(item, dict):
                continue
            hotel = item.get("hotel") or {}
            offers = item.get("offers") or []
            offer = offers[0] if offers and isinstance(offers[0], dict) else {}
            price = offer.get("price") or {}
            policies = offer.get("policies") or {}
            lines.append(
                f"{index}. {hotel.get('name') or hotel.get('hotelId') or 'Hotel'} | hotel_id: {hotel.get('hotelId', '')} | "
                f"offer_id: {offer.get('id', '')} | total: {price.get('total', '')} {price.get('currency', currency)} | "
                f"available: {item.get('available', '')}"
            )
            room = offer.get("room") or {}
            if room:
                lines.append(f"   room: {_short(_as_text(room.get('description') or room), 500)}")
            if policies:
                lines.append(f"   policies: {_short(json.dumps(policies, ensure_ascii=False), 600)}")
        if not (offers_payload.get("data") or []):
            lines.append("No priced offers returned for the selected dates.")
            if hotel_ids:
                lines.append(f"Unpriced hotel IDs: {','.join(hotel_ids)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_amadeus_hotel_sentiments(
        self,
        hotel_ids: str,
        timeout_seconds: int = 30,
    ) -> str:
        ids = [value.strip().upper() for value in str(hotel_ids or "").split(",") if value.strip()]
        if not ids:
            return "ERROR: hotel_ids is required"
        payload = self._amadeus_get(
            "v2/e-reputation/hotel-sentiments",
            {"hotelIds": ",".join(ids[:20])},
            timeout_seconds,
        )
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("amadeus-hotel-sentiments", "_".join(ids[:5]), payload)
        data = payload.get("data") or []
        lines = ["SUCCESS: Amadeus aggregate hotel-review sentiments:"]
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            sentiments = item.get("sentiments") or {}
            lines.append(
                f"{index}. hotel_id: {item.get('hotelId', '')} | overall: {item.get('overallRating', '')}/100 | "
                f"reviews: {item.get('numberOfReviews', '')} | ratings: {item.get('numberOfRatings', '')}"
            )
            if sentiments:
                lines.append(f"   categories: {_short(json.dumps(sentiments, ensure_ascii=False), 900)}")
        if not data:
            lines.append("No sentiment records returned for these hotel IDs.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def _opentripmap_get(self, path: str, params: dict[str, Any], timeout_seconds: int = 20) -> Any:
        api_key = str(os.environ.get("OPENTRIPMAP_API_KEY", "") or "").strip()
        if not api_key:
            return "ERROR: OpenTripMap requires OPENTRIPMAP_API_KEY (a free key is available)."
        base_url = str(
            os.environ.get("OPENTRIPMAP_BASE_URL", "https://api.opentripmap.com/0.1")
            or "https://api.opentripmap.com/0.1"
        ).rstrip("/")
        request_params = dict(params)
        request_params["apikey"] = api_key
        try:
            response = requests.get(
                f"{base_url}/{path.lstrip('/')}",
                params=request_params,
                timeout=max(5, int(timeout_seconds or 20)),
            )
        except requests.exceptions.Timeout:
            return "ERROR: OpenTripMap request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to OpenTripMap"
        if response.status_code >= 400:
            return f"ERROR: OpenTripMap returned HTTP {response.status_code} ({_short(response.text, 300)})"
        try:
            return response.json()
        except ValueError:
            return "ERROR: OpenTripMap returned invalid JSON"

    def search_opentripmap_places(
        self,
        destination: str,
        radius_meters: int = 5000,
        kinds: str = "",
        minimum_rating: int = 2,
        language: str = "en",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        destination = str(destination or "").strip()
        if not destination:
            return "ERROR: destination is required"
        language = str(language or "en").strip().lower()
        geocode = self._opentripmap_get(
            f"{language}/places/geoname", {"name": destination}, timeout_seconds
        )
        if isinstance(geocode, str):
            return geocode
        lat, lon = geocode.get("lat"), geocode.get("lon")
        if lat is None or lon is None:
            return f"ERROR: OpenTripMap could not resolve destination '{destination}'"
        params: dict[str, Any] = {
            "radius": _clamp(_coerce_int(radius_meters, 5000), 100, 50000),
            "lat": lat,
            "lon": lon,
            "rate": _clamp(_coerce_int(minimum_rating, 2), 1, 3),
            "limit": self._max_results(max_results),
            "format": "json",
        }
        if str(kinds or "").strip():
            params["kinds"] = str(kinds).strip()
        payload = self._opentripmap_get(f"{language}/places/radius", params, timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact(
            "opentripmap-places", destination, {"geocode": geocode, "places": payload}
        )
        data = payload if isinstance(payload, list) else payload.get("features") or payload.get("data") or []
        lines = [f"SUCCESS: OpenTripMap places near {destination} ({lat}, {lon}):"]
        for index, item in enumerate(data[: self._max_results(max_results)], start=1):
            if not isinstance(item, dict):
                continue
            properties = item.get("properties") if isinstance(item.get("properties"), dict) else item
            lines.append(
                f"{index}. {properties.get('name') or 'Unnamed place'} | xid: {properties.get('xid', '')} | "
                f"kinds: {properties.get('kinds', '')} | rate: {properties.get('rate', '')} | "
                f"distance_m: {properties.get('dist', '')}"
            )
        if not data:
            lines.append("No places returned for these filters.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_opentripmap_place_details(
        self,
        xid: str,
        language: str = "en",
        timeout_seconds: int = 20,
    ) -> str:
        xid = str(xid or "").strip()
        if not xid:
            return "ERROR: xid is required"
        language = str(language or "en").strip().lower()
        payload = self._opentripmap_get(f"{language}/places/xid/{xid}", {}, timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("opentripmap-place-details", xid, payload)
        lines = [f"SUCCESS: OpenTripMap details for {payload.get('name') or xid}:"]
        for label, key in (
            ("xid", "xid"),
            ("kinds", "kinds"),
            ("address", "address"),
            ("description", "wikipedia_extracts"),
            ("official URL", "url"),
            ("Wikipedia", "wikipedia"),
            ("image", "image"),
        ):
            value = _as_text(payload.get(key))
            if value:
                lines.append(f"{label}: {_short(value, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_weather_forecast(
        self,
        location: str,
        forecast_days: int = 7,
        temperature_unit: str = "celsius",
        wind_speed_unit: str = "kmh",
        timeout_seconds: int = 20,
    ) -> str:
        """Resolve a place and return a keyless Open-Meteo daily forecast."""
        location = str(location or "").strip()
        if not location:
            return "ERROR: location is required"

        def get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = requests.get(
                        url,
                        params=params,
                        timeout=max(5, int(timeout_seconds or 20)),
                    )
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.HTTPError as exc:
                    last_error = exc
                    status = getattr(exc.response, "status_code", 0) or 0
                    if status < 500 and status != 429:
                        raise
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError) as exc:
                    last_error = exc
                if attempt < 2:
                    time.sleep(0.4 * (2 ** attempt))
            assert last_error is not None
            raise last_error

        try:
            geo_payload = get_json(
                "https://geocoding-api.open-meteo.com/v1/search",
                {"name": location, "count": 1, "language": "en", "format": "json"},
            )
            results = geo_payload.get("results") or []
            if not results:
                return f"ERROR: Open-Meteo could not resolve location '{location}'"
            place = results[0]
            params = {
                "latitude": place.get("latitude"),
                "longitude": place.get("longitude"),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,precipitation_sum,wind_speed_10m_max,uv_index_max"
                ),
                "timezone": "auto",
                "forecast_days": _clamp(_coerce_int(forecast_days, 7), 1, 16),
                "temperature_unit": "fahrenheit" if temperature_unit.lower().startswith("f") else "celsius",
                "wind_speed_unit": wind_speed_unit if wind_speed_unit in {"kmh", "ms", "mph", "kn"} else "kmh",
            }
            payload = get_json(
                "https://api.open-meteo.com/v1/forecast",
                params,
            )
        except requests.exceptions.Timeout:
            return "ERROR: Open-Meteo request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Open-Meteo"
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            return f"ERROR: Open-Meteo returned HTTP {status}"
        except ValueError:
            return "ERROR: Open-Meteo returned invalid JSON"
        combined = {"geocoding": place, "forecast": payload}
        artifact = _write_json_artifact("open-meteo", location, combined)
        daily = payload.get("daily") or {}
        units = payload.get("daily_units") or {}
        times = daily.get("time") or []
        lines = [
            f"SUCCESS: Open-Meteo forecast for {place.get('name', location)}, "
            f"{place.get('admin1', '')}, {place.get('country', '')}:"
        ]
        for index, day in enumerate(times):
            def value(key: str) -> Any:
                values = daily.get(key) or []
                return values[index] if index < len(values) else ""

            lines.append(
                f"{day}: code {value('weather_code')} | "
                f"{value('temperature_2m_min')}-{value('temperature_2m_max')} {units.get('temperature_2m_max', '')} | "
                f"rain chance {value('precipitation_probability_max')}{units.get('precipitation_probability_max', '')} | "
                f"rain {value('precipitation_sum')} {units.get('precipitation_sum', '')} | "
                f"wind {value('wind_speed_10m_max')} {units.get('wind_speed_10m_max', '')} | "
                f"UV {value('uv_index_max')}"
            )
        lines.append("Forecasts are predictions and become less reliable farther into the future.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)


def _tool_or_raise():
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")


def get_google_flights_search_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="search_google_flights")
    def search_google_flights(
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        return_date: str = "",
        currency: str = "USD",
        travel_class: int = 1,
        adults: int = 1,
        children: int = 0,
        infants_in_seat: int = 0,
        infants_on_lap: int = 0,
        stops: int = 0,
        sort_by: int = 1,
        max_price: Optional[int] = None,
        bags: int = 0,
        include_airlines: str = "",
        exclude_airlines: str = "",
        deep_search: bool = False,
        gl: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Search live Google Flights prices, schedules, stops, and price insights via SerpAPI.

        Use IATA airport codes or Google location IDs. Omit return_date for one-way travel.
        Prices are snapshots and must be rechecked before booking.
        """
        values = _wrapper_values(locals())
        return _run_logged("search_google_flights", values, lambda: helper.search_flights(**values))

    return _with_travel_output(search_google_flights)


def get_google_travel_explore_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="explore_google_travel_destinations")
    def explore_google_travel_destinations(
        departure_id: str,
        arrival_id: str = "",
        currency: str = "USD",
        gl: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Find flexible-date destinations with indicative flight and hotel prices via Google Travel Explore."""
        values = _wrapper_values(locals())
        return _run_logged(
            "explore_google_travel_destinations",
            values,
            lambda: helper.explore_destinations(**values),
        )

    return _with_travel_output(explore_google_travel_destinations)


def get_google_stays_search_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="search_google_stays")
    def search_google_stays(
        query: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        children: int = 0,
        children_ages: str = "",
        currency: str = "USD",
        vacation_rentals: bool = False,
        bedrooms: int = 0,
        bathrooms: int = 0,
        sort_by: int = 0,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        rating: int = 0,
        hotel_class: str = "",
        free_cancellation: bool = False,
        gl: str = "",
        hl: str = "",
        next_page_token: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Search Google Hotels or vacation rentals with nightly/total rates, ratings, and booking sources.

        Set vacation_rentals=true for Airbnb-like whole-home/apartment inventory aggregated by
        Google. Results are not guaranteed to be Airbnb listings and prices remain snapshots.
        """
        values = _wrapper_values(locals())
        return _run_logged("search_google_stays", values, lambda: helper.search_stays(**values))

    return _with_travel_output(search_google_stays)


def get_google_stay_details_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_google_stay_details")
    def get_google_stay_details(
        query: str,
        property_token: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        children: int = 0,
        currency: str = "USD",
        gl: str = "",
        hl: str = "",
        timeout_seconds: int = 30,
    ) -> str:
        """Get one Google Hotels property's details, amenities, nearby places, and booking prices."""
        values = _wrapper_values(locals())
        return _run_logged("get_google_stay_details", values, lambda: helper.get_stay_details(**values))

    return _with_travel_output(get_google_stay_details)


def get_google_stay_reviews_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_google_stay_reviews")
    def get_google_stay_reviews(
        property_token: str,
        sort_by: int = 1,
        source_number: int = 0,
        category_token: str = "",
        next_page_token: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Get detailed Google Hotels reviews for a property token, optionally sorted or filtered."""
        values = _wrapper_values(locals())
        return _run_logged("get_google_stay_reviews", values, lambda: helper.get_stay_reviews(**values))

    return _with_travel_output(get_google_stay_reviews)


def get_booking_cities_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="find_booking_cities")
    def find_booking_cities(
        country: str,
        city_query: str,
        language: str = "en-gb",
        max_results: Optional[int] = None,
        max_pages: int = 3,
        timeout_seconds: int = 30,
    ) -> str:
        """Find Booking.com city IDs required by the official Demand API accommodation search."""
        values = _wrapper_values(locals())
        return _run_logged("find_booking_cities", values, lambda: helper.find_booking_cities(**values))

    return _with_travel_output(find_booking_cities)


def get_booking_stays_search_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="search_booking_stays")
    def search_booking_stays(
        city_id: int,
        check_in_date: str,
        check_out_date: str,
        booker_country: str,
        adults: int = 2,
        rooms: int = 1,
        children_ages: str = "",
        currency: str = "EUR",
        platform: str = "desktop",
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Search Booking.com's official Demand API for available stays and best returned prices."""
        values = _wrapper_values(locals())
        return _run_logged("search_booking_stays", values, lambda: helper.search_booking_stays(**values))

    return _with_travel_output(search_booking_stays)


def get_booking_stay_details_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_booking_stay_details")
    def get_booking_stay_details(
        accommodation_ids: str,
        language: str = "en-gb",
        timeout_seconds: int = 30,
    ) -> str:
        """Get official Booking.com property descriptions, facilities, photos, and room metadata."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_booking_stay_details", values, lambda: helper.get_booking_stay_details(**values)
        )

    return _with_travel_output(get_booking_stay_details)


def get_booking_stay_reviews_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_booking_stay_reviews")
    def get_booking_stay_reviews(
        accommodation_id: int,
        language: str = "en-gb",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Get Booking.com guest reviews when the affiliate agreement permits the review endpoint."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_booking_stay_reviews", values, lambda: helper.get_booking_stay_reviews(**values)
        )

    return _with_travel_output(get_booking_stay_reviews)


def get_amadeus_hotel_prices_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="search_amadeus_hotel_prices")
    def search_amadeus_hotel_prices(
        city_code: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 2,
        rooms: int = 1,
        currency: str = "EUR",
        radius_km: int = 5,
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Find Amadeus hotels by city and return real-time room offers and prices for selected dates."""
        values = _wrapper_values(locals())
        return _run_logged(
            "search_amadeus_hotel_prices", values, lambda: helper.search_amadeus_hotel_prices(**values)
        )

    return _with_travel_output(search_amadeus_hotel_prices)


def get_amadeus_hotel_sentiments_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_amadeus_hotel_sentiments")
    def get_amadeus_hotel_sentiments(
        hotel_ids: str,
        timeout_seconds: int = 30,
    ) -> str:
        """Get Amadeus aggregate ratings and review-sentiment categories for hotel IDs."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_amadeus_hotel_sentiments", values, lambda: helper.get_amadeus_hotel_sentiments(**values)
        )

    return _with_travel_output(get_amadeus_hotel_sentiments)


def get_opentripmap_places_search_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="search_destination_places")
    def search_destination_places(
        destination: str,
        radius_meters: int = 5000,
        kinds: str = "",
        minimum_rating: int = 2,
        language: str = "en",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search OpenTripMap's worldwide points of interest for itinerary planning."""
        values = _wrapper_values(locals())
        return _run_logged(
            "search_destination_places", values, lambda: helper.search_opentripmap_places(**values)
        )

    return _with_travel_output(search_destination_places)


def get_opentripmap_place_details_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_destination_place_details")
    def get_destination_place_details(
        xid: str,
        language: str = "en",
        timeout_seconds: int = 20,
    ) -> str:
        """Get address, description, links, and image metadata for an OpenTripMap place."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_destination_place_details",
            values,
            lambda: helper.get_opentripmap_place_details(**values),
        )

    return _with_travel_output(get_destination_place_details)


def get_open_meteo_forecast_tool(helper: TravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_destination_weather")
    def get_destination_weather(
        location: str,
        forecast_days: int = 7,
        temperature_unit: str = "celsius",
        wind_speed_unit: str = "kmh",
        timeout_seconds: int = 20,
    ) -> str:
        """Get a free, keyless Open-Meteo daily forecast for a destination (up to 16 days)."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_destination_weather",
            values,
            lambda: helper.get_weather_forecast(**values),
        )

    return _with_travel_output(get_destination_weather)
