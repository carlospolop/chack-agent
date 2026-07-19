from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from typing import Any, Optional

import requests

from .config import ToolsConfig
from .telemetry import run_with_tool_logging
from .travel_search import (
    _as_text,
    _clamp,
    _coerce_int,
    _short,
    _validate_date,
    _write_json_artifact,
)

try:
    from agents import function_tool
except ImportError:
    function_tool = None


_USER_AGENT = "chack-agent/0.1 (https://github.com/carlospolop/chack-agent)"
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_WIKI_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]+)?$")
_COORDINATES = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")
_COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")


def _run_logged(tool: str, tool_input: dict[str, Any], func):
    try:
        return run_with_tool_logging(tool, tool_input, func)
    except Exception as exc:
        return f"ERROR: {tool} failed ({exc})"


def _wrapper_values(values: dict[str, Any]) -> dict[str, Any]:
    """Remove closure cells that CPython may expose through nested-function locals()."""
    return {name: value for name, value in values.items() if name != "helper"}


def _with_open_travel_output(tool):
    descriptions = {
        "location": "Human-readable destination name to geocode.",
        "forecast_days": "Air-quality forecast length from 1 to 7 days.",
        "country_code": "Two-letter ISO 3166-1 destination country code.",
        "start_date": "Inclusive start date in YYYY-MM-DD format.",
        "end_date": "Inclusive end date in YYYY-MM-DD format.",
        "keyword": "Optional event name, performer, sport, festival, or category keyword.",
        "base_currency": "Three-letter currency code for the amount being converted.",
        "quote_currency": "Three-letter destination currency code.",
        "amount": "Positive monetary amount to convert using the reference rate.",
        "rate_date": "Optional historical rate date in YYYY-MM-DD format; empty means latest.",
        "provider": "Optional Frankfurter provider code such as ECB for one official source.",
        "query": "Destination, region, airport, or itinerary topic to find on Wikivoyage.",
        "language": "Wikivoyage or Transitous language code, such as en, es, or fr.",
        "max_results": "Maximum compact results to return, capped by configuration.",
        "from_location": "Trip origin address, station, landmark, or place name.",
        "to_location": "Trip destination address, station, landmark, or place name.",
        "departure_time": "Optional ISO 8601 departure or arrival time, preferably with UTC offset.",
        "arrive_by": "Interpret departure_time as the desired arrival time when true.",
        "max_transfers": "Optional maximum number of public-transport interchanges.",
        "timeout_seconds": "HTTP timeout in seconds.",
    }
    schema = getattr(tool, "params_json_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, property_schema in properties.items():
            if isinstance(property_schema, dict) and not property_schema.get("description"):
                property_schema["description"] = descriptions.get(
                    name, f"Value for the {name.replace('_', ' ')} travel parameter."
                )
    current = str(getattr(tool, "description", "") or "").strip()
    tool.description = (
        f"{current}\n\n"
        "Parameters: Provide the destination, currencies, dates, routing constraints, language, limits, and timeout described in the schema.\n"
        "Output: Compact SUCCESS/ERROR text with attributed travel records and an Artifact JSON path containing the complete raw response."
    ).strip()
    return tool


class OpenTravelSearchTool:
    """Keyless travel APIs for air quality, exchange rates, guides, and transit."""

    def __init__(self, config: ToolsConfig):
        self.config = config

    def _max_results(self, requested: Optional[int]) -> int:
        configured = _clamp(
            _coerce_int(getattr(self.config, "travel_max_results", 10), 10), 1, 30
        )
        if requested is None:
            return configured
        return _clamp(_coerce_int(requested, configured), 1, 30)

    @staticmethod
    def _get_json(
        url: str,
        params: dict[str, Any],
        timeout_seconds: int,
        provider_name: str,
    ) -> Any:
        for attempt in range(3):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
                    timeout=max(5, int(timeout_seconds or 20)),
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt < 2:
                    time.sleep(0.25 * (2**attempt))
                    continue
                if isinstance(exc, requests.exceptions.Timeout):
                    return f"ERROR: {provider_name} request timed out"
                return f"ERROR: Failed to connect to {provider_name}"
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    retry_after = str(getattr(response, "headers", {}).get("Retry-After") or "").strip()
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0.25 * (2**attempt)
                    time.sleep(min(2.0, max(0.0, delay)))
                    continue
            if response.status_code >= 400:
                return f"ERROR: {provider_name} returned HTTP {response.status_code}"
            try:
                return response.json()
            except ValueError:
                return f"ERROR: {provider_name} returned invalid JSON"
        return f"ERROR: {provider_name} request failed"

    def _geocode_open_meteo(self, location: str, timeout_seconds: int) -> Any:
        payload = self._get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": location, "count": 1, "language": "en", "format": "json"},
            timeout_seconds,
            "Open-Meteo geocoding",
        )
        if isinstance(payload, str):
            return payload
        results = payload.get("results") or []
        if not results:
            return f"ERROR: Open-Meteo could not resolve destination '{location}'"
        return results[0]

    def get_air_quality(
        self,
        location: str,
        forecast_days: int = 3,
        timeout_seconds: int = 20,
    ) -> str:
        location = str(location or "").strip()
        if not location:
            return "ERROR: location is required"
        geocode = self._geocode_open_meteo(location, timeout_seconds)
        if isinstance(geocode, str):
            return geocode
        variables = [
            "european_aqi",
            "us_aqi",
            "pm10",
            "pm2_5",
            "nitrogen_dioxide",
            "ozone",
            "alder_pollen",
            "birch_pollen",
            "grass_pollen",
            "mugwort_pollen",
            "olive_pollen",
            "ragweed_pollen",
        ]
        days = _clamp(_coerce_int(forecast_days, 3), 1, 7)
        payload = self._get_json(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            {
                "latitude": geocode.get("latitude"),
                "longitude": geocode.get("longitude"),
                "current": ",".join(variables),
                "hourly": ",".join(variables),
                "forecast_days": days,
                "timezone": "auto",
            },
            timeout_seconds,
            "Open-Meteo Air Quality",
        )
        if isinstance(payload, str):
            return payload
        combined = {"geocode": geocode, "air_quality": payload}
        artifact = _write_json_artifact("open-meteo-air-quality", location, combined)
        label = ", ".join(
            value
            for value in (
                str(geocode.get("name") or ""),
                str(geocode.get("admin1") or ""),
                str(geocode.get("country") or ""),
            )
            if value
        )
        current = payload.get("current") or {}
        units = payload.get("current_units") or {}
        lines = [f"SUCCESS: Open-Meteo air quality for {label or location}:"]
        lines.append(f"time: {current.get('time', '')} {payload.get('timezone_abbreviation', '')}")
        for key in variables:
            value = current.get(key)
            if value is not None:
                lines.append(f"{key}: {value} {units.get(key, '')}".rstrip())
        hourly = payload.get("hourly") or {}
        summary: dict[str, Any] = {}
        for key in ("european_aqi", "us_aqi", "pm10", "pm2_5", "grass_pollen"):
            values = [value for value in hourly.get(key, []) if isinstance(value, (int, float))]
            if values:
                summary[key] = {"average": round(sum(values) / len(values), 2), "maximum": max(values)}
        if summary:
            lines.append(f"forecast summary: {_short(json.dumps(summary, ensure_ascii=False), 1200)}")
        lines.append("Attribution: Open-Meteo and CAMS air-quality data; values are forecasts, not medical advice.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_marine_forecast(
        self,
        location: str,
        forecast_days: int = 5,
        timeout_seconds: int = 20,
    ) -> str:
        location = str(location or "").strip()
        if not location:
            return "ERROR: location is required"
        geocode = self._geocode_open_meteo(location, timeout_seconds)
        if isinstance(geocode, str):
            return geocode
        days = _clamp(_coerce_int(forecast_days, 5), 1, 8)
        payload = self._get_json(
            "https://marine-api.open-meteo.com/v1/marine",
            {
                "latitude": geocode.get("latitude"), "longitude": geocode.get("longitude"),
                "daily": "wave_height_max,wave_period_max,swell_wave_height_max",
                "hourly": "sea_surface_temperature", "forecast_days": days,
                "timezone": "auto", "cell_selection": "sea",
            },
            timeout_seconds,
            "Open-Meteo Marine",
        )
        if isinstance(payload, str):
            return payload
        combined = {"geocode": geocode, "marine": payload}
        artifact = _write_json_artifact("open-meteo-marine", location, combined)
        daily = payload.get("daily") or {}
        units = payload.get("daily_units") or {}
        temperatures = [
            value for value in (payload.get("hourly") or {}).get("sea_surface_temperature", [])
            if isinstance(value, (int, float))
        ]
        label = ", ".join(str(geocode.get(key) or "") for key in ("name", "country") if geocode.get(key))
        lines = [f"SUCCESS: Open-Meteo marine forecast near {label or location}:"]
        for index, day in enumerate((daily.get("time") or [])[:days]):
            values = []
            for key in ("wave_height_max", "swell_wave_height_max", "wave_period_max"):
                series = daily.get(key) or []
                if index < len(series) and series[index] is not None:
                    values.append(f"{key}: {series[index]} {units.get(key, '')}".rstrip())
            lines.append(f"{day}: " + " | ".join(values))
        if temperatures:
            lines.append(
                f"Sea-surface temperature range: {min(temperatures):g}–{max(temperatures):g} "
                f"{(payload.get('hourly_units') or {}).get('sea_surface_temperature', '')}".rstrip()
            )
        lines.append(
            "Caveat: modelled coastal conditions are not suitable for navigation or a substitute for local safety flags and operator advice."
        )
        lines.append("Attribution: Open-Meteo and its listed marine model providers.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_public_holidays(
        self,
        country_code: str,
        start_date: str,
        end_date: str,
        timeout_seconds: int = 20,
    ) -> str:
        country = str(country_code or "").strip().upper()
        if not _COUNTRY_CODE.fullmatch(country):
            return "ERROR: country_code must be a two-letter ISO country code"
        start_error = _validate_date(str(start_date or "").strip(), "start_date")
        end_error = _validate_date(str(end_date or "").strip(), "end_date")
        if start_error or end_error:
            return start_error or end_error
        start = date.fromisoformat(str(start_date).strip())
        end = date.fromisoformat(str(end_date).strip())
        if end < start or (end - start).days > 370:
            return "ERROR: holiday date range must be ordered and no longer than 370 days"
        records: list[dict[str, Any]] = []
        raw_by_year: dict[str, Any] = {}
        for year in range(start.year, end.year + 1):
            payload = self._get_json(
                f"https://date.nager.at/api/v4/Holidays/{country}/{year}", {}, timeout_seconds, "Nager.Date"
            )
            if isinstance(payload, str):
                return payload
            raw_by_year[str(year)] = payload
            if isinstance(payload, list):
                records.extend(item for item in payload if isinstance(item, dict))
        matches = [item for item in records if str(start) <= str(item.get("date") or "") <= str(end)]
        artifact = _write_json_artifact("nager-public-holidays", f"{country}_{start}_{end}", raw_by_year)
        lines = [f"SUCCESS: Nager.Date public holidays for {country}, {start} to {end}:"]
        for item in matches:
            subdivision = ",".join(item.get("subdivisionCodes") or [])
            lines.append(
                f"{item.get('date', '')}: {item.get('name', '')} | types: {','.join(item.get('holidayTypes') or [])}"
                f" | {'national' if item.get('nationalHoliday') else 'subdivisions: ' + subdivision}"
            )
        if not matches:
            lines.append("No listed public holidays fall inside the requested range.")
        lines.append("Closure and local-observance effects still require confirmation with the venue or operator.")
        lines.append("Attribution: Nager.Date community-maintained holiday data.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_ticketmaster_events(
        self,
        location: str,
        start_date: str,
        end_date: str,
        keyword: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        api_key = str(os.environ.get("TICKETMASTER_API_KEY", "") or "").strip()
        if not api_key:
            return "ERROR: Ticketmaster Discovery requires TICKETMASTER_API_KEY"
        city = str(location or "").strip()
        if not city:
            return "ERROR: location is required"
        start_error = _validate_date(str(start_date or "").strip(), "start_date")
        end_error = _validate_date(str(end_date or "").strip(), "end_date")
        if start_error or end_error:
            return start_error or end_error
        start = date.fromisoformat(str(start_date).strip())
        end = date.fromisoformat(str(end_date).strip())
        if end < start or (end - start).days > 370:
            return "ERROR: event date range must be ordered and no longer than 370 days"
        limit = self._max_results(max_results)
        params = {
            "apikey": api_key, "city": city, "startDateTime": f"{start}T00:00:00Z",
            "endDateTime": f"{end}T23:59:59Z", "sort": "date,asc", "size": limit,
        }
        if str(keyword or "").strip():
            params["keyword"] = str(keyword).strip()[:120]
        payload = self._get_json(
            "https://app.ticketmaster.com/discovery/v2/events.json", params, timeout_seconds, "Ticketmaster Discovery"
        )
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("ticketmaster-events", f"{city}_{start}_{end}_{keyword}", payload)
        events = (payload.get("_embedded") or {}).get("events") or []
        lines = [f"SUCCESS: Ticketmaster events in {city}, {start} to {end}:"]
        for event in events[:limit]:
            dates = event.get("dates") or {}
            start_data = dates.get("start") or {}
            venue = (((event.get("_embedded") or {}).get("venues") or [{}])[0] or {})
            lines.append(
                f"{start_data.get('localDate', '')} {start_data.get('localTime', '')}: {event.get('name', '')} | "
                f"{venue.get('name', '')} | status: {(dates.get('status') or {}).get('code', '')} | {event.get('url', '')}"
            )
        if not events:
            lines.append("No Ticketmaster events matched the requested city, dates, and keyword.")
        lines.append("Inventory coverage is not universal; verify the official venue/organizer and ticket availability.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def convert_currency(
        self,
        base_currency: str,
        quote_currency: str,
        amount: float = 1.0,
        rate_date: str = "",
        provider: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        base = str(base_currency or "").strip().upper()
        quote = str(quote_currency or "").strip().upper()
        if not _CURRENCY.fullmatch(base) or not _CURRENCY.fullmatch(quote):
            return "ERROR: base_currency and quote_currency must be three-letter currency codes"
        try:
            numeric_amount = float(amount)
        except (TypeError, ValueError):
            return "ERROR: amount must be numeric"
        if numeric_amount <= 0:
            return "ERROR: amount must be greater than zero"
        params: dict[str, Any] = {}
        if str(rate_date or "").strip():
            error = _validate_date(str(rate_date).strip(), "rate_date")
            if error:
                return error
            params["date"] = str(rate_date).strip()
        if str(provider or "").strip():
            params["providers"] = str(provider).strip().upper()
        base_url = str(
            os.environ.get("FRANKFURTER_BASE_URL", "https://api.frankfurter.dev")
            or "https://api.frankfurter.dev"
        ).rstrip("/")
        payload = self._get_json(
            f"{base_url}/v2/rate/{base}/{quote}", params, timeout_seconds, "Frankfurter"
        )
        if isinstance(payload, str):
            return payload
        rate = payload.get("rate")
        if not isinstance(rate, (int, float)):
            return "ERROR: Frankfurter response did not contain a numeric rate"
        converted = numeric_amount * float(rate)
        artifact = _write_json_artifact(
            "frankfurter-rates", f"{base}_{quote}_{payload.get('date') or rate_date or 'latest'}", payload
        )
        source = str(provider or "blended central-bank sources").upper()
        return (
            f"SUCCESS: {numeric_amount:g} {base} = {converted:.4f} {quote}\n"
            f"Reference rate: 1 {base} = {rate} {quote} | date: {payload.get('date', '')} | source: {source}\n"
            "Reference rates can differ from card, cash, or transfer-provider rates and fees.\n"
            f"Artifact JSON: {artifact}"
        )

    def search_wikivoyage(
        self,
        query: str,
        language: str = "en",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        language = str(language or "en").strip().lower()
        if not query:
            return "ERROR: query is required"
        if not _WIKI_LANGUAGE.fullmatch(language):
            return "ERROR: language must be a valid Wikivoyage language code"
        limit = self._max_results(max_results)
        payload = self._get_json(
            f"https://{language}.wikivoyage.org/w/api.php",
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
                "formatversion": 2,
            },
            timeout_seconds,
            "Wikivoyage",
        )
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("wikivoyage-guides", f"{language}_{query}", payload)
        pages = payload.get("query", {}).get("pages", [])
        lines = [f"SUCCESS: Wikivoyage guide matches for {query} ({language}):"]
        for index, page in enumerate(pages[:limit], start=1):
            if not isinstance(page, dict):
                continue
            lines.append(f"{index}. {page.get('title', '')} | {page.get('fullurl', '')}")
            extract = str(page.get("extract") or "").strip()
            if extract:
                lines.append(f"   {_short(extract, 900)}")
        if not pages:
            lines.append("No Wikivoyage guide matches returned.")
        lines.append("Attribution: Wikivoyage contributors, CC BY-SA; verify changeable details with primary sources.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def _transitous_geocode(self, text: str, language: str, timeout_seconds: int) -> Any:
        coordinate_match = _COORDINATES.fullmatch(text)
        if coordinate_match:
            latitude, longitude = map(float, coordinate_match.groups())
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                return f"ERROR: coordinates outside the valid range in '{text}'"
            return [{"name": text, "type": "COORDINATE", "lat": latitude, "lon": longitude}]

        queries = [text]
        parts = [part.strip() for part in text.split(",") if part.strip()]
        # Property names often confuse address autocomplete. Also try the address
        # without the first comma-delimited label and prefer the provider's best score.
        if len(parts) >= 4:
            queries.append(", ".join(parts[1:]))
        matches: list[dict[str, Any]] = []
        for query in queries:
            payload = self._get_json(
                "https://api.transitous.org/api/v1/geocode",
                {"text": query, "language": language, "numResults": 5},
                timeout_seconds,
                "Transitous geocoding",
            )
            if isinstance(payload, str):
                return payload
            if isinstance(payload, list):
                matches.extend(item for item in payload if isinstance(item, dict))
        matches.sort(
            key=lambda item: float(item.get("score"))
            if isinstance(item.get("score"), (int, float))
            else float("inf")
        )
        if not matches:
            return f"ERROR: Transitous could not resolve '{text}'"
        return matches[:5]

    def plan_public_transport(
        self,
        from_location: str,
        to_location: str,
        departure_time: str = "",
        arrive_by: bool = False,
        language: str = "en",
        max_transfers: Optional[int] = None,
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        origin_text = str(from_location or "").strip()
        destination_text = str(to_location or "").strip()
        if not origin_text or not destination_text:
            return "ERROR: from_location and to_location are required"
        language = str(language or "en").strip().lower()
        if not _WIKI_LANGUAGE.fullmatch(language):
            return "ERROR: language must be a valid language code"
        when = str(departure_time or "").strip()
        if when:
            try:
                datetime.fromisoformat(when.replace("Z", "+00:00"))
            except ValueError:
                return "ERROR: departure_time must be a valid ISO 8601 date-time"
        origin_matches = self._transitous_geocode(origin_text, language, timeout_seconds)
        if isinstance(origin_matches, str):
            return origin_matches
        destination_matches = self._transitous_geocode(destination_text, language, timeout_seconds)
        if isinstance(destination_matches, str):
            return destination_matches
        origin = origin_matches[0]
        destination = destination_matches[0]
        params: dict[str, Any] = {
            "fromPlace": f"{origin.get('lat')},{origin.get('lon')}",
            "toPlace": f"{destination.get('lat')},{destination.get('lon')}",
            "arriveBy": str(bool(arrive_by)).lower(),
            "numItineraries": self._max_results(max_results),
            "detailedLegs": "false",
        }
        if when:
            params["time"] = when
        if max_transfers is not None:
            params["maxTransfers"] = _clamp(_coerce_int(max_transfers, 4), 0, 12)
        payload = self._get_json(
            "https://api.transitous.org/api/v6/plan",
            params,
            timeout_seconds,
            "Transitous",
        )
        if isinstance(payload, str):
            return payload
        combined = {
            "origin_matches": origin_matches,
            "destination_matches": destination_matches,
            "plan": payload,
        }
        artifact = _write_json_artifact(
            "transitous-routes", f"{origin_text}_{destination_text}", combined
        )
        itineraries = payload.get("itineraries") or []
        lines = [
            f"SUCCESS: Transitous public-transport routes from {origin.get('name', origin_text)} "
            f"to {destination.get('name', destination_text)}:"
        ]
        for index, itinerary in enumerate(
            itineraries[: self._max_results(max_results)], start=1
        ):
            if not isinstance(itinerary, dict):
                continue
            duration_minutes = round(_coerce_int(itinerary.get("duration"), 0) / 60)
            lines.append(
                f"{index}. {itinerary.get('startTime', '')} to {itinerary.get('endTime', '')} | "
                f"{duration_minutes} min | transfers: {itinerary.get('transfers', '')}"
            )
            for leg in itinerary.get("legs") or []:
                if not isinstance(leg, dict):
                    continue
                start = leg.get("from") or {}
                end = leg.get("to") or {}
                route = leg.get("routeShortName") or leg.get("displayName") or ""
                agency = leg.get("agencyName") or ""
                detail = " ".join(value for value in (str(route), str(agency)) if value).strip()
                lines.append(
                    f"   {leg.get('mode', '')}{f' {detail}' if detail else ''}: "
                    f"{start.get('name', '')} -> {end.get('name', '')} "
                    f"({leg.get('startTime', '')} to {leg.get('endTime', '')})"
                )
                if leg.get("cancelled"):
                    lines.append("   WARNING: this leg is marked cancelled.")
                alerts = leg.get("alerts") or []
                if alerts:
                    lines.append(f"   alerts: {_short(_as_text(alerts), 700)}")
        if not itineraries:
            lines.append("No public-transport itinerary returned for the selected locations and time.")
        lines.append("Attribution: Transitous and its listed GTFS/OpenStreetMap data sources; service is best-effort.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)


def _tool_or_raise():
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available in this runtime.")


def get_open_meteo_air_quality_tool(helper: OpenTravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_destination_air_quality")
    def get_destination_air_quality(
        location: str,
        forecast_days: int = 3,
        timeout_seconds: int = 20,
    ) -> str:
        """Get a keyless Open-Meteo air-quality and pollen forecast for a destination."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_destination_air_quality", values, lambda: helper.get_air_quality(**values)
        )

    return _with_open_travel_output(get_destination_air_quality)


def get_open_meteo_marine_tool(helper: OpenTravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_destination_marine_forecast")
    def get_destination_marine_forecast(
        location: str,
        forecast_days: int = 5,
        timeout_seconds: int = 20,
    ) -> str:
        """Get a keyless Open-Meteo wave, swell, period, and sea-temperature forecast."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_destination_marine_forecast", values, lambda: helper.get_marine_forecast(**values)
        )

    return _with_open_travel_output(get_destination_marine_forecast)


def get_public_holidays_tool(helper: OpenTravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="get_destination_public_holidays")
    def get_destination_public_holidays(
        country_code: str,
        start_date: str,
        end_date: str,
        timeout_seconds: int = 20,
    ) -> str:
        """Get keyless Nager.Date holidays that may affect crowds, openings, and transport."""
        values = _wrapper_values(locals())
        return _run_logged(
            "get_destination_public_holidays", values, lambda: helper.get_public_holidays(**values)
        )

    return _with_open_travel_output(get_destination_public_holidays)


def get_ticketmaster_events_tool(helper: OpenTravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="search_ticketmaster_events")
    def search_ticketmaster_events(
        location: str,
        start_date: str,
        end_date: str,
        keyword: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search Ticketmaster Discovery for dated events and official ticket URLs."""
        values = _wrapper_values(locals())
        return _run_logged(
            "search_ticketmaster_events", values, lambda: helper.search_ticketmaster_events(**values)
        )

    return _with_open_travel_output(search_ticketmaster_events)


def get_travel_currency_tool(helper: OpenTravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="convert_travel_currency")
    def convert_travel_currency(
        base_currency: str,
        quote_currency: str,
        amount: float = 1.0,
        rate_date: str = "",
        provider: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Convert a travel budget using Frankfurter's keyless central-bank reference rates."""
        values = _wrapper_values(locals())
        return _run_logged(
            "convert_travel_currency", values, lambda: helper.convert_currency(**values)
        )

    return _with_open_travel_output(convert_travel_currency)


def get_wikivoyage_search_tool(helper: OpenTravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="search_wikivoyage_guides")
    def search_wikivoyage_guides(
        query: str,
        language: str = "en",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search keyless Wikivoyage destination, region, airport, and itinerary guides."""
        values = _wrapper_values(locals())
        return _run_logged(
            "search_wikivoyage_guides", values, lambda: helper.search_wikivoyage(**values)
        )

    return _with_open_travel_output(search_wikivoyage_guides)


def get_transitous_route_tool(helper: OpenTravelSearchTool):
    _tool_or_raise()

    @function_tool(name_override="plan_public_transport_trip")
    def plan_public_transport_trip(
        from_location: str,
        to_location: str,
        departure_time: str = "",
        arrive_by: bool = False,
        language: str = "en",
        max_transfers: Optional[int] = None,
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Plan a keyless public-transport journey with Transitous schedule and realtime data."""
        values = _wrapper_values(locals())
        return _run_logged(
            "plan_public_transport_trip", values, lambda: helper.plan_public_transport(**values)
        )

    return _with_open_travel_output(plan_public_transport_trip)
