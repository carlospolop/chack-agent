from __future__ import annotations

import json
import os
import re
from html import unescape
from typing import Any, Optional
from uuid import uuid4

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests

from .config import ToolsConfig
from .research_artifacts import record_research_json_artifact, research_artifacts_root
from .serpapi_keys import (
    is_serpapi_rate_limited,
    note_serpapi_response_error,
    usable_serpapi_keys,
)
from .telemetry import run_with_tool_logging


_SEC_HEADERS = {
    "User-Agent": os.environ.get(
        "SEC_USER_AGENT",
        "chack-agent business-research contact@example.com",
    )
}
_FINANCE_WINDOWS = {"1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"}
_ADS_TRANSPARENCY_REGION_ALIASES = {
    "AR": "2032",
    "AU": "2036",
    "BR": "2076",
    "CA": "2124",
    "DE": "2276",
    "ES": "2724",
    "FR": "2250",
    "GB": "2826",
    "IN": "2356",
    "IT": "2380",
    "JP": "2392",
    "MX": "2484",
    "UK": "2826",
    "US": "2840",
}


def _run_logged(tool: str, tool_input: dict, func):
    try:
        return run_with_tool_logging(tool, tool_input, func)
    except Exception as exc:
        return f"ERROR: {tool} failed ({exc})"


def _set_param_descriptions(tool, descriptions: dict[str, str]):
    schema = getattr(tool, "params_json_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, description in descriptions.items():
            if isinstance(properties.get(name), dict):
                properties[name]["description"] = description
    current = str(getattr(tool, "description", "") or "").strip()
    if current and "Output:" not in current:
        tool.description = (
            f"{current}\n\n"
            "Parameters: Use the endpoint-specific parameter descriptions in the schema to provide company IDs, search text, market, locale, pagination, filters, and timeouts.\n"
            "Output: Returns a compact SUCCESS/ERROR text report with business, market, finance, listing, review, ad, filing, or product records. "
            "When raw endpoint data is preserved, the output includes an Artifact JSON path."
        )
    return tool


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short(text: str, max_chars: int = 260) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _html_to_text(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", str(value or ""))
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return " ".join(unescape(text).split())


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(part for part in (_as_text(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("name", "title", "value", "label", "price", "content", "text", "description", "snippet", "code", "id", "link", "url"):
            text = _as_text(value.get(key))
            if text:
                return text
    return ""


def _safe_filename(value: str, fallback: str = "business-data") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _artifact_dir(kind: str) -> str:
    root = research_artifacts_root()
    base = os.path.join(root, kind) if root else os.path.join("/tmp", "chack-business", kind)
    os.makedirs(base, exist_ok=True)
    return base


def _write_json_artifact(kind: str, label: str, payload: Any) -> str:
    output_dir = _artifact_dir(kind)
    path = os.path.join(output_dir, f"{_safe_filename(label)}_{uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    record_research_json_artifact(path, payload, provenance=f"{kind}:{label}", kind=kind, label=label)
    return path


def _normalize_cik(value: str) -> str:
    digits = re.sub(r"\D+", "", str(value or ""))
    if not digits:
        return ""
    return digits.zfill(10)


def _maybe(params: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        params[key] = text


def _price_text(value: Any) -> str:
    if isinstance(value, dict):
        amount = value.get("amount") or value.get("price") or value.get("value")
        currency = value.get("currency") or ""
        if amount not in (None, ""):
            return f"{amount} {currency}".strip()
    return _as_text(value)


def _first_yelp_place_id(item: dict[str, Any]) -> str:
    value = item.get("place_id")
    if value:
        return str(value)
    place_ids = item.get("place_ids")
    if isinstance(place_ids, list):
        for place_id in place_ids:
            if str(place_id or "").strip():
                return str(place_id).strip()
    link = str(item.get("link") or "").strip()
    match = re.search(r"/biz/([^/?#]+)", link)
    if match:
        return match.group(1)
    return ""


def _ads_transparency_region(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return raw
    upper = raw.upper()
    return _ADS_TRANSPARENCY_REGION_ALIASES.get(upper, raw)


class BusinessSearchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _max_results(self, requested: Optional[int], default_limit: int = 10) -> int:
        cfg_limit = _coerce_int(getattr(self.config, "business_max_results", default_limit), default_limit)
        cfg_limit = _clamp(cfg_limit, 1, 50)
        if requested is None:
            return cfg_limit
        return _clamp(_coerce_int(requested, cfg_limit), 1, 50)

    def _serpapi_request(self, params: dict[str, Any], timeout_seconds: int = 20) -> Any:
        api_keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if not api_keys:
            return "ERROR: No usable SerpAPI key (not configured or all keys exhausted)."
        for idx, api_key in enumerate(api_keys):
            req_params = dict(params)
            req_params["api_key"] = api_key
            req_params["output"] = "json"
            try:
                response = requests.get("https://serpapi.com/search", params=req_params, timeout=timeout_seconds)
            except requests.exceptions.Timeout:
                return "ERROR: SerpAPI request timed out"
            except requests.exceptions.ConnectionError:
                return "ERROR: Failed to connect to SerpAPI"
            if response.status_code >= 400:
                body = (response.text or "").strip().replace("\n", " ")
                if len(body) > 220:
                    body = body[:217] + "..."
                note_serpapi_response_error(api_key, response.status_code, body)
                if is_serpapi_rate_limited(response.status_code, body) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI returned HTTP {response.status_code} ({body})"
            try:
                payload = response.json()
            except ValueError:
                return "ERROR: SerpAPI returned invalid JSON"
            if isinstance(payload, dict) and payload.get("error"):
                error_text = str(payload.get("error") or "")
                note_serpapi_response_error(api_key, response.status_code, error_text)
                if is_serpapi_rate_limited(response.status_code, error_text) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI error ({error_text})"
            return payload
        return "ERROR: All configured SerpAPI keys are rate limited."

    def _sec_tickers(self, timeout_seconds: int = 20) -> Any:
        try:
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=_SEC_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return "ERROR: SEC company tickers request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to SEC"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: SEC returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: SEC returned invalid JSON"

    def _resolve_sec_company(self, value: str, timeout_seconds: int = 20) -> tuple[str, str, str] | str:
        raw = str(value or "").strip()
        raw_digits = re.sub(r"\D+", "", raw)
        cik = _normalize_cik(raw)
        if cik and raw.isdigit():
            return cik, "", ""
        payload = self._sec_tickers(timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        rows = payload.values() if isinstance(payload, dict) else []
        target = raw.lower()
        for item in rows:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "").strip()
            title = str(item.get("title") or "").strip()
            if ticker.lower() == target or title.lower() == target:
                return str(item.get("cik_str") or "").zfill(10), ticker, title
        for item in rows:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if target and target in title.lower():
                return str(item.get("cik_str") or "").zfill(10), str(item.get("ticker") or ""), title
        if cik and len(raw_digits) >= 6:
            return cik, "", ""
        return f"ERROR: Could not resolve SEC company '{value}' to a CIK"

    def search_sec_companies(
        self,
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        payload = self._sec_tickers(timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("sec", f"company_tickers_{query}", payload)
        limit = self._max_results(max_results)
        rows = []
        needle = query.lower()
        for item in (payload.values() if isinstance(payload, dict) else []):
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "")
            title = str(item.get("title") or "")
            cik = str(item.get("cik_str") or "").zfill(10)
            if needle in ticker.lower() or needle in title.lower() or needle in cik:
                rows.append((ticker, title, cik))
        if not rows:
            return f"SUCCESS: No SEC company ticker matches found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: SEC company matches for '{query}' (top {min(len(rows), limit)}):"]
        for idx, (ticker, title, cik) in enumerate(rows[:limit], start=1):
            lines.append(f"{idx}. {title} | ticker: {ticker} | CIK: {cik}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_sec_company_submissions(
        self,
        company: str,
        form_filter: str = "",
        max_filings: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        resolved = self._resolve_sec_company(company, timeout_seconds=timeout_seconds)
        if isinstance(resolved, str):
            return resolved
        cik, ticker, title = resolved
        try:
            response = requests.get(
                f"https://data.sec.gov/submissions/CIK{cik}.json",
                headers=_SEC_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: SEC submissions request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to SEC"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: SEC returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: SEC returned invalid JSON"
        artifact = _write_json_artifact("sec", f"submissions_{cik}", payload)
        recent = (payload.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        accession = recent.get("accessionNumber") or []
        filing_dates = recent.get("filingDate") or []
        report_dates = recent.get("reportDate") or []
        primary_docs = recent.get("primaryDocument") or []
        limit = self._max_results(max_filings, default_limit=12)
        wanted = str(form_filter or "").strip().upper()
        lines = [
            f"SUCCESS: SEC submissions for {title or payload.get('name') or company} | ticker: {ticker or payload.get('tickers')} | CIK: {cik}"
        ]
        count = 0
        for idx, form in enumerate(forms):
            form_text = str(form or "")
            if wanted and form_text.upper() != wanted:
                continue
            acc = str(accession[idx] if idx < len(accession) else "")
            doc = str(primary_docs[idx] if idx < len(primary_docs) else "")
            filing_date = str(filing_dates[idx] if idx < len(filing_dates) else "")
            report_date = str(report_dates[idx] if idx < len(report_dates) else "")
            accession_no_dash = acc.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}/{doc}" if acc and doc else ""
            count += 1
            lines.append(f"{count}. {form_text} | filed: {filing_date} | report: {report_date} | accession: {acc}")
            if filing_url:
                lines.append(f"   filing: {filing_url}")
            if count >= limit:
                break
        if count == 0:
            lines.append(f"No recent filings matched form_filter='{form_filter}'.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_sec_company_facts(
        self,
        company: str,
        tags: str = "Revenues,RevenueFromContractWithCustomerExcludingAssessedTax,NetIncomeLoss,Assets,Liabilities,StockholdersEquity",
        max_facts_per_tag: int = 5,
        timeout_seconds: int = 20,
    ) -> str:
        resolved = self._resolve_sec_company(company, timeout_seconds=timeout_seconds)
        if isinstance(resolved, str):
            return resolved
        cik, ticker, title = resolved
        try:
            response = requests.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                headers=_SEC_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: SEC company facts request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to SEC"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: SEC returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: SEC returned invalid JSON"
        artifact = _write_json_artifact("sec", f"companyfacts_{cik}", payload)
        facts = ((payload.get("facts") or {}).get("us-gaap") or {})
        wanted_tags = [tag.strip() for tag in str(tags or "").split(",") if tag.strip()]
        limit = _clamp(_coerce_int(max_facts_per_tag, 5), 1, 20)
        lines = [f"SUCCESS: SEC company facts for {title or payload.get('entityName') or company} | ticker: {ticker} | CIK: {cik}"]
        for tag in wanted_tags:
            fact = facts.get(tag)
            if not isinstance(fact, dict):
                continue
            units = fact.get("units") or {}
            unit_rows = []
            for unit_name, rows in units.items():
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            unit_rows.append((unit_name, row))
            unit_rows.sort(key=lambda pair: str(pair[1].get("filed") or pair[1].get("end") or ""), reverse=True)
            if not unit_rows:
                continue
            label = fact.get("label") or tag
            lines.append(f"{tag} ({label}):")
            for unit_name, row in unit_rows[:limit]:
                value = row.get("val")
                fy = row.get("fy") or ""
                fp = row.get("fp") or ""
                end = row.get("end") or ""
                filed = row.get("filed") or ""
                form = row.get("form") or ""
                lines.append(f"- {value} {unit_name} | fy/fp: {fy}{fp} | end: {end} | filed: {filed} | form: {form}")
        if len(lines) == 1:
            lines.append("No requested us-gaap tags were found.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_gleif_lei(
        self,
        query: str,
        country: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {
            "filter[fulltext]": query,
            "page[size]": limit,
        }
        if country.strip():
            params["filter[entity.legalAddress.country]"] = country.strip().upper()
        try:
            response = requests.get(
                "https://api.gleif.org/api/v1/lei-records",
                params=params,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: GLEIF request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to GLEIF"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: GLEIF returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: GLEIF returned invalid JSON"
        artifact = _write_json_artifact("gleif", f"lei_search_{query}", payload)
        rows = payload.get("data") or []
        if not isinstance(rows, list) or not rows:
            return f"SUCCESS: No GLEIF LEI records found for '{query}'.\nArtifact JSON: {artifact}"
        total = ((payload.get("meta") or {}).get("pagination") or {}).get("total")
        lines = [f"SUCCESS: GLEIF LEI matches for '{query}' (top {min(len(rows), limit)}; total: {total}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            attrs = item.get("attributes") or {}
            entity = attrs.get("entity") or {}
            registration = attrs.get("registration") or {}
            legal_name = _as_text((entity.get("legalName") or {}).get("name") or entity.get("legalName"))
            lei = attrs.get("lei") or item.get("id") or ""
            status = _as_text(registration.get("status") or entity.get("status"))
            country_code = _as_text((entity.get("legalAddress") or {}).get("country"))
            jurisdiction = _as_text(entity.get("jurisdiction"))
            lines.append(f"{idx}. {legal_name} | LEI: {lei} | status: {status} | jurisdiction: {jurisdiction} | country: {country_code}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_gleif_lei_record(self, lei: str, timeout_seconds: int = 20) -> str:
        lei = str(lei or "").strip()
        if not lei:
            return "ERROR: lei cannot be empty"
        try:
            response = requests.get(
                f"https://api.gleif.org/api/v1/lei-records/{lei}",
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: GLEIF request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to GLEIF"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: GLEIF returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: GLEIF returned invalid JSON"
        artifact = _write_json_artifact("gleif", f"lei_{lei}", payload)
        data = payload.get("data") or {}
        attrs = data.get("attributes") or {}
        entity = attrs.get("entity") or {}
        registration = attrs.get("registration") or {}
        legal_name = _as_text((entity.get("legalName") or {}).get("name") or entity.get("legalName"))
        lines = [f"SUCCESS: GLEIF LEI record for '{lei}':", f"Legal name: {legal_name}"]
        for label, value in [
            ("LEI", attrs.get("lei") or data.get("id")),
            ("Entity status", entity.get("status")),
            ("Jurisdiction", entity.get("jurisdiction")),
            ("Legal form", entity.get("legalForm")),
            ("Registration status", registration.get("status")),
            ("Initial registration", registration.get("initialRegistrationDate")),
            ("Last update", registration.get("lastUpdateDate")),
            ("Next renewal", registration.get("nextRenewalDate")),
            ("Managing LOU", registration.get("managingLou")),
        ]:
            text = _as_text(value)
            if text:
                lines.append(f"{label}: {text}")
        for label, key in [("Legal address", "legalAddress"), ("Headquarters address", "headquartersAddress")]:
            address = entity.get(key) or {}
            if isinstance(address, dict):
                parts = []
                for part in address.get("addressLines") or []:
                    if part:
                        parts.append(str(part))
                for key2 in ("city", "region", "postalCode", "country"):
                    value = address.get(key2)
                    if value:
                        parts.append(str(value))
                if parts:
                    lines.append(f"{label}: {', '.join(parts)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_google_finance(
        self,
        query: str,
        window: str = "1D",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        window = str(window or "1D").strip().upper()
        if window not in _FINANCE_WINDOWS:
            return "ERROR: window must be one of 1D, 5D, 1M, 6M, YTD, 1Y, 5Y, MAX"
        params = {"engine": "google_finance", "q": query, "window": window}
        if hl.strip():
            params["hl"] = hl.strip().lower()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-finance", query, payload)
        lines = [f"SUCCESS: Google Finance results for '{query}':"]
        summary = payload.get("summary") or {}
        if isinstance(summary, dict):
            title = summary.get("title") or summary.get("stock") or query
            price = summary.get("price") or summary.get("extracted_price") or ""
            exchange = summary.get("exchange") or ""
            movement = summary.get("price_movement") or {}
            move_text = ""
            if isinstance(movement, dict):
                move_text = " | ".join(
                    str(part)
                    for part in [
                        movement.get("movement"),
                        movement.get("percentage"),
                        movement.get("value"),
                    ]
                    if part not in (None, "")
                )
            lines.append(f"Summary: {title} | exchange: {exchange} | price: {price} | movement: {move_text}")
            for key in ("date", "currency", "market", "previous_close", "open", "day_range", "year_range", "market_cap", "pe_ratio", "dividend_yield"):
                value = summary.get(key)
                if value:
                    lines.append(f"- {key}: {_price_text(value) or _as_text(value)}")
        knowledge = payload.get("knowledge_graph") or {}
        if isinstance(knowledge, dict) and knowledge:
            description = knowledge.get("description") or knowledge.get("about") or ""
            website = knowledge.get("website") or ""
            if description or website:
                lines.append(f"Knowledge graph: {_short(description, 600)}" + (f" | website: {website}" if website else ""))
        financials = payload.get("financials") or []
        if isinstance(financials, list) and financials:
            lines.append("Financials:")
            for item in financials[:8]:
                if isinstance(item, dict):
                    label = item.get("title") or item.get("name") or item.get("metric") or ""
                    value = item.get("value") or item.get("amount") or item.get("description") or ""
                    if label or value:
                        lines.append(f"- {label}: {value}")
        news = payload.get("news_results") or []
        if isinstance(news, list) and news:
            lines.append("News:")
            for item in news[:6]:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or ""
                link = item.get("link") or ""
                source = _as_text(item.get("source"))
                date = item.get("date") or ""
                lines.append(f"- {title} | {source} | {date} | {link}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_google_finance_markets(
        self,
        gl: str = "",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        params = {"engine": "google_finance_markets", "trend": "indexes"}
        if gl.strip():
            params["gl"] = gl.strip().lower()
        if hl.strip():
            params["hl"] = hl.strip().lower()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-finance", "markets_indexes", payload)
        markets = payload.get("markets") or {}
        lines = ["SUCCESS: Google Finance market indexes:"]
        if isinstance(markets, dict):
            for region, rows in markets.items():
                if not isinstance(rows, list):
                    continue
                lines.append(f"{region}:")
                for item in rows[:8]:
                    if not isinstance(item, dict):
                        continue
                    movement = item.get("price_movement") or {}
                    move_text = ""
                    if isinstance(movement, dict):
                        move_text = " | ".join(
                            str(part)
                            for part in [movement.get("movement"), movement.get("percentage"), movement.get("value")]
                            if part not in (None, "")
                        )
                    lines.append(f"- {item.get('name') or item.get('stock')}: {item.get('price')} | {move_text} | {item.get('stock') or ''}")
        news = payload.get("news_results") or []
        if isinstance(news, list) and news:
            lines.append("Market news:")
            for item in news[:6]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('title') or ''} | {item.get('link') or ''}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_google_maps(
        self,
        query: str,
        location: str = "",
        ll: str = "",
        z: str = "14",
        m: str = "",
        gl: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {"engine": "google_maps", "type": "search", "q": query}
        _maybe(params, "location", location)
        _maybe(params, "ll", ll)
        if location.strip() and not ll.strip():
            _maybe(params, "m", m)
            if not str(m or "").strip():
                _maybe(params, "z", z or "14")
        _maybe(params, "gl", gl.lower() if gl else "")
        _maybe(params, "hl", hl.lower() if hl else "")
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-maps", query, payload)
        rows = payload.get("local_results") or []
        lines = [f"SUCCESS: Google Maps local results for '{query}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            if not isinstance(item, dict):
                continue
            address = item.get("address") or item.get("place") or ""
            phone = item.get("phone") or ""
            website = item.get("website") or ""
            types = _as_text(item.get("types") or item.get("type"))
            lines.append(
                f"{idx}. {item.get('title') or item.get('name') or ''} | rating: {item.get('rating') or ''} | reviews: {item.get('reviews') or ''} | type: {types}"
            )
            lines.append(f"   address: {address} | phone: {phone} | website: {website}")
            lines.append(f"   place_id: {item.get('place_id') or ''} | data_id: {item.get('data_id') or ''} | data_cid: {item.get('data_cid') or ''}")
        if not rows:
            lines.append("No local_results returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_google_maps_reviews(
        self,
        data_id: str = "",
        place_id: str = "",
        sort_by: str = "",
        hl: str = "",
        gl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not str(data_id or "").strip() and not str(place_id or "").strip():
            return "ERROR: data_id or place_id is required"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {"engine": "google_maps_reviews"}
        if str(data_id or "").strip():
            _maybe(params, "data_id", data_id)
        else:
            _maybe(params, "place_id", place_id)
        _maybe(params, "sort_by", sort_by)
        _maybe(params, "hl", hl.lower() if hl else "")
        _maybe(params, "gl", gl.lower() if gl else "")
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        label = data_id or place_id
        artifact = _write_json_artifact("google-maps-reviews", label, payload)
        reviews = payload.get("reviews") or []
        place_info = payload.get("place_info") or payload.get("place_results") or {}
        lines = [f"SUCCESS: Google Maps reviews for '{label}' (top {min(len(reviews), limit)}):"]
        if isinstance(place_info, dict) and place_info:
            lines.append(f"Place: {place_info.get('title') or place_info.get('name') or ''} | rating: {place_info.get('rating') or ''} | reviews: {place_info.get('reviews') or ''}")
        for idx, item in enumerate(reviews[:limit], start=1):
            if not isinstance(item, dict):
                continue
            user = item.get("user") or {}
            username = _as_text(user.get("name") if isinstance(user, dict) else user)
            lines.append(f"{idx}. rating: {item.get('rating') or ''} | date: {item.get('date') or item.get('iso_date') or ''} | user: {username}")
            text = item.get("snippet") or item.get("text") or item.get("review") or ""
            if text:
                lines.append(f"   {_short(text, 500)}")
        if not reviews:
            lines.append("No reviews returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_yelp_businesses(
        self,
        find_desc: str,
        find_loc: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        find_desc = str(find_desc or "").strip()
        if not find_desc:
            return "ERROR: find_desc cannot be empty"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {"engine": "yelp", "find_desc": find_desc}
        _maybe(params, "find_loc", find_loc)
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("yelp", f"{find_desc}_{find_loc}", payload)
        rows = payload.get("organic_results") or []
        lines = [f"SUCCESS: Yelp business results for '{find_desc}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            if not isinstance(item, dict):
                continue
            place_id = _first_yelp_place_id(item)
            lines.append(f"{idx}. {item.get('title') or ''} | rating: {item.get('rating') or ''} | reviews: {item.get('reviews') or ''} | price: {item.get('price') or ''}")
            lines.append(f"   place_id: {place_id} | phone: {item.get('phone') or ''} | neighborhoods: {_as_text(item.get('neighborhoods'))}")
            snippet = item.get("snippet") or item.get("description") or ""
            if snippet:
                lines.append(f"   {_short(snippet, 350)}")
        if not rows:
            lines.append("No organic_results returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_yelp_place(self, place_id: str, timeout_seconds: int = 20) -> str:
        place_id = str(place_id or "").strip()
        if not place_id:
            return "ERROR: place_id cannot be empty"
        payload = self._serpapi_request({"engine": "yelp_place", "place_id": place_id}, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("yelp", f"place_{place_id}", payload)
        place = payload.get("place_results") or {}
        lines = [f"SUCCESS: Yelp place for '{place_id}':"]
        if isinstance(place, dict):
            lines.append(f"{place.get('name') or place.get('title') or ''} | rating: {place.get('rating') or ''} | reviews: {place.get('reviews') or ''} | price: {place.get('price') or ''}")
            for key in ("address", "phone", "website", "hours", "categories", "highlights"):
                value = _as_text(place.get(key))
                if value:
                    lines.append(f"{key}: {_short(value, 600)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_yelp_reviews(
        self,
        place_id: str,
        query: str = "",
        sortby: str = "",
        rating: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        place_id = str(place_id or "").strip()
        if not place_id:
            return "ERROR: place_id cannot be empty"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {"engine": "yelp_reviews", "place_id": place_id}
        _maybe(params, "q", query)
        _maybe(params, "sortby", sortby)
        _maybe(params, "rating", rating)
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("yelp", f"reviews_{place_id}", payload)
        reviews = payload.get("reviews") or []
        lines = [f"SUCCESS: Yelp reviews for '{place_id}' (top {min(len(reviews), limit)}):"]
        for idx, item in enumerate(reviews[:limit], start=1):
            if not isinstance(item, dict):
                continue
            user = item.get("user") or item.get("author") or {}
            lines.append(f"{idx}. rating: {item.get('rating') or ''} | date: {item.get('date') or ''} | user: {_as_text(user)}")
            text = _as_text(item.get("comment") or item.get("text") or item.get("snippet") or "")
            if text:
                lines.append(f"   {_short(text, 500)}")
            response = item.get("owner_response") or item.get("business_response") or ""
            if response:
                lines.append(f"   owner response: {_short(_as_text(response), 350)}")
        if not reviews:
            lines.append("No reviews returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_apple_maps(
        self,
        query: str,
        location: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {"engine": "apple_maps", "query": query}
        _maybe(params, "location", location or "Austin, Texas, United States")
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("apple-maps", f"{query}_{location}", payload)
        rows = payload.get("place_results") or payload.get("local_results") or []
        lines = [f"SUCCESS: Apple Maps results for '{query}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(f"{idx}. {item.get('title') or item.get('name') or ''} | rating: {item.get('rating') or ''} | reviews: {item.get('reviews') or ''}")
            lines.append(f"   place_id: {item.get('place_id') or ''} | muid: {item.get('muid') or ''} | provider_id: {item.get('provider_id') or ''}")
            address = _as_text(item.get("address"))
            if address:
                lines.append(f"   address: {address}")
        if not rows:
            lines.append("No place/local results returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_apple_maps_place(self, muid: str, timeout_seconds: int = 20) -> str:
        muid = str(muid or "").strip()
        if not muid:
            return "ERROR: muid cannot be empty"
        payload = self._serpapi_request({"engine": "apple_maps_places", "muid": muid}, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("apple-maps", f"place_{muid}", payload)
        rows = payload.get("place_results") or []
        lines = [f"SUCCESS: Apple Maps place for '{muid}':"]
        if isinstance(rows, dict):
            rows = [rows]
        for idx, item in enumerate((rows or [])[:3], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(f"{idx}. {item.get('title') or item.get('name') or ''} | rating: {item.get('rating') or ''} | reviews: {item.get('reviews') or ''}")
            for key in ("address", "phone", "website", "hours", "categories", "amenities", "actions"):
                value = _as_text(item.get(key))
                if value:
                    lines.append(f"{key}: {_short(value, 600)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_google_ads(
        self,
        query: str,
        location: str = "",
        gl: str = "",
        hl: str = "",
        device: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {"engine": "google_ads", "q": query}
        _maybe(params, "location", location or "United States")
        _maybe(params, "gl", gl.lower() if gl else "")
        _maybe(params, "hl", hl.lower() if hl else "")
        _maybe(params, "device", device)
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-ads", query, payload)
        lines = [f"SUCCESS: Google Ads results for '{query}':"]
        for section in ("ads", "ad_results", "shopping_results", "local_results", "organic_results"):
            rows = payload.get(section) or []
            if not isinstance(rows, list) or not rows:
                continue
            lines.append(f"{section}:")
            for item in rows[:limit]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('title') or item.get('name') or ''} | {item.get('displayed_link') or item.get('source') or ''} | {item.get('link') or ''}")
                    text = item.get("description") or item.get("snippet") or ""
                    if text:
                        lines.append(f"  {_short(text, 350)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_google_ads_transparency(
        self,
        advertiser_id: str = "",
        text: str = "",
        region: str = "",
        platform: str = "",
        creative_format: str = "",
        start_date: str = "",
        end_date: str = "",
        next_page_token: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not str(advertiser_id or "").strip() and not str(text or "").strip():
            return "ERROR: advertiser_id or text is required"
        limit = _clamp(self._max_results(max_results, default_limit=40), 1, 100)
        params: dict[str, Any] = {"engine": "google_ads_transparency_center", "num": limit}
        _maybe(params, "advertiser_id", advertiser_id)
        _maybe(params, "text", text)
        _maybe(params, "region", _ads_transparency_region(region))
        _maybe(params, "platform", platform.upper() if platform else "")
        _maybe(params, "creative_format", creative_format.lower() if creative_format else "")
        _maybe(params, "start_date", start_date)
        _maybe(params, "end_date", end_date)
        _maybe(params, "next_page_token", next_page_token)
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        label = advertiser_id or text
        artifact = _write_json_artifact("google-ads-transparency", label, payload)
        rows = payload.get("ad_creatives") or payload.get("ads") or payload.get("organic_results") or []
        lines = [f"SUCCESS: Google Ads Transparency results for '{label}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(f"{idx}. advertiser: {_as_text(item.get('advertiser')) or item.get('advertiser_id') or ''} | format: {item.get('format') or item.get('creative_format') or ''}")
            lines.append(f"   shown: {item.get('first_shown') or item.get('first_seen') or ''} - {item.get('last_shown') or item.get('last_seen') or ''} | link: {item.get('link') or item.get('ad_link') or ''}")
            text_value = item.get("text") or item.get("title") or item.get("description") or ""
            if text_value:
                lines.append(f"   {_short(text_value, 400)}")
        if not rows:
            lines.append("No ad creatives/results returned.")
        next_token = ((payload.get("serpapi_pagination") or {}).get("next_page_token") or payload.get("next_page_token"))
        if next_token:
            lines.append(f"Next page token: {next_token}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_google_shopping(
        self,
        query: str,
        location: str = "",
        gl: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        return self._search_product_listing(
            engine="google_shopping",
            artifact_kind="google-shopping",
            query_param="q",
            query=query,
            extra_params={"location": location, "gl": gl.lower() if gl else "", "hl": hl.lower() if hl else ""},
            result_keys=("shopping_results", "inline_shopping_results", "categorized_shopping_results"),
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )

    def search_google_shopping_light(
        self,
        query: str,
        gl: str = "",
        hl: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        return self._search_product_listing(
            engine="google_shopping_light",
            artifact_kind="google-shopping",
            query_param="q",
            query=query,
            extra_params={"gl": gl.lower() if gl else "", "hl": hl.lower() if hl else ""},
            result_keys=("shopping_results", "inline_shopping_results", "categorized_shopping_results"),
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )

    def get_google_immersive_product(self, page_token: str, more_stores: bool = True, timeout_seconds: int = 20) -> str:
        page_token = str(page_token or "").strip()
        if not page_token:
            return "ERROR: page_token cannot be empty"
        params: dict[str, Any] = {"engine": "google_immersive_product", "page_token": page_token}
        if more_stores:
            params["more_stores"] = "true"
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-shopping", f"immersive_{page_token}", payload)
        lines = [f"SUCCESS: Google Immersive Product for token '{page_token}':"]
        product = payload.get("product_results") or payload.get("product") or payload
        if isinstance(product, dict):
            for key in ("title", "brand", "rating", "reviews", "price", "description", "pros", "cons"):
                value = _as_text(product.get(key))
                if value:
                    lines.append(f"{key}: {_short(value, 800)}")
        stores = payload.get("stores") or payload.get("sellers") or []
        if isinstance(stores, list) and stores:
            lines.append("Stores:")
            for item in stores[:12]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('name') or item.get('source') or ''} | price: {_price_text(item.get('price') or item.get('extracted_price'))} | {item.get('link') or ''}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_amazon_products(
        self,
        query: str,
        amazon_domain: str = "amazon.com",
        page: int = 1,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        return self._search_product_listing(
            engine="amazon",
            artifact_kind="amazon",
            query_param="k",
            query=query,
            extra_params={"amazon_domain": amazon_domain, "page": max(1, _coerce_int(page, 1))},
            result_keys=("organic_results", "featured_products", "product_ads"),
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )

    def get_amazon_product(self, asin: str, amazon_domain: str = "amazon.com", timeout_seconds: int = 20) -> str:
        return self._get_product_detail(
            engine="amazon_product",
            artifact_kind="amazon",
            id_param="asin",
            product_id=asin,
            extra_params={"amazon_domain": amazon_domain},
            timeout_seconds=timeout_seconds,
        )

    def search_walmart_products(
        self,
        query: str,
        walmart_domain: str = "walmart.com",
        page: int = 1,
        store_id: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        return self._search_product_listing(
            engine="walmart",
            artifact_kind="walmart",
            query_param="query",
            query=query,
            extra_params={"walmart_domain": walmart_domain, "page": max(1, _coerce_int(page, 1)), "store_id": store_id},
            result_keys=("organic_results", "featured_item"),
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )

    def get_walmart_product(self, product_id: str, timeout_seconds: int = 20) -> str:
        return self._get_product_detail(
            engine="walmart_product",
            artifact_kind="walmart",
            id_param="product_id",
            product_id=product_id,
            timeout_seconds=timeout_seconds,
        )

    def search_ebay_products(
        self,
        query: str,
        ebay_domain: str = "ebay.com",
        page: int = 1,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        return self._search_product_listing(
            engine="ebay",
            artifact_kind="ebay",
            query_param="_nkw",
            query=query,
            extra_params={"ebay_domain": ebay_domain, "_pgn": max(1, _coerce_int(page, 1))},
            result_keys=("organic_results", "inline_results", "deals"),
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )

    def get_ebay_product(
        self,
        product_id: str,
        ebay_domain: str = "ebay.com",
        shipping_country: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        return self._get_product_detail(
            engine="ebay_product",
            artifact_kind="ebay",
            id_param="product_id",
            product_id=product_id,
            extra_params={"ebay_domain": ebay_domain, "shipping_country": shipping_country},
            timeout_seconds=timeout_seconds,
        )

    def search_home_depot_products(
        self,
        query: str,
        country: str = "us",
        store_id: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        return self._search_product_listing(
            engine="home_depot",
            artifact_kind="home-depot",
            query_param="q",
            query=query,
            extra_params={"country": country.lower() if country else "", "store_id": store_id},
            result_keys=("products", "organic_results"),
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )

    def get_home_depot_product(
        self,
        product_id: str,
        country: str = "us",
        store_id: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        return self._get_product_detail(
            engine="home_depot_product",
            artifact_kind="home-depot",
            id_param="product_id",
            product_id=product_id,
            extra_params={"country": country.lower() if country else "", "store_id": store_id},
            timeout_seconds=timeout_seconds,
        )

    def search_tripadvisor(
        self,
        query: str,
        ssrc: str = "a",
        tripadvisor_domain: str = "www.tripadvisor.com",
        offset: int = 0,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params = {
            "engine": "tripadvisor",
            "q": query,
            "ssrc": ssrc or "a",
            "tripadvisor_domain": tripadvisor_domain or "www.tripadvisor.com",
            "offset": max(0, _coerce_int(offset, 0)),
            "limit": limit,
        }
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("tripadvisor", query, payload)
        rows = payload.get("places") or payload.get("results") or []
        lines = [f"SUCCESS: Tripadvisor results for '{query}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            if isinstance(item, dict):
                lines.append(f"{idx}. {item.get('name') or item.get('title') or ''} | place_id: {item.get('place_id') or ''} | rating: {item.get('rating') or ''} | reviews: {item.get('reviews') or ''}")
                address = _as_text(item.get("address") or item.get("location"))
                if address:
                    lines.append(f"   address/location: {address}")
                if item.get("link"):
                    lines.append(f"   {item.get('link')}")
        if not rows:
            lines.append("No places/results returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_tripadvisor_place(
        self,
        place_id: str,
        tripadvisor_domain: str = "www.tripadvisor.com",
        timeout_seconds: int = 20,
    ) -> str:
        return self._get_product_detail(
            engine="tripadvisor_place",
            artifact_kind="tripadvisor",
            id_param="place_id",
            product_id=place_id,
            extra_params={"tripadvisor_domain": tripadvisor_domain or "www.tripadvisor.com"},
            timeout_seconds=timeout_seconds,
        )

    def get_tripadvisor_reviews(
        self,
        place_id: str,
        tripadvisor_domain: str = "www.tripadvisor.com",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        place_id = str(place_id or "").strip()
        if not place_id:
            return "ERROR: place_id cannot be empty"
        limit = self._max_results(max_results)
        params = {"engine": "tripadvisor_reviews", "place_id": place_id, "tripadvisor_domain": tripadvisor_domain or "www.tripadvisor.com"}
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("tripadvisor", f"reviews_{place_id}", payload)
        reviews = payload.get("reviews") or []
        lines = [f"SUCCESS: Tripadvisor reviews for '{place_id}' (top {min(len(reviews), limit)}):"]
        for idx, item in enumerate(reviews[:limit], start=1):
            if isinstance(item, dict):
                lines.append(f"{idx}. rating: {item.get('rating') or ''} | date: {item.get('date') or item.get('published_date') or ''} | user: {_as_text(item.get('user') or item.get('author'))}")
                text = item.get("text") or item.get("review") or item.get("snippet") or item.get("title") or ""
                if text:
                    lines.append(f"   {_short(text, 500)}")
        if not reviews:
            lines.append("No reviews returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def _search_product_listing(
        self,
        *,
        engine: str,
        artifact_kind: str,
        query_param: str,
        query: str,
        extra_params: dict[str, Any] | None = None,
        result_keys: tuple[str, ...],
        max_results: Optional[int],
        timeout_seconds: int,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {"engine": engine, query_param: query}
        for key, value in (extra_params or {}).items():
            _maybe(params, key, value)
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact(artifact_kind, f"{engine}_{query}", payload)
        rows: list[Any] = []
        for key in result_keys:
            section = payload.get(key)
            if isinstance(section, list):
                rows.extend(section)
            elif isinstance(section, dict):
                rows.append(section)
        lines = [f"SUCCESS: {engine} product/listing results for '{query}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            if not isinstance(item, dict):
                continue
            product_id = item.get("product_id") or item.get("asin") or item.get("us_item_id") or item.get("item_id") or item.get("epid") or ""
            price = _price_text(item.get("price") or item.get("primary_offer") or item.get("extracted_price"))
            lines.append(f"{idx}. {item.get('title') or item.get('name') or ''} | id: {product_id} | price: {price} | rating: {item.get('rating') or ''} | reviews: {item.get('reviews') or item.get('reviews_count') or ''}")
            seller = item.get("seller") or item.get("source") or item.get("merchant") or item.get("brand") or ""
            if seller or item.get("link") or item.get("product_link"):
                lines.append(f"   seller/source: {_as_text(seller)} | link: {item.get('link') or item.get('product_link') or ''}")
            snippet = item.get("snippet") or item.get("description") or item.get("delivery") or ""
            if snippet:
                lines.append(f"   {_short(_as_text(snippet), 350)}")
            serpapi_product = item.get("serpapi_product_api") or item.get("serpapi_product_link") or item.get("serpapi_immersive_product_api") or item.get("serpapi_link") or ""
            if serpapi_product:
                lines.append(f"   serpapi detail: {serpapi_product}")
        if not rows:
            lines.append("No product/listing rows returned.")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def _get_product_detail(
        self,
        *,
        engine: str,
        artifact_kind: str,
        id_param: str,
        product_id: str,
        extra_params: dict[str, Any] | None = None,
        timeout_seconds: int,
    ) -> str:
        product_id = str(product_id or "").strip()
        if not product_id:
            return f"ERROR: {id_param} cannot be empty"
        params: dict[str, Any] = {"engine": engine, id_param: product_id}
        for key, value in (extra_params or {}).items():
            _maybe(params, key, value)
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact(artifact_kind, f"{engine}_{product_id}", payload)
        product = (
            payload.get("product_results")
            or payload.get("product_result")
            or payload.get("place_results")
            or payload.get("place_result")
            or payload.get("product")
            or payload.get("search_information")
            or {}
        )
        lines = [f"SUCCESS: {engine} detail for '{product_id}':"]
        if isinstance(product, dict):
            for key in (
                "title",
                "name",
                "brand",
                "rating",
                "reviews",
                "price",
                "description",
                "short_description_html",
                "short_description",
                "availability",
                "stock",
                "seller",
                "address",
                "phone",
                "website",
                "hours",
            ):
                value = product.get(key)
                text = _price_text(value) if key == "price" else _as_text(value)
                if key.endswith("_html"):
                    text = _html_to_text(text)
                if text:
                    lines.append(f"{key}: {_short(text, 800)}")
            for key in ("features", "about_this_item", "product_details", "specifications", "nearby"):
                value = _as_text(product.get(key))
                if value:
                    lines.append(f"{key}: {_short(value, 1000)}")
        for section in ("reviews", "reviews_results", "related_products", "other_sellers", "stores"):
            rows = payload.get(section)
            if isinstance(rows, list) and rows:
                lines.append(f"{section}:")
                for item in rows[:8]:
                    if isinstance(item, dict):
                        lines.append(f"- {item.get('title') or item.get('name') or _short(_as_text(item), 160)} | {_price_text(item.get('price'))}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)


def get_sec_company_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_sec_companies")
    def search_sec_companies(query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search SEC company tickers by company name, ticker, or CIK."""
        tool_input = {"query": query, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_sec_companies", tool_input, lambda: helper.search_sec_companies(query, max_results, timeout_seconds))

    return _set_param_descriptions(search_sec_companies, {
        "query": "Company name, ticker symbol, or CIK text to resolve against the SEC company ticker dataset.",
        "max_results": "Maximum number of matching SEC company records to return.",
        "timeout_seconds": "Maximum seconds to wait for loading/searching SEC company metadata.",
    })


def get_sec_company_submissions_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_sec_company_submissions")
    def get_sec_company_submissions(company: str, form_filter: str = "", max_filings: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Fetch recent SEC filings/submissions for a ticker, CIK, or company name."""
        tool_input = {"company": company, "form_filter": form_filter, "max_filings": max_filings, "timeout_seconds": timeout_seconds}
        return _run_logged("get_sec_company_submissions", tool_input, lambda: helper.get_sec_company_submissions(company, form_filter, max_filings, timeout_seconds))

    return _set_param_descriptions(get_sec_company_submissions, {
        "company": "Ticker, CIK, or company name to resolve before fetching SEC submissions.",
        "form_filter": "Optional SEC form type filter such as 10-K, 10-Q, 8-K, S-1, or DEF 14A.",
        "max_filings": "Maximum number of recent SEC filings to include.",
        "timeout_seconds": "Maximum seconds to wait for resolving the company and fetching submissions.",
    })


def get_sec_company_facts_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_sec_company_facts")
    def get_sec_company_facts(company: str, tags: str = "Revenues,RevenueFromContractWithCustomerExcludingAssessedTax,NetIncomeLoss,Assets,Liabilities,StockholdersEquity", max_facts_per_tag: int = 5, timeout_seconds: int = 20) -> str:
        """Fetch recent SEC XBRL company facts for selected us-gaap tags."""
        tool_input = {"company": company, "tags": tags, "max_facts_per_tag": max_facts_per_tag, "timeout_seconds": timeout_seconds}
        return _run_logged("get_sec_company_facts", tool_input, lambda: helper.get_sec_company_facts(company, tags, max_facts_per_tag, timeout_seconds))

    return _set_param_descriptions(get_sec_company_facts, {
        "company": "Ticker, CIK, or company name to resolve before fetching SEC XBRL company facts.",
        "tags": "Comma-separated us-gaap taxonomy tags to extract, such as Revenues, Assets, or NetIncomeLoss.",
        "max_facts_per_tag": "Maximum number of recent fact values to include for each requested tag.",
        "timeout_seconds": "Maximum seconds to wait for resolving the company and fetching XBRL facts.",
    })


def get_gleif_lei_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_gleif_lei")
    def search_gleif_lei(query: str, country: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search GLEIF's free Global LEI Index for legal entity identity records."""
        tool_input = {"query": query, "country": country, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_gleif_lei", tool_input, lambda: helper.search_gleif_lei(query, country, max_results, timeout_seconds))

    return _set_param_descriptions(search_gleif_lei, {
        "query": "Legal entity name, LEI, registration identifier, or address text to search in GLEIF.",
        "country": "Optional ISO 3166-1 alpha-2 country code to filter legal entity records.",
        "max_results": "Maximum number of GLEIF LEI records to return.",
        "timeout_seconds": "Maximum seconds to wait for the GLEIF search.",
    })


def get_gleif_lei_record_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_gleif_lei_record")
    def get_gleif_lei_record(lei: str, timeout_seconds: int = 20) -> str:
        """Fetch a single GLEIF LEI record by LEI code."""
        tool_input = {"lei": lei, "timeout_seconds": timeout_seconds}
        return _run_logged("get_gleif_lei_record", tool_input, lambda: helper.get_gleif_lei_record(lei, timeout_seconds))

    return _set_param_descriptions(get_gleif_lei_record, {
        "lei": "Exact 20-character Legal Entity Identifier to fetch from GLEIF.",
        "timeout_seconds": "Maximum seconds to wait for the GLEIF LEI record fetch.",
    })


def get_google_finance_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_finance")
    def search_google_finance(query: str, window: str = "1D", hl: str = "", timeout_seconds: int = 20) -> str:
        """Search Google Finance via SerpAPI for quotes, summary, financials, and related news."""
        tool_input = {"query": query, "window": window, "hl": hl, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_finance", tool_input, lambda: helper.search_google_finance(query, window, hl, timeout_seconds))

    return _set_param_descriptions(search_google_finance, {
        "query": "Google Finance query, usually a ticker, exchange-qualified symbol, company name, market index, or fund.",
        "window": "Google Finance chart window such as 1D, 5D, 1M, 6M, YTD, 1Y, 5Y, or MAX.",
        "hl": "Optional Google interface language code for finance results, such as en or es.",
        "timeout_seconds": "Maximum seconds to wait for the Google Finance SerpAPI request.",
    })


def get_google_finance_markets_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_finance_markets")
    def search_google_finance_markets(gl: str = "", hl: str = "", timeout_seconds: int = 20) -> str:
        """Search Google Finance Markets via SerpAPI for market index snapshots."""
        tool_input = {"gl": gl, "hl": hl, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_finance_markets", tool_input, lambda: helper.search_google_finance_markets(gl, hl, timeout_seconds))

    return _set_param_descriptions(search_google_finance_markets, {
        "gl": "Optional Google country code for localized market snapshots, such as us or es.",
        "hl": "Optional Google interface language code for market snapshots, such as en or es.",
        "timeout_seconds": "Maximum seconds to wait for the Google Finance Markets SerpAPI request.",
    })


def get_google_maps_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_maps_businesses")
    def search_google_maps_businesses(query: str, location: str = "", ll: str = "", z: str = "14", m: str = "", gl: str = "", hl: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Google Maps local business listings via SerpAPI."""
        tool_input = {"query": query, "location": location, "ll": ll, "z": z, "m": m, "gl": gl, "hl": hl, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_maps_businesses", tool_input, lambda: helper.search_google_maps(query, location, ll, z, m, gl, hl, max_results, timeout_seconds))

    return _set_param_descriptions(search_google_maps_businesses, {
        "query": "Business/category/place search query for Google Maps, such as coffee near gran via or Acme Inc.",
        "location": "Optional SerpAPI location string used as the Maps search origin, such as Madrid, Spain.",
        "ll": "Optional Google Maps latitude/longitude/@ zoom string to anchor the map search.",
        "z": "Optional Google Maps zoom level used with ll.",
        "m": "Optional Google Maps mode/data parameter passed through to SerpAPI when needed.",
        "gl": "Optional Google country code for localized Maps results, such as us or es.",
        "hl": "Optional Google interface language code for Maps results, such as en or es.",
        "max_results": "Maximum number of Google Maps business listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Google Maps SerpAPI request.",
    })


def get_google_maps_reviews_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_google_maps_reviews")
    def get_google_maps_reviews(data_id: str = "", place_id: str = "", sort_by: str = "", hl: str = "", gl: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Fetch Google Maps reviews by data_id or place_id via SerpAPI."""
        tool_input = {"data_id": data_id, "place_id": place_id, "sort_by": sort_by, "hl": hl, "gl": gl, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("get_google_maps_reviews", tool_input, lambda: helper.get_google_maps_reviews(data_id, place_id, sort_by, hl, gl, max_results, timeout_seconds))

    return _set_param_descriptions(get_google_maps_reviews, {
        "data_id": "Google Maps data_id from a search_google_maps_businesses result; use this or place_id.",
        "place_id": "Google place_id for the Maps location; use this when data_id is unavailable.",
        "sort_by": "Optional Google Maps reviews sort value, such as newest, highest_rating, lowest_rating, or most_relevant.",
        "hl": "Optional Google interface language code for reviews, such as en or es.",
        "gl": "Optional Google country code for localized reviews, such as us or es.",
        "max_results": "Maximum number of Google Maps reviews to return.",
        "timeout_seconds": "Maximum seconds to wait for the Google Maps Reviews SerpAPI request.",
    })


def get_yelp_business_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_yelp_businesses")
    def search_yelp_businesses(find_desc: str, find_loc: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Yelp business listings via SerpAPI."""
        tool_input = {"find_desc": find_desc, "find_loc": find_loc, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_yelp_businesses", tool_input, lambda: helper.search_yelp_businesses(find_desc, find_loc, max_results, timeout_seconds))

    return _set_param_descriptions(search_yelp_businesses, {
        "find_desc": "Yelp business/category/keyword search text, equivalent to Yelp's find_desc.",
        "find_loc": "Optional Yelp location text, equivalent to Yelp's find_loc, such as San Francisco, CA.",
        "max_results": "Maximum number of Yelp business listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Yelp Search SerpAPI request.",
    })


def get_yelp_place_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_yelp_place")
    def get_yelp_place(place_id: str, timeout_seconds: int = 20) -> str:
        """Fetch a Yelp place page by place_id via SerpAPI."""
        tool_input = {"place_id": place_id, "timeout_seconds": timeout_seconds}
        return _run_logged("get_yelp_place", tool_input, lambda: helper.get_yelp_place(place_id, timeout_seconds))

    return _set_param_descriptions(get_yelp_place, {
        "place_id": "Yelp place/business identifier from search_yelp_businesses.",
        "timeout_seconds": "Maximum seconds to wait for the Yelp place SerpAPI request.",
    })


def get_yelp_reviews_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_yelp_reviews")
    def get_yelp_reviews(place_id: str, query: str = "", sortby: str = "", rating: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Fetch Yelp reviews by place_id via SerpAPI."""
        tool_input = {"place_id": place_id, "query": query, "sortby": sortby, "rating": rating, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("get_yelp_reviews", tool_input, lambda: helper.get_yelp_reviews(place_id, query, sortby, rating, max_results, timeout_seconds))

    return _set_param_descriptions(get_yelp_reviews, {
        "place_id": "Yelp place/business identifier whose reviews should be fetched.",
        "query": "Optional text filter for Yelp reviews.",
        "sortby": "Optional Yelp reviews sort key such as relevance_desc, date_desc, rating_desc, or rating_asc.",
        "rating": "Optional Yelp star-rating filter.",
        "max_results": "Maximum number of Yelp reviews to return.",
        "timeout_seconds": "Maximum seconds to wait for the Yelp Reviews SerpAPI request.",
    })


def get_apple_maps_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_apple_maps_businesses")
    def search_apple_maps_businesses(query: str, location: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Apple Maps local business listings via SerpAPI."""
        tool_input = {"query": query, "location": location, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_apple_maps_businesses", tool_input, lambda: helper.search_apple_maps(query, location, max_results, timeout_seconds))

    return _set_param_descriptions(search_apple_maps_businesses, {
        "query": "Business/category/place search query for Apple Maps.",
        "location": "Optional location text used to localize Apple Maps results.",
        "max_results": "Maximum number of Apple Maps listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Apple Maps SerpAPI request.",
    })


def get_apple_maps_place_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_apple_maps_place")
    def get_apple_maps_place(muid: str, timeout_seconds: int = 20) -> str:
        """Fetch an Apple Maps place by muid via SerpAPI."""
        tool_input = {"muid": muid, "timeout_seconds": timeout_seconds}
        return _run_logged("get_apple_maps_place", tool_input, lambda: helper.get_apple_maps_place(muid, timeout_seconds))

    return _set_param_descriptions(get_apple_maps_place, {
        "muid": "Apple Maps MUID identifier from search_apple_maps_businesses.",
        "timeout_seconds": "Maximum seconds to wait for the Apple Maps place SerpAPI request.",
    })


def get_google_ads_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_ads")
    def search_google_ads(query: str, location: str = "", gl: str = "", hl: str = "", device: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Google paid ads results via SerpAPI."""
        tool_input = {"query": query, "location": location, "gl": gl, "hl": hl, "device": device, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_ads", tool_input, lambda: helper.search_google_ads(query, location, gl, hl, device, max_results, timeout_seconds))

    return _set_param_descriptions(search_google_ads, {
        "query": "Commercial query to search paid Google ad results for.",
        "location": "Optional SerpAPI location string for localized ads, such as Austin, Texas, United States.",
        "gl": "Optional Google country code for localized ad results, such as us or es.",
        "hl": "Optional Google interface language code for ad results, such as en or es.",
        "device": "Optional device type to emulate for ads, such as desktop, mobile, or tablet.",
        "max_results": "Maximum number of paid ad results to return.",
        "timeout_seconds": "Maximum seconds to wait for the Google Ads SerpAPI request.",
    })


def get_google_ads_transparency_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_ads_transparency")
    def search_google_ads_transparency(advertiser_id: str = "", text: str = "", region: str = "", platform: str = "", creative_format: str = "", start_date: str = "", end_date: str = "", next_page_token: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Google Ads Transparency Center by advertiser_id or text via SerpAPI."""
        tool_input = {"advertiser_id": advertiser_id, "text": text, "region": region, "platform": platform, "creative_format": creative_format, "start_date": start_date, "end_date": end_date, "next_page_token": next_page_token, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_ads_transparency", tool_input, lambda: helper.search_google_ads_transparency(advertiser_id, text, region, platform, creative_format, start_date, end_date, next_page_token, max_results, timeout_seconds))

    return _set_param_descriptions(search_google_ads_transparency, {
        "advertiser_id": "Optional Google Ads Transparency advertiser ID to inspect.",
        "text": "Optional text query for advertiser/ad creative search when advertiser_id is not enough.",
        "region": "Optional ad transparency region code or country alias, such as US, ES, or 2840.",
        "platform": "Optional platform filter for transparency results, such as GOOGLE_SEARCH or YOUTUBE where supported.",
        "creative_format": "Optional creative format filter such as text, image, or video where supported.",
        "start_date": "Optional earliest ad transparency date in YYYY-MM-DD format.",
        "end_date": "Optional latest ad transparency date in YYYY-MM-DD format.",
        "next_page_token": "Optional pagination token returned by a previous Ads Transparency request.",
        "max_results": "Maximum number of ad transparency records to return.",
        "timeout_seconds": "Maximum seconds to wait for the Ads Transparency SerpAPI request.",
    })


def get_google_shopping_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_shopping")
    def search_google_shopping(query: str, location: str = "", gl: str = "", hl: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Google Shopping listings via SerpAPI."""
        tool_input = {"query": query, "location": location, "gl": gl, "hl": hl, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_shopping", tool_input, lambda: helper.search_google_shopping(query, location, gl, hl, max_results, timeout_seconds))

    return _set_param_descriptions(search_google_shopping, {
        "query": "Product search query for Google Shopping.",
        "location": "Optional SerpAPI location string for localized Shopping results.",
        "gl": "Optional Google country code for Shopping results, such as us or es.",
        "hl": "Optional Google interface language code for Shopping results, such as en or es.",
        "max_results": "Maximum number of Google Shopping listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Google Shopping SerpAPI request.",
    })


def get_google_shopping_light_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_shopping_light")
    def search_google_shopping_light(query: str, gl: str = "", hl: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Google Shopping Light listings via SerpAPI."""
        tool_input = {"query": query, "gl": gl, "hl": hl, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_shopping_light", tool_input, lambda: helper.search_google_shopping_light(query, gl, hl, max_results, timeout_seconds))

    return _set_param_descriptions(search_google_shopping_light, {
        "query": "Product search query for the Google Shopping Light endpoint.",
        "gl": "Optional Google country code for Shopping Light results, such as us or es.",
        "hl": "Optional Google interface language code for Shopping Light results, such as en or es.",
        "max_results": "Maximum number of Google Shopping Light listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Google Shopping Light SerpAPI request.",
    })


def get_google_immersive_product_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_google_immersive_product")
    def get_google_immersive_product(page_token: str, more_stores: bool = True, timeout_seconds: int = 20) -> str:
        """Fetch Google Immersive Product details by page_token via SerpAPI."""
        tool_input = {"page_token": page_token, "more_stores": more_stores, "timeout_seconds": timeout_seconds}
        return _run_logged("get_google_immersive_product", tool_input, lambda: helper.get_google_immersive_product(page_token, more_stores, timeout_seconds))

    return _set_param_descriptions(get_google_immersive_product, {
        "page_token": "Google Immersive Product page_token from Google Shopping/Shopping Light results.",
        "more_stores": "When true, request additional seller/store offers for the immersive product.",
        "timeout_seconds": "Maximum seconds to wait for the Google Immersive Product SerpAPI request.",
    })


def get_amazon_product_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_amazon_products")
    def search_amazon_products(query: str, amazon_domain: str = "amazon.com", page: int = 1, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Amazon products via SerpAPI."""
        tool_input = {"query": query, "amazon_domain": amazon_domain, "page": page, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_amazon_products", tool_input, lambda: helper.search_amazon_products(query, amazon_domain, page, max_results, timeout_seconds))

    return _set_param_descriptions(search_amazon_products, {
        "query": "Amazon product search query.",
        "amazon_domain": "Amazon marketplace domain to search, such as amazon.com, amazon.es, or amazon.co.uk.",
        "page": "Amazon search results page number to request.",
        "max_results": "Maximum number of Amazon product listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Amazon Search SerpAPI request.",
    })


def get_amazon_product_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_amazon_product")
    def get_amazon_product(asin: str, amazon_domain: str = "amazon.com", timeout_seconds: int = 20) -> str:
        """Fetch Amazon product details by ASIN via SerpAPI."""
        tool_input = {"asin": asin, "amazon_domain": amazon_domain, "timeout_seconds": timeout_seconds}
        return _run_logged("get_amazon_product", tool_input, lambda: helper.get_amazon_product(asin, amazon_domain, timeout_seconds))

    return _set_param_descriptions(get_amazon_product, {
        "asin": "Amazon ASIN product identifier to fetch.",
        "amazon_domain": "Amazon marketplace domain for the ASIN, such as amazon.com, amazon.es, or amazon.co.uk.",
        "timeout_seconds": "Maximum seconds to wait for the Amazon Product SerpAPI request.",
    })


def get_walmart_product_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_walmart_products")
    def search_walmart_products(query: str, walmart_domain: str = "walmart.com", page: int = 1, store_id: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Walmart products via SerpAPI."""
        tool_input = {"query": query, "walmart_domain": walmart_domain, "page": page, "store_id": store_id, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_walmart_products", tool_input, lambda: helper.search_walmart_products(query, walmart_domain, page, store_id, max_results, timeout_seconds))

    return _set_param_descriptions(search_walmart_products, {
        "query": "Walmart product search query.",
        "walmart_domain": "Walmart marketplace domain to search, usually walmart.com.",
        "page": "Walmart search results page number to request.",
        "store_id": "Optional Walmart store ID to localize price/availability.",
        "max_results": "Maximum number of Walmart product listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Walmart Search SerpAPI request.",
    })


def get_walmart_product_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_walmart_product")
    def get_walmart_product(product_id: str, timeout_seconds: int = 20) -> str:
        """Fetch Walmart product details by product_id via SerpAPI."""
        tool_input = {"product_id": product_id, "timeout_seconds": timeout_seconds}
        return _run_logged("get_walmart_product", tool_input, lambda: helper.get_walmart_product(product_id, timeout_seconds))

    return _set_param_descriptions(get_walmart_product, {
        "product_id": "Walmart product ID from search_walmart_products.",
        "timeout_seconds": "Maximum seconds to wait for the Walmart Product SerpAPI request.",
    })


def get_ebay_product_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_ebay_products")
    def search_ebay_products(query: str, ebay_domain: str = "ebay.com", page: int = 1, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search eBay products via SerpAPI."""
        tool_input = {"query": query, "ebay_domain": ebay_domain, "page": page, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_ebay_products", tool_input, lambda: helper.search_ebay_products(query, ebay_domain, page, max_results, timeout_seconds))

    return _set_param_descriptions(search_ebay_products, {
        "query": "eBay product search query.",
        "ebay_domain": "eBay marketplace domain to search, such as ebay.com, ebay.es, or ebay.co.uk.",
        "page": "eBay search results page number to request.",
        "max_results": "Maximum number of eBay product listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the eBay Search SerpAPI request.",
    })


def get_ebay_product_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_ebay_product")
    def get_ebay_product(product_id: str, ebay_domain: str = "ebay.com", shipping_country: str = "", timeout_seconds: int = 20) -> str:
        """Fetch eBay product details by product_id via SerpAPI."""
        tool_input = {"product_id": product_id, "ebay_domain": ebay_domain, "shipping_country": shipping_country, "timeout_seconds": timeout_seconds}
        return _run_logged("get_ebay_product", tool_input, lambda: helper.get_ebay_product(product_id, ebay_domain, shipping_country, timeout_seconds))

    return _set_param_descriptions(get_ebay_product, {
        "product_id": "eBay product/listing ID from search_ebay_products.",
        "ebay_domain": "eBay marketplace domain for the listing, such as ebay.com, ebay.es, or ebay.co.uk.",
        "shipping_country": "Optional destination country code for shipping/availability context.",
        "timeout_seconds": "Maximum seconds to wait for the eBay Product SerpAPI request.",
    })


def get_home_depot_product_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_home_depot_products")
    def search_home_depot_products(query: str, country: str = "us", store_id: str = "", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Home Depot products via SerpAPI."""
        tool_input = {"query": query, "country": country, "store_id": store_id, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_home_depot_products", tool_input, lambda: helper.search_home_depot_products(query, country, store_id, max_results, timeout_seconds))

    return _set_param_descriptions(search_home_depot_products, {
        "query": "Home Depot product search query.",
        "country": "Home Depot country/market code, usually us or ca.",
        "store_id": "Optional Home Depot store ID to localize price/availability.",
        "max_results": "Maximum number of Home Depot product listings to return.",
        "timeout_seconds": "Maximum seconds to wait for the Home Depot Search SerpAPI request.",
    })


def get_home_depot_product_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_home_depot_product")
    def get_home_depot_product(product_id: str, country: str = "us", store_id: str = "", timeout_seconds: int = 20) -> str:
        """Fetch Home Depot product details by product_id via SerpAPI."""
        tool_input = {"product_id": product_id, "country": country, "store_id": store_id, "timeout_seconds": timeout_seconds}
        return _run_logged("get_home_depot_product", tool_input, lambda: helper.get_home_depot_product(product_id, country, store_id, timeout_seconds))

    return _set_param_descriptions(get_home_depot_product, {
        "product_id": "Home Depot product ID from search_home_depot_products.",
        "country": "Home Depot country/market code for the product, usually us or ca.",
        "store_id": "Optional Home Depot store ID to localize price/availability.",
        "timeout_seconds": "Maximum seconds to wait for the Home Depot Product SerpAPI request.",
    })


def get_tripadvisor_search_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_tripadvisor")
    def search_tripadvisor(query: str, ssrc: str = "a", tripadvisor_domain: str = "www.tripadvisor.com", offset: int = 0, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Tripadvisor places via SerpAPI."""
        tool_input = {"query": query, "ssrc": ssrc, "tripadvisor_domain": tripadvisor_domain, "offset": offset, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_tripadvisor", tool_input, lambda: helper.search_tripadvisor(query, ssrc, tripadvisor_domain, offset, max_results, timeout_seconds))

    return _set_param_descriptions(search_tripadvisor, {
        "query": "Tripadvisor place, hotel, restaurant, attraction, or destination search query.",
        "ssrc": "Tripadvisor search category code, such as a for all, h for hotels, r for restaurants, or A for attractions.",
        "tripadvisor_domain": "Tripadvisor domain to query, such as www.tripadvisor.com or www.tripadvisor.es.",
        "offset": "Tripadvisor result offset for pagination.",
        "max_results": "Maximum number of Tripadvisor search results to return.",
        "timeout_seconds": "Maximum seconds to wait for the Tripadvisor Search SerpAPI request.",
    })


def get_tripadvisor_place_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_tripadvisor_place")
    def get_tripadvisor_place(place_id: str, tripadvisor_domain: str = "www.tripadvisor.com", timeout_seconds: int = 20) -> str:
        """Fetch Tripadvisor place details by place_id via SerpAPI."""
        tool_input = {"place_id": place_id, "tripadvisor_domain": tripadvisor_domain, "timeout_seconds": timeout_seconds}
        return _run_logged("get_tripadvisor_place", tool_input, lambda: helper.get_tripadvisor_place(place_id, tripadvisor_domain, timeout_seconds))

    return _set_param_descriptions(get_tripadvisor_place, {
        "place_id": "Tripadvisor place/location ID from search_tripadvisor.",
        "tripadvisor_domain": "Tripadvisor domain for the place, such as www.tripadvisor.com or www.tripadvisor.es.",
        "timeout_seconds": "Maximum seconds to wait for the Tripadvisor place SerpAPI request.",
    })


def get_tripadvisor_reviews_tool(helper: BusinessSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_tripadvisor_reviews")
    def get_tripadvisor_reviews(place_id: str, tripadvisor_domain: str = "www.tripadvisor.com", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Fetch Tripadvisor reviews by place_id via SerpAPI."""
        tool_input = {"place_id": place_id, "tripadvisor_domain": tripadvisor_domain, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("get_tripadvisor_reviews", tool_input, lambda: helper.get_tripadvisor_reviews(place_id, tripadvisor_domain, max_results, timeout_seconds))

    return _set_param_descriptions(get_tripadvisor_reviews, {
        "place_id": "Tripadvisor place/location ID whose reviews should be fetched.",
        "tripadvisor_domain": "Tripadvisor domain for the reviews, such as www.tripadvisor.com or www.tripadvisor.es.",
        "max_results": "Maximum number of Tripadvisor reviews to return.",
        "timeout_seconds": "Maximum seconds to wait for the Tripadvisor Reviews SerpAPI request.",
    })
