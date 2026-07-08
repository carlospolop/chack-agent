from __future__ import annotations

import json
import os
import re
import time
from io import BytesIO
from html import unescape
from typing import Any, Optional
from urllib.parse import quote, urlparse
from uuid import uuid4
import xml.etree.ElementTree as ET

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests
from pypdf import PdfReader

from .config import ToolsConfig
from .research_artifacts import record_research_artifact, record_research_json_artifact, research_artifacts_root
from .telemetry import run_with_tool_logging


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 "
        "chack-research/1.0"
    ),
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}


def _run_logged(tool: str, tool_input: dict, func):
    try:
        return run_with_tool_logging(tool, tool_input, func)
    except Exception as exc:
        return f"ERROR: {tool} failed ({exc})"


def _set_param_descriptions(tool: Any, descriptions: dict[str, str]):
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
            "Parameters: Use the endpoint-specific parameter descriptions in the schema to choose IDs, filters, limits, dates, languages, and timeouts.\n"
            "Output: Returns a compact SUCCESS/ERROR text report with the matched records, extracted content, or metadata. "
            "When content is fetched or downloaded, the output includes local artifact paths for saved JSON/raw/text/PDF evidence."
        )
    return tool


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short(text: Any, max_chars: int = 320) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _safe_filename(value: str, fallback: str = "source") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _artifact_dir(kind: str) -> str:
    root = research_artifacts_root()
    base = os.path.join(root, kind) if root else os.path.join("/tmp", "chack-open-research", kind)
    os.makedirs(base, exist_ok=True)
    return base


def _write_json_artifact(kind: str, label: str, payload: Any) -> str:
    path = os.path.join(_artifact_dir(kind), f"{_safe_filename(label)}_{uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    record_research_json_artifact(path, payload, provenance=f"{kind}:{label}", kind=kind, label=label)
    return path


def _write_text_artifacts(
    kind: str,
    label: str,
    raw_ext: str,
    raw_text: str,
    plain_text: str,
    *,
    source_url: str = "",
    tool: str = "",
) -> tuple[str, str]:
    base = _safe_filename(label)
    token = uuid4().hex
    out_dir = _artifact_dir(kind)
    raw_path = os.path.join(out_dir, f"{base}_{token}.{raw_ext}")
    text_path = os.path.join(out_dir, f"{base}_{token}.txt")
    with open(raw_path, "w", encoding="utf-8") as handle:
        handle.write(raw_text)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(plain_text)
    record_research_artifact(
        raw_path,
        source_url=source_url,
        provenance=f"{kind}:{label}",
        tool=tool,
        kind=kind,
        label=label,
    )
    record_research_artifact(
        text_path,
        source_url=source_url,
        provenance=f"{kind}:{label}",
        tool=tool,
        kind=kind,
        label=label,
    )
    return raw_path, text_path


def _write_text_artifact(
    kind: str,
    label: str,
    plain_text: str,
    *,
    source_url: str = "",
    tool: str = "",
) -> str:
    path = os.path.join(_artifact_dir(kind), f"{_safe_filename(label)}_{uuid4().hex}.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(plain_text or ""))
    record_research_artifact(
        path,
        source_url=source_url,
        provenance=f"{kind}:{label}",
        tool=tool,
        kind=kind,
        label=label,
    )
    return path


def _html_to_text(html_text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", str(html_text or ""))
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _xml_to_text(xml_text: str) -> str:
    try:
        root = ET.fromstring(str(xml_text or "").encode("utf-8"))
    except Exception:
        return _html_to_text(str(xml_text or ""))
    chunks: list[str] = []
    for node in root.iter():
        if node.text and node.text.strip():
            chunks.append(node.text.strip())
        if node.tail and node.tail.strip():
            chunks.append(node.tail.strip())
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()


def _value(data: Any, *keys: str) -> str:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                return str(value).strip()
            if isinstance(value, list):
                text = ", ".join(_value(item, "name", "title", "label", "value", "id") for item in value[:8])
                text = ", ".join(part for part in text.split(", ") if part)
                if text:
                    return text
            if isinstance(value, dict):
                text = _value(value, "name", "title", "label", "value", "id")
                if text:
                    return text
    return ""


def _boe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("texto", "text", "titulo", "title", "nombre", "name", "valor", "value", "codigo", "id"):
            text = _boe_text(value.get(key))
            if text:
                return text
    if isinstance(value, list):
        return ", ".join(part for part in (_boe_text(item) for item in value[:8]) if part)
    return ""


def _join_json_text(value: Any) -> str:
    chunks: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            text = _join_json_text(item)
            if text:
                chunks.append(text)
    elif isinstance(value, list):
        for item in value:
            text = _join_json_text(item)
            if text:
                chunks.append(text)
    elif isinstance(value, (str, int, float, bool)):
        text = _html_to_text(str(value))
        if text:
            chunks.append(text)
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()


def _suttacentral_bilara_path(uid: str) -> tuple[str, str] | None:
    text_uid = str(uid or "").strip().lower()
    if not re.match(r"^[a-z0-9.-]+$", text_uid):
        return None
    match = re.match(r"^(dn|mn)(\d+[a-z]?)$", text_uid)
    if match:
        collection = match.group(1)
        return f"sutta/{collection}", text_uid
    match = re.match(r"^(sn|an)(\d+)\.(\d+[a-z]?)$", text_uid)
    if match:
        collection = match.group(1)
        chapter = match.group(2)
        return f"sutta/{collection}/{collection}{chapter}", text_uid
    match = re.match(r"^(dhp)(\d+-\d+)$", text_uid)
    if match:
        return "sutta/kn/dhp", text_uid
    return None


class OpenResearchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _max_results(self, requested: Optional[int], default_limit: int = 10) -> int:
        cfg = _coerce_int(
            getattr(self.config, "open_research_max_results", default_limit),
            default_limit,
        )
        if not cfg:
            cfg = default_limit
        if requested is None:
            return _clamp(cfg, 1, 50)
        return _clamp(_coerce_int(requested, cfg), 1, 50)

    def fetch_url_text(self, url: str, timeout_seconds: int = 30) -> str:
        url = str(url or "").strip()
        if not url:
            return "ERROR: url cannot be empty"
        if not url.lower().startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        try:
            response = requests.get(url, headers=_HEADERS, timeout=timeout_seconds, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: URL fetch timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to URL"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: URL returned HTTP {exc.response.status_code}"
        content_type = str(response.headers.get("content-type") or "").lower()
        raw = response.text or ""
        text = _html_to_text(raw) if "html" in content_type or "<html" in raw[:500].lower() else raw
        if len(text.strip()) < 40:
            return "ERROR: fetched URL did not contain enough extractable text"
        host = urlparse(str(response.url or url)).netloc or "page"
        raw_ext = "html" if "html" in content_type or "<html" in raw[:500].lower() else "txt"
        final_url = str(response.url or url)
        raw_path, text_path = _write_text_artifacts(
            "web-pages",
            host,
            raw_ext,
            raw,
            text,
            source_url=final_url,
            tool="fetch_url_text",
        )
        return (
            "SUCCESS: Fetched page text.\n"
            f"URL: {final_url}\n"
            f"Characters: {len(text)}\n"
            f"Saved raw: {raw_path}\n"
            f"Saved text: {text_path}\n"
            f"Excerpt: {_short(text, 900)}"
        )

    def search_wayback_cdx(
        self,
        url: str,
        from_year: str = "",
        to_year: str = "",
        match_type: str = "prefix",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        url = str(url or "").strip()
        if not url:
            return "ERROR: url cannot be empty"
        match_type = str(match_type or "prefix").strip().lower()
        if match_type not in {"exact", "prefix", "host", "domain"}:
            return "ERROR: match_type must be one of exact, prefix, host, domain"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {
            "url": url,
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype,digest,length",
            "filter": "statuscode:200",
            "collapse": "digest",
            "limit": limit,
            "matchType": match_type,
        }
        if str(from_year or "").strip():
            params["from"] = str(from_year).strip()
        if str(to_year or "").strip():
            params["to"] = str(to_year).strip()
        try:
            response = requests.get(
                "https://web.archive.org/cdx",
                params=params,
                headers=_HEADERS,
                timeout=timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Wayback CDX request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Wayback CDX"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Wayback CDX returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Wayback CDX returned invalid JSON"
        artifact = _write_json_artifact("wayback-cdx", url, payload)
        if not isinstance(payload, list) or len(payload) <= 1:
            return f"SUCCESS: No Wayback captures found for '{url}'.\nArtifact JSON: {artifact}"
        rows = payload[1:limit + 1]
        lines = [f"SUCCESS: Wayback captures for '{url}' (top {len(rows)}):"]
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, list) or len(row) < 2:
                continue
            timestamp = str(row[0])
            original = str(row[1])
            archive_url = f"https://web.archive.org/web/{timestamp}/{original}"
            meta = []
            for label, pos in [("status", 2), ("type", 3), ("digest", 4), ("length", 5)]:
                if len(row) > pos and str(row[pos]).strip():
                    meta.append(f"{label}: {row[pos]}")
            lines.append(f"{idx}. {timestamp} - {archive_url}")
            if meta:
                lines.append(f"   {' | '.join(meta)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def fetch_wayback_capture(self, url: str, timestamp: str = "", timeout_seconds: int = 30) -> str:
        url = str(url or "").strip()
        timestamp = str(timestamp or "").strip()
        if not url:
            return "ERROR: url cannot be empty"
        if not timestamp:
            try:
                response = requests.get(
                    "https://archive.org/wayback/available",
                    params={"url": url},
                    headers=_HEADERS,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                closest = ((payload.get("archived_snapshots") or {}).get("closest") or {})
                timestamp = str(closest.get("timestamp") or "").strip()
                archive_url = str(closest.get("url") or "").strip()
            except requests.exceptions.Timeout:
                return "ERROR: Wayback availability request timed out"
            except requests.exceptions.ConnectionError:
                return "ERROR: Failed to connect to Wayback availability API"
            except requests.exceptions.HTTPError as exc:
                return f"ERROR: Wayback availability returned HTTP {exc.response.status_code}"
            except ValueError:
                return "ERROR: Wayback availability returned invalid JSON"
            if not timestamp:
                return f"SUCCESS: No available Wayback capture found for '{url}'."
        else:
            archive_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
        try:
            response = requests.get(archive_url, headers=_HEADERS, timeout=timeout_seconds, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: Wayback capture fetch timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Wayback capture"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Wayback capture returned HTTP {exc.response.status_code}"
        raw = response.text or ""
        text = _html_to_text(raw)
        if len(text.strip()) < 40:
            return "ERROR: Wayback capture did not contain enough extractable text"
        raw_path, text_path = _write_text_artifacts("wayback-captures", url, "html", raw, text)
        return (
            "SUCCESS: Downloaded Wayback capture.\n"
            f"Original URL: {url}\n"
            f"Timestamp: {timestamp}\n"
            f"Archive URL: {response.url}\n"
            f"Characters: {len(text)}\n"
            f"Saved HTML: {raw_path}\n"
            f"Saved text: {text_path}\n"
            f"Excerpt: {_short(text, 900)}"
        )

    def search_gdelt_news(
        self,
        query: str,
        timespan: str = "7d",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": limit,
            "sort": "datedesc",
            "timespan": str(timespan or "7d").strip() or "7d",
        }
        last_429_body = ""
        try:
            response = None
            for attempt in range(3):
                response = requests.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params=params,
                    headers=_HEADERS,
                    timeout=timeout_seconds,
                )
                if response.status_code != 429:
                    break
                last_429_body = _short(response.text, 300)
                if attempt < 2:
                    time.sleep(6 * (attempt + 1))
            if response is None:
                return "ERROR: GDELT request did not return a response"
            if response.status_code == 429:
                return (
                    "ERROR: GDELT rate limited the request after retries. "
                    f"Upstream message: {last_429_body or 'HTTP 429 Too Many Requests'}"
                )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: GDELT request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to GDELT"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: GDELT returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: GDELT returned invalid JSON"
        artifact = _write_json_artifact("gdelt-news", query, payload)
        articles = payload.get("articles") if isinstance(payload, dict) else []
        if not isinstance(articles, list) or not articles:
            return f"SUCCESS: No GDELT news articles found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: GDELT news articles for '{query}' (top {min(len(articles), limit)}):"]
        for idx, item in enumerate(articles[:limit], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(f"{idx}. {_value(item, 'title') or '(no title)'} - {_value(item, 'url')}")
            meta = []
            for key, label in [
                ("seendate", "seen"),
                ("sourceCountry", "country"),
                ("domain", "domain"),
                ("language", "language"),
            ]:
                text = _value(item, key)
                if text:
                    meta.append(f"{label}: {text}")
            if meta:
                lines.append(f"   {' | '.join(meta)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_crossref(self, query: str, rows: int = 10, from_year: str = "", until_year: str = "", timeout_seconds: int = 20) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        rows = self._max_results(rows)
        params: dict[str, Any] = {"query": query, "rows": rows}
        filters = []
        if str(from_year or "").strip():
            filters.append(f"from-pub-date:{str(from_year).strip()}")
        if str(until_year or "").strip():
            filters.append(f"until-pub-date:{str(until_year).strip()}")
        if filters:
            params["filter"] = ",".join(filters)
        return self._crossref_request("https://api.crossref.org/works", params, f"Crossref works for '{query}'", "crossref-search", query, timeout_seconds)

    def lookup_crossref_doi(self, doi: str, timeout_seconds: int = 20) -> str:
        doi = str(doi or "").strip()
        if not doi:
            return "ERROR: doi cannot be empty"
        url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        return self._crossref_request(url, {}, f"Crossref DOI lookup for '{doi}'", "crossref-doi", doi, timeout_seconds)

    def search_crossref_retractions(self, query: str = "", rows: int = 10, timeout_seconds: int = 20) -> str:
        rows = self._max_results(rows)
        params: dict[str, Any] = {"filter": "update-type:retraction", "rows": rows}
        if str(query or "").strip():
            params["query"] = str(query).strip()
        label = str(query or "all retractions").strip()
        return self._crossref_request("https://api.crossref.org/works", params, f"Crossref Retraction Watch records for '{label}'", "crossref-retractions", label, timeout_seconds)

    def _crossref_request(self, url: str, params: dict[str, Any], heading: str, kind: str, label: str, timeout_seconds: int) -> str:
        try:
            response = requests.get(url, params=params, headers=_HEADERS, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Crossref request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Crossref"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Crossref returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Crossref returned invalid JSON"
        artifact = _write_json_artifact(kind, label, payload)
        message = payload.get("message") if isinstance(payload, dict) else {}
        items = message.get("items") if isinstance(message, dict) else None
        if isinstance(items, list):
            if not items:
                return f"SUCCESS: No {heading} found.\nArtifact JSON: {artifact}"
            lines = [f"SUCCESS: {heading} (top {len(items)}):"]
            for idx, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                title = _value(item, "title") or "(no title)"
                doi = _value(item, "DOI")
                url_value = _value(item, "URL")
                year = ""
                date_parts = (((item.get("published-print") or item.get("published-online") or item.get("created") or {}).get("date-parts") or [[]])[0])
                if date_parts:
                    year = str(date_parts[0])
                lines.append(f"{idx}. {title} - {url_value}")
                meta = [part for part in [f"DOI: {doi}" if doi else "", f"year: {year}" if year else "", _value(item, "publisher")] if part]
                if meta:
                    lines.append(f"   {' | '.join(meta)}")
                abstract = _html_to_text(_value(item, "abstract"))
                if abstract:
                    lines.append(f"   {_short(abstract, 500)}")
                updates = item.get("update-to") or item.get("update-policy") or item.get("relation")
                if updates:
                    lines.append(f"   updates/relation: {_short(json.dumps(updates, ensure_ascii=False, separators=(',', ':')), 500)}")
            lines.append(f"Artifact JSON: {artifact}")
            return "\n".join(lines)
        item = message if isinstance(message, dict) else {}
        if not item:
            return f"SUCCESS: No Crossref data found.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: {heading}:"]
        for key, label_name in [("title", "Title"), ("DOI", "DOI"), ("URL", "URL"), ("publisher", "Publisher"), ("type", "Type")]:
            text = _value(item, key)
            if text:
                lines.append(f"{label_name}: {text}")
        if _value(item, "abstract"):
            lines.append(f"Abstract: {_short(_html_to_text(_value(item, 'abstract')), 900)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_clinical_trials(self, query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params = {"query.term": query, "pageSize": limit, "format": "json"}
        try:
            response = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params, headers=_HEADERS, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: ClinicalTrials.gov request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to ClinicalTrials.gov"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: ClinicalTrials.gov returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: ClinicalTrials.gov returned invalid JSON"
        artifact = _write_json_artifact("clinicaltrials-search", query, payload)
        studies = payload.get("studies") if isinstance(payload, dict) else []
        if not isinstance(studies, list) or not studies:
            return f"SUCCESS: No clinical trials found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: ClinicalTrials.gov studies for '{query}' (top {min(len(studies), limit)}):"]
        for idx, study in enumerate(studies[:limit], start=1):
            proto = study.get("protocolSection") if isinstance(study, dict) else {}
            ident = (proto or {}).get("identificationModule") or {}
            status = (proto or {}).get("statusModule") or {}
            cond = (proto or {}).get("conditionsModule") or {}
            nct = _value(ident, "nctId")
            title = _value(ident, "briefTitle", "officialTitle") or "(no title)"
            lines.append(f"{idx}. {title} - https://clinicaltrials.gov/study/{nct}")
            meta = [part for part in [nct, _value(status, "overallStatus"), _value(status, "startDateStruct"), _value(cond, "conditions")] if part]
            if meta:
                lines.append(f"   {' | '.join(meta)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_clinical_trial(self, nct_id: str, timeout_seconds: int = 20) -> str:
        nct_id = str(nct_id or "").strip().upper()
        if not re.match(r"^NCT\d{8}$", nct_id):
            return "ERROR: nct_id must look like NCT12345678"
        try:
            response = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}", headers=_HEADERS, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: ClinicalTrials.gov request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to ClinicalTrials.gov"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: ClinicalTrials.gov returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: ClinicalTrials.gov returned invalid JSON"
        artifact = _write_json_artifact("clinicaltrials-study", nct_id, payload)
        proto = payload.get("protocolSection") if isinstance(payload, dict) else {}
        ident = (proto or {}).get("identificationModule") or {}
        desc = (proto or {}).get("descriptionModule") or {}
        lines = [f"SUCCESS: ClinicalTrials.gov study {nct_id}:"]
        lines.append(f"Title: {_value(ident, 'briefTitle', 'officialTitle')}")
        if _value(desc, "briefSummary"):
            lines.append(f"Summary: {_short(_value(desc, 'briefSummary'), 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_biorxiv(self, query: str, server: str = "biorxiv", from_date: str = "2026-01-01", to_date: str = "2026-12-31", max_results: Optional[int] = None, timeout_seconds: int = 30) -> str:
        query = str(query or "").strip().lower()
        server = str(server or "biorxiv").strip().lower()
        if server not in {"biorxiv", "medrxiv"}:
            return "ERROR: server must be biorxiv or medrxiv"
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        url = f"https://api.biorxiv.org/details/{server}/{from_date}/{to_date}/0/json"
        try:
            response = requests.get(url, headers=_HEADERS, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: bioRxiv request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to bioRxiv"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: bioRxiv returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: bioRxiv returned invalid JSON"
        collection = payload.get("collection") if isinstance(payload, dict) else []
        matches = []
        if isinstance(collection, list):
            for item in collection:
                haystack = " ".join(
                    str(item.get(key) or "") for key in ("title", "abstract", "authors", "doi")
                ).lower() if isinstance(item, dict) else ""
                if query in haystack:
                    matches.append(item)
                    if len(matches) >= limit:
                        break
        artifact = _write_json_artifact(f"{server}-search", query, {"matches": matches, "source": payload})
        if not matches:
            return f"SUCCESS: No {server} preprints found for '{query}' in {from_date}:{to_date}.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: {server} preprints for '{query}' (top {len(matches)}):"]
        for idx, item in enumerate(matches, start=1):
            doi = _value(item, "doi")
            lines.append(f"{idx}. {_value(item, 'title') or '(no title)'} - https://doi.org/{doi}")
            meta = [part for part in [_value(item, "date"), _value(item, "authors"), doi] if part]
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if _value(item, "abstract"):
                lines.append(f"   {_short(_value(item, 'abstract'), 500)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def download_biorxiv_pdf(self, doi: str, server: str = "biorxiv", timeout_seconds: int = 30) -> str:
        doi = str(doi or "").strip()
        server = str(server or "biorxiv").strip().lower()
        if server not in {"biorxiv", "medrxiv"}:
            return "ERROR: server must be biorxiv or medrxiv"
        if not doi:
            return "ERROR: doi cannot be empty"
        url = f"https://www.{server}.org/content/{doi}.full.pdf"
        try:
            response = requests.get(url, headers=_HEADERS, timeout=timeout_seconds, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return f"ERROR: {server} PDF download timed out"
        except requests.exceptions.ConnectionError:
            return f"ERROR: Failed to connect to {server}"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: {server} PDF returned HTTP {exc.response.status_code}"
        out_dir = _artifact_dir(f"{server}-pdf")
        token = uuid4().hex
        path = os.path.join(out_dir, f"{_safe_filename(doi)}_{token}.pdf")
        text_path = os.path.join(out_dir, f"{_safe_filename(doi)}_{token}.txt")
        with open(path, "wb") as handle:
            handle.write(response.content)
        record_research_artifact(
            path,
            source_url=str(response.url or url),
            provenance=f"{server} PDF download DOI={doi}",
            tool=f"{server}_download",
            kind=f"{server}-pdf",
            label=doi,
        )
        extracted = ""
        try:
            reader = PdfReader(BytesIO(response.content))
            chunks = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    chunks.append(page_text.strip())
            extracted = "\n\n".join(chunks).strip()
        except Exception:
            extracted = ""
        if extracted:
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write(extracted)
            record_research_artifact(
                text_path,
                source_url=str(response.url or url),
                provenance=f"{server} extracted PDF text DOI={doi}",
                tool=f"{server}_download",
                kind=f"{server}-pdf",
                label=doi,
            )
        lines = [
            f"SUCCESS: Downloaded {server} PDF.",
            f"DOI: {doi}",
            f"URL: {response.url}",
            f"Bytes: {len(response.content)}",
            f"Saved PDF: {path}",
        ]
        if extracted:
            lines.append(f"Saved text: {text_path}")
            lines.append(f"Text characters: {len(extracted)}")
        else:
            lines.append("Text extraction: unavailable")
        return "\n".join(lines)

    def search_pubchem(self, query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        try:
            cids_resp = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(query, safe='')}/cids/JSON",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            cids_resp.raise_for_status()
            cids_payload = cids_resp.json()
            cids = (cids_payload.get("IdentifierList") or {}).get("CID") or []
            cids = [str(cid) for cid in cids[:limit]]
            if not cids:
                artifact = _write_json_artifact("pubchem", query, cids_payload)
                return f"SUCCESS: No PubChem compounds found for '{query}'.\nArtifact JSON: {artifact}"
            props_resp = requests.get(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                + ",".join(cids)
                + "/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            props_resp.raise_for_status()
            payload = props_resp.json()
        except requests.exceptions.Timeout:
            return "ERROR: PubChem request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to PubChem"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: PubChem returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: PubChem returned invalid JSON"
        artifact = _write_json_artifact("pubchem", query, payload)
        props = (payload.get("PropertyTable") or {}).get("Properties") or []
        lines = [f"SUCCESS: PubChem compounds for '{query}' (top {min(len(props), limit)}):"]
        for idx, item in enumerate(props[:limit], start=1):
            lines.append(f"{idx}. CID {item.get('CID')} | {item.get('IUPACName') or ''}")
            lines.append(f"   formula: {item.get('MolecularFormula') or ''} | weight: {item.get('MolecularWeight') or ''} | smiles: {item.get('CanonicalSMILES') or ''}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_bible_passage(
        self,
        reference: str,
        translation: str = "kjv",
        timeout_seconds: int = 20,
    ) -> str:
        reference = str(reference or "").strip()
        translation = str(translation or "kjv").strip().lower() or "kjv"
        if not reference:
            return "ERROR: reference cannot be empty"
        try:
            response = requests.get(
                f"https://bible-api.com/{quote(reference)}",
                params={"translation": translation},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Bible API request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Bible API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Bible API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Bible API returned invalid JSON"
        artifact = _write_json_artifact("bible-api", reference, payload)
        text = _boe_text(payload.get("text")) if isinstance(payload, dict) else ""
        if text:
            text_path = _write_text_artifact("bible-api", reference, text)
        else:
            text_path = ""
        lines = [f"SUCCESS: Bible passage for '{reference}':"]
        if isinstance(payload, dict):
            lines.append(f"Reference: {_boe_text(payload.get('reference'))}")
            lines.append(f"Translation: {_boe_text(payload.get('translation_name'))} ({_boe_text(payload.get('translation_id'))})")
            if text:
                lines.append(f"Text: {_short(text, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_sefaria_text(
        self,
        reference: str,
        version: str = "",
        language: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        reference = str(reference or "").strip()
        if not reference:
            return "ERROR: reference cannot be empty"
        params: dict[str, Any] = {}
        if str(version or "").strip():
            params["version"] = str(version).strip()
        if str(language or "").strip():
            params["lang"] = str(language).strip()
        try:
            response = requests.get(
                f"https://www.sefaria.org/api/v3/texts/{quote(reference, safe='')}",
                params=params,
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Sefaria text request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Sefaria"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Sefaria returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Sefaria returned invalid JSON"
        artifact = _write_json_artifact("sefaria-text", reference, payload)
        text_chunks = []
        versions = payload.get("versions") if isinstance(payload, dict) else []
        if isinstance(versions, list):
            for item in versions[:5]:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, list):
                    text = " ".join(_boe_text(part) for part in text)
                text = _html_to_text(_boe_text(text))
                if text:
                    title = _boe_text(item.get("versionTitle") or item.get("shortVersionTitle"))
                    prefix = f"{title}: " if title else ""
                    text_chunks.append(prefix + text)
        plain_text = "\n\n".join(text_chunks).strip()
        text_path = ""
        if plain_text:
            text_path = _write_text_artifact("sefaria-text", reference, plain_text)
        lines = [f"SUCCESS: Sefaria text for '{reference}':"]
        if isinstance(payload, dict):
            lines.append(f"Ref: {_boe_text(payload.get('ref')) or reference}")
            lines.append(f"Categories: {_boe_text(payload.get('categories'))}")
            if plain_text:
                lines.append(f"Text excerpt: {_short(plain_text, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_sefaria(
        self,
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        try:
            response = requests.post(
                "https://www.sefaria.org/api/search-wrapper",
                json={"query": query, "size": limit},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Sefaria search request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Sefaria"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Sefaria search returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Sefaria search returned invalid JSON"
        artifact = _write_json_artifact("sefaria-search", query, payload)
        hits = (((payload.get("hits") or {}).get("hits") or []) if isinstance(payload, dict) else [])[:limit]
        if not hits:
            return f"SUCCESS: No Sefaria search results found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Sefaria search results for '{query}' (top {len(hits)}):"]
        for idx, item in enumerate(hits, start=1):
            if not isinstance(item, dict):
                continue
            ref = _boe_text(item.get("_id"))
            lines.append(f"{idx}. {ref or '(no reference)'}")
            highlight = item.get("highlight") or {}
            if isinstance(highlight, dict):
                snippets = []
                for values in highlight.values():
                    if isinstance(values, list):
                        snippets.extend(_html_to_text(str(value)) for value in values[:3])
                if snippets:
                    lines.append(f"   {_short(' | '.join(snippets), 700)}")
        text_path = _write_text_artifact("sefaria-search", query, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_quran(
        self,
        query: str,
        max_results: Optional[int] = None,
        language: str = "en",
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        try:
            response = requests.get(
                "https://api.quran.com/api/v4/search",
                params={"q": query, "size": limit, "language": str(language or "en").strip() or "en"},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Quran search request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Quran.com API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Quran search returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Quran search returned invalid JSON"
        artifact = _write_json_artifact("quran-search", query, payload)
        results = (((payload.get("search") or {}).get("results") or []) if isinstance(payload, dict) else [])[:limit]
        if not results:
            return f"SUCCESS: No Quran search results found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Quran search results for '{query}' (top {len(results)}):"]
        for idx, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            words = item.get("words") or []
            text = " ".join(_boe_text(word.get("text") if isinstance(word, dict) else word) for word in words) if isinstance(words, list) else _boe_text(item.get("text"))
            lines.append(f"{idx}. {_boe_text(item.get('verse_key'))} | verse_id: {_boe_text(item.get('verse_id'))}")
            if text:
                lines.append(f"   {_short(text, 500)}")
        text_path = _write_text_artifact("quran-search", query, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_quran_verse(
        self,
        verse_key: str,
        translation_ids: str = "131",
        language: str = "en",
        timeout_seconds: int = 20,
    ) -> str:
        verse_key = str(verse_key or "").strip()
        if not re.match(r"^\d+:\d+$", verse_key):
            return "ERROR: verse_key must look like 1:1"
        try:
            response = requests.get(
                f"https://api.quran.com/api/v4/verses/by_key/{quote(verse_key, safe=':')}",
                params={
                    "language": str(language or "en").strip() or "en",
                    "translations": str(translation_ids or "131").strip() or "131",
                    "words": "true",
                    "word_fields": "text_uthmani,text_indopak,translation,transliteration",
                },
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Quran verse request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Quran.com API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Quran verse returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Quran verse returned invalid JSON"
        artifact = _write_json_artifact("quran-verse", verse_key, payload)
        verse = payload.get("verse") if isinstance(payload, dict) else {}
        words = verse.get("words") if isinstance(verse, dict) else []
        text = ""
        if isinstance(words, list):
            text = " ".join(_boe_text(word.get("text_uthmani") if isinstance(word, dict) else word) for word in words)
        translations = verse.get("translations") if isinstance(verse, dict) else []
        translation_text = " ".join(_html_to_text(_boe_text(item.get("text"))) for item in translations if isinstance(item, dict)) if isinstance(translations, list) else ""
        plain_text = "\n".join(part for part in [text, translation_text] if part).strip()
        text_path = ""
        if plain_text:
            text_path = _write_text_artifact("quran-verse", verse_key, plain_text)
        lines = [f"SUCCESS: Quran verse {verse_key}:"]
        if text:
            lines.append(f"Arabic: {_short(text, 800)}")
        if translation_text:
            lines.append(f"Translation: {_short(translation_text, 800)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_quran_chapters(self, language: str = "en", timeout_seconds: int = 20) -> str:
        try:
            response = requests.get(
                "https://api.quran.com/api/v4/chapters",
                params={"language": str(language or "en").strip() or "en"},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Quran chapters request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Quran.com API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Quran chapters returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Quran chapters returned invalid JSON"
        artifact = _write_json_artifact("quran-chapters", str(language or "en"), payload)
        chapters = payload.get("chapters") if isinstance(payload, dict) else []
        lines = [f"SUCCESS: Quran chapters ({len(chapters) if isinstance(chapters, list) else 0} total):"]
        if isinstance(chapters, list):
            for item in chapters[:114]:
                if not isinstance(item, dict):
                    continue
                translated = item.get("translated_name") if isinstance(item.get("translated_name"), dict) else {}
                lines.append(
                    f"{item.get('id')}. {item.get('name_simple')} / {item.get('name_arabic')} | {_boe_text(translated.get('name'))} | verses: {item.get('verses_count')}"
                )
        text_path = _write_text_artifact("quran-chapters", str(language or "en"), "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_gita_chapters(self, timeout_seconds: int = 20) -> str:
        try:
            response = requests.get(
                "https://vedicscriptures.github.io/chapters",
                headers=_HEADERS,
                timeout=timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Bhagavad Gita chapters request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Bhagavad Gita API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Bhagavad Gita API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Bhagavad Gita API returned invalid JSON"
        artifact = _write_json_artifact("gita-chapters", "chapters", payload)
        chapters = payload if isinstance(payload, list) else []
        lines = [f"SUCCESS: Bhagavad Gita chapters ({len(chapters)} total):"]
        for item in chapters[:18]:
            if not isinstance(item, dict):
                continue
            meaning = item.get("meaning") if isinstance(item.get("meaning"), dict) else {}
            lines.append(
                f"{item.get('chapter_number')}. {item.get('translation') or item.get('name')} | {_boe_text(meaning.get('en'))} | verses: {item.get('verses_count')}"
            )
        text_path = _write_text_artifact("gita-chapters", "chapters", "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_gita_chapter(self, chapter: int, timeout_seconds: int = 20) -> str:
        chapter_num = _coerce_int(chapter, 0)
        if chapter_num < 1 or chapter_num > 18:
            return "ERROR: chapter must be between 1 and 18"
        try:
            response = requests.get(
                f"https://vedicscriptures.github.io/chapter/{chapter_num}",
                headers=_HEADERS,
                timeout=timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Bhagavad Gita chapter request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Bhagavad Gita API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Bhagavad Gita API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Bhagavad Gita API returned invalid JSON"
        artifact = _write_json_artifact("gita-chapter", str(chapter_num), payload)
        plain_text = _join_json_text(payload)
        text_path = ""
        if plain_text:
            text_path = _write_text_artifact("gita-chapter", str(chapter_num), plain_text)
        lines = [f"SUCCESS: Bhagavad Gita chapter {chapter_num}:"]
        if isinstance(payload, dict):
            lines.append(f"Title: {_boe_text(payload.get('translation') or payload.get('name'))}")
            lines.append(f"Verses: {_boe_text(payload.get('verses_count'))}")
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            if summary:
                lines.append(f"Summary: {_short(_boe_text(summary.get('en')), 900)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_gita_verse(self, chapter: int, verse: int, timeout_seconds: int = 20) -> str:
        chapter_num = _coerce_int(chapter, 0)
        verse_num = _coerce_int(verse, 0)
        if chapter_num < 1 or chapter_num > 18:
            return "ERROR: chapter must be between 1 and 18"
        if verse_num < 1:
            return "ERROR: verse must be positive"
        try:
            response = requests.get(
                f"https://vedicscriptures.github.io/slok/{chapter_num}/{verse_num}",
                headers=_HEADERS,
                timeout=timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Bhagavad Gita verse request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Bhagavad Gita API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Bhagavad Gita API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Bhagavad Gita API returned invalid JSON"
        label = f"{chapter_num}.{verse_num}"
        artifact = _write_json_artifact("gita-verse", label, payload)
        parts = []
        if isinstance(payload, dict):
            for key in ("slok", "transliteration"):
                text = _boe_text(payload.get(key))
                if text:
                    parts.append(text)
            for key in ("siva", "tej", "purohit", "chinmay", "san", "adi", "gambir", "madhav", "anand", "rams"):
                item = payload.get(key)
                if not isinstance(item, dict):
                    continue
                author = _boe_text(item.get("author")) or key
                translated = _boe_text(item.get("et") or item.get("ht") or item.get("sc"))
                if translated:
                    parts.append(f"{author}: {translated}")
        plain_text = "\n\n".join(parts).strip() or _join_json_text(payload)
        text_path = ""
        if plain_text:
            text_path = _write_text_artifact("gita-verse", label, plain_text)
        lines = [f"SUCCESS: Bhagavad Gita verse {label}:"]
        if isinstance(payload, dict):
            if _boe_text(payload.get("slok")):
                lines.append(f"Sanskrit: {_short(_boe_text(payload.get('slok')), 700)}")
            if _boe_text(payload.get("transliteration")):
                lines.append(f"Transliteration: {_short(_boe_text(payload.get('transliteration')), 700)}")
        if plain_text:
            lines.append(f"Text excerpt: {_short(plain_text, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_hadith_editions(self, timeout_seconds: int = 20) -> str:
        try:
            response = requests.get(
                "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions.json",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Hadith editions request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Hadith API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Hadith API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Hadith API returned invalid JSON"
        artifact = _write_json_artifact("hadith-editions", "editions", payload)
        lines = ["SUCCESS: Hadith editions:"]
        if isinstance(payload, dict):
            for collection, meta in list(payload.items())[:30]:
                name = _boe_text(meta.get("name")) if isinstance(meta, dict) else collection
                editions = meta.get("collection") if isinstance(meta, dict) else []
                names = []
                if isinstance(editions, list):
                    for item in editions[:8]:
                        if isinstance(item, dict):
                            edition = _boe_text(item.get("name"))
                            language = _boe_text(item.get("language"))
                            if edition:
                                names.append(f"{edition} ({language})" if language else edition)
                lines.append(f"- {collection}: {name}; editions: {', '.join(names)}")
        text_path = _write_text_artifact("hadith-editions", "editions", "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_hadith_collection(self, edition: str, max_hadiths: Optional[int] = 20, timeout_seconds: int = 20) -> str:
        edition = str(edition or "").strip().lower()
        if not re.match(r"^[a-z0-9-]+$", edition):
            return "ERROR: edition must look like eng-bukhari"
        limit = _clamp(_coerce_int(max_hadiths, 20), 1, 200)
        try:
            response = requests.get(
                f"https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{quote(edition, safe='')}.json",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Hadith collection request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Hadith API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Hadith API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Hadith API returned invalid JSON"
        artifact = _write_json_artifact("hadith-collection", edition, payload)
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        hadiths = payload.get("hadiths") if isinstance(payload, dict) else []
        lines = [f"SUCCESS: Hadith collection {edition}:"]
        if isinstance(metadata, dict):
            lines.append(f"Name: {_boe_text(metadata.get('name'))}")
            sections = metadata.get("sections")
            if isinstance(sections, dict):
                lines.append(f"Sections: {len(sections)}")
        plain_lines = []
        if isinstance(hadiths, list):
            lines.append(f"Hadiths returned: {len(hadiths)}; showing/saving first {min(len(hadiths), limit)}")
            for item in hadiths[:limit]:
                if not isinstance(item, dict):
                    continue
                num = _boe_text(item.get("hadithnumber") or item.get("arabicnumber"))
                text = _html_to_text(_boe_text(item.get("text")))
                if text:
                    plain_lines.append(f"{num}. {text}" if num else text)
        plain_text = "\n\n".join(plain_lines).strip()
        text_path = ""
        if plain_text:
            text_path = _write_text_artifact("hadith-collection", edition, plain_text)
            lines.append(f"Text excerpt: {_short(plain_text, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_hadith_section(self, edition: str, section: int, timeout_seconds: int = 20) -> str:
        edition = str(edition or "").strip().lower()
        section_num = _coerce_int(section, -1)
        if not re.match(r"^[a-z0-9-]+$", edition):
            return "ERROR: edition must look like eng-bukhari"
        if section_num < 0:
            return "ERROR: section must be zero or a positive integer"
        label = f"{edition}-{section_num}"
        try:
            response = requests.get(
                f"https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{quote(edition, safe='')}/{section_num}.json",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Hadith section request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Hadith API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Hadith API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Hadith API returned invalid JSON"
        artifact = _write_json_artifact("hadith-section", label, payload)
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        hadiths = payload.get("hadiths") if isinstance(payload, dict) else []
        lines = [f"SUCCESS: Hadith section {edition}/{section_num}:"]
        if isinstance(metadata, dict):
            section_meta = metadata.get("section")
            lines.append(f"Name: {_boe_text(metadata.get('name'))}")
            lines.append(f"Section: {_boe_text(section_meta)}")
        plain_lines = []
        if isinstance(hadiths, list):
            lines.append(f"Hadiths: {len(hadiths)}")
            for item in hadiths:
                if not isinstance(item, dict):
                    continue
                num = _boe_text(item.get("hadithnumber") or item.get("arabicnumber"))
                text = _html_to_text(_boe_text(item.get("text")))
                if text:
                    plain_lines.append(f"{num}. {text}" if num else text)
        plain_text = "\n\n".join(plain_lines).strip()
        text_path = ""
        if plain_text:
            text_path = _write_text_artifact("hadith-section", label, plain_text)
            lines.append(f"Text excerpt: {_short(plain_text, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_hadith(self, edition: str, query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        edition = str(edition or "").strip().lower()
        query = str(query or "").strip()
        if not re.match(r"^[a-z0-9-]+$", edition):
            return "ERROR: edition must look like eng-bukhari"
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        terms = [term.lower() for term in re.findall(r"[\w']+", query) if term.strip()]
        if not terms:
            return "ERROR: query must contain searchable terms"
        try:
            response = requests.get(
                f"https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{quote(edition, safe='')}.json",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Hadith search request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Hadith API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Hadith API returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Hadith API returned invalid JSON"
        hadiths = payload.get("hadiths") if isinstance(payload, dict) else []
        matches = []
        if isinstance(hadiths, list):
            for item in hadiths:
                if not isinstance(item, dict):
                    continue
                text = _html_to_text(_boe_text(item.get("text")))
                haystack = text.lower()
                if all(term in haystack for term in terms):
                    matches.append(item)
                if len(matches) >= limit:
                    break
        label = f"{edition}-{query}"
        artifact = _write_json_artifact("hadith-search", label, {"edition": edition, "query": query, "matches": matches})
        if not matches:
            return f"SUCCESS: No hadith matches found for '{query}' in {edition}.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Hadith matches for '{query}' in {edition} (top {len(matches)}):"]
        plain_lines = []
        for idx, item in enumerate(matches, start=1):
            num = _boe_text(item.get("hadithnumber") or item.get("arabicnumber"))
            text = _html_to_text(_boe_text(item.get("text")))
            reference = item.get("reference") if isinstance(item.get("reference"), dict) else {}
            reference_text = _boe_text(reference)
            lines.append(f"{idx}. {num or '(no number)'}")
            if reference_text:
                lines.append(f"   reference: {reference_text}")
            if text:
                lines.append(f"   {_short(text, 600)}")
                plain_lines.append(f"{num}. {text}" if num else text)
        text_path = _write_text_artifact("hadith-search", label, "\n\n".join(plain_lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_suttacentral_suttaplex(self, uid: str, timeout_seconds: int = 20) -> str:
        uid = str(uid or "").strip().lower()
        if not re.match(r"^[a-z0-9.-]+$", uid):
            return "ERROR: uid must look like mn1, sn12.1, an3.65, or dhp1-20"
        try:
            response = requests.get(
                f"https://suttacentral.net/api/suttaplex/{quote(uid, safe='')}",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: SuttaCentral suttaplex request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to SuttaCentral"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: SuttaCentral returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: SuttaCentral returned invalid JSON"
        artifact = _write_json_artifact("suttacentral-suttaplex", uid, payload)
        item = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else {}
        lines = [f"SUCCESS: SuttaCentral metadata for {uid}:"]
        if item:
            lines.append(f"Acronym: {_boe_text(item.get('acronym'))}")
            lines.append(f"Original title: {_boe_text(item.get('original_title'))}")
            lines.append(f"Translated title: {_boe_text(item.get('translated_title'))}")
            lines.append(f"Root language: {_boe_text(item.get('root_lang_name'))}")
            translations = item.get("translations")
            if isinstance(translations, list):
                sample = []
                for trans in translations[:12]:
                    if isinstance(trans, dict):
                        sample.append(
                            f"{_boe_text(trans.get('id'))} [{_boe_text(trans.get('lang'))}, {_boe_text(trans.get('author_uid'))}]"
                        )
                lines.append(f"Translations: {', '.join(part for part in sample if part)}")
        text_path = _write_text_artifact("suttacentral-suttaplex", uid, "\n".join(lines))
        lines.append(f"Artifact JSON: {artifact}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def get_suttacentral_text(
        self,
        uid: str,
        language: str = "en",
        author: str = "sujato",
        root: bool = False,
        timeout_seconds: int = 20,
    ) -> str:
        uid = str(uid or "").strip().lower()
        language = str(language or "en").strip().lower() or "en"
        author = str(author or "sujato").strip().lower() or "sujato"
        if not re.match(r"^[a-z0-9.-]+$", uid):
            return "ERROR: uid must look like mn1, sn12.1, an3.65, or dhp1-20"
        if not re.match(r"^[a-z0-9_-]+$", language) or not re.match(r"^[a-z0-9_-]+$", author):
            return "ERROR: language and author must be simple ids such as en and sujato"
        resolved = _suttacentral_bilara_path(uid)
        if not resolved:
            return "ERROR: could not derive a Bilara path for this uid. Try common ids like dn1, mn1, sn12.1, an3.65, or dhp1-20"
        bilara_path, file_stem = resolved
        if root:
            url = f"https://raw.githubusercontent.com/suttacentral/bilara-data/published/root/pli/ms/{bilara_path}/{file_stem}_root-pli-ms.json"
            label = f"{uid}-root-pli-ms"
        else:
            url = f"https://raw.githubusercontent.com/suttacentral/bilara-data/published/translation/{language}/{author}/{bilara_path}/{file_stem}_translation-{language}-{author}.json"
            label = f"{uid}-translation-{language}-{author}"
        try:
            response = requests.get(url, headers=_HEADERS, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: SuttaCentral Bilara text request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to SuttaCentral Bilara data"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: SuttaCentral Bilara data returned HTTP {exc.response.status_code} for {url}"
        except ValueError:
            return "ERROR: SuttaCentral Bilara data returned invalid JSON"
        artifact = _write_json_artifact("suttacentral-text", label, payload)
        if isinstance(payload, dict):
            plain_text = "\n".join(_html_to_text(str(value)) for value in payload.values() if str(value).strip()).strip()
        else:
            plain_text = _join_json_text(payload)
        text_path = ""
        if plain_text:
            text_path = _write_text_artifact("suttacentral-text", label, plain_text)
        lines = [f"SUCCESS: SuttaCentral Bilara text for {uid}:"]
        lines.append(f"Source URL: {url}")
        lines.append(f"Characters: {len(plain_text)}")
        if plain_text:
            lines.append(f"Text excerpt: {_short(plain_text, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        if text_path:
            lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_mastodon(self, query: str, instance: str = "mastodon.social", result_type: str = "statuses", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        query = str(query or "").strip()
        instance = str(instance or "mastodon.social").strip().removeprefix("https://").removeprefix("http://").strip("/")
        result_type = str(result_type or "statuses").strip().lower()
        if result_type not in {"accounts", "hashtags", "statuses"}:
            return "ERROR: result_type must be accounts, hashtags, or statuses"
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        try:
            response = requests.get(
                f"https://{instance}/api/v2/search",
                params={"q": query, "type": result_type, "limit": limit},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Mastodon request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Mastodon instance"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Mastodon returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Mastodon returned invalid JSON"
        artifact = _write_json_artifact("mastodon", f"{instance}_{query}", payload)
        results = payload.get(result_type) if isinstance(payload, dict) else []
        if not isinstance(results, list) or not results:
            return f"SUCCESS: No Mastodon {result_type} found for '{query}' on {instance}.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Mastodon {result_type} for '{query}' on {instance} (top {min(len(results), limit)}):"]
        for idx, item in enumerate(results[:limit], start=1):
            if result_type == "statuses":
                text = _html_to_text(_value(item, "content"))
                lines.append(f"{idx}. {_short(text, 220)} - {_value(item, 'url', 'uri')}")
                meta = [part for part in [_value(item, "created_at"), _value(item.get("account") if isinstance(item, dict) else {}, "acct", "username")] if part]
            else:
                lines.append(f"{idx}. {_value(item, 'acct', 'name', 'title') or '(result)'} - {_value(item, 'url')}")
                meta = [part for part in [_value(item, "display_name"), _value(item, "name"), _value(item, "history")] if part]
            if meta:
                lines.append(f"   {' | '.join(meta)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_cpsc_recalls(self, query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        try:
            response = requests.get(
                "https://www.saferproducts.gov/RestWebServices/Recall",
                params={"format": "json", "RecallTitle": query},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: CPSC recall request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to CPSC recall API"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: CPSC returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: CPSC returned invalid JSON"
        artifact = _write_json_artifact("cpsc-recalls", query, payload)
        rows = payload if isinstance(payload, list) else []
        if not rows:
            return f"SUCCESS: No CPSC recalls found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: CPSC recalls for '{query}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            lines.append(f"{idx}. {_short(_value(item, 'Title', 'Name', 'RecallNumber'), 180)} - {_value(item, 'URL')}")
            meta = [part for part in [_value(item, "RecallID"), _value(item, "RecallNumber"), _value(item, "RecallDate")] if part]
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if _value(item, "Description"):
                lines.append(f"   {_short(_value(item, 'Description'), 500)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_cisa_kev(self, query: str = "", cve_id: str = "", timeout_seconds: int = 20) -> str:
        query = str(query or "").strip().lower()
        cve_id = str(cve_id or "").strip().upper()
        try:
            response = requests.get(
                "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: CISA KEV request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to CISA KEV"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: CISA KEV returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: CISA KEV returned invalid JSON"
        vulns = payload.get("vulnerabilities") if isinstance(payload, dict) else []
        matches = []
        for item in vulns or []:
            haystack = json.dumps(item, ensure_ascii=False).lower() if isinstance(item, dict) else ""
            if (cve_id and _value(item, "cveID").upper() == cve_id) or (query and query in haystack) or (not query and not cve_id):
                matches.append(item)
            if len(matches) >= 20:
                break
        artifact = _write_json_artifact("cisa-kev", query or cve_id or "catalog", {"matches": matches, "source_date": payload.get("dateReleased") if isinstance(payload, dict) else ""})
        if not matches:
            return f"SUCCESS: No CISA KEV entries matched.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: CISA KEV matches (top {len(matches)}):"]
        for idx, item in enumerate(matches, start=1):
            lines.append(f"{idx}. {_value(item, 'cveID')} | {_value(item, 'vendorProject')} | {_value(item, 'product')}")
            lines.append(f"   added: {_value(item, 'dateAdded')} | due: {_value(item, 'dueDate')} | action: {_short(_value(item, 'requiredAction'), 260)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def query_osv_package(self, package: str, ecosystem: str, version: str = "", timeout_seconds: int = 20) -> str:
        package = str(package or "").strip()
        ecosystem = str(ecosystem or "").strip()
        if not package or not ecosystem:
            return "ERROR: package and ecosystem are required"
        body: dict[str, Any] = {"package": {"name": package, "ecosystem": ecosystem}}
        if str(version or "").strip():
            body["version"] = str(version).strip()
        try:
            response = requests.post("https://api.osv.dev/v1/query", json=body, headers=_HEADERS, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: OSV request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to OSV"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: OSV returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: OSV returned invalid JSON"
        artifact = _write_json_artifact("osv", package, payload)
        vulns = payload.get("vulns") if isinstance(payload, dict) else []
        if not isinstance(vulns, list) or not vulns:
            return f"SUCCESS: No OSV vulnerabilities found for {ecosystem}/{package}.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: OSV vulnerabilities for {ecosystem}/{package} (top {min(len(vulns), 20)}):"]
        for idx, item in enumerate(vulns[:20], start=1):
            lines.append(f"{idx}. {_value(item, 'id')} | {_short(_value(item, 'summary'), 180)}")
            lines.append(f"   modified: {_value(item, 'modified')} | published: {_value(item, 'published')}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_federal_register(self, query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        try:
            response = requests.get(
                "https://www.federalregister.gov/api/v1/documents.json",
                params={"conditions[term]": query, "per_page": limit, "order": "newest"},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Federal Register request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Federal Register"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Federal Register returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Federal Register returned invalid JSON"
        artifact = _write_json_artifact("federal-register", query, payload)
        results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(results, list) or not results:
            return f"SUCCESS: No Federal Register documents found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Federal Register documents for '{query}' (top {min(len(results), limit)}):"]
        for idx, item in enumerate(results[:limit], start=1):
            lines.append(f"{idx}. {_value(item, 'title') or '(no title)'} - {_value(item, 'html_url', 'pdf_url')}")
            meta = [part for part in [_value(item, "publication_date"), _value(item, "type"), _value(item, "agency_names")] if part]
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if _value(item, "abstract"):
                lines.append(f"   {_short(_value(item, 'abstract'), 500)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_boe_laws(
        self,
        query: str,
        field: str = "all",
        max_results: Optional[int] = None,
        publication_from: str = "",
        publication_to: str = "",
        sort_by: str = "",
        sort_order: str = "desc",
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        field = str(field or "all").strip().lower()
        if field not in {"all", "title", "text", "raw"}:
            return "ERROR: field must be one of all, title, text, raw"
        limit = self._max_results(max_results)
        if field == "raw":
            query_string = query
        else:
            escaped = query.replace('"', '\\"')
            if field == "title":
                query_string = f'titulo:"{escaped}"'
            elif field == "text":
                query_string = f'texto:"{escaped}"'
            else:
                query_string = f'(titulo:"{escaped}" or texto:"{escaped}")'
        range_query: dict[str, Any] = {}
        pub_range = {}
        if str(publication_from or "").strip():
            pub_range["gte"] = str(publication_from).strip()
        if str(publication_to or "").strip():
            pub_range["lte"] = str(publication_to).strip()
        if pub_range:
            range_query["fecha_publicacion"] = pub_range
        sort = []
        sort_by = str(sort_by or "").strip()
        if sort_by:
            if sort_by not in {
                "fecha_publicacion",
                "fecha_disposicion",
                "fecha_actualizacion",
                "titulo",
                "departamento",
                "rango",
            }:
                return "ERROR: sort_by must be empty or one of fecha_publicacion, fecha_disposicion, fecha_actualizacion, titulo, departamento, rango"
            order = str(sort_order or "desc").strip().lower()
            if order not in {"asc", "desc"}:
                return "ERROR: sort_order must be asc or desc"
            sort.append({sort_by: order})
        boe_query = {
            "query": {
                "query_string": {"query": query_string},
                "range": range_query,
            },
            "sort": sort,
        }
        try:
            response = requests.get(
                "https://www.boe.es/datosabiertos/api/legislacion-consolidada",
                params={
                    "query": json.dumps(boe_query, ensure_ascii=False, separators=(",", ":")),
                    "limit": limit,
                },
                headers={**_HEADERS, "Accept": "application/json"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: BOE legislation search request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to BOE OpenData"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: BOE legislation search returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: BOE legislation search returned invalid JSON"
        artifact = _write_json_artifact("boe-legislation-search", query, payload)
        status = payload.get("status") if isinstance(payload, dict) else {}
        if _boe_text((status or {}).get("code")) not in {"", "200"}:
            return f"ERROR: BOE returned status {_boe_text(status.get('code'))}: {_boe_text(status.get('text'))}\nArtifact JSON: {artifact}"
        rows = payload.get("data") if isinstance(payload, dict) else []
        if not isinstance(rows, list) or not rows:
            return f"SUCCESS: No BOE consolidated-law results found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: BOE consolidated-law results for '{query}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            boe_id = _boe_text(item.get("identificador")) if isinstance(item, dict) else ""
            title = _boe_text(item.get("titulo")) if isinstance(item, dict) else ""
            lines.append(f"{idx}. {title or '(no title)'}")
            meta = []
            for key, label in [
                ("identificador", "id"),
                ("ambito", "ambito"),
                ("departamento", "departamento"),
                ("rango", "rango"),
                ("fecha_publicacion", "publicacion"),
                ("fecha_vigencia", "vigencia"),
                ("estado_consolidacion", "estado"),
            ]:
                text = _boe_text(item.get(key)) if isinstance(item, dict) else ""
                if text:
                    meta.append(f"{label}: {text}")
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            url = _boe_text(item.get("url_html_consolidada")) if isinstance(item, dict) else ""
            if not url and boe_id:
                url = f"https://www.boe.es/buscar/act.php?id={boe_id}"
            if url:
                lines.append(f"   URL: {url}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_boe_law_metadata(
        self,
        boe_id: str,
        part: str = "metadatos",
        timeout_seconds: int = 20,
    ) -> str:
        boe_id = str(boe_id or "").strip().upper()
        part = str(part or "metadatos").strip().lower()
        if not re.match(r"^BOE-[A-Z]-\d{4}-\d+$", boe_id):
            return "ERROR: boe_id must look like BOE-A-2018-16673"
        if part not in {"metadatos", "analisis", "metadata-eli", "indice"}:
            return "ERROR: part must be one of metadatos, analisis, metadata-eli, indice"
        suffix = "texto/indice" if part == "indice" else part
        try:
            response = requests.get(
                f"https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/{boe_id}/{suffix}",
                headers={**_HEADERS, "Accept": "application/json"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: BOE metadata request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to BOE OpenData"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: BOE metadata returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: BOE metadata returned invalid JSON"
        artifact = _write_json_artifact(f"boe-law-{part}", boe_id, payload)
        status = payload.get("status") if isinstance(payload, dict) else {}
        if _boe_text((status or {}).get("code")) not in {"", "200"}:
            return f"ERROR: BOE returned status {_boe_text(status.get('code'))}: {_boe_text(status.get('text'))}\nArtifact JSON: {artifact}"
        data = payload.get("data") if isinstance(payload, dict) else []
        item = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else data if isinstance(data, dict) else {}
        lines = [f"SUCCESS: BOE {part} for {boe_id}:"]
        if isinstance(item, dict):
            for key, label in [
                ("identificador", "ID"),
                ("titulo", "Title"),
                ("ambito", "Ambito"),
                ("departamento", "Departamento"),
                ("rango", "Rango"),
                ("fecha_disposicion", "Fecha disposicion"),
                ("fecha_publicacion", "Fecha publicacion"),
                ("fecha_vigencia", "Fecha vigencia"),
                ("estado_consolidacion", "Estado consolidacion"),
                ("url_html_consolidada", "URL HTML consolidada"),
                ("url_xml", "URL XML"),
                ("url_eli", "URL ELI"),
            ]:
                text = _boe_text(item.get(key))
                if text:
                    lines.append(f"{label}: {text}")
            extra = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if len(extra) > 1200:
                lines.append(f"Data excerpt: {_short(extra, 1200)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def download_boe_law_text(self, boe_id: str, timeout_seconds: int = 30) -> str:
        boe_id = str(boe_id or "").strip().upper()
        if not re.match(r"^BOE-[A-Z]-\d{4}-\d+$", boe_id):
            return "ERROR: boe_id must look like BOE-A-2018-16673"
        try:
            response = requests.get(
                f"https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/{boe_id}/texto",
                headers={**_HEADERS, "Accept": "application/xml"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: BOE law text request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to BOE OpenData"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: BOE law text returned HTTP {exc.response.status_code}"
        xml_text = response.text or ""
        plain_text = _xml_to_text(xml_text)
        if len(plain_text) < 100:
            return "ERROR: BOE law text response did not contain enough extractable text"
        raw_path, text_path = _write_text_artifacts("boe-law-text", boe_id, "xml", xml_text, plain_text)
        return (
            "SUCCESS: Downloaded BOE consolidated law text.\n"
            f"BOE ID: {boe_id}\n"
            f"URL: {response.url}\n"
            f"Characters: {len(plain_text)}\n"
            f"Saved XML: {raw_path}\n"
            f"Saved text: {text_path}\n"
            f"Excerpt: {_short(plain_text, 900)}"
        )

    def get_boe_aux_table(
        self,
        table: str,
        timeout_seconds: int = 20,
    ) -> str:
        table = str(table or "").strip().lower()
        allowed = {
            "materias",
            "ambitos",
            "estados-consolidacion",
            "departamentos",
            "rangos",
        }
        if table not in allowed:
            return "ERROR: table must be one of materias, ambitos, estados-consolidacion, departamentos, rangos"
        try:
            response = requests.get(
                f"https://www.boe.es/datosabiertos/api/datos-auxiliares/{table}",
                headers={**_HEADERS, "Accept": "application/json"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: BOE auxiliary-table request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to BOE OpenData"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: BOE auxiliary-table request returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: BOE auxiliary-table request returned invalid JSON"
        artifact = _write_json_artifact("boe-auxiliary", table, payload)
        data = payload.get("data") if isinstance(payload, dict) else {}
        lines = [f"SUCCESS: BOE auxiliary table '{table}':"]
        if isinstance(data, dict):
            for idx, (key, value) in enumerate(list(data.items())[:30], start=1):
                lines.append(f"{idx}. {key}: {_boe_text(value)}")
        else:
            lines.append(_short(json.dumps(data, ensure_ascii=False, separators=(",", ":")), 1200))
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_world_bank_indicator(self, indicator: str, country: str = "all", per_page: int = 20, timeout_seconds: int = 20) -> str:
        indicator = str(indicator or "").strip()
        country = str(country or "all").strip() or "all"
        if not indicator:
            return "ERROR: indicator cannot be empty"
        per_page = self._max_results(per_page)
        try:
            response = requests.get(
                f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}",
                params={"format": "json", "per_page": per_page, "mrv": per_page},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: World Bank request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to World Bank"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: World Bank returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: World Bank returned invalid JSON"
        artifact = _write_json_artifact("world-bank", f"{country}_{indicator}", payload)
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        if not isinstance(rows, list) or not rows:
            return f"SUCCESS: No World Bank data found for {country}/{indicator}.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: World Bank indicator {indicator} for {country} (top {min(len(rows), per_page)}):"]
        for idx, item in enumerate(rows[:per_page], start=1):
            country_name = _value(item.get("country") if isinstance(item, dict) else {}, "value")
            lines.append(f"{idx}. {country_name} | {item.get('date')}: {item.get('value')}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_wikidata_entities(self, query: str, language: str = "en", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        try:
            response = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={"action": "wbsearchentities", "search": query, "language": language, "format": "json", "limit": limit},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Wikidata request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Wikidata"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Wikidata returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Wikidata returned invalid JSON"
        artifact = _write_json_artifact("wikidata-search", query, payload)
        rows = payload.get("search") if isinstance(payload, dict) else []
        if not isinstance(rows, list) or not rows:
            return f"SUCCESS: No Wikidata entities found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Wikidata entities for '{query}' (top {min(len(rows), limit)}):"]
        for idx, item in enumerate(rows[:limit], start=1):
            lines.append(f"{idx}. {_value(item, 'label')} ({_value(item, 'id')}) - https://www.wikidata.org/wiki/{_value(item, 'id')}")
            if _value(item, "description"):
                lines.append(f"   {_value(item, 'description')}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def query_wikidata_sparql(self, sparql: str, timeout_seconds: int = 30) -> str:
        sparql = str(sparql or "").strip()
        if not sparql:
            return "ERROR: sparql cannot be empty"
        try:
            response = requests.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql, "format": "json"},
                headers=_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Wikidata SPARQL request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Wikidata SPARQL"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Wikidata SPARQL returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Wikidata SPARQL returned invalid JSON"
        artifact = _write_json_artifact("wikidata-sparql", "query", payload)
        bindings = ((payload.get("results") or {}).get("bindings") or []) if isinstance(payload, dict) else []
        lines = [f"SUCCESS: Wikidata SPARQL returned {len(bindings)} rows.", f"Artifact JSON: {artifact}"]
        for idx, row in enumerate(bindings[:20], start=1):
            if not isinstance(row, dict):
                continue
            values = []
            for key, cell in row.items():
                if isinstance(cell, dict) and cell.get("value"):
                    values.append(f"{key}: {cell['value']}")
            if values:
                lines.append(f"{idx}. {' | '.join(values)}")
        return "\n".join(lines)


def _require_sdk():
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")


def get_fetch_url_text_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="fetch_url_text")
    def fetch_url_text(url: str, timeout_seconds: int = 30) -> str:
        """Fetch a URL, extract readable text, and save raw/text artifacts."""
        return _run_logged("fetch_url_text", {"url": url, "timeout_seconds": timeout_seconds}, lambda: helper.fetch_url_text(url, timeout_seconds))

    return _set_param_descriptions(fetch_url_text, {
        "url": "Absolute HTTP/HTTPS URL to fetch, extract into readable page text, and save as raw/text artifacts.",
        "timeout_seconds": "Maximum seconds to wait for this page download and text extraction.",
    })


def get_web_archive_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="web_archive_search")
    def web_archive_search(url: str, from_year: str = "", to_year: str = "", match_type: str = "prefix", max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search Internet Archive Wayback CDX captures for a URL/domain."""
        return _run_logged("web_archive_search", locals(), lambda: helper.search_wayback_cdx(url, from_year, to_year, match_type, max_results, timeout_seconds))

    return _set_param_descriptions(web_archive_search, {
        "url": "URL, URL prefix, or domain to search in the Internet Archive CDX index.",
        "from_year": "Optional first capture year to include, formatted as YYYY.",
        "to_year": "Optional last capture year to include, formatted as YYYY.",
        "match_type": "Wayback CDX match type for the url value, usually exact, prefix, host, or domain.",
        "max_results": "Maximum number of captures to return from the archive search.",
        "timeout_seconds": "Maximum seconds to wait for the CDX archive search request.",
    })


def get_wayback_fetch_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="wayback_fetch")
    def wayback_fetch(url: str, timestamp: str = "", timeout_seconds: int = 30) -> str:
        """Download a Wayback capture and save raw HTML plus extracted text."""
        return _run_logged("wayback_fetch", locals(), lambda: helper.fetch_wayback_capture(url, timestamp, timeout_seconds))

    return _set_param_descriptions(wayback_fetch, {
        "url": "Original URL or Wayback URL to fetch from the Internet Archive.",
        "timestamp": "Optional Wayback capture timestamp such as 20240102123456; latest available capture is used when empty.",
        "timeout_seconds": "Maximum seconds to wait for downloading and extracting the archived page.",
    })


def get_gdelt_news_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="gdelt_news_search")
    def gdelt_news_search(query: str, timespan: str = "7d", max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search GDELT DOC news articles and save compact JSON evidence."""
        return _run_logged("gdelt_news_search", locals(), lambda: helper.search_gdelt_news(query, timespan, max_results, timeout_seconds))

    return _set_param_descriptions(gdelt_news_search, {
        "query": "GDELT DOC query for news articles, using GDELT's keyword/operator syntax where useful.",
        "timespan": "GDELT time window such as 1d, 7d, 30d, or 3m.",
        "max_results": "Maximum number of GDELT article records to return.",
        "timeout_seconds": "Maximum seconds to wait for the GDELT request.",
    })


def get_crossref_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="crossref_search")
    def crossref_search(query: str, rows: int = 10, from_year: str = "", until_year: str = "", timeout_seconds: int = 20) -> str:
        """Search Crossref works metadata."""
        return _run_logged("crossref_search", locals(), lambda: helper.search_crossref(query, rows, from_year, until_year, timeout_seconds))

    return _set_param_descriptions(crossref_search, {
        "query": "Crossref works search query, typically title keywords, author names, DOI fragments, or topic terms.",
        "rows": "Maximum number of Crossref work records to return.",
        "from_year": "Optional earliest publication year to include, formatted as YYYY.",
        "until_year": "Optional latest publication year to include, formatted as YYYY.",
        "timeout_seconds": "Maximum seconds to wait for the Crossref works search.",
    })


def get_crossref_doi_lookup_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="crossref_doi_lookup")
    def crossref_doi_lookup(doi: str, timeout_seconds: int = 20) -> str:
        """Lookup Crossref metadata for one DOI."""
        return _run_logged("crossref_doi_lookup", locals(), lambda: helper.lookup_crossref_doi(doi, timeout_seconds))

    return _set_param_descriptions(crossref_doi_lookup, {
        "doi": "Exact DOI to look up in Crossref, with or without a https://doi.org/ prefix.",
        "timeout_seconds": "Maximum seconds to wait for the Crossref DOI lookup.",
    })


def get_retraction_watch_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="retraction_watch")
    def retraction_watch(query: str = "", rows: int = 10, timeout_seconds: int = 20) -> str:
        """Search Crossref Retraction Watch retraction update records."""
        return _run_logged("retraction_watch", locals(), lambda: helper.search_crossref_retractions(query, rows, timeout_seconds))

    return _set_param_descriptions(retraction_watch, {
        "query": "Optional title, author, DOI, journal, institution, or topic text to filter retraction records.",
        "rows": "Maximum number of Retraction Watch update records to return.",
        "timeout_seconds": "Maximum seconds to wait for the retraction search.",
    })


def get_clinicaltrials_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="clinicaltrials_search")
    def clinicaltrials_search(query: str, max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search ClinicalTrials.gov v2 studies."""
        return _run_logged("clinicaltrials_search", locals(), lambda: helper.search_clinical_trials(query, max_results, timeout_seconds))

    return _set_param_descriptions(clinicaltrials_search, {
        "query": "ClinicalTrials.gov query expression for condition, intervention, sponsor, title, or NCT terms.",
        "max_results": "Maximum number of matching clinical trial summaries to return.",
        "timeout_seconds": "Maximum seconds to wait for the ClinicalTrials.gov search.",
    })


def get_clinicaltrial_get_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="clinicaltrial_get")
    def clinicaltrial_get(nct_id: str, timeout_seconds: int = 20) -> str:
        """Fetch a ClinicalTrials.gov v2 study by NCT id."""
        return _run_logged("clinicaltrial_get", locals(), lambda: helper.get_clinical_trial(nct_id, timeout_seconds))

    return _set_param_descriptions(clinicaltrial_get, {
        "nct_id": "ClinicalTrials.gov NCT identifier for one study, such as NCT01234567.",
        "timeout_seconds": "Maximum seconds to wait for the ClinicalTrials.gov study fetch.",
    })


def get_biorxiv_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="biorxiv_search")
    def biorxiv_search(query: str, server: str = "biorxiv", from_date: str = "2026-01-01", to_date: str = "2026-12-31", max_results: int = 10, timeout_seconds: int = 30) -> str:
        """Search bioRxiv/medRxiv details API over a date interval."""
        return _run_logged("biorxiv_search", locals(), lambda: helper.search_biorxiv(query, server, from_date, to_date, max_results, timeout_seconds))

    return _set_param_descriptions(biorxiv_search, {
        "query": "Keyword text to match against bioRxiv/medRxiv titles, abstracts, authors, categories, or DOI metadata.",
        "server": "Preprint server to search: biorxiv or medrxiv.",
        "from_date": "Start date for the details API interval, formatted as YYYY-MM-DD.",
        "to_date": "End date for the details API interval, formatted as YYYY-MM-DD.",
        "max_results": "Maximum number of matching preprint records to return.",
        "timeout_seconds": "Maximum seconds to wait for the preprint search request.",
    })


def get_biorxiv_download_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="biorxiv_download")
    def biorxiv_download(doi: str, server: str = "biorxiv", timeout_seconds: int = 30) -> str:
        """Download a bioRxiv/medRxiv PDF by DOI."""
        return _run_logged("biorxiv_download", locals(), lambda: helper.download_biorxiv_pdf(doi, server, timeout_seconds))

    return _set_param_descriptions(biorxiv_download, {
        "doi": "bioRxiv or medRxiv DOI to download as a PDF.",
        "server": "Preprint server hosting the DOI: biorxiv or medrxiv.",
        "timeout_seconds": "Maximum seconds to wait for the preprint PDF download and text extraction.",
    })


def get_pubchem_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="pubchem_search")
    def pubchem_search(query: str, max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search PubChem compounds by name and return key properties."""
        return _run_logged("pubchem_search", locals(), lambda: helper.search_pubchem(query, max_results, timeout_seconds))

    return _set_param_descriptions(pubchem_search, {
        "query": "Compound name, synonym, CID, formula, or chemical term to search in PubChem.",
        "max_results": "Maximum number of PubChem compound records to return.",
        "timeout_seconds": "Maximum seconds to wait for the PubChem requests.",
    })


def get_bible_passage_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="bible_passage_get")
    def bible_passage_get(reference: str, translation: str = "kjv", timeout_seconds: int = 20) -> str:
        """Retrieve a Bible passage by reference from bible-api.com and save JSON/text evidence."""
        return _run_logged("bible_passage_get", locals(), lambda: helper.get_bible_passage(reference, translation, timeout_seconds))

    return _set_param_descriptions(bible_passage_get, {
        "reference": "Bible passage reference such as John 3:16, Genesis 1:1-5, or Romans 8.",
        "translation": "Bible translation/version accepted by bible-api.com, such as kjv.",
        "timeout_seconds": "Maximum seconds to wait for the Bible passage request.",
    })


def get_sefaria_text_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="sefaria_text_get")
    def sefaria_text_get(reference: str, version: str = "", language: str = "", timeout_seconds: int = 20) -> str:
        """Retrieve a Jewish text reference from Sefaria, including Torah/Tanakh/Talmud/commentary where available."""
        return _run_logged("sefaria_text_get", locals(), lambda: helper.get_sefaria_text(reference, version, language, timeout_seconds))

    return _set_param_descriptions(sefaria_text_get, {
        "reference": "Sefaria text reference such as Genesis 1:1, Exodus 20, Mishnah Berakhot 1:1, or Rashi on Genesis 1:1.",
        "version": "Optional exact Sefaria version title to request; leave empty for Sefaria's default.",
        "language": "Optional text language code such as en or he; leave empty for available defaults.",
        "timeout_seconds": "Maximum seconds to wait for the Sefaria text request.",
    })


def get_sefaria_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="sefaria_search")
    def sefaria_search(query: str, max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search Sefaria Jewish texts and save JSON/text evidence."""
        return _run_logged("sefaria_search", locals(), lambda: helper.search_sefaria(query, max_results, timeout_seconds))

    return _set_param_descriptions(sefaria_search, {
        "query": "Search phrase for Sefaria Jewish texts, commentaries, titles, or references.",
        "max_results": "Maximum number of Sefaria search hits to return.",
        "timeout_seconds": "Maximum seconds to wait for the Sefaria search request.",
    })


def get_quran_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="quran_search")
    def quran_search(query: str, max_results: int = 10, language: str = "en", timeout_seconds: int = 20) -> str:
        """Search Quran.com/Quran Foundation content API and save JSON/text evidence."""
        return _run_logged("quran_search", locals(), lambda: helper.search_quran(query, max_results, language, timeout_seconds))

    return _set_param_descriptions(quran_search, {
        "query": "Word or phrase to search in Quran.com/Quran Foundation text and translation content.",
        "max_results": "Maximum number of Quran search hits to return.",
        "language": "Language code for Quran API labels/translations where supported, such as en or ar.",
        "timeout_seconds": "Maximum seconds to wait for the Quran search request.",
    })


def get_quran_verse_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="quran_verse_get")
    def quran_verse_get(verse_key: str, translation_ids: str = "131", language: str = "en", timeout_seconds: int = 20) -> str:
        """Retrieve a Quran verse by key such as 1:1 and save JSON/text evidence."""
        return _run_logged("quran_verse_get", locals(), lambda: helper.get_quran_verse(verse_key, translation_ids, language, timeout_seconds))

    return _set_param_descriptions(quran_verse_get, {
        "verse_key": "Quran verse key in surah:ayah form, such as 1:1 or 2:255.",
        "translation_ids": "Comma-separated Quran.com translation IDs to include, such as 131.",
        "language": "Language code for Quran API labels where supported, such as en or ar.",
        "timeout_seconds": "Maximum seconds to wait for the Quran verse request.",
    })


def get_quran_chapters_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="quran_chapters_get")
    def quran_chapters_get(language: str = "en", timeout_seconds: int = 20) -> str:
        """Fetch Quran chapter metadata and save JSON/text evidence."""
        return _run_logged("quran_chapters_get", locals(), lambda: helper.get_quran_chapters(language, timeout_seconds))

    return _set_param_descriptions(quran_chapters_get, {
        "language": "Language code for Quran chapter names/metadata where supported, such as en or ar.",
        "timeout_seconds": "Maximum seconds to wait for the Quran chapters request.",
    })


def get_gita_chapters_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="gita_chapters_get")
    def gita_chapters_get(timeout_seconds: int = 20) -> str:
        """Fetch Bhagavad Gita chapter metadata and save JSON/text evidence."""
        return _run_logged("gita_chapters_get", locals(), lambda: helper.get_gita_chapters(timeout_seconds))

    return _set_param_descriptions(gita_chapters_get, {
        "timeout_seconds": "Maximum seconds to wait for the Bhagavad Gita chapters request.",
    })


def get_gita_chapter_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="gita_chapter_get")
    def gita_chapter_get(chapter: int, timeout_seconds: int = 20) -> str:
        """Fetch Bhagavad Gita chapter metadata/summary by chapter number and save evidence."""
        return _run_logged("gita_chapter_get", locals(), lambda: helper.get_gita_chapter(chapter, timeout_seconds))

    return _set_param_descriptions(gita_chapter_get, {
        "chapter": "Bhagavad Gita chapter number to fetch.",
        "timeout_seconds": "Maximum seconds to wait for the Bhagavad Gita chapter request.",
    })


def get_gita_verse_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="gita_verse_get")
    def gita_verse_get(chapter: int, verse: int, timeout_seconds: int = 20) -> str:
        """Fetch a Bhagavad Gita verse with Sanskrit, transliteration, translations, and commentaries."""
        return _run_logged("gita_verse_get", locals(), lambda: helper.get_gita_verse(chapter, verse, timeout_seconds))

    return _set_param_descriptions(gita_verse_get, {
        "chapter": "Bhagavad Gita chapter number containing the verse.",
        "verse": "Verse number within the selected Bhagavad Gita chapter.",
        "timeout_seconds": "Maximum seconds to wait for the Bhagavad Gita verse request.",
    })


def get_hadith_editions_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="hadith_editions_get")
    def hadith_editions_get(timeout_seconds: int = 20) -> str:
        """List available no-key Hadith API editions and save JSON/text evidence."""
        return _run_logged("hadith_editions_get", locals(), lambda: helper.get_hadith_editions(timeout_seconds))

    return _set_param_descriptions(hadith_editions_get, {
        "timeout_seconds": "Maximum seconds to wait for the Hadith editions list request.",
    })


def get_hadith_collection_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="hadith_collection_get")
    def hadith_collection_get(edition: str, max_hadiths: int = 20, timeout_seconds: int = 20) -> str:
        """Fetch a hadith edition such as eng-bukhari and save a compact evidence excerpt plus full JSON."""
        return _run_logged("hadith_collection_get", locals(), lambda: helper.get_hadith_collection(edition, max_hadiths, timeout_seconds))

    return _set_param_descriptions(hadith_collection_get, {
        "edition": "Hadith API edition identifier to fetch, such as eng-bukhari or ara-muslim.",
        "max_hadiths": "Maximum number of hadith records from the edition to include in the returned excerpt.",
        "timeout_seconds": "Maximum seconds to wait for the Hadith collection request.",
    })


def get_hadith_section_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="hadith_section_get")
    def hadith_section_get(edition: str, section: int, timeout_seconds: int = 20) -> str:
        """Fetch one section/book from a hadith edition such as eng-bukhari/1 and save JSON/text evidence."""
        return _run_logged("hadith_section_get", locals(), lambda: helper.get_hadith_section(edition, section, timeout_seconds))

    return _set_param_descriptions(hadith_section_get, {
        "edition": "Hadith API edition identifier containing the section, such as eng-bukhari.",
        "section": "Section/book number inside the selected Hadith edition.",
        "timeout_seconds": "Maximum seconds to wait for the Hadith section request.",
    })


def get_hadith_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="hadith_search")
    def hadith_search(edition: str, query: str, max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search a hadith edition such as eng-bukhari locally after downloading it and save matching records/text."""
        return _run_logged("hadith_search", locals(), lambda: helper.search_hadith(edition, query, max_results, timeout_seconds))

    return _set_param_descriptions(hadith_search, {
        "edition": "Hadith API edition identifier to download and search, such as eng-bukhari.",
        "query": "Word or phrase to match inside hadith text and metadata for the selected edition.",
        "max_results": "Maximum number of matching hadith records to return.",
        "timeout_seconds": "Maximum seconds to wait for downloading/searching the Hadith edition.",
    })


def get_suttacentral_suttaplex_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="suttacentral_suttaplex_get")
    def suttacentral_suttaplex_get(uid: str, timeout_seconds: int = 20) -> str:
        """Fetch SuttaCentral metadata/translations for a text id such as mn1, sn12.1, an3.65, or dhp1-20."""
        return _run_logged("suttacentral_suttaplex_get", locals(), lambda: helper.get_suttacentral_suttaplex(uid, timeout_seconds))

    return _set_param_descriptions(suttacentral_suttaplex_get, {
        "uid": "SuttaCentral text identifier such as mn1, sn12.1, an3.65, or dhp1-20.",
        "timeout_seconds": "Maximum seconds to wait for the SuttaCentral metadata request.",
    })


def get_suttacentral_text_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="suttacentral_text_get")
    def suttacentral_text_get(uid: str, language: str = "en", author: str = "sujato", root: bool = False, timeout_seconds: int = 20) -> str:
        """Fetch full segmented SuttaCentral/Bilara text for common sutta ids and save JSON/text evidence."""
        return _run_logged("suttacentral_text_get", locals(), lambda: helper.get_suttacentral_text(uid, language, author, root, timeout_seconds))

    return _set_param_descriptions(suttacentral_text_get, {
        "uid": "SuttaCentral/Bilara text identifier such as mn1, sn12.1, an3.65, or dhp1-20.",
        "language": "Bilara translation language code to fetch, commonly en.",
        "author": "Bilara translation author slug to fetch, such as sujato.",
        "root": "When true, fetch root/source text instead of a translation.",
        "timeout_seconds": "Maximum seconds to wait for the SuttaCentral/Bilara text request.",
    })


def get_mastodon_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="mastodon_search")
    def mastodon_search(query: str, instance: str = "mastodon.social", result_type: str = "statuses", max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search a Mastodon instance's public API for statuses, accounts, or hashtags."""
        return _run_logged("mastodon_search", locals(), lambda: helper.search_mastodon(query, instance, result_type, max_results, timeout_seconds))

    return _set_param_descriptions(mastodon_search, {
        "query": "Search query for the selected Mastodon instance public API.",
        "instance": "Mastodon server hostname to query, such as mastodon.social.",
        "result_type": "Result category to return from Mastodon search: statuses, accounts, or hashtags.",
        "max_results": "Maximum number of Mastodon results to return.",
        "timeout_seconds": "Maximum seconds to wait for the Mastodon instance request.",
    })


def get_cpsc_recalls_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="cpsc_recalls_search")
    def cpsc_recalls_search(query: str, max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search CPSC/SaferProducts consumer-product recalls."""
        return _run_logged("cpsc_recalls_search", locals(), lambda: helper.search_cpsc_recalls(query, max_results, timeout_seconds))

    return _set_param_descriptions(cpsc_recalls_search, {
        "query": "Product, brand, company, hazard, or recall keyword to search in CPSC recall records.",
        "max_results": "Maximum number of CPSC recall records to return.",
        "timeout_seconds": "Maximum seconds to wait for the CPSC recall search.",
    })


def get_cisa_kev_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="cisa_kev_search")
    def cisa_kev_search(query: str = "", cve_id: str = "", timeout_seconds: int = 20) -> str:
        """Search the CISA Known Exploited Vulnerabilities catalog."""
        return _run_logged("cisa_kev_search", locals(), lambda: helper.search_cisa_kev(query, cve_id, timeout_seconds))

    return _set_param_descriptions(cisa_kev_search, {
        "query": "Optional vendor, product, vulnerability name, notes, or keyword to match in the CISA KEV catalog.",
        "cve_id": "Optional exact CVE identifier to find, such as CVE-2024-12345.",
        "timeout_seconds": "Maximum seconds to wait for the CISA KEV catalog request.",
    })


def get_osv_package_query_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="osv_package_query")
    def osv_package_query(package: str, ecosystem: str, version: str = "", timeout_seconds: int = 20) -> str:
        """Query OSV vulnerabilities for a package ecosystem/name/version."""
        return _run_logged("osv_package_query", locals(), lambda: helper.query_osv_package(package, ecosystem, version, timeout_seconds))

    return _set_param_descriptions(osv_package_query, {
        "package": "Package name to query in OSV, such as requests, lodash, or openssl.",
        "ecosystem": "OSV ecosystem for the package, such as PyPI, npm, Maven, Go, crates.io, or Debian.",
        "version": "Optional package version to narrow vulnerabilities; leave empty for all known versions.",
        "timeout_seconds": "Maximum seconds to wait for the OSV package query.",
    })


def get_federal_register_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="federal_register_search")
    def federal_register_search(query: str, max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search FederalRegister.gov documents."""
        return _run_logged("federal_register_search", locals(), lambda: helper.search_federal_register(query, max_results, timeout_seconds))

    return _set_param_descriptions(federal_register_search, {
        "query": "Full-text search query for FederalRegister.gov documents.",
        "max_results": "Maximum number of Federal Register document records to return.",
        "timeout_seconds": "Maximum seconds to wait for the Federal Register search.",
    })


def get_boe_law_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="boe_law_search")
    def boe_law_search(
        query: str,
        field: str = "all",
        max_results: int = 10,
        publication_from: str = "",
        publication_to: str = "",
        sort_by: str = "",
        sort_order: str = "desc",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Spanish BOE consolidated legislation by title/text/raw BOE query."""
        return _run_logged(
            "boe_law_search",
            locals(),
            lambda: helper.search_boe_laws(
                query,
                field,
                max_results,
                publication_from,
                publication_to,
                sort_by,
                sort_order,
                timeout_seconds,
            ),
        )

    return _set_param_descriptions(boe_law_search, {
        "query": "Spanish BOE legislation search text, title words, free-text legal terms, or raw BOE query.",
        "field": "Search field to use: all, title, text, or raw depending on the BOE query strategy.",
        "max_results": "Maximum number of BOE consolidated-law records to return.",
        "publication_from": "Optional earliest BOE publication date in YYYY-MM-DD format.",
        "publication_to": "Optional latest BOE publication date in YYYY-MM-DD format.",
        "sort_by": "Optional BOE sort field when supported by the endpoint.",
        "sort_order": "BOE sort direction, usually asc or desc.",
        "timeout_seconds": "Maximum seconds to wait for the BOE legislation search.",
    })


def get_boe_law_metadata_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="boe_law_metadata_get")
    def boe_law_metadata_get(boe_id: str, part: str = "metadatos", timeout_seconds: int = 20) -> str:
        """Get Spanish BOE consolidated-law metadata, analysis, ELI metadata, or index."""
        return _run_logged("boe_law_metadata_get", locals(), lambda: helper.get_boe_law_metadata(boe_id, part, timeout_seconds))

    return _set_param_descriptions(boe_law_metadata_get, {
        "boe_id": "BOE consolidated legislation identifier returned by boe_law_search, such as BOE-A-1978-31229.",
        "part": "BOE metadata part to retrieve, such as metadatos, analisis, eli, or indice.",
        "timeout_seconds": "Maximum seconds to wait for the BOE metadata request.",
    })


def get_boe_law_text_download_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="boe_law_text_download")
    def boe_law_text_download(boe_id: str, timeout_seconds: int = 30) -> str:
        """Download Spanish BOE consolidated-law full text as XML and extracted text."""
        return _run_logged("boe_law_text_download", locals(), lambda: helper.download_boe_law_text(boe_id, timeout_seconds))

    return _set_param_descriptions(boe_law_text_download, {
        "boe_id": "BOE consolidated legislation identifier to download as XML/text, such as BOE-A-1978-31229.",
        "timeout_seconds": "Maximum seconds to wait for the BOE full-text download and extraction.",
    })


def get_boe_aux_table_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="boe_aux_table_get")
    def boe_aux_table_get(table: str, timeout_seconds: int = 20) -> str:
        """Fetch Spanish BOE auxiliary lookup tables such as materias, rangos, departamentos."""
        return _run_logged("boe_aux_table_get", locals(), lambda: helper.get_boe_aux_table(table, timeout_seconds))

    return _set_param_descriptions(boe_aux_table_get, {
        "table": "BOE auxiliary table name to fetch, such as materias, rangos, departamentos, or diarios.",
        "timeout_seconds": "Maximum seconds to wait for the BOE auxiliary table request.",
    })


def get_world_bank_indicator_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="world_bank_indicator")
    def world_bank_indicator(indicator: str, country: str = "all", per_page: int = 20, timeout_seconds: int = 20) -> str:
        """Fetch recent World Bank indicator values."""
        return _run_logged("world_bank_indicator", locals(), lambda: helper.search_world_bank_indicator(indicator, country, per_page, timeout_seconds))

    return _set_param_descriptions(world_bank_indicator, {
        "indicator": "World Bank indicator code such as NY.GDP.MKTP.CD or SP.POP.TOTL.",
        "country": "ISO country code such as US or ES, or all for all countries.",
        "per_page": "Maximum number of World Bank indicator values to return.",
        "timeout_seconds": "Maximum seconds to wait for the World Bank request.",
    })


def get_wikidata_entity_search_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="wikidata_entity_search")
    def wikidata_entity_search(query: str, language: str = "en", max_results: int = 10, timeout_seconds: int = 20) -> str:
        """Search Wikidata entities by label/alias."""
        return _run_logged("wikidata_entity_search", locals(), lambda: helper.search_wikidata_entities(query, language, max_results, timeout_seconds))

    return _set_param_descriptions(wikidata_entity_search, {
        "query": "Entity label, alias, or term to search in Wikidata.",
        "language": "Wikidata label/search language code, such as en or es.",
        "max_results": "Maximum number of Wikidata entity matches to return.",
        "timeout_seconds": "Maximum seconds to wait for the Wikidata entity search.",
    })


def get_wikidata_sparql_tool(helper: OpenResearchTool):
    _require_sdk()

    @function_tool(name_override="wikidata_sparql")
    def wikidata_sparql(sparql: str, timeout_seconds: int = 30) -> str:
        """Run a Wikidata SPARQL query and save compact JSON evidence."""
        return _run_logged("wikidata_sparql", locals(), lambda: helper.query_wikidata_sparql(sparql, timeout_seconds))

    return _set_param_descriptions(wikidata_sparql, {
        "sparql": "Complete Wikidata SPARQL query to execute against query.wikidata.org.",
        "timeout_seconds": "Maximum seconds to wait for the Wikidata SPARQL query.",
    })
