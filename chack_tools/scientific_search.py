import json
import os
import re
import time
from datetime import date, timedelta
from html import unescape
from typing import Any, Optional
from urllib.parse import quote
from uuid import uuid4
import xml.etree.ElementTree as ET

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests
from .config import ToolsConfig
from .research_artifacts import record_research_artifact, research_artifacts_root
from .serpapi_keys import (
    is_serpapi_rate_limited,
    note_serpapi_response_error,
    usable_serpapi_keys,
)
from .telemetry import run_with_tool_logging


def _run_logged(tool: str, tool_input: dict, func):
    try:
        return run_with_tool_logging(tool, tool_input, func)
    except Exception as exc:
        return f"ERROR: {tool} failed ({exc})"


def _with_scientific_output(tool: Any):
    current = str(getattr(tool, "description", "") or "").strip()
    if current and "Output:" not in current:
        tool.description = (
            f"{current}\n\n"
            "Parameters: Use the schema descriptions to provide scientific queries, identifiers, date ranges, result limits, locale/filter options, and request timeouts.\n"
            "Output: Returns compact SUCCESS/ERROR text with paper/book/patent/video metadata, links, abstracts/snippets, citation or transcript data, and pagination IDs when available. "
            "Download tools also report local raw/text/PDF/XML artifact paths."
        )
    return tool


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short(text: str, max_chars: int = 200) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _plain_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(part for part in (_plain_text(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("content", "text", "description", "snippet", "title", "name", "value"):
            text = _plain_text(value.get(key))
            if text:
                return text
        return " ".join(
            part for part in (_plain_text(item) for item in value.values()) if part
        )
    return str(value)


def _artifact_dir(kind: str) -> str:
    root = research_artifacts_root()
    base = os.path.join(root, kind) if root else os.path.join("/tmp", "chack-scientific", kind)
    os.makedirs(base, exist_ok=True)
    return base


def _safe_filename(value: str, fallback: str = "document") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _xml_to_text(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except Exception:
        return _clean_text(re.sub(r"<[^>]+>", " ", xml_text))
    chunks = []
    for node in root.iter():
        if node.text and node.text.strip():
            chunks.append(node.text.strip())
        if node.tail and node.tail.strip():
            chunks.append(node.tail.strip())
    return _clean_text(" ".join(chunks))


def _html_to_text(html_text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return _clean_text(unescape(text))


def _write_artifacts(kind: str, label: str, raw_ext: str, raw_text: str, plain_text: str) -> tuple[str, str]:
    output_dir = _artifact_dir(kind)
    token = uuid4().hex
    base = _safe_filename(label)
    raw_path = os.path.join(output_dir, f"{base}_{token}.{raw_ext}")
    text_path = os.path.join(output_dir, f"{base}_{token}.txt")
    with open(raw_path, "w", encoding="utf-8") as handle:
        handle.write(raw_text)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(plain_text)
    record_research_artifact(raw_path, provenance=f"{kind}:{label}", kind=kind, label=label)
    record_research_artifact(text_path, provenance=f"{kind}:{label}", kind=kind, label=label)
    return raw_path, text_path


def _normalize_pmc_id(value: str) -> str:
    raw = str(value or "").strip()
    match = re.search(r"PMC(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    digits = re.sub(r"\D+", "", raw)
    return digits


def _normalize_youtube_video_id(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    patterns = [
        r"(?:v=|/shorts/|/embed/|youtu\.be/)([A-Za-z0-9_-]{6,})",
        r"^([A-Za-z0-9_-]{6,})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return match.group(1)
    return raw


_NCBI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    )
}


def _get_with_retries(url: str, *, retries: int = 3, **kwargs):
    response = None
    for attempt in range(max(1, retries)):
        try:
            response = requests.get(url, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                time.sleep(1.0 + attempt)
                continue
            raise
        if response.status_code not in {429, 500, 502, 503, 504}:
            return response
        if attempt < retries - 1:
            time.sleep(1.0 + attempt)
    return response


class ScientificSearchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _max_results(self, requested: Optional[int], default_limit: int = 10) -> int:
        cfg_limit = _coerce_int(getattr(self.config, "scientific_max_results", default_limit), default_limit)
        cfg_limit = _clamp(cfg_limit, 1, 50)
        if requested is None:
            return cfg_limit
        return _clamp(_coerce_int(requested, cfg_limit), 1, 50)

    def _serpapi_key(self) -> str:
        keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        return keys[0] if keys else ""

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
                note_serpapi_response_error(api_key, response.status_code, body)
                if is_serpapi_rate_limited(response.status_code, body) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI returned HTTP {response.status_code}"

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

    @staticmethod
    def _format_results(source: str, query: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return f"SUCCESS: No {source} full-text results found for '{query}'."
        lines = [f"SUCCESS: {source} full-text results for '{query}' (top {len(rows)}):"]
        for idx, row in enumerate(rows, start=1):
            title = row.get("title") or "(no title)"
            url = row.get("url") or ""
            meta_parts = []
            if row.get("year"):
                meta_parts.append(f"year: {row['year']}")
            if row.get("source"):
                meta_parts.append(f"source: {row['source']}")
            if row.get("authors"):
                meta_parts.append(f"authors: {row['authors']}")
            lines.append(f"{idx}. {title} - {url}")
            if meta_parts:
                lines.append(f"   {' | '.join(meta_parts)}")
            if row.get("snippet"):
                lines.append(f"   {_short(str(row['snippet']))}")
        return "\n".join(lines)

    @staticmethod
    def _is_pdf_url_accessible(url: str, timeout_seconds: int) -> bool:
        if not url:
            return False
        try:
            response = requests.get(url, timeout=timeout_seconds, allow_redirects=True)
        except requests.RequestException:
            return False
        if response.status_code >= 400:
            return False
        ctype = str(response.headers.get("content-type") or "").lower()
        if "pdf" in ctype:
            return True
        final_url = str(response.url or "").lower()
        return final_url.endswith(".pdf")

    def _search_arxiv_html_fallback(self, query: str, limit: int, timeout_seconds: int, reason: str) -> str:
        params = {
            "query": query,
            "searchtype": "all",
            "abstracts": "show",
            "order": "-announced_date_first",
            "size": max(25, limit),
        }
        try:
            response = _get_with_retries(
                "https://arxiv.org/search/",
                params=params,
                headers=_NCBI_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return f"ERROR: arXiv API failed ({reason}) and arXiv HTML fallback timed out"
        except requests.exceptions.ConnectionError:
            return f"ERROR: arXiv API failed ({reason}) and arXiv HTML fallback could not connect"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: arXiv API failed ({reason}) and arXiv HTML fallback returned HTTP {exc.response.status_code}"
        html = response.text or ""
        blocks = re.findall(r'<li class="arxiv-result">(.*?)</li>', html, flags=re.DOTALL)
        rows = []
        for block in blocks:
            abs_match = re.search(r'https://arxiv\.org/abs/([0-9][0-9A-Za-z._/-]+)', block)
            pdf_match = re.search(r'https://arxiv\.org/pdf/([0-9][0-9A-Za-z._/-]+)', block)
            arxiv_id = (pdf_match or abs_match).group(1) if (pdf_match or abs_match) else ""
            if not arxiv_id:
                continue
            title_match = re.search(r'<p class="title[^"]*">\s*(.*?)\s*</p>', block, flags=re.DOTALL)
            authors_match = re.search(r'<p class="authors">\s*(.*?)\s*</p>', block, flags=re.DOTALL)
            abstract_match = re.search(r'<span class="abstract-full[^"]*"[^>]*>\s*(.*?)\s*<a ', block, flags=re.DOTALL)
            if not abstract_match:
                abstract_match = re.search(r'<span class="abstract-short[^"]*"[^>]*>\s*(.*?)\s*<a ', block, flags=re.DOTALL)
            submitted_match = re.search(r"<span[^>]*>Submitted</span>\s*([^;<]+)", block, flags=re.DOTALL)
            year = ""
            if submitted_match:
                year_match = re.search(r"(19|20)\d{2}", submitted_match.group(1))
                year = year_match.group(0) if year_match else ""
            rows.append(
                {
                    "title": _html_to_text(title_match.group(1)) if title_match else f"arXiv:{arxiv_id}",
                    "url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "year": year,
                    "source": "arXiv HTML fallback",
                    "authors": _html_to_text(authors_match.group(1)).replace("Authors:", "").strip() if authors_match else "",
                    "snippet": _html_to_text(abstract_match.group(1)) if abstract_match else "",
                }
            )
            if len(rows) >= limit:
                break
        if rows:
            return self._format_results("arXiv", query, rows)
        return f"ERROR: arXiv API failed ({reason}) and arXiv HTML fallback found no PDF results"

    def search_arxiv(
        self,
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        limit = self._max_results(max_results)
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
        }
        try:
            response = _get_with_retries(
                "https://export.arxiv.org/api/query",
                params=params,
                headers=_NCBI_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return self._search_arxiv_html_fallback(query, limit, timeout_seconds, "API timed out")
        except requests.exceptions.ConnectionError:
            return self._search_arxiv_html_fallback(query, limit, timeout_seconds, "API connection failed")
        except requests.exceptions.HTTPError as exc:
            return self._search_arxiv_html_fallback(query, limit, timeout_seconds, f"API HTTP {exc.response.status_code}")

        atom = response.text
        entries = re.findall(r"<entry>(.*?)</entry>", atom, flags=re.DOTALL)
        rows = []
        for entry in entries:
            title_match = re.search(r"<title>(.*?)</title>", entry, flags=re.DOTALL)
            title = (title_match.group(1).strip() if title_match else "arXiv paper").replace("\n", " ")
            pdf_match = re.search(r'<link[^>]+href="([^"]+)"[^>]+type="application/pdf"', entry)
            if not pdf_match:
                pdf_match = re.search(r'<link[^>]+href="([^"]+)"[^>]+title="pdf"', entry)
            pdf_url = pdf_match.group(1) if pdf_match else ""
            if not pdf_url:
                continue
            if not pdf_url.endswith(".pdf"):
                pdf_url = f"{pdf_url}.pdf"
            summary_match = re.search(r"<summary>(.*?)</summary>", entry, flags=re.DOTALL)
            published_match = re.search(r"<published>(.*?)</published>", entry, flags=re.DOTALL)
            year = ""
            if published_match:
                year = str(published_match.group(1)).strip()[:4]
            rows.append(
                {
                    "title": title,
                    "url": pdf_url,
                    "year": year,
                    "source": "arXiv",
                    "snippet": (summary_match.group(1).strip() if summary_match else ""),
                }
            )
        return self._format_results("arXiv", query, rows[:limit])

    def search_europe_pmc(
        self,
        query: str,
        page: int = 1,
        page_size: int = 25,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        page = max(1, _coerce_int(page, 1))
        page_size = _clamp(_coerce_int(page_size, self._max_results(None)), 1, 50)
        params = {
            "query": query,
            "page": page,
            "pageSize": page_size,
            "format": "json",
        }
        try:
            response = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Europe PMC request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Europe PMC"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Europe PMC returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Europe PMC returned invalid JSON"

        rows = []
        items = (payload.get("resultList") or {}).get("result") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            pmcid = str(item.get("pmcid") or "").strip()
            has_pdf = str(item.get("hasPDF") or "").upper() == "Y"
            is_oa = str(item.get("isOpenAccess") or "").upper() == "Y"
            if not (pmcid and has_pdf and is_oa):
                continue
            rows.append(
                {
                    "title": item.get("title") or "Europe PMC paper",
                    "url": f"https://europepmc.org/articles/{pmcid}?pdf=render",
                    "year": item.get("pubYear") or "",
                    "source": item.get("journalTitle") or "Europe PMC",
                    "authors": item.get("authorString") or "",
                    "snippet": "",
                }
            )
        limit = self._max_results(page_size, default_limit=page_size)
        return self._format_results("Europe PMC", query, rows[:limit])

    def search_pmc_full_text(
        self,
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        limit = self._max_results(max_results)
        try:
            search_response = _get_with_retries(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pmc", "term": query, "retmode": "json", "retmax": limit},
                headers=_NCBI_HEADERS,
                timeout=timeout_seconds,
            )
            search_response.raise_for_status()
            search_payload = search_response.json()
        except requests.exceptions.Timeout:
            return "ERROR: PMC request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to PMC"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: PMC returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: PMC returned invalid JSON"

        ids = ((search_payload.get("esearchresult") or {}).get("idlist") or [])[:limit]
        if not ids:
            return f"SUCCESS: No PMC full-text results found for '{query}'."
        try:
            summary_response = _get_with_retries(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pmc", "id": ",".join(ids), "retmode": "json"},
                headers=_NCBI_HEADERS,
                timeout=timeout_seconds,
            )
            summary_response.raise_for_status()
            summary_payload = summary_response.json()
        except requests.exceptions.Timeout:
            return "ERROR: PMC summary request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to PMC summary"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: PMC summary returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: PMC summary returned invalid JSON"

        result = summary_payload.get("result") or {}
        lines = [f"SUCCESS: PMC full-text results for '{query}' (top {len(ids)}):"]
        for idx, pmc_id in enumerate(ids, start=1):
            item = result.get(str(pmc_id)) or {}
            pmcid = ""
            for article_id in item.get("articleids") or []:
                if isinstance(article_id, dict) and str(article_id.get("idtype") or "").lower() == "pmcid":
                    pmcid = str(article_id.get("value") or "").strip()
                    break
            pmcid = pmcid or f"PMC{pmc_id}"
            title = item.get("title") or "PMC article"
            journal = item.get("fulljournalname") or item.get("source") or "PMC"
            pubdate = item.get("pubdate") or item.get("epubdate") or ""
            article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
            xml_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pmc&id={pmc_id}&retmode=xml"
            )
            lines.append(f"{idx}. {title} - {article_url}")
            lines.append(f"   PMCID: {pmcid} | source: {journal}" + (f" | date: {pubdate}" if pubdate else ""))
            lines.append(f"   Full-text XML: {xml_url}")
        return "\n".join(lines)

    def download_pmc_full_text(
        self,
        pmcid_or_id: str,
        timeout_seconds: int = 30,
    ) -> str:
        pmc_id = _normalize_pmc_id(pmcid_or_id)
        if not pmc_id:
            return "ERROR: pmcid_or_id must contain a PMC identifier or numeric PMC id"
        try:
            response = _get_with_retries(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pmc", "id": pmc_id, "retmode": "xml"},
                headers=_NCBI_HEADERS,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: PMC full-text download timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect while downloading PMC full text"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: PMC full-text download returned HTTP {exc.response.status_code}"
        xml_text = response.text or ""
        plain_text = _xml_to_text(xml_text)
        if len(plain_text) < 200:
            return "ERROR: PMC full-text response did not contain enough extractable text"
        raw_path, text_path = _write_artifacts("pmc-full-text", f"PMC{pmc_id}", "xml", xml_text, plain_text)
        return (
            "SUCCESS: Downloaded PMC full text.\n"
            f"PMCID/ID: {pmcid_or_id}\n"
            f"Characters: {len(plain_text)}\n"
            f"Saved XML: {raw_path}\n"
            f"Saved text: {text_path}"
        )

    def search_ncbi_bookshelf(
        self,
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        limit = self._max_results(max_results)
        try:
            search_response = _get_with_retries(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "books", "term": query, "retmode": "json", "retmax": limit},
                headers=_NCBI_HEADERS,
                timeout=timeout_seconds,
            )
            search_response.raise_for_status()
            search_payload = search_response.json()
        except requests.exceptions.Timeout:
            return "ERROR: NCBI Bookshelf search timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to NCBI Bookshelf"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: NCBI Bookshelf returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: NCBI Bookshelf returned invalid JSON"
        ids = ((search_payload.get("esearchresult") or {}).get("idlist") or [])[:limit]
        if not ids:
            return f"SUCCESS: No NCBI Bookshelf results found for '{query}'."
        try:
            summary_response = _get_with_retries(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "books", "id": ",".join(ids), "retmode": "json"},
                headers=_NCBI_HEADERS,
                timeout=timeout_seconds,
            )
            summary_response.raise_for_status()
            summary_payload = summary_response.json()
        except requests.exceptions.Timeout:
            return "ERROR: NCBI Bookshelf summary timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to NCBI Bookshelf summary"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: NCBI Bookshelf summary returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: NCBI Bookshelf summary returned invalid JSON"

        result = summary_payload.get("result") or {}
        lines = [f"SUCCESS: NCBI Bookshelf full-content results for '{query}' (top {len(ids)}):"]
        for idx, book_id in enumerate(ids, start=1):
            item = result.get(str(book_id)) or {}
            accession = str(item.get("accessionid") or item.get("bookaccessionid") or "").strip()
            title = item.get("title") or "NCBI Bookshelf item"
            record_type = item.get("rtype") or ""
            pubdate = item.get("pubdate") or ""
            reader_url = f"https://www.ncbi.nlm.nih.gov/books/{accession}/?report=reader" if accession else ""
            lines.append(f"{idx}. {title}" + (f" - {reader_url}" if reader_url else ""))
            meta = [f"NCBI Books ID: {book_id}"]
            if accession:
                meta.append(f"accession: {accession}")
            if record_type:
                meta.append(f"type: {record_type}")
            if pubdate:
                meta.append(f"date: {pubdate}")
            lines.append(f"   {' | '.join(meta)}")
        return "\n".join(lines)

    def download_ncbi_bookshelf(
        self,
        accession_or_id: str,
        timeout_seconds: int = 30,
    ) -> str:
        raw = str(accession_or_id or "").strip()
        if not raw:
            return "ERROR: accession_or_id cannot be empty"
        accession_match = re.search(r"NBK\d+", raw, flags=re.IGNORECASE)
        accession = accession_match.group(0).upper() if accession_match else ""
        if not accession:
            numeric_id = re.sub(r"\D+", "", raw)
            if not numeric_id:
                return "ERROR: accession_or_id must contain an NBK accession or numeric Bookshelf id"
            try:
                summary_response = _get_with_retries(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                    params={"db": "books", "id": numeric_id, "retmode": "json"},
                    headers=_NCBI_HEADERS,
                    timeout=timeout_seconds,
                )
                summary_response.raise_for_status()
                item = (summary_response.json().get("result") or {}).get(numeric_id) or {}
                accession = str(item.get("accessionid") or item.get("bookaccessionid") or "").strip()
            except Exception:
                accession = ""
        if not accession:
            return "ERROR: Could not resolve NCBI Bookshelf accession"
        reader_url = f"https://www.ncbi.nlm.nih.gov/books/{accession}/?report=reader"
        try:
            response = _get_with_retries(reader_url, headers=_NCBI_HEADERS, timeout=timeout_seconds)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: NCBI Bookshelf download timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect while downloading NCBI Bookshelf content"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: NCBI Bookshelf download returned HTTP {exc.response.status_code}"
        html_text = response.text or ""
        plain_text = _html_to_text(html_text)
        if len(plain_text) < 200:
            return "ERROR: NCBI Bookshelf response did not contain enough extractable text"
        raw_path, text_path = _write_artifacts("ncbi-bookshelf", accession, "html", html_text, plain_text)
        return (
            "SUCCESS: Downloaded NCBI Bookshelf full content.\n"
            f"Accession: {accession}\n"
            f"URL: {reader_url}\n"
            f"Characters: {len(plain_text)}\n"
            f"Saved HTML: {raw_path}\n"
            f"Saved text: {text_path}"
        )

    def search_semantic_scholar(
        self,
        query: str,
        limit: int = 20,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        limit = _clamp(_coerce_int(limit, self._max_results(None)), 1, 20)
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,openAccessPdf,url",
        }
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            headers = dict(_NCBI_HEADERS)
            api_key = (
                os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
                or os.environ.get("S2_API_KEY", "").strip()
            )
            if api_key:
                headers["x-api-key"] = api_key
            response = requests.get(url, params=params, headers=headers, timeout=timeout_seconds)
            retries = 0
            while response.status_code == 429 and retries < 3:
                retry_after = _coerce_int(response.headers.get("Retry-After"), 2 + retries * 2)
                time.sleep(max(1, min(retry_after, 10)))
                response = requests.get(url, params=params, headers=headers, timeout=timeout_seconds)
                retries += 1
            if response.status_code == 429:
                return (
                    "ERROR: Semantic Scholar rate limited the request. "
                    "Set SEMANTIC_SCHOLAR_API_KEY for a free higher-quota API key."
                )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Semantic Scholar request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Semantic Scholar"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Semantic Scholar returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Semantic Scholar returned invalid JSON"

        rows = []
        for item in payload.get("data", []) or []:
            if not isinstance(item, dict):
                continue
            pdf_url = ((item.get("openAccessPdf") or {}).get("url") or "").strip()
            if not pdf_url:
                continue
            if not self._is_pdf_url_accessible(pdf_url, timeout_seconds=min(timeout_seconds, 12)):
                continue
            authors = ", ".join(
                [a.get("name", "") for a in (item.get("authors") or []) if isinstance(a, dict) and a.get("name")]
            )
            rows.append(
                {
                    "title": item.get("title") or "Semantic Scholar paper",
                    "url": pdf_url,
                    "year": item.get("year") or "",
                    "source": "Semantic Scholar",
                    "authors": authors,
                    "snippet": item.get("abstract") or "",
                }
            )
        return self._format_results("Semantic Scholar", query, rows[: self._max_results(limit, 20)])

    def search_openalex(
        self,
        query: str,
        page: int = 1,
        per_page: int = 10,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        page = max(1, _coerce_int(page, 1))
        per_page = _clamp(_coerce_int(per_page, self._max_results(None)), 1, 25)
        params = {"search": query, "page": page, "per_page": per_page}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            )
        }
        try:
            response = requests.get("https://api.openalex.org/works", params=params, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: OpenAlex request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to OpenAlex"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: OpenAlex returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: OpenAlex returned invalid JSON"

        rows = []
        for work in payload.get("results", []) or []:
            if not isinstance(work, dict):
                continue
            best_loc = work.get("best_oa_location") or {}
            pdf_url = str(best_loc.get("pdf_url") or "").strip()
            if not pdf_url:
                for loc in work.get("locations", []) or []:
                    if isinstance(loc, dict) and loc.get("pdf_url"):
                        pdf_url = str(loc["pdf_url"]).strip()
                        break
            if not pdf_url:
                continue
            if not self._is_pdf_url_accessible(pdf_url, timeout_seconds=min(timeout_seconds, 12)):
                continue
            year = work.get("publication_year") or work.get("year") or ""
            rows.append(
                {
                    "title": work.get("title") or work.get("display_name") or "OpenAlex paper",
                    "url": pdf_url,
                    "year": year,
                    "source": "OpenAlex",
                    "snippet": "",
                }
            )
        return self._format_results("OpenAlex", query, rows[: self._max_results(per_page, 25)])

    def search_plos(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        rows = _clamp(_coerce_int(rows, self._max_results(None)), 1, 50)
        start = max(0, _coerce_int(start, 0))
        params = {"q": query, "rows": rows, "start": start}
        headers = {"User-Agent": "chack/1.0"}
        try:
            response = requests.get("https://api.plos.org/search", params=params, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: PLOS request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to PLOS"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: PLOS returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: PLOS returned invalid JSON"

        docs = (payload.get("response") or {}).get("docs") or []
        rows_out = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            doi = str(doc.get("id") or doc.get("doi") or "").strip()
            if not doi.startswith("10.1371/"):
                continue
            rows_out.append(
                {
                    "title": doc.get("title_display") or doc.get("title") or "PLOS paper",
                    "url": f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable",
                    "year": doc.get("publication_date") or "",
                    "source": "PLOS",
                    "authors": ", ".join(doc.get("author_display") or []),
                    "snippet": " ".join(doc.get("abstract") or []),
                }
            )
        return self._format_results("PLOS", query, rows_out[: self._max_results(rows, 50)])

    def search_google_patents(
        self,
        query: str,
        page: int = 1,
        num: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        page = max(1, _coerce_int(page, 1))
        limit = self._max_results(num, default_limit=10)
        # SerpAPI google_patents requires num in [10, 100].
        serp_num = max(10, min(100, _coerce_int(num, max(10, limit))))
        payload = self._serpapi_request(
            {
                "engine": "google_patents",
                "q": query,
                "page": page,
                "num": serp_num,
            },
            timeout_seconds=timeout_seconds,
        )
        if isinstance(payload, str):
            return payload
        items = payload.get("organic_results") or []
        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            link = str(
                item.get("link")
                or item.get("patent_link")
                or item.get("serpapi_link")
                or ""
            ).strip()
            if not link:
                patent_id = str(item.get("patent_id") or "").strip()
                if patent_id:
                    link = f"https://patents.google.com/{patent_id}"
            if not link:
                continue
            date_hint = (
                item.get("grant_date")
                or item.get("publication_date")
                or item.get("filing_date")
                or ""
            )
            rows.append(
                {
                    "title": item.get("title") or "Google Patents result",
                    "url": link,
                    "year": str(date_hint)[:4] if date_hint else "",
                    "source": "Google Patents",
                    "authors": item.get("assignee") or item.get("inventor") or "",
                    "snippet": item.get("snippet") or item.get("abstract") or "",
                    "pdf_url": item.get("pdf") or "",
                }
            )
        return self._format_results("Google Patents", query, rows[:limit])

    def search_google_patents_details(
        self,
        patent_id: str,
        timeout_seconds: int = 20,
    ) -> str:
        patent_id = str(patent_id or "").strip()
        if not patent_id:
            return "ERROR: patent_id cannot be empty"
        payload = self._serpapi_request(
            {
                "engine": "google_patents_details",
                "patent_id": patent_id,
            },
            timeout_seconds=timeout_seconds,
        )
        if isinstance(payload, str):
            return payload
        details = (
            payload.get("patent_results")
            or payload.get("patent_result")
            or payload.get("scholar_results")
            or payload.get("scholar_result")
            or payload
        )
        if not isinstance(details, dict):
            return "ERROR: Unexpected Google Patents details response format"

        def _value(*keys: str) -> str:
            for key in keys:
                value = details.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, (int, float)):
                    return str(value)
                if isinstance(value, list):
                    parts = []
                    for item in value[:10]:
                        if isinstance(item, (str, int, float)):
                            parts.append(str(item))
                        elif isinstance(item, dict):
                            text = " | ".join(
                                str(part)
                                for part in item.values()
                                if isinstance(part, (str, int, float)) and str(part).strip()
                            )
                            if text:
                                parts.append(text)
                    if parts:
                        return "; ".join(parts)
                if isinstance(value, dict):
                    text = " | ".join(
                        str(part)
                        for part in value.values()
                        if isinstance(part, (str, int, float)) and str(part).strip()
                    )
                    if text:
                        return text
            return ""

        lines = [f"SUCCESS: Google Patents details for '{patent_id}':"]
        title = _value("title", "name")
        if title:
            lines.append(f"Title: {title}")
        for label, keys in [
            ("Patent ID", ("patent_id", "publication_number")),
            ("Application", ("application_number", "application")),
            ("Publication date", ("publication_date", "publication")),
            ("Filing date", ("filing_date", "filing")),
            ("Grant date", ("grant_date", "grant")),
            ("Assignee", ("assignee", "current_assignee")),
            ("Inventor", ("inventor", "inventors")),
            ("PDF", ("pdf", "pdf_link")),
            ("Link", ("link", "patent_link")),
        ]:
            value = _value(*keys)
            if value:
                lines.append(f"{label}: {_short(value, 500)}")

        abstract = _value("abstract")
        if abstract:
            lines.append(f"Abstract: {_short(abstract, 900)}")
        claims = details.get("claims")
        if isinstance(claims, list) and claims:
            lines.append("Claims:")
            for idx, claim in enumerate(claims[:5], start=1):
                if isinstance(claim, dict):
                    text = str(claim.get("text") or claim.get("claim") or "").strip()
                else:
                    text = str(claim or "").strip()
                if text:
                    lines.append(f"{idx}. {_short(text, 500)}")
        description = _value("description")
        if description:
            lines.append(f"Description excerpt: {_short(description, 1200)}")
        for heading, key in [
            ("Classifications", "classifications"),
            ("Citations", "citations"),
            ("Cited by", "cited_by"),
            ("Patent family", "family"),
            ("Similar documents", "similar_documents"),
            ("Events", "events"),
            ("Legal events", "legal_events"),
        ]:
            rows = details.get(key)
            if isinstance(rows, dict):
                flattened = []
                for value in rows.values():
                    if isinstance(value, list):
                        flattened.extend(value)
                rows = flattened
            if isinstance(rows, list) and rows:
                lines.append(f"{heading}:")
                for idx, row in enumerate(rows[:6], start=1):
                    if isinstance(row, dict):
                        text = " | ".join(
                            str(value)
                            for value in row.values()
                            if isinstance(value, (str, int, float)) and str(value).strip()
                        )
                    else:
                        text = str(row or "")
                    if text.strip():
                        lines.append(f"{idx}. {_short(text, 400)}")
        return "\n".join(lines)

    def search_google_scholar(
        self,
        query: str = "",
        num: Optional[int] = None,
        include_patents: bool = False,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        sort_by_date: bool = False,
        review_articles_only: bool = False,
        exclude_citations: bool = False,
        cites: str = "",
        cluster: str = "",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        cites = str(cites or "").strip()
        cluster = str(cluster or "").strip()
        if not query and not cites and not cluster:
            return "ERROR: query, cites, or cluster is required"
        limit = _clamp(self._max_results(num, default_limit=10), 1, 20)
        params: dict[str, Any] = {
            "engine": "google_scholar",
            "num": limit,
            "as_sdt": "7" if include_patents else "0",
        }
        if cluster:
            params["cluster"] = cluster
            params.pop("as_sdt", None)
        else:
            if query:
                params["q"] = query
            if cites:
                params["cites"] = cites
        if start_year is not None:
            params["as_ylo"] = _coerce_int(start_year, 0)
        if end_year is not None:
            params["as_yhi"] = _coerce_int(end_year, 0)
        if sort_by_date:
            params["scisbd"] = "2"
        if review_articles_only:
            params["as_rr"] = "1"
        if exclude_citations:
            params["as_vis"] = "1"
        if hl.strip():
            params["hl"] = hl.strip()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        results = payload.get("organic_results") or []
        if not isinstance(results, list) or not results:
            label = query or (f"cites:{cites}" if cites else f"cluster:{cluster}")
            return f"SUCCESS: No Google Scholar results found for '{label}'."
        label = query or (f"cites:{cites}" if cites else f"cluster:{cluster}")
        lines = [f"SUCCESS: Google Scholar results for '{label}' (top {min(len(results), limit)}):"]
        for idx, item in enumerate(results[:limit], start=1):
            if not isinstance(item, dict):
                continue
            link = ""
            resources = item.get("resources") or []
            resource_lines = []
            if isinstance(resources, list):
                for resource in resources:
                    if not isinstance(resource, dict):
                        continue
                    resource_link = str(resource.get("link") or "").strip()
                    if not resource_link:
                        continue
                    file_format = str(resource.get("file_format") or "").strip().lower()
                    if "pdf" in file_format or resource_link.lower().endswith(".pdf"):
                        link = resource_link
                        break
                    title = str(resource.get("title") or resource.get("file_format") or "resource").strip()
                    resource_lines.append(f"{title}: {resource_link}")
                if not link:
                    for resource in resources:
                        if isinstance(resource, dict) and resource.get("link"):
                            link = str(resource.get("link")).strip()
                            break
            if not link:
                link = str(item.get("link") or "").strip()
            pub = item.get("publication_info") or {}
            summary = ""
            if isinstance(pub, dict):
                summary = str(pub.get("summary") or "")
            year_match = re.search(r"(19|20)\d{2}", summary)
            year = year_match.group(0) if year_match else ""
            title = item.get("title") or "Google Scholar result"
            lines.append(f"{idx}. {title}" + (f" - {link}" if link else ""))
            meta = []
            if year:
                meta.append(f"year: {year}")
            result_id = str(item.get("result_id") or "").strip()
            if result_id:
                meta.append(f"result_id: {result_id}")
            if summary:
                meta.append(summary)
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if item.get("snippet"):
                lines.append(f"   {_short(str(item['snippet']), 320)}")
            inline_links = item.get("inline_links") or {}
            if isinstance(inline_links, dict):
                cited_by = inline_links.get("cited_by") or {}
                if isinstance(cited_by, dict):
                    cited_id = str(cited_by.get("cites_id") or "").strip()
                    cited_total = cited_by.get("total")
                    if cited_id or cited_total:
                        lines.append(
                            "   Cited by: "
                            + " | ".join(
                                part
                                for part in [
                                    f"total {cited_total}" if cited_total is not None else "",
                                    f"cites_id {cited_id}" if cited_id else "",
                                ]
                                if part
                            )
                        )
                versions = inline_links.get("versions") or {}
                if isinstance(versions, dict):
                    cluster_id = str(versions.get("cluster_id") or "").strip()
                    total = versions.get("total")
                    if cluster_id or total:
                        lines.append(
                            "   Versions: "
                            + " | ".join(
                                part
                                for part in [
                                    f"total {total}" if total is not None else "",
                                    f"cluster {cluster_id}" if cluster_id else "",
                                ]
                                if part
                            )
                        )
                serpapi_cite_link = inline_links.get("serpapi_cite_link")
                if serpapi_cite_link:
                    lines.append(f"   Cite lookup: {serpapi_cite_link}")
            if resource_lines:
                lines.append("   Resources:")
                for resource_line in resource_lines[:4]:
                    lines.append(f"   - {resource_line}")
        return "\n".join(lines)

    def search_google_scholar_cite(
        self,
        result_id: str,
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        result_id = str(result_id or "").strip()
        if not result_id:
            return "ERROR: result_id cannot be empty"
        params = {
            "engine": "google_scholar_cite",
            "q": result_id,
        }
        if hl.strip():
            params["hl"] = hl.strip()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        citations = payload.get("citations") or payload.get("citation") or []
        links = payload.get("links") or payload.get("resources") or []
        lines = [f"SUCCESS: Google Scholar citations for result_id '{result_id}':"]
        if isinstance(citations, list):
            for item in citations[:8]:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or item.get("format") or "Citation"
                snippet = item.get("snippet") or item.get("citation") or ""
                if snippet:
                    lines.append(f"- {title}: {_short(str(snippet), 700)}")
        elif isinstance(citations, dict):
            for key, value in list(citations.items())[:8]:
                if isinstance(value, str) and value.strip():
                    lines.append(f"- {key}: {_short(value, 700)}")
        if isinstance(links, list) and links:
            lines.append("Citation export links:")
            for item in links[:8]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("title") or item.get("format") or "link"
                link = item.get("link") or item.get("url") or ""
                if link:
                    lines.append(f"- {name}: {link}")
        if len(lines) == 1:
            return f"SUCCESS: No Google Scholar citation formats found for result_id '{result_id}'."
        return "\n".join(lines)

    def search_youtube_videos(
        self,
        query: str,
        limit: Optional[int] = None,
        gl: str = "",
        hl: str = "",
        sp: str = "",
        next_page_token: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        limit = self._max_results(limit, default_limit=10)
        params: dict[str, Any] = {
            "engine": "youtube",
            "search_query": query,
        }
        if gl.strip():
            params["gl"] = gl.strip().lower()
        if hl.strip():
            params["hl"] = hl.strip().lower()
        if next_page_token.strip():
            params["sp"] = next_page_token.strip()
        elif sp.strip():
            params["sp"] = sp.strip()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        videos = []
        for key in ("video_results", "videos_results", "shorts_results", "results"):
            items = payload.get(key)
            if isinstance(items, list):
                videos.extend(items)
        rows = []
        for item in videos:
            if not isinstance(item, dict):
                continue
            video_link = str(item.get("link") or item.get("url") or "").strip()
            video_id = str(item.get("id") or item.get("video_id") or "").strip()
            if not video_id:
                video_id = _normalize_youtube_video_id(video_link)
            if not video_link:
                if video_id:
                    video_link = f"https://www.youtube.com/watch?v={video_id}"
            if not video_link:
                continue
            channel = item.get("channel") or {}
            if isinstance(channel, dict):
                channel_name = channel.get("name") or ""
            else:
                channel_name = ""
            meta_bits = []
            if video_id:
                meta_bits.append(f"video_id: {video_id}")
            for key in ("published_date", "views", "length", "duration"):
                value = item.get(key)
                if value:
                    meta_bits.append(f"{key}: {value}")
            rows.append(
                {
                    "title": item.get("title") or "YouTube video",
                    "url": video_link,
                    "year": "",
                    "source": "YouTube",
                    "authors": channel_name,
                    "snippet": " | ".join(meta_bits),
                }
            )
        output = self._format_results("YouTube", query, rows[:limit])
        pagination = payload.get("serpapi_pagination") or payload.get("pagination") or {}
        if isinstance(pagination, dict) and pagination.get("next_page_token"):
            output += f"\nNext page token: {pagination['next_page_token']}"
        related_searches = payload.get("related_searches") or []
        if isinstance(related_searches, list) and related_searches:
            output += "\nRelated searches:"
            for item in related_searches[:8]:
                if isinstance(item, dict):
                    query_text = str(item.get("query") or item.get("title") or "").strip()
                    if query_text:
                        output += f"\n- {query_text}"
        return output

    def get_youtube_video_details(
        self,
        video_id: str,
        gl: str = "",
        hl: str = "",
        next_page_token: str = "",
        timeout_seconds: int = 30,
    ) -> str:
        video_id = _normalize_youtube_video_id(video_id)
        if not video_id:
            return "ERROR: video_id is required"
        params: dict[str, Any] = {
            "engine": "youtube_video",
            "v": video_id,
        }
        if gl.strip():
            params["gl"] = gl.strip().lower()
        if hl.strip():
            params["hl"] = hl.strip().lower()
        if next_page_token.strip():
            params["next_page_token"] = next_page_token.strip()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return "ERROR: Unexpected SerpAPI response format"
        video = (
            payload.get("video_result")
            or payload.get("video_results")
            or payload.get("video")
            or {}
        )
        if isinstance(video, list):
            video = video[0] if video and isinstance(video[0], dict) else {}
        if not isinstance(video, dict):
            video = {}
        title = video.get("title") or payload.get("title") or video_id
        link = video.get("link") or video.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        lines = [f"SUCCESS: YouTube video details for '{video_id}':", f"Video: {title} - {link}"]
        channel = video.get("channel") or payload.get("channel") or {}
        channel_name = channel.get("name") if isinstance(channel, dict) else str(channel or "")
        meta_parts = []
        for key, label in (
            ("published_date", "published"),
            ("views", "views"),
            ("likes", "likes"),
            ("duration", "duration"),
        ):
            value = video.get(key) or payload.get(key)
            if value:
                meta_parts.append(f"{label}: {value}")
        if channel_name:
            meta_parts.append(f"channel: {channel_name}")
        if meta_parts:
            lines.append(" | ".join(meta_parts))
        description = _plain_text(video.get("description") or payload.get("description") or "")
        if description:
            lines.append(f"Description: {_short(description, 1200)}")
        pagination = payload.get("serpapi_pagination") or payload.get("pagination") or {}
        for token_key, label in (
            ("comments_next_page_token", "Comments next page token"),
            ("related_videos_next_page_token", "Related videos next page token"),
            ("replies_next_page_token", "Replies next page token"),
        ):
            token = payload.get(token_key)
            if not token and isinstance(pagination, dict):
                token = pagination.get(token_key)
            if token:
                lines.append(f"{label}: {token}")
        sorting = payload.get("comments_sorting_token")
        if not sorting and isinstance(pagination, dict):
            sorting = pagination.get("comments_sorting_token")
        if isinstance(sorting, dict) and sorting.get("token"):
            lines.append(f"Comments sorting token: {sorting['token']}")
        comments = payload.get("comments") or payload.get("comment_results") or []
        if isinstance(comments, list) and comments:
            lines.append("Comments:")
            for idx, item in enumerate(comments[:12], start=1):
                if not isinstance(item, dict):
                    continue
                author = item.get("author") or item.get("author_name") or ""
                text = item.get("text") or item.get("content") or item.get("comment") or ""
                likes = item.get("likes") or item.get("vote_count") or ""
                meta = []
                if author:
                    meta.append(f"author: {author}")
                if likes:
                    meta.append(f"likes: {likes}")
                prefix = f"{idx}. "
                if meta:
                    prefix += f"{' | '.join(meta)} - "
                lines.append(f"{prefix}{_short(str(text), 500)}")
        related = payload.get("related_videos") or payload.get("related_video_results") or []
        if isinstance(related, list) and related:
            lines.append("Related videos:")
            for idx, item in enumerate(related[:10], start=1):
                if not isinstance(item, dict):
                    continue
                rel_title = item.get("title") or "YouTube video"
                rel_link = item.get("link") or item.get("url") or ""
                rel_id = item.get("video_id") or item.get("id") or _normalize_youtube_video_id(rel_link)
                suffix = f" | video_id: {rel_id}" if rel_id else ""
                lines.append(f"{idx}. {rel_title} - {rel_link}{suffix}")
        return "\n".join(lines)

    def get_youtube_video_transcript(
        self,
        video_id: str,
        language_code: str = "",
        max_segments: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        video_id = _normalize_youtube_video_id(video_id)
        if not video_id:
            return "ERROR: video_id is required"
        params: dict[str, Any] = {
            "engine": "youtube_video_transcript",
            "v": video_id,
        }
        if language_code.strip():
            params["language_code"] = language_code.strip()
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        segments = payload.get("transcript") or payload.get("transcripts") or []
        if not isinstance(segments, list) or not segments:
            return f"SUCCESS: No transcript segments found for video '{video_id}'."
        if max_segments is None:
            shown_segments = segments
            descriptor = f"all {len(segments)} segments"
        else:
            count = _clamp(_coerce_int(max_segments, len(segments)), 1, len(segments))
            shown_segments = segments[:count]
            descriptor = f"top {len(shown_segments)} of {len(segments)} segments"
        lines = [f"SUCCESS: YouTube transcript for '{video_id}' ({descriptor}):"]
        for idx, seg in enumerate(shown_segments, start=1):
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("snippet") or seg.get("text") or "").strip()
            start = seg.get("start") or seg.get("start_ms") or ""
            if not text:
                continue
            prefix = f"[{start}] " if start != "" else ""
            lines.append(f"{idx}. {prefix}{text}")
        transcript_text = "\n".join(lines)
        raw_path, text_path = _write_artifacts(
            "youtube-transcripts",
            video_id,
            "json",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            transcript_text,
        )
        lines.append(f"Artifact JSON: {raw_path}")
        lines.append(f"Artifact text: {text_path}")
        return "\n".join(lines)

    def search_medrxiv_preprints(
        self,
        query: str,
        start_date: str = "",
        end_date: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        limit = self._max_results(max_results)
        today = date.today()
        if not end_date.strip():
            end_date = today.isoformat()
        if not start_date.strip():
            start_date = (today - timedelta(days=365)).isoformat()
        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not date_re.match(start_date) or not date_re.match(end_date):
            return "ERROR: start_date and end_date must use YYYY-MM-DD format"
        terms = [term for term in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(term) > 2]
        rows = []
        cursor = 0
        max_pages = 5
        try:
            for _ in range(max_pages):
                payload = None
                last_response = None
                for url in (
                    f"https://api.biorxiv.org/details/medrxiv/{start_date}/{end_date}/{cursor}/json",
                    f"https://api.medrxiv.org/details/medrxiv/{start_date}/{end_date}/{cursor}",
                ):
                    response = _get_with_retries(
                        url,
                        headers=_NCBI_HEADERS,
                        timeout=timeout_seconds,
                    )
                    last_response = response
                    response.raise_for_status()
                    candidate = response.json()
                    if isinstance(candidate, dict):
                        payload = candidate
                        break
                if payload is None:
                    if last_response is not None:
                        last_response.raise_for_status()
                    return "ERROR: medRxiv returned invalid JSON"
                collection = payload.get("collection") or []
                if not collection:
                    break
                for item in collection:
                    if not isinstance(item, dict):
                        continue
                    haystack = " ".join(
                        str(item.get(key) or "")
                        for key in ["title", "abstract", "authors", "category"]
                    ).lower()
                    if terms and not all(term in haystack for term in terms):
                        continue
                    jatsxml = str(item.get("jatsxml") or "").strip()
                    if not jatsxml:
                        continue
                    rows.append(item)
                    if len(rows) >= limit:
                        break
                if len(rows) >= limit:
                    break
                cursor += len(collection)
        except requests.exceptions.Timeout:
            return "ERROR: medRxiv search timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to medRxiv"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: medRxiv returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: medRxiv returned invalid JSON"
        if not rows:
            return (
                f"SUCCESS: No medRxiv full-text preprints found for '{query}' "
                f"between {start_date} and {end_date}."
            )
        lines = [
            f"SUCCESS: medRxiv full-text preprints for '{query}' "
            f"between {start_date} and {end_date} (top {len(rows)}):"
        ]
        for idx, item in enumerate(rows, start=1):
            doi = str(item.get("doi") or "").strip()
            title = item.get("title") or "medRxiv preprint"
            date_value = item.get("date") or ""
            category = item.get("category") or ""
            jatsxml = str(item.get("jatsxml") or "").strip()
            lines.append(f"{idx}. {title}")
            meta = []
            if doi:
                meta.append(f"DOI: {doi}")
            if date_value:
                meta.append(f"date: {date_value}")
            if category:
                meta.append(f"category: {category}")
            if meta:
                lines.append(f"   {' | '.join(meta)}")
            if item.get("authors"):
                lines.append(f"   authors: {_short(str(item['authors']), 260)}")
            if item.get("abstract"):
                lines.append(f"   {_short(str(item['abstract']), 320)}")
            lines.append(f"   Full-text JATS XML: {jatsxml}")
        return "\n".join(lines)

    def download_medrxiv_full_text(
        self,
        jatsxml_url_or_doi: str,
        timeout_seconds: int = 30,
    ) -> str:
        raw = str(jatsxml_url_or_doi or "").strip()
        if not raw:
            return "ERROR: jatsxml_url_or_doi cannot be empty"
        target_url = raw
        if not target_url.lower().startswith("http"):
            doi = raw
            try:
                response = _get_with_retries(
                    f"https://api.biorxiv.org/details/medrxiv/{quote(doi, safe='/')}",
                    headers=_NCBI_HEADERS,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                collection = payload.get("collection") or []
                if collection and isinstance(collection[0], dict):
                    target_url = str(collection[0].get("jatsxml") or "").strip()
            except Exception:
                target_url = ""
        if not target_url:
            return "ERROR: Could not resolve medRxiv JATS XML URL"
        try:
            response = _get_with_retries(
                target_url,
                timeout=timeout_seconds,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: medRxiv full-text download timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect while downloading medRxiv full text"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: medRxiv full-text download returned HTTP {exc.response.status_code}"
        xml_text = response.text or ""
        plain_text = _xml_to_text(xml_text)
        if len(plain_text) < 200:
            return "ERROR: medRxiv full-text response did not contain enough extractable text"
        label = raw if not raw.lower().startswith("http") else os.path.basename(raw)
        raw_path, text_path = _write_artifacts("medrxiv-full-text", label, "xml", xml_text, plain_text)
        return (
            "SUCCESS: Downloaded medRxiv full text.\n"
            f"Source: {raw}\n"
            f"JATS XML URL: {target_url}\n"
            f"Characters: {len(plain_text)}\n"
            f"Saved XML: {raw_path}\n"
            f"Saved text: {text_path}"
        )


def get_arxiv_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_arxiv")
    def search_arxiv(query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search arXiv papers with direct PDF URLs.

        Args:
            query: Search query string.
            max_results: Optional max number of results.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_arxiv",
            tool_input,
            lambda: helper.search_arxiv(query=query, max_results=max_results, timeout_seconds=timeout_seconds),
        )

    return _with_scientific_output(search_arxiv)


def get_europe_pmc_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_europe_pmc")
    def search_europe_pmc(
        query: str,
        page: int = 1,
        page_size: int = 25,
        timeout_seconds: int = 20,
    ) -> str:
        """Search Europe PMC and return open-access papers with PDF URLs.

        Args:
            query: Search query string.
            page: Page number (1+).
            page_size: Number of results per page (1-50).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "page_size": page_size,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_europe_pmc",
            tool_input,
            lambda: helper.search_europe_pmc(
                query=query,
                page=page,
                page_size=page_size,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_europe_pmc)


def get_pmc_full_text_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_pmc_full_text")
    def search_pmc_full_text(
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search PubMed Central for articles with downloadable full-text XML.

        Args:
            query: Search query string.
            max_results: Optional max number of results.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_pmc_full_text",
            tool_input,
            lambda: helper.search_pmc_full_text(
                query=query,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_pmc_full_text)


def get_pmc_full_text_download_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="download_pmc_full_text")
    def download_pmc_full_text(pmcid_or_id: str, timeout_seconds: int = 30) -> str:
        """Download a PubMed Central article as full-text XML and extracted text.

        Args:
            pmcid_or_id: PMCID such as PMC1234567, a PMC article URL, or numeric PMC id.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {"pmcid_or_id": pmcid_or_id, "timeout_seconds": timeout_seconds}
        return _run_logged(
            "download_pmc_full_text",
            tool_input,
            lambda: helper.download_pmc_full_text(
                pmcid_or_id=pmcid_or_id,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(download_pmc_full_text)


def get_ncbi_bookshelf_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_ncbi_bookshelf")
    def search_ncbi_bookshelf(
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search NCBI Bookshelf for books, chapters, tables, and reports with readable full content.

        Args:
            query: Search query string.
            max_results: Optional max number of results.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_ncbi_bookshelf",
            tool_input,
            lambda: helper.search_ncbi_bookshelf(
                query=query,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_ncbi_bookshelf)


def get_ncbi_bookshelf_download_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="download_ncbi_bookshelf")
    def download_ncbi_bookshelf(accession_or_id: str, timeout_seconds: int = 30) -> str:
        """Download an NCBI Bookshelf item as reader HTML and extracted text.

        Args:
            accession_or_id: NBK accession, Bookshelf URL, or numeric Bookshelf id.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {"accession_or_id": accession_or_id, "timeout_seconds": timeout_seconds}
        return _run_logged(
            "download_ncbi_bookshelf",
            tool_input,
            lambda: helper.download_ncbi_bookshelf(
                accession_or_id=accession_or_id,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(download_ncbi_bookshelf)


def get_semantic_scholar_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_semantic_scholar")
    def search_semantic_scholar(query: str, limit: int = 20, timeout_seconds: int = 20) -> str:
        """Search Semantic Scholar and return papers with open-access URLs.

        Args:
            query: Search query string.
            limit: Number of results to request (1-20).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "limit": limit,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_semantic_scholar",
            tool_input,
            lambda: helper.search_semantic_scholar(query=query, limit=limit, timeout_seconds=timeout_seconds),
        )

    return _with_scientific_output(search_semantic_scholar)


def get_openalex_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_openalex")
    def search_openalex(
        query: str,
        page: int = 1,
        per_page: int = 10,
        timeout_seconds: int = 20,
    ) -> str:
        """Search OpenAlex and return works with open-access PDF URLs.

        Args:
            query: Search query string.
            page: Page number (1+).
            per_page: Number of results per page.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_openalex",
            tool_input,
            lambda: helper.search_openalex(
                query=query,
                page=page,
                per_page=per_page,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_openalex)


def get_plos_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_plos")
    def search_plos(query: str, rows: int = 20, start: int = 0, timeout_seconds: int = 20) -> str:
        """Search PLOS and return direct full-text PDF URLs.

        Args:
            query: Search query string.
            rows: Number of results to return.
            start: Result offset.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "rows": rows,
            "start": start,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_plos",
            tool_input,
            lambda: helper.search_plos(query=query, rows=rows, start=start, timeout_seconds=timeout_seconds),
        )

    return _with_scientific_output(search_plos)


def get_google_patents_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_patents")
    def search_google_patents(
        query: str,
        page: int = 1,
        num: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        """Search Google Patents via SerpAPI.

        Args:
            query: Search query string.
            page: Page number (1+).
            num: Number of results (default 10).
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "page": page,
            "num": num,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_patents",
            tool_input,
            lambda: helper.search_google_patents(
                query=query,
                page=page,
                num=num,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_google_patents)


def get_google_patents_details_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_patents_details")
    def search_google_patents_details(patent_id: str, timeout_seconds: int = 20) -> str:
        """Fetch detailed Google Patents metadata via SerpAPI.

        Args:
            patent_id: Patent ID from search_google_patents, e.g. patent/US11734097B1/en.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {"patent_id": patent_id, "timeout_seconds": timeout_seconds}
        return _run_logged(
            "search_google_patents_details",
            tool_input,
            lambda: helper.search_google_patents_details(
                patent_id=patent_id,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_google_patents_details)


def get_google_scholar_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_scholar")
    def search_google_scholar(
        query: str = "",
        num: Optional[int] = None,
        include_patents: bool = False,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        sort_by_date: bool = False,
        review_articles_only: bool = False,
        exclude_citations: bool = False,
        cites: str = "",
        cluster: str = "",
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search Google Scholar via SerpAPI.

        Args:
            query: Search query string. Optional when using cites or cluster.
            num: Number of results (default 10).
            include_patents: Whether to include patents in search.
            start_year: Include results from this year onward.
            end_year: Include results up to this year.
            sort_by_date: Sort/filter for recent articles.
            review_articles_only: Return only review articles.
            exclude_citations: Exclude citation-only results.
            cites: Google Scholar cites_id to find citing papers.
            cluster: Google Scholar cluster_id to find all versions.
            hl: Optional language code.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "num": num,
            "include_patents": include_patents,
            "start_year": start_year,
            "end_year": end_year,
            "sort_by_date": sort_by_date,
            "review_articles_only": review_articles_only,
            "exclude_citations": exclude_citations,
            "cites": cites,
            "cluster": cluster,
            "hl": hl,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_scholar",
            tool_input,
            lambda: helper.search_google_scholar(
                query=query,
                num=num,
                include_patents=include_patents,
                start_year=start_year,
                end_year=end_year,
                sort_by_date=sort_by_date,
                review_articles_only=review_articles_only,
                exclude_citations=exclude_citations,
                cites=cites,
                cluster=cluster,
                hl=hl,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_google_scholar)


def get_google_scholar_cite_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_scholar_cite")
    def search_google_scholar_cite(
        result_id: str,
        hl: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Fetch citation formats and export links for a Google Scholar result.

        Args:
            result_id: result_id from search_google_scholar.
            hl: Optional language code.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "result_id": result_id,
            "hl": hl,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_google_scholar_cite",
            tool_input,
            lambda: helper.search_google_scholar_cite(
                result_id=result_id,
                hl=hl,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_google_scholar_cite)


def get_youtube_video_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_youtube_videos")
    def search_youtube_videos(
        query: str,
        limit: Optional[int] = None,
        gl: str = "",
        hl: str = "",
        sp: str = "",
        next_page_token: str = "",
        timeout_seconds: int = 20,
    ) -> str:
        """Search YouTube videos via SerpAPI.

        Args:
            query: Search query string.
            limit: Max number of results.
            gl: Country code (e.g. 'us').
            hl: Language code (e.g. 'en').
            sp: Optional YouTube filter/pagination token from YouTube or SerpAPI.
            next_page_token: Optional token from a previous search_youtube_videos call.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "limit": limit,
            "gl": gl,
            "hl": hl,
            "sp": sp,
            "next_page_token": next_page_token,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_youtube_videos",
            tool_input,
            lambda: helper.search_youtube_videos(
                query=query,
                limit=limit,
                gl=gl,
                hl=hl,
                sp=sp,
                next_page_token=next_page_token,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_youtube_videos)


def get_youtube_video_details_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_youtube_video_details")
    def get_youtube_video_details(
        video_id: str,
        gl: str = "",
        hl: str = "",
        next_page_token: str = "",
        timeout_seconds: int = 30,
    ) -> str:
        """Get YouTube video details, comments, related videos, and follow-up pagination tokens.

        Args:
            video_id: YouTube video ID or URL.
            gl: Optional country code.
            hl: Optional language code.
            next_page_token: Optional token for comments, replies, or related video pagination.
            timeout_seconds: Request timeout.
        """
        tool_input = {
            "video_id": video_id,
            "gl": gl,
            "hl": hl,
            "next_page_token": next_page_token,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "get_youtube_video_details",
            tool_input,
            lambda: helper.get_youtube_video_details(
                video_id=video_id,
                gl=gl,
                hl=hl,
                next_page_token=next_page_token,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(get_youtube_video_details)


def get_youtube_transcript_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_youtube_video_transcript")
    def get_youtube_video_transcript(
        video_id: str,
        language_code: str = "",
        max_segments: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Get the full transcript of a YouTube video by default.

        Args:
            video_id: The YouTube video ID or URL.
            language_code: Optional language code.
            max_segments: Optional cap. Leave empty to return the complete transcript.
            timeout_seconds: Request timeout.
        """
        tool_input = {
            "video_id": video_id,
            "language_code": language_code,
            "max_segments": max_segments,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "get_youtube_video_transcript",
            tool_input,
            lambda: helper.get_youtube_video_transcript(
                video_id=video_id,
                language_code=language_code,
                max_segments=max_segments,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(get_youtube_video_transcript)


def get_medrxiv_preprint_search_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_medrxiv_preprints")
    def search_medrxiv_preprints(
        query: str,
        start_date: str = "",
        end_date: str = "",
        max_results: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Search recent medRxiv preprints and return direct full-text JATS XML URLs.

        Args:
            query: Search query string matched against title, abstract, authors, and category.
            start_date: Optional start date in YYYY-MM-DD; defaults to one year ago.
            end_date: Optional end date in YYYY-MM-DD; defaults to today.
            max_results: Optional max number of matching results.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "search_medrxiv_preprints",
            tool_input,
            lambda: helper.search_medrxiv_preprints(
                query=query,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(search_medrxiv_preprints)


def get_medrxiv_full_text_download_tool(helper: ScientificSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="download_medrxiv_full_text")
    def download_medrxiv_full_text(jatsxml_url_or_doi: str, timeout_seconds: int = 30) -> str:
        """Download a medRxiv preprint as full-text JATS XML and extracted text.

        Args:
            jatsxml_url_or_doi: Full-text JATS XML URL from search_medrxiv_preprints, or a supported medRxiv DOI.
            timeout_seconds: Request timeout in seconds.
        """
        tool_input = {
            "jatsxml_url_or_doi": jatsxml_url_or_doi,
            "timeout_seconds": timeout_seconds,
        }
        return _run_logged(
            "download_medrxiv_full_text",
            tool_input,
            lambda: helper.download_medrxiv_full_text(
                jatsxml_url_or_doi=jatsxml_url_or_doi,
                timeout_seconds=timeout_seconds,
            ),
        )

    return _with_scientific_output(download_medrxiv_full_text)
