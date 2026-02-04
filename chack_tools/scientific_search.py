import re
import time
from typing import Any, Optional

import requests
from langchain_core.tools import StructuredTool

from .config import ToolsConfig


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
        return str(getattr(self.config, "serpapi_api_key", "") or "").strip()

    def _serpapi_request(self, params: dict[str, Any], timeout_seconds: int = 20) -> Any:
        api_key = self._serpapi_key()
        if not api_key:
            return "ERROR: SerpAPI key not configured."
        req_params = dict(params)
        req_params["api_key"] = api_key
        req_params["output"] = "json"
        try:
            response = requests.get("https://serpapi.com/search", params=req_params, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: SerpAPI request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to SerpAPI"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: SerpAPI returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: SerpAPI returned invalid JSON"
        if isinstance(payload, dict) and payload.get("error"):
            return f"ERROR: SerpAPI error ({payload.get('error')})"
        return payload

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
            response = requests.get("http://export.arxiv.org/api/query", params=params, timeout=timeout_seconds)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: arXiv request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to arXiv"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: arXiv returned HTTP {exc.response.status_code}"

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
            response = requests.get(url, params=params, timeout=timeout_seconds)
            retries = 0
            while response.status_code == 429 and retries < 3:
                retry_after = _coerce_int(response.headers.get("Retry-After"), 2 + retries * 2)
                time.sleep(max(1, min(retry_after, 10)))
                response = requests.get(url, params=params, timeout=timeout_seconds)
                retries += 1
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

    def search_google_scholar(
        self,
        query: str,
        num: Optional[int] = None,
        include_patents: bool = False,
        timeout_seconds: int = 20,
    ) -> str:
        if not query.strip():
            return "ERROR: Query cannot be empty"
        limit = self._max_results(num, default_limit=10)
        payload = self._serpapi_request(
            {
                "engine": "google_scholar",
                "q": query,
                "num": limit,
                "as_sdt": "7" if include_patents else "0",
            },
            timeout_seconds=timeout_seconds,
        )
        if isinstance(payload, str):
            return payload
        rows = []
        for item in payload.get("organic_results") or []:
            if not isinstance(item, dict):
                continue
            link = ""
            resources = item.get("resources") or []
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
                if not link:
                    for resource in resources:
                        if isinstance(resource, dict) and resource.get("link"):
                            link = str(resource.get("link")).strip()
                            break
            if not link:
                link = str(item.get("link") or "").strip()
            if not link:
                continue
            pub = item.get("publication_info") or {}
            summary = ""
            if isinstance(pub, dict):
                summary = str(pub.get("summary") or "")
            year_match = re.search(r"(19|20)\d{2}", summary)
            year = year_match.group(0) if year_match else ""
            rows.append(
                {
                    "title": item.get("title") or "Google Scholar result",
                    "url": link,
                    "year": year,
                    "source": "Google Scholar",
                    "authors": summary,
                    "snippet": item.get("snippet") or "",
                }
            )
        return self._format_results("Google Scholar", query, rows[:limit])

    def search_youtube_videos(
        self,
        query: str,
        limit: Optional[int] = None,
        gl: str = "",
        hl: str = "",
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
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        videos = (
            payload.get("video_results")
            or payload.get("videos_results")
            or payload.get("results")
            or []
        )
        rows = []
        for item in videos:
            if not isinstance(item, dict):
                continue
            video_link = str(item.get("link") or item.get("url") or "").strip()
            if not video_link:
                video_id = str(item.get("id") or item.get("video_id") or "").strip()
                if video_id:
                    video_link = f"https://www.youtube.com/watch?v={video_id}"
            if not video_link:
                continue
            channel = item.get("channel") or {}
            if isinstance(channel, dict):
                channel_name = channel.get("name") or ""
            else:
                channel_name = ""
            rows.append(
                {
                    "title": item.get("title") or "YouTube video",
                    "url": video_link,
                    "year": "",
                    "source": "YouTube",
                    "authors": channel_name,
                    "snippet": item.get("published_date") or item.get("views") or "",
                }
            )
        return self._format_results("YouTube", query, rows[:limit])

    def get_youtube_video_transcript(
        self,
        video_id: str,
        language_code: str = "",
        timeout_seconds: int = 30,
    ) -> str:
        video_id = (video_id or "").strip()
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
        lines = [f"SUCCESS: YouTube transcript for '{video_id}' (top {min(len(segments), 60)} segments):"]
        for idx, seg in enumerate(segments[:60], start=1):
            if not isinstance(seg, dict):
                continue
            text = _short(str(seg.get("snippet") or seg.get("text") or ""), 260)
            start = seg.get("start") or seg.get("start_ms") or ""
            if not text:
                continue
            prefix = f"[{start}] " if start != "" else ""
            lines.append(f"{idx}. {prefix}{text}")
        return "\n".join(lines)


def build_arxiv_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ScientificSearchTool(config)

    def _search_arxiv(query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search arXiv papers with direct PDF URLs.

        Args:
            query: Search query string.
            max_results: Optional max number of results.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.search_arxiv(query=query, max_results=max_results, timeout_seconds=timeout_seconds)

    return StructuredTool.from_function(
        name="search_arxiv",
        description=_search_arxiv.__doc__ or "Search arXiv papers with direct PDF URLs.",
        func=_search_arxiv,
    )


def build_europe_pmc_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ScientificSearchTool(config)

    def _search_europe_pmc(
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
        return helper.search_europe_pmc(
            query=query,
            page=page,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="search_europe_pmc",
        description=_search_europe_pmc.__doc__ or "Search Europe PMC papers with PDF URLs.",
        func=_search_europe_pmc,
    )


def build_semantic_scholar_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ScientificSearchTool(config)

    def _search_semantic_scholar(query: str, limit: int = 20, timeout_seconds: int = 20) -> str:
        """Search Semantic Scholar and return papers with open-access URLs.

        Args:
            query: Search query string.
            limit: Number of results to request (1-20).
            timeout_seconds: Request timeout in seconds.
        """
        return helper.search_semantic_scholar(query=query, limit=limit, timeout_seconds=timeout_seconds)

    return StructuredTool.from_function(
        name="search_semantic_scholar",
        description=_search_semantic_scholar.__doc__ or "Search Semantic Scholar papers with open-access URLs.",
        func=_search_semantic_scholar,
    )


def build_openalex_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ScientificSearchTool(config)

    def _search_openalex(
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
        return helper.search_openalex(
            query=query,
            page=page,
            per_page=per_page,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="search_openalex",
        description=_search_openalex.__doc__ or "Search OpenAlex papers with open-access PDF URLs.",
        func=_search_openalex,
    )


def build_plos_search_tool(config: ToolsConfig) -> StructuredTool:
    helper = ScientificSearchTool(config)

    def _search_plos(query: str, rows: int = 20, start: int = 0, timeout_seconds: int = 20) -> str:
        """Search PLOS and return direct full-text PDF URLs.

        Args:
            query: Search query string.
            rows: Number of results to return.
            start: Result offset.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.search_plos(query=query, rows=rows, start=start, timeout_seconds=timeout_seconds)

    return StructuredTool.from_function(
        name="search_plos",
        description=_search_plos.__doc__ or "Search PLOS and return direct full-text PDF URLs.",
        func=_search_plos,
    )
