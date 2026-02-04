from io import BytesIO
from typing import Optional

import requests
from langchain_core.tools import StructuredTool
from pypdf import PdfReader

from .config import ToolsConfig


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[truncated]"


class PdfTextTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def download_pdf_as_text(
        self,
        url: str,
        max_chars: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        if not url or not str(url).strip():
            return "ERROR: url cannot be empty"
        limit = int(max_chars or self.config.pdf_text_max_chars or 12000)
        if limit < 500:
            limit = 500
        if limit > 100000:
            limit = 100000
        try:
            response = requests.get(
                url,
                timeout=timeout_seconds,
                allow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "ERROR: PDF download timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect while downloading PDF"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: PDF download returned HTTP {exc.response.status_code}"

        content_type = str(response.headers.get("content-type") or "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            return (
                "ERROR: URL did not return a PDF content-type. "
                f"Got: {content_type or 'unknown'}"
            )

        try:
            reader = PdfReader(BytesIO(response.content))
        except Exception as exc:
            return f"ERROR: Failed to parse PDF ({exc})"

        chunks = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text.strip():
                chunks.append(page_text.strip())
        full_text = "\n\n".join(chunks).strip()
        if not full_text:
            return "ERROR: No extractable text found in PDF"

        text = _truncate(full_text, limit)
        return (
            "SUCCESS: Extracted PDF text.\n"
            f"URL: {url}\n"
            f"Characters: {len(full_text)}\n\n"
            f"{text}"
        )


def build_pdf_text_tool(config: ToolsConfig) -> StructuredTool:
    helper = PdfTextTool(config)

    def _download_pdf_as_text(
        url: str,
        max_chars: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Download a PDF URL and extract readable text.

        Args:
            url: PDF URL to download and parse.
            max_chars: Optional max output size.
            timeout_seconds: Request timeout in seconds.
        """
        return helper.download_pdf_as_text(
            url=url,
            max_chars=max_chars,
            timeout_seconds=timeout_seconds,
        )

    return StructuredTool.from_function(
        name="download_pdf_as_text",
        description=_download_pdf_as_text.__doc__ or "Download a PDF and extract text.",
        func=_download_pdf_as_text,
    )
