from io import BytesIO
import os
import re
from urllib.parse import urlparse
from uuid import uuid4
from typing import Optional

import requests
from langchain_core.tools import StructuredTool
from pypdf import PdfReader

from .config import ToolsConfig


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

        output_dir = "/tmp/chack-pdf-text"
        os.makedirs(output_dir, exist_ok=True)
        parsed = urlparse(url)
        base_name = os.path.basename(parsed.path or "").strip() or "document.pdf"
        base_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
        if base_name.lower().endswith(".pdf"):
            base_name = base_name[:-4]
        file_path = os.path.join(output_dir, f"{base_name}_{uuid4().hex}.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(full_text)

        return (
            "SUCCESS: Extracted PDF text and saved to filesystem.\n"
            f"URL: {url}\n"
            f"Characters: {len(full_text)}\n\n"
            f"Saved file: {file_path}\n"
            "Use exec tool with grep/sed/cat on this file to locate relevant data."
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
            max_chars: Deprecated, ignored (kept for compatibility).
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
