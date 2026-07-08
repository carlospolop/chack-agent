from io import BytesIO
import os
import re
from urllib.parse import urlparse
from uuid import uuid4
from typing import Optional

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests
from pypdf import PdfReader

from .config import ToolsConfig
from .research_artifacts import record_research_artifact, research_artifacts_root
from .telemetry import run_with_tool_logging


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
            "Parameters: Use the schema descriptions to provide the PDF URL, optional inline character cap, and request timeout.\n"
            "Output: Returns SUCCESS/ERROR text. On success it reports extracted character count plus local paths for the saved PDF and full extracted text file."
        )
    return tool



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

        evidence_root = research_artifacts_root()
        output_dir = (
            os.path.join(evidence_root, "pdf-text")
            if evidence_root
            else "/tmp/chack-pdf-text"
        )
        os.makedirs(output_dir, exist_ok=True)
        parsed = urlparse(url)
        base_name = os.path.basename(parsed.path or "").strip() or "document.pdf"
        base_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
        if base_name.lower().endswith(".pdf"):
            base_name = base_name[:-4]
        token = uuid4().hex
        pdf_path = os.path.join(output_dir, f"{base_name}_{token}.pdf")
        file_path = os.path.join(output_dir, f"{base_name}_{token}.txt")
        with open(pdf_path, "wb") as handle:
            handle.write(response.content)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(full_text)
        record_research_artifact(
            pdf_path,
            source_url=url,
            provenance=f"download_pdf_as_text PDF source={url}",
            tool="download_pdf_as_text",
            kind="pdf-text",
            label=url,
        )
        record_research_artifact(
            file_path,
            source_url=url,
            provenance=f"download_pdf_as_text extracted text source={url}",
            tool="download_pdf_as_text",
            kind="pdf-text",
            label=url,
        )

        return (
            "SUCCESS: Extracted PDF text and saved to filesystem.\n"
            f"URL: {url}\n"
            f"Characters: {len(full_text)}\n\n"
            f"Saved PDF: {pdf_path}\n"
            f"Saved text: {file_path}\n"
            "Use exec tool with grep/sed/cat on this file to locate relevant data."
        )


def get_pdf_text_tool(helper: PdfTextTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="download_pdf_as_text")
    def download_pdf_as_text(
        url: str,
        max_chars: Optional[int] = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Download a PDF URL and extract readable text.

        Use this to read papers or reports; then inspect the saved text file with exec + grep/sed.
        """
        tool_input = {
            "url": url,
            "max_chars": max_chars,
            "timeout_seconds": timeout_seconds,
        }
        try:
            return run_with_tool_logging(
                "download_pdf_as_text",
                tool_input,
                lambda: helper.download_pdf_as_text(
                    url=url,
                    max_chars=max_chars,
                    timeout_seconds=timeout_seconds,
                ),
            )
        except Exception as exc:
            return f"ERROR: PDF extraction failed ({exc})"

    return _set_param_descriptions(download_pdf_as_text, {
        "url": "Absolute HTTP/HTTPS URL expected to return a PDF file to download and extract as text.",
        "max_chars": "Optional maximum number of extracted text characters to return inline; the saved text file still contains the full extraction.",
        "timeout_seconds": "Maximum seconds to wait for the PDF download and text extraction.",
    })
