import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from uuid import uuid4

try:
    from agents import function_tool
except ImportError:
    function_tool = None

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:
    PlaywrightError = Exception
    PlaywrightTimeoutError = TimeoutError
    sync_playwright = None

from .config import ToolsConfig
from .formatting import _truncate
from .research_artifacts import record_research_artifact, research_artifacts_root
from .telemetry import run_with_tool_logging


_WAIT_UNTIL_VALUES = {"commit", "domcontentloaded", "load", "networkidle"}


def _sanitize_filename(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]", "_", value or "").strip("._")
    return text or fallback


@lru_cache(maxsize=1)
def is_playwright_available() -> bool:
    if sync_playwright is None:
        return False
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


class PlaywrightFetchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def fetch_page(
        self,
        url: str,
        wait_until: str = "networkidle",
        wait_for_selector: str = "",
        timeout_seconds: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        if not is_playwright_available():
            return (
                "ERROR: Playwright is not available. Install the Python playwright package "
                "and browser binaries, then enable playwright_enabled."
            )

        target = str(url or "").strip()
        if not target:
            return "ERROR: url cannot be empty"
        parsed = urlparse(target)
        if parsed.scheme not in {"http", "https"}:
            return "ERROR: url must start with http:// or https://"

        normalized_wait_until = str(wait_until or "networkidle").strip().lower()
        if normalized_wait_until not in _WAIT_UNTIL_VALUES:
            allowed = ", ".join(sorted(_WAIT_UNTIL_VALUES))
            return f"ERROR: wait_until must be one of {allowed}"

        timeout_ms = max(
            1,
            int(timeout_seconds or self.config.playwright_timeout_seconds or 30),
        ) * 1000
        output_limit = max(
            1,
            int(max_chars or self.config.playwright_max_output_chars or 12000),
        )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=bool(self.config.playwright_headless),
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/142.0.0.0 Safari/537.36"
                ),
                ignore_https_errors=True,
            )
            page = context.new_page()
            try:
                page.goto(target, wait_until=normalized_wait_until, timeout=timeout_ms)
                selector = str(wait_for_selector or "").strip()
                if selector:
                    page.wait_for_selector(selector, timeout=timeout_ms)

                title = (page.title() or "").strip()
                final_url = (page.url or target).strip()
                main_text = (page.locator("body").inner_text(timeout=timeout_ms) or "").strip()
                html = page.content()
            except PlaywrightTimeoutError:
                return "ERROR: Playwright page load timed out"
            except PlaywrightError as exc:
                return f"ERROR: Playwright failed ({exc})"
            finally:
                context.close()
                browser.close()

        if not main_text:
            main_text = "(page body text was empty)"

        evidence_root = research_artifacts_root()
        output_dir = (
            Path(evidence_root) / "playwright"
            if evidence_root
            else Path("/tmp/chack-playwright")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        url_label = _sanitize_filename(
            os.path.basename(urlparse(final_url).path or "") or title,
            "page",
        )
        token = uuid4().hex
        text_path = output_dir / f"{url_label}_{token}.txt"
        html_path = output_dir / f"{url_label}_{token}.html"
        text_path.write_text(main_text, encoding="utf-8")
        html_path.write_text(html, encoding="utf-8")
        record_research_artifact(
            text_path,
            source_url=final_url,
            provenance=f"playwright_fetch requested={target}; title={title}",
            tool="playwright_fetch",
            kind="playwright",
            label=final_url,
        )
        record_research_artifact(
            html_path,
            source_url=final_url,
            provenance=f"playwright_fetch requested={target}; title={title}",
            tool="playwright_fetch",
            kind="playwright",
            label=final_url,
        )

        excerpt = _truncate(main_text, output_limit)
        lines = [
            "SUCCESS: Fetched page with Playwright.",
            f"Requested URL: {target}",
            f"Final URL: {final_url}",
            f"Title: {title or '(no title)'}",
            f"Wait until: {normalized_wait_until}",
            f"Text characters: {len(main_text)}",
        ]
        if str(wait_for_selector or "").strip():
            lines.append(f"Waited for selector: {wait_for_selector.strip()}")
        lines.extend(
            [
                f"Saved text: {text_path}",
                f"Saved HTML: {html_path}",
                "",
                "Visible text excerpt:",
                excerpt,
            ]
        )
        return "\n".join(lines)


def get_playwright_fetch_tool(helper: PlaywrightFetchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="playwright_fetch")
    def playwright_fetch(
        url: str,
        wait_until: str = "networkidle",
        wait_for_selector: str = "",
        timeout_seconds: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """Open a web page in a real browser and extract the rendered page text.

        Use this when normal search results are not enough or when a site relies on JavaScript.
        Prefer it for reading a specific page, verifying dynamic content, and saving the rendered
        page text/HTML for later inspection.

        Args:
            url: Absolute HTTP/HTTPS URL to open in the browser.
            wait_until: Browser navigation wait condition such as load, domcontentloaded, networkidle, or commit.
            wait_for_selector: Optional CSS selector that must appear before extraction.
            timeout_seconds: Optional browser navigation timeout; defaults to tool configuration.
            max_chars: Optional maximum number of extracted text characters to return inline.

        Output: Returns SUCCESS/ERROR text with page title, final URL, extracted rendered text, and local artifact paths for saved HTML/text when available.
        """
        tool_input = {
            "url": url,
            "wait_until": wait_until,
            "wait_for_selector": wait_for_selector,
            "timeout_seconds": timeout_seconds,
            "max_chars": max_chars,
        }
        try:
            return run_with_tool_logging(
                "playwright_fetch",
                tool_input,
                lambda: helper.fetch_page(
                    url=url,
                    wait_until=wait_until,
                    wait_for_selector=wait_for_selector,
                    timeout_seconds=timeout_seconds,
                    max_chars=max_chars,
                ),
            )
        except Exception as exc:
            return f"ERROR: Playwright fetch failed ({exc})"

    return playwright_fetch
