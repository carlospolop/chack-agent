"""ChatGPT Web research agents backed by an authenticated Chrome CDP session.

These researchers deliberately use the ChatGPT web product instead of an API.
They attach to a user-managed Chrome profile, launch one clean conversation, wait
for a terminal UI state, extract the complete answer, and return the normal Chack
researcher JSON contract.  This makes them usable by ResearcherAdministrator and
the shared researcher queue like every other specialist researcher.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from .config import ToolsConfig
from .research_artifacts import cleanup_research_artifacts
from .subagent_config import (
    create_subagent_evidence_dir,
    enforce_prompt_str_or_list_schema,
    normalize_subagent_prompts,
    record_researcher_response,
    run_parallel_subagent_prompts,
)
from .task_steps_manager_state import current_session_id
from .telemetry import current_log_context, run_with_tool_logging

try:
    from agents import function_tool
except ImportError:  # pragma: no cover - mirrors the other researcher modules
    function_tool = None


Mode = Literal["deep", "pro"]


class ChatGPTWebResearchError(RuntimeError):
    """A launch, terminal-state, or extraction failure in ChatGPT Web."""


def _compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ChatGPTWebResearchAgentTool:
    """Launch a Deep Research or Pro request in an existing Chrome session."""

    def __init__(self, config: ToolsConfig, *, mode: Mode):
        if mode not in {"deep", "pro"}:
            raise ValueError(f"Unsupported ChatGPT research mode: {mode}")
        self.config = config
        self.mode = mode

    @property
    def tool_name(self) -> str:
        return f"{self.mode}chatgpt_researcher"

    def _cdp_url(self) -> str:
        configured = str(getattr(self.config, "chatgpt_cdp_url", "") or "").strip()
        return configured or os.environ.get("CHACK_CHATGPT_CDP_URL", "").strip() or "http://127.0.0.1:9226"

    def _timeout_seconds(self) -> int:
        default = 5400 if self.mode == "deep" else 3600
        configured = int(getattr(self.config, "chatgpt_research_timeout_seconds", 0) or 0)
        return max(60, configured or default)

    def _poll_seconds(self) -> int:
        configured = int(getattr(self.config, "chatgpt_research_poll_seconds", 0) or 0)
        return max(2, configured or 15)

    @staticmethod
    def _composer(page):
        selectors = (
            "#prompt-textarea",
            "div.ProseMirror[contenteditable='true']",
            "[contenteditable='true'][data-virtualkeyboard='true']",
            "textarea[placeholder]",
        )
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() and locator.first.is_visible():
                return locator.first
        raise ChatGPTWebResearchError("ChatGPT composer was not found; the Chrome profile may be signed out or the UI changed.")

    @staticmethod
    def _clear_stale_attachments(page) -> None:
        for pattern in (re.compile(r"remove file", re.I), re.compile(r"remove attachment", re.I)):
            locator = page.get_by_role("button", name=pattern)
            for index in range(locator.count()):
                try:
                    if locator.nth(index).is_visible():
                        locator.nth(index).click(timeout=2000)
                except Exception:
                    continue

    def _select_pro(self, page) -> None:
        # Current ChatGPT UI exposes the selected reasoning level as a compact
        # menu button (commonly "Medium"). Prefer that exact visible control so
        # the many conversation-option menus in the sidebar are never inspected.
        mode_button = page.get_by_role(
            "button",
            name=re.compile(r"^\s*(auto|instant|medium|high|extra high|thinking|pro|gpt[- ]?\d.*)\s*$", re.I),
        )
        for index in reversed(range(mode_button.count())):
            try:
                if mode_button.nth(index).is_visible():
                    mode_button.nth(index).click(timeout=5000)
                    opened = True
                    break
            except Exception:
                continue
        else:
            opened = False

        candidates = (
            "button[data-testid='model-switcher-dropdown-button']",
            "button[aria-haspopup='menu']",
        )
        if not opened:
            for selector in candidates:
                buttons = page.locator(selector)
                for index in range(buttons.count()):
                    button = buttons.nth(index)
                    try:
                        label = " ".join(((button.inner_text() or "") + " " + (button.get_attribute("aria-label") or "")).split())
                        if button.is_visible() and ("model" in label.lower() or re.search(r"\b(auto|instant|medium|thinking|pro|gpt)\b", label, re.I)):
                            button.click(timeout=5000)
                            opened = True
                            break
                    except Exception:
                        continue
                if opened:
                    break
        if not opened:
            raise ChatGPTWebResearchError("Could not open the ChatGPT model/mode selector required for Pro mode.")

        pro = page.get_by_text(re.compile(r"^\s*Pro\s*$", re.I))
        for index in reversed(range(pro.count())):
            try:
                if pro.nth(index).is_visible():
                    pro.nth(index).click(timeout=5000)
                    page.wait_for_timeout(500)
                    return
            except Exception:
                continue
        raise ChatGPTWebResearchError("The Pro option was not present in the ChatGPT model selector.")

    @staticmethod
    def _send(page, prompt: str) -> None:
        ChatGPTWebResearchAgentTool._clear_stale_attachments(page)
        composer = ChatGPTWebResearchAgentTool._composer(page)
        composer.click()
        try:
            composer.fill("")
            composer.fill(prompt)
        except Exception:
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
            page.keyboard.insert_text(prompt)

        send_selectors = (
            "button[data-testid='send-button']",
            "button[aria-label*='Send']",
            "button[aria-label*='Enviar']",
        )
        for selector in send_selectors:
            button = page.locator(selector)
            if button.count() and button.first.is_visible() and button.first.is_enabled():
                button.first.click(timeout=5000)
                return
        composer.press("Enter")

    @staticmethod
    def _click_deep_start_if_present(page) -> bool:
        for name in (re.compile(r"^\s*Start\s*$", re.I), re.compile(r"^\s*Iniciar\s*$", re.I)):
            buttons = page.get_by_role("button", name=name)
            for index in range(buttons.count()):
                try:
                    if buttons.nth(index).is_visible() and buttons.nth(index).is_enabled():
                        buttons.nth(index).click(timeout=5000)
                        return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _clean_source_url(url: str) -> str:
        raw = str(url or "").strip()
        if not re.match(r"^https?://", raw, re.I):
            return ""
        try:
            parts = urllib.parse.urlsplit(raw)
            query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
            query = [(key, value) for key, value in query if key.lower() not in {"utm_source", "utm_medium", "utm_campaign"}]
            return urllib.parse.urlunsplit(
                (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query, doseq=True), parts.fragment)
            )
        except Exception:
            return raw

    @classmethod
    def _append_source_links(cls, text: str, links: list[dict[str, str]]) -> str:
        answer = str(text or "").strip()
        sources: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in links or []:
            url = cls._clean_source_url(str(item.get("url") or item.get("href") or ""))
            if not url or url in seen or url in answer:
                continue
            seen.add(url)
            label = " ".join(str(item.get("label") or item.get("text") or "Source").split())
            sources.append((label or "Source", url))
        if not sources:
            return answer
        source_block = "Source links:\n" + "\n".join(f"- {label}: {url}" for label, url in sources)
        lines = answer.rstrip().splitlines()
        terminal_marker = ""
        if lines and re.fullmatch(r"[A-Z][A-Z0-9_]{5,}", lines[-1].strip()):
            terminal_marker = lines.pop().strip()
        combined = "\n".join(lines).rstrip() + "\n\n" + source_block
        if terminal_marker:
            combined += "\n\n" + terminal_marker
        return combined.strip()

    @classmethod
    def _element_text_with_links(cls, element) -> str:
        text = element.inner_text(timeout=3000).strip()
        links: list[dict[str, str]] = []
        anchors = element.locator("a[href]")
        for index in range(anchors.count()):
            try:
                anchor = anchors.nth(index)
                links.append(
                    {
                        "label": (anchor.inner_text(timeout=1000) or "").strip(),
                        "url": str(anchor.get_attribute("href") or ""),
                    }
                )
            except Exception:
                continue
        return cls._append_source_links(text, links)

    @classmethod
    def _longest_answer(cls, page) -> str:
        candidates: list[str] = []
        assistant = page.locator('[data-message-author-role="assistant"]')
        for index in range(assistant.count()):
            try:
                text = cls._element_text_with_links(assistant.nth(index))
                if text:
                    candidates.append(text)
            except Exception:
                continue

        # Deep Research is often rendered in an OOPIF and then a nested #root
        # iframe. Playwright exposes both as Frame objects, so inspect every frame.
        for frame in page.frames:
            try:
                parent_url = frame.parent_frame.url if frame.parent_frame else ""
                is_research_frame = (
                    frame is not page.main_frame
                    and (
                        "deep_research" in frame.url
                        or "oaiusercontent.com" in frame.url
                        or "deep_research" in parent_url
                        or "oaiusercontent.com" in parent_url
                    )
                )
                if not is_research_frame:
                    continue
                body = frame.locator("body")
                if body.count():
                    text = cls._element_text_with_links(body)
                    if text:
                        candidates.append(text)
            except Exception:
                continue
        return max(candidates, key=len, default="").strip()

    @staticmethod
    def _is_running(page) -> bool:
        running_patterns = (
            re.compile(r"stop (generating|research|thinking)", re.I),
            re.compile(r"detener (la )?(generaci[oó]n|investigaci[oó]n)", re.I),
        )
        for pattern in running_patterns:
            if page.get_by_role("button", name=pattern).count():
                return True
        return False

    def _deep_connector_target(self, parent_target_id: str, timeout_seconds: int = 30) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self._cdp_url().rstrip('/')}/json/list", timeout=5) as response:
                    targets = json.load(response)
                connector = next(
                    (
                        target
                        for target in targets
                        if target.get("type") == "iframe"
                        and target.get("parentId") == parent_target_id
                        and "connector_openai_deep_research" in str(target.get("url") or "")
                        and target.get("webSocketDebuggerUrl")
                    ),
                    None,
                )
                if connector:
                    return connector
            except Exception:
                pass
            time.sleep(1)
        raise ChatGPTWebResearchError(
            "The sent request did not create a verified Deep Research connector target; refusing to count it as Deep Research."
        )

    def _target_url(self, target_id: str, fallback: str = "") -> str:
        try:
            with urllib.request.urlopen(f"{self._cdp_url().rstrip('/')}/json/list", timeout=5) as response:
                targets = json.load(response)
            target = next((row for row in targets if row.get("id") == target_id), None)
            url = str((target or {}).get("url") or "").strip()
            if url:
                return url
        except Exception:
            pass
        return fallback

    @staticmethod
    def _deep_connector_state(websocket_url: str, *, click_start: bool = False) -> dict[str, Any]:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - dependency comes via openai-agents
            raise ChatGPTWebResearchError("The websockets package is required to monitor the Deep Research connector.") from exc

        expression = r"""(()=>{
const root=document.querySelector('#root'),doc=root?.contentDocument;
if(!doc)return{text:'',textLen:0,buttons:[],links:[],hasStop:false,completed:false,planning:false,clickedStart:false};
const buttons=[...doc.querySelectorAll('button')];
let clickedStart=false;
if(CLICK_START){const start=buttons.find(b=>/^\s*(Start|Iniciar)\s*$/i.test((b.innerText||b.getAttribute('aria-label')||'')));if(start){start.click();clickedStart=true;}}
const text=doc.body?.innerText||root?.innerText||'';
const labels=buttons.map(b=>(b.innerText||b.getAttribute('aria-label')||'').trim()).filter(Boolean);
const links=[...doc.querySelectorAll('a[href]')].map(a=>({label:(a.innerText||a.getAttribute('aria-label')||'Source').trim(),url:a.href||''}));
const hasStop=labels.some(x=>/Stop research|Detener.*investigaci/i.test(x));
const completed=/Research completed|Investigaci[oó]n completada/i.test(text)||(/\bSources\b|\bFuentes\b/i.test(text)&&text.length>1200&&!hasStop);
const planning=labels.some(x=>/^\s*(Start|Iniciar)\s*$/i.test(x));
return{text,textLen:text.length,buttons:labels,links,hasStop,completed,planning,clickedStart};
})()""".replace("CLICK_START", "true" if click_start else "false")
        with connect(websocket_url, origin=None, open_timeout=10, close_timeout=5) as websocket:
            websocket.send(
                json.dumps(
                    {
                        "id": 1,
                        "method": "Runtime.evaluate",
                        "params": {"expression": expression, "returnByValue": True, "timeout": 30000},
                    }
                )
            )
            raw = json.loads(websocket.recv(timeout=35))
        try:
            return raw["result"]["result"]["value"]
        except (KeyError, TypeError) as exc:
            raise ChatGPTWebResearchError(f"Could not evaluate the Deep Research connector target: {raw}") from exc

    def _wait_and_extract_deep(self, connector: dict[str, Any]) -> str:
        websocket_url = str(connector.get("webSocketDebuggerUrl") or "")
        deadline = time.monotonic() + self._timeout_seconds()
        previous = ""
        stable_polls = 0
        while time.monotonic() < deadline:
            state = self._deep_connector_state(websocket_url, click_start=True)
            answer = self._append_source_links(
                str(state.get("text") or "").strip(),
                list(state.get("links") or []),
            )
            if answer and answer == previous:
                stable_polls += 1
            else:
                previous = answer
                stable_polls = 0
            if bool(state.get("completed")) and not bool(state.get("hasStop")) and len(answer) >= 1200 and stable_polls >= 1:
                return answer
            time.sleep(self._poll_seconds())
        raise ChatGPTWebResearchError(
            f"ChatGPT deep request did not reach an extractable terminal state within {self._timeout_seconds()} seconds."
        )

    def _wait_and_extract(self, page) -> str:
        deadline = time.monotonic() + self._timeout_seconds()
        previous = ""
        stable_polls = 0
        while time.monotonic() < deadline:
            if self.mode == "deep":
                self._click_deep_start_if_present(page)
            answer = self._longest_answer(page)
            running = self._is_running(page)
            if answer and answer == previous:
                stable_polls += 1
            else:
                stable_polls = 0
                previous = answer
            # Two identical polls plus no stop control avoids saving a streaming
            # partial answer. Deep reports need a larger minimum than Pro answers.
            min_chars = 1200 if self.mode == "deep" else 200
            if len(answer) >= min_chars and not running and stable_polls >= 2:
                return answer
            page.wait_for_timeout(self._poll_seconds() * 1000)
        raise ChatGPTWebResearchError(
            f"ChatGPT {self.mode} request did not reach an extractable terminal state within {self._timeout_seconds()} seconds."
        )

    def _browser_research(self, prompt: str) -> tuple[str, str, dict[str, Any]]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - packaging error
            raise ChatGPTWebResearchError("Playwright is required for ChatGPT Web researchers.") from exc

        started_at = time.time()
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(self._cdp_url(), timeout=30000)
            if not browser.contexts:
                raise ChatGPTWebResearchError("The Chrome CDP endpoint has no browser context.")
            page = browser.contexts[0].new_page()
            try:
                page.goto("https://chatgpt.com/deep-research" if self.mode == "deep" else "https://chatgpt.com/", wait_until="domcontentloaded", timeout=60000)
                # domcontentloaded fires before the authenticated React app has
                # hydrated. Wait for the real composer, otherwise concurrent
                # launches can falsely look signed-out or mode-less.
                page.wait_for_selector(
                    "#prompt-textarea, div.ProseMirror[contenteditable='true']",
                    state="visible",
                    timeout=30000,
                )
                page.wait_for_timeout(1500)
                if self.mode == "deep":
                    marker = page.get_by_text(
                        re.compile(r"Ask a complex question|Get a full report|Deep research|investigaci[oó]n profunda|informe detallado", re.I)
                    )
                    try:
                        marker.first.wait_for(state="visible", timeout=15000)
                    except Exception:
                        pass
                    body = page.locator("body").inner_text(timeout=5000)
                    if not re.search(r"deep research|full report|detailed report|investigaci[oó]n profunda|informe detallado", body, re.I):
                        raise ChatGPTWebResearchError("The /deep-research route did not expose Deep Research mode; refusing to send a normal chat.")
                else:
                    self._select_pro(page)
                self._send(page, prompt)
                page.wait_for_timeout(1000)
                if self.mode == "deep":
                    cdp_session = page.context.new_cdp_session(page)
                    try:
                        target_info = cdp_session.send("Target.getTargetInfo")["targetInfo"]
                    finally:
                        cdp_session.detach()
                    connector = self._deep_connector_target(str(target_info.get("targetId") or ""))
                    answer = self._wait_and_extract_deep(connector)
                    conversation_url = self._target_url(str(connector.get("parentId") or ""), page.url)
                else:
                    answer = self._wait_and_extract(page)
                    conversation_url = page.url
                url = conversation_url
                return answer, url, {
                    "mode": self.mode,
                    "conversation_url": url,
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "answer_chars": len(answer),
                    "terminal_state": "extracted",
                }
            finally:
                try:
                    page.close()
                except Exception:
                    pass

    def _run_single(self, prompt: str, *, save_artifacts: bool) -> str:
        ctx = current_log_context()
        evidence_dir = create_subagent_evidence_dir(self.tool_name, str(ctx.get("session_id") or current_session_id() or ""))
        root = Path(evidence_dir)
        root.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {"mode": self.mode, "terminal_state": "error"}
        try:
            answer, conversation_url, metadata = self._browser_research(prompt)
            filename = f"chatgpt-{self.mode}-response.md"
            (root / filename).write_text(answer, encoding="utf-8")
            (root / "chatgpt-request.md").write_text(prompt, encoding="utf-8")
            (root / "chatgpt-run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            payload: dict[str, Any] = {
                "research_worked": True,
                "failure_reason": "",
                "final_research_review": answer,
                "evidence_data_path": evidence_dir if save_artifacts else "",
                "key_artifacts": [],
                "tool_call_counts": {"chatgpt_web": 1},
                "total_tool_calls": 1,
            }
            if save_artifacts:
                payload["key_artifacts"] = [
                    {
                        "filename": filename,
                        "source_url": conversation_url,
                        "description": "Complete extracted ChatGPT Web response from the requested research mode, preserved as the primary research evidence and synthesis input.",
                    },
                    {
                        "filename": "chatgpt-request.md",
                        "source_url": conversation_url,
                        "description": "Exact prompt submitted to ChatGPT Web, preserved to make the research request, scope, and provenance independently auditable.",
                    },
                    {
                        "filename": "chatgpt-run.json",
                        "source_url": conversation_url,
                        "description": "Run metadata containing the selected ChatGPT mode, conversation URL, timestamps, terminal extraction state, and extracted answer length.",
                    },
                ]
            return _compact(payload)
        except Exception as exc:
            metadata.update({"finished_at": time.time(), "error": f"{type(exc).__name__}: {exc}"})
            try:
                root.mkdir(parents=True, exist_ok=True)
                (root / "chatgpt-run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            payload = {
                "research_worked": False,
                "failure_reason": str(exc),
                "final_research_review": "",
                "evidence_data_path": evidence_dir if save_artifacts else "",
                "key_artifacts": ([{
                    "filename": "chatgpt-run.json",
                    "source_url": "",
                    "description": "Terminal failure metadata for the ChatGPT Web research attempt, retained so the launch or extraction problem can be audited and retried.",
                }] if save_artifacts else []),
                "tool_call_counts": {"chatgpt_web": 1},
                "total_tool_calls": 1,
            }
            return _compact(payload)
        finally:
            cleanup_research_artifacts(evidence_dir, save_artifacts=save_artifacts)

    def run(self, prompt: str | list[str], save_artifacts: bool = False) -> str:
        prompts, error = normalize_subagent_prompts(prompt, min_chars=100, max_prompts=3)
        if error:
            return error
        return run_parallel_subagent_prompts(
            prompts,
            lambda item: self._run_single(item, save_artifacts=save_artifacts),
        )


def _make_tool(helper: ChatGPTWebResearchAgentTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    mode_label = "Deep Research" if helper.mode == "deep" else "Pro mode"
    description = f"""Run one authenticated ChatGPT Web {mode_label} research agent in a clean Chrome tab and wait for the complete extracted response.

Use it for an independent ChatGPT {mode_label} research or reasoning pass. Give a self-contained prompt with the topic, scope, source/evidence requirements, uncertainties to test, and expected output. The browser must already be signed in to ChatGPT and reachable through the configured Chrome CDP endpoint.

Args:
    prompt: One detailed research prompt, or a list of up to 3 prompts to run independently.
    save_artifacts: Preserve the exact prompt, complete response, run metadata, and conversation URL in the research evidence folder.

Output: Standard Chack researcher JSON with terminal worked/failure status, the complete extracted review, and preserved artifact metadata when requested.
"""

    def research(prompt: str | list[str], save_artifacts: bool = False) -> str:
        try:
            return run_with_tool_logging(
                helper.tool_name,
                {"prompt": prompt, "save_artifacts": save_artifacts},
                lambda: _run_and_record(helper.tool_name, helper.run(prompt, save_artifacts=save_artifacts)),
            )
        except Exception as exc:
            return f"ERROR: {helper.tool_name} failed ({exc})"

    tool = enforce_prompt_str_or_list_schema(
        function_tool(research, name_override=helper.tool_name, description_override=description)
    )
    properties = (getattr(tool, "params_json_schema", {}) or {}).get("properties", {})
    if "prompt" in properties:
        properties["prompt"]["description"] = "One detailed research prompt, or a list of up to three independent detailed prompts."
    if "save_artifacts" in properties:
        properties["save_artifacts"]["description"] = "Preserve the exact request, response, run metadata, and conversation URL when true."
    return tool


def _run_and_record(tool_name: str, output: str) -> str:
    record_researcher_response(tool_name, output)
    return output


def get_deepchatgpt_researcher_tool(config: ToolsConfig):
    return _make_tool(ChatGPTWebResearchAgentTool(config, mode="deep"))


def get_prochatgpt_researcher_tool(config: ToolsConfig):
    return _make_tool(ChatGPTWebResearchAgentTool(config, mode="pro"))
