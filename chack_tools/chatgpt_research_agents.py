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
import uuid
from pathlib import Path
from typing import Any, Literal

from .chatgpt_async_client import ChatGPTAsyncApiClient
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


Mode = Literal["deep", "pro", "xhigh"]

CHATGPT_PRO_OUTPUT_TIMEOUT_SECONDS = 90 * 60
CHATGPT_XHIGH_OUTPUT_TIMEOUT_SECONDS = 90 * 60
CHATGPT_DEEP_OUTPUT_TIMEOUT_SECONDS = 75 * 60
_MODE_TOOL_NAMES: dict[Mode, str] = {
    "deep": "deepchatgpt_researcher",
    "pro": "prochatgpt_researcher",
    "xhigh": "chatgptxhigh",
}
_REMOTE_METADATA_FIELDS = {
    "mode",
    "started_at",
    "finished_at",
    "answer_chars",
    "terminal_state",
    "stage",
    "forced_answer",
    "output_timeout_seconds",
    "execution_backend",
}


def resolve_chatgpt_timeout_seconds(config: ToolsConfig, mode: Mode) -> int:
    """Return the total output deadline for one ChatGPT browser request.

    Mode-specific configuration is authoritative. The old shared setting is
    retained as a compatibility fallback for callers that have not migrated.
    """
    field_name = {
        "deep": "chatgpt_deep_timeout_seconds",
        "pro": "chatgpt_pro_timeout_seconds",
        "xhigh": "chatgpt_xhigh_timeout_seconds",
    }[mode]
    configured = getattr(config, field_name, None)
    if configured is not None and int(configured or 0) > 0:
        return max(60, int(configured))
    legacy = int(getattr(config, "chatgpt_research_timeout_seconds", 0) or 0)
    if legacy > 0:
        return max(60, legacy)
    return {
        "deep": CHATGPT_DEEP_OUTPUT_TIMEOUT_SECONDS,
        "pro": CHATGPT_PRO_OUTPUT_TIMEOUT_SECONDS,
        "xhigh": CHATGPT_XHIGH_OUTPUT_TIMEOUT_SECONDS,
    }[mode]


class ChatGPTWebResearchError(RuntimeError):
    """A launch, terminal-state, or extraction failure in ChatGPT Web."""


def _compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ChatGPTWebResearchAgentTool:
    """Launch a Deep Research, Pro, or Extra High request in Chrome."""

    def __init__(self, config: ToolsConfig, *, mode: Mode):
        if mode not in {"deep", "pro", "xhigh"}:
            raise ValueError(f"Unsupported ChatGPT research mode: {mode}")
        self.config = config
        self.mode: Mode = mode

    @property
    def tool_name(self) -> str:
        return _MODE_TOOL_NAMES[self.mode]

    def _cdp_url(self) -> str:
        configured = str(getattr(self.config, "chatgpt_cdp_url", "") or "").strip()
        return configured or os.environ.get("CHACK_CHATGPT_CDP_URL", "").strip() or "http://127.0.0.1:9226"

    def _timeout_seconds(self) -> int:
        return resolve_chatgpt_timeout_seconds(self.config, self.mode)

    def _poll_seconds(self) -> int:
        configured = int(getattr(self.config, "chatgpt_research_poll_seconds", 0) or 0)
        return max(2, configured or 15)

    def _force_answer_grace_seconds(self) -> int:
        configured = int(getattr(self.config, "chatgpt_force_answer_grace_seconds", 0) or 0)
        return max(60, configured or 300)

    def _execution_backend(self) -> str:
        configured = str(getattr(self.config, "chatgpt_execution_backend", "") or "").strip().lower()
        backend = configured or os.environ.get("CHACK_CHATGPT_EXECUTION_BACKEND", "").strip().lower() or "auto"
        if backend not in {"auto", "local", "remote"}:
            raise ChatGPTWebResearchError(f"Unsupported ChatGPT execution backend: {backend}")
        if backend == "auto":
            # The presence of either broker setting means this is a remote client.
            # A partial deployment must fail closed instead of touching local CDP.
            return "remote" if self._async_api_url() or self._async_api_secret() else "local"
        return backend

    def _async_api_url(self) -> str:
        configured = str(getattr(self.config, "chatgpt_async_api_url", "") or "").strip()
        return configured or os.environ.get("CHACK_CHATGPT_ASYNC_API_URL", "").strip()

    def _async_api_secret(self) -> str:
        configured = str(getattr(self.config, "chatgpt_async_api_secret", "") or "").strip()
        return configured or os.environ.get("CHACK_CHATGPT_ASYNC_API_SECRET", "").strip()

    def _async_poll_seconds(self) -> int:
        configured = int(getattr(self.config, "chatgpt_async_poll_seconds", 0) or 0)
        environment = int(os.environ.get("CHACK_CHATGPT_ASYNC_POLL_SECONDS", "0") or 0)
        return max(2, configured or environment or 10)

    def _async_max_wait_seconds(self) -> int:
        configured = int(getattr(self.config, "chatgpt_async_max_wait_seconds", 0) or 0)
        environment = int(os.environ.get("CHACK_CHATGPT_ASYNC_MAX_WAIT_SECONDS", "0") or 0)
        return max(self._timeout_seconds(), configured or environment or 10800)

    def _async_client(self) -> ChatGPTAsyncApiClient:
        url = self._async_api_url()
        secret = self._async_api_secret()
        if not url or not secret:
            raise ChatGPTWebResearchError(
                "Remote ChatGPT execution requires CHACK_CHATGPT_ASYNC_API_URL and CHACK_CHATGPT_ASYNC_API_SECRET."
            )
        request_timeout = int(getattr(self.config, "chatgpt_async_request_timeout_seconds", 0) or 0) or 30
        return ChatGPTAsyncApiClient(url, secret, request_timeout_seconds=request_timeout)

    def _remote_research(
        self,
        prompt: str,
        *,
        run_state_path: Path | None = None,
        partial_path: Path | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """Submit through the cloud broker and poll without touching local CDP."""
        client = self._async_client()
        output_timeout_seconds = self._timeout_seconds()
        submitted = client.submit(
            mode=self.mode,
            prompt=prompt,
            idempotency_key=str(uuid.uuid4()),
            output_timeout_seconds=output_timeout_seconds,
        )
        job_id = str(submitted.get("job_id") or "")
        if not job_id:
            raise ChatGPTWebResearchError("ChatGPT async API did not return a job id.")

        started = time.time()
        deadline = time.monotonic() + self._async_max_wait_seconds()
        metadata: dict[str, Any] = {
            "mode": self.mode,
            "execution_backend": "remote",
            "remote_job_id": job_id,
            "submitted_at": started,
            "terminal_state": "queued",
            "output_timeout_seconds": output_timeout_seconds,
        }
        self._write_json(run_state_path, metadata)
        last_stage = "queued"
        last_chars = 0
        while True:
            if time.monotonic() >= deadline:
                try:
                    client.cancel(job_id)
                except Exception:
                    pass
                metadata.update({"terminal_state": "timeout", "finished_at": time.time()})
                self._write_json(run_state_path, metadata)
                raise ChatGPTWebResearchError(
                    f"Remote ChatGPT {self.mode} job exceeded the configured client wait deadline."
                )

            status_payload = client.status(job_id)
            status = str(status_payload.get("status") or "").upper()
            stage = str(status_payload.get("stage") or status or "queued").lower()
            answer_chars = int(status_payload.get("answer_chars") or 0)
            if stage != last_stage or answer_chars != last_chars:
                self._emit_progress(f"remote_{stage}", answer_chars=answer_chars, running=status not in {"SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED", "EXPIRED"})
                last_stage, last_chars = stage, answer_chars
            metadata.update(
                {
                    "remote_status": status,
                    "terminal_state": stage,
                    "answer_chars": answer_chars,
                    "last_polled_at": time.time(),
                }
            )
            self._write_json(run_state_path, metadata)

            if status in {"SUCCEEDED", "FAILED", "TIMED_OUT", "CANCELLED", "EXPIRED"}:
                result_payload = client.result(job_id)
                answer = str(result_payload.get("result") or "")
                partial = str(result_payload.get("partial_result") or "")
                raw_metadata = result_payload.get("metadata")
                untrusted_metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
                remote_metadata = {
                    key: value
                    for key, value in untrusted_metadata.items()
                    if key in _REMOTE_METADATA_FIELDS and isinstance(value, (str, int, float, bool))
                }
                # Remote clients never receive or propagate authenticated browser
                # conversation URLs, even if a compromised broker tried to add one.
                conversation_url = ""
                metadata.update(remote_metadata)
                metadata.update(
                    {
                        "remote_status": status,
                        "terminal_state": "extracted" if status == "SUCCEEDED" else status.lower(),
                        "finished_at": time.time(),
                        "answer_chars": len(answer or partial),
                    }
                )
                self._write_json(run_state_path, metadata)
                if status == "SUCCEEDED" and answer.strip():
                    self._emit_progress("remote_extracted", answer_chars=len(answer), running=False)
                    return answer, conversation_url, metadata
                if partial:
                    self._write_partial(partial_path, partial)
                error_code = str(result_payload.get("error_code") or status or "remote_failed")
                error_message = str(result_payload.get("error_message") or "")
                raise ChatGPTWebResearchError(
                    f"Remote ChatGPT {self.mode} job ended as {status} ({error_code})"
                    + (f": {error_message}" if error_message else "")
                )
            time.sleep(self._async_poll_seconds())

    def _research(
        self,
        prompt: str,
        *,
        run_state_path: Path | None = None,
        partial_path: Path | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        if self._execution_backend() == "remote":
            return self._remote_research(prompt, run_state_path=run_state_path, partial_path=partial_path)
        return self._browser_research(prompt, run_state_path=run_state_path, partial_path=partial_path)

    @staticmethod
    def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            merged: dict[str, Any] = {}
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict):
                        merged.update(existing)
                except Exception:
                    pass
            merged.update(payload)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        except Exception:
            pass

    @staticmethod
    def _write_partial(path: Path | None, text: str) -> None:
        if path is None or not text:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        except Exception:
            pass

    def _emit_progress(self, stage: str, *, answer_chars: int = 0, running: bool = True, forced_answer: bool = False) -> None:
        """Refresh async-job activity without counting a new researcher tool call."""
        callback = current_log_context().get("_chack_tool_progress_callback")
        if not callable(callback):
            return
        try:
            callback(
                "research_progress",
                {
                    "tool": self.tool_name,
                    "tool_start_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stage": stage,
                    "answer_chars": int(answer_chars or 0),
                    "running": bool(running),
                    "forced_answer": bool(forced_answer),
                },
            )
        except Exception:
            pass

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

    def _select_reasoning_mode(self, page) -> None:
        if self.mode not in {"pro", "xhigh"}:
            raise ChatGPTWebResearchError(
                f"ChatGPT selector is not valid for mode {self.mode}."
            )
        target_label = "Pro" if self.mode == "pro" else "Extra High"
        target_pattern = re.compile(rf"^\s*{re.escape(target_label)}\s*$", re.I)
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
            raise ChatGPTWebResearchError(
                f"Could not open the ChatGPT model/mode selector required for {target_label} mode."
            )

        options = page.get_by_role("menuitemradio", name=target_pattern)
        if not options.count():
            options = page.get_by_text(target_pattern)
        for index in reversed(range(options.count())):
            try:
                if options.nth(index).is_visible():
                    options.nth(index).click(timeout=5000)
                    page.wait_for_timeout(500)
                    break
            except Exception:
                continue
        else:
            raise ChatGPTWebResearchError(
                f"The {target_label} option was not present in the ChatGPT model selector."
            )

        # Do not trust the click alone: the mode must be visibly selected before
        # a potentially expensive prompt is submitted. This also catches a stale
        # menu or account-level UI change instead of silently using another mode.
        selected = page.get_by_role("button", name=target_pattern)
        for index in reversed(range(selected.count())):
            try:
                if selected.nth(index).is_visible():
                    return
            except Exception:
                continue
        raise ChatGPTWebResearchError(
            f"ChatGPT did not confirm {target_label} mode after selection; refusing to send."
        )

    def _select_pro(self, page) -> None:
        """Backward-compatible helper retained for integrations/tests."""
        if self.mode != "pro":
            raise ChatGPTWebResearchError("_select_pro is only valid for Pro mode.")
        self._select_reasoning_mode(page)

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

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Remove ChatGPT Deep Research counter/citation UI noise from extracted prose.

        The Deep Research iframe sometimes renders animated 0-9 counters and
        citation superscripts as standalone lines. Only activate the cleanup when
        many such lines are present so legitimate short numbered answers remain
        untouched.
        """
        raw = str(text or "").strip()
        lines = raw.splitlines()
        short_number_lines = sum(1 for line in lines if re.fullmatch(r"\s*\d{1,2}\s*", line))
        if short_number_lines < 10:
            return raw
        ui_labels = {"citations ·", "searches", "text", "copy"}
        cleaned = [
            line for line in lines
            if not re.fullmatch(r"\s*\d{1,2}\s*", line)
            and line.strip().lower() not in ui_labels
        ]
        normalized: list[str] = []
        for line in cleaned:
            if not line.strip() and normalized and not normalized[-1].strip():
                continue
            normalized.append(line)
        return "\n".join(normalized).strip()

    @classmethod
    def _element_text_with_links(cls, element) -> str:
        text = cls._clean_extracted_text(element.inner_text(timeout=3000))
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
    def _click_answer_now_if_present(page) -> bool:
        patterns = (
            re.compile(r"^\s*Answer now\s*$", re.I),
            re.compile(r"^\s*Responder ahora\s*$", re.I),
            re.compile(r"^\s*Answer with current findings\s*$", re.I),
        )
        for pattern in patterns:
            buttons = page.get_by_role("button", name=pattern)
            for index in range(buttons.count()):
                try:
                    button = buttons.nth(index)
                    if button.is_visible() and button.is_enabled():
                        button.click(timeout=5000)
                        return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _is_running(page) -> bool:
        running_patterns = (
            re.compile(r"stop (generating|research|thinking|answering)", re.I),
            re.compile(r"detener (la )?(generaci[oó]n|investigaci[oó]n|respuesta)", re.I),
            re.compile(r"^\s*Answer now\s*$", re.I),
            re.compile(r"^\s*Responder ahora\s*$", re.I),
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
                        and re.search(
                            r"connector[-_]openai[-_]deep[-_]research",
                            str(target.get("url") or ""),
                            re.I,
                        )
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

    def _wait_and_extract_deep(
        self,
        connector: dict[str, Any],
        *,
        partial_path: Path | None = None,
        run_state_path: Path | None = None,
    ) -> str:
        websocket_url = str(connector.get("webSocketDebuggerUrl") or "")
        timeout_seconds = self._timeout_seconds()
        deadline = time.monotonic() + timeout_seconds
        previous = ""
        stable_polls = 0
        last_progress_at = 0.0
        while time.monotonic() < deadline:
            state = self._deep_connector_state(websocket_url, click_start=True)
            answer = self._append_source_links(
                self._clean_extracted_text(str(state.get("text") or "")),
                list(state.get("links") or []),
            )
            if answer and answer == previous:
                stable_polls += 1
            else:
                previous = answer
                stable_polls = 0
                self._write_partial(partial_path, answer)
            now = time.monotonic()
            if now - last_progress_at >= 60:
                self._emit_progress(
                    "waiting_for_deep_research",
                    answer_chars=len(answer),
                    running=bool(state.get("hasStop") or not state.get("completed")),
                )
                last_progress_at = now
            self._write_json(
                run_state_path,
                {
                    "mode": self.mode,
                    "terminal_state": "running",
                    "updated_at": time.time(),
                    "answer_chars": len(answer),
                    "output_timeout_seconds": timeout_seconds,
                },
            )
            if bool(state.get("completed")) and not bool(state.get("hasStop")) and len(answer) >= 1200 and stable_polls >= 1:
                return answer
            remaining = max(0.0, deadline - time.monotonic())
            if remaining > 0:
                time.sleep(min(float(self._poll_seconds()), remaining))
        raise ChatGPTWebResearchError(
            f"ChatGPT deep request did not reach an extractable terminal state within its "
            f"{timeout_seconds}-second total output deadline."
        )

    def _wait_and_extract(
        self,
        page,
        *,
        partial_path: Path | None = None,
        run_state_path: Path | None = None,
    ) -> str:
        timeout_seconds = self._timeout_seconds()
        started_monotonic = time.monotonic()
        hard_deadline = started_monotonic + timeout_seconds
        force_window = min(self._force_answer_grace_seconds(), timeout_seconds)
        force_at = hard_deadline - force_window
        previous = ""
        stable_polls = 0
        last_progress_at = 0.0
        forced_answer = False
        force_baseline = ""
        while True:
            now = time.monotonic()
            # Pro's Answer-now recovery window is inside the total output
            # deadline. It must never extend a broken browser request beyond the
            # configured total output deadline.
            if (
                self.mode in {"pro", "xhigh"}
                and not forced_answer
                and now >= force_at
                and self._click_answer_now_if_present(page)
            ):
                forced_answer = True
                force_baseline = previous
                stable_polls = 0
                self._emit_progress(
                    "forced_answer_requested",
                    answer_chars=len(previous),
                    running=True,
                    forced_answer=True,
                )
                self._write_json(
                    run_state_path,
                    {
                        "mode": self.mode,
                        "terminal_state": "forcing_answer",
                        "updated_at": time.time(),
                        "answer_chars": len(previous),
                        "forced_answer": True,
                        "output_timeout_seconds": timeout_seconds,
                    },
                )
            if self.mode == "deep":
                self._click_deep_start_if_present(page)
            answer = self._longest_answer(page)
            running = self._is_running(page)
            if answer and answer == previous:
                stable_polls += 1
            else:
                stable_polls = 0
                previous = answer
                self._write_partial(partial_path, answer)
            now = time.monotonic()
            if now - last_progress_at >= 60:
                self._emit_progress(
                    "waiting_for_forced_answer" if forced_answer else "waiting_for_chatgpt",
                    answer_chars=len(answer),
                    running=running,
                    forced_answer=forced_answer,
                )
                last_progress_at = now
            self._write_json(
                run_state_path,
                {
                    "mode": self.mode,
                    "terminal_state": "forcing_answer" if forced_answer else "running",
                    "conversation_url": str(getattr(page, "url", "") or ""),
                    "updated_at": time.time(),
                    "answer_chars": len(answer),
                    "running": running,
                    "forced_answer": forced_answer,
                    "output_timeout_seconds": timeout_seconds,
                },
            )
            # Two identical polls plus no running control avoids saving a streaming
            # partial answer. After forcing, require material growth beyond the
            # pre-force acknowledgement before accepting a stable terminal answer.
            min_chars = 1200 if self.mode == "deep" else 200
            changed_after_force = (
                not forced_answer
                or (answer != force_baseline and len(answer) >= max(min_chars, len(force_baseline) + 100))
            )
            if len(answer) >= min_chars and changed_after_force and not running and stable_polls >= 2:
                return answer
            if now >= hard_deadline:
                raise ChatGPTWebResearchError(
                    f"ChatGPT {self.mode} request did not reach an extractable terminal state within its "
                    f"{timeout_seconds}-second total output deadline"
                    f"{' (including the Answer now recovery window)' if self.mode == 'pro' else ''}."
                )
            remaining = max(0.0, hard_deadline - now)
            page.wait_for_timeout(min(float(self._poll_seconds()), remaining) * 1000)

    def _browser_research(
        self,
        prompt: str,
        *,
        run_state_path: Path | None = None,
        partial_path: Path | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - packaging error
            raise ChatGPTWebResearchError("Playwright is required for ChatGPT Web researchers.") from exc

        started_at = time.time()
        output_timeout_seconds = self._timeout_seconds()
        self._write_json(
            run_state_path,
            {
                "mode": self.mode,
                "started_at": started_at,
                "output_deadline_at": started_at + output_timeout_seconds,
                "output_timeout_seconds": output_timeout_seconds,
                "terminal_state": "launching",
                "answer_chars": 0,
            },
        )
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
                    self._select_reasoning_mode(page)
                self._send(page, prompt)
                page.wait_for_timeout(1000)
                try:
                    page.wait_for_url(re.compile(r"https://chatgpt\.com/c/"), timeout=30000)
                except Exception:
                    pass
                conversation_url = str(page.url or "")
                self._write_json(
                    run_state_path,
                    {
                        "terminal_state": "running",
                        "conversation_url": conversation_url,
                        "updated_at": time.time(),
                    },
                )
                self._emit_progress("browser_research_started", running=True)
                if self.mode == "deep":
                    cdp_session = page.context.new_cdp_session(page)
                    try:
                        target_info = cdp_session.send("Target.getTargetInfo")["targetInfo"]
                    finally:
                        cdp_session.detach()
                    connector = self._deep_connector_target(str(target_info.get("targetId") or ""))
                    conversation_url = self._target_url(str(connector.get("parentId") or ""), page.url)
                    self._write_json(run_state_path, {"conversation_url": conversation_url})
                    answer = self._wait_and_extract_deep(
                        connector,
                        partial_path=partial_path,
                        run_state_path=run_state_path,
                    )
                else:
                    answer = self._wait_and_extract(
                        page,
                        partial_path=partial_path,
                        run_state_path=run_state_path,
                    )
                    conversation_url = page.url
                metadata = {
                    "mode": self.mode,
                    "conversation_url": conversation_url,
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "answer_chars": len(answer),
                    "terminal_state": "extracted",
                }
                self._write_json(run_state_path, metadata)
                self._emit_progress("answer_extracted", answer_chars=len(answer), running=False)
                return answer, conversation_url, metadata
            except Exception as exc:
                state = "timeout" if "did not reach an extractable terminal state" in str(exc) else "error"
                self._write_json(
                    run_state_path,
                    {
                        "mode": self.mode,
                        "conversation_url": str(getattr(page, "url", "") or ""),
                        "finished_at": time.time(),
                        "terminal_state": state,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                self._emit_progress(state, running=False)
                raise
            finally:
                try:
                    page.close()
                except Exception:
                    pass

    def _run_single(self, prompt: str, *, save_artifacts: bool) -> str:
        ctx = current_log_context()
        evidence_parent = Path(
            create_subagent_evidence_dir(
                self.tool_name,
                str(ctx.get("session_id") or current_session_id() or ""),
            )
        )
        # The administrator intentionally groups same-type researchers under one
        # parent. Fixed filenames must still be isolated per invocation when up
        # to five same-mode requests execute concurrently.
        root = evidence_parent / f"run-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        evidence_dir = str(root)
        run_state_path = root / "chatgpt-run.json"
        partial_path = root / f"chatgpt-{self.mode}-partial.md"
        request_path = root / "chatgpt-request.md"
        request_path.write_text(prompt, encoding="utf-8")
        metadata: dict[str, Any] = {"mode": self.mode, "terminal_state": "error"}
        try:
            answer, conversation_url, metadata = self._research(
                prompt,
                run_state_path=run_state_path,
                partial_path=partial_path,
            )
            filename = f"chatgpt-{self.mode}-response.md"
            (root / filename).write_text(answer, encoding="utf-8")
            try:
                partial_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._write_json(run_state_path, metadata)
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
            try:
                existing = json.loads(run_state_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    metadata.update(existing)
            except Exception:
                pass
            metadata.update({"finished_at": time.time(), "error": f"{type(exc).__name__}: {exc}"})
            if str(metadata.get("terminal_state") or "") not in {"timeout", "forcing_answer"}:
                metadata["terminal_state"] = "error"
            self._write_json(run_state_path, metadata)
            partial_review = ""
            try:
                if partial_path.exists():
                    partial_review = partial_path.read_text(encoding="utf-8").strip()
            except Exception:
                partial_review = ""
            source_url = str(metadata.get("conversation_url") or "")
            failure_artifacts: list[dict[str, str]] = []
            if save_artifacts:
                failure_artifacts.extend(
                    [
                        {
                            "filename": "chatgpt-run.json",
                            "source_url": source_url,
                            "description": "Terminal failure metadata including the recoverable ChatGPT conversation URL, timestamps, last progress state, and extraction error.",
                        },
                        {
                            "filename": "chatgpt-request.md",
                            "source_url": source_url,
                            "description": "Exact prompt submitted before browser launch, retained even if the browser worker times out.",
                        },
                    ]
                )
                if partial_review:
                    failure_artifacts.append(
                        {
                            "filename": partial_path.name,
                            "source_url": source_url,
                            "description": "Latest incrementally saved ChatGPT response text recovered before the terminal failure.",
                        }
                    )
            payload = {
                "research_worked": False,
                "failure_reason": str(exc),
                "final_research_review": partial_review,
                "partial_result": bool(partial_review),
                "evidence_data_path": evidence_dir if save_artifacts else "",
                "key_artifacts": failure_artifacts,
                "tool_call_counts": {"chatgpt_web": 1},
                "total_tool_calls": 1,
            }
            return _compact(payload)
        finally:
            cleanup_research_artifacts(evidence_dir, save_artifacts=save_artifacts)

    def run(self, prompt: str | list[str], save_artifacts: bool = False) -> str:
        prompts, error = normalize_subagent_prompts(prompt, min_chars=100, max_prompts=5)
        if error:
            return error
        return run_parallel_subagent_prompts(
            prompts,
            lambda item: self._run_single(item, save_artifacts=save_artifacts),
        )


def _make_tool(helper: ChatGPTWebResearchAgentTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    mode_label = {
        "deep": "Deep Research",
        "pro": "Pro mode",
        "xhigh": "Extra High reasoning mode",
    }[helper.mode]
    description = f"""Run one authenticated ChatGPT Web {mode_label} research agent in a clean Chrome tab and wait for the complete extracted response.

Use it for an independent ChatGPT {mode_label} research or reasoning pass. Give a self-contained prompt with the topic, scope, source/evidence requirements, uncertainties to test, and expected output. Normal clients submit through the configured authenticated async HTTPS broker; only the outbound workstation worker uses the signed-in local Chrome/CDP executor.

Args:
    prompt: One detailed research prompt, or a list of up to 5 prompts to run independently.
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
        properties["prompt"]["description"] = "One detailed research prompt, or a list of up to five independent detailed prompts."
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


def get_chatgptxhigh_tool(config: ToolsConfig):
    return _make_tool(ChatGPTWebResearchAgentTool(config, mode="xhigh"))
