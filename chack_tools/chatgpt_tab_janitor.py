"""Close only Chack-owned ChatGPT CDP tabs after a terminal job or two hours.

The browser is shared with other work. Never close by URL, tab title, or age
alone: ownership comes from the target IDs recorded when a worker creates a
page. This also cleans up tabs left by SIGKILL, a stuck Playwright call, or a
worker hard deadline that terminates its process with os._exit().
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

MAX_TAB_AGE_SECONDS = 2 * 60 * 60
_JOB_ID = re.compile(r"job_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_TERMINAL = {"extracted", "error", "timeout", "cancelled", "worker_hard_timeout"}
_CLEANED_FILE = "browser-tabs-cleaned.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _candidates(state_root: Path, *, now: float) -> dict[Path, tuple[str, ...]]:
    """Select only recorded, not-yet-cleaned worker targets."""
    pending: dict[Path, tuple[str, ...]] = {}
    for job_dir in (state_root / "jobs").glob("job_*"):
        if not job_dir.is_dir() or not _JOB_ID.fullmatch(job_dir.name):
            continue
        state = _read_json(job_dir / "chatgpt-run.json")
        raw_ids = state.get("browser_target_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            continue
        ids = tuple(dict.fromkeys(t for t in raw_ids if isinstance(t, str) and t))
        if len(ids) != len(raw_ids):
            continue
        cleaned = _read_json(job_dir / _CLEANED_FILE)
        if cleaned.get("browser_target_ids") == list(ids):
            continue
        try:
            started_at = float(state["started_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(started_at) or started_at <= 0 or started_at > now:
            continue
        if (str(state.get("terminal_state") or "").lower() not in _TERMINAL
                and now - started_at < MAX_TAB_AGE_SECONDS):
            continue
        pending[job_dir] = ids
    return pending


def _mark_cleaned(job_dir: Path, ids: tuple[str, ...], *, now: float) -> None:
    path = job_dir / _CLEANED_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"browser_target_ids": list(ids), "closed_at": now}), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def clean_tabs(state_root: Path, *, cdp_url: str = "http://127.0.0.1:9226", now: float | None = None) -> int:
    """Close proven Chack-owned tabs; keep all unrelated and young live tabs."""
    now = time.time() if now is None else now
    pending = _candidates(state_root, now=now)
    if not pending:
        return 0

    from playwright.sync_api import sync_playwright

    wanted = {target for ids in pending.values() for target in ids}
    remaining = set(wanted)
    closed = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url, timeout=15000)
        for context in browser.contexts:
            for page in context.pages:
                try:
                    session = context.new_cdp_session(page)
                    try:
                        target_id = str(session.send("Target.getTargetInfo")["targetInfo"]["targetId"])
                    finally:
                        session.detach()
                except Exception:
                    # If a target cannot be identified, leave it alone.
                    continue
                if target_id not in wanted:
                    continue
                try:
                    page.close()
                    closed += 1
                    remaining.discard(target_id)
                except Exception:
                    # Retry next timer tick; never mark an unclosed tab clean.
                    continue

    for job_dir, ids in pending.items():
        if not remaining.intersection(ids):
            _mark_cleaned(job_dir, ids, now=now)
    return closed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path(os.environ.get(
        "CHACK_CHATGPT_WORKER_STATE_DIR", str(Path.home() / ".local/state/chack-chatgpt-worker")
    )))
    parser.add_argument("--cdp-url", default=os.environ.get("CHACK_CHATGPT_CDP_URL", "http://127.0.0.1:9226"))
    args = parser.parse_args()
    count = clean_tabs(args.state_root, cdp_url=args.cdp_url)
    if count:
        print(f"closed {count} completed or two-hour-old Chack-owned browser tabs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
