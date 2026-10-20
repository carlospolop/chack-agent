import json
import time

from chack_tools.chatgpt_tab_janitor import MAX_TAB_AGE_SECONDS, _candidates, clean_tabs


def _job(root, suffix, target, *, started_at, terminal_state="running"):
    folder = root / "jobs" / f"job_{suffix:08x}-0000-0000-0000-000000000000"
    folder.mkdir(parents=True)
    (folder / "chatgpt-run.json").write_text(json.dumps({
        "started_at": started_at,
        "terminal_state": terminal_state,
        "browser_target_ids": [target],
    }))
    return folder


class FakePage:
    def __init__(self, target):
        self.target = target
        self.closed = False
        self.fail_close = False

    def close(self):
        if self.fail_close:
            raise RuntimeError("browser refused to close tab")
        self.closed = True


class FakeSession:
    def __init__(self, page):
        self.page = page

    def send(self, method):
        assert method == "Target.getTargetInfo"
        return {"targetInfo": {"targetId": self.page.target}}

    def detach(self):
        pass


class FakeContext:
    def __init__(self, pages):
        self.pages = pages

    def new_cdp_session(self, page):
        return FakeSession(page)


class FakePlaywright:
    def __init__(self, pages):
        self.pages = pages
        self.connections = 0
        self.chromium = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def connect_over_cdp(self, url, *, timeout):
        assert url == "http://127.0.0.1:9226"
        assert timeout == 15000
        self.connections += 1
        return type("Browser", (), {"contexts": [FakeContext(self.pages)]})()


def test_terminal_tabs_close_immediately_and_live_tabs_only_after_two_hours(monkeypatch, tmp_path):
    now = time.time()
    done = _job(tmp_path, 1, "owned-done", started_at=now - 10, terminal_state="extracted")
    stale = _job(tmp_path, 2, "owned-stale", started_at=now - MAX_TAB_AGE_SECONDS, terminal_state="running")
    _job(tmp_path, 3, "owned-live", started_at=now - 10, terminal_state="running")
    pages = [FakePage(name) for name in ("owned-done", "owned-stale", "owned-live", "unrelated-browser-tab")]
    fake = FakePlaywright(pages)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: fake)

    assert clean_tabs(tmp_path, now=now) == 2
    assert [p.closed for p in pages] == [True, True, False, False]
    for job in (done, stale):
        assert (job / "browser-tabs-cleaned.json").exists()
    assert clean_tabs(tmp_path, now=now + 5) == 0
    assert fake.connections == 1  # Cleaned jobs do not reconnect every timer tick.


def test_failed_close_is_retried_and_unrelated_pages_stay_open(monkeypatch, tmp_path):
    now = time.time()
    job = _job(tmp_path, 4, "owned-failure", started_at=now - 30, terminal_state="error")
    page, unrelated = FakePage("owned-failure"), FakePage("unrelated")
    page.fail_close = True
    fake = FakePlaywright([page, unrelated])
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: fake)

    assert clean_tabs(tmp_path, now=now) == 0
    assert not (job / "browser-tabs-cleaned.json").exists()
    page.fail_close = False
    assert clean_tabs(tmp_path, now=now + 1) == 1
    assert page.closed and not unrelated.closed


def test_unowned_or_recent_live_jobs_never_connect(monkeypatch, tmp_path):
    now = time.time()
    _job(tmp_path, 5, "owned-live", started_at=now - 120, terminal_state="running")
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: (_ for _ in ()).throw(AssertionError("unexpected CDP connection")))
    assert not _candidates(tmp_path, now=now)
    assert clean_tabs(tmp_path, now=now) == 0
