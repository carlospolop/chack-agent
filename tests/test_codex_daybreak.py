from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from chack_agent.backends import codex_backend as codex_backend_module
from chack_agent.backends import codex_daybreak
from chack_agent.backends.codex_backend import CodexExecutor


SECRET = "sk-test-secret-value-never-persisted"


@pytest.fixture(autouse=True)
def _isolated_caches(tmp_path, monkeypatch):
    codex_daybreak.reset_caches_for_tests()
    monkeypatch.setenv(
        codex_daybreak.DAYBREAK_CACHE_PATH_ENV,
        str(tmp_path / "daybreak_support.json"),
    )
    monkeypatch.delenv(codex_daybreak.DAYBREAK_MODE_ENV, raising=False)
    yield
    codex_daybreak.reset_caches_for_tests()


def _env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _jwt(claims: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{body}.signature"


def _fake_codex(tmp_path: Path, *, account: dict | None, models: list[dict]) -> str:
    """A stand-in `codex` that answers --version and the app-server probe."""
    script = tmp_path / "codex"
    script.write_text(
        f"""#!{sys.executable}
import json, sys
if sys.argv[1:] == ["--version"]:
    print("codex-cli 0.165.0")
    sys.exit(0)
assert sys.argv[1:] == ["app-server"], sys.argv
account = json.loads({json.dumps(json.dumps(account))})
models = json.loads({json.dumps(json.dumps(models))})
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if "id" not in message:
        continue
    if method == "initialize":
        result = {{"userAgent": "fake"}}
    elif method == "account/read":
        result = {{"account": account, "requiresOpenaiAuth": True}}
    elif method == "model/list":
        result = {{"data": models, "nextCursor": None}}
    else:
        result = {{}}
    print(json.dumps({{"id": message["id"], "result": result}}), flush=True)
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_daybreak_mode_defaults_to_auto_and_accepts_off_switches():
    assert codex_daybreak.daybreak_mode("") == "auto"
    assert codex_daybreak.daybreak_mode("auto") == "auto"
    for value in ("0", "false", "off", "OFF", "disabled", "no"):
        assert codex_daybreak.daybreak_mode(value) == "off"
    for value in ("1", "true", "on", "force"):
        assert codex_daybreak.daybreak_mode(value) == "on"


def test_catalog_selection_prefers_blue_and_reads_both_wire_spellings():
    models = [
        {"model": "gpt-a", "availableAccessPrograms": {"cyber": ["standard", "daybreakRed", "daybreakBlue"]}},
        {"model": "gpt-b", "available_access_programs": {"cyber": ["daybreak_red"]}},
        {"model": "gpt-c", "availableAccessPrograms": {"cyber": ["standard"]}},
        {"model": "gpt-d", "availableAccessPrograms": None},
    ]

    assert codex_daybreak.program_from_catalog(models, "gpt-a") == "daybreak_blue"
    assert codex_daybreak.program_from_catalog(models, "gpt-b") == "daybreak_red"
    assert codex_daybreak.program_from_catalog(models, "gpt-c") == ""
    assert codex_daybreak.program_from_catalog(models, "gpt-d") == ""
    assert codex_daybreak.program_from_catalog(models, "missing") == ""


def test_fingerprint_hides_secret_and_survives_chatgpt_token_rotation():
    claims = {
        "sub": "user-1",
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acct-1",
            "chatgpt_user_id": "user-1",
        },
    }
    first = codex_daybreak.credential_fingerprint("chatgpt", _jwt({**claims, "iat": 1}), "acct-1")
    rotated = codex_daybreak.credential_fingerprint("chatgpt", _jwt({**claims, "iat": 2}), "acct-1")
    other_user = codex_daybreak.credential_fingerprint(
        "chatgpt",
        _jwt({**claims, "https://api.openai.com/auth": {"chatgpt_user_id": "user-2"}}),
        "acct-1",
    )
    api_key = codex_daybreak.credential_fingerprint("api_key", SECRET)

    assert first == rotated
    assert first != other_user
    assert SECRET not in api_key and len(api_key) == 64
    assert codex_daybreak.credential_fingerprint("api_key", "") == ""


@pytest.mark.parametrize(
    "text",
    [
        "Error: Daybreak is unavailable for model gpt-6-sol; turn it off or choose a compatible model",
        "Error: Daybreak requires a signed-in ChatGPT account or an OpenAI API key",
        "Daybreak account availability could not be determined: boom",
        '{"error":{"message":"access_programs.cyber daybreak_blue is not allowed for this account"}}',
        # Verbatim backend replies captured 2026-10-29 from chatgpt.com/backend-api/codex/responses.
        '403 {"detail":"Daybreak isn\'t available for this model. Turn off Daybreak or choose another model."}',
        '{"type":"error","error":{"message":"The selected model requires a different Daybreak access '
        'program.","param":"access_programs.cyber","code":"invalid_access_program"}}',
    ],
)
def test_rejection_signals_are_recognized(text):
    assert codex_daybreak.looks_like_daybreak_rejection(text)


@pytest.mark.parametrize(
    "text",
    [
        "stream disconnected before completion",
        "401 Unauthorized",
        "Daybreak enabled",
        # Verbatim content flag captured 2026-10-29 from a prod run with daybreak_blue selected.
        '{"type":"turn.failed","error":{"message":"This content was flagged for possible cybersecurity risk. '
        "If this seems wrong, try rephrasing your request. Otherwise, confirm that the access_programs.cyber "
        "parameter is set to the appropriate tier, and note that some cybersecurity requests are still limited, "
        'even when Daybreak is on."}}',
    ],
)
def test_unrelated_failures_are_not_daybreak_rejections(text):
    assert not codex_daybreak.looks_like_daybreak_rejection(text)


def test_probe_enables_daybreak_only_when_catalog_offers_it(tmp_path):
    codex = _fake_codex(
        tmp_path,
        account={"type": "chatgpt", "planType": "pro"},
        models=[{"model": "gpt-6-sol", "availableAccessPrograms": {"cyber": ["standard", "daybreakBlue"]}}],
    )
    program, reason = codex_daybreak.probe_catalog(codex, "gpt-6-sol", dict(os.environ))
    assert program == "daybreak_blue", reason

    program, _ = codex_daybreak.probe_catalog(codex, "gpt-5.5", dict(os.environ))
    assert program == ""


def test_probe_rejects_accounts_that_cannot_select_programs(tmp_path):
    codex = _fake_codex(
        tmp_path,
        account=None,
        models=[{"model": "gpt-6-sol", "availableAccessPrograms": {"cyber": ["daybreakBlue"]}}],
    )
    program, reason = codex_daybreak.probe_catalog(codex, "gpt-6-sol", dict(os.environ))
    assert program == ""
    assert "account" in reason


def test_probe_never_runs_in_the_agent_codex_home(tmp_path, monkeypatch):
    seen = {}
    real_popen = codex_daybreak.subprocess.Popen

    def launch(args, **kwargs):
        seen["home"] = kwargs["env"]["CODEX_HOME"]
        seen["auth_copied"] = os.path.exists(os.path.join(seen["home"], "auth.json"))
        return real_popen(args, **kwargs)

    monkeypatch.setattr(codex_daybreak.subprocess, "Popen", launch)
    codex = _fake_codex(tmp_path, account={"type": "apiKey"}, models=[])
    auth = tmp_path / "agent-home" / "auth.json"
    auth.parent.mkdir()
    auth.write_text("{}")

    codex_daybreak.probe_catalog(
        codex,
        "gpt-6-sol",
        {**os.environ, "CODEX_HOME": str(auth.parent)},
        auth_file=str(auth),
    )

    assert seen["home"] != str(auth.parent)
    assert seen["auth_copied"] is True
    assert not os.path.exists(seen["home"])


def test_resolve_caches_by_fingerprint_without_storing_the_secret(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(codex_daybreak, "codex_version", lambda _path: (0, 165, 0))
    monkeypatch.setattr(
        codex_daybreak,
        "probe_catalog",
        lambda *args, **kwargs: calls.append(1) or ("daybreak_blue", "catalog offers it"),
    )
    fingerprint = codex_daybreak.credential_fingerprint("api_key", SECRET)

    kwargs = dict(
        codex_path="codex",
        model="gpt-6-sol",
        fingerprint=fingerprint,
        env={},
        env_value=_env_value,
    )
    assert codex_daybreak.resolve_program(**kwargs) == "daybreak_blue"
    codex_daybreak.reset_caches_for_tests()
    monkeypatch.setattr(codex_daybreak, "codex_version", lambda _path: (0, 165, 0))
    assert codex_daybreak.resolve_program(**kwargs) == "daybreak_blue"

    assert calls == [1]
    body = (tmp_path / "daybreak_support.json").read_text()
    assert SECRET not in body
    assert fingerprint not in body


def test_resolve_respects_off_switch_and_old_codex(monkeypatch):
    monkeypatch.setattr(
        codex_daybreak,
        "probe_catalog",
        lambda *args, **kwargs: pytest.fail("probe must not run"),
    )
    kwargs = dict(codex_path="codex", model="gpt-6-sol", fingerprint="f" * 64, env={}, env_value=_env_value)

    monkeypatch.setattr(codex_daybreak, "codex_version", lambda _path: (0, 165, 0))
    monkeypatch.setenv(codex_daybreak.DAYBREAK_MODE_ENV, "off")
    assert codex_daybreak.resolve_program(**kwargs) == ""

    monkeypatch.delenv(codex_daybreak.DAYBREAK_MODE_ENV)
    monkeypatch.setattr(codex_daybreak, "codex_version", lambda _path: (0, 158, 0))
    assert codex_daybreak.resolve_program(**kwargs) == ""


def test_undecided_probe_leaves_daybreak_off_without_persisting(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_daybreak, "codex_version", lambda _path: (0, 165, 0))
    monkeypatch.setattr(
        codex_daybreak,
        "probe_catalog",
        lambda *args, **kwargs: (None, "model/list failed"),
    )

    program = codex_daybreak.resolve_program(
        codex_path="codex",
        model="gpt-6-sol",
        fingerprint="f" * 64,
        env={},
        env_value=_env_value,
    )

    assert program == ""
    assert not (tmp_path / "daybreak_support.json").exists()


def _executor() -> CodexExecutor:
    executor = object.__new__(CodexExecutor)
    executor._runtime_env_json = "{}"
    executor._uses_openrouter_route = False
    executor._model_name = "gpt-6-sol"
    executor._model_provider = "codex"
    executor._codex_path = "codex"
    executor._codex_home = ""
    executor._thread_id = None
    executor._sub_action = ""
    executor._use_codex_access_token = False
    executor._use_existing_codex_auth_file = False
    executor._existing_codex_auth_file = ""
    executor._openai_api_key = SECRET
    executor._disable_native_shell = False
    executor._disable_native_web_search = False
    executor._thinking_effort = "high"
    executor._output_schema_path = None
    return executor


def test_build_command_adds_daybreak_only_when_resolved():
    executor = _executor()
    assert "daybreak=true" not in executor._build_command()

    executor._daybreak_program = "daybreak_blue"
    command = executor._build_command()
    assert command[command.index("daybreak=true") - 1] == "-c"
    assert command[command.index("cli_daybreak") - 1] == "--enable"

    executor._thread_id = "11111111-2222-3333-4444-555555555555"
    resume = executor._build_command()
    assert resume[:3] == ["codex", "exec", "resume"]
    assert "daybreak=true" in resume


def test_executor_resolves_daybreak_from_hashed_credentials(monkeypatch):
    seen = {}

    def resolve(**kwargs):
        seen.update(kwargs)
        return "daybreak_blue"

    monkeypatch.setattr(codex_daybreak, "resolve_program", resolve)
    executor = _executor()
    executor._resolve_daybreak()

    assert executor._daybreak_program == "daybreak_blue"
    assert seen["fingerprint"] == codex_daybreak.credential_fingerprint("api_key", SECRET)
    assert seen["model"] == "gpt-6-sol"


def test_executor_skips_daybreak_when_switched_off_or_routed_elsewhere(monkeypatch):
    monkeypatch.setattr(
        codex_daybreak,
        "resolve_program",
        lambda **_kwargs: pytest.fail("must not resolve"),
    )
    executor = _executor()
    executor._runtime_env_json = json.dumps({codex_daybreak.DAYBREAK_MODE_ENV: "off"})
    executor._resolve_daybreak()
    assert executor._daybreak_program == ""

    routed = _executor()
    routed._uses_openrouter_route = True
    routed._resolve_daybreak()
    assert routed._daybreak_program == ""


def test_rejected_daybreak_run_is_retried_without_it_and_remembered(monkeypatch):
    executor = _executor()
    executor._daybreak_program = "daybreak_blue"
    executor._daybreak_fingerprint = "f" * 64
    launched = []

    def build_command():
        with_daybreak = bool(executor._daybreak_program)
        launched.append(with_daybreak)
        if with_daybreak:
            script = (
                "import sys; print('Error: Daybreak is unavailable for model gpt-6-sol; "
                "turn it off or choose a compatible model'); sys.exit(1)"
            )
        else:
            script = (
                "import json; "
                "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'done'}}))"
            )
        return [sys.executable, "-u", "-c", script]

    remembered = []
    monkeypatch.setattr(executor, "_build_command", build_command)
    monkeypatch.setattr(executor, "_build_env", lambda: os.environ.copy())
    monkeypatch.setattr(executor, "_write_codex_explicit_mcp_env", lambda _env: None)
    monkeypatch.setattr(executor, "_log_codex_failure", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        codex_daybreak,
        "mark_unavailable",
        lambda **kwargs: remembered.append(kwargs["fingerprint"]),
    )

    output, _steps, _raw = executor._run_codex_once("hi", allow_api_key_fallback=False)

    assert launched == [True, False]
    assert output == "done"
    assert remembered == ["f" * 64]
    assert executor._daybreak_program == ""


def test_unrelated_failure_keeps_daybreak_and_does_not_retry(monkeypatch):
    executor = _executor()
    executor._daybreak_program = "daybreak_blue"
    launched = []

    def build_command():
        launched.append(1)
        return [sys.executable, "-u", "-c", "import sys; print('stream disconnected'); sys.exit(1)"]

    monkeypatch.setattr(executor, "_build_command", build_command)
    monkeypatch.setattr(executor, "_build_env", lambda: os.environ.copy())
    monkeypatch.setattr(executor, "_write_codex_explicit_mcp_env", lambda _env: None)
    monkeypatch.setattr(executor, "_log_codex_failure", lambda *_a, **_kw: None)

    output, _steps, _raw = executor._run_codex_once("hi", allow_api_key_fallback=False)

    assert launched == [1]
    assert output.startswith("ERROR: Codex exec failed")
    assert executor._daybreak_program == "daybreak_blue"


def test_direct_cached_request_carries_the_resolved_program():
    executor = _executor()
    executor._cacheable_developer_prompt = "stable"
    executor._prompt_cache_prefix_key = "chack-key"
    executor._output_schema_strict = False
    executor._output_schema_json = ""
    executor._direct_cache_credentials = lambda: ("api_key", SECRET, "")

    _url, _headers, body = executor._direct_cache_request("task")
    assert "access_programs" not in body

    executor._daybreak_program = "daybreak_blue"
    _url, _headers, body = executor._direct_cache_request("task")
    assert body["access_programs"] == {"cyber": "daybreak_blue"}


def test_backend_module_exposes_daybreak_helpers():
    assert codex_backend_module.codex_daybreak is codex_daybreak


def test_bundled_catalog_without_metadata_is_undecided(tmp_path):
    codex = _fake_codex(
        tmp_path,
        account={"type": "chatgpt", "planType": "pro"},
        models=[{"model": "gpt-6-sol", "availableAccessPrograms": None}],
    )
    program, reason = codex_daybreak.probe_catalog(codex, "gpt-6-sol", dict(os.environ))
    assert program is None
    assert "metadata" in reason


class _CatalogResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def test_chatgpt_catalog_probe_reads_the_codex_models_endpoint(monkeypatch):
    import requests

    seen = {}

    def fake_get(url, *, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return _CatalogResponse(
            200,
            {
                "models": [
                    {"slug": "gpt-6-sol", "available_access_programs": {"cyber": ["standard", "daybreak_blue"]}},
                    {"slug": "gpt-6.1-sol", "available_access_programs": {"cyber": ["standard"]}},
                ]
            },
        )

    monkeypatch.setattr(requests, "get", fake_get)
    probe = codex_daybreak.probe_chatgpt_catalog

    assert probe("tok", "acct", "gpt-6-sol", client_version=(0, 165, 0))[0] == "daybreak_blue"
    assert seen["url"] == "https://chatgpt.com/backend-api/codex/models"
    assert seen["params"] == {"client_version": "0.165.0"}
    assert seen["headers"]["ChatGPT-Account-ID"] == "acct"
    assert probe("tok", "acct", "gpt-6.1-sol", client_version=(0, 165, 0))[0] == ""
    assert probe("tok", "acct", "unlisted", client_version=(0, 165, 0))[0] == ""


@pytest.mark.parametrize("status", [401, 403, 500])
def test_chatgpt_catalog_errors_are_undecided(monkeypatch, status):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **kw: _CatalogResponse(status, {}))
    program, _ = codex_daybreak.probe_chatgpt_catalog(
        "tok", "acct", "gpt-6-sol", client_version=(0, 165, 0)
    )
    assert program is None


def test_resolve_uses_the_chatgpt_catalog_for_chatgpt_credentials(monkeypatch):
    monkeypatch.setattr(codex_daybreak, "codex_version", lambda _path: (0, 165, 0))
    monkeypatch.setattr(
        codex_daybreak,
        "probe_catalog",
        lambda *a, **kw: pytest.fail("app-server probe is only for other credentials"),
    )
    monkeypatch.setattr(
        codex_daybreak,
        "probe_chatgpt_catalog",
        lambda bearer, account, model, **kw: ("daybreak_blue", "ok"),
    )

    program = codex_daybreak.resolve_program(
        codex_path="codex",
        model="gpt-6-sol",
        fingerprint="f" * 64,
        env={},
        env_value=_env_value,
        chatgpt_bearer="tok",
        chatgpt_account_id="acct",
    )

    assert program == "daybreak_blue"


def test_executor_passes_chatgpt_identity_to_the_probe(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        codex_daybreak,
        "resolve_program",
        lambda **kwargs: seen.update(kwargs) or "",
    )
    executor = _executor()
    executor._direct_cache_credentials = lambda: ("chatgpt", "jwt-token", "acct-1")
    executor._resolve_daybreak()

    assert seen["chatgpt_bearer"] == "jwt-token"
    assert seen["chatgpt_account_id"] == "acct-1"
    assert executor._daybreak_program == ""
