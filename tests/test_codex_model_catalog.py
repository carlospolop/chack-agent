from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import tomllib

import pytest

from chack_agent.backends import codex_backend
from chack_agent.backends.codex_backend import build_executor
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

_CATALOG = {
    "models": [
        {"slug": "gpt-6-sol", "multi_agent_version": "v2", "tool_mode": "code_mode_only",
         "context_window": 1000},
        {"slug": "gpt-5.6-luna", "multi_agent_version": "v1", "tool_mode": "code_mode_only"},
    ]
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_CODEX_HOME_BASE", str(tmp_path / "homes"))
    monkeypatch.setattr(codex_backend, "_CODEX_CATALOGS", {})


def _fake_codex(tmp_path, monkeypatch, *, exit_code=0, catalog=_CATALOG):
    """A ``codex`` that only answers ``debug models``, counting its calls."""
    calls = tmp_path / "calls"
    script = tmp_path / "codex"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys
            with open({str(calls)!r}, "a") as handle:
                handle.write(" ".join(sys.argv[1:]) + "\\n")
            if sys.argv[1:] == ["debug", "models"]:
                print({json.dumps(json.dumps(catalog))})
            sys.exit({exit_code})
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CODEX_PATH", str(script))
    return calls


def _executor(model: str, session: str = "s1", monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(codex_backend, "current_session_id", lambda: session, raising=False)
    config = ChackConfig(
        model=ModelConfig(primary=model, provider="codex"),
        agent=AgentConfig(main_action="test", sub_action="test"),
        session=SessionConfig(max_turns=2),
        tools=ToolsConfig(),
        credentials=CredentialsConfig(openai_api_key="sk-test"),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )
    executor = build_executor(
        config, system_prompt="system", max_turns=2, memory_max_messages=10, memory_reset_to_messages=5
    )
    executor._ensure_codex_home_and_config()
    return executor


def _config(executor) -> dict:
    with open(os.path.join(executor._codex_home, "config.toml"), "rb") as handle:
        return tomllib.load(handle)


def test_runs_use_the_catalog_with_multi_agent_mode_off(tmp_path, monkeypatch):
    _fake_codex(tmp_path, monkeypatch)

    config = _config(_executor("gpt-6-sol"))

    with open(config["model_catalog_json"], encoding="utf-8") as handle:
        models = {entry["slug"]: entry for entry in json.load(handle)["models"]}
    assert set(models) == {"gpt-6-sol", "gpt-5.6-luna"}
    assert all(entry["multi_agent_version"] is None for entry in models.values())
    # Everything else about each model is kept.
    assert models["gpt-6-sol"]["tool_mode"] == "code_mode_only"
    assert models["gpt-6-sol"]["context_window"] == 1000


def test_a_gateway_prefixed_model_matches_its_catalog_entry(tmp_path, monkeypatch):
    _fake_codex(tmp_path, monkeypatch)

    assert "model_catalog_json" in _config(_executor("bakkt/gpt-6-sol"))


def test_a_model_missing_from_the_catalog_keeps_the_cli_default(tmp_path, monkeypatch):
    _fake_codex(tmp_path, monkeypatch)

    config = _config(_executor("some-other-model"))

    assert "model_catalog_json" not in config
    assert config["model"] == "some-other-model"


def test_a_cli_without_a_catalog_command_still_gets_a_valid_config(tmp_path, monkeypatch):
    _fake_codex(tmp_path, monkeypatch, exit_code=2)

    config = _config(_executor("gpt-6-sol"))

    assert "model_catalog_json" not in config
    assert config["model"] == "gpt-6-sol"


def test_the_catalog_is_read_once_per_cli(tmp_path, monkeypatch):
    calls = _fake_codex(tmp_path, monkeypatch)

    _executor("gpt-6-sol", "a", monkeypatch)
    _executor("gpt-6-sol", "b", monkeypatch)

    assert calls.read_text().splitlines() == ["debug models"]


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI not installed")
def test_the_real_cli_sends_no_agent_management_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_PATH", shutil.which("codex"))
    listed = codex_backend._codex_catalog_models(shutil.which("codex"), str(tmp_path))
    multi_agent_model = next(
        (entry["slug"] for entry in listed or [] if entry.get("multi_agent_version")), None
    )
    if multi_agent_model is None:
        pytest.skip("this Codex CLI has no multi-agent models")
    executor = _executor(multi_agent_model)

    done = subprocess.run(
        [shutil.which("codex"), "debug", "prompt-input", "hello"],
        env={**os.environ, "CODEX_HOME": executor._codex_home},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    items = json.loads(done.stdout)
    tool_groups = [
        tool.get("name") for item in items if item.get("type") == "additional_tools" for tool in item["tools"]
    ]
    developer_text = json.dumps([item for item in items if item.get("role") == "developer"])
    assert tool_groups, "the prompt lists no tools at all"
    assert "collaboration" not in tool_groups
    assert "spawn_agent" not in developer_text
