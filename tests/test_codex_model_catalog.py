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


def _executor(model: str, session: str = "s1", monkeypatch=None, tools: ToolsConfig | None = None):
    if monkeypatch is not None:
        monkeypatch.setattr(codex_backend, "current_session_id", lambda: session, raising=False)
    config = ChackConfig(
        model=ModelConfig(primary=model, provider="codex"),
        agent=AgentConfig(main_action="test", sub_action="test"),
        session=SessionConfig(max_turns=2),
        tools=tools or ToolsConfig(),
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


@pytest.mark.parametrize("exec_enabled", [True, False])
def test_codex_never_offers_its_own_shell(tmp_path, monkeypatch, exec_enabled):
    _fake_codex(tmp_path, monkeypatch)
    executor = _executor("gpt-6-sol", tools=ToolsConfig(exec_enabled=exec_enabled))

    command = executor._build_command()

    # chack's exec tool is the only shell.
    disabled = {command[i + 1] for i, arg in enumerate(command) if arg == "--disable"}
    assert {"shell_tool", "unified_exec"} <= disabled


@pytest.mark.parametrize("exec_enabled", [True, False])
def test_codex_file_tools_come_only_with_chack_exec(tmp_path, monkeypatch, exec_enabled):
    _fake_codex(tmp_path, monkeypatch, catalog={"models": [{"slug": "gpt-6-sol", "apply_patch_tool_type": "freeform"}]})
    executor = _executor("gpt-6-sol", tools=ToolsConfig(exec_enabled=exec_enabled))

    with open(_config(executor)["model_catalog_json"], encoding="utf-8") as handle:
        [model] = json.load(handle)["models"]
    command = executor._build_command()
    disabled = {command[i + 1] for i, arg in enumerate(command) if arg == "--disable"}

    assert (model["apply_patch_tool_type"] == "freeform") is exec_enabled
    assert ("view_image" in disabled) is not exec_enabled


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


def _rendered_prompt(codex_home: str) -> list:
    done = subprocess.run(
        [shutil.which("codex"), "debug", "prompt-input", "hello"],
        env={**os.environ, "CODEX_HOME": codex_home},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(done.stdout)


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI not installed")
def test_the_real_cli_sends_no_interactive_only_prompt_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_PATH", shutil.which("codex"))
    # A skill the run must not be told about.
    skill = tmp_path / "home" / ".agents" / "skills" / "canary-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: canary-skill\ndescription: canary skill description\n---\nbody\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    listed = codex_backend._codex_catalog_models(shutil.which("codex"), str(tmp_path))
    if not listed:
        pytest.skip("this Codex CLI lists no models")
    executor = _executor(listed[0]["slug"])

    items = _rendered_prompt(executor._codex_home)

    tools = {
        tool.get("name")
        for item in items
        if item.get("type") == "additional_tools"
        for group in item["tools"]
        for tool in (group.get("tools") or [group])
    }
    text = json.dumps(items)
    assert tools, "the prompt lists no tools at all"
    assert not tools & {"request_user_input", "request_user_input_async", "sleep"}
    assert "canary-skill" not in text
    assert "<skills_instructions>" not in text
    assert "<permissions instructions>" not in text
    assert "<collaboration_mode>" not in text
    assert "request_plugin_install" not in text
    # The task itself is still sent.
    assert "hello" in text


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI not installed")
def test_the_real_cli_gives_a_run_without_exec_no_file_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_PATH", shutil.which("codex"))
    listed = codex_backend._codex_catalog_models(shutil.which("codex"), str(tmp_path))
    if not listed:
        pytest.skip("this Codex CLI lists no models")
    executor = _executor(listed[0]["slug"], tools=ToolsConfig(exec_enabled=False))
    command = executor._build_command()
    flags = [
        arg
        for i, flag in enumerate(command)
        if flag in {"--disable", "-c"}
        for arg in (flag, command[i + 1])
    ]

    done = subprocess.run(
        [shutil.which("codex"), *flags, "debug", "prompt-input", "hello"],
        env={**os.environ, "CODEX_HOME": executor._codex_home},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    text = done.stdout
    assert "hello" in text
    for tool in ("apply_patch", "view_image", "exec_command", "shell"):
        assert f"### `{tool}`" not in text and f'"name": "{tool}"' not in text
