from __future__ import annotations

from pathlib import Path

from chack_agent.backends.codex_backend import (
    CodexExecutor,
    _cleanup_isolated_codex_home,
    _preview_text,
)


def test_failure_preview_preserves_start_and_terminal_error():
    text = "startup:" + ("x" * 100) + ":terminal provider error"

    preview = _preview_text(text, max_chars=80)

    assert len(preview) == 80
    assert preview.startswith("startup:")
    assert preview.endswith(":terminal provider error")
    assert "truncated middle" in preview


def test_cleanup_isolated_codex_home_removes_only_child(tmp_path: Path):
    home_base = tmp_path / "chack"
    target = home_base / "session-1"
    target.mkdir(parents=True)
    (target / "config.toml").write_text("model = 'test'", encoding="utf-8")

    assert _cleanup_isolated_codex_home(str(target), str(home_base)) is True
    assert not target.exists()
    assert home_base.is_dir()


def test_cleanup_isolated_codex_home_refuses_base_and_outside(tmp_path: Path):
    home_base = tmp_path / "chack"
    outside = tmp_path / "outside"
    home_base.mkdir()
    outside.mkdir()

    assert _cleanup_isolated_codex_home(str(home_base), str(home_base)) is False
    assert _cleanup_isolated_codex_home(str(outside), str(home_base)) is False
    assert home_base.is_dir()
    assert outside.is_dir()


def test_executor_cleanup_resets_runtime_paths(tmp_path: Path):
    home_base = tmp_path / "chack"
    target = home_base / "session-2"
    target.mkdir(parents=True)
    executor = object.__new__(CodexExecutor)
    executor._runtime_env_json = (
        '{"CHACK_CODEX_HOME_BASE": "' + str(home_base) + '"}'
    )
    executor._codex_home = str(target)
    executor._output_schema_path = str(target / "output-schema.json")

    assert executor.cleanup_runtime_artifacts() is True
    assert executor._codex_home is None
    assert executor._output_schema_path is None
    assert home_base.is_dir()
