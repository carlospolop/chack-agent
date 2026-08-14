from types import SimpleNamespace

import pytest

from chack_agent.github_action import _publish_result


def test_publish_result_writes_successful_output(monkeypatch, tmp_path, capsys):
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    _publish_result(SimpleNamespace(output="review complete", error=""))

    assert capsys.readouterr().out == "review complete\n"
    assert "review complete" in github_output.read_text(encoding="utf-8")


def test_publish_result_fails_instead_of_returning_backend_error_as_review(
    monkeypatch, tmp_path, capsys
):
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    result = SimpleNamespace(
        output="ERROR: Codex exec failed (exit=1).",
        error="backend_failure",
    )

    with pytest.raises(SystemExit) as exc_info:
        _publish_result(result)

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "ERROR: Codex exec failed" in captured.out
    assert "Chack Agent failed: backend_failure" in captured.err
    assert "ERROR: Codex exec failed" in github_output.read_text(encoding="utf-8")
