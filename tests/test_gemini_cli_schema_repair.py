"""Gemini CLI answers that drift from the output schema get one repair turn.

A fake ``gemini`` executable records its arguments and replays a scripted
stream-json run per call, so the real subprocess path is exercised.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

from chack_agent.backends import gemini_cli_backend
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

FAKE_GEMINI = textwrap.dedent(
    """
    import json, os, sys
    record = os.environ["FAKE_GEMINI_RECORD"]
    calls = []
    if os.path.exists(record):
        with open(record) as handle:
            calls = json.load(handle)
    calls.append(sys.argv[1:])
    with open(record, "w") as handle:
        json.dump(calls, handle)
    settings_path = os.path.join(os.environ["GEMINI_CLI_HOME"], ".gemini", "settings.json")
    with open(settings_path) as handle:
        turn_budget = json.load(handle)["model"]["maxSessionTurns"]
    with open(record + ".turns", "a") as handle:
        handle.write(f"{turn_budget}\\n")
    with open(record + ".prompts", "a") as handle:
        handle.write(json.dumps(sys.stdin.read()) + "\\n")
    if os.path.exists("/proc/self/oom_score_adj"):
        import subprocess
        started = subprocess.run(
            [sys.executable, "-c", "print(open('/proc/self/oom_score_adj').read().strip())"],
            capture_output=True, text=True,
        ).stdout.strip()
        with open("/proc/self/oom_score_adj") as handle:
            own = handle.read().strip()
        with open(record + ".oom", "a") as handle:
            handle.write(json.dumps({"cli": own, "started": started}) + "\\n")
    with open(os.environ["FAKE_GEMINI_SCENARIO"]) as handle:
        runs = json.load(handle)
    run = runs[min(len(calls), len(runs)) - 1]
    for line in run.get("stderr", []):
        print(line, file=sys.stderr, flush=True)
    for event in run.get("events", []):
        print(json.dumps(event), flush=True)
    sys.exit(run.get("exit", 0))
    """
)

SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "object",
            "properties": {"service_id": {"type": "string"}, "assessment": {"type": "string"}},
            "required": ["service_id", "assessment"],
            "additionalProperties": False,
        }
    },
    "required": ["result"],
    "additionalProperties": False,
}
VALID = {"result": {"service_id": "svc-1", "assessment": "exploitable"}}
DRIFTED = {"findings": [{"observation": "open port"}], "summary": "done"}


def _run(answer, *, session="gem-1", tool_id="t1", input_tokens=100, output_tokens=10):
    return {
        "events": [
            {"type": "init", "session_id": session},
            {"type": "tool_use", "tool_id": tool_id, "tool_name": "exec", "parameters": {"command": "id"}},
            {"type": "tool_result", "tool_id": tool_id, "status": "success", "output": "uid=0"},
            {"type": "message", "role": "assistant", "content": answer},
            {
                "type": "result",
                "status": "success",
                "stats": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cached": 0},
            },
        ]
    }


@pytest.fixture
def fake_gemini(tmp_path, monkeypatch):
    script = tmp_path / "gemini"
    script.write_text(f"#!{sys.executable}\n{FAKE_GEMINI}", encoding="utf-8")
    script.chmod(0o755)
    record = tmp_path / "calls.json"
    scenario = tmp_path / "scenario.json"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-homes"))
    monkeypatch.setenv("GEMINI_CLI_PATH", str(script))
    monkeypatch.setenv("FAKE_GEMINI_RECORD", str(record))
    monkeypatch.setenv("FAKE_GEMINI_SCENARIO", str(scenario))
    for name in ("GOOGLE_GEMINI_BASE_URL", "CHACK_GEMINI_EXEC_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    class Fake:
        def script(self, runs):
            scenario.write_text(json.dumps(runs), encoding="utf-8")

        @property
        def calls(self):
            return json.loads(record.read_text(encoding="utf-8")) if record.exists() else []

        @property
        def turn_budgets(self):
            path = tmp_path / "calls.json.turns"
            return [int(line) for line in path.read_text(encoding="utf-8").split()] if path.exists() else []

        @property
        def oom_scores(self):
            path = tmp_path / "calls.json.oom"
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            return [json.loads(line) for line in lines]

        @property
        def prompts(self):
            path = tmp_path / "calls.json.prompts"
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            return [json.loads(line) for line in lines]

    return Fake()


def _executor(schema=SCHEMA):
    config = ChackConfig(
        model=ModelConfig(primary="gemini-3-flash", provider="gemini"),
        agent=AgentConfig(main_action="test", sub_action="test", output_schema_json=schema, output_schema_name="answer"),
        session=SessionConfig(max_turns=4),
        tools=ToolsConfig(exec_enabled=True, task_steps_manager_enabled=False, native_web_search_enabled=False),
        credentials=CredentialsConfig(gemini_api_key="fake-key"),
        logging=LoggingConfig(),
        system_prompt="system",
        env={},
    )
    return gemini_cli_backend.build_executor(
        config, system_prompt="system", max_turns=4, memory_max_messages=10, memory_reset_to_messages=5
    )


def test_drifted_answer_is_repaired_in_the_same_session_and_both_turns_are_counted(fake_gemini):
    fake_gemini.script([
        _run(json.dumps(DRIFTED), input_tokens=1000, output_tokens=50),
        _run(json.dumps(VALID), tool_id="t2", input_tokens=1100, output_tokens=20),
    ])

    result = _executor().invoke({"input": "Assess the service."})

    assert json.loads(result["output"]) == VALID
    first, second = fake_gemini.calls
    assert "-r" not in first
    assert second[second.index("-r") + 1] == "gem-1"
    usage = [response["usage"] for response in result["raw_result"].raw_responses]
    assert sum(item["input_tokens"] for item in usage) == 2100
    assert sum(item["output_tokens"] for item in usage) == 70
    assert [step[0].tool_input["tool_id"] for step in result["intermediate_steps"]] == ["t1", "t2"]


def test_valid_answer_is_not_repaired(fake_gemini):
    fake_gemini.script([_run("Done:\n```json\n" + json.dumps(VALID) + "\n```")])

    result = _executor().invoke({"input": "Assess the service."})

    assert json.loads(result["output"]) == VALID
    assert len(fake_gemini.calls) == 1


def test_answer_still_drifting_after_the_repair_is_returned_as_given(fake_gemini):
    fake_gemini.script([_run(json.dumps(DRIFTED)), _run("still prose")])

    result = _executor().invoke({"input": "Assess the service."})

    assert result["output"] == "still prose"
    assert len(fake_gemini.calls) == 2


def test_backend_error_is_not_repaired(fake_gemini):
    fake_gemini.script([{"events": [{"type": "init", "session_id": "gem-1"}], "exit": 3}])

    result = _executor().invoke({"input": "Assess the service."})

    assert result["output"].startswith("ERROR: Gemini exec failed (exit=3)")
    assert len(fake_gemini.calls) == 1


def test_without_a_session_to_resume_the_answer_is_returned_unrepaired(fake_gemini):
    fake_gemini.script([_run(json.dumps(DRIFTED), session="")])

    result = _executor().invoke({"input": "Assess the service."})

    assert json.loads(result["output"]) == DRIFTED
    assert len(fake_gemini.calls) == 1


def test_without_a_schema_any_answer_is_final(fake_gemini):
    fake_gemini.script([_run("plain prose")])

    result = _executor(schema=None).invoke({"input": "Say something."})

    assert result["output"] == "plain prose"
    assert len(fake_gemini.calls) == 1


def _turn_limit_run(session="gem-1"):
    return {
        "events": [
            {"type": "init", "session_id": session},
            {"type": "tool_use", "tool_id": "t1", "tool_name": "exec", "parameters": {"command": "id"}},
            {"type": "tool_result", "tool_id": "t1", "status": "success", "output": "uid=0"},
            {
                "type": "result",
                "status": "error",
                "error": {"type": "FatalTurnLimitedError", "message": "Reached max session turns for this session."},
                "stats": {"input_tokens": 500, "output_tokens": 5, "cached": 0},
            },
        ]
    }


def test_run_stopped_at_the_turn_limit_resumes_once_for_the_final_answer(fake_gemini):
    fake_gemini.script([_turn_limit_run(), _run(json.dumps(VALID), tool_id="t2")])

    result = _executor().invoke({"input": "Assess the service."})

    assert json.loads(result["output"]) == VALID
    first, second = fake_gemini.calls
    assert second[second.index("-r") + 1] == "gem-1"
    assert fake_gemini.turn_budgets == [4, 2]
    assert "final answer" in fake_gemini.prompts[1]
    assert [step[0].tool_input["tool_id"] for step in result["intermediate_steps"]] == ["t1", "t2"]


def test_next_request_gets_the_full_turn_budget_again(fake_gemini):
    fake_gemini.script([_turn_limit_run(), _run(json.dumps(VALID)), _run(json.dumps(VALID))])
    executor = _executor()

    executor.invoke({"input": "Assess the service."})
    executor.invoke({"input": "Assess another service."})

    assert fake_gemini.turn_budgets == [4, 2, 4]


def test_turn_limit_again_on_the_final_answer_run_is_reported(fake_gemini):
    fake_gemini.script([_turn_limit_run(), _turn_limit_run()])

    result = _executor().invoke({"input": "Assess the service."})

    assert result["output"].startswith("ERROR: Gemini result error: Reached max session turns")
    assert len(fake_gemini.calls) == 2


def test_turn_limit_without_a_session_to_resume_is_reported(fake_gemini):
    fake_gemini.script([_turn_limit_run(session="")])

    result = _executor().invoke({"input": "Assess the service."})

    assert result["output"].startswith("ERROR: Gemini result error: Reached max session turns")
    assert len(fake_gemini.calls) == 1


def test_turn_ending_without_an_answer_resumes_to_finish_instead_of_reshaping_cli_events(fake_gemini):
    no_answer = _run("")
    no_answer["events"] = [event for event in no_answer["events"] if event["type"] != "message"]
    fake_gemini.script([no_answer, _run(json.dumps(VALID), tool_id="t2")])

    result = _executor().invoke({"input": "Assess the service."})

    assert json.loads(result["output"]) == VALID
    first, second = fake_gemini.calls
    assert second[second.index("-r") + 1] == "gem-1"
    assert "stats" not in fake_gemini.prompts[1] and "tool_id" not in fake_gemini.prompts[1]



def test_cli_warnings_printed_on_a_turn_without_an_answer_are_not_taken_as_the_answer(fake_gemini):
    no_answer = _run("")
    no_answer["events"] = [event for event in no_answer["events"] if event["type"] != "message"]
    no_answer["stderr"] = ["[WARN] Skipping unreadable directory: /proc/1/root", "Loaded cached credentials."]
    fake_gemini.script([no_answer, _run(json.dumps(VALID), tool_id="t2")])

    result = _executor().invoke({"input": "Assess the service."})

    assert json.loads(result["output"]) == VALID
    assert len(fake_gemini.calls) == 2
    # The model is asked to finish its task, not to reshape the CLI's warnings.
    assert fake_gemini.prompts[1].startswith("Your reply is not the final answer")


@pytest.mark.skipif(not os.path.exists("/proc/self/oom_score_adj"), reason="needs Linux /proc")
def test_out_of_memory_kills_pick_the_cli_and_what_it_starts_before_the_host_process(fake_gemini):
    fake_gemini.script([_run(json.dumps(VALID))])
    with open("/proc/self/oom_score_adj") as handle:
        host_score = handle.read().strip()

    result = _executor().invoke({"input": "Assess the service."})

    assert json.loads(result["output"]) == VALID
    assert fake_gemini.oom_scores == [{"cli": "500", "started": "500"}]
    with open("/proc/self/oom_score_adj") as handle:
        assert handle.read().strip() == host_score

