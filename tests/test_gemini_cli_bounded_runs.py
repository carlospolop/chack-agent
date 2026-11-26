"""A long Gemini CLI tool loop is compressed instead of growing without bound.

The installed CLI talks to a local server standing in for the model. It keeps
calling a tool and reports a prompt past the agent's compaction point, until
the CLI sends it a history summary; after that it answers. No model is called.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chack_agent.backends import gemini_cli_backend
from chack_agent.backends.gemini_cli_backend import build_executor
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
    ToolsConfig,
)

_GEMINI = shutil.which(os.environ.get("GEMINI_CLI_PATH", "") or "gemini")

pytestmark = pytest.mark.skipif(_GEMINI is None, reason="Gemini CLI not installed")

# Past the agent's compaction point (350k x 0.75) and below the CLI's own
# default (half of a 1,048,576-token window).
_PROMPT_TOKENS = 300_000


def _answer(parts: list[dict], prompt_tokens: int = 0) -> dict:
    answer: dict = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}
    if prompt_tokens:
        answer["usageMetadata"] = {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": 1,
            "totalTokenCount": prompt_tokens + 1,
        }
    return answer


def _run(tmp_path, monkeypatch, *, max_context_tokens: int, max_turns: int = 12, growth: int = _PROMPT_TOKENS,
         answer_after_summary: bool = True):
    turns: list[dict] = []
    summaries: list[int] = []  # how many turns preceded each summary

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            last = (payload.get("contents") or [{}])[-1]
            if "<state_snapshot>" in json.dumps(last.get("parts", [])):
                summaries.append(len(turns))
                answer = _answer([{"text": "<state_snapshot>listed files</state_snapshot>"}])
            elif summaries and answer_after_summary or "final answer now" in json.dumps(last).lower():
                turns.append(payload)
                answer = _answer([{"text": "done"}])
            else:
                turns.append(payload)
                # The prompt grows by `growth` tokens a turn and restarts small after a summary.
                since = len(turns) - (summaries[-1] if summaries else 0)
                answer = _answer(
                    [{"functionCall": {"name": "list_files", "args": {"page": len(turns)}}}], growth * since
                )
            body = json.dumps(answer).encode()
            if "alt=sse" in self.path:
                self._send(b"data: " + body + b"\n\n", "text/event-stream")
            else:
                self._send(body, "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(gemini_cli_backend, "_RUN_TURNS", 3)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("CHACK_GEMINI_HOME", str(tmp_path / "gemini-homes"))
        monkeypatch.setenv("GEMINI_CLI_PATH", _GEMINI)
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        config = ChackConfig(
            model=ModelConfig(primary="long-run-model", provider="gemini", max_context_tokens=max_context_tokens),
            agent=AgentConfig(main_action="test", sub_action="test", compaction_threshold_ratio=0.75),
            session=SessionConfig(max_turns=max_turns),
            tools=ToolsConfig(exec_enabled=False, task_steps_manager_enabled=False),
            credentials=CredentialsConfig(gemini_api_key="test-key"),
            logging=LoggingConfig(),
            system_prompt="system",
            env={},
        )
        executor = build_executor(
            config, system_prompt="system", max_turns=max_turns, memory_max_messages=10, memory_reset_to_messages=5
        )
        executor._gemini_base_url = f"http://127.0.0.1:{server.server_address[1]}"
        executor._compaction_threshold_ratio = 0.75
        result = executor.invoke({"input": "List every file of the repository."})
    finally:
        server.shutdown()
    return result, turns, summaries


def test_a_tool_loop_past_the_compaction_point_is_summarized_and_continues_from_the_summary(tmp_path, monkeypatch):
    result, turns, summaries = _run(tmp_path, monkeypatch, max_context_tokens=350_000)

    assert summaries, "the CLI never summarized the growing history"
    assert result["output"] == "done"
    before, after = turns[summaries[0] - 1], turns[summaries[0]]
    assert "listed files" in json.dumps(after["contents"])
    assert len(json.dumps(after["contents"])) < len(json.dumps(before["contents"]))


def test_a_tool_loop_below_the_compaction_point_keeps_its_history(tmp_path, monkeypatch):
    # At most 12 x 40k = 480k tokens, under the 600k compaction point.
    _result, turns, summaries = _run(tmp_path, monkeypatch, max_context_tokens=800_000, growth=40_000)

    assert not summaries
    contents = [json.dumps(turn["contents"]) for turn in turns]
    assert all(len(a) <= len(b) for a, b in zip(contents, contents[1:]))


def test_a_gradually_growing_loop_is_summarized_once_it_reaches_the_compaction_point(tmp_path, monkeypatch):
    # 30k tokens a turn: under the 262,500-token point for eight turns, past it on the ninth. The CLI
    # learns a turn's prompt size from its answer, so it may summarize one turn later.
    result, turns, summaries = _run(
        tmp_path, monkeypatch, max_context_tokens=350_000, max_turns=40, growth=30_000
    )

    assert result["output"] == "done"
    assert summaries and summaries[0] in (9, 10)


def test_continued_runs_share_one_turn_budget(tmp_path, monkeypatch):
    from chack_agent.backends.final_answer import FINAL_ANSWER_PROMPT

    _result, turns, summaries = _run(
        tmp_path, monkeypatch, max_context_tokens=800_000, max_turns=7, growth=1_000
    )

    assert not summaries
    final = [i for i, turn in enumerate(turns) if FINAL_ANSWER_PROMPT[:40] in json.dumps(turn["contents"][-1])]
    # Runs of 3, 3 and 1 turns, then the final-answer request, as one 7-turn run would.
    assert final == [7]


def test_a_run_that_kept_no_session_is_not_continued(monkeypatch):
    executor = gemini_cli_backend.GeminiCliExecutor.__new__(gemini_cli_backend.GeminiCliExecutor)
    executor._conversation, executor._memory_limit = [], 0
    executor._max_turns, executor._max_context_tokens, executor._compaction_threshold_ratio = 100, 350_000, 0.75
    executor._output_schema_json, executor._output_schema_name, executor._output_schema_strict = "", "o", False
    executor._gemini_session_id = None
    executor._compose_prompt = lambda text: text
    prompts = []

    class Raw:
        raw_responses: list = []

    executor._run_gemini = lambda prompt: (prompts.append(prompt) or gemini_cli_backend._TURN_LIMIT_ERROR, [], Raw())

    result = executor.invoke({"input": "Question"})

    assert prompts == ["Question"]
    assert result["output"].startswith(gemini_cli_backend._TURN_LIMIT_ERROR)
