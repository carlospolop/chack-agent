from __future__ import annotations

import http.server
import os
import subprocess
import sys
import threading
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from chack_agent import Chack
from chack_agent.backends.chack_tools_mcp_server import _ServerPolicyState, _register_tools
from chack_agent.config import (
    AgentConfig,
    ChackConfig,
    CredentialsConfig,
    LoggingConfig,
    ModelConfig,
    SessionConfig,
)
from chack_tools.config import ToolsConfig
from chack_tools.telemetry import (
    child_process_env,
    log_event,
    log_tool_started,
    run_with_tool_logging,
)
from chack_tools.telemetry.otel import start_tool_span, end_tool_span
from chack_tools.tool_call_policy import ToolCallPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The tests share one in-process provider, as an application embedding chack
# would install its own; chack then exports nothing itself and uses it.
_EXPORTER = InMemorySpanExporter()
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture(autouse=True)
def _spans(tmp_path, monkeypatch):
    monkeypatch.setenv("CHACK_RUN_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("CHACK_TASK_SESSION_ID", raising=False)
    monkeypatch.delenv("TRACEPARENT", raising=False)
    _EXPORTER.clear()
    yield _EXPORTER


def _finished(name: str):
    return [span for span in _EXPORTER.get_finished_spans() if span.name == name]


class _ToolUsingExecutor:
    """Calls tools the way backends do, then answers."""

    def __init__(self, tools):
        self.tools = tools

    def invoke(self, payload, context=None):
        del payload, context
        steps = []
        for name, args, func in self.tools:
            try:
                result = run_with_tool_logging(name, args, func)
            except Exception as exc:  # the backend reports the failure to the model
                result = f"ERROR: {exc}"
            steps.append((SimpleNamespace(tool=name, tool_input=args), result))
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=50,
            input_tokens_details=SimpleNamespace(cached_tokens=400),
        )
        return {
            "output": "final answer",
            "intermediate_steps": steps,
            "raw_result": SimpleNamespace(raw_responses=[SimpleNamespace(usage=usage)]),
        }


class _InjectedChack(Chack):
    executor = None

    def _get_executor(self, *args, **kwargs):
        return self.executor


def _config() -> ChackConfig:
    return ChackConfig(
        model=ModelConfig(primary="test-model", provider="openai"),
        agent=AgentConfig(self_critique_enabled=False, main_action="audit", sub_action="check"),
        session=SessionConfig(long_term_memory_enabled=False),
        tools=replace(ToolsConfig(), min_tools_used=0, task_steps_manager_enabled=False),
        credentials=CredentialsConfig(),
        logging=LoggingConfig(level="ERROR"),
        system_prompt="system",
        env={},
    )


def _run(tools):
    agent = _InjectedChack(_config())
    agent.executor = _ToolUsingExecutor(tools)
    return agent.run(
        "session-1",
        "find the bug",
        enable_self_critique=False,
        require_task_steps_manager_init_first=False,
    )


def test_a_run_is_one_agent_span_with_its_tools_inside():
    def _search():
        # The tool reports itself too; that must not add a second span.
        log_tool_started("search_web", {"q": "bug"})
        return "found it"

    _run([("search_web", {"q": "bug"}, _search)])

    [run] = _finished("audit.check")
    [tool] = _finished("search_web")
    assert run.attributes["openinference.span.kind"] == "AGENT"
    assert run.attributes["session.id"] == "session-1"
    assert run.attributes["input.value"] == "find the bug"
    assert run.attributes["output.value"] == "final answer"
    assert run.attributes["llm.model_name"] == "test-model"
    assert run.attributes["llm.token_count.prompt"] == 1000
    assert run.attributes["llm.token_count.completion"] == 50
    assert run.attributes["llm.token_count.total"] == 1050
    assert run.attributes["llm.token_count.prompt_details.cache_read"] == 400
    # Context set while the run is already going is on the span too.
    assert run.attributes["chack.context.model"] == "test-model"
    assert run.attributes["chack.context.task_session_id"]
    assert run.attributes["chack.context.max_turns"] > 0
    assert tool.parent.span_id == run.context.span_id
    assert tool.attributes["openinference.span.kind"] == "TOOL"
    assert tool.attributes["output.value"] == "found it"
    assert {event.name for event in run.events} >= {"agent_start", "agent_end"}


def test_a_failing_tool_is_an_error_span_and_the_run_continues():
    def _broken():
        raise TimeoutError("upstream timed out")

    result = _run([("fetch_page", {"url": "https://x"}, _broken)])

    [tool] = _finished("fetch_page")
    assert tool.status.status_code is StatusCode.ERROR
    assert "upstream timed out" in tool.attributes["chack.error"]
    assert result.output == "final answer"
    [run] = _finished("audit.check")
    assert run.status.status_code is not StatusCode.ERROR


def test_tools_of_a_run_with_limits_stay_inside_its_span():
    # Runs with a runtime or cost limit invoke the backend on a worker thread.
    agent = _InjectedChack(_config())
    agent.config.agent.max_runtime_minutes = 5
    agent.config.agent.max_cost_usd = 1.0
    seen = {}

    def _search():
        seen["env"] = child_process_env()
        return "found it"

    agent.executor = _ToolUsingExecutor([("search_web", {"q": "bug"}, _search)])
    agent.run("session-1", "find the bug", enable_self_critique=False,
              require_task_steps_manager_init_first=False)

    [run] = _finished("audit.check")
    [tool] = _finished("search_web")
    assert tool.parent.span_id == run.context.span_id
    # A CLI backend's MCP child gets the run's trace context too.
    assert f"{run.context.trace_id:032x}" in seen["env"]["TRACEPARENT"]


@pytest.mark.parametrize(
    "result",
    [
        "ERROR: no such file",
        "\u274c outcome must be safe, vulnerable, or unresolved",
        '{"error": "credentials unavailable"}',
        '{"success": false, "status": "failed", "exit_code": 2}',
        "An error occurred while running the tool. Please try again.",
    ],
)
def test_a_tool_returning_an_error_is_an_error_span(result):
    run_with_tool_logging("read_file", {"path": "a"}, lambda: result)

    [tool] = _finished("read_file")
    assert tool.status.status_code is StatusCode.ERROR
    assert tool.attributes["chack.error"] == result.splitlines()[0]


def test_a_tool_returning_data_is_not_an_error_span():
    run_with_tool_logging("read_file", {"path": "a"}, lambda: '{"count": 1, "error_count": 0}')

    [tool] = _finished("read_file")
    assert tool.status.status_code is not StatusCode.ERROR


def test_large_tool_results_are_cut_in_the_trace_only(monkeypatch):
    monkeypatch.setenv("CHACK_TELEMETRY_TOOL_RESULT_MAX_BYTES", "1000")
    big = "x" * 50_000

    result = run_with_tool_logging("read_file", {"path": "a"}, lambda: big)

    [tool] = _finished("read_file")
    assert result == big
    assert len(tool.attributes["output.value"]) <= 1000


def test_events_outside_any_span_are_dropped_not_raised():
    assert log_event("compaction", {"tokens": 5}) is False


def test_child_processes_receive_export_settings_and_the_trace_context(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://collector.example/v1/traces")
    monkeypatch.setenv("UNRELATED_SECRET", "nope")
    handle = start_tool_span("exec", {"command": "ls"})
    try:
        env = child_process_env()
    finally:
        end_tool_span(handle, result="ok")

    [span] = _finished("exec")
    assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == "https://collector.example/v1/traces"
    assert format(span.context.trace_id, "032x") in env["TRACEPARENT"]
    assert format(span.context.span_id, "016x") in env["TRACEPARENT"]
    assert "UNRELATED_SECRET" not in env


def test_spans_in_a_child_process_join_the_parent_trace(monkeypatch):
    parent = start_tool_span("parent_tool", {})
    traceparent = child_process_env()["TRACEPARENT"]
    end_tool_span(parent, result="")
    monkeypatch.setenv("TRACEPARENT", traceparent)

    run_with_tool_logging("child_tool", {}, lambda: "ok")

    [parent_span] = _finished("parent_tool")
    [child] = _finished("child_tool")
    assert child.context.trace_id == parent_span.context.trace_id
    assert child.parent.span_id == parent_span.context.span_id


def test_mcp_tool_calls_are_tool_spans():
    from agents import function_tool
    from mcp.server.fastmcp import FastMCP
    import asyncio

    @function_tool(name_override="lookup")
    def _lookup(query: str) -> str:
        """Look something up."""
        return f"answer to {query}"

    server = FastMCP("otel-test")
    _register_tools(
        server,
        [_lookup],
        _ServerPolicyState(
            require_task_steps_manager_init_first=False,
            max_non_task_tools=0,
            session_id="",
            call_policy=ToolCallPolicy(ToolsConfig()),
        ),
    )
    asyncio.run(server.call_tool("lookup", {"query": "x"}))

    [span] = _finished("lookup")
    assert span.attributes["output.value"] == "answer to x"
    assert '"query":"x"' in span.attributes["input.value"]


class _Collector(http.server.BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).received.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        return


def test_config_endpoint_exports_traces_over_otlp_http(tmp_path):
    _Collector.received = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {ROOT!r})
            from chack_tools.telemetry import configure_telemetry, run_with_tool_logging
            from chack_tools.telemetry.otel import start_run_span, end_run_span
            assert configure_telemetry(
                endpoint="http://127.0.0.1:{server.server_port}",
                headers="authorization=Bearer test-key",
                project="chack-tests",
            )
            span = start_run_span(name="demo", session_id="s1", model="m", provider="p",
                                  input_text="hi", log_context={{}})
            run_with_tool_logging("search_web", {{"q": "x"}}, lambda: "ok")
            end_run_span(span, output="bye")
            """
        )
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OTEL_") and key not in {"TRACEPARENT", "TRACESTATE"}
    }
    try:
        subprocess.run([sys.executable, str(script)], env=env, check=True, timeout=60)
    finally:
        server.shutdown()

    assert _Collector.received, "nothing was exported"
    names, projects = set(), set()
    for path, headers, body in _Collector.received:
        assert path == "/v1/traces"
        assert {k.lower(): v for k, v in headers.items()}["authorization"] == "Bearer test-key"
        request = ExportTraceServiceRequest.FromString(body)
        for resource_spans in request.resource_spans:
            for attribute in resource_spans.resource.attributes:
                if attribute.key == "openinference.project.name":
                    projects.add(attribute.value.string_value)
            for scope_spans in resource_spans.scope_spans:
                names.update(span.name for span in scope_spans.spans)
    assert names == {"demo", "search_web"}
    assert projects == {"chack-tests"}


def test_nothing_is_exported_or_printed_without_an_endpoint(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {ROOT!r})
            from chack_tools.telemetry import configure_telemetry, log_event, run_with_tool_logging
            assert configure_telemetry() is False
            run_with_tool_logging("search_web", {{"q": "x"}}, lambda: "ok")
            log_event("agent_end", {{"error": ""}})
            """
        )
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}

    done = subprocess.run(
        [sys.executable, str(script)], env=env, check=True, timeout=60, capture_output=True, text=True
    )

    assert done.stdout == ""
