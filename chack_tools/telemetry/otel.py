"""OpenTelemetry traces for chack runs and tool calls.

Every ``Chack.run`` is an AGENT span and every tool call a TOOL span, using
the OpenInference attribute names that LLM trace viewers (Arize Phoenix,
Langfuse, ...) understand. Everything else chack used to log becomes an event
on the current span.

Export is configured with the standard OpenTelemetry environment variables:

* ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` (or ``OTEL_EXPORTER_OTLP_ENDPOINT``):
  where to send traces, over OTLP/HTTP. Nothing is exported without it.
* ``OTEL_EXPORTER_OTLP_HEADERS``: for example ``authorization=Bearer <key>``.
* ``OTEL_SERVICE_NAME`` and ``OTEL_RESOURCE_ATTRIBUTES``
  (``openinference.project.name=<project>`` picks the Phoenix project).

When the host application already installed its own tracer provider, chack's
spans go there and chack installs nothing. Child processes (CLI backends and
their chack MCP tool server) receive the same variables plus ``TRACEPARENT``,
so their spans join the run's trace.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

try:  # pragma: no cover - exercised by the import itself
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.propagate import extract, inject
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover - telemetry is optional at runtime
    otel_context = None
    trace = None

_TRACER_NAME = "chack-agent"
_DEFAULT_SERVICE_NAME = "chack-agent"
_PROJECT_ATTRIBUTE = "openinference.project.name"
_SPAN_KIND = "openinference.span.kind"

# Bounds on what one span may carry, so traces stay cheap to send and store.
_VALUE_MAX_CHARS_ENV = "CHACK_TELEMETRY_MAX_VALUE_CHARS"
_TOOL_RESULT_MAX_BYTES_ENV = "CHACK_TELEMETRY_TOOL_RESULT_MAX_BYTES"
_DEFAULT_VALUE_MAX_CHARS = 16000
_DEFAULT_TOOL_RESULT_MAX_BYTES = 8192
_EVENT_VALUE_MAX_CHARS = 4000

# Variables a child process needs to export into the same place and trace.
OTEL_CHILD_ENV_NAMES = (
    "OTEL_SDK_DISABLED",
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TIMEOUT",
    "OTEL_EXPORTER_OTLP_CERTIFICATE",
    "OTEL_SERVICE_NAME",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_BSP_SCHEDULE_DELAY",
    "CHACK_OTEL_LLM_SPANS",
    _VALUE_MAX_CHARS_ENV,
    _TOOL_RESULT_MAX_BYTES_ENV,
    "TRACEPARENT",
    "TRACESTATE",
)

_LOCK = threading.Lock()
_PROVIDER: Any = None
_SETUP_DONE = False


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(str(os.environ.get(name, "") or "").strip() or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def export_configured() -> bool:
    """Whether the environment asks for traces to be exported."""
    if trace is None or _truthy(os.environ.get("OTEL_SDK_DISABLED")):
        return False
    if str(os.environ.get("OTEL_TRACES_EXPORTER", "") or "").strip().lower() == "none":
        return False
    return bool(
        str(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "") or "").strip()
        or str(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "") or "").strip()
    )


def configure_telemetry(
    *,
    endpoint: str = "",
    headers: str = "",
    project: str = "",
    service_name: str = "",
) -> bool:
    """Fill the OTel environment from chack config, then set up export.

    Values already in the environment win, so deployments can always override
    a config file. Returns whether chack's own exporter is installed.
    """
    endpoint = str(endpoint or "").strip()
    if endpoint:
        os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", _traces_url(endpoint))
    if str(headers or "").strip():
        os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", str(headers).strip())
    if str(service_name or "").strip():
        os.environ.setdefault("OTEL_SERVICE_NAME", str(service_name).strip())
    project = str(project or "").strip()
    if project and _PROJECT_ATTRIBUTE not in str(os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")):
        existing = str(os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "") or "").strip().strip(",")
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(
            part for part in (existing, f"{_PROJECT_ATTRIBUTE}={project}") if part
        )
    return ensure_tracer_provider()


def _traces_url(endpoint: str) -> str:
    """Accept either a collector base URL or the full ``/v1/traces`` URL."""
    url = endpoint.rstrip("/")
    return url if url.endswith("/v1/traces") else f"{url}/v1/traces"


def _host_has_provider() -> bool:
    provider = trace.get_tracer_provider()
    return type(provider).__name__ not in {"ProxyTracerProvider", "NoOpTracerProvider"}


def ensure_tracer_provider() -> bool:
    """Install an OTLP exporter once per process when one is configured."""
    global _PROVIDER, _SETUP_DONE
    if _SETUP_DONE:
        return _PROVIDER is not None
    with _LOCK:
        if _SETUP_DONE:
            return _PROVIDER is not None
        _SETUP_DONE = True
        if not export_configured() or _host_has_provider():
            return False
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except Exception:
            return False
        resource = Resource.create(
            {"service.name": os.environ.get("OTEL_SERVICE_NAME") or _DEFAULT_SERVICE_NAME}
        )
        if _PROJECT_ATTRIBUTE not in resource.attributes:
            resource = resource.merge(
                Resource({_PROJECT_ATTRIBUTE: resource.attributes.get("service.name", _DEFAULT_SERVICE_NAME)})
            )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _PROVIDER = provider
        _instrument_llm_calls(provider)
        return True


def _instrument_llm_calls(provider: Any) -> None:
    """Add per-LLM-call spans from OpenInference instrumentors when asked.

    Off by default: those spans carry whole prompts, which multiplies trace
    size. Enable with ``CHACK_OTEL_LLM_SPANS=1`` and the instrumentor packages.
    """
    if not _truthy(os.environ.get("CHACK_OTEL_LLM_SPANS")):
        return
    for module, name in (
        ("openinference.instrumentation.openai_agents", "OpenAIAgentsInstrumentor"),
        ("openinference.instrumentation.langchain", "LangChainInstrumentor"),
    ):
        try:
            instrumentor = getattr(__import__(module, fromlist=[name]), name)
            instrumentor().instrument(tracer_provider=provider)
        except Exception:
            continue


def flush_in_background(timeout_millis: int = 5000) -> None:
    """Send finished spans now without blocking the caller."""
    provider = _PROVIDER
    if provider is None:
        return
    threading.Thread(
        target=lambda: provider.force_flush(timeout_millis),
        name="chack-otel-flush",
        daemon=True,
    ).start()


def _tracer() -> Any:
    ensure_tracer_provider()
    return trace.get_tracer(_TRACER_NAME)


def _env_parent() -> Any:
    """The trace context a parent process passed in ``TRACEPARENT``."""
    carrier = {
        key.lower(): value
        for key in ("TRACEPARENT", "TRACESTATE")
        if (value := str(os.environ.get(key, "") or "").strip())
    }
    return extract(carrier) if carrier else None


def _parent_context() -> Any:
    """Current span if any, else the parent process's span, else a new trace."""
    if trace.get_current_span().get_span_context().is_valid:
        return None
    return _env_parent()


def child_process_env() -> Dict[str, str]:
    """OTel variables for a child process, including the current trace context."""
    env = {name: os.environ[name] for name in OTEL_CHILD_ENV_NAMES if os.environ.get(name)}
    if trace is not None and trace.get_current_span().get_span_context().is_valid:
        carrier: Dict[str, str] = {}
        inject(carrier)
        for key, value in carrier.items():
            env[key.upper()] = value
    return env


# ---------------------------------------------------------------------------
# Attribute encoding
# ---------------------------------------------------------------------------


def _truncate(value: str, max_chars: int) -> str:
    value = value.replace("\x00", "")
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)] + "..."


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def _attribute_value(value: Any, max_chars: int) -> Any:
    """Encode ``value`` as an OTel attribute: primitives as-is, the rest as JSON."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return [_truncate(v, max_chars) for v in value]
    return _truncate(_json(value), max_chars)


def _set_attributes(span: Any, attributes: Mapping[str, Any], max_chars: int) -> None:
    for key, value in attributes.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        span.set_attribute(key, _attribute_value(value, max_chars))


def _value_max_chars() -> int:
    return _int_env(_VALUE_MAX_CHARS_ENV, _DEFAULT_VALUE_MAX_CHARS, low=512, high=1_000_000)


def _tool_result_max_chars() -> int:
    return _int_env(_TOOL_RESULT_MAX_BYTES_ENV, _DEFAULT_TOOL_RESULT_MAX_BYTES, low=512, high=65536)


def _context_attributes(log_context: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        f"chack.context.{key}": value
        for key, value in log_context.items()
        if not str(key).startswith("_") and not callable(value)
    }


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


@dataclass
class SpanHandle:
    """A started span and the context token that made it current."""

    span: Any = None
    token: Any = None
    kind: str = ""
    name: str = ""
    owns_span: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def recording(self) -> bool:
        return self.span is not None


_NO_SPAN = SpanHandle()


def _start(name: str, kind: str, attributes: Mapping[str, Any], max_chars: int) -> SpanHandle:
    if trace is None:
        return _NO_SPAN
    try:
        span = _tracer().start_span(name, context=_parent_context())
        span.set_attribute(_SPAN_KIND, kind)
        _set_attributes(span, attributes, max_chars)
        token = otel_context.attach(trace.set_span_in_context(span))
        return SpanHandle(span=span, token=token, kind=kind, name=name)
    except Exception:
        return _NO_SPAN


def _end(handle: SpanHandle, *, error: str = "") -> None:
    if not handle.recording or not handle.owns_span:
        return
    try:
        if error:
            handle.span.set_attribute("chack.error", _truncate(str(error), _EVENT_VALUE_MAX_CHARS))
            handle.span.set_status(Status(StatusCode.ERROR, _truncate(str(error), 500)))
        handle.span.end()
    finally:
        try:
            otel_context.detach(handle.token)
        except Exception:
            # Ended from another context than it started in; the span is
            # already closed, only the context restore is skipped.
            pass


def start_run_span(
    *,
    name: str,
    session_id: str,
    model: str,
    provider: str,
    input_text: str,
    log_context: Mapping[str, Any],
) -> SpanHandle:
    """Start the AGENT span of one ``Chack.run`` and make it current."""
    return _start(
        name or "chack.run",
        "AGENT",
        {
            "session.id": session_id,
            "llm.model_name": model,
            "llm.provider": provider,
            "input.value": input_text,
            **_context_attributes(log_context),
        },
        _value_max_chars(),
    )


def end_run_span(handle: SpanHandle, *, output: Any = None, error: str = "") -> None:
    if handle.recording and output not in (None, ""):
        _set_attributes(handle.span, {"output.value": output}, _value_max_chars())
    top_level = handle.recording and not _env_parent() and not getattr(
        getattr(handle.span, "parent", None), "is_valid", False
    )
    _end(handle, error=error)
    if top_level:
        flush_in_background()


def start_tool_span(tool: str, tool_input: Any) -> SpanHandle:
    """Start the TOOL span of one call, unless that call already has one.

    A tool runs through several layers that each report it (the MCP proxy,
    the backend's tool wrapper, the tool's own logging). The outermost one
    creates the span; inner reports of the same tool reuse it.
    """
    if trace is None:
        return _NO_SPAN
    current = trace.get_current_span()
    attributes = getattr(current, "attributes", None) or {}
    if (
        current.get_span_context().is_valid
        and attributes.get(_SPAN_KIND) == "TOOL"
        and attributes.get("tool.name") == tool
    ):
        return SpanHandle(span=current, kind="TOOL", name=tool, owns_span=False)
    return _start(
        tool or "tool",
        "TOOL",
        {"tool.name": tool, "input.value": tool_input},
        _value_max_chars(),
    )


def end_tool_span(handle: SpanHandle, *, result: Any = None, error: str = "") -> None:
    if handle.recording and handle.owns_span and result is not None:
        _set_attributes(handle.span, {"output.value": result}, _tool_result_max_chars())
    _end(handle, error=error)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

# agent_end fields that become first-class span attributes.
_RUN_END_ATTRIBUTES = {
    "prompt_tokens": "llm.token_count.prompt",
    "completion_tokens": "llm.token_count.completion",
    "cached_prompt_tokens": "llm.token_count.prompt_details.cache_read",
    "cache_write_prompt_tokens": "llm.token_count.prompt_details.cache_write",
    "total_cost": "chack.cost_usd",
    "main_cost": "chack.main_cost_usd",
    "nested_cost": "chack.nested_cost_usd",
    "cache_hit_rate": "chack.cache_hit_rate",
    "tools_used": "chack.tools_used",
    "rounds_used": "chack.rounds_used",
    "limit_reached": "chack.limit_reached",
    "time_to_first_token_seconds": "chack.time_to_first_token_seconds",
}


def record_event(event_type: str, payload: Mapping[str, Any]) -> bool:
    """Attach a former log event to the current span."""
    if trace is None:
        return False
    span = trace.get_current_span()
    if not span.get_span_context().is_valid or not span.is_recording():
        return False
    try:
        if event_type == "agent_end":
            _record_run_end(span, payload)
        elif event_type == "agent_error":
            error = str(payload.get("error") or "")
            span.set_attribute("chack.error", _truncate(error, _EVENT_VALUE_MAX_CHARS))
            span.set_status(Status(StatusCode.ERROR, _truncate(error, 500)))
        attributes = {
            key: _attribute_value(value, _EVENT_VALUE_MAX_CHARS)
            for key, value in payload.items()
            if value is not None and value != "" and value != [] and value != {}
        }
        span.add_event(event_type, attributes=attributes)
        return True
    except Exception:
        return False


def _record_run_end(span: Any, payload: Mapping[str, Any]) -> None:
    attributes = {
        target: payload.get(source)
        for source, target in _RUN_END_ATTRIBUTES.items()
        if payload.get(source) not in (None, "")
    }
    prompt = int(payload.get("prompt_tokens") or 0)
    completion = int(payload.get("completion_tokens") or 0)
    if prompt or completion:
        attributes["llm.token_count.total"] = prompt + completion
    tool_counts = payload.get("tool_counts")
    if isinstance(tool_counts, Mapping) and tool_counts:
        attributes["chack.tool_counts"] = dict(tool_counts)
    breakdown = payload.get("cost_breakdown")
    if breakdown:
        attributes["chack.cost_breakdown"] = breakdown
    error = str(payload.get("error") or "")
    if error:
        attributes["chack.error"] = error
        span.set_status(Status(StatusCode.ERROR, _truncate(error, 500)))
    _set_attributes(span, attributes, _EVENT_VALUE_MAX_CHARS)

