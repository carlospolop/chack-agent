"""Local relay between the Gemini CLI and an LLM gateway's Gemini API.

LiteLLM translates its Gemini API for every model it serves, but when it
streams (``:streamGenerateContent``) a model that is not Gemini its
``usageMetadata`` is all zeros, so a run's tokens and cost are lost. The same
request without streaming (``:generateContent``) reports the real usage.

The relay listens on 127.0.0.1 and forwards requests to the gateway. A
streaming request is sent as ``:generateContent`` and its JSON answer is
returned to the CLI as one server-sent event. If a gateway explicitly rejects
one of the Gemini CLI's temperature, topP, or topK fields for a model, a
bounded retry omits that field. A transient gateway failure (a 502, 503 or
504 answer such as an upstream timeout, or a connection dropped or refused
before the answer was read) is sent again with jittered exponential backoff, a
bounded number of times (``transient_retry.TransientRetry``); nothing of the
answer has reached the CLI yet, so a retry is safe. Other gateway errors, and a
transient one that outlasts the retries, pass through with their status and body.

The Gemini CLI also calls Google models of its own for helper work (chat
compression, loop detection, routing, summaries), which a gateway may not
serve or a profile never chose. The relay sends every call to the run's model.

An unstreamed answer can take longer than the Gemini CLI waits for response
headers, so a slow streaming call gets its event-stream headers early and
blank keepalive lines until the answer, or the gateway's error, arrives.

Each gateway call has its own time limit (``CHACK_GEMINI_RELAY_REQUEST_TIMEOUT_SECONDS``,
never more than the run's limit), well below a run's. A streaming model call
with no answer after half that limit (``_HEDGE_AFTER_SHARE``) gets one duplicate
request sent alongside it (a hedge), while the first keeps waiting: whichever
answers first with a success reaches the CLI and the other's connection is
closed. A call never has more than these two requests in flight, and both end at
the call's deadline, the hedge point plus the whole limit; with no answer by
then the CLI gets a 504 ``DEADLINE_EXCEEDED`` error instead of keepalives until
the whole run times out. A request that fails with an error before the hedge
point is returned as it is; after it, the other request is waited for. Each
attempt's socket timeout is capped at the time left before the deadline
(``_MIN_ATTEMPT_TIMEOUT_SECONDS`` at least), and a request the relay stopped
waiting for starts no further retry, so it ends by the deadline instead of
running on, billed, upstream. A call the relay itself gave up on at its socket
timeout is logged as ``outcome=timeout``, not as a gateway 5xx, and a request
closed because the other one answered as ``outcome=cancelled``; a hedged call
also logs one ``Gemini gateway hedge`` line (``primary_won``, ``hedge_won``,
``both_timeout``, ``both_failed`` or ``abandoned``).
Answered calls are not logged one by one; at most once per
``progress_every_seconds`` an INFO line counts them, with the last request's
size and prompt tokens, so a quiet log still shows the run is being answered.

The relay also keeps the token usage of every answer it relays, so a run that
is stopped before the CLI prints its closing stats still counts the calls it
made. LiteLLM's Gemini API omits ``cachedContentTokenCount`` for models that are
not Gemini even when the provider served most of the prompt from its prefix
cache, so every token would be billed at the full input price. When an answer
omits that field, the relay estimates the cached tokens from the longest prefix
the request shares with a recent request to the same gateway and model
(``PrefixCacheEstimator``); an answer that reports the field is trusted as is. A CLI that goes away while the relay answers it (a cancelled or timed-out
run) is no error: the answer is dropped quietly.

Some models (DeepSeek behind LiteLLM's Gemini route) answer a request that
offers tools with their own tool-call markup in the text (``<｜DSML｜ calls>``)
instead of function calls, so the CLI would run no tool. The relay reads each
answer whole before the CLI gets it, so it turns such a block into
``functionCall`` parts (``_with_text_tool_calls``) when the block ends the text
outside a code fence and every call in it names a tool the request declared
with valid arguments; otherwise the answer is passed on unchanged.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import queue
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, NamedTuple

from . import gateway_pressure
from .transient_retry import TRANSIENT_STATUSES, TransientRetry, retry_after_seconds


_LOGGER = logging.getLogger("chack.gemini_gateway_relay")
_CALL_LOGGER = _LOGGER.getChild("calls")
if os.environ.get("CHACK_GEMINI_RELAY_LOG_CALLS") == "1":
    # Workers can collect call telemetry even when their root logger suppresses INFO.
    call_handler = logging.StreamHandler()
    call_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    _CALL_LOGGER.setLevel(logging.INFO)
    _CALL_LOGGER.addHandler(call_handler)
    _CALL_LOGGER.propagate = False

_STREAM_METHOD = ":streamGenerateContent"
_UNARY_METHOD = ":generateContent"
_CHUNK_BYTES = 64 * 1024
# Errors of writing to a client that has gone away.
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

# Headers that describe one hop, or that the relay sets itself.
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
    "content-encoding",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect goes back to the CLI as it is, like any other answer.
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _hang_up(sock: Any) -> None:
    try:
        # The plain socket's shutdown wakes a thread blocked reading it, a TLS one
        # too, without touching the TLS state that thread is using.
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except (OSError, TypeError):
        pass


class _AttemptControl(threading.Event):
    """Ends one gateway request of a call. Set, the request starts no further
    attempt (its caller stopped waiting); ``cancel`` also hangs up its open
    connection at once, for a request whose answer is no longer needed.

    Each request has its own control, so concurrent calls share none of it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._sockets: list[Any] = []
        self.cancelled = False

    def connected(self, sock: Any) -> None:
        with self._lock:
            if not self.cancelled:
                # One socket per attempt, and a request has a bounded number of attempts.
                self._sockets.append(sock)
                return
        _hang_up(sock)

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            sockets, self._sockets = self._sockets, []
        self.set()
        for sock in sockets:
            _hang_up(sock)


class _CancellableConnections:
    """Registers each gateway connection with its request's ``_AttemptControl``
    (``Request.chack_control``) once connected, so ``cancel`` can hang it up."""

    def do_open(self, http_class: Any, req: Any, **kwargs: Any) -> Any:
        control = getattr(req, "chack_control", None)
        if control is None:
            return super().do_open(http_class, req, **kwargs)  # type: ignore[misc]

        def connection(*args: Any, **connection_kwargs: Any) -> Any:
            conn = http_class(*args, **connection_kwargs)
            connect = conn.connect

            def connect_and_register() -> None:
                connect()
                control.connected(conn.sock)

            conn.connect = connect_and_register
            return conn

        return super().do_open(connection, req, **kwargs)  # type: ignore[misc]


class _HTTPHandler(_CancellableConnections, urllib.request.HTTPHandler):
    pass


class _HTTPSHandler(_CancellableConnections, urllib.request.HTTPSHandler):
    pass


_MODELS_SEGMENT = "/models/"


def _model_in_path(request_path: str) -> tuple[int, int] | None:
    """Span of the model name in ``.../models/<model>:<method>``, if the path names one."""

    start = request_path.find(_MODELS_SEGMENT)
    if start < 0:
        return None
    start += len(_MODELS_SEGMENT)
    end = request_path.rfind(":")
    return (start, end) if end > start else None


def _upstream_target(base_url: str, path: str, model: str = "") -> tuple[str, bool, str]:
    """Gateway URL for a CLI request path, whether it was a streaming call, and
    the other model the CLI asked for when the call is sent to ``model`` instead."""

    parsed = urllib.parse.urlsplit(path)
    request_path = parsed.path
    query = parsed.query
    replaced = ""
    span = _model_in_path(request_path) if model else None
    if span is not None and request_path[span[0] : span[1]] != model:
        replaced = request_path[span[0] : span[1]]
        request_path = request_path[: span[0]] + model + request_path[span[1] :]
    streaming = request_path.endswith(_STREAM_METHOD)
    if streaming:
        request_path = request_path[: -len(_STREAM_METHOD)] + _UNARY_METHOD
        query = urllib.parse.urlencode(
            [
                (name, value)
                for name, value in urllib.parse.parse_qsl(query, keep_blank_values=True)
                if name != "alt"
            ]
        )
    url = base_url.rstrip("/") + request_path
    if query:
        url += "?" + query
    return url, streaming, replaced


def _usage_of(answer: Any) -> dict[str, int] | None:
    """Token usage of one Gemini API answer, counted the way the Gemini CLI counts it.

    ``cached_tokens`` is None when the answer does not report it.
    """

    metadata = answer.get("usageMetadata") if isinstance(answer, dict) else None
    if not isinstance(metadata, dict):
        return None
    cached = metadata.get("cachedContentTokenCount")
    return {
        "input_tokens": int(metadata.get("promptTokenCount", 0) or 0),
        "output_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
        "cached_tokens": None if cached is None else int(cached or 0),
    }


def _empty_answer_note(answer: Any, request_body: bytes | None) -> dict[str, Any] | None:
    """Why a model answer carried neither text nor a tool call, without its content.

    Holds the finish reason, the answer's input/output/thinking tokens and the
    request's output limit, so an output cutoff can be told from an answer the
    model chose to leave empty. None when the answer has text or a tool call.
    """

    if not isinstance(answer, dict):
        return None
    candidates = answer.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    reasons = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            if part.get("functionCall") or str(part.get("text") or "").strip():
                return None
        reasons.append(str(candidate.get("finishReason") or "none"))
    metadata = answer.get("usageMetadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    limit = None
    try:
        generation = json.loads(request_body or b"{}").get("generationConfig")
        limit = generation.get("maxOutputTokens") if isinstance(generation, dict) else None
    except (ValueError, AttributeError):
        pass
    feedback = answer.get("promptFeedback")
    return {
        "finish_reason": ",".join(reasons) or "no candidates",
        "block_reason": feedback.get("blockReason") if isinstance(feedback, dict) else None,
        "input_tokens": int(metadata.get("promptTokenCount", 0) or 0),
        "output_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
        "thinking_tokens": int(metadata.get("thoughtsTokenCount", 0) or 0),
        "max_output_tokens": limit,
    }


def _answer_shape(answer: Any) -> dict[str, Any]:
    """What one model answer held, without its content: finish reasons, how many
    parts of each kind (tool call, text, thought, empty) and its token counts."""

    shape: dict[str, Any] = {"finish_reason": [], "function_call": 0, "text": 0, "thought": 0, "empty": 0}
    candidates = answer.get("candidates") if isinstance(answer, dict) else None
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        shape["finish_reason"].append(str(candidate.get("finishReason") or "none"))
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("functionCall"):
                shape["function_call"] += 1
            elif part.get("thought"):
                shape["thought"] += 1
            elif str(part.get("text") or "").strip():
                shape["text"] += 1
            else:
                shape["empty"] += 1
    shape["finish_reason"] = ",".join(shape["finish_reason"]) or "no candidates"
    metadata = answer.get("usageMetadata") if isinstance(answer, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    shape["input_tokens"] = int(metadata.get("promptTokenCount", 0) or 0)
    shape["output_tokens"] = int(metadata.get("candidatesTokenCount", 0) or 0)
    shape["thinking_tokens"] = int(metadata.get("thoughtsTokenCount", 0) or 0)
    return shape


# Provider prefix caches store whole blocks of a recent request's tokens and
# skip prompts too short to cache: measured on a LiteLLM gateway, the cached
# count of each call was the previous request's prompt rounded down to a
# 128-token block. Earlier requests count for a while, not forever.
_CACHE_BLOCK_TOKENS = 128
_CACHE_MARGIN_TOKENS = 64
_CACHE_MIN_TOKENS = 1024
_CACHE_HISTORY = 16
_CACHE_TTL_SECONDS = 600.0
# Prompts kept per session, and sessions kept per process: a session's next
# turn repeats its own last prompts, however many other sessions call between.
_CACHE_SESSION_HISTORY = 4
_CACHE_SESSIONS = 128


def _prompt_text(request_body: bytes | None) -> str:
    """The request in the order a model reads it: instructions, tools, then the conversation."""

    if not request_body:
        return ""
    try:
        payload = json.loads(request_body)
    except ValueError:
        return request_body.decode("utf-8", "replace")
    if not isinstance(payload, dict):
        return json.dumps(payload, separators=(",", ":"))
    ordered = {
        key: payload[key]
        for key in ("systemInstruction", "tools", "toolConfig", "cachedContent", "contents")
        if key in payload
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    low, high = 0, limit
    # Binary search over slices compares in C instead of one character at a time.
    while low < high:
        middle = (low + high + 1) // 2
        if left[:middle] == right[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


class PrefixCacheEstimator:
    """Estimates the prompt tokens a provider served from its prefix cache.

    Keeps the last ``_CACHE_HISTORY`` prompts per gateway and model, and the
    last ``_CACHE_SESSION_HISTORY`` of each of the ``_CACHE_SESSIONS`` most
    recent sessions, for ``_CACHE_TTL_SECONDS``. A prompt's estimate is its
    share of characters that repeat the longest-matching earlier prompt,
    applied to its token count, never above that earlier prompt's own tokens,
    less a small margin and rounded down to whole cache blocks; under
    ``_CACHE_MIN_TOKENS`` it is zero.
    """

    def __init__(self, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._history: dict[tuple[str, str], list[tuple[float, str, int]]] = {}
        self._sessions: OrderedDict[tuple[str, str, Any], list[tuple[float, str, int]]] = OrderedDict()

    def estimate(self, key: tuple[str, str], text: str, prompt_tokens: int, session: Any = None) -> int:
        now = self._clock()
        with self._lock:
            recent = [
                entry for entry in self._history.get(key, []) if now - entry[0] <= _CACHE_TTL_SECONDS
            ]
            while self._sessions and now - next(iter(self._sessions.values()))[-1][0] > _CACHE_TTL_SECONDS:
                self._sessions.popitem(last=False)
            own_key = (*key, session)
            own = [
                entry for entry in self._sessions.pop(own_key, []) if now - entry[0] <= _CACHE_TTL_SECONDS
            ] if session is not None else []
            cached = 0
            if text and prompt_tokens > 0:
                for _, earlier, earlier_tokens in recent + own:
                    shared = _common_prefix_length(text, earlier)
                    tokens = min(prompt_tokens * shared // len(text), earlier_tokens)
                    tokens = (tokens - _CACHE_MARGIN_TOKENS) // _CACHE_BLOCK_TOKENS * _CACHE_BLOCK_TOKENS
                    cached = max(cached, tokens)
                recent.append((now, text, prompt_tokens))
                own.append((now, text, prompt_tokens))
            self._history[key] = recent[-_CACHE_HISTORY:]
            if own:
                self._sessions[own_key] = own[-_CACHE_SESSION_HISTORY:]
                while len(self._sessions) > _CACHE_SESSIONS:
                    self._sessions.popitem(last=False)
        return cached if cached >= _CACHE_MIN_TOKENS else 0


# Shared by every relay of a process: a run's relays name their session, so
# each turn sees its own earlier prompts however many other runs call between.
_PREFIX_CACHE = PrefixCacheEstimator()


# DeepSeek's tool-call markup: special tokens with fullwidth bars (U+FF5C).
_DSML_CALLS_OPEN = re.compile(r"<｜DSML｜\s*(?:function_)?calls>")
_DSML_CALLS_REST = re.compile(r"(?P<body>.*?)</｜DSML｜\s*(?:function_)?calls>\s*", re.S)

_DSML_INVOKE = re.compile(
    r'<｜DSML｜\s*invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</｜DSML｜\s*invoke>', re.S
)
_DSML_PARAMETER = re.compile(
    r'<｜DSML｜\s*parameter\s+name="(?P<name>[^"]+)"(?:\s+string="(?P<string>true|false)")?\s*>'
    r"(?P<value>.*?)</｜DSML｜\s*parameter>",
    re.S,
)


def _declared_tools(request_body: bytes | None) -> dict[str, dict[str, Any]]:
    """The function declarations a model request offers, by name."""
    try:
        request = json.loads(request_body or b"{}")
    except ValueError:
        return {}
    tools = request.get("tools") if isinstance(request, dict) else None
    declared: dict[str, dict[str, Any]] = {}
    for tool in tools if isinstance(tools, list) else []:
        functions = tool.get("functionDeclarations") if isinstance(tool, dict) else None
        for function in functions if isinstance(functions, list) else []:
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                schema = function.get("parametersJsonSchema")
                if schema is None and isinstance(function.get("parameters"), dict):
                    schema = _lower_types(function["parameters"])  # Gemini's OpenAPI subset: "OBJECT".
                declared[function["name"]] = schema if isinstance(schema, dict) else {"type": "object"}
    return declared


def _lower_types(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {key: value.lower() if key == "type" and isinstance(value, str) else _lower_types(value)
                for key, value in schema.items()}
    if isinstance(schema, list):
        return [_lower_types(item) for item in schema]
    return schema


def _declared_name(name: str, declared: dict[str, dict[str, Any]]) -> str | None:
    """The declared tool a call means: its exact name, else the one MCP tool
    whose name only adds the CLI's server prefix (``mcp_<server>_<name>``)."""
    if name in declared:
        return name
    matches = [candidate for candidate in declared
               if candidate.startswith("mcp_") and candidate.endswith(f"_{name}") and len(candidate) > len(name) + 5]
    return matches[0] if len(matches) == 1 else None


def _valid_arguments(args: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Whether the arguments satisfy the tool's whole declared schema (an unusable schema rejects them)."""
    import jsonschema

    try:
        validator = jsonschema.validators.validator_for(schema)
        validator.check_schema(schema)
        return not any(True for _ in validator(schema).iter_errors(args))
    except jsonschema.SchemaError:
        return False


def _markup_calls(markup: str, declared: dict[str, dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Function calls of one markup block, or None unless every call is valid."""
    calls = []
    if _DSML_INVOKE.sub("", markup).strip():
        return None
    for invoke in _DSML_INVOKE.finditer(markup):
        name = _declared_name(invoke["name"], declared)
        if name is None or _DSML_PARAMETER.sub("", invoke["body"]).strip():
            return None
        args: dict[str, Any] = {}
        for parameter in _DSML_PARAMETER.finditer(invoke["body"]):
            key, value = parameter["name"], parameter["value"]
            if key in args:
                return None
            if parameter["string"] == "false":
                try:
                    value = json.loads(value)
                except ValueError:
                    return None
            args[key] = value
        properties = declared[name].get("properties")
        # A parameter the tool never declared is debris, even where its schema would allow extra keys.
        if isinstance(properties, dict) and any(key not in properties for key in args):
            return None
        if not _valid_arguments(args, declared[name]):
            return None
        calls.append({"functionCall": {"name": name, "args": args}})
    return calls or None


def _with_text_tool_calls(body: bytes, request_body: bytes | None) -> bytes:
    """The answer with tool-call markup in its text turned into function calls.

    Only a block that ends a text part, outside a code fence, and whose calls
    all name declared tools with arguments valid against their declared schema is converted; the text before
    it stays a text part. Any other answer is returned unchanged.
    """
    if b"DSML" not in body:  # The bars may arrive JSON-escaped (\uff5c).
        return body
    declared = _declared_tools(request_body)
    if not declared:
        return body
    try:
        answer = json.loads(body)
    except ValueError:
        return body
    changed = False
    candidates = answer.get("candidates") if isinstance(answer, dict) else None
    for candidate in candidates if isinstance(candidates, list) else []:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        rewritten = []
        for part in parts:
            text = part.get("text") if isinstance(part, dict) and not part.get("thought") else None
            opening = list(_DSML_CALLS_OPEN.finditer(text)) if isinstance(text, str) else []
            block = _DSML_CALLS_REST.fullmatch(text, opening[-1].end()) if opening else None
            calls = None
            if block and text[: opening[-1].start()].count("```") % 2 == 0:
                calls = _markup_calls(block["body"], declared)
            if calls is None:
                if opening:
                    _LOGGER.warning("Gemini gateway answer kept tool-call markup it could not read as declared tool calls")
                rewritten.append(part)
                continue
            prefix = text[: opening[-1].start()].rstrip()
            if prefix:
                rewritten.append({**part, "text": prefix})
            rewritten.extend(calls)
            changed = True
        content["parts"] = rewritten
    if not changed:
        return body
    return json.dumps(answer, ensure_ascii=False).encode("utf-8")


def _sse_event(body: bytes) -> bytes:
    # The Gemini SDK reads each event as one ``data:`` line, so the JSON
    # must not span lines.
    payload = json.dumps(json.loads(body), ensure_ascii=False, separators=(",", ":"))
    return b"data: " + payload.encode("utf-8") + b"\r\n\r\n"


_RETRYABLE_GENERATION_FIELDS = frozenset({"temperature", "topP", "topK"})


def _without_rejected_generation_field(request_body: bytes | None, error_body: bytes) -> bytes | None:
    """Only remove a named generation field after its explicit rejection."""
    match = re.search(rb"doesn't support the ([A-Za-z][A-Za-z0-9]*) field", error_body, re.I)
    if not request_body or match is None:
        return None
    field = match.group(1).decode("ascii")
    if field not in _RETRYABLE_GENERATION_FIELDS:
        return None
    try:
        data = json.loads(request_body)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    generation = data.get("generationConfig")
    if not isinstance(generation, dict) or field not in generation:
        return None
    data["generationConfig"] = {key: value for key, value in generation.items() if key != field}
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


REQUEST_TIMEOUT_ENV = "CHACK_GEMINI_RELAY_REQUEST_TIMEOUT_SECONDS"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
# A gateway that hangs on a call hangs on most requests for the whole limit, while
# a slow answer may still come after this share of it. A streaming call with no
# answer by then gets one duplicate request alongside the first (a hedge), which
# keeps waiting; the hedge gets the whole limit, and both end at the hedge point
# plus that limit, so a hang costs this share and a slow answer is not discarded.
_HEDGE_AFTER_SHARE = 0.5
# Socket timeout of an attempt started just before its call's deadline, so it never gets a zero or negative one.
_MIN_ATTEMPT_TIMEOUT_SECONDS = 0.5


def gateway_call_timeout_seconds(run_timeout_seconds: float | None = None) -> float:
    """Time limit of one gateway call: the configured limit, never above the run's own."""
    try:
        limit = float(os.environ.get(REQUEST_TIMEOUT_ENV) or DEFAULT_REQUEST_TIMEOUT_SECONDS)
    except ValueError:
        limit = DEFAULT_REQUEST_TIMEOUT_SECONDS
    if limit <= 0:
        limit = DEFAULT_REQUEST_TIMEOUT_SECONDS
    if run_timeout_seconds and run_timeout_seconds > 0:
        limit = min(limit, float(run_timeout_seconds))
    return limit


class _Reply(NamedTuple):
    """A whole answer for the CLI."""

    status: int
    headers: dict[str, str]
    body: bytes
    # The relay's own 502 for a gateway connection dropped or refused, which may be retried.
    dropped: bool = False
    # The relay's own 502 for a gateway call it stopped waiting for at its socket timeout.
    timed_out: bool = False


def _error_reply(code: int, status: str, message: str) -> _Reply:
    error = {"error": {"code": code, "status": status, "message": message}}
    return _Reply(code, {"Content-Type": "application/json"}, json.dumps(error).encode("utf-8"))


def _error_chunk(reply: _Reply) -> bytes:
    """A gateway error as the Gemini API's error JSON, carrying its HTTP status as the code."""
    try:
        data = json.loads(reply.body)
    except ValueError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        error = {"message": reply.body.decode("utf-8", "replace")[:2000]}
    code = error.get("code")
    if not isinstance(code, int) or not 400 <= code < 600:
        error = {**error, "code": reply.status}
    return json.dumps({"error": error}, ensure_ascii=False).encode("utf-8")


def _dropped_connection(exc: BaseException) -> bool:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    # A connection closed before or while its answer was read; a timeout is not one.
    return isinstance(reason, (ConnectionError, http.client.RemoteDisconnected, http.client.IncompleteRead))


def _timed_out(exc: BaseException) -> bool:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(reason, TimeoutError)


def _timeout_reply(exc: BaseException) -> _Reply:
    return _error_reply(502, "UNAVAILABLE", f"gateway call timed out: {exc}")._replace(timed_out=True)


def _notify_pressure(response: Any, status: int, transient: bool, url: str, *, timeout_reported: bool) -> None:
    """Tell ``gateway_pressure`` listeners about a timed-out, rate-limited or unavailable call."""
    if isinstance(response, _Reply) and response.timed_out:
        if not timeout_reported:
            gateway_pressure.notify(gateway_pressure.TIMEOUT, f"call to {url} timed out")
    elif status == 429:
        gateway_pressure.notify(gateway_pressure.RATE_LIMITED, f"call to {url}: HTTP 429")
    elif transient or status >= 500:
        gateway_pressure.notify(gateway_pressure.UNAVAILABLE, f"call to {url}: {_status_label(response)}")


def _status_label(response: Any) -> Any:
    """The status a log line shows: the relay's own failures by name, not as their 502."""
    if isinstance(response, _Reply):
        if response.timed_out:
            return "timeout"
        if response.dropped:
            return "connection dropped"
    return _status_of(response)


def _status_of(response: Any) -> int:
    if isinstance(response, _Reply):
        return response.status
    return int(response.status if hasattr(response, "status") else response.code)


def _end_to_end(headers: Any) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in _HOP_HEADERS}


def _write_chunk(handler: BaseHTTPRequestHandler, chunk: bytes) -> None:
    handler.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
    handler.wfile.flush()


class GeminiGatewayRelay:
    """Serves the Gemini CLI on 127.0.0.1 for one gateway Gemini API base URL."""

    def __init__(
        self,
        upstream_base_url: str,
        *,
        model: str = "",
        timeout_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
        keepalive_after_seconds: float = 20.0,
        keepalive_every_seconds: float = 15.0,
        progress_every_seconds: float = 300.0,
        cache_session: Any = None,
    ) -> None:
        # Names the run whose turns this relay carries, for its cache estimates.
        self._cache_session = cache_session
        self._upstream_base_url = str(upstream_base_url or "").rstrip("/")
        self._model = str(model or "").strip()
        self._replaced_models: set[str] = set()
        # ``timeout_seconds`` is the run's limit; each gateway call gets its own, lower one.
        self._timeout_seconds = (
            min(request_timeout_seconds, timeout_seconds or request_timeout_seconds)
            if request_timeout_seconds
            else gateway_call_timeout_seconds(timeout_seconds)
        )
        self._keepalive_after_seconds = keepalive_after_seconds
        self._keepalive_every_seconds = keepalive_every_seconds
        self._opener = urllib.request.build_opener(_NoRedirect(), _HTTPHandler(), _HTTPSHandler())
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._usage_lock = threading.Lock()
        self._usage: list[dict[str, int]] = []
        self._estimating = False
        # The last answer to a request offering tools (the agent's own turn), and
        # how many answers to the CLI's helper calls (no tools) came after it.
        self._last_turn: dict[str, Any] | None = None
        self._helper_answers = 0
        # Set once a call spent its retries on a transient failure, until the gateway answers again.
        self._retries_spent = False
        # Answered calls are otherwise silent: one progress line at most every
        # ``progress_every_seconds`` shows a quiet run is still being answered.
        self._progress_every_seconds = progress_every_seconds
        self._progress_at = time.monotonic()
        self._answered = 0
        self._answered_since_progress = 0

    def take_usage(self) -> list[dict[str, int]]:
        """Usage of the answers relayed since the last call, one entry per answer."""
        with self._usage_lock:
            usage, self._usage = self._usage, []
        return usage

    def last_answer_shape(self) -> dict[str, Any] | None:
        """The shape (``_answer_shape``) of the last answer to an agent turn, with
        the number of helper-call answers relayed after it; None before any."""

        with self._usage_lock:
            if self._last_turn is None:
                return None
            return {**self._last_turn, "helper_answers_after": self._helper_answers}

    def _record_usage(self, body: bytes, request_body: bytes | None = None) -> dict[str, int] | None:
        """Count usage and return the provider's measurements before cache estimation."""
        try:
            answer = json.loads(body)
        except ValueError:
            return None
        shape = _answer_shape(answer)
        now = time.monotonic()
        with self._usage_lock:
            if request_body and b'"functionDeclarations"' in request_body:
                self._last_turn, self._helper_answers = shape, 0
            else:
                self._helper_answers += 1
            self._answered += 1
            self._answered_since_progress += 1
            elapsed = now - self._progress_at
            progress = None
            if elapsed >= self._progress_every_seconds:
                progress = (self._answered_since_progress, elapsed, self._answered)
                self._progress_at, self._answered_since_progress = now, 0
        if progress is not None:
            _LOGGER.info(
                "Gemini gateway relay answered %d model calls in the last %.0fs (%d in total); "
                "last request_bytes=%d prompt_tokens=%s",
                progress[0],
                progress[1],
                progress[2],
                len(request_body or b""),
                shape.get("input_tokens"),
            )
        empty = _empty_answer_note(answer, request_body)
        if empty is not None:
            _LOGGER.warning(
                "Gemini gateway answer for %s had no text or tool call: finish_reason=%s block_reason=%s "
                "input_tokens=%s output_tokens=%s thinking_tokens=%s max_output_tokens=%s",
                self._model or "the requested model",
                empty["finish_reason"],
                empty["block_reason"],
                empty["input_tokens"],
                empty["output_tokens"],
                empty["thinking_tokens"],
                empty["max_output_tokens"],
            )
        usage = _usage_of(answer)
        if usage is None:
            return None
        reported_usage = dict(usage)
        key = (self._upstream_base_url, self._model)
        estimated = _PREFIX_CACHE.estimate(
            key, _prompt_text(request_body), usage["input_tokens"], self._cache_session
        )
        if usage["cached_tokens"] is None:
            usage["cached_tokens"] = min(estimated, usage["input_tokens"])
            if not self._estimating:
                self._estimating = True
                _LOGGER.info(
                    "Gemini gateway %s reports no cached tokens for %s; estimating them from repeated prompt prefixes",
                    self._upstream_base_url,
                    self._model or "the requested model",
                )
        with self._usage_lock:
            self._usage.append(usage)
        return reported_usage

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Gemini gateway relay is not running")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> "GeminiGatewayRelay":
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                _LOGGER.debug("relay %s", format % args)

            def _relay(self) -> None:
                try:
                    relay._forward(self)
                except _CLIENT_GONE:
                    # The CLI was stopped (cancelled, timed out) while it waited.
                    self.close_connection = True
                    _LOGGER.debug("Gemini CLI went away before the relay answered %s", self.path)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _relay

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="chack-gemini-gateway-relay",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "GeminiGatewayRelay":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _attempt_timeout(self, deadline: float | None) -> float:
        """Socket timeout of one attempt: the call's limit, or the time left before its ``deadline``."""
        if deadline is None:
            return self._timeout_seconds
        return max(_MIN_ATTEMPT_TIMEOUT_SECONDS, deadline - time.monotonic())

    def _open(
        self,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        method: str,
        deadline: float | None = None,
        control: _AttemptControl | None = None,
    ) -> Any:
        """The gateway's response, or a 502 ``_Reply`` when it cannot be reached.

        A timeout is not retried here (see ``_forward_stream``) and is marked
        ``timed_out``; a dropped or refused connection is marked so
        ``_gateway_response`` may retry it.
        """
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        request.chack_control = control  # type: ignore[attr-defined]
        try:
            return self._opener.open(request, timeout=self._attempt_timeout(deadline))
        except urllib.error.HTTPError as exc:
            return exc
        except Exception as exc:
            cancelled = control is not None and control.cancelled
            if _timed_out(exc) and not cancelled:
                _LOGGER.warning("Gemini gateway relay timed out waiting for %s: %s", url, exc)
                return _timeout_reply(exc)
            (_LOGGER.debug if cancelled else _LOGGER.warning)("Gemini gateway relay could not reach %s: %s", url, exc)
            reply = _error_reply(502, "UNAVAILABLE", f"gateway unreachable: {exc}")
            return reply._replace(dropped=_dropped_connection(exc))

    def _retry(
        self,
        url: str,
        body: bytes | None,
        deadline: float | None = None,
        abandoned: threading.Event | None = None,
    ) -> TransientRetry:
        # The Gemini CLI re-sends a failed call a few times itself. Once a call
        # spent the relay's retries, those re-sends get one attempt each until
        # the gateway answers again, so the attempts do not multiply.
        # The request's size goes in the log, to tell whether only the largest requests fail.
        with self._usage_lock:
            last_input = (self._last_turn or {}).get("input_tokens")
        size = f"{len(body or b'')} request bytes" + (
            f", last turn {last_input} input tokens" if last_input else ""
        )
        return TransientRetry(
            f"Gemini gateway call to {url} ({size})",
            logger=_LOGGER,
            deadline=deadline,
            abandoned=abandoned,
            attempts=1 if self._retries_spent else None,
        )

    def _log_call(
        self,
        body: bytes | None,
        status: int | None,
        retry: TransientRetry,
        *,
        outcome: str | None = None,
        usage: dict[str, int] | None = None,
        finished_at: float | None = None,
        control: _AttemptControl | None = None,
    ) -> None:
        """Report the attempt without the URL, headers, request or answer body."""
        if outcome is None:
            outcome = "answered" if status is not None and 200 <= status < 300 else "http_error"
        if outcome != "answered" and control is not None and control.cancelled:
            # Hung up by the relay because the call's other request answered.
            outcome = "cancelled"
        elapsed = (time.monotonic() if finished_at is None else finished_at) - retry.attempt_started
        usage = usage or {}
        _CALL_LOGGER.info(
            "Gemini gateway call outcome=%s status=%s attempt=%d/%d elapsed_seconds=%.3f "
            "request_bytes=%d prompt_tokens=%s completion_tokens=%s cached_tokens=%s",
            outcome,
            status,
            retry.attempt,
            retry.attempts,
            elapsed,
            len(body or b""),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("cached_tokens"),
        )

    def _gateway_response(
        self,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        method: str,
        retry: TransientRetry,
        control: _AttemptControl | None = None,
    ) -> tuple[Any, bytes | None]:
        """The gateway's response once rejected generation fields are dropped and
        transient failures retried, with the body actually sent."""
        while True:
            retry.attempt_started = time.monotonic()
            response = self._open(url, body, headers, method, retry.deadline, control)
            for _ in range(len(_RETRYABLE_GENERATION_FIELDS)):
                if not isinstance(response, urllib.error.HTTPError) or response.code != 400:
                    break
                try:
                    error_body = response.read()
                finally:
                    self._log_call(body, 400, retry, control=control)
                retry_body = _without_rejected_generation_field(body, error_body)
                response_headers = _end_to_end(response.headers)
                response.close()
                if retry_body is None or retry.abandoned:
                    return _Reply(400, response_headers, error_body), body
                _LOGGER.info("Retrying Gemini gateway request without unsupported generation field")
                body = retry_body
                retry.attempt_started = time.monotonic()
                response = self._open(url, body, headers, method, retry.deadline, control)
            status = _status_of(response)
            transient = response.dropped if isinstance(response, _Reply) else status in TRANSIENT_STATUSES
            if not retry.abandoned:
                # A call with a deadline is waited for by ``_forward_stream``, which reports its timeout.
                _notify_pressure(response, status, transient, url, timeout_reported=retry.deadline is not None)
            if isinstance(response, _Reply):
                outcome = "timeout" if response.timed_out else "connection_error"
                self._log_call(body, None, retry, outcome=outcome, control=control)
            if control is not None and control.cancelled:
                # The call's other request answered; this one's failure needs no retry.
                return response, body
            if not transient:
                self._retries_spent = False
                return response, body
            if isinstance(response, _Reply):
                failure, retry_after, error_body, error_headers = "connection dropped", None, response.body, response.headers
            else:
                try:
                    error_body = response.read()
                finally:
                    self._log_call(body, status, retry, control=control)
                error_headers = _end_to_end(response.headers)
                response.close()
                failure = f"HTTP {status}: {error_body[:200].decode('utf-8', 'replace')}"
                retry_after = retry_after_seconds(response.headers.get("Retry-After"))
            if not retry.wait(failure, retry_after=retry_after):
                self._retries_spent = True
                return _Reply(status, error_headers, error_body), body

    def _read_answer(
        self,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        method: str,
        deadline: float | None = None,
        abandoned: _AttemptControl | None = None,
    ) -> "_Reply":
        """A model call's whole answer, its usage counted, or the gateway's error.

        A connection dropped while the answer was read is retried like a
        transient error status, with the same allowance; a timeout while it was
        read is the relay's own timeout reply.
        """
        retry = self._retry(url, body, deadline, abandoned)
        while True:
            response, sent_body = self._gateway_response(url, body, headers, method, retry, abandoned)
            if isinstance(response, _Reply):
                return response
            status = _status_of(response)
            try:
                with response:
                    response_headers = _end_to_end(response.headers)
                    raw = response.read()
            except Exception as exc:
                if _timed_out(exc):
                    reply = _timeout_reply(exc)
                    if deadline is None:
                        gateway_pressure.notify(gateway_pressure.TIMEOUT, f"answer from {url} timed out")
                    self._log_call(sent_body, status, retry, outcome="timeout", control=abandoned)
                    return reply
                if not _dropped_connection(exc):
                    self._log_call(sent_body, status, retry, outcome="read_error", control=abandoned)
                    raise
                self._log_call(sent_body, status, retry, outcome="answer_cut_off", control=abandoned)
                if abandoned is not None and abandoned.cancelled:
                    return _error_reply(502, "UNAVAILABLE", "gateway request cancelled")
                if retry.wait(f"answer cut off: {exc!r}"):
                    continue
                self._retries_spent = True
                return _error_reply(502, "UNAVAILABLE", f"gateway answer cut off: {exc}")
            finished_at = time.monotonic()
            usage = None
            try:
                if 200 <= status < 300:
                    raw = _with_text_tool_calls(raw, body)
                    usage = self._record_usage(raw, body)
            finally:
                self._log_call(sent_body, status, retry, usage=usage, finished_at=finished_at, control=abandoned)
            return _Reply(status, response_headers, raw)

    def _stream_reply(
        self,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        method: str,
        deadline: float | None = None,
        abandoned: _AttemptControl | None = None,
    ) -> "_Reply":
        """A streaming call's whole answer: one server-sent event, or the gateway's error."""
        reply = self._read_answer(url, body, headers, method, deadline, abandoned)
        if not 200 <= reply.status < 300:
            return reply
        try:
            event = _sse_event(reply.body)
        except ValueError:
            _LOGGER.warning("Gemini gateway relay got a non-JSON answer from %s", url)
            return _Reply(502, reply.headers, reply.body)
        response_headers = {
            name: value for name, value in reply.headers.items() if name.lower() != "content-type"
        }
        response_headers["Content-Type"] = "text/event-stream"
        return _Reply(reply.status, response_headers, event)

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        url, streaming, replaced = _upstream_target(self._upstream_base_url, handler.path, self._model)
        if replaced and replaced not in self._replaced_models:
            self._replaced_models.add(replaced)
            _LOGGER.info("Gemini CLI asked for %s; sending its calls to the run's model %s", replaced, self._model)
        length = int(handler.headers.get("content-length") or 0)
        body = handler.rfile.read(length) if length else None
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() not in _HOP_HEADERS
        }
        if streaming:
            headers = {name: value for name, value in headers.items() if name.lower() != "accept"}
            headers["Accept"] = "application/json"
            self._forward_stream(handler, url, body, headers)
            return
        if urllib.parse.urlsplit(url).path.endswith(_UNARY_METHOD):
            # A model answer: read it whole to count its tokens before the CLI gets it.
            reply = self._read_answer(url, body, headers, handler.command)
            self._send(handler, reply.status, reply.headers, reply.body)
            return
        retry = self._retry(url, body)
        response, sent_body = self._gateway_response(url, body, headers, handler.command, retry)
        if isinstance(response, _Reply):
            self._send(handler, response.status, response.headers, response.body)
            return
        status = _status_of(response)
        try:
            with response:
                self._send_streamed(handler, status, _end_to_end(response.headers), response)
        except Exception as exc:
            self._log_call(sent_body, status, retry, outcome="timeout" if _timed_out(exc) else "read_error")
            raise
        else:
            self._log_call(sent_body, status, retry)

    def _forward_stream(
        self, handler: BaseHTTPRequestHandler, url: str, body: bytes | None, headers: dict[str, str]
    ) -> None:
        """Answers a streaming call, keeping the CLI's connection alive while the gateway works.

        The Gemini CLI gives up on a response whose headers take over a minute,
        or whose body stalls for five, and a long model answer through the
        unstreamed call can take longer. A quick answer is sent as it is; a slow
        one gets its event-stream headers early and blank lines until the event,
        which the CLI's stream parser skips. A gateway error after the headers
        went out is sent as the Gemini API's error JSON, which the CLI raises
        with the error's status code; so is a transient failure that outlasted
        the relay's retries, so the CLI does not retry it once more.
        """
        reply, headers_sent = self._hedged_call(handler, url, body, headers)
        if reply is None:
            reply = _error_reply(
                504,
                "DEADLINE_EXCEEDED",
                f"gateway gave no answer within {self._timeout_seconds * (1 + _HEDGE_AFTER_SHARE):g}s, "
                f"a duplicate request sent after {self._timeout_seconds * _HEDGE_AFTER_SHARE:g}s included",
            )
        if not headers_sent and not (reply.status in TRANSIENT_STATUSES and self._retries_spent):
            self._send(handler, reply.status, reply.headers, reply.body)
            return
        if not headers_sent:
            # The relay already spent the retry allowance on this transient
            # failure. As an error inside the stream the CLI ends the turn on
            # it instead of retrying the call again, which would multiply the
            # attempts and the wait.
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
        _write_chunk(handler, reply.body if 200 <= reply.status < 300 else _error_chunk(reply))
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()

    def _hedged_call(
        self,
        handler: BaseHTTPRequestHandler,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> tuple["_Reply | None", bool]:
        """A streaming call's answer, or None when none of its requests answered by its deadline.

        Sends the call once and, when it has no answer after ``_HEDGE_AFTER_SHARE``
        of the limit, once more alongside it, never more than these two. The
        first success is the answer and the other request is hung up; an error
        is the answer only when no other request is still running. Both end at
        the deadline (the hedge point plus the whole limit): their attempts'
        socket timeouts end there and they start no attempt after it.

        Keeps the CLI's connection alive meanwhile, sending the event-stream
        headers once the call has taken ``keepalive_after_seconds``. Returns
        whether those headers have gone out. Everything a call tracks is its
        own, so concurrent calls share no state here.
        """
        started = time.monotonic()
        hedge_after = self._timeout_seconds * _HEDGE_AFTER_SHARE
        hedge_at = started + hedge_after
        deadline = hedge_at + self._timeout_seconds
        finished: "queue.SimpleQueue[tuple[str, _Reply]]" = queue.SimpleQueue()
        controls: dict[str, _AttemptControl] = {}

        def launch(name: str) -> None:
            control = controls[name] = _AttemptControl()

            def call() -> None:
                try:
                    reply = self._stream_reply(url, body, headers, handler.command, deadline, control)
                except Exception as exc:  # the CLI must get an answer
                    if not control.cancelled:
                        _LOGGER.warning("Gemini gateway relay failed on %s: %s", url, exc)
                    reply = _error_reply(502, "UNAVAILABLE", f"gateway relay failed: {exc}")
                finished.put((name, reply))

            # Nothing joins it: it ends by the deadline (socket timeouts capped, no
            # attempt after it), or at once when hung up.
            threading.Thread(target=call, name=f"chack-gemini-gateway-{name}", daemon=True).start()

        headers_sent = False
        keepalive_at = started + self._keepalive_after_seconds
        running = 0
        winner: str | None = None
        failure: _Reply | None = None
        reply: _Reply | None = None
        outcome = "abandoned"
        launch("primary")
        running += 1
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    gateway_pressure.notify(
                        gateway_pressure.TIMEOUT, f"no answer from {url} within {round(deadline - started, 1)}s"
                    )
                    _LOGGER.warning(
                        "Gemini gateway gave no answer to %s within %ss, with a duplicate request sent after %ss",
                        url,
                        round(deadline - started, 1),
                        round(hedge_after, 1),
                    )
                    reply = failure
                    outcome = "both_failed" if failure is not None else "both_timeout"
                    break
                if "hedge" not in controls and now >= hedge_at:
                    gateway_pressure.notify(
                        gateway_pressure.TIMEOUT, f"no answer from {url} within {round(hedge_after, 1)}s"
                    )
                    _LOGGER.warning(
                        "Gemini gateway gave no answer to %s within %ss; sending it once more alongside the first",
                        url,
                        round(hedge_after, 1),
                    )
                    launch("hedge")
                    running += 1
                if now >= keepalive_at:
                    if not headers_sent:
                        handler.send_response(200)
                        handler.send_header("Content-Type", "text/event-stream")
                        handler.send_header("Transfer-Encoding", "chunked")
                        handler.end_headers()
                        headers_sent = True
                    else:
                        _write_chunk(handler, b"\n")
                    keepalive_at = time.monotonic() + self._keepalive_every_seconds
                wake = min(deadline, keepalive_at, deadline if "hedge" in controls else hedge_at)
                try:
                    name, answer = finished.get(timeout=max(0.0, wake - time.monotonic()))
                except queue.Empty:
                    continue
                running -= 1
                if 200 <= answer.status < 300:
                    winner, reply = name, answer
                    outcome = f"{name}_won"
                    break
                if not answer.timed_out:
                    # The relay's own timeout comes only at the deadline: it is no answer.
                    failure = answer
                    if "hedge" not in controls:
                        # An error before the hedge point is the call's answer, as without a hedge.
                        reply = failure
                        break
                if running == 0:
                    if "hedge" not in controls:
                        hedge_at = time.monotonic()
                        continue
                    reply = failure
                    outcome = "both_failed" if failure is not None else "both_timeout"
                    break
        finally:
            for name, control in controls.items():
                if winner is not None and name != winner:
                    control.cancel()
                else:
                    control.set()
            if "hedge" in controls:
                _CALL_LOGGER.info(
                    "Gemini gateway hedge outcome=%s status=%s hedge_after_seconds=%.3f elapsed_seconds=%.3f "
                    "request_bytes=%d",
                    outcome,
                    None if reply is None else reply.status,
                    hedge_after,
                    time.monotonic() - started,
                    len(body or b""),
                )
        return reply, headers_sent

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, headers: dict[str, str], body: bytes) -> None:
        handler.send_response(status)
        for name, value in headers.items():
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _send_streamed(
        handler: BaseHTTPRequestHandler, status: int, headers: dict[str, str], response: Any
    ) -> None:
        handler.send_response(status)
        for name, value in headers.items():
            handler.send_header(name, value)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        while True:
            chunk = response.read1(_CHUNK_BYTES) if hasattr(response, "read1") else response.read(_CHUNK_BYTES)
            if not chunk:
                break
            _write_chunk(handler, chunk)
        handler.wfile.write(b"0\r\n\r\n")
