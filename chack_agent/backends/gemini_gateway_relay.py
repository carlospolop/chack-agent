"""Local relay between the Gemini CLI and an LLM gateway's Gemini API.

LiteLLM translates its Gemini API for every model it serves, but when it
streams (``:streamGenerateContent``) a model that is not Gemini its
``usageMetadata`` is all zeros, so a run's tokens and cost are lost. The same
request without streaming (``:generateContent``) reports the real usage.

The relay listens on 127.0.0.1 and forwards requests to the gateway. A
streaming request is sent as ``:generateContent`` and its JSON answer is
returned to the CLI as one server-sent event. If a gateway explicitly rejects
one of the Gemini CLI's temperature, topP, or topK fields for a model, a
bounded retry omits that field. A connection the gateway drops or refuses is
tried again a bounded number of times.
Other gateway errors pass through with their status and body.

The Gemini CLI also calls Google models of its own for helper work (chat
compression, loop detection, routing, summaries), which a gateway may not
serve or a profile never chose. The relay sends every call to the run's model.

An unstreamed answer can take longer than the Gemini CLI waits for response
headers, so a slow streaming call gets its event-stream headers early and
blank keepalive lines until the answer, or the gateway's error, arrives.

Each gateway call has its own time limit (``CHACK_GEMINI_RELAY_REQUEST_TIMEOUT_SECONDS``,
never more than the run's limit), well below a run's: a model call the gateway
never answers is abandoned at that limit and sent once more, and if that one
also gets no answer the CLI gets a 504 ``DEADLINE_EXCEEDED`` error instead of
keepalives until the whole run times out.

The relay also keeps the token usage of every answer it relays, so a run that
is stopped before the CLI prints its closing stats still counts the calls it
made. LiteLLM's Gemini API omits ``cachedContentTokenCount`` for models that are
not Gemini even when the provider served most of the prompt from its prefix
cache, so every token would be billed at the full input price. When an answer
omits that field, the relay estimates the cached tokens from the longest prefix
the request shares with a recent request to the same gateway and model
(``PrefixCacheEstimator``); an answer that reports the field is trusted as is. A CLI that goes away while the relay answers it (a cancelled or timed-out
run) is no error: the answer is dropped quietly.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, NamedTuple


_LOGGER = logging.getLogger("chack.gemini_gateway_relay")

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

    Keeps the last ``_CACHE_HISTORY`` prompts per gateway and model for
    ``_CACHE_TTL_SECONDS``. A prompt's estimate is its share of characters
    that repeat the longest-matching earlier prompt, applied to its token count,
    never above that earlier prompt's own tokens, less a small margin and
    rounded down to whole cache blocks; under ``_CACHE_MIN_TOKENS`` it is zero.
    """

    def __init__(self, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._history: dict[tuple[str, str], list[tuple[float, str, int]]] = {}

    def estimate(self, key: tuple[str, str], text: str, prompt_tokens: int) -> int:
        now = self._clock()
        with self._lock:
            recent = [
                entry for entry in self._history.get(key, []) if now - entry[0] <= _CACHE_TTL_SECONDS
            ]
            cached = 0
            if text and prompt_tokens > 0:
                for _, earlier, earlier_tokens in recent:
                    shared = _common_prefix_length(text, earlier)
                    tokens = min(prompt_tokens * shared // len(text), earlier_tokens)
                    tokens = (tokens - _CACHE_MARGIN_TOKENS) // _CACHE_BLOCK_TOKENS * _CACHE_BLOCK_TOKENS
                    cached = max(cached, tokens)
                recent.append((now, text, prompt_tokens))
            self._history[key] = recent[-_CACHE_HISTORY:]
        return cached if cached >= _CACHE_MIN_TOKENS else 0


# Shared by every relay of a process, so each turn of a run sees the prompts of its earlier turns.
_PREFIX_CACHE = PrefixCacheEstimator()


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


# A gateway connection that is dropped or refused is tried this many times.
_CONNECTION_ATTEMPTS = 3
_RECONNECT_DELAY_SECONDS = 2.0

REQUEST_TIMEOUT_ENV = "CHACK_GEMINI_RELAY_REQUEST_TIMEOUT_SECONDS"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
# A model call that gets no answer within its time limit is sent this many times in all.
_TIMED_OUT_CALL_ATTEMPTS = 2


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
    return isinstance(reason, ConnectionError)


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
    ) -> None:
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
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._usage_lock = threading.Lock()
        self._usage: list[dict[str, int]] = []
        self._estimating = False
        # The last answer to a request offering tools (the agent's own turn), and
        # how many answers to the CLI's helper calls (no tools) came after it.
        self._last_turn: dict[str, Any] | None = None
        self._helper_answers = 0

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

    def _record_usage(self, body: bytes, request_body: bytes | None = None) -> None:
        try:
            answer = json.loads(body)
        except ValueError:
            return
        shape = _answer_shape(answer)
        with self._usage_lock:
            if request_body and b'"functionDeclarations"' in request_body:
                self._last_turn, self._helper_answers = shape, 0
            else:
                self._helper_answers += 1
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
            return
        key = (self._upstream_base_url, self._model)
        estimated = _PREFIX_CACHE.estimate(key, _prompt_text(request_body), usage["input_tokens"])
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

    def _open(self, url: str, body: bytes | None, headers: dict[str, str], method: str) -> Any:
        """The gateway's response, or a 502 ``_Reply`` when it cannot be reached.

        A connection the gateway drops or refuses is tried again a bounded
        number of times; a timeout is not retried here (see ``_forward_stream``).
        """
        failure: Exception | None = None
        for attempt in range(_CONNECTION_ATTEMPTS):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                return self._opener.open(request, timeout=self._timeout_seconds)
            except urllib.error.HTTPError as exc:
                return exc
            except Exception as exc:
                failure = exc
                if not _dropped_connection(exc) or attempt + 1 == _CONNECTION_ATTEMPTS:
                    break
                _LOGGER.warning("Gemini gateway dropped the connection to %s (%s); retrying", url, exc)
                time.sleep(_RECONNECT_DELAY_SECONDS * (attempt + 1))
        _LOGGER.warning("Gemini gateway relay could not reach %s: %s", url, failure)
        return _error_reply(502, "UNAVAILABLE", f"gateway unreachable: {failure}")

    def _gateway_response(self, url: str, body: bytes | None, headers: dict[str, str], method: str) -> Any:
        """The gateway's response once rejected generation fields are dropped, or a final ``_Reply``."""
        response = self._open(url, body, headers, method)
        for _ in range(len(_RETRYABLE_GENERATION_FIELDS)):
            if not isinstance(response, urllib.error.HTTPError) or response.code != 400:
                break
            error_body = response.read()
            retry_body = _without_rejected_generation_field(body, error_body)
            response_headers = _end_to_end(response.headers)
            response.close()
            if retry_body is None:
                return _Reply(400, response_headers, error_body)
            _LOGGER.info("Retrying Gemini gateway request without unsupported generation field")
            body = retry_body
            response = self._open(url, body, headers, method)
        return response

    def _stream_reply(self, url: str, body: bytes | None, headers: dict[str, str], method: str) -> "_Reply":
        """A streaming call's whole answer: one server-sent event, or the gateway's error."""
        response = self._gateway_response(url, body, headers, method)
        if isinstance(response, _Reply):
            return response
        with response:
            status = int(response.status if hasattr(response, "status") else response.code)
            response_headers = _end_to_end(response.headers)
            raw = response.read()
        if not 200 <= status < 300:
            return _Reply(status, response_headers, raw)
        self._record_usage(raw, body)
        try:
            event = _sse_event(raw)
        except ValueError:
            _LOGGER.warning("Gemini gateway relay got a non-JSON answer from %s", url)
            return _Reply(502, response_headers, raw)
        response_headers = {
            name: value for name, value in response_headers.items() if name.lower() != "content-type"
        }
        response_headers["Content-Type"] = "text/event-stream"
        return _Reply(status, response_headers, event)

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
        response = self._gateway_response(url, body, headers, handler.command)
        if isinstance(response, _Reply):
            self._send(handler, response.status, response.headers, response.body)
            return
        with response:
            status = int(response.status if hasattr(response, "status") else response.code)
            response_headers = _end_to_end(response.headers)
            if urllib.parse.urlsplit(url).path.endswith(_UNARY_METHOD) and 200 <= status < 300:
                # A model answer: read it whole to count its tokens before the CLI gets it.
                raw = response.read()
                self._record_usage(raw, body)
                self._send(handler, status, response_headers, raw)
                return
            self._send_streamed(handler, status, response_headers, response)

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
        with the error's status code.
        """
        headers_sent = False
        reply: _Reply | None = None
        for attempt in range(1, _TIMED_OUT_CALL_ATTEMPTS + 1):
            reply, headers_sent = self._wait_for_call(handler, url, body, headers, headers_sent)
            if reply is not None:
                break
            _LOGGER.warning(
                "Gemini gateway gave no answer to %s within %ss (attempt %d/%d)%s",
                url,
                self._timeout_seconds,
                attempt,
                _TIMED_OUT_CALL_ATTEMPTS,
                "; retrying" if attempt < _TIMED_OUT_CALL_ATTEMPTS else "",
            )
        if reply is None:
            reply = _error_reply(
                504,
                "DEADLINE_EXCEEDED",
                f"gateway gave no answer within {self._timeout_seconds:g}s, "
                f"{_TIMED_OUT_CALL_ATTEMPTS} attempts",
            )
        if not headers_sent:
            self._send(handler, reply.status, reply.headers, reply.body)
            return
        _write_chunk(handler, reply.body if 200 <= reply.status < 300 else _error_chunk(reply))
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()

    def _wait_for_call(
        self,
        handler: BaseHTTPRequestHandler,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        headers_sent: bool,
    ) -> tuple["_Reply | None", bool]:
        """One gateway call's answer, or None when it has none within the call's time limit.

        Keeps the CLI's connection alive meanwhile, sending the event-stream
        headers once the call has taken ``keepalive_after_seconds``. Returns
        whether those headers have gone out.
        """
        outcome: list[_Reply] = []
        done = threading.Event()

        def call() -> None:
            try:
                outcome.append(self._stream_reply(url, body, headers, handler.command))
            except Exception as exc:  # the CLI must get an answer
                _LOGGER.warning("Gemini gateway relay failed on %s: %s", url, exc)
                outcome.append(_error_reply(502, "UNAVAILABLE", f"gateway relay failed: {exc}"))
            finally:
                done.set()

        # An abandoned call ends at its socket timeout; nothing waits for it.
        threading.Thread(target=call, name="chack-gemini-gateway-call", daemon=True).start()
        deadline = time.monotonic() + self._timeout_seconds
        wait = self._keepalive_every_seconds if headers_sent else self._keepalive_after_seconds
        while True:
            if done.wait(max(0.0, min(wait, deadline - time.monotonic()))):
                return outcome[0], headers_sent
            if time.monotonic() >= deadline:
                return None, headers_sent
            if not headers_sent:
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.send_header("Transfer-Encoding", "chunked")
                handler.end_headers()
                headers_sent = True
            else:
                _write_chunk(handler, b"\n")
            wait = self._keepalive_every_seconds

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
