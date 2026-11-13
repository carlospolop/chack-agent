import json
import logging
import re
from typing import Any, Callable, Optional

from agents.agent_output import AgentOutputSchemaBase, ensure_strict_json_schema
from agents.exceptions import ModelBehaviorError

_LOGGER = logging.getLogger("chack.output_schema")


def _strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks emitted by thinking models (e.g. Qwen3).

    Returns the remaining text stripped of leading/trailing whitespace.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_from_thinking_output(text: str) -> str:
    """Strip <think> blocks and return the remaining text.

    If nothing remains after stripping (i.e. the model placed its entire answer
    inside a <think> block), fall back to finding the last JSON object or array
    within the thinking block itself.
    """
    stripped = _strip_thinking_blocks(text)
    if stripped:
        return stripped

    # Fallback: extract JSON from inside the <think> block
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        inner = think_match.group(1).strip()
        # Return the last JSON object/array found (model tends to put it last)
        candidates = list(re.finditer(r"\{[\s\S]*\}|\[[\s\S]*\]", inner))
        if candidates:
            return candidates[-1].group(0).strip()

    return stripped  # empty string if nothing usable found


def _extract_json_from_text(text: str) -> str:
    """Best-effort extraction of a JSON value from arbitrary free-form text.

    Models often produce output like:
      - JSON wrapped in ```json ... ``` code fences
      - Narrative sentences before or after the fence / JSON object
      - A mix of thinking prose and JSON embedded anywhere in the reply

    Strategy (in order):
    1. Entire text is a single code fence  → return its content.
    2. First code fence found anywhere     → return its content.
    3. First '{' or '[' found              → return everything from there.
    """
    stripped = text.strip()
    if not stripped:
        return stripped

    # 1 & 2: markdown code fences (full wrap or anywhere in text)
    fence_re = re.compile(r'`{3,}[a-zA-Z0-9]*[ \t]*\n?([\s\S]*?)\n?`{3,}', re.DOTALL)
    m = fence_re.search(stripped)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            return candidate

    # 3: find first JSON object or array
    for ch in ('{', '['):
        idx = stripped.find(ch)
        if idx >= 0:
            return stripped[idx:]

    return stripped


def _filter_invalid_array_items(data: Any, schema: dict[str, Any]) -> Any:
    """Remove array items that are missing required properties.

    Open-weight models sometimes produce array elements that are missing one
    or more required fields.  Rather than failing the whole output, drop the
    offending items so the rest of the data can be used.
    """
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return data

    props = schema.get("properties", {})
    result = dict(data)
    for key, prop_schema in props.items():
        if key not in result:
            continue
        value = result[key]
        if prop_schema.get("type") != "array" or not isinstance(value, list):
            continue
        item_schema = prop_schema.get("items", {})
        if not isinstance(item_schema, dict):
            continue
        required_fields = set(item_schema.get("required", []))
        if not required_fields:
            continue
        filtered = []
        for item in value:
            if not isinstance(item, dict):
                filtered.append(item)
                continue
            if required_fields.issubset(item.keys()):
                filtered.append(item)
            # else: silently drop the item that's missing required fields
        result[key] = filtered
    return result


def _strip_extra_properties(data: Any, schema: dict[str, Any]) -> Any:
    """Recursively remove keys not declared in a JSON schema with additionalProperties=False.

    Open-weight models often append extra fields (e.g. 'environment_context')
    that are not in the declared schema.  Rather than failing validation, prune
    them so that strict jsonschema validation can succeed.

    Also recurses into array items so that properties like 'sensitive_actions'
    whose item schema has additionalProperties=false are cleaned up too.
    """
    if not isinstance(schema, dict):
        return data

    if isinstance(data, list):
        # Recurse into each array element using the items sub-schema.
        item_schema = schema.get("items", {})
        if isinstance(item_schema, dict):
            return [_strip_extra_properties(elem, item_schema) for elem in data]
        return data

    if not isinstance(data, dict):
        return data

    props = schema.get("properties", {})
    result = {}
    for k, v in data.items():
        prop_schema = props.get(k, {})
        if isinstance(prop_schema, dict) and prop_schema.get("type") == "array" and isinstance(v, list):
            # Always recurse into array-typed properties regardless of additionalProperties flag.
            result[k] = _strip_extra_properties(v, prop_schema)
        elif schema.get("additionalProperties") is not False or k in props:
            result[k] = _strip_extra_properties(v, prop_schema)
        # else: key is not in props and additionalProperties=false → drop it
    return result


def _coerce_types(data: Any, schema: dict[str, Any]) -> Any:
    """Best-effort coercion of values that don't match the declared JSON schema types.

    Open-weight models occasionally return a list where a string is expected
    (e.g. common_vulns as ['...', '...'] instead of a single string).  Rather
    than failing hard, coerce the value to the expected type when possible.

    Handles: list/dict/number/bool → string  |  string → int/float/bool/list
    """
    if not isinstance(schema, dict) or not isinstance(data, dict):
        return data

    props = schema.get("properties", {})
    result = dict(data)
    for key, prop_schema in props.items():
        if key not in result:
            continue
        value = result[key]
        expected_type = prop_schema.get("type")
        if expected_type is None:
            continue

        if expected_type == "string" and not isinstance(value, str):
            if isinstance(value, list):
                result[key] = "\n".join(str(v) for v in value)
            elif isinstance(value, dict):
                result[key] = json.dumps(value, ensure_ascii=False)
            else:
                result[key] = str(value)

        elif expected_type in ("integer", "number") and isinstance(value, str):
            try:
                result[key] = int(value) if expected_type == "integer" else float(value)
            except (ValueError, TypeError):
                pass

        elif expected_type == "array" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    result[key] = parsed
            except (ValueError, TypeError):
                result[key] = [value]

        elif expected_type == "object" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    result[key] = parsed
            except (ValueError, TypeError):
                pass

        # Recurse into nested objects
        if expected_type == "object" and isinstance(result[key], dict):
            result[key] = _coerce_types(result[key], prop_schema)

    return result


def _wrap_single_array_root(data: Any, schema: dict[str, Any]) -> Any:
    """Wrap a bare item or list in the root's only property when that is an array.

    Some models answer ``{"items": [x]}`` schemas with ``x`` or ``[x]``; there
    is only one place either can belong.
    """
    properties = schema.get("properties")
    if schema.get("type") != "object" or not isinstance(properties, dict) or len(properties) != 1:
        return data
    (key, prop_schema), = properties.items()
    if not isinstance(prop_schema, dict) or prop_schema.get("type") != "array":
        return data
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, dict) and data and key not in data:
        return {key: [data]}
    return data


def _normalize_root_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if "type" in schema:
        return schema

    # OpenAI Responses requires response_format JSON schemas to have an object root.
    # Older configs may use a top-level composition of object refs instead.
    if any(key in schema for key in ("anyOf", "oneOf", "allOf")):
        normalized = dict(schema)
        normalized["type"] = "object"
        return normalized

    return schema


class JsonSchemaOutput(AgentOutputSchemaBase):
    def __init__(
        self,
        schema: dict[str, Any],
        *,
        name: str = "output_schema",
        strict: bool = True,
    ) -> None:
        if not isinstance(schema, dict):
            raise ValueError("output schema must be a JSON object")
        schema = _normalize_root_object_schema(schema)
        self._schema = ensure_strict_json_schema(schema) if strict else schema
        self._name = name or "output_schema"
        self._strict = bool(strict)

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self._name

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return self._strict

    def _mismatch(self, exc: BaseException, data: Any) -> ModelBehaviorError:
        """The schema error, carrying every error of ``data`` so one repair turn can fix them all."""
        error = ModelBehaviorError(f"Output did not match schema: {exc}")
        try:
            import jsonschema  # type: ignore

            validator = jsonschema.validators.validator_for(self._schema)(self._schema)
            error.schema_errors = sorted(
                validator.iter_errors(data), key=lambda e: [str(part) for part in e.absolute_path]
            )[:_SCHEMA_ERRORS_LISTED]
        except Exception:
            pass
        return error

    def validate_json(self, json_str: str) -> Any:
        # Thinking models (e.g. qwen3-*-thinking-*) wrap their reasoning in
        # <think>...</think> blocks before the actual JSON answer.  Strip those
        # blocks so that json.loads sees only the structured output.
        if json_str and "<think>" in json_str:
            json_str = _extract_json_from_thinking_output(json_str)
        # First attempt: parse as-is (handles clean JSON with no wrapping).
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Second attempt: model may have wrapped JSON in markdown fences or
            # added narrative text before/after – extract the JSON portion and retry.
            extracted = _extract_json_from_text(json_str or "")
            try:
                data = json.loads(extracted)
            except json.JSONDecodeError as exc:
                raise ModelBehaviorError(f"Invalid JSON output: {exc}") from exc

        data = _wrap_single_array_root(data, self._schema)
        try:
            import jsonschema  # type: ignore

            try:
                jsonschema.validate(instance=data, schema=self._schema)
            except jsonschema.ValidationError as exc:
                # Open-weight models sometimes add extra fields not in the
                # schema.  When the only issue is additionalProperties, strip
                # the unexpected keys and re-validate instead of failing.
                err_msg = str(exc).lower()
                if "additional properties are not allowed" in err_msg:
                    data = _strip_extra_properties(data, self._schema)
                elif "is not of type" in err_msg:
                    # Model returned wrong type for a field (e.g. a list where
                    # a string is expected).  Try to coerce the values.
                    data = _coerce_types(data, self._schema)
                elif "is a required property" in err_msg:
                    # Model returned array items missing required fields.
                    # Drop the offending items rather than failing entirely.
                    data = _filter_invalid_array_items(data, self._schema)
                else:
                    raise self._mismatch(exc, data) from exc

                # Re-validate after the fix attempt.  If it still fails, raise.
                try:
                    jsonschema.validate(instance=data, schema=self._schema)
                except jsonschema.ValidationError as exc2:
                    # One last attempt: apply all fixes together.
                    data = _coerce_types(
                        _filter_invalid_array_items(
                            _strip_extra_properties(data, self._schema),
                            self._schema,
                        ),
                        self._schema,
                    )
                    try:
                        jsonschema.validate(instance=data, schema=self._schema)
                    except jsonschema.ValidationError as exc3:
                        raise self._mismatch(exc3, data) from exc3
        except ModelBehaviorError:
            raise
        except ModuleNotFoundError:
            # Best-effort validation when jsonschema isn't installed.
            pass
        except Exception as exc:
            raise ModelBehaviorError(f"Output did not match schema: {exc}") from exc

        return data


# ---------------------------------------------------------------------------
# Schema validation and the one-turn repair shared by the backends that only
# state the schema in the prompt.

SCHEMA_ERROR_MAX_CHARS = 1000
# Models that fix one named error often break another; naming all of them
# (bounded) lets one repair turn fix the answer.
_SCHEMA_ERRORS_LISTED = 6
_BOUND_UNITS = {
    "maxLength": ("above the maximum", "character"),
    "minLength": ("below the minimum", "character"),
    "maxItems": ("above the maximum", "item"),
    "minItems": ("below the minimum", "item"),
    "maxProperties": ("above the maximum", "property"),
    "minProperties": ("below the minimum", "property"),
}
# Models cannot count characters, so an over-long text also gets a word
# budget that lands below the limit. Technical prose (identifiers, paths,
# security terms) runs ~9 characters a word with its space, well above
# plain English's ~6.
_CHARACTERS_PER_WORD_TARGET = 10
_SCHEMA_ERROR_EACH_MAX_CHARS = 400
_SCHEMA_ERROR_MESSAGE_MAX_CHARS = 300
_SCHEMA_ERROR_INSTANCE_MAX_CHARS = 80


# Read the way the model meant them: raw newlines inside strings, and
# backslashes JSON does not define (a regex or shell command copied into a
# string, "grep \.py") kept as literal backslashes. Valid JSON is unchanged.
_LENIENT_DECODER = json.JSONDecoder(strict=False)
_ESCAPE_PAIR = re.compile(r"\\(.)", re.DOTALL)
_OBJECT_START = re.compile(r'\{\s*"')


def _escape_stray_backslashes(text: str) -> str:
    return _ESCAPE_PAIR.sub(
        lambda match: match.group(0) if match.group(1) in '"\\/bfnrtu' else "\\" + match.group(0),
        text,
    )


def _scan_json_values(output: str) -> tuple[list[Any], json.JSONDecodeError | None]:
    """The complete JSON values in ``output`` in order, and why the object enclosing the last did not parse.

    The error is set only when the last value sits inside an object that
    failed to decode, i.e. the value is a fragment of a broken answer, not the
    answer.
    """
    text = _escape_stray_backslashes(str(output or ""))
    idx = 0
    values: list[Any] = []
    found_start = found_end = -1
    broken: tuple[int, json.JSONDecodeError] | None = None
    while idx < len(text):
        starts = [pos for pos in (text.find("{", idx), text.find("[", idx)) if pos >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            parsed, end = _LENIENT_DECODER.raw_decode(text[start:])
            values.append(parsed)
            found_start, found_end = start, start + end
            idx = start + max(end, 1)
        except json.JSONDecodeError as exc:
            if broken is None and _OBJECT_START.match(text, start):
                broken = (start, exc)
            idx = start + 1
    if broken is None or found_start < broken[0]:
        return values, None
    # Nested when the broken object is still open at the value and the text
    # after the value closes it again (a truncated draft followed by a
    # complete answer is not).
    before, after = text[broken[0]:found_start], text[found_end:]
    depth = before.count("{") + before.count("[") - before.count("}") - before.count("]")
    nested = 0 < depth <= after.count("}") + after.count("]")
    return values, broken[1] if nested else None


def extract_last_json_value(output: str) -> Any | None:
    """The last complete JSON object or array embedded anywhere in ``output``."""
    values = _scan_json_values(output)[0]
    return values[-1] if values else None


def _json_path(path: Any) -> str:
    text = ""
    for part in path or ():
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else str(part)
    return text or "the top level"


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _validation_error_text(error: Any) -> str:
    """One jsonschema error as "<what failed> (at <path>)", or "" when it is not one."""
    message = getattr(error, "message", None)
    path = getattr(error, "absolute_path", None)
    if not isinstance(message, str) or path is None:
        return ""
    bound = _BOUND_UNITS.get(str(getattr(error, "validator", "") or ""))
    instance = getattr(error, "instance", None)
    if bound and isinstance(instance, (str, list, dict)):
        # The bare "... is too long" leaves the model guessing how much to cut.
        direction, unit = bound
        limit = getattr(error, "validator_value", "?")
        text = f"{unit} count {len(instance)} is {direction} {limit} (at {_json_path(path)})"
        if isinstance(instance, str) and direction == "above the maximum" and isinstance(limit, int):
            text += f"; rewrite it in at most {max(1, limit // _CHARACTERS_PER_WORD_TARGET)} words"
        return _clip(text, _SCHEMA_ERROR_EACH_MAX_CHARS)
    # Messages quote the failing value ("[...] is not of type 'string'");
    # a short excerpt of it keeps the reason itself visible.
    try:
        instance_text = repr(getattr(error, "instance"))
    except Exception:
        instance_text = ""
    if len(instance_text) > _SCHEMA_ERROR_INSTANCE_MAX_CHARS and instance_text in message:
        message = message.replace(instance_text, _clip(instance_text, _SCHEMA_ERROR_INSTANCE_MAX_CHARS))
    return _clip(
        f"{_clip(message, _SCHEMA_ERROR_MESSAGE_MAX_CHARS)} (at {_json_path(path)})",
        _SCHEMA_ERROR_EACH_MAX_CHARS,
    )


def short_schema_error(exc: BaseException) -> str:
    """A short, bounded reason an answer failed the schema.

    jsonschema's own message embeds the whole schema and instance; this keeps
    only what failed and where.
    """
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, json.JSONDecodeError):
            return _clip(f"Invalid JSON output: {cause.msg} (line {cause.lineno} column {cause.colno})", SCHEMA_ERROR_MAX_CHARS)
        listed = [text for text in map(_validation_error_text, getattr(cause, "schema_errors", None) or []) if text]
        if len(listed) > 1:
            return _clip("; ".join(listed), SCHEMA_ERROR_MAX_CHARS)
        text = _validation_error_text(cause)
        if text:
            return text
        cause = cause.__cause__
    return _clip(str(exc), SCHEMA_ERROR_MAX_CHARS)


def _load_schema(schema: Any) -> dict[str, Any] | None:
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return None
    return schema if isinstance(schema, dict) and schema else None


def validate_schema_output(
    output: str,
    schema: Any,
    *,
    name: str = "output_schema",
    strict: bool = True,
) -> tuple[str, str]:
    """Return ``(schema-shaped JSON text, "")`` or ``(output as given, short error)``.

    The answer itself is tried first, then the JSON values embedded in it from
    the last to the first: a model may follow its answer with a second, smaller
    JSON document (a summary, a note, a tool call).
    """
    loaded = _load_schema(schema)
    if loaded is None:
        return output, ""
    validator = JsonSchemaOutput(loaded, name=name or "output_schema", strict=strict)
    values, broken = _scan_json_values(output or "")
    # Inside a broken answer only the last value is tried: the others are
    # fragments of it.
    embedded = values[-1:] if broken is not None else list(reversed(values))
    candidates = [output or ""] + [json.dumps(value, ensure_ascii=False) for value in embedded]
    errors: list[BaseException] = []
    for candidate in candidates:
        try:
            return json.dumps(validator.validate_json(candidate), ensure_ascii=False), ""
        except Exception as exc:
            errors.append(exc)
    if broken is not None:
        # The only parsable value is a fragment of the answer; its schema
        # errors would send the model after fields it did write.
        return output, short_schema_error(broken)
    if embedded:
        # A schema mismatch says more than a parse error of the surrounding
        # prose, and the largest value is the answer the model meant.
        largest = max(range(len(embedded)), key=lambda index: len(candidates[index + 1]))
        exc = errors[largest + 1]
        if not isinstance(exc.__cause__, json.JSONDecodeError):
            return output, _with_answer_keys(short_schema_error(exc), embedded[largest])
    return output, short_schema_error(errors[0])


_ANSWER_KEYS_SHOWN = 8


def _with_answer_keys(error: str, answer: Any) -> str:
    """Name the keys the answer did have when its top-level fields are missing.

    "'results' is a required property" alone does not tell a model that it
    answered with one result item, or with a tool call, instead of the object.
    """
    if not isinstance(answer, dict) or "required property (at the top level)" not in error:
        return error
    keys = [str(key)[:40] for key in answer][:_ANSWER_KEYS_SHOWN]
    shown = ", ".join(keys) if keys else "none"
    if len(answer) > _ANSWER_KEYS_SHOWN:
        shown += ", ..."
    return f"{error}; the answer's top-level keys were: {shown}"


def schema_repair_prompt(schema_error: str, schema: Any, *, name: str = "output_schema") -> str:
    """The user turn asking for the answer again in the required shape."""
    loaded = _load_schema(schema)
    schema_text = (
        json.dumps(loaded, ensure_ascii=False, separators=(",", ":"))
        if loaded is not None
        else str(schema or "")
    )
    schema_name = str(name or "").strip() or "output_schema"
    return (
        f"Your final answer does not match the required output schema `{schema_name}`: "
        f"{_clip(schema_error, SCHEMA_ERROR_MAX_CHARS)}\n"
        "Reply again with only the complete final answer, keeping every item and field of your previous "
        "answer and changing only what the error names, as one JSON object that matches this schema, "
        "with no text before or after it:\n"
        f"{schema_text}"
    )


def unfinished_task_prompt(schema_error: str, schema: Any, *, name: str = "output_schema") -> str:
    """The user turn for a reply that is a JSON object with none of the answer's fields.

    Such a reply is almost always a tool call written as text: the model
    stopped before finishing, so asking it to reshape that reply would only
    get the same object back.
    """
    loaded = _load_schema(schema)
    schema_text = (
        json.dumps(loaded, ensure_ascii=False, separators=(",", ":"))
        if loaded is not None
        else str(schema or "")
    )
    schema_name = str(name or "").strip() or "output_schema"
    return (
        f"Your reply is not the final answer `{schema_name}`: {_clip(schema_error, SCHEMA_ERROR_MAX_CHARS)}\n"
        "A tool call written as text is not run. If the task is not finished, continue it by calling your "
        "tools; when it is done, reply with only the complete final answer as one JSON object that matches "
        "this schema, with no text before or after it:\n"
        f"{schema_text}"
    )


def _shares_no_answer_field(output: str, schema: Any) -> bool:
    loaded = _load_schema(schema)
    fields = (loaded or {}).get("properties")
    answer = extract_last_json_value(output or "")
    return isinstance(fields, dict) and bool(fields) and isinstance(answer, dict) and bool(answer) and not (
        set(answer) & set(fields)
    )


def is_backend_error_output(output: str) -> bool:
    return str(output or "").strip().lower().startswith("error:")


def repair_schema_output(
    output: str,
    *,
    schema: Any,
    name: str = "output_schema",
    strict: bool = True,
    resumable: bool,
    run_turn: Callable[[str], tuple[str, list[Any], Any]],
    label: str = "agent",
) -> tuple[str, list[Any], list[Any]]:
    """Validate a CLI answer and give the same session one turn to fix it.

    Returns ``(answer, steps of the repair turn, raw responses of the repair
    turn)``. The answer is the schema-shaped JSON when the first answer or the
    repair validates, otherwise the latest output as given. Backend errors are
    returned untouched, and no repair runs without a session to resume.
    """
    if _load_schema(schema) is None or is_backend_error_output(output):
        return output, [], []
    normalized, error = validate_schema_output(output, schema, name=name, strict=strict)
    if not error:
        return normalized, [], []
    if not resumable:
        _LOGGER.warning("%s answer did not match the output schema; no session to repair it in: %s", label, error)
        return output, [], []
    if _shares_no_answer_field(output, schema):
        _LOGGER.warning("%s reply has none of the answer's fields; asking it to finish the task: %s", label, error)
        prompt = unfinished_task_prompt(error, schema, name=name)
    else:
        _LOGGER.warning("%s answer did not match the output schema; asking for a repair: %s", label, error)
        prompt = schema_repair_prompt(error, schema, name=name)
    repaired, steps, raw = run_turn(prompt)
    raw_responses = list(getattr(raw, "raw_responses", None) or [])
    if is_backend_error_output(repaired):
        _LOGGER.warning("%s schema repair turn failed: %s", label, _clip(repaired, SCHEMA_ERROR_MAX_CHARS))
        return output, list(steps or []), raw_responses
    restored = _restore_dropped_lists(output, repaired, schema, name=name, strict=strict, label=label)
    if restored is not None:
        return restored, list(steps or []), raw_responses
    normalized, error = validate_schema_output(repaired, schema, name=name, strict=strict)
    if error:
        _LOGGER.warning("%s answer still did not match the output schema after a repair turn: %s", label, error)
        return repaired, list(steps or []), raw_responses
    return normalized, list(steps or []), raw_responses


def _restore_dropped_lists(
    original: str, repaired: str, schema: Any, *, name: str, strict: bool, label: str
) -> str | None:
    """The repaired answer with the top-level lists it emptied or left out put back.

    Asked to fix one field, some models resend only that field, or an empty
    list for the rest, which discards the work of the whole run. A list the
    first answer filled is kept when the merged answer validates; otherwise
    (or when nothing was dropped) this returns None.
    """
    first = extract_last_json_value(original)
    answer = extract_last_json_value(repaired)
    if not isinstance(first, dict) or not isinstance(answer, dict):
        return None
    dropped = [key for key, value in first.items() if isinstance(value, list) and value and not answer.get(key)]
    if not dropped:
        return None
    merged, error = validate_schema_output(
        json.dumps({**answer, **{key: first[key] for key in dropped}}, ensure_ascii=False),
        schema,
        name=name,
        strict=strict,
    )
    if error:
        return None
    _LOGGER.warning("%s repair turn dropped the items of %s; keeping the first answer's", label, ", ".join(dropped))
    return merged


def answer_repair_turns(model: str) -> int:
    """Repair turns a rejected final answer gets in its session.

    GPT and Claude models fix a named error in one turn; other models often fix
    one error and introduce another, and a second turn still costs far less
    than rerunning the whole task.
    """
    bare = str(model or "").strip().lower().rsplit("/", 1)[-1]
    if "claude" in bare or bare.startswith(("gpt", "codex")) or re.match(r"o\d", bare):
        return 1
    return 2


def final_answer_rejection(
    output: str,
    *,
    schema: Any = None,
    name: str = "output_schema",
    strict: bool = True,
    validator: Callable[[str], str] | None = None,
) -> str:
    """Why a final answer is unusable, or "" when it is accepted.

    The answer must match the output schema and pass the caller's validator,
    which returns an error message (or raises) for rules a schema cannot state.
    Backend errors and empty answers are not answers to repair.
    """
    if not str(output or "").strip() or is_backend_error_output(output):
        return ""
    if _load_schema(schema) is not None:
        output, error = validate_schema_output(output, schema, name=name, strict=strict)
        if error:
            return error
    if validator is None:
        return ""
    try:
        return str(validator(output) or "").strip()
    except Exception as exc:  # the validator reports the rejection by raising
        return str(exc).strip() or type(exc).__name__


def answer_repair_prompt(rejection: str, *, structured: bool) -> str:
    """The user turn asking to fix a rejected final answer."""
    shape = (
        "as one JSON document matching the required output schema, with no text before or after it"
        if structured
        else "in the same format"
    )
    return (
        f"Your final answer was rejected: {_clip(rejection, SCHEMA_ERROR_MAX_CHARS)}\n"
        f"Fix that and reply again with only the complete corrected final answer {shape}."
    )
