"""Checks that a CLI backend's final message is an answer.

Some gateway models end a run with a message that is not an answer: a tool
call written out as text instead of called, or the schema filled with a
placeholder. Both come back from the CLI as a successful run, so a caller
would take them as the agent's result. These checks name them so the backend
can report a provider failure instead.

They only reject what can never be an answer. Whether an agent had to use its
tools is the agent's call (``tools.required_tool_names``), not checked here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from ..output_schema import JsonSchemaOutput

# Tool-call markup of chat templates that some providers fail to parse and
# return as text (Kimi, MiniMax, Qwen/Hermes, GLM, Anthropic XML), at the
# start of a line of the final message.
_TEXT_TOOL_CALL_RE = re.compile(
    r"^\s*(?:<\|tool_calls?_(?:section_)?begin\|>|<\|tool_call|<(?:minimax:)?tool_call\b"
    r"|<function_calls>|<invoke\s+name=|<function=)",
    re.IGNORECASE | re.MULTILINE,
)
_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_CALL_ARGUMENT_KEYS = ("arguments", "parameters", "input", "args")
_PLACEHOLDERS = frozenset(
    {"pending", "tbd", "todo", "placeholder", "to be determined", "fill in", "...", "xxx"}
)


def tool_argument_shapes(tools: list[Any]) -> dict[str, dict[str, list[str]]]:
    """Each tool's argument names and required arguments, by tool name."""
    shapes: dict[str, dict[str, list[str]]] = {}
    for tool in tools or []:
        name = str(getattr(tool, "name", "") or "").strip()
        schema = getattr(tool, "params_json_schema", None)
        if not name or not isinstance(schema, Mapping):
            continue
        properties = schema.get("properties")
        shapes[name] = {
            "properties": sorted(properties) if isinstance(properties, Mapping) else [],
            "required": sorted(str(item) for item in schema.get("required") or []),
        }
    return shapes


def _schema_validator(output_schema_json: str, strict: bool) -> JsonSchemaOutput | None:
    try:
        schema = json.loads(output_schema_json or "")
    except json.JSONDecodeError:
        return None
    if not isinstance(schema, dict):
        return None
    return JsonSchemaOutput(schema, name="output_schema", strict=strict)


def _validated(output: str, validator: JsonSchemaOutput | None) -> Any | None:
    if validator is None:
        return None
    try:
        return validator.validate_json(output)
    except Exception:
        return None


def _json_object(output: str) -> dict[str, Any] | None:
    text = output.strip()
    fenced = _FENCED_JSON_RE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _is_tool_arguments(value: Mapping[str, Any], shape: Mapping[str, list[str]]) -> bool:
    required = set(shape.get("required") or [])
    return bool(required) and required <= set(value) <= set(shape.get("properties") or [])


def _text_tool_call(output: str, tools: Mapping[str, Mapping[str, list[str]]]) -> str:
    """The tool the message calls in text, "markup" for tool-call markup, or ""."""
    if _TEXT_TOOL_CALL_RE.search(output):
        return "markup"
    value = _json_object(output)
    if value is None:
        return ""
    name = str(value.get("name") or value.get("tool") or "")
    if name in tools and set(value) - {"name", "tool"} <= set(_CALL_ARGUMENT_KEYS):
        return name
    for tool_name, shape in tools.items():
        if _is_tool_arguments(value, shape):
            return tool_name
    return ""


def _free_text_values(value: Any, schema: Any) -> list[str]:
    """String values of ``value`` whose schema does not fix them (no enum/const)."""
    schema = schema if isinstance(schema, Mapping) else {}
    if isinstance(value, str):
        return [] if "enum" in schema or "const" in schema else [value]
    if isinstance(value, Mapping):
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        return [
            text for key, item in value.items() for text in _free_text_values(item, properties.get(key))
        ]
    if isinstance(value, list):
        return [text for item in value for text in _free_text_values(item, schema.get("items"))]
    return []


def _is_placeholder(value: Any, output_schema_json: str) -> bool:
    try:
        schema = json.loads(output_schema_json or "")
    except json.JSONDecodeError:
        schema = {}
    texts = [text.strip().strip(".:;!").lower() for text in _free_text_values(value, schema)]
    texts = [text for text in texts if text]
    return bool(texts) and all(text in _PLACEHOLDERS for text in texts)


TOOL_CALL_AS_TEXT = "tool_call_as_text"
PLACEHOLDER = "placeholder"


def final_answer_problem(
    output: str,
    *,
    output_schema_json: str = "",
    output_schema_strict: bool = True,
    tools: Mapping[str, Mapping[str, list[str]]] | None = None,
) -> tuple[str, str]:
    """Why ``output`` is not an answer, as (kind, detail); ("", "") when it can be one."""
    text = str(output or "").strip()
    if not text:
        return "", ""
    validator = _schema_validator(output_schema_json, output_schema_strict) if output_schema_json else None
    answer = _validated(text, validator)
    if answer is None:
        tool = _text_tool_call(text, tools or {})
        if tool == "markup":
            return TOOL_CALL_AS_TEXT, f"the final message is tool-call markup: {text[:300]!r}"
        if tool:
            return TOOL_CALL_AS_TEXT, f"the final message is a call of tool {tool!r}: {text[:300]!r}"
        return "", ""
    if _is_placeholder(answer, output_schema_json):
        return PLACEHOLDER, f"every free-text field of the answer is a placeholder: {text[:300]!r}"
    return "", ""
