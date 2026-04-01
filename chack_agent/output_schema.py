import json
import re
from typing import Any, Optional

from agents.agent_output import AgentOutputSchemaBase, ensure_strict_json_schema
from agents.exceptions import ModelBehaviorError


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


def _strip_extra_properties(data: Any, schema: dict[str, Any]) -> Any:
    """Recursively remove keys not declared in a JSON schema with additionalProperties=False.

    Open-weight models often append extra fields (e.g. 'environment_context')
    that are not in the declared schema.  Rather than failing validation, prune
    them so that strict jsonschema validation can succeed.
    """
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return data
    if schema.get("additionalProperties") is not False:
        return data
    allowed = set(schema.get("properties", {}).keys())
    nested = schema.get("properties", {})
    return {
        k: _strip_extra_properties(v, nested.get(k, {}))
        for k, v in data.items()
        if k in allowed
    }


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

    def validate_json(self, json_str: str) -> Any:
        # Thinking models (e.g. qwen3-*-thinking-*) wrap their reasoning in
        # <think>...</think> blocks before the actual JSON answer.  Strip those
        # blocks so that json.loads sees only the structured output.
        if json_str and "<think>" in json_str:
            json_str = _extract_json_from_thinking_output(json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ModelBehaviorError(f"Invalid JSON output: {exc}") from exc

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
                else:
                    raise ModelBehaviorError(
                        f"Output did not match schema: {exc}"
                    ) from exc

                # Re-validate after the fix attempt.  If it still fails, raise.
                try:
                    jsonschema.validate(instance=data, schema=self._schema)
                except jsonschema.ValidationError as exc2:
                    # One last attempt: apply both fixes together.
                    data = _coerce_types(_strip_extra_properties(data, self._schema), self._schema)
                    try:
                        jsonschema.validate(instance=data, schema=self._schema)
                    except jsonschema.ValidationError as exc3:
                        raise ModelBehaviorError(
                            f"Output did not match schema: {exc3}"
                        ) from exc3
        except ModelBehaviorError:
            raise
        except ModuleNotFoundError:
            # Best-effort validation when jsonschema isn't installed.
            pass
        except Exception as exc:
            raise ModelBehaviorError(f"Output did not match schema: {exc}") from exc

        return data
