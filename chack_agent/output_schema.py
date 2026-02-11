import json
from typing import Any, Optional

from agents.agent_output import AgentOutputSchemaBase, ensure_strict_json_schema
from agents.exceptions import ModelBehaviorError


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
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ModelBehaviorError(f"Invalid JSON output: {exc}") from exc

        try:
            import jsonschema  # type: ignore

            jsonschema.validate(instance=data, schema=self._schema)
        except ModuleNotFoundError:
            # Best-effort validation when jsonschema isn't installed.
            pass
        except Exception as exc:
            raise ModelBehaviorError(f"Output did not match schema: {exc}") from exc

        return data
