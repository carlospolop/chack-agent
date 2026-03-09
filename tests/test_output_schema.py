import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "output_schema.py"
SPEC = importlib.util.spec_from_file_location("output_schema_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
JsonSchemaOutput = MODULE.JsonSchemaOutput


def test_json_schema_output_normalizes_composition_root_to_object():
    schema = {
        "$defs": {
            "Success": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            "Failure": {
                "type": "object",
                "properties": {"error": {"type": "string"}},
                "required": ["error"],
                "additionalProperties": False,
            },
        },
        "anyOf": [
            {"$ref": "#/$defs/Success"},
            {"$ref": "#/$defs/Failure"},
        ],
    }

    output = JsonSchemaOutput(schema, strict=False)

    assert output.json_schema()["type"] == "object"
    assert "anyOf" in output.json_schema()
