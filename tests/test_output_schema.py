import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


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


validate_schema_output = MODULE.validate_schema_output
repair_schema_output = MODULE.repair_schema_output
SCHEMA_ERROR_MAX_CHARS = MODULE.SCHEMA_ERROR_MAX_CHARS

_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "object",
            "properties": {
                "assessment": {"type": "string"},
                "severity": {"type": "integer"},
            },
            "required": ["assessment", "severity"],
            "additionalProperties": False,
        }
    },
    "required": ["result"],
    "additionalProperties": False,
}


def test_missing_required_property_is_reported_with_its_path():
    answer = '{"result": {"severity": 3}}'

    output, error = validate_schema_output(answer, _FINDING_SCHEMA, strict=False)

    assert output == answer
    assert error == "'assessment' is a required property (at result)"


def test_wrong_type_error_stays_short_even_for_a_huge_value():
    huge = ["x" * 5000]
    answer = json.dumps({"result": {"assessment": "ok", "severity": huge}})

    output, error = validate_schema_output(answer, _FINDING_SCHEMA, strict=False)

    assert output == answer
    assert len(error) <= SCHEMA_ERROR_MAX_CHARS
    assert "is not of type 'integer'" in error
    assert error.endswith("(at result.severity)")
    # Neither the schema nor the whole value is echoed back.
    assert "x" * 200 not in error
    assert "additionalProperties" not in error


def test_invalid_json_is_reported_briefly():
    answer = "I found three issues; see the summary above. " * 50

    output, error = validate_schema_output(answer, _FINDING_SCHEMA, strict=False)

    assert output == answer
    assert error.startswith("Invalid JSON output")
    assert len(error) <= SCHEMA_ERROR_MAX_CHARS


def test_json_embedded_in_prose_is_normalized():
    answer = 'Done. Final: {"result": {"assessment": "ok", "severity": 2}} Thanks.'

    output, error = validate_schema_output(answer, _FINDING_SCHEMA, strict=False)

    assert error == ""
    assert json.loads(output) == {"result": {"assessment": "ok", "severity": 2}}


def test_without_a_schema_the_answer_is_left_alone():
    assert validate_schema_output("prose", "") == ("prose", "")


def test_failed_repair_turn_keeps_the_first_answer_and_counts_its_usage():
    turns = []

    def run_turn(prompt):
        turns.append(prompt)
        return "ERROR: provider unavailable", [("step", None)], SimpleNamespace(raw_responses=[{"usage": {"input_tokens": 7}}])

    output, steps, raw = repair_schema_output(
        "not json", schema=_FINDING_SCHEMA, strict=False, resumable=True, run_turn=run_turn
    )

    assert output == "not json"
    assert len(turns) == 1
    assert steps == [("step", None)]
    assert raw == [{"usage": {"input_tokens": 7}}]


_GROUP_RESULTS_SCHEMA = {
    "type": "object",
    "properties": {
        "group_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"group_id": {"type": "string"}, "reasoning": {"type": "string"}},
                "required": ["group_id", "reasoning"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["group_results"],
    "additionalProperties": False,
}


def test_bare_item_or_list_is_wrapped_in_the_only_root_array():
    item = {"group_id": "g1", "reasoning": "same sink"}
    for answer in (item, [item]):
        normalized, error = MODULE.validate_schema_output(json.dumps(answer), _GROUP_RESULTS_SCHEMA)

        assert error == ""
        assert json.loads(normalized) == {"group_results": [item]}


def test_bare_answer_is_not_wrapped_when_the_root_has_other_fields():
    schema = json.loads(json.dumps(_GROUP_RESULTS_SCHEMA))
    schema["properties"]["summary"] = {"type": "string"}
    schema["required"].append("summary")

    _, error = MODULE.validate_schema_output(json.dumps({"group_id": "g1", "reasoning": "x"}), schema)

    assert "required property" in error


_STEPS_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "branch_id": {"type": "string"},
                            "flow_title": {"type": "string"},
                        },
                        "required": ["text", "branch_id", "flow_title"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


def test_repair_turn_names_every_schema_error_so_one_turn_can_fix_them_all():
    answer = json.dumps({"steps": {"items": [
        {"text": "a", "branch_id": "b1", "flow_title": "f"},
        {"text": "b"},
        {"text": 3, "branch_id": "b2", "flow_title": "f"},
    ]}})
    prompts = []

    def run_turn(prompt):
        prompts.append(prompt)
        return json.dumps({"steps": {"items": [{"text": "b", "branch_id": "b1", "flow_title": "f"}]}}), [], None

    output, _steps, _raw = repair_schema_output(
        answer, schema=_STEPS_SCHEMA, strict=False, resumable=True, run_turn=run_turn
    )

    assert json.loads(output)["steps"]["items"][0]["branch_id"] == "b1"
    assert len(prompts) == 1
    for reported in (
        "'branch_id' is a required property (at steps.items[1])",
        "'flow_title' is a required property (at steps.items[1])",
        "3 is not of type 'string' (at steps.items[2].text)",
    ):
        assert reported in prompts[0]


def test_listed_schema_errors_stay_bounded():
    answer = json.dumps({"steps": {"items": [{"text": "x" * 5000}] * 50}})

    _output, error = validate_schema_output(answer, _STEPS_SCHEMA, strict=False)

    assert "'branch_id' is a required property" in error
    assert len(error) <= SCHEMA_ERROR_MAX_CHARS
    assert error.count("is a required property") <= 6
