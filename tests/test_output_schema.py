import re
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


_COMMANDS_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "proof": {"type": "object"}},
                "required": ["command"],
            },
        },
        "error": {"type": "string"},
    },
    "required": ["results", "error"],
}


def test_stray_backslashes_and_raw_newlines_are_read_as_the_model_meant_them():
    answer = (
        "Verified.\n```json\n"
        '{"results": [{"command": "grep -E \'id=\\d+\' app\\.py\nexit", "proof": {"status": 200}}],'
        ' "error": ""}\n```'
    )

    output, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    assert error == ""
    assert json.loads(output) == {
        "results": [{"command": "grep -E 'id=\\d+' app\\.py\nexit", "proof": {"status": 200}}],
        "error": "",
    }


def test_valid_escapes_are_left_exactly_as_written():
    value = {"results": [{"command": 'echo "a\\\\.b" \\ \t \u00e9'}], "error": ""}

    output, error = validate_schema_output(json.dumps(value), _COMMANDS_SCHEMA, strict=False)

    assert error == ""
    assert json.loads(output) == value


def test_a_broken_answer_reports_its_syntax_error_not_its_fragments_shape():
    # The unescaped quotes break the whole answer; the nested proof object
    # still parses but is not the answer, so its missing fields must not be
    # what the model is asked to fix.
    answer = '{"results": [{"command": "echo "hi"", "proof": {"status": 200}}], "error": ""}'

    output, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    assert output == answer
    assert error.startswith("Invalid JSON output: Expecting ',' delimiter")
    assert "required property" not in error


def test_a_complete_wrong_answer_after_broken_prose_keeps_its_schema_error():
    answer = 'Draft: {"results": [oops\nFinal: {"results": []}'

    _, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    assert error.startswith("'error' is a required property (at the top level)")


def test_an_answer_missing_its_top_level_fields_is_told_which_keys_it_had():
    # One result item instead of the object that wraps it.
    answer = '{"command": "curl /rce", "proof": {"status": 200}}'

    output, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    assert output == answer
    assert error.endswith("the answer's top-level keys were: command, proof")
    assert "'results' is a required property (at the top level)" in error


def test_an_answer_with_many_keys_names_only_a_bounded_few():
    answer = json.dumps({f"field_{index}": index for index in range(50)})

    _, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    shown = error.split("keys were: ", 1)[1]
    assert shown.endswith(", ...")
    assert shown.count("field_") == 8


def test_nested_errors_do_not_list_the_answer_keys():
    answer = '{"results": "none", "error": ""}'

    _, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    assert "(at results" in error
    assert "keys were" not in error


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


def test_length_errors_state_the_bound_and_the_actual_size():
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 50},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        },
        "required": ["summary", "tags"],
        "additionalProperties": False,
    }
    answer = json.dumps({"summary": "y" * 1834, "tags": ["one"]})

    _output, error = validate_schema_output(answer, schema, strict=False)

    assert "character count 1834 is above the maximum 50 (at summary)" in error
    assert "item count 1 is below the minimum 2 (at tags)" in error
    assert "y" * 100 not in error


def test_an_over_long_text_gets_a_word_budget_that_fits_its_limit():
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string", "maxLength": 1200}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    answer = json.dumps({"summary": "word " * 260})

    _output, error = validate_schema_output(answer, schema, strict=False)

    budget = int(re.search(r"rewrite it in at most (\d+) words", error).group(1))
    # A text of that many typical words (6 characters and a space) fits the limit.
    assert budget * 7 <= 1200
    assert budget >= 100


def test_short_lists_and_texts_get_no_word_budget():
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 50},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 1},
        },
        "required": ["summary", "tags"],
        "additionalProperties": False,
    }
    answer = json.dumps({"summary": "short", "tags": ["a", "b"]})

    _output, error = validate_schema_output(answer, schema, strict=False)

    assert "item count 2 is above the maximum 1 (at tags)" in error
    assert "words" not in error


# Over-long dedup reasons a model wrote, rewritten to the hinted word count.
_TECHNICAL_REASONS = [
    "Same unvalidated send_file traversal flaw on GET /path_traversal; v18 adds the str(e) path disclosure in 500 responses only",
    "Same root-runtime least-privilege flaw: the Dockerfile has no USER directive so flask run executes as uid 0 inside the container",
    "Both report pickle.loads on the base64url-decoded request.form['pickled'] body of POST /pickled reaching arbitrary code execution",
    "Identical unpinned Flask/Werkzeug requirement resolution; the same version pin in requirements.txt removes both reports entirely",
]


def _rewrite_to_hint(text, error):
    words = int(re.search(r"rewrite it in at most (\d+) words", error).group(1))
    return " ".join(text.split()[:words])


def test_an_answer_rewritten_to_the_hinted_word_count_fits_the_limit():
    for limit in (100, 300):
        schema = {
            "type": "object",
            "properties": {"reasoning": {"type": "string", "maxLength": limit}},
            "required": ["reasoning"],
            "additionalProperties": False,
        }
        for reason in _TECHNICAL_REASONS:
            long_reason = " ".join([reason] * (limit // len(reason) + 1))
            _, error = validate_schema_output(json.dumps({"reasoning": long_reason}), schema, strict=False)

            rewritten = json.dumps({"reasoning": _rewrite_to_hint(long_reason, error)})

            assert not validate_schema_output(rewritten, schema, strict=False)[1]


_CHECKS_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "title": {"type": "string"}},
                "required": ["id", "title"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string", "maxLength": 40},
    },
    "required": ["checks", "notes"],
    "additionalProperties": False,
}
_CHECKS = [{"id": "C1", "title": "eval on query"}, {"id": "C2", "title": "pickle loads"}]


def _repair_with(first, repaired):
    return repair_schema_output(
        json.dumps(first),
        schema=_CHECKS_SCHEMA,
        resumable=True,
        run_turn=lambda prompt: (json.dumps(repaired), [], SimpleNamespace(raw_responses=[])),
    )[0]


def test_a_repair_that_empties_the_answers_items_keeps_the_first_answers():
    output = _repair_with({"checks": _CHECKS, "notes": "x" * 80}, {"checks": [], "notes": "short"})

    assert json.loads(output) == {"checks": _CHECKS, "notes": "short"}


def test_a_repair_that_leaves_a_list_out_keeps_the_first_answers():
    output = _repair_with({"checks": _CHECKS, "notes": "x" * 80}, {"notes": "short"})

    assert json.loads(output) == {"checks": _CHECKS, "notes": "short"}


def test_a_repair_that_rewrites_the_items_is_kept_as_written():
    fixed = [{"id": "C1", "title": "eval on query"}]
    output = _repair_with({"checks": _CHECKS, "notes": "x" * 80}, {"checks": fixed, "notes": "short"})

    assert json.loads(output)["checks"] == fixed


def test_a_first_answer_whose_items_are_the_error_is_not_restored():
    broken = [{"id": "C1"}]
    output = _repair_with({"checks": broken, "notes": "ok"}, {"checks": [], "notes": "ok"})

    assert json.loads(output) == {"checks": [], "notes": "ok"}


def test_an_answer_followed_by_a_second_json_document_is_accepted():
    answer = json.dumps({"results": [{"command": "id"}], "error": ""}) + '\n\n{"summary": "1 command verified"}'

    output, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    assert error == ""
    assert json.loads(output) == {"results": [{"command": "id"}], "error": ""}


def test_a_wrong_answer_followed_by_a_note_reports_the_answers_error():
    answer = json.dumps({"results": [{"command": "id"}] * 5}) + '\n{"note": "done"}'

    output, error = validate_schema_output(answer, _COMMANDS_SCHEMA, strict=False)

    assert output == answer
    assert error.startswith("'error' is a required property (at the top level)")
    assert "keys were: results" in error


def _first_turn_prompt(answer):
    turns = []

    def run_turn(prompt):
        turns.append(prompt)
        return json.dumps({"result": {"assessment": "done", "severity": 2}}), [], None

    output, _steps, _raw = repair_schema_output(
        answer, schema=_FINDING_SCHEMA, strict=False, resumable=True, run_turn=run_turn
    )
    assert json.loads(output) == {"result": {"assessment": "done", "severity": 2}}
    return turns[0]


def test_a_tool_call_written_as_text_is_sent_back_to_finish_the_task():
    prompt = _first_turn_prompt('{"command": "node scripts/oracle.mjs", "wait_for_previous": true}')
    _normalized, error = validate_schema_output(
        '{"command": "node scripts/oracle.mjs", "wait_for_previous": true}', _FINDING_SCHEMA, strict=False
    )

    assert prompt == MODULE.unfinished_task_prompt(error, _FINDING_SCHEMA)


def test_an_answer_with_some_of_its_fields_gets_the_reshape_repair():
    answer = '{"result": {"assessment": "done"}}'
    prompt = _first_turn_prompt(answer)
    _normalized, error = validate_schema_output(answer, _FINDING_SCHEMA, strict=False)

    assert prompt == MODULE.schema_repair_prompt(error, _FINDING_SCHEMA)


def test_an_empty_reply_is_sent_back_to_finish_the_task():
    prompt = _first_turn_prompt("")
    _normalized, error = validate_schema_output("", _FINDING_SCHEMA, strict=False)

    assert prompt == MODULE.unfinished_task_prompt(error, _FINDING_SCHEMA)
