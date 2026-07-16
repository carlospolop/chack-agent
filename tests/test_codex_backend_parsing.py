import ast
import json
import importlib.util
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "backends" / "codex_backend.py"
CLAUDE_MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "backends" / "claude_code_backend.py"
GEMINI_MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "backends" / "gemini_cli_backend.py"
OUTPUT_SCHEMA_MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "output_schema.py"


def _load_codex_helper(function_name: str):
    module_ast = ast.parse(MODULE_PATH.read_text())
    class_node = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == "CodexExecutor"
    )
    helper_names = {"_extract_message_text", "_extract_text_candidate"}
    method_nodes = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    for node in method_nodes:
        node.decorator_list = []

    isolated_module = ast.Module(body=method_nodes, type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {"Any": Any}
    exec(compile(isolated_module, str(MODULE_PATH), "exec"), namespace)
    return namespace[function_name]


def _load_list_literal(path: Path, class_name: str, function_name: str, target_name: str):
    module_ast = ast.parse(path.read_text())
    class_node = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    for node in function_node.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == target_name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{target_name} not found in {path}:{function_name}")


def _load_claude_helper(function_name: str):
    module_ast = ast.parse(CLAUDE_MODULE_PATH.read_text())
    class_node = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == "ClaudeCodeExecutor"
    )
    method_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    method_node.decorator_list = []
    isolated_module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(isolated_module)

    spec = importlib.util.spec_from_file_location("output_schema_module", OUTPUT_SCHEMA_MODULE_PATH)
    output_schema_module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(output_schema_module)

    class _Logger:
        @staticmethod
        def warning(*args, **kwargs):
            return None

    namespace = {
        "json": json,
        "JsonSchemaOutput": output_schema_module.JsonSchemaOutput,
        "_LOGGER": _Logger(),
    }
    exec(compile(isolated_module, str(CLAUDE_MODULE_PATH), "exec"), namespace)
    return namespace[function_name]


extract_message_text = _load_codex_helper("_extract_message_text")
extract_text_candidate = _load_codex_helper("_extract_text_candidate")


class _CodexHelperShim:
    _extract_message_text = classmethod(extract_message_text)
    _extract_text_candidate = classmethod(extract_text_candidate)


def test_extract_message_text_supports_content_parts():
    payload = {
        "type": "assistant_message",
        "content": [
            {"type": "text", "text": '{"analysis_summary":"ok"}'},
        ],
    }

    assert _CodexHelperShim._extract_message_text(payload) == '{"analysis_summary":"ok"}'


def test_extract_message_text_supports_nested_item_message():
    payload = {
        "type": "message",
        "item": {
            "type": "message",
            "message": {
                "content": [
                    {"type": "output_text", "text": '{"new_vulnerabilities":[]}'},
                ],
            },
        },
    }

    assert _CodexHelperShim._extract_message_text(payload) == '{"new_vulnerabilities":[]}'


def test_extract_message_text_joins_multiple_text_parts():
    payload = {
        "content": [
            {"type": "text", "text": '{"part":1}'},
            {"type": "text", "text": '{"part":2}'},
        ]
    }

    assert _CodexHelperShim._extract_message_text(payload) == '{"part":1}\n{"part":2}'


def test_codex_mcp_env_allowlist_includes_local_vulnerability_store_path():
    env_vars = _load_list_literal(
        MODULE_PATH,
        "CodexExecutor",
        "_write_codex_config",
        "env_vars",
    )

    assert "AISEC_LOCAL_VULN_STORE_PATH" in env_vars


def test_claude_mcp_env_allowlist_includes_local_vulnerability_store_path():
    env_vars = _load_list_literal(
        CLAUDE_MODULE_PATH,
        "ClaudeCodeExecutor",
        "_mcp_env_map",
        "env_keys",
    )

    assert "AISEC_LOCAL_VULN_STORE_PATH" in env_vars


def test_gemini_mcp_env_allowlist_includes_local_vulnerability_store_path():
    env_vars = _load_list_literal(
        GEMINI_MODULE_PATH,
        "GeminiCliExecutor",
        "_gemini_mcp_env_map",
        "env_keys",
    )

    assert "AISEC_LOCAL_VULN_STORE_PATH" in env_vars


def test_codex_backend_logs_codex_cli_failure_event():
    module_ast = ast.parse(MODULE_PATH.read_text())
    found = False

    for node in ast.walk(module_ast):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "log_event":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and first_arg.value == "codex_cli_failure":
            found = True
            break

    assert found is True


def test_codex_backend_does_not_pass_cd_to_exec_resume():
    module_ast = ast.parse(MODULE_PATH.read_text())

    for node in ast.walk(module_ast):
        if not isinstance(node, ast.FunctionDef) or node.name != "_build_command":
            continue
        for if_node in [child for child in ast.walk(node) if isinstance(child, ast.If)]:
            test = if_node.test
            if not isinstance(test, ast.Attribute):
                continue
            if not isinstance(test.value, ast.Name) or test.value.id != "self":
                continue
            if test.attr != "_thread_id":
                continue

            found_resume = False
            found_cd = False
            for child in ast.walk(if_node):
                if isinstance(child, ast.Constant) and child.value == "resume":
                    found_resume = True
                if isinstance(child, ast.Constant) and child.value == "--cd":
                    found_cd = True

            assert found_resume is True
            assert found_cd is False
            return

    raise AssertionError("Could not find _thread_id resume branch in _build_command")


def test_claude_backend_normalizes_schema_output_from_wrapped_text():
    normalize_schema_output = _load_claude_helper("_normalize_schema_output")

    class _ClaudeShim:
        _output_schema_json = json.dumps(
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "groups": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["summary", "groups"],
                "additionalProperties": False,
            }
        )
        _output_schema_name = "grouping_output"
        _output_schema_strict = True

    normalized = normalize_schema_output(
        _ClaudeShim(),
        "Here is the result:\n```json\n{\"summary\":\"ok\",\"groups\":[]}\n```"
    )

    assert json.loads(normalized) == {"summary": "ok", "groups": []}


def test_claude_backend_keeps_structured_output_from_result_event():
    source = CLAUDE_MODULE_PATH.read_text()

    assert 'event.get("structured_output")' in source


def test_claude_backend_prefers_claude_access_token_over_anthropic_env_vars():
    source = CLAUDE_MODULE_PATH.read_text()

    assert 'env["CLAUDE_CODE_OAUTH_TOKEN"] = self._claude_access_token' in source
    assert 'env.pop("ANTHROPIC_API_KEY", None)' in source
    assert 'env.pop("CLAUDE_API_KEY", None)' in source
