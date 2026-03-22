import ast
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "backends" / "codex_backend.py"
CLAUDE_MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "backends" / "claude_code_backend.py"
GEMINI_MODULE_PATH = Path(__file__).resolve().parents[1] / "chack_agent" / "backends" / "gemini_cli_backend.py"


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
