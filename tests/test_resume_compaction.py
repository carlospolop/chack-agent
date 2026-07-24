from types import SimpleNamespace

from chack_agent.backends.claude_code_backend import ClaudeCodeExecutor
from chack_agent.backends.codex_backend import CodexExecutor
from chack_agent.backends.copilot_cli_backend import CopilotCliExecutor
from chack_agent.backends.gemini_cli_backend import GeminiCliExecutor
from chack_agent.backends.langgraph_backend import LangGraphExecutor
from chack_agent.backends.openai_compaction_backend import (
    AgentsExecutor as OpenAIExecutor,
)
from chack_agent.backends.openrouter_openai_backend import (
    AgentsExecutor as OpenRouterExecutor,
)


def _raw_result():
    return SimpleNamespace(
        raw_responses=[
            {
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "input_tokens_details": {
                        "cached_tokens": 4,
                        "cache_write_tokens": 0,
                    },
                }
            }
        ]
    )


def test_codex_compaction_is_skipped_without_a_thread_and_uses_native_api():
    executor = CodexExecutor.__new__(CodexExecutor)
    executor._thread_id = None
    assert executor.compact_for_resume().attempted is False

    calls = []
    executor._thread_id = "thread-1"
    executor._compact_codex_thread = lambda focus: calls.append(focus) or [
        {"usage": {"input_tokens": 10, "output_tokens": 2}}
    ]
    result = executor.compact_for_resume("preserve checks")

    assert result.succeeded is True
    assert result.method == "thread/compact/start"
    assert calls == ["preserve checks"]
    assert result.raw_responses[0]["usage"]["input_tokens"] == 10


def test_claude_compaction_uses_manual_command_with_focus():
    executor = ClaudeCodeExecutor.__new__(ClaudeCodeExecutor)
    executor._claude_session_id = "session-1"
    calls = []
    executor._run_claude_once = (
        lambda prompt, resume_compaction=False: (
            calls.append((prompt, resume_compaction)) or ("Compacted", [], _raw_result())
        )
    )

    result = executor.compact_for_resume("preserve checks")

    assert result.succeeded is True
    assert calls == [("/compact preserve checks", True)]


def test_gemini_compaction_uses_compress_command():
    executor = GeminiCliExecutor.__new__(GeminiCliExecutor)
    executor._gemini_session_id = "session-1"
    calls = []
    executor._run_gemini = (
        lambda prompt: calls.append(prompt) or ("Compressed", [], _raw_result())
    )

    result = executor.compact_for_resume("unsupported focus")

    assert result.succeeded is True
    assert calls == ["/compress"]


def test_copilot_compaction_uses_manual_command_with_focus():
    executor = CopilotCliExecutor.__new__(CopilotCliExecutor)
    executor._copilot_session_id = "session-1"
    calls = []
    executor._run_copilot = (
        lambda prompt: calls.append(prompt) or ("Compacted", [], _raw_result())
    )

    result = executor.compact_for_resume("preserve checks")

    assert result.succeeded is True
    assert calls == ["/compact preserve checks"]


def test_openai_compaction_rotates_to_compacted_response():
    executor = OpenAIExecutor.__new__(OpenAIExecutor)
    executor._previous_response_id = "response-1"
    executor._conversation = [{"role": "user"}, {"role": "assistant"}]
    executor._run_compaction = lambda response_id: (
        "response-2" if response_id == "response-1" else None
    )
    executor._normalized_memory_reset_to = lambda: 1

    result = executor.compact_for_resume()

    assert result.succeeded is True
    assert executor._previous_response_id == "response-2"
    assert executor._conversation == [{"role": "assistant"}]


def test_openrouter_compaction_summarizes_and_rotates_server_chain():
    executor = OpenRouterExecutor.__new__(OpenRouterExecutor)
    executor._conversation = [{"role": "user", "content": "context"}]
    executor._summary = ""
    executor._previous_response_id = "response-1"
    executor._conversation_id = "conversation-1"
    calls = []
    executor._summarize_items = (
        lambda *args, **kwargs: calls.append((args, kwargs)) or "summary"
    )

    result = executor.compact_for_resume("preserve checks")

    assert result.succeeded is True
    assert executor._summary == "summary"
    assert executor._conversation == []
    assert executor._previous_response_id is None
    assert executor._conversation_id is None
    assert calls[0][1]["focus_instructions"] == "preserve checks"


def test_langgraph_compaction_summarizes_and_rotates_checkpoint_thread():
    executor = LangGraphExecutor.__new__(LangGraphExecutor)
    executor._thread_id = "thread-1"
    executor._conversation = [{"role": "user", "content": "context"}]
    executor._graph = SimpleNamespace(
        get_state=lambda _config: SimpleNamespace(
            values={
                "messages": [SimpleNamespace(content="context")],
                "summary": "",
            }
        )
    )
    calls = []
    executor._summarize_messages = (
        lambda *args, **kwargs: calls.append((args, kwargs)) or "summary"
    )

    result = executor.compact_for_resume("preserve checks")

    assert result.succeeded is True
    assert executor._thread_id != "thread-1"
    assert executor._pending_resume_summary == "summary"
    assert executor._conversation == []
    assert calls[0][1]["focus_instructions"] == "preserve checks"
