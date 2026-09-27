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


def test_gemini_compacts_after_turn_crosses_configured_threshold():
    executor = GeminiCliExecutor.__new__(GeminiCliExecutor)
    executor._conversation = []
    executor._memory_limit = 0
    executor._max_context_tokens = 20
    executor._compaction_threshold_ratio = 0.5
    executor._gemini_session_id = "session-1"
    executor._compose_prompt = lambda text: text
    calls = []
    executor._run_gemini = lambda prompt: (
        calls.append(prompt) or ("Compressed" if prompt == "/compress" else "Answer"),
        [],
        _raw_result(),
    )

    result = executor.invoke({"input": "Question"})

    assert result["output"] == "Answer"
    assert calls == ["Question", "/compress"]
    assert len(result["raw_result"].raw_responses) == 2


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


def test_copilot_compacts_after_reported_context_crosses_threshold():
    executor = CopilotCliExecutor.__new__(CopilotCliExecutor)
    executor._conversation = []
    executor._memory_limit = 0
    executor._max_context_tokens = 350
    executor._compaction_threshold_ratio = 0.75
    executor._copilot_context_tokens = 0
    executor._copilot_context_limit = 0
    executor._copilot_session_id = "session-1"
    executor._compose_prompt = lambda text: text
    calls = []

    def run(prompt):
        calls.append(prompt)
        if prompt != "/compact":
            executor._record_context_usage({"currentTokens": 75, "tokenLimit": 100})
        return ("Compacted" if prompt == "/compact" else "Answer", [], _raw_result())

    executor._run_copilot = run
    result = executor.invoke({"input": "Question"})

    assert result["output"] == "Answer"
    assert calls == ["Question", "/compact"]
    assert executor._copilot_context_tokens == 0
    assert len(result["raw_result"].raw_responses) == 2


def test_copilot_ignores_invalid_context_usage_snapshot():
    executor = CopilotCliExecutor.__new__(CopilotCliExecutor)
    executor._copilot_context_tokens = 42
    executor._copilot_context_limit = 100
    executor._record_context_usage({"currentTokens": "not-a-number"})
    assert executor._copilot_context_tokens == 42
    assert executor._copilot_context_limit == 100


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


def test_openrouter_automatic_threshold_compaction_rotates_server_chain():
    executor = OpenRouterExecutor.__new__(OpenRouterExecutor)
    executor._conversation = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "keep"},
    ]
    executor._summary = ""
    executor._summary_keep_messages = 1
    executor._summary_trigger_messages = 100
    executor._max_context_tokens = 100
    executor._compaction_threshold_ratio = 0.75
    executor._previous_response_id = "response-1"
    executor._conversation_id = "conversation-1"
    executor._summarize_items = lambda *args, **kwargs: "summary"

    executor._maybe_summarize_for_next_turn(input_tokens=75)

    assert executor._summary == "summary"
    assert executor._conversation == [{"role": "user", "content": "keep"}]
    assert executor._previous_response_id is None
    assert executor._conversation_id is None


def test_openrouter_automatic_compaction_preserves_chain_if_summary_is_empty():
    executor = OpenRouterExecutor.__new__(OpenRouterExecutor)
    executor._conversation = [{"role": "user", "content": "context"}]
    executor._summary = ""
    executor._summary_keep_messages = 0
    executor._summary_trigger_messages = 100
    executor._max_context_tokens = 100
    executor._compaction_threshold_ratio = 0.75
    executor._previous_response_id = "response-1"
    executor._conversation_id = "conversation-1"
    executor._summarize_items = lambda *args, **kwargs: ""

    executor._maybe_summarize_for_next_turn(input_tokens=75)

    assert executor._summary == ""
    assert executor._conversation == [{"role": "user", "content": "context"}]
    assert executor._previous_response_id == "response-1"
    assert executor._conversation_id == "conversation-1"


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


def test_langgraph_threshold_compaction_rotates_checkpoint_after_turn():
    executor = LangGraphExecutor.__new__(LangGraphExecutor)
    executor._thread_id = "thread-1"
    executor._recursion_limit = 10
    executor._conversation = []
    executor._memory_limit = 0
    executor._max_context_tokens = 100
    executor._compaction_threshold_ratio = 0.75
    executor._pending_resume_summary = ""
    executor._extract_output = lambda messages: "Answer"
    executor._summarize_messages = lambda *args, **kwargs: "summary"
    state = {"invoked": False}

    def get_state(_config):
        if not state["invoked"]:
            return SimpleNamespace(values={})
        return SimpleNamespace(values={
            "messages": [SimpleNamespace(content="Question"), SimpleNamespace(content="Answer")],
            "summary": "",
            "tool_events": [],
            "usage_events": [{"input_tokens": 75, "output_tokens": 3}],
        })

    def invoke(_input, config):
        state["invoked"] = True
        return get_state(config).values

    executor._graph = SimpleNamespace(get_state=get_state, invoke=invoke)

    result = executor.invoke({"input": "Question"})

    assert result["output"] == "Answer"
    assert executor._thread_id != "thread-1"
    assert executor._pending_resume_summary == "summary"
    assert executor._conversation == []
